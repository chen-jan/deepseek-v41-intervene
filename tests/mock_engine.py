"""CPU mock of the engine's HTTP surface, for a dry run of ``tools/gates.py`` before a GPU boot.

Emulates ``/v1/completions`` (token-id prompts, ``prompt_logprobs``, decode ``logprobs``, ``n``, ``vllm_xargs``,
``cache_salt``, a 128-token-block prefix cache that ``prompt_logprobs`` requests do not read), ``/collective_rpc`` with
the five ``dsv41_*`` methods (4 identical ranks), the device counters by INTERFACE.md section 3, capture dumps,
``/metrics`` (always idle) and the key on every route but ``/health`` (like ``dsv41_intervene.auth``).
Logprobs are a hash of (token, position) plus a shift that depends on the arm's table content, so arms that edit
differ from the sham and teacher-forced scores match decode logprobs. Not a model: it checks the gate code, not the
engine.

    python tests/mock_engine.py --port 18101 --key-file KEYFILE --capture-dir DIR
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from dsv41_intervene import DEFAULT_CAPTURE_LAYERS, DEFAULT_EDIT_LAYERS, MAX_EDIT_TO, parse_layers  # noqa: E402

BLOCK = 128
DECODE_CHUNK, CHUNK_S = 64, 0.02  # uncaptured long decodes count their tokens in steps, as a real server does
LAYERS = parse_layers(DEFAULT_EDIT_LAYERS)
CAP_LAYERS = parse_layers(DEFAULT_CAPTURE_LAYERS)
A, I = 16, len(LAYERS)


def h(*xs) -> int:
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), "little")


class Mock:
    def __init__(self, capture_dir: str):
        self.capture_dir = capture_dir
        self.t = {"p": np.zeros((A, I, 4, 5120), np.float32), "e": np.zeros((A, I, 4, 5120), np.float32),
                  "mu": np.zeros((A, I, 4), np.float32), "a": np.zeros((A, I), np.float32),
                  "g": np.zeros((A, I), np.float32), "mc": np.zeros(A, np.int32)}
        self.sha, self.global_arm, self.invalid = "", -1, 0
        self.boot = f"mock-{os.getpid()}-{int(time.time())}"
        self.cache: set = set()
        self.rng = np.random.default_rng(0)
        self.reset_counters()
        self.reset_capture()

    def reset_counters(self):
        self.seen = np.zeros(A, np.int64)
        self.edited = np.zeros((A, I), np.int64)
        self.sad = np.zeros((A, I), np.float32)
        self.max_err = np.zeros((A, I), np.float32)
        self.nonfinite = np.zeros((A, I), np.int64)
        self.cast_violations = np.zeros((A, I), np.int64)

    def reset_capture(self):
        self.coef, self.vec, self.cmap, self.vmap, self.req_ids, self.tags = [], [], [], [], [], []

    def sig(self, arm: int) -> float:
        """0 for an arm that edits nothing; else a content hash in [1.2, 2.2) (so every editing arm's median effect on
        a text, about s/2, clears the decode_effect gate's 0.5 testability line)."""
        t = self.t
        if not (np.any(t["a"][arm]) or np.any(t["g"][arm])):
            return 0.0
        key = b"".join(t[k][arm].tobytes() for k in ("p", "e", "mu", "a", "g")) + t["mc"][arm].tobytes()
        return 1.2 + (int.from_bytes(hashlib.sha256(key).digest()[:4], "little") % 1000) / 1000

    def lp(self, tok: int, pos: int, s: float) -> float:
        return -((h(tok, pos) % 3000) / 1000.0) - s * ((h(tok, pos, 1) % 1000) / 1000.0) - 1e-4 * (h(pos) % 3)

    # ------------------------------------------------------------------------------------------------------------------
    def rpc(self, method: str, args: list[str]) -> list[dict]:
        base = {"layers": LAYERS, "arms_sha": self.sha, "boot": self.boot}
        if method == "dsv41_installed":
            out = {"installed": True, "capture_layers": CAP_LAYERS, "fuse_mhc_all_reduce": False, "mhc_stream": True,
                   "model_fuse_mhc_all_reduce": False, "deferred_finalize": [], "all_reduce_mhc_built": False,
                   "n_arms": A, "n_inst": I, "impl": "mock", "sequence_parallel": False, "spec_decode": False,
                   "max_t": 32768, "coef_rows": 1 << 20, "vec_rows": 1 << 15, "global_arm": self.global_arm,
                   "wrapped": LAYERS}
        elif method == "dsv41_load_arms":
            from safetensors.numpy import load_file

            path, sha = args
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("sha mismatch")
            f = load_file(path)
            assert list(f["layers"]) == LAYERS and f["p"].shape == (A, I, 4, 5120)
            for k in self.t:
                self.t[k] = f[k]
            self.sha = sha
            base["arms_sha"] = sha
            out = {"checksums": {k: float(np.asarray(v, np.float64).sum()) for k, v in self.t.items()}}
        elif method == "dsv41_stats":
            out = {"tokens_seen": self.seen.tolist(), "tokens_edited": self.edited.tolist(),
                   "sum_abs_delta": self.sad.tolist(), "max_err": self.max_err.tolist(),
                   "nonfinite": self.nonfinite.tolist(), "cast_violations": self.cast_violations.tolist(),
                   "invalid_arm_requests": self.invalid,
                   "invalid_capture_requests": 0, "global_arm": self.global_arm, "coef_rows_used": len(self.coef),
                   "vec_rows_used": len(self.vec), "coef_overflow": 0, "vec_overflow": 0, "t": time.time()}
            if args and args[0] == "1":
                self.reset_counters()
        elif method == "dsv41_capture_dump":
            out = self.dump(args[0])
        elif method == "dsv41_set_global_arm":
            self.global_arm = int(args[0])
            out = {"global_arm": self.global_arm}
        else:
            raise ValueError(method)
        return [{"rank": r, **base, **(out if r == 0 or method != "dsv41_capture_dump" else
                                      {"path": None, "coef_rows": 0, "vec_rows": 0})} for r in range(4)]

    def dump(self, path: str) -> dict:
        import torch
        from safetensors.torch import save_file

        assert os.path.realpath(path).startswith(os.path.realpath(self.capture_dir)), path
        assert not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cm = np.asarray(self.cmap, np.int32).reshape(-1, 4)
        vm = np.asarray(self.vmap, np.int32).reshape(-1, 4)
        tens = {"coef": np.asarray(self.coef, np.float32).reshape(-1, I, 3, 4),
                "vec": np.asarray(self.vec, np.float32).reshape(-1, len(CAP_LAYERS), 5120)}
        for i, f in enumerate(("pos", "arm", "req", "tag")):
            tens[f"coef_{f}"], tens[f"vec_{f}"] = cm[:, i].copy(), vm[:, i].copy()
        tens["coef_pos_dev"] = cm[:, 0].copy()  # the kernel's device positions (equal when rows line up)
        tens = {k: torch.from_numpy(v) for k, v in tens.items()}
        tens["vec"] = tens["vec"].to(torch.bfloat16)  # the engine's dump format (INTERFACE.md section 6)
        save_file(tens, path)
        Path(path + ".json").write_text(json.dumps({"layers": LAYERS, "capture_layers": CAP_LAYERS,
                                                    "arms_sha": self.sha, "req_ids": self.req_ids,
                                                    "tags": self.tags}))
        n, m = len(self.coef), len(self.vec)
        self.reset_capture()
        return {"path": path, "coef_rows": n, "vec_rows": m}

    # ------------------------------------------------------------------------------------------------------------------
    def complete(self, body: dict, pending: list | None = None) -> dict:
        ids = [int(x) for x in body["prompt"]]
        P, N, n = len(ids), int(body.get("max_tokens", 16)), int(body.get("n", 1))
        xa = body.get("vllm_xargs") or {}
        arm = xa.get("arm", 0)
        if type(arm) is not int or not 0 <= arm < A:
            self.invalid += 1
            arm = 0
        if self.global_arm >= 0:
            arm = self.global_arm
        p0, p1 = int(xa.get("edit_from", 1)), int(xa.get("edit_to", MAX_EDIT_TO))
        s = self.sig(arm)
        salt = body.get("cache_salt") or ""
        blocks = [h(salt, tuple(ids[: (b + 1) * BLOCK])) for b in range(P // BLOCK)]
        cached = 0
        if body.get("prompt_logprobs") is None:
            for b in blocks:
                if b not in self.cache:
                    break
                cached += BLOCK
        cached = min(cached, P - 1)
        self.cache.update(blocks)
        req_id = f"cmpl-{h(time.time(), tuple(ids[:8]), salt)}"
        choices = []
        for k in range(n):
            gen = [(h(tuple(ids[-4:]), j, k, round(s, 3)) % 100000) for j in range(N)]
            seq = ids + gen
            if pending is not None and "capture" not in xa and N > DECODE_CHUNK:
                # Prompt now, then the decode positions chunk by chunk (the handler sleeps between chunks), so a
                # client polling tokens_seen sees the counter grow while requests decode (the PIECEWISE late join).
                self.account(arm, seq, cached, P, p0, p1, xa, f"{req_id}-{k}")
                for j in range(P, P + N - 1, DECODE_CHUNK):
                    pending.append((arm, seq, j, min(j + DECODE_CHUNK, P + N - 1), p0, p1, xa, f"{req_id}-{k}"))
            else:
                self.account(arm, seq, cached, P + N - 1, p0, p1, xa, f"{req_id}-{k}")
            ch = {"index": k, "text": "", "finish_reason": "length",
                  "logprobs": {"tokens": [f"token_id:{t}" for t in gen],
                               "token_logprobs": [self.lp(t, P + j, s) for j, t in enumerate(gen)]}}
            if body.get("prompt_logprobs") is not None:
                plp = [None]
                for i in range(1, P):
                    top = ids[i] if s == 0 or h(i, 7) % 5 else ids[i] + 1
                    d = {str(ids[i]): {"logprob": self.lp(ids[i], i, s), "rank": 1 if top == ids[i] else 2}}
                    if top != ids[i]:
                        d[str(top)] = {"logprob": -0.01, "rank": 1}
                    plp.append(d)
                ch["prompt_logprobs"] = plp
            choices.append(ch)
        return {"id": req_id, "choices": choices,
                "usage": {"prompt_tokens": P, "completion_tokens": N * n,
                          "prompt_tokens_details": {"cached_tokens": cached}}}

    def account(self, arm, seq, lo, hi, p0, p1, xa, req_id):
        """Counters (and capture rows) for computed positions ``lo .. hi - 1``."""
        pos = np.arange(lo, hi)
        self.seen[arm] += pos.size
        on = (self.t["a"][arm] != 0) | (self.t["g"][arm] != 0)
        win = (pos >= max(1, p0)) & (pos < p1)
        self.edited[arm][on] += int(win.sum())
        cap = xa.get("capture")
        if cap not in ("coef", "vec"):
            return
        if "cap_pos" in xa:
            sel = np.isin(pos, np.asarray(xa["cap_pos"]))
        else:
            sel = (pos >= int(xa.get("cap_from", 0))) & (pos < int(xa.get("cap_to", MAX_EDIT_TO)))
        tag = xa.get("tag", "")
        if tag not in self.tags:
            self.tags.append(tag)
        self.req_ids.append(req_id)
        t = self.t
        ep = np.einsum("isd,isd->is", t["p"][arm].astype(np.float64), t["e"][arm].astype(np.float64))
        shift = (h(tag) % 7) / 7.0
        for q, w in zip(pos[sel], win[sel]):
            c = self.rng.normal(5.0, 2.0, (I, 4))
            delta = (c.mean(1, keepdims=True) - t["mu"][arm].mean(1, keepdims=True)) if t["mc"][arm] else c - t["mu"][arm]
            a, g = t["a"][arm][:, None], t["g"][arm][:, None]
            cp = np.where(w & on[:, None], c - a * delta * ep + g * ep, c)
            self.coef.append(np.stack([c, cp, self.rng.random((I, 4))], 1))
            self.cmap.append([int(q), arm, len(self.req_ids) - 1, self.tags.index(tag)])
            if cap == "vec":
                self.vec.append(self.rng.standard_normal((len(CAP_LAYERS), 5120)) * 0.1 + shift)
                self.vmap.append([int(q), arm, len(self.req_ids) - 1, self.tags.index(tag)])


def main() -> None:
    from aiohttp import web

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18101)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--capture-dir", required=True)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    m = Mock(args.capture_dir)

    async def handle(request):
        if request.path == "/health":
            return web.Response(text="")
        if request.headers.get("Authorization") != f"Bearer {key}":  # every route but /health, as auth.py
            return web.json_response({"error": "unauthorized"}, status=401)
        if request.path == "/metrics":
            return web.Response(text='vllm:num_requests_running{model_name="dsv41"} 0.0\n'
                                     'vllm:num_requests_waiting{model_name="dsv41"} 0.0\n')
        if request.path == "/v1/models":
            return web.json_response({"data": [{"id": "dsv41", "max_model_len": 131072}]})
        if request.path == "/v1/completions":
            pending = []
            out = m.complete(await request.json(), pending)
            for chunk in pending:
                await asyncio.sleep(CHUNK_S)
                m.account(*chunk)
            return web.json_response(out)
        if request.path == "/collective_rpc":
            b = await request.json()
            try:
                return web.json_response({"results": m.rpc(b["method"], b.get("args", []))})
            except Exception as e:
                return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response({"error": "nope"}, status=404)

    app = web.Application(client_max_size=256 * 2**20)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
