"""HTTP client for the engine: completions, teacher-forced scoring, RPC wrappers, CLI.

Control routes go to ``DSV41_ENGINE_URL`` (default ``http://127.0.0.1:8101``) only. The API key is read from
``DSV41_KEY_FILE`` (default ``$DSV41_ROOT/.serve_key``) inside the process and never printed. Uses python urllib
(no curl dependency).

    python tools/client.py installed | stats [--reset] | load PATH | dump PATH | global-arm K
    python tools/client.py score --data data/floor_prompts.json [--arm K --arms-sha SHA] --out X.json
    python tools/client.py --url http://127.0.0.1:8100 floor --out $DSV41_LOG_DIR/engine/floor_native_<ts>.json
"""

import argparse
import hashlib
import json
import os
import random
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsv41_intervene import env_path  # noqa: E402
from tools.arms import record_global_arm, record_loaded_sha, request_fields  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8101"
MODEL = "dsv41"
N_RANKS = 4
# Per-rank fields that must agree across ranks whenever a method returns them.
AGREE = ("layers", "arms_sha", "checksums")
NAN_ERROR = "Out of range float values are not JSON compliant: nan"
NAN_RETRIES = 8


class EngineError(RuntimeError):
    pass


class EngineClient:
    """Thin client; ``rpc`` asserts 4 ranks agree on ``layers``, ``arms_sha`` and checksums.

    ``nan`` counts teacher-forced requests the server answered with a NaN error and that were retried:
    ``requests``, ``retried``, ``retries`` and ``retried_c1`` (retries in ``score`` calls made at concurrency 1)."""

    def __init__(self, url: str | None = None, key_file: str | None = None, timeout: float = 600.0):
        self.url = (url or os.environ.get("DSV41_ENGINE_URL") or DEFAULT_URL).rstrip("/")
        kf = key_file or env_path("DSV41_KEY_FILE", ".serve_key", required=False) or "none"  # no key: requests get 401
        self._key = Path(kf).read_text().strip() if kf != "none" and Path(kf).is_file() else ""
        self.timeout = timeout
        self.n_ranks = int(os.environ.get("DSV41_N_RANKS", N_RANKS))
        self._nan_lock = threading.Lock()
        self.reset_nan()

    def reset_nan(self) -> dict:
        """Return the NaN-retry counts since the last reset and zero them."""
        old = getattr(self, "nan", None)
        self.nan = {"requests": 0, "retried": 0, "retries": 0, "retried_c1": 0}
        return old

    def __repr__(self) -> str:  # never show the key
        return f"EngineClient({self.url!r}, key={'set' if self._key else 'none'})"

    def request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None,
                ok: tuple[int, ...] = (200,)) -> tuple[int, dict | str]:
        """One HTTP call; returns ``(status, json or text)``; raises ``EngineError`` on a status not in ``ok``."""
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        try:
            out = json.loads(raw)
        except ValueError:
            out = raw.decode(errors="replace")
        if ok and status not in ok:
            raise EngineError(f"{method} {path} -> {status}: {str(out)[:500]}")
        return status, out

    def rpc(self, method: str, *args: str) -> list[dict]:
        """``POST /collective_rpc``; returns the per-rank dicts."""
        _, out = self.request("POST", "/collective_rpc", {"method": method, "args": [str(a) for a in args],
                                                          "timeout": self.timeout})
        res = out["results"] if isinstance(out, dict) else None
        if not isinstance(res, list) or len(res) != self.n_ranks or not all(isinstance(r, dict) for r in res):
            raise EngineError(f"{method}: expected {self.n_ranks} dict results, got {str(out)[:500]}")
        for k in AGREE:
            vals = [json.dumps(r[k], sort_keys=True) for r in res if k in r]
            if vals and len(set(vals)) != 1:
                raise EngineError(f"{method}: ranks disagree on {k}: {vals}")
        return sorted(res, key=lambda r: r.get("rank", 0))

    def installed(self) -> dict:
        return self.rpc("dsv41_installed")[0]

    def stats(self, reset: bool = False) -> dict:
        """Rank 0's counters (all ranks run the same edits on the replicated stack), plus the per-rank list."""
        res = self.rpc("dsv41_stats", "1" if reset else "0")
        return dict(res[0], ranks=res)

    def load_arms(self, path: str) -> dict:
        """Load an arms file on all ranks and record ``<boot> <sha>`` for the proxy (``$DSV41_ARMS_DIR``).
        Always load through this method: the proxy accepts only salts of shas recorded for the current boot."""
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        res = self.rpc("dsv41_load_arms", str(Path(path).resolve()), sha)
        if res[0]["arms_sha"] != sha:
            raise EngineError(f"load_arms: engine reports {res[0]['arms_sha']}, file is {sha}")
        record_loaded_sha(res[0].get("boot", ""), sha)
        return res[0]

    def dump(self, path: str) -> dict:
        return self.rpc("dsv41_capture_dump", path)[0]

    def set_global_arm(self, k: int) -> dict:
        """Set the global arm on all ranks and record it for the proxy (which then passes only that arm's traffic)."""
        out = self.rpc("dsv41_set_global_arm", str(int(k)))[0]
        record_global_arm(out.get("boot", ""), out["global_arm"])
        return out

    def metrics(self) -> dict[str, float]:
        """``GET /metrics``: the sums of ``vllm:num_requests_running`` and ``vllm:num_requests_waiting``."""
        _, text = self.request("GET", "/metrics")
        out = {"running": 0.0, "waiting": 0.0}
        for line in str(text).splitlines():
            name = line.split("{", 1)[0].split(" ", 1)[0]
            for k in out:
                if name == f"vllm:num_requests_{k}":
                    out[k] += float(line.rsplit(" ", 1)[1])
        return out

    def wait_idle(self, timeout: float = 15.0) -> dict:
        """Wait until the engine runs and queues no request (exact counters need no stray traffic); raises
        ``EngineError`` with "stray traffic" if it is still busy after ``timeout`` seconds."""
        t0 = time.time()
        while True:
            m = self.metrics()
            if m["running"] == 0 and m["waiting"] == 0:
                return m
            if time.time() - t0 > timeout:
                raise EngineError(f"stray traffic: {m} after {timeout:.0f} s; counted blocks need an idle engine")
            time.sleep(0.5)

    def complete(self, body: dict) -> dict:
        """``POST /v1/completions`` with the key."""
        return self.request("POST", "/v1/completions", {"model": MODEL, **body})[1]

    def generate(self, prompt: list[int], fields: dict, max_tokens: int, n: int = 1, unique: bool = True) -> dict:
        """Greedy (n=1) or sampled generation with decode logprobs; returns the response plus ``gen`` (token ids
        and logprobs per choice). Refuses a capture with ``n > 1``: the children share the salt, so only the first
        child computes (and captures) the full prompt blocks."""
        if n > 1 and "capture" in (fields.get("vllm_xargs") or {}):
            raise ValueError("capture with n > 1 is not supported (children share prompt blocks)")
        fields = fresh_salt(fields) if unique else fields
        body = {"prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True, "logprobs": 1, "n": n,
                "temperature": 0.0 if n == 1 else 1.0, "return_tokens_as_token_ids": True, **fields}
        out = self.complete(body)
        gen = []
        for ch in sorted(out["choices"], key=lambda c: c["index"]):
            lp = ch["logprobs"]
            gen.append({"ids": [int(t.split(":", 1)[1]) for t in lp["tokens"]],
                        "lp": [float(x) for x in lp["token_logprobs"]]})
        out["gen"] = gen
        return out

    def score(self, prompts: list[list[int]], fields: dict, concurrency: int = 64) -> list[dict]:
        """Teacher-forced: ``max_tokens 1``, ``prompt_logprobs 1``, ``temperature 0``, per-request unique salt.
        Returns per prompt the logprob of each actual next token and the top-1 flag."""

        def one(ids: list[int]) -> dict:
            for nan_retries in range(NAN_RETRIES + 1):
                body = {"prompt": ids, "max_tokens": 1, "prompt_logprobs": 1, "temperature": 0.0,
                        **fresh_salt(fields)}
                try:
                    out = self.complete(body)
                    break
                except EngineError as e:  # the server cannot serialize a NaN logprob (seen under batching, 28 Oct)
                    if NAN_ERROR not in str(e) or nan_retries == NAN_RETRIES:
                        raise
            with self._nan_lock:
                self.nan["requests"] += 1
                if nan_retries:
                    self.nan["retried"] += 1
                    self.nan["retries"] += nan_retries
                    self.nan["retried_c1"] += int(concurrency == 1)
            return dict(parse_prompt_logprobs(ids, out["choices"][0]["prompt_logprobs"]), id=out.get("id"),
                        usage=out.get("usage"), nan_retries=nan_retries)

        with ThreadPoolExecutor(max(1, min(concurrency, len(prompts)))) as ex:
            return list(ex.map(one, prompts))


def fresh_salt(fields: dict) -> dict:
    """Copy of ``fields`` with a per-request unique ``cache_salt`` (forces a full recompute)."""
    base = fields.get("cache_salt", "")
    salt = hashlib.sha256(f"{base}|fresh|{secrets.token_hex(8)}".encode()).hexdigest()[:24]
    return {**fields, "cache_salt": salt}


def parse_prompt_logprobs(ids: list[int], plp: list) -> dict:
    """vLLM ``prompt_logprobs`` (``[None, {tok: {logprob, rank, ...}}, ...]``) -> ``lp`` of the actual token,
    ``top1`` token id and ``hit`` (actual token is top-1) for positions 1..n-1."""
    if len(plp) != len(ids):
        raise EngineError(f"prompt_logprobs has {len(plp)} entries for {len(ids)} tokens")
    lp, top1 = np.empty(len(ids) - 1, np.float64), np.empty(len(ids) - 1, np.int64)
    for i in range(1, len(ids)):
        d = plp[i]
        lp[i - 1] = d[str(ids[i])]["logprob"]
        top1[i - 1] = int(next(k for k, v in d.items() if v["rank"] == 1)) if any(
            v["rank"] == 1 for v in d.values()) else -1
    return {"lp": lp, "top1": top1, "hit": top1 == np.asarray(ids[1:])}


# ----------------------------------------------------------------------------------------------------------------------
# Comparisons against the noise floor (docs/GATES.md)
# ----------------------------------------------------------------------------------------------------------------------

def compare(x: list[dict], y: list[dict], kinds: list[str], sl: list[slice] | None = None) -> dict:
    """Pooled ``|Δ lp|`` (median, p95, max) and top-1 agreement between two scored runs, per kind and ``all``.
    ``sl`` optionally restricts each prompt to a slice of positions."""
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for i, (a, b) in enumerate(zip(x, y)):
        s = sl[i] if sl else slice(None)
        d, agree = np.abs(a["lp"][s] - b["lp"][s]), a["top1"][s] == b["top1"][s]
        for k in (kinds[i], "all"):
            groups.setdefault(k, []).append((d, agree))
    out = {}
    for k, parts in groups.items():
        d = np.concatenate([p[0] for p in parts])
        ag = np.concatenate([p[1] for p in parts])
        out[k] = {"n_pos": int(d.size), "n_prompts": len(parts), "median": float(np.median(d)),
                  "p95": float(np.percentile(d, 95)), "max": float(d.max()), "mean": float(d.mean()),
                  "top1_agree": float(ag.mean())}
    return out


def sub_kind(prompt_id: str) -> str:
    """``code``, ``prose`` or ``agent`` from a floor prompt id (``generic_code_00``, ``generic_prose_01``, ``agent_*``)."""
    for k in ("code", "prose"):
        if f"_{k}_" in prompt_id:
            return k
    return "agent" if prompt_id.startswith("agent") else "other"


def floor_reference(floor: dict) -> dict:
    """Per kind, the worse of (a vs b) and (a vs c): the run-to-run noise including batch-composition changes."""
    ref = {}
    for k in floor["a_vs_b"]:
        ab, ac = floor["a_vs_b"][k], floor["a_vs_c"][k]
        ref[k] = {"median": max(ab["median"], ac["median"]), "p95": max(ab["p95"], ac["p95"]),
                  "top1_agree": min(ab["top1_agree"], ac["top1_agree"])}
    return ref


def within_floor(cmp: dict, ref: dict) -> dict:
    """Floor rule: median <= 1.5x floor median, p95 <= 2x floor p95, top-1 >= floor - 1 point, for every kind."""
    rows, ok = {}, True
    for k, c in cmp.items():
        r = ref.get(k, ref["all"])
        row = {"median_ok": c["median"] <= 1.5 * r["median"], "p95_ok": c["p95"] <= 2 * r["p95"],
               "top1_ok": c["top1_agree"] >= r["top1_agree"] - 0.01}
        row["ok"] = all(row.values())
        ok &= row["ok"]
        rows[k] = row
    return {"ok": bool(ok), "per_kind": rows}


def save_runs(path: str, runs: dict[str, list[dict]], ids: list[str]) -> None:
    """Raw scored runs as one ``.npz`` (concatenated arrays + offsets), for later gates."""
    arrs = {"prompt_ids": np.array(ids)}
    for name, run in runs.items():
        arrs[f"{name}_off"] = np.cumsum([0] + [r["lp"].size for r in run])
        arrs[f"{name}_lp"] = np.concatenate([r["lp"] for r in run])
        arrs[f"{name}_top1"] = np.concatenate([r["top1"] for r in run])
    np.savez_compressed(path, **arrs)


def load_runs(path: str) -> dict[str, list[dict]]:
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.endswith("_off"):
            name, off = k[:-4], z[k]
            lp, t1 = z[f"{name}_lp"], z[f"{name}_top1"]
            out[name] = [{"lp": lp[a:b], "top1": t1[a:b]} for a, b in zip(off[:-1], off[1:])]
    return out


def run_floor(client: EngineClient, data: dict, out: str, fields: dict | None = None, seed: int = 0) -> dict:
    """Native floor: (a) all at once, (b) again shuffled, (c) one by one; floor = (a vs b) and (a vs c)."""
    fields = fields or {}
    prompts = data["prompts"]
    ids, kinds = [p["ids"] for p in prompts], [p["kind"] for p in prompts]
    t0 = time.time()
    a = client.score(ids, fields, concurrency=len(ids))
    t1 = time.time()
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    b_sh = client.score([ids[i] for i in order], fields, concurrency=len(ids))
    b = [None] * len(ids)
    for j, i in enumerate(order):
        b[i] = b_sh[j]
    t2 = time.time()
    c = client.score(ids, fields, concurrency=1)
    t3 = time.time()
    res = {"url": client.url, "fields": fields, "tokenizer_sha256": data.get("tokenizer_sha256"),
           "n_prompts": len(ids), "n_tokens": int(sum(len(x) for x in ids)),
           "kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
           "seconds": {"a": t1 - t0, "b": t2 - t1, "c": t3 - t2},
           "nan_retries": {name: {p["id"]: r["nan_retries"] for p, r in zip(prompts, run) if r["nan_retries"]}
                           for name, run in (("a", a), ("b", b), ("c", c))},
           "a_vs_b": compare(a, b, kinds), "a_vs_c": compare(a, c, kinds), "b_vs_c": compare(b, c, kinds),
           "sub_kinds": {name: compare(x, y, [sub_kind(p["id"]) for p in prompts])
                         for name, x, y in (("a_vs_b", a, b), ("a_vs_c", a, c), ("b_vs_c", b, c))},
           "per_prompt_a_vs_b": [{"id": p["id"], "kind": p["kind"], "n": len(p["ids"]),
                                  "median": float(np.median(np.abs(x["lp"] - y["lp"]))),
                                  "top1_agree": float((x["top1"] == y["top1"]).mean())}
                                 for p, x, y in zip(prompts, a, b)],
           "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    res["reference"] = floor_reference(res)
    res["a_vs_b_top1_below_90"] = res["a_vs_b"]["all"]["top1_agree"] < 0.90
    # one-by-one requests never hit the batched NaN path on the plain server (0/40, 28 Oct): a retry there is a signal
    res["c_nan_free"] = not res["nan_retries"]["c"]
    runs_path = str(Path(out).with_suffix("")) + ".runs.npz"
    save_runs(runs_path, {"a": a, "b": b, "c": c}, [p["id"] for p in prompts])
    res["runs"] = runs_path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    """CLI: ``installed``, ``stats [--reset]``, ``load PATH``, ``dump PATH``, ``global-arm K``, ``score``, ``floor``."""
    ap = argparse.ArgumentParser(description="DeepSeek V4.1 engine client")
    ap.add_argument("--url", default=None, help=f"server (default $DSV41_ENGINE_URL or {DEFAULT_URL})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("installed")
    s = sub.add_parser("stats")
    s.add_argument("--reset", action="store_true")
    for name in ("load", "dump"):
        sub.add_parser(name).add_argument("path")
    sub.add_parser("global-arm").add_argument("k", type=int)
    for name in ("score", "floor"):
        s = sub.add_parser(name)
        s.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "floor_prompts.json"))
        s.add_argument("--arm", type=int, default=0)
        s.add_argument("--arms-sha", default="")
        s.add_argument("--edit-from", type=int, default=1)
        s.add_argument("--edit-to", type=int, default=2**31 - 1)
        s.add_argument("--out", required=True)
    args = ap.parse_args()
    c = EngineClient(args.url)
    if args.cmd == "installed":
        out = c.installed()
    elif args.cmd == "stats":
        out = c.stats(args.reset)
    elif args.cmd == "load":
        out = c.load_arms(args.path)
    elif args.cmd == "dump":
        out = c.dump(args.path)
    elif args.cmd == "global-arm":
        out = c.set_global_arm(args.k)
    else:
        data = json.loads(Path(args.data).read_text())
        fields = request_fields(args.arm, args.edit_from, args.edit_to, arms_sha=args.arms_sha)
        if args.cmd == "floor":
            res = run_floor(c, data, args.out, fields)
            out = {k: res[k] for k in ("a_vs_b", "a_vs_c", "reference", "seconds", "runs", "a_vs_b_top1_below_90",
                                       "nan_retries", "c_nan_free")}
        else:
            runs = c.score([p["ids"] for p in data["prompts"]], fields)
            save_runs(args.out, {"run": runs}, [p["id"] for p in data["prompts"]])
            out = {"out": args.out, "n": len(runs)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
