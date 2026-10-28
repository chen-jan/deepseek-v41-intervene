#!/usr/bin/env python3
"""Multi-layer O-lens readouts for DeepSeek V4.1 Flash: capture the base model's stream means at several layers, then
read them with the release lens (one layer) and/or a multilayer lens (several layers). Background: docs/MULTILAYER.md.

NOT RUN END TO END FROM THIS REPO. The 28 Oct 2026 readouts came from a private harness with the same maths; this file
is a cleaned rewrite of it and was only exercised on CPU with a stub engine (``--check`` and the code paths below).
Read a few positions and compare them with your own lens setup before a large run.

Needs the lens engine, not this repo's edit engine: the vllm-metamodels DeepSeek V4.1 build (our runs used build
``7ab496a``) and the usage modules that come with it, ``olens_deepseek_v41_flash_usage`` and ``dsv41_engine``, in one
folder (``--lens-engine-dir``), in a venv where that build is installed. Nothing of them is included here; they are
imported at runtime. The engine takes all 4 GPUs (TP 4), so stop the edit engine and the plain server first: the tool
refuses to boot while a GPU holds more than ``--max-gpu-used-mib``.

Steps
 1. Capture: unmerged base weights, one early-exit forward per transcript, the stream mean (mean of the 4 mHC streams)
    at every ``--layers`` layer and requested position. Saved bf16 to <out>/h.pt ``[N, len(layers), 5120]`` with the
    layers and the shas of the positions and inputs; a rerun reuses it only when all three match.
 2. Release lens (``--release-lens``, at ``--release-layer``): merge the lens and inject the captured vector at the
    marker of the usage module's prompt with its ``steer_embed(v, marker, ALPHA)``.
 3. Multilayer lens (``--multilayer-lens``, at ``--ml-layers``): merge the lens; per layer L its input head maps the
    vector to what is injected, as the lens was trained: ``v <- v / |v|`` (``input_norm: unit``), ``u = W_L v + b_L``,
    then ``steer_embed(u, marker_L, alpha)`` puts ``alpha * u / |u|`` at the marker of the layer-tagged prompt
    (``input_heads.json`` ``prompts[L]``). With both lenses, ``--sanity`` random positions are also read with the
    multilayer head at the release layer, to compare the two lenses on the same vectors.
 4. Per position: greedy plus ``--samples`` samples at T = 1 (top-p 1), each one bullet, constrained by the usage
    module's bullet grammar (``GrammarSpec(cap=--bullet-cap)``). Every request must report that its injection landed.
Non-finite vectors are flagged (``{"i": i, "nonfinite": true}``) and never read. Readouts go to
<out>/readouts_<stage>.jsonl (``{"i", "greedy", "samples"}``, i = row of the positions file); a rerun skips rows already
there (and refuses when that stage's lens or settings changed). <out>/report.json holds shas, settings, counts, times.

Inputs
  --positions P   {"reads": [{"key": <prompt key>, "pos": <0-based token position>, ...}, ...]} or a bare list
  --items I       {"items": [{"key": ...}, ...], "npz": "<file next to I>"}; the npz holds ``ids`` (all prompts
                  concatenated) and ``off`` (``[n + 1]`` offsets)
  --token-ids T   instead of --items: {"<key>": [ids], ...} (JSON) or JSONL lines {"key": ..., "ids": [...]}
  Every prompt starts with BOS (id 0).

    python tools/lens_read.py --check --positions pos.json --token-ids prompts.json --layers 22,26,30,34 \\
        --release-lens <release lens dir> --multilayer-lens <multilayer lens dir> --out <out dir>
    python tools/lens_read.py --positions pos.json --token-ids prompts.json --layers 22,26,30,34 \\
        --release-lens <dir> --multilayer-lens <dir> --lens-engine-dir <dir> --model-dir $DSV41_MODEL_DIR --out <dir>

``--check`` validates inputs and lens folders on CPU and prints the plan; it imports neither vLLM nor the lens engine.
Paths can also come from the environment: DSV41_MODEL_DIR, DSV41_LENS_ENGINE_DIR, DSV41_RELEASE_LENS,
DSV41_MULTILAYER_LENS.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

BOS_ID = 0
HIDDEN = 5120
SEED_STRIDE = 64  # seed = (row * 1_000_003) * 64 + draw, so draws 0..63 of a row never share a seed


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_files(*paths: str) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------------

def load_inputs(a) -> tuple[list[dict], dict[str, np.ndarray], str, str]:
    """(reads, {key: ids}, positions sha, inputs sha); checks keys, positions and BOS."""
    doc = json.load(open(a.positions))
    reads = doc["reads"] if isinstance(doc, dict) else doc
    if a.items:
        head = json.load(open(a.items))
        npz = os.path.join(os.path.dirname(os.path.abspath(a.items)), head["npz"])
        z = np.load(npz)
        ids, off = z["ids"], z["off"]
        prompts = {it["key"]: ids[off[i]:off[i + 1]] for i, it in enumerate(head["items"])}
        in_sha = sha256_files(a.items, npz)
    else:
        with open(a.token_ids) as f:
            text = f.read()
        try:
            d = json.loads(text)
        except json.JSONDecodeError:  # JSONL
            d = None
        if isinstance(d, dict) and set(d) != {"key", "ids"}:
            prompts = {k: np.asarray(v, np.int64) for k, v in d.items()}
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            prompts = {r["key"]: np.asarray(r["ids"], np.int64) for r in rows}
        in_sha = sha256_files(a.token_ids)
    errors = []
    for i, r in enumerate(reads):
        p = prompts.get(r["key"])
        if p is None:
            errors.append(f"read {i}: no prompt {r['key']!r}")
        elif not 0 <= int(r["pos"]) < len(p):
            errors.append(f"read {i}: position {r['pos']} outside {r['key']!r} ({len(p)} tokens)")
    for k in {r["key"] for r in reads} & set(prompts):
        if len(prompts[k]) == 0 or int(prompts[k][0]) != BOS_ID:
            errors.append(f"prompt {k!r} does not start with BOS ({BOS_ID})")
    if errors:
        raise SystemExit(f"{len(errors)} input errors, e.g.: " + "; ".join(errors[:5]))
    return reads, prompts, sha256_files(a.positions), in_sha


class MLLens:
    """A multilayer lens's input heads (``input_heads.json`` + ``input_heads.safetensors``), single-stream mean kind."""

    def __init__(self, d: str):
        from safetensors.torch import load_file

        self.meta = json.load(open(os.path.join(d, "input_heads.json")))
        if int(self.meta.get("streams", 1)) != 1 or self.meta.get("kind", "mean") != "mean":
            raise SystemExit(f"{d}: only single-stream mean heads are supported, got {self.meta.get('streams')} "
                             f"streams, kind {self.meta.get('kind')!r}")
        self.sd = {k: v.float() for k, v in load_file(os.path.join(d, "input_heads.safetensors")).items()}
        self.prompts = {int(k): ([int(t) for t in v["ids"]], int(v["marker_pos"]))
                        for k, v in self.meta["prompts"].items()}
        self.alpha = float(self.meta["alpha"])
        if self.meta.get("input_norm") != "unit":  # the only kind we used; others may need a scale we do not know
            raise SystemExit(f"{d}: input_norm {self.meta.get('input_norm')!r} is not supported (only 'unit')")
        self.sha = sha256_files(os.path.join(d, "input_heads.json"), os.path.join(d, "input_heads.safetensors"))

    def layers(self) -> list[int]:
        return sorted(L for L in self.prompts if f"layer{L}.weight" in self.sd)

    def head(self, v: torch.Tensor, L: int) -> torch.Tensor:
        """``[n, 5120]`` captured vectors -> ``[n, d]`` injections before ``steer_embed`` (NaN in, NaN out)."""
        v = F.normalize(v.float(), dim=-1)
        return v @ self.sd[f"layer{L}.weight"].T + self.sd[f"layer{L}.bias"]


# ---------------------------------------------------------------------------------------------------------------------
# the lens engine (vllm-metamodels DeepSeek V4.1 build, imported at runtime)
# ---------------------------------------------------------------------------------------------------------------------

def gpu_preflight(max_used_mib: int) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used = [int(float(line.split(",")[1])) for line in out.strip().splitlines() if line.strip()]
    if any(u > max_used_mib for u in used):
        raise SystemExit(f"GPU memory in use {used} MiB (> {max_used_mib}): stop the server holding the GPUs first")
    return used


class LensEngine:
    """The usage module's ``DSV41OlensEngine`` at TP 4, set up like the fork's DeepSeek benchmark: FLASHMLA_SPARSE_DSV41
    attention (the default FP8 attention path needs ``wq_b`` scales the bf16 LoRA targets lack) and vLLM's custom
    all-reduce off (its fused all-reduce + mHC path defers the MoE finalize, which the fork's capture refuses)."""

    def __init__(self, a):
        self.info = {"gpu_mem_before_mib": gpu_preflight(a.max_gpu_used_mib)}
        for k, v in (("VLLM_LENS_CUDA_GRAPHS", "0"), ("VLLM_ALLOW_INSECURE_SERIALIZATION", "1"),
                     ("VLLM_USE_V2_MODEL_RUNNER", "0"), ("VLLM_LENS_MERGE_ALLOW_TP", "1"), ("VLLM_LENS_SHM", "0"),
                     ("DSV41_GRAMMAR_CACHE", os.path.join(tempfile.gettempdir(), "dsv41_grammar_v3.pt"))):
            os.environ.setdefault(k, v)
        if a.lens_engine_dir not in sys.path:
            sys.path.insert(0, a.lens_engine_dir)
        import olens_deepseek_v41_flash_usage as U  # noqa: E402

        U._stack(a.model_dir)
        import dsv41_engine as E  # noqa: E402  (sets the fork's environment; must come before vllm)

        base_kwargs = E.engine_kwargs

        def engine_kwargs(*args, **kwargs):
            kw = base_kwargs(*args, **kwargs)
            if a.attn_backend:
                kw["attention_config"] = {"backend": a.attn_backend}
            if not a.custom_all_reduce:
                kw["disable_custom_all_reduce"] = True
            return kw

        E.engine_kwargs = engine_kwargs
        t0 = time.time()
        self.eng = U.DSV41OlensEngine(a.model_dir, tp=a.tp, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len)
        self.E, self.U = E, U
        builds = sorted(glob.glob(os.path.join(a.lens_engine_dir, "vllm_metamodels*.tar.gz")))
        self.info.update({"boot_s": round(time.time() - t0, 1), "tp": a.tp, "gpu_mem": a.gpu_mem,
                          "attention_backend": a.attn_backend, "custom_all_reduce": a.custom_all_reduce,
                          "fork_builds": {os.path.basename(b): sha256_files(b) for b in builds}})

    def capture(self, prompts: list[list[int]], positions: list[list[int]], layers: list[int]) -> list[torch.Tensor]:
        """Stream means ``[len(layers), len(pos), 5120]`` per prompt, one early-exit forward each (unmerged model)."""
        llm = self.eng.llm
        self.E.set_mode(llm, "mean")
        uniq = [sorted(set(int(p) for p in pos)) for pos in positions]
        _, acts, _ = self.E.run_capture(llm, prompts, uniq, [int(L) for L in layers], early_exit=True)
        out = []
        for x, u, pos in zip(acts, uniq, positions):
            where = {p: i for i, p in enumerate(u)}
            out.append(x[:, [where[int(p)] for p in pos]].float().cpu())
        return out

    def merge(self, lens_dir: str | None) -> dict:
        res = self.eng.set_lens(lens_dir) or {}
        return {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool, type(None)))}

    def release_prompt(self) -> tuple[list[int], int, float]:
        return list(self.eng.prompt_ids), int(self.eng.mpos), float(self.U.ALPHA)

    def read(self, vecs: torch.Tensor, rows: list[int], prompt_ids: list[int], marker: int, alpha: float,
             n_samples: int, cap: int) -> list[tuple[str, list[str]]]:
        """Greedy + ``n_samples`` one-bullet readouts per vector, in one generate call."""
        from vllm import SamplingParams

        eng, U = self.eng, self.U
        gram = eng.C.GrammarSpec(cap=cap)
        reqs, sps = [], []
        for v, row in zip(vecs, rows):
            sv = U.steer_embed(v, marker, alpha)
            for g in range(1 + n_samples):
                reqs.append({"prompt_token_ids": prompt_ids})
                sps.append(SamplingParams(
                    n=1, temperature=0.0 if g == 0 else 1.0, top_p=1.0, max_tokens=gram.max_gen,
                    seed=(row * 1_000_003) * SEED_STRIDE + g, stop_token_ids=[U.TERM_ID], skip_special_tokens=False,
                    detokenize=False, extra_args={"apply_steering_vectors": [sv], "bullet_grammar": gram.to_dict()}))
        eng._stats()  # reset the per-rank injection counters
        outs = eng.llm.generate(reqs, sps, use_tqdm=False)
        replaced = [int(x.get("rows_replaced", 0)) for x in eng._stats()]
        if not all(r == len(reqs) for r in replaced):
            raise RuntimeError(f"injection did not land on every request: rows_replaced {replaced} != {len(reqs)}")
        texts = [eng.C.bullet_text(eng.tok, [int(t) for t in o.outputs[0].token_ids], cap) for o in outs]
        k = 1 + n_samples
        return [(texts[i * k], texts[i * k + 1:(i + 1) * k]) for i in range(len(rows))]


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

def plan(a, reads, prompts, ml: MLLens | None) -> dict:
    """Stages and their layers; raises SystemExit on a layer the lenses cannot read."""
    layers = list(a.layers)
    st = {}
    if a.release_lens:
        if a.release_layer not in layers:
            raise SystemExit(f"--release-layer {a.release_layer} is not in --layers {layers}")
        st[f"rel{a.release_layer}"] = a.release_layer
    if ml is not None:
        ml_layers = a.ml_layers if a.ml_layers is not None else [L for L in layers if not (
            a.release_lens and L == a.release_layer)]
        bad = [L for L in ml_layers if L not in layers or L not in ml.layers()]
        if bad:
            raise SystemExit(f"--ml-layers {bad}: not captured (--layers {layers}) or no input head/prompt in the "
                             f"multilayer lens (has {ml.layers()})")
        if a.release_lens and a.sanity and a.release_layer in ml.layers():
            st[f"ml{a.release_layer}_sanity"] = a.release_layer
        for L in ml_layers:
            st[f"ml{L}"] = L
    if not st:
        raise SystemExit("nothing to read: give --release-lens and/or --multilayer-lens")
    keys = sorted({r["key"] for r in reads})
    n_rows = sum(min(a.sanity, len(reads)) if s.endswith("_sanity") else len(reads) for s in st)
    return {"reads": len(reads), "prompts": len(keys), "tokens": int(sum(len(prompts[k]) for k in keys)),
            "layers": layers, "stages": st, "samples": a.samples, "bullets_max": n_rows * (1 + a.samples)}


def capture_all(be, reads, prompts, layers, chunk_tokens: int) -> torch.Tensor:
    """``[N, len(layers), 5120]`` bf16, rows in positions-file order; prompts batched up to ``chunk_tokens`` tokens."""
    by_key: dict[str, list[int]] = {}
    for i, r in enumerate(reads):
        by_key.setdefault(r["key"], []).append(i)
    H = torch.zeros(len(reads), len(layers), HIDDEN, dtype=torch.bfloat16)
    keys = list(by_key)
    batch, ntok = [], 0
    for k in keys + [None]:
        n = len(prompts[k]) if k is not None else 0
        if batch and (k is None or ntok + n > chunk_tokens):
            caps = be.capture([[int(t) for t in prompts[b]] for b in batch],
                              [[int(reads[i]["pos"]) for i in by_key[b]] for b in batch], layers)
            for b, cap in zip(batch, caps):  # cap [n_layers, n_pos, H]
                H[by_key[b]] = cap.permute(1, 0, 2).to(torch.bfloat16)
            log(f"captured {sum(len(by_key[b]) for b in batch)} positions of {len(batch)} prompts")
            batch, ntok = [], 0
        if k is not None:
            batch.append(k)
            ntok += n
    return H


def run(a, be=None) -> dict:
    reads, prompts, pos_sha, in_sha = load_inputs(a)
    ml = MLLens(a.multilayer_lens) if a.multilayer_lens else None
    P = plan(a, reads, prompts, ml)
    if a.check:
        P["release_lens_exists"] = bool(a.release_lens and os.path.isdir(a.release_lens))
        print(json.dumps(P, indent=1))
        return P
    os.makedirs(a.out, exist_ok=True)
    R = {"script_sha256": sha256_files(os.path.abspath(__file__)), "positions_sha256": pos_sha, "inputs_sha256": in_sha,
         "release_lens": a.release_lens, "multilayer_lens": a.multilayer_lens,
         "multilayer_heads_sha256": ml.sha if ml else None, "plan": P, "bullet_cap": a.bullet_cap,
         "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "stages": {}}

    def save():
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(R, f, indent=1)

    be = be or LensEngine(a)
    R["engine"] = be.info
    layers = list(a.layers)
    hpath = os.path.join(a.out, "h.pt")
    if os.path.exists(hpath):  # 1. capture, or reuse a matching earlier capture
        d = torch.load(hpath)
        for k, want in (("layers", layers), ("positions_sha256", pos_sha), ("inputs_sha256", in_sha)):
            if d.get(k) != want:
                raise SystemExit(f"{hpath} holds {k} {d.get(k)!r}, this run has {want!r}: use another --out")
        H = d["h"]
        R["stages"]["capture"] = {"reused": hpath}
    else:
        t0 = time.time()
        be.merge(None)
        H = capture_all(be, reads, prompts, layers, a.chunk_tokens)
        torch.save({"h": H, "layers": layers, "positions_sha256": pos_sha, "inputs_sha256": in_sha}, hpath)
        bad = (~torch.isfinite(H.float()).all(-1).all(-1)).nonzero().flatten().tolist()
        R["stages"]["capture"] = {"seconds": round(time.time() - t0, 1), "positions": len(reads),
                                  "nonfinite_rows": len(bad), "nonfinite_list": bad[:1000]}
        save()

    def stage(name: str, vecs: torch.Tensor, prompt_ids: list[int], marker: int, alpha: float, lens_id: str,
              rows: list[int] | None = None):
        path = os.path.join(a.out, f"readouts_{name}.jsonl")
        setting = {"lens": lens_id, "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids).encode()).hexdigest(), "marker": marker, "alpha": alpha, "samples": a.samples,
            "bullet_cap": a.bullet_cap, "h_positions_sha256": pos_sha}
        mpath = path + ".settings.json"
        if os.path.exists(mpath) and json.load(open(mpath)) != setting:
            raise SystemExit(f"{path} was written with other settings ({mpath}): use another --out")
        with open(mpath, "w") as f:
            json.dump(setting, f, indent=1)
        rows = list(range(len(vecs))) if rows is None else rows
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(line)["i"] for line in f if line.strip()}
        ok = torch.isfinite(vecs).all(-1)
        todo = [i for i in rows if i not in done]
        bad = [i for i in todo if not bool(ok[i])]
        todo = [i for i in todo if bool(ok[i])]
        t0, nb = time.time(), 0
        with open(path, "a") as f:
            for i in bad:
                f.write(json.dumps({"i": i, "nonfinite": True}) + "\n")
            for c in range(0, len(todo), a.chunk_vecs):
                ix = todo[c:c + a.chunk_vecs]
                t1 = time.time()
                for i, (g, s) in zip(ix, be.read(vecs[ix], ix, prompt_ids, marker, alpha, a.samples, a.bullet_cap)):
                    f.write(json.dumps({"i": i, "greedy": g, "samples": s}) + "\n")
                f.flush()
                nb += len(ix) * (1 + a.samples)
                log(f"{name}: {c + len(ix)}/{len(todo)} positions, "
                    f"{len(ix) * (1 + a.samples) / max(time.time() - t1, 1e-9):.0f} bullets/s")
        R["stages"][name] = {"layer": P["stages"][name], "positions": len(todo), "skipped_done": len(done),
                             "nonfinite": len(bad), "bullets": nb, "seconds": round(time.time() - t0, 1)}
        save()

    if a.release_lens:  # 2. release lens
        R["merge_release"] = be.merge(a.release_lens)
        pid, mp, alpha = be.release_prompt()
        L = a.release_layer
        stage(f"rel{L}", H[:, layers.index(L)].float(), pid, mp, alpha, os.path.realpath(a.release_lens))
    if ml is not None:  # 3. multilayer lens (+ sanity sample at the release layer)
        R["merge_multilayer"] = be.merge(a.multilayer_lens)
        for name, L in P["stages"].items():
            if not name.startswith("ml"):
                continue
            rows = None
            if name.endswith("_sanity"):
                rng = np.random.default_rng(0)
                rows = sorted(rng.choice(len(reads), size=min(a.sanity, len(reads)), replace=False).tolist())
            pid, mp = ml.prompts[L]
            stage(name, ml.head(H[:, layers.index(L)].float(), L), pid, mp, ml.alpha, ml.sha, rows=rows)
    R["unmerge"] = be.merge(None)
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    save()
    return R


def parse_args(argv=None):
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--positions", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--items", help="items.json + npz (ids, off)")
    src.add_argument("--token-ids", help='{"<key>": [ids]} JSON or {"key", "ids"} JSONL')
    ap.add_argument("--layers", type=ints, default=[22, 26, 30, 34], help="layers to capture (default 22,26,30,34)")
    ap.add_argument("--release-lens", default=os.environ.get("DSV41_RELEASE_LENS", ""))
    ap.add_argument("--release-layer", type=int, default=26)
    ap.add_argument("--multilayer-lens", default=os.environ.get("DSV41_MULTILAYER_LENS", ""))
    ap.add_argument("--ml-layers", type=ints, default=None,
                    help="layers read with the multilayer lens (default: --layers without the release layer when "
                         "--release-lens is given)")
    ap.add_argument("--samples", type=int, default=4, help="T = 1 samples per position besides greedy (max 63)")
    ap.add_argument("--sanity", type=int, default=512, help="positions read with both lenses at the release layer")
    ap.add_argument("--bullet-cap", type=int, default=16, help="bullet grammar cap (GrammarSpec(cap=...))")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="validate and print the plan only (CPU, no lens engine)")
    ap.add_argument("--model-dir", default=os.environ.get("DSV41_MODEL_DIR", ""))
    ap.add_argument("--lens-engine-dir", default=os.environ.get("DSV41_LENS_ENGINE_DIR", ""),
                    help="folder with olens_deepseek_v41_flash_usage.py and dsv41_engine.py")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--attn-backend", default="FLASHMLA_SPARSE_DSV41")
    ap.add_argument("--custom-all-reduce", action="store_true", help="keep vLLM's custom all-reduce (breaks capture)")
    ap.add_argument("--max-gpu-used-mib", type=int, default=2000)
    ap.add_argument("--chunk-tokens", type=int, default=400_000, help="prompt tokens per capture batch")
    ap.add_argument("--chunk-vecs", type=int, default=4096, help="positions per readout batch")
    a = ap.parse_args(argv)
    if not 0 <= a.samples < SEED_STRIDE:
        ap.error(f"--samples must be in 0..{SEED_STRIDE - 1}")
    if not a.check:
        missing = [n for n, v in (("--model-dir", a.model_dir), ("--lens-engine-dir", a.lens_engine_dir)) if not v]
        if missing:
            ap.error(f"set {' and '.join(missing)} (or DSV41_MODEL_DIR / DSV41_LENS_ENGINE_DIR)")
    return a


if __name__ == "__main__":
    run(parse_args())
