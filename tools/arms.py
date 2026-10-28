"""Build arms files, mu, controls and request fields.

The provider and every tool get ``vllm_xargs`` / ``cache_salt`` only from ``request_fields``. File format and salt
rule: docs/INTERFACE.md sections 4 and 7. Imports only ``dsv41_intervene/__init__.py`` from the engine package.

    python tools/arms.py fields --arms-sha SHA --arm 3 [--edit-from 1 --edit-to N]   # JSON for --extra-body
    python tools/arms.py build --recipe recipe.json --out arms/name                  # -> name.safetensors/.json
    python tools/arms.py mu --dump cap.safetensors --arm 1 [--tag T] --out mu.json
    python tools/arms.py controls --dump vec.safetensors --layer 26 --d d.npy --n 8 --seed 0 --out c.npy
"""

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsv41_intervene import DEFAULT_N_ARMS, HC, HIDDEN, MAX_CAP_POS, MAX_EDIT_TO, env_path  # noqa: E402

KINDS = ("measure", "project", "project_mc", "shift", "add")
LOADED_SHAS = "LOADED_SHAS"  # "<boot> <sha>" per line, appended by tools/client.py load_arms
GLOBAL_ARM = "GLOBAL_ARM"  # "<boot> <k>", written by tools/client.py set_global_arm


def salt(arms_sha: str, arm: int, edit_from: int, edit_to: int, nonce: str | None = None) -> str:
    """24-hex cache salt; ``nonce`` (capture requests) is hashed in as ``|cap|<nonce>``."""
    key = f"{arms_sha}|{arm}|{edit_from}|{edit_to}" + (f"|cap|{nonce}" if nonce is not None else "")
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def request_fields(arm: int = 0, edit_from: int = 1, edit_to: int = MAX_EDIT_TO, *, arms_sha: str,
                   capture: str | None = None, cap_from: int | None = None, cap_to: int | None = None,
                   cap_pos: list[int] | None = None, tag: str | None = None, nonce: str | None = None) -> dict:
    """Return ``{"vllm_xargs": {...}, "cache_salt": ...}`` for one request (no salt for plain sham; defaults omitted
    from ``vllm_xargs``; a capture request gets a fresh random nonce unless one is given)."""
    arm, edit_from, edit_to = int(arm), int(edit_from), int(edit_to)
    if arm < 0:
        raise ValueError(f"arm must be >= 0, got {arm}")
    if not 0 <= edit_from < edit_to <= MAX_EDIT_TO:
        raise ValueError(f"need 0 <= edit_from < edit_to <= {MAX_EDIT_TO}, got [{edit_from}, {edit_to})")
    if (arm != 0 or capture) and not arms_sha:
        raise ValueError("arms_sha is required for arm != 0 and for captures (it keys the salt)")
    xargs: dict = {}
    if arm != 0:
        xargs["arm"] = arm
    if edit_from != 1:
        xargs["edit_from"] = edit_from
    if edit_to != MAX_EDIT_TO:
        xargs["edit_to"] = edit_to
    if capture is not None:
        if capture not in ("coef", "vec"):
            raise ValueError(f"capture must be 'coef' or 'vec', got {capture!r}")
        xargs["capture"] = capture
        if cap_pos is not None:
            pos = sorted({int(p) for p in cap_pos})
            if not pos or len(pos) > MAX_CAP_POS or pos[0] < 0:
                raise ValueError(f"cap_pos needs 1..{MAX_CAP_POS} positions >= 0")
            xargs["cap_pos"] = pos
        else:
            if cap_from is not None:
                xargs["cap_from"] = int(cap_from)
            if cap_to is not None:
                xargs["cap_to"] = int(cap_to)
        nonce = nonce if nonce is not None else secrets.token_hex(8)
    if tag:
        xargs["tag"] = str(tag)
    out: dict = {"vllm_xargs": xargs}
    if arm != 0 or capture is not None or nonce is not None:
        out["cache_salt"] = salt(arms_sha, arm, edit_from, edit_to, nonce)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Engine state files for the proxy (which arms shas this boot loaded, the global arm)
# ----------------------------------------------------------------------------------------------------------------------

def arms_dir() -> Path:
    return Path(env_path("DSV41_ARMS_DIR", "arms"))


def record_loaded_sha(boot: str, sha: str) -> None:
    """Append ``<boot> <sha>`` to ``$DSV41_ARMS_DIR/LOADED_SHAS`` (the proxy accepts salts of these shas)."""
    d = arms_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LOADED_SHAS, "a") as f:
        f.write(f"{boot} {sha}\n")


def record_global_arm(boot: str, k: int) -> None:
    """Write ``<boot> <k>`` to ``$DSV41_ARMS_DIR/GLOBAL_ARM`` (``-1`` = per-request arms)."""
    d = arms_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{GLOBAL_ARM}.{os.getpid()}"
    tmp.write_text(f"{boot} {int(k)}\n")
    os.replace(tmp, d / GLOBAL_ARM)


def read_state(path: Path, boot: str) -> list[str]:
    """Values of the ``<boot> <value>`` lines of ``path`` that belong to ``boot`` (missing file: none)."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    return [v for b, _, v in (ln.strip().partition(" ") for ln in lines) if b == boot and v]


# ----------------------------------------------------------------------------------------------------------------------
# Arms tables and files
# ----------------------------------------------------------------------------------------------------------------------

def empty_tables(n_inst: int, n_arms: int = DEFAULT_N_ARMS) -> dict:
    """All-zero arms tables (every arm sham), numpy, the dtypes of INTERFACE.md section 7."""
    import numpy as np

    return {"p": np.zeros((n_arms, n_inst, HC, HIDDEN), np.float32),
            "e": np.zeros((n_arms, n_inst, HC, HIDDEN), np.float32),
            "mu": np.zeros((n_arms, n_inst, HC), np.float32), "a": np.zeros((n_arms, n_inst), np.float32),
            "g": np.zeros((n_arms, n_inst), np.float32), "mc": np.zeros((n_arms,), np.int32)}


def per_stream(v) -> "np.ndarray":
    """``[5120]`` (mean-space, copied to all 4 streams) or ``[4, 5120]`` -> unit-norm-per-stream ``[4, 5120]`` fp64."""
    import numpy as np

    v = np.asarray(v, np.float64)
    v = np.broadcast_to(v, (HC, HIDDEN)) if v.shape == (HIDDEN,) else v
    if v.shape != (HC, HIDDEN):
        raise ValueError(f"direction must be [{HIDDEN}] or [{HC}, {HIDDEN}], got {v.shape}")
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("zero direction")
    return v / n


def set_slot(t: dict, arm: int, slot: int, kind: str, d=None, c=None, mu=None, lam: float = 1.0,
             alpha: float = 0.0) -> None:
    """Write one (arm, slot) entry (docs/INTERFACE.md section 3). ``d`` read direction, ``c`` write direction for ``shift``,
    ``mu`` per-stream ``[4]`` reference (scalar ok), ``lam`` removal strength, ``alpha`` add amount."""
    import numpy as np

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    p = per_stream(d)
    if kind == "project_mc" and not np.allclose(p, p[:1]):
        raise ValueError("mean-coupled projection needs one direction replicated on all 4 streams")
    e = per_stream(c) if kind == "shift" else (np.zeros_like(p) if kind == "measure" else p)
    t["p"][arm, slot] = p
    t["e"][arm, slot] = e
    t["mu"][arm, slot] = np.broadcast_to(np.asarray(0.0 if mu is None else mu, np.float64), (HC,))
    t["a"][arm, slot] = lam if kind in ("project", "project_mc", "shift") else 0.0
    t["g"][arm, slot] = alpha if kind == "add" else 0.0
    if kind == "project_mc":
        t["mc"][arm] = 1


def write_arms_file(path: str, layers: list[int], p, e, mu, a, g, mc, labels: dict[int, str],
                    recipe: dict) -> str:
    """Write ``<path>.safetensors`` + ``<path>.json``; checks arm 0 is all zero. Returns the sha256 (= arms_sha)."""
    import numpy as np
    from safetensors.numpy import save_file

    base = Path(str(path).removesuffix(".safetensors"))
    t = {"layers": np.asarray(layers, np.int32), "p": np.ascontiguousarray(p, np.float32),
         "e": np.ascontiguousarray(e, np.float32), "mu": np.ascontiguousarray(mu, np.float32),
         "a": np.ascontiguousarray(a, np.float32), "g": np.ascontiguousarray(g, np.float32),
         "mc": np.ascontiguousarray(mc, np.int32)}
    A, I = t["a"].shape
    want = {"layers": (I,), "p": (A, I, HC, HIDDEN), "e": (A, I, HC, HIDDEN), "mu": (A, I, HC), "a": (A, I),
            "g": (A, I), "mc": (A,)}
    for k, shape in want.items():
        if t[k].shape != shape:
            raise ValueError(f"{k}: shape {t[k].shape}, want {shape}")
        if not np.all(np.isfinite(t[k])):
            raise ValueError(f"{k} has non-finite values")
    if list(t["layers"]) != sorted(set(layers)):
        raise ValueError("layers must be ascending and unique")
    if any(np.any(t[k][0] != 0) for k in ("p", "e", "mu", "a", "g", "mc")):
        raise ValueError("arm 0 (sham) must be all zero")
    if np.any((t["mc"] != 0) & (t["mc"] != 1)):
        raise ValueError("mc must be 0 or 1")
    base.parent.mkdir(parents=True, exist_ok=True)
    st_path = base.with_name(base.name + ".safetensors")
    save_file(t, str(st_path))
    sha = hashlib.sha256(st_path.read_bytes()).hexdigest()
    base.with_name(base.name + ".json").write_text(json.dumps(
        {"sha256": sha, "file": st_path.name, "layers": [int(x) for x in t["layers"]], "n_arms": A,
         "labels": {str(k): v for k, v in sorted(labels.items())}, "recipe": recipe}, indent=1))
    return sha


def load_direction(spec, layers: list[int], slot: int):
    """A direction from a recipe: a list of 5120 floats, ``"file.npy"`` (``[5120]``, ``[4,5120]``, ``[I,5120]`` or
    ``[I,4,5120]``) or ``"file.npy:<layer>"`` with a ``{layer: vector}`` ``.npz``."""
    import numpy as np

    if isinstance(spec, list):
        return np.asarray(spec, np.float64)
    name, _, key = str(spec).partition(":")
    arr = np.load(name)
    if hasattr(arr, "files"):  # npz keyed by layer
        return arr[key or str(layers[slot])]
    if arr.ndim >= 2 and arr.shape[0] == len(layers) and arr.shape[-1] == HIDDEN and arr.shape != (HC, HIDDEN):
        return arr[slot]
    return arr


def build_from_recipe(recipe: dict, out: str) -> str:
    """Recipe JSON -> arms file. ``{"layers": [...installed], "n_arms": 16, "arms": [{"arm": 1, "label": "...",
    "slots": [{"layers": [26] | "all", "kind": "project", "d": ..., "c": ..., "mu": "mu.json" | [4] | 0,
    "lambda": 1.0, "alpha": 0.0 | "alpha_sd": 8}]}]}``. ``mu.json`` is the output of ``mu`` (keyed by layer)."""
    layers = [int(x) for x in recipe["layers"]]
    t = empty_tables(len(layers), int(recipe.get("n_arms", DEFAULT_N_ARMS)))
    labels = {}
    for arm in recipe["arms"]:
        k = int(arm["arm"])
        if k == 0:
            raise ValueError("arm 0 is the sham and stays all zero")
        labels[k] = arm.get("label", "")
        for s in arm["slots"]:
            sl = layers if s.get("layers", "all") == "all" else [int(x) for x in s["layers"]]
            for L in sl:
                slot = layers.index(L)
                mu, sd = s.get("mu", 0.0), None
                if isinstance(mu, str):
                    m = json.loads(Path(mu).read_text())
                    mu, sd = m["mu"][str(L)], m["sd"][str(L)]
                alpha = float(s.get("alpha", 0.0))
                if "alpha_sd" in s:
                    if sd is None:
                        raise ValueError("alpha_sd needs a mu.json with sd")
                    alpha = float(s["alpha_sd"]) * float(sum(sd) / len(sd))
                set_slot(t, k, slot, s["kind"], d=load_direction(s["d"], layers, slot),
                         c=load_direction(s["c"], layers, slot) if "c" in s else None, mu=mu,
                         lam=float(s.get("lambda", 1.0)), alpha=alpha)
    return write_arms_file(out, layers, labels=labels, recipe=recipe, **t)


# ----------------------------------------------------------------------------------------------------------------------
# mu, gap report, controls
# ----------------------------------------------------------------------------------------------------------------------

def load_dump(dump_path: str) -> tuple[dict, dict]:
    """A capture dump (``.safetensors`` + ``.json`` sidecar) as numpy arrays and its metadata. ``vec`` is bf16 in
    the file (numpy has no bf16), so it is read through torch and returned as float32.

    Raises ValueError when the kernel's device positions (``coef_pos_dev``) differ from the host row map
    (``coef_pos``) on any row: then captured rows sit under the wrong request or position."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    arr = {k: (v.float() if v.dtype == torch.bfloat16 else v).numpy() for k, v in load_file(dump_path).items()}
    if "coef_pos_dev" in arr and not np.array_equal(arr["coef_pos_dev"], arr["coef_pos"]):
        bad = int((arr["coef_pos_dev"] != arr["coef_pos"]).sum())
        raise ValueError(f"{dump_path}: {bad} of {arr['coef_pos'].size} coef rows have a device position that differs "
                         "from the host row map (capture rows misaligned with tokens)")
    return arr, json.loads(Path(dump_path + ".json").read_text())


def nonfinite_requests(dump: dict):
    """Request indices (``coef_req``) with any non-finite c or c' in the dump. The base model returns NaN on some
    batched prefill requests (native too: 17/40 batched, 0/40 one by one, 28 Oct); the client retries them as new
    requests, so a failed attempt's rows (NaN rows plus duplicates of the retry's finite rows) stay in the ring."""
    import numpy as np

    bad = ~np.isfinite(dump["coef"][:, :, :2]).all(axis=(1, 2, 3))
    return np.unique(dump["coef_req"][bad])


def coef_rows(dump: dict, meta: dict, arm: int, tag: str | None = None, min_pos: int = 4,
              finite_requests_only: bool = False):
    """Coefficient rows ``[n, I, 3, 4]`` of one arm (and tag), positions >= ``min_pos``. ``finite_requests_only``
    drops every row of a request that has any non-finite row (estimates only; never for the Projection exact gate,
    which must see every row)."""
    import numpy as np

    sel = (dump["coef_arm"] == arm) & (dump["coef_pos"] >= min_pos)
    if tag is not None:
        sel &= dump["coef_tag"] == (meta["tags"].index(tag) if tag in meta["tags"] else -2)
    if finite_requests_only:
        sel &= ~np.isin(dump["coef_req"], nonfinite_requests(dump))
    return dump["coef"][sel].astype(np.float64)


def mu_from_capture(dump_path: str, arm: int, tag: str | None = None, min_pos: int = 4,
                    median_from_layer: int = 30) -> dict:
    """Per-slot, per-stream mu from a coef capture (mean; median for outputs >= ``median_from_layer``), plus sd."""
    import numpy as np

    dump, meta = load_dump(dump_path)
    n_all = len(coef_rows(dump, meta, arm, tag, min_pos))
    # requests with a non-finite row are dropped whole (their retry is in the dump as a new request)
    c = coef_rows(dump, meta, arm, tag, min_pos, finite_requests_only=True)[:, :, 0, :]  # [n, I, 4] before the edit
    if len(c) < 2:
        raise ValueError(f"only {len(c)} capture rows for arm {arm} tag {tag!r}")
    layers = meta["layers"]
    mu, sd = {}, {}
    for i, L in enumerate(layers):
        x = c[:, i, :]
        mu[str(L)] = (np.median(x, 0) if L >= median_from_layer else x.mean(0)).tolist()
        sd[str(L)] = x.std(0, ddof=1).tolist()
    return {"dump": dump_path, "arm": arm, "tag": tag, "min_pos": min_pos, "median_from_layer": median_from_layer,
            "n": int(len(c)), "n_dropped_nonfinite_requests": int(n_all - len(c)), "layers": layers, "mu": mu,
            "sd": sd}


def gap_report(mu_generic, mu_agent, sd) -> dict:
    """``(mu_generic - mu_agent) / sd`` per slot and stream, flagged above 0.25."""
    import numpy as np

    out, flags = {}, []
    for L in mu_generic:
        z = (np.asarray(mu_generic[L]) - np.asarray(mu_agent[L])) / np.maximum(np.asarray(sd[L]), 1e-12)
        out[L] = z.tolist()
        if np.any(np.abs(z) > 0.25):
            flags.append(int(L))
    return {"gap_sd": out, "flagged_layers": flags, "threshold": 0.25}


def control_sampler(states, d, n: int, seed: int, top_pcs: int = 200, var_tol: float = 0.25):
    """Random unit vectors in the top-``top_pcs`` PC span of captured states, orthogonal to ``d``, with
    ``Var(h.c)`` within ``var_tol`` of ``Var(h.d)``.

    Candidates draw PC weights with scale ``lambda_i ** (beta / 2)``; ``beta`` is bisected so the median variance ratio
    is 1, then candidates outside the tolerance are rejected. Returns ``(vectors [n, 5120] fp32, info)``."""
    import numpy as np

    x = np.asarray(states, np.float64)
    x = x - x.mean(0)
    cov = x.T @ x / max(1, len(x) - 1)
    d = np.asarray(d, np.float64) / np.linalg.norm(d)
    target = float(d @ cov @ d)
    evals, evecs = np.linalg.eigh(cov)
    lam, U = evals[::-1][:top_pcs].clip(min=0), evecs[:, ::-1][:, :top_pcs]
    rng = np.random.default_rng(seed)

    def draw(beta: float, m: int):
        w = rng.standard_normal((m, lam.size)) * (lam + 1e-30) ** (beta / 2)
        v = w @ U.T
        v -= np.outer(v @ d, d)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v, np.sum((v @ cov) * v, axis=1) / target  # = einsum("ij,jk,ik->i"), via BLAS

    lo, hi = -4.0, 4.0
    for _ in range(30):
        beta = (lo + hi) / 2
        med = np.median(draw(beta, 256)[1])
        lo, hi = (beta, hi) if med < 1 else (lo, beta)
    beta = (lo + hi) / 2
    keep, ratios, tries = [], [], 0
    while len(keep) < n and tries < 200:
        v, r = draw(beta, 256)
        ok = np.abs(r - 1) <= var_tol
        keep += list(v[ok])
        ratios += list(r[ok])
        tries += 1
    if len(keep) < n:
        raise ValueError(f"control sampler: {len(keep)}/{n} candidates within {var_tol} (Var(h.d)={target:.4g}, "
                         f"top-{top_pcs} eigenvalues {lam[-1]:.4g}..{lam[0]:.4g})")
    vecs = np.asarray(keep[:n], np.float32)
    return vecs, {"var_d": target, "var_ratio": [float(r) for r in ratios[:n]], "beta": beta, "seed": seed,
                  "top_pcs": top_pcs, "var_tol": var_tol, "n_states": int(len(x)),
                  "max_abs_cos_d": float(np.abs(vecs.astype(np.float64) @ d).max())}


def vec_means(dump: dict, meta: dict, layer: int, arm: int | None = None, tag: str | None = None, min_pos: int = 4):
    """Stream-mean vectors ``[m, 5120]`` fp64 at one capture layer (rows of one arm/tag, positions >= ``min_pos``)."""
    import numpy as np

    sel = dump["vec_pos"] >= min_pos
    if arm is not None:
        sel &= dump["vec_arm"] == arm
    if tag is not None:
        sel &= dump["vec_tag"] == (meta["tags"].index(tag) if tag in meta["tags"] else -2)
    return dump["vec"][sel][:, meta["capture_layers"].index(layer)].astype(np.float64)


def layer_capture(dump: dict, meta: dict, layers: list[int], tag: str) -> dict:
    """One prompt's rows of a capture dump, split by layer (docs/MULTILAYER.md, "Multi-layer capture").

    A capture request writes, per captured token, one coefficient row covering every installed slot
    (``meta["layers"]``) and, for ``capture="vec"``, one vector row covering every boot-time capture layer
    (``meta["capture_layers"]``). This picks the rows of ``tag``'s last request (a NaN retry is a new request with the
    same tag, sent after the failed one) and, inside it, the last row per position (a preempted and recomputed request
    gets fresh rows).

    Returns ``{"req", "coef_pos" [n], "coef": {L: [n, 3, 4]} (c before, c' after the bf16 cast, carried pre-mix; per
    stream), "vec_pos" [m], "vec": {L: [m, 5120]} (pre-edit stream mean; only layers in ``capture_layers`` and only
    if vectors were captured), "nonfinite_pos" (positions with a non-finite c, c' or vector at a requested layer;
    flagged, not dropped), "zero_coef_layers" (requested layers where c is 0 on every row: the arm has no read
    direction there, as for arm 0)}``. Raises ValueError for a layer that is not installed."""
    import numpy as np

    installed, cap_layers = list(meta["layers"]), list(meta["capture_layers"])
    bad = [L for L in layers if L not in installed]
    if bad:
        raise ValueError(f"layers {bad} are not installed edit slots {installed}")
    t = meta["tags"].index(tag) if tag in meta["tags"] else -2

    def last_rows(ring: str):
        sel = np.flatnonzero(dump[f"{ring}_tag"] == t)
        if not sel.size:
            return None, np.zeros(0, np.int64), np.zeros(0, np.int64)
        req = int(dump[f"{ring}_req"][sel].max())
        sel = sel[dump[f"{ring}_req"][sel] == req]
        last = {int(p): int(r) for p, r in zip(dump[f"{ring}_pos"][sel], sel)}  # later rows overwrite earlier ones
        pos = np.array(sorted(last), np.int64)
        return req, pos, np.array([last[p] for p in pos], np.int64)

    req, cpos, crows = last_rows("coef")
    _, vpos, vrows = last_rows("vec")
    coef = {L: dump["coef"][crows][:, installed.index(L)] for L in layers}
    vec = {L: dump["vec"][vrows][:, cap_layers.index(L)] for L in layers if L in cap_layers} if vrows.size else {}
    bad_pos = set()
    for L, c in coef.items():
        bad_pos.update(cpos[~np.isfinite(c[:, :2]).all(axis=(1, 2))].tolist())
    for L, v in vec.items():
        bad_pos.update(vpos[~np.isfinite(v).all(axis=1)].tolist())
    zero = [L for L in layers if crows.size and not np.any(coef[L][:, 0])]
    return {"req": req, "coef_pos": cpos, "coef": coef, "vec_pos": vpos, "vec": vec,
            "nonfinite_pos": sorted(int(p) for p in bad_pos), "zero_coef_layers": zero}


def main() -> None:
    """CLI: ``build`` (gate arms or a recipe JSON), ``mu``, ``controls``, ``fields``."""
    import numpy as np

    ap = argparse.ArgumentParser(description="arms files, mu, controls, request fields")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fields")
    f.add_argument("--arms-sha", default="")
    f.add_argument("--arm", type=int, default=0)
    f.add_argument("--edit-from", type=int, default=1)
    f.add_argument("--edit-to", type=int, default=MAX_EDIT_TO)
    b = sub.add_parser("build")
    b.add_argument("--recipe", required=True)
    b.add_argument("--out", required=True)
    m = sub.add_parser("mu")
    m.add_argument("--dump", required=True)
    m.add_argument("--arm", type=int, required=True)
    m.add_argument("--tag", default=None)
    m.add_argument("--min-pos", type=int, default=4)
    m.add_argument("--median-from-layer", type=int, default=30)
    m.add_argument("--out", required=True)
    c = sub.add_parser("controls")
    c.add_argument("--dump", required=True, help="vec capture dump")
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--d", required=True, help=".npy direction [5120]")
    c.add_argument("--n", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--top-pcs", type=int, default=200)
    c.add_argument("--var-tol", type=float, default=0.25)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "fields":
        out = request_fields(args.arm, args.edit_from, args.edit_to, arms_sha=args.arms_sha)
    elif args.cmd == "build":
        sha = build_from_recipe(json.loads(Path(args.recipe).read_text()), args.out)
        out = {"sha256": sha, "out": args.out}
    elif args.cmd == "mu":
        out = mu_from_capture(args.dump, args.arm, args.tag, args.min_pos, args.median_from_layer)
        Path(args.out).write_text(json.dumps(out, indent=1))
        out = {"out": args.out, "n": out["n"]}
    else:
        dump, meta = load_dump(args.dump)
        vecs, info = control_sampler(vec_means(dump, meta, args.layer), np.load(args.d), args.n, args.seed,
                                     args.top_pcs, args.var_tol)
        np.save(args.out, vecs)
        Path(args.out + ".json").write_text(json.dumps(info, indent=1))
        out = {"out": args.out, **{k: info[k] for k in ("var_d", "beta", "max_abs_cos_d")}}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
