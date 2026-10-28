"""Capture stream means and coefficients at several layers, one request per prompt (docs/MULTILAYER.md).

Runs on the GPU node against a running edit engine (scripts/serve_engine.sh, :8101) with an arms file loaded. Vectors
exist only at the boot-time capture layers (DSV41_CAPTURE_LAYERS, default 12,15,18,22,25,26,27,30,34). Coefficients
exist at every installed slot, along the read direction of the chosen arm there; arm 0 (sham) reads nothing, so for
your own direction load an arms file with a measure-only arm at the layers you want, for example:

    {"layers": [12, 14, 15, 16, ..., 38], "n_arms": 16,
     "arms": [{"arm": 1, "label": "measure d at 22/26/30/34",
               "slots": [{"layers": [22, 26, 30, 34], "kind": "measure", "d": "d.npz"}]}]}

(``layers`` = all installed slots, as `tools/client.py installed` prints them; ``d.npz`` holds one [5120] or [4, 5120]
direction per layer, keyed "22", "26", ...). Then:

    python tools/arms.py build --recipe measure.json --out $DSV41_ARMS_DIR/measure_l4
    python tools/client.py load $DSV41_ARMS_DIR/measure_l4.safetensors
    python examples/multilayer_capture.py --prompts prompts.json --layers 22,26,30,34 --arm 1 \\
        --dump $DSV41_CAPTURE_DIR/ml_run1.safetensors --out ml_run1.npz

prompts.json: {"<prompt id>": [token ids, BOS (0) first], ...}. Writes ``--out`` (.npz; per prompt k = its index:
``p<k>_coef_pos``, ``p<k>_c_L<L>`` [n, 3, 4] = c, c' and pre-mix per stream, ``p<k>_vec_pos``, ``p<k>_vec_L<L>``
[m, 5120]) and ``--out``.json (prompt ids, NaN retries, positions with non-finite values, layers with no read
direction). Loading arms replaces every arm in the engine, and the dump empties the capture rings: run this on an
engine nobody else is using.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.client import EngineClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=None, help="engine (default $DSV41_ENGINE_URL or http://127.0.0.1:8101)")
    ap.add_argument("--prompts", required=True, help='JSON {"<id>": [token ids], ...}')
    ap.add_argument("--layers", required=True, help="comma-separated output layers, e.g. 22,26,30,34")
    ap.add_argument("--kind", choices=["vec", "coef"], default="vec", help="vec = vectors and coefficients")
    ap.add_argument("--arm", type=int, default=0, help="arm whose read directions give the coefficients")
    ap.add_argument("--cap-from", type=int, default=4, help="first captured position (default 4, as the gates and "
                                                             "mu use; the first position has a huge norm)")
    ap.add_argument("--cap-to", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--dump", required=True, help="new .safetensors path under the engine's DSV41_CAPTURE_DIR")
    ap.add_argument("--out", required=True, help=".npz output")
    a = ap.parse_args()

    prompts = json.loads(Path(a.prompts).read_text())
    ids = list(prompts)
    layers = [int(x) for x in a.layers.split(",")]
    res = EngineClient(a.url).capture_layers([prompts[k] for k in ids], layers, a.dump, kind=a.kind, arm=a.arm,
                                             cap_from=a.cap_from, cap_to=a.cap_to, concurrency=a.concurrency)
    arrays, info = {}, []
    for k, (pid, r) in enumerate(zip(ids, res["prompts"])):
        arrays[f"p{k}_coef_pos"] = r["coef_pos"]
        arrays[f"p{k}_vec_pos"] = r["vec_pos"]
        for L in layers:
            arrays[f"p{k}_c_L{L}"] = r["coef"][L]
            if L in r["vec"]:
                arrays[f"p{k}_vec_L{L}"] = r["vec"][L]
        info.append({"k": k, "id": pid, "tag": r["tag"], "n_coef": int(r["coef_pos"].size),
                     "n_vec": int(r["vec_pos"].size), "nan_retries": r["nan_retries"],
                     "nonfinite_pos": r["nonfinite_pos"], "zero_coef_layers": r["zero_coef_layers"]})
    np.savez(a.out, **arrays)
    summary = {k: res[k] for k in ("dump", "layers", "kind", "arm", "arms_sha", "coef_overflow", "vec_overflow")}
    Path(a.out + ".json").write_text(json.dumps({**summary, "prompts": info}, indent=1))
    print(json.dumps({**summary, "out": a.out, "n_prompts": len(ids),
                      "prompts_with_nonfinite": sum(bool(x["nonfinite_pos"]) for x in info),
                      "zero_coef_layers": sorted({L for x in info for L in x["zero_coef_layers"]})}, indent=1))


if __name__ == "__main__":
    main()
