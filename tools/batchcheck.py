"""Batch-size vs mixing check for an edited arm (added after the 28 Oct 2026 retry boot).

Question: an edited request moved several times its repeat noise when put in a mixed batch (16 sham + 16 arm) vs
alone (16 arm). Is that batch-size rounding amplified by the edit, or does mixing with sham neighbours change (weaken)
the edit?

Runs on 16 prompts (``iso``) plus 16 neighbour prompts (``nb``), every request with a fresh salt:
  S  iso arm k alone (16 at once), twice          P  iso arm k + nb arm k (32 at once), twice
  X  iso arm k + nb sham (32 at once), twice      S0 iso sham alone (16), N0 iso sham + nb sham (32)
Distances on the iso requests: per request the median over positions of |delta logprob| (``req_medians``).
Retention of the edit in X relative to P, per request: sum((x - s0)(p - s0)) / sum((p - s0)^2).

Reading rule (fixed before the first run):
- harmless rounding: median d(X, P) <= q95(rep_P + rep_X) + F, and median retention in [0.95, 1.05];
- mixing bug: median d(X, P) above that bound, or median retention < 0.95;
- either way, d(P, S) vs d(X, S) says how much of the original difference batch size alone explains.
Needs a gates folder whose ``repeat`` gate ran (``arms_ext.json``), e.g. with ``DSV41_WEAK_ADD_SD=2,4``.
Usage: python tools/batchcheck.py --followup <gates folder> [--kind generic|agent]
``--kind agent`` uses the first 16 agent floor prompts (long agent transcripts) as ``iso`` and 16 generic prompts as
``nb``; output ``batchcheck_agent.json``. Results of the 28 Oct runs: docs/RESULTS_2026-10-28.md.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.client import EngineClient, floor_reference  # noqa: E402
from tools.gates import arms_info, floor_prompts, latest_floor, req_medians, request_fields, short_prompts  # noqa: E402

ARMS = (11, 4)  # 4 sd add (the follow-up's sensitive weak arm), 8 sd add (strong)


def together(client, groups):
    """All (prompt, fields) pairs at once, like the isolation gate's mixed batch; results in input order."""
    pairs = [(x, f) for ids, f in groups for x in ids]
    with ThreadPoolExecutor(len(pairs)) as ex:
        futs = [ex.submit(client.score, [x], f, len(pairs)) for x, f in pairs]
        return [fu.result()[0] for fu in futs]


def retention(xs, ps, s0s):
    out = []
    for x, p, s in zip(xs, ps, s0s):
        x, p, s = (np.asarray(v["lp"], dtype=np.float64) for v in (x, p, s))
        den = float(np.sum((p - s) ** 2))
        out.append(float(np.sum((x - s) * (p - s)) / den) if den > 0 else float("nan"))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--followup", required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--kind", default="generic", choices=("generic", "agent"))
    a = ap.parse_args()
    ext = json.loads(Path(a.followup, "arms_ext.json").read_text())
    info = arms_info(a.followup)
    F = float(floor_reference(json.loads(Path(latest_floor()).read_text()))["generic"]["median"])
    client = EngineClient(a.url)
    generic = [p["ids"] for p in floor_prompts() if p["kind"] == "generic"]
    if a.kind == "agent":
        iso, nb = [p["ids"] for p in floor_prompts() if p["kind"] == "agent"][:16], generic[:16]
    else:
        iso, nb = generic[:16], short_prompts(16, seed=1)
    fe = lambda k: request_fields(k, arms_sha=ext["sha"])  # noqa: E731
    n = len(iso)
    res = {"F": F, "kind": a.kind, "iso_len_median": int(np.median([len(x) for x in iso])), "ext_sha": ext["sha"],
           "add_sd": ext["add_sd"], "arms": {}}
    try:
        client.load_arms(ext["path"])
        S0 = client.score(iso, fe(0), concurrency=16)
        N0 = together(client, [(iso, fe(0)), (nb, fe(0))])[:n]
        res["native_size_effect_median"] = float(np.median(req_medians(N0, S0)))
        for k in ARMS:
            S = [client.score(iso, fe(k), concurrency=16) for _ in range(2)]
            P = [together(client, [(iso, fe(k)), (nb, fe(k))])[:n] for _ in range(2)]
            X = [together(client, [(iso, fe(k)), (nb, fe(0))])[:n] for _ in range(2)]
            rep_p, rep_x, rep_s = req_medians(*P), req_medians(*X), req_medians(*S)
            d_xp = np.concatenate([req_medians(X[0], P[0]), req_medians(X[1], P[1])])
            d_ps = np.concatenate([req_medians(P[0], S[0]), req_medians(P[1], S[1])])
            d_xs = np.concatenate([req_medians(X[0], S[0]), req_medians(X[1], S[1])])
            ret = np.concatenate([retention(X[0], P[0], S0), retention(X[1], P[1], S0)])
            bound = float(np.quantile(np.concatenate([rep_p, rep_x]), 0.95)) + F
            med_ret = float(np.nanmedian(ret))
            rounding = bool(np.median(d_xp) <= bound and 0.95 <= med_ret <= 1.05)
            res["arms"][str(k)] = {
                "add_sd": ext["add_sd"][str(k)] if str(k) in ext["add_sd"] else 8.0,
                "effect_vs_sham_median": float(np.median(req_medians(S[0], S0))),
                "rep_S": float(np.median(rep_s)), "rep_P": float(np.median(rep_p)), "rep_X": float(np.median(rep_x)),
                "d_PS_batch_size": float(np.median(d_ps)), "d_XS_size_plus_mixing": float(np.median(d_xs)),
                "d_XP_mixing_only": float(np.median(d_xp)), "bound": bound,
                "retention_median": med_ret, "retention_q05": float(np.nanquantile(ret, 0.05)),
                "retention_q95": float(np.nanquantile(ret, 0.95)),
                "reading": "harmless rounding" if rounding else "MIXING BUG candidate"}
    finally:
        client.load_arms(info["gate_path"])
    out = Path(a.followup, "batchcheck.json" if a.kind == "generic" else f"batchcheck_{a.kind}.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
