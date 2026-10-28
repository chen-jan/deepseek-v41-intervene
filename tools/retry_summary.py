"""Side-by-side summary of a retry gates folder: the old gates' verdicts and headline numbers next to
the retry gates (repeat, outliers, decode_effect) and their pre-written readings (docs/GATES.md, "Retry gates").
Reads JSON only; talks to no engine.

    python tools/retry_summary.py $DSV41_LOG_DIR/engine/gates_retry_<MMDD_HHMM>
"""

import json
import sys
from pathlib import Path


def _j(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, n=4):
    return "-" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))


def main(d: Path) -> None:
    s = _j(d, "summary")
    print("summary:", s and s.get("results"))
    for g in ("install", "build", "sham", "positive", "decode", "isolation", "prefix", "reload", "projection"):
        r = _j(d, g)
        if r is None:
            continue
        print(f"\n== {g} (old rule): {'PASS' if r.get('pass') else 'FAIL'} {r.get('error') or ''} nan={r.get('nan')}")
        if g == "decode":
            for x in r.get("rows", []):
                print(f"   {x['path']:13s} c={x['c']:<3} med4 {_f(x['median_arm4'])} med0 {_f(x['median_arm0'])} "
                      f"med_tf4 {_f(x['median_tf4_vs_tf4'])} effect {_f(x['effect_tf4_vs_sham'])} ok {x['ok']}")
        if g == "isolation" and "per_request" in r:
            bad = [f"{x['id']}/arm{x['arm']} med {_f(x['median'])} top1 {_f(x['top1_agree'])}"
                   for x in r["per_request"] if not x["ok"]]
            print(f"   counts_ok {r['counts_ok']}; outside own floor {len(bad)}: {bad[:6]}")
        if g == "reload" and "checks" in r:
            print(f"   checks {r['checks']}; r2 vs arm 4 median {_f(r['r2_vs_arm4']['all']['median'])}")
        if g == "projection" and "rows" in r:
            for x in r["rows"]:
                if x["layer"] == 26:
                    print(f"   arm {x['arm']} @26: max {_f(x['max_err_sd'])} sd, frac<=0.02 {_f(x['frac_le_0.02sd'], 6)}, "
                          f"exact {x['exact_ok']}, sd rule {x['sd_ok']}")
    r = _j(d, "repeat")
    if r:
        print(f"\n== repeat (new): {'PASS' if r.get('pass') else 'FAIL'} {r.get('error') or ''} "
              f"isolation {r.get('isolation_pass')} reload {r.get('reload_pass')} F {_f(r.get('F'))} "
              f"weak arm {r.get('gated_weak_arm')} {r.get('seconds', 0):.0f} s")
        for k, a in (r.get("part_a") or {}).items():
            q = a["repeat"]
            print(f"   arm {k:>2} ({a['add_sd']} sd): rep median {_f(q['rep_median'])} q95 {_f(q['rep_q95'])} | cross "
                  f"median {_f(q['cross_median'])} <= {_f(q['limit'])}: {q['ok']} | original rule {a['original_rule_ok']}"
                  f" | effect vs sham {_f(a['effect_vs_sham'])} sensitive {a['sensitive']} | shams M1 ok "
                  f"{a['sham_m1_ok']} M2 ok {a['sham_m2_ok']} | counts {a['counts_ok']}")
        b = r.get("part_b") or {}
        if b:
            q = b["strong"]
            print(f"   reload: rep median {_f(q['rep_median'])} q95 {_f(q['rep_q95'])} | cross median "
                  f"{_f(q['cross_median'])} <= {_f(q['limit'])}: {q['ok']} | old r2_is_arm4 {b['old_r2_is_arm4']['ok']}")
            for k, w in b["weak"].items():
                print(f"   weak reload arm {k} -> slot {w['reloaded_slot']}: vs own slot median "
                      f"{_f(w['vs_own_slot']['median'])}, original rule {w['original_rule_ok']}, differs from sham "
                      f"{w['differs_from_sham']}")
        for line in r.get("reading", []):
            print("   reading:", line)
    r = _j(d, "outliers")
    if r:
        print(f"\n== outliers (new): {'PASS' if r.get('pass') else 'FAIL'} {r.get('error') or ''} "
              f"overflow {r.get('overflow')} {r.get('seconds', 0):.0f} s")
        for x in r.get("rows", []):
            if not x.get("gated"):
                continue
            print(f"   arm {x['arm']} @26: max {_f(x['max_err_sd'])} sd, >0.05 sd {x['n_gt_0.05sd']} tokens, "
                  f"frac<=0.02 {_f(x['frac_le_0.02sd'], 6)}, unexplained {x.get('n_unexplained')}, all flagged in top "
                  f"0.1% |h| {x.get('all_flagged_in_top_0.1pct_norm')}, |h| median {_f(x.get('hnorm_median'))} q99.9 "
                  f"{_f(x.get('hnorm_q999'))}, exact {x['exact_ok']}, old rule {x['old_rule_ok']}, ok {x['ok']}")
            for t in x["top"][:5]:
                print(f"      {t['prompt']} pos {t['pos']}: err {_f(max(t['err_sd']))} sd (stream {t['worst_stream']}),"
                      f" |h| {_f(t.get('hnorm'))} (pct {_f(t.get('hnorm_pct'))}), bound "
                      f"{_f(max(t.get('bound_sd') or [0.0]))} sd, explained {t.get('explained')}")
        others = [x for x in r.get("rows", []) if not x.get("gated") and x["n_gt_0.05sd"]]
        for x in sorted(others, key=lambda x: -x["max_err_sd"])[:6]:
            print(f"   (recorded) arm {x['arm']} @{x['layer']}: max {_f(x['max_err_sd'])} sd, >0.05 sd "
                  f"{x['n_gt_0.05sd']}, flagged pos<4 {x['flagged_positions']['n_pos_lt_4']}, unexplained "
                  f"{x.get('n_unexplained', 'no |h| here')}")
    r = _j(d, "decode_effect")
    if r:
        print(f"\n== decode_effect (new): {'PASS' if r.get('pass') else 'FAIL'} {r.get('error') or ''} medium arm "
              f"{r.get('medium_arm')} ({r.get('medium_add_sd')} sd), calibration {r.get('calibration_effect')} "
              f"{r.get('seconds', 0):.0f} s")
        for x in r.get("rows", []):
            print(f"   {x['path']:13s} c={x['c']:<3} med {_f(x['median_arm'])} med0 {_f(x['median_arm0'])} med_tf "
                  f"{_f(x['median_tf_vs_tf'])} effect {_f(x['effect_tf_vs_sham'])} | testable {x['testable']} noise "
                  f"{x['noise_ok']} cap {x['cap_ok']} counts {x['counts_ok']} path {x['path_ok']} | med/floor "
                  f"{_f(x['median_arm_over_floor'], 3)}")
        for line in r.get("reading", []):
            print("   reading:", line)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
