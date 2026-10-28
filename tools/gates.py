"""In-engine gate suite (docs/GATES.md); each gate writes JSON to ``$DSV41_LOG_DIR/engine/gates_<ts>/`` and a
failure stops the suite. Order: install check, build arms, sham vs native, positive control per path, decode vs
prefill, arm isolation, prefix cache, arm reload, projection exact, info (recorded numbers, no pass/fail), proxy.

Runs on the GPU node (captures are node-local), against the engine on ``:8101``. The default run is every gate but
``proxy``; start the proxy only after it (stray traffic would break the exact counters), then ``--only proxy``::

    python tools/gates.py --engine-log LOG [--only build,sham,...] [--floor FLOOR.json] [--out DIR]
    python tools/gates.py --only proxy --out DIR

``build`` writes ``arms.json`` into the out folder; later gates read it from there, so ``--only`` works on a folder
where ``build`` already ran (``--out`` the same folder). Without ``build`` in ``--only``, the gate arms are reloaded
when the engine holds other arms (after a reboot every table is zero), and cached generations of another boot are
ignored. Every counted block first waits for an idle engine (``/metrics``) and aborts on stray traffic.

Retry gates (added 28 Oct ~11:30 CET, before any run of them; reading table in ``docs/GATES.md``), after ``info``:
``repeat`` (measured repeat floor for isolation and reload), ``outliers`` (projection errors vs |h| and position),
``decode_effect`` (decode vs prefill for an arm whose effect on its own text is large). They never replace an old
gate: the old gates are unchanged, and each new result carries the old one next to it.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from dsv41_intervene import DEFAULT_EDIT_LAYERS, MAX_EDIT_TO, env_path, parse_layers  # noqa: E402
from tools.arms import (coef_rows, empty_tables, load_dump, mu_from_capture, nonfinite_requests,  # noqa: E402
                        request_fields, set_slot, vec_means, write_arms_file)
from tools.client import (EngineClient, compare, floor_reference, load_runs, sub_kind,  # noqa: E402
                          within_floor)


def _opt_dir(var: str, sub: str) -> Path | None:
    p = env_path(var, sub, required=False)
    return Path(p) if p else None


# All from the environment (see .env.example); main() stops with a clear message when one is missing.
ARMS_DIR = _opt_dir("DSV41_ARMS_DIR", "arms")
CAPTURE_DIR = _opt_dir("DSV41_CAPTURE_DIR", "captures")  # must be the engine's DSV41_CAPTURE_DIR (dumps are checked)
LOG_DIR = _opt_dir("DSV41_LOG_DIR", "logs")
XCHECK_DIR = _opt_dir("DSV41_XCHECK_DIR", "xcheck")
DATA = Path(os.environ.get("DSV41_DATA_DIR") or ENGINE / "data")  # tools/build_data.py writes it
GATES = ("install", "build", "sham", "positive", "decode", "isolation", "prefix", "reload", "projection", "info",
         "repeat", "outliers", "decode_effect", "proxy")
L26 = 26
N_GEN = 64
SHORT = 256  # prompt length for the decode-path cases
# Per-prompt floor minimums for the arm isolation gate (fixed 28 Oct before any engine run): some native prompt pairs
# are bit-identical, which would make any nondeterminism fail. An 8 sd arm-4 leak moves the median by >= 0.13
# (positive control: 10x the pooled floor) and the exact per-arm counters catch it too, so these cannot hide one.
MIN_PROMPT_MEDIAN, MIN_PROMPT_P95, MAX_PROMPT_TOP1 = 0.005, 0.05, 0.99
_CLIENT: "EngineClient | None" = None  # set by main; write() adds its NaN-retry counts to every gate JSON


def write(out_dir: str, name: str, res: dict) -> dict:
    res = {"gate": name, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **res}
    if _CLIENT is not None:  # NaN retries hide non-finite logits; one at concurrency 1 fails the gate
        res["nan"] = _CLIENT.reset_nan()
        if res["nan"]["retried_c1"]:
            res["pass"] = False
            res["nan_fail"] = "a teacher-forced request at concurrency 1 needed a NaN retry"
    Path(out_dir, f"{name}.json").write_text(json.dumps(res, indent=1, default=_np))
    print(f"[gate] {name}: {'PASS' if res.get('pass') else 'FAIL'}", flush=True)
    return res


def _np(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def arms_info(out_dir: str) -> dict:
    return json.loads(Path(out_dir, "arms.json").read_text())


def floor_prompts() -> list[dict]:
    return json.loads((DATA / "floor_prompts.json").read_text())["prompts"]


def short_prompts(n: int, length: int = SHORT, seed: int = 0) -> list[list[int]]:
    """``n`` BOS-prefixed windows of generic text (floor generic prompts and plumbing sets A/B)."""
    texts = json.loads((DATA / "plumbing_texts.json").read_text())
    src = [p["ids"][1:] for p in floor_prompts() if p["kind"] == "generic"] + [x[1:] for x in texts["A"] + texts["B"]]
    pool = [[0] + s[o: o + length - 1] for s in src for o in range(0, len(s) - length + 1, length // 2)]
    random.Random(seed).shuffle(pool)
    if len(pool) < n:
        pool = (pool * (n // len(pool) + 1))[:n]
    return pool[:n]


def dev(s: dict) -> dict:
    """Device counters of a ``stats`` result as arrays (counts since the last ``stats(reset=True)``)."""
    return {k: np.asarray(s[k]) for k in ("tokens_seen", "tokens_edited", "sum_abs_delta", "max_err", "nonfinite",
                                          "cast_violations")}


def reset_counted(client) -> dict:
    """Start a counted block: wait for an idle engine (no stray traffic), then read-and-zero the counters."""
    client.wait_idle()
    return client.stats(reset=True)


def prompt_floors(floor: dict) -> dict[str, dict]:
    """Per floor prompt id, the worse of (a vs b) and (a vs c) on that prompt alone, with the minimums above."""
    runs = load_runs(floor["runs"])
    ids = [str(x) for x in np.load(floor["runs"])["prompt_ids"]]
    out = {}
    for i, pid in enumerate(ids):
        ab = compare([runs["a"][i]], [runs["b"][i]], ["x"])["x"]
        ac = compare([runs["a"][i]], [runs["c"][i]], ["x"])["x"]
        out[pid] = {"median": max(ab["median"], ac["median"], MIN_PROMPT_MEDIAN),
                    "p95": max(ab["p95"], ac["p95"], MIN_PROMPT_P95),
                    "top1_agree": min(ab["top1_agree"], ac["top1_agree"], MAX_PROMPT_TOP1)}
    return out


def within_prompt_floor(cmp: dict, ref: dict) -> bool:
    """The floor rule (1.5x median, 2x p95, top-1 - 1 point) against one prompt's own floor."""
    return (cmp["median"] <= 1.5 * ref["median"] and cmp["p95"] <= 2 * ref["p95"]
            and cmp["top1_agree"] >= ref["top1_agree"] - 0.01)


def check_floor_prompts(floor: dict) -> None:
    """The floor runs must be of the current ``data/floor_prompts.json`` (same ids, same lengths, same order)."""
    z = np.load(floor["runs"])
    ids, prompts = [str(x) for x in z["prompt_ids"]], floor_prompts()
    lens = np.diff(z["a_off"]).tolist()
    if ids != [p["id"] for p in prompts] or lens != [len(p["ids"]) - 1 for p in prompts]:
        raise SystemExit(f"floor {floor['runs']} was not scored on the current data/floor_prompts.json")


def ensure_gate_arms(client, out_dir: str) -> dict:
    """Reload the gate arms when the engine holds other arms (e.g. all zero after a reboot)."""
    info, inst = arms_info(out_dir), client.installed()
    if inst.get("arms_sha") == info["gate_sha"]:
        return {"reloaded": False, "boot": inst.get("boot")}
    client.load_arms(info["gate_path"])
    return {"reloaded": True, "was": inst.get("arms_sha"), "boot": inst.get("boot")}


def fields_for(info: dict, arm: int, **kw) -> dict:
    return request_fields(arm, arms_sha=info["gate_sha"], **kw)


# ----------------------------------------------------------------------------------------------------------------------
# install check
# ----------------------------------------------------------------------------------------------------------------------

def gate_install(client, out_dir: str, engine_log: str | None = None, edit_layers: str = DEFAULT_EDIT_LAYERS,
                 mhc_overlap: bool = True) -> dict:
    """``dsv41_installed`` on all 4 ranks: the slots of ``edit_layers`` (default ``12,14..38``, 26 slots), all wrapped,
    fusion off (layer flag, model flag, no deferred MoE finalize, fused kernel never built), side stream present
    (absent with ``DSV41_MHC_OVERLAP=0``), no sequence parallel or spec decoding; the engine log (required) has neither
    fused all-reduce line."""
    want = parse_layers(edit_layers)
    ranks = client.rpc("dsv41_installed")
    checks = {"ranks": len(ranks) == 4}
    for r in ranks:
        k = f"rank{r.get('rank')}"
        checks[f"{k}_installed"] = r.get("installed") is True and r.get("layers") == want
        checks[f"{k}_wrapped"] = r.get("wrapped") == want
        checks[f"{k}_fusion_off"] = (r.get("fuse_mhc_all_reduce") is False and r.get("model_fuse_mhc_all_reduce") is False
                                     and r.get("deferred_finalize") == [] and r.get("all_reduce_mhc_built") is False)
        checks[f"{k}_mhc_stream"] = r.get("mhc_stream") is mhc_overlap
        checks[f"{k}_no_sp_no_spec"] = r.get("sequence_parallel") is False and r.get("spec_decode") is False
    log = {}
    checks["engine_log_given"] = bool(engine_log)
    if engine_log:
        text = Path(engine_log).read_text(errors="replace")
        log = {"lamport_fused_lines": text.count("Lamport all-reduce fused"),
               "finalize_fused_lines": text.count("finalize fused"),
               "dsv41_line": "dsv41_intervene: fusion off" in text,
               "autotune_cc063872": "cc063872" in text}
        checks["log_no_fused_lines"] = log["lamport_fused_lines"] == 0 and log["finalize_fused_lines"] == 0
        checks["log_dsv41_line"] = log["dsv41_line"]
    return write(out_dir, "install", {"pass": all(checks.values()), "checks": checks, "log": log,
                                      "rank0": ranks[0] if ranks else None})


# ----------------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _capture(client, prompts, fields_list, path) -> dict:
    """Score ``prompts`` under each capture field set, then dump the rings to ``path``."""
    for f in fields_list:
        client.score(prompts, f, concurrency=len(prompts))
    return client.dump(str(path))


def build_gate_arms(client, out_dir: str, alpha_sd: float = 8.0) -> dict:
    """Capture set A/B/C means at 26 under measure-only, build d26, d', c, mu/sd, write the 9-arm gate file, load it."""
    inst = client.installed()
    layers, n_arms = list(inst["layers"]), int(inst["n_arms"])
    I = len(layers)
    slot = {L: i for i, L in enumerate(layers)}
    ts = Path(out_dir).name
    run = int(time.time())  # dumps never overwrite, so a re-run of build in the same folder needs new names
    arms_dir, cap_dir = ARMS_DIR / ts, CAPTURE_DIR / ts
    arms_dir.mkdir(parents=True, exist_ok=True)
    info = {"layers": layers, "n_arms": n_arms, "slot26": slot[L26], "arms_dir": str(arms_dir),
            "capture_dir": str(cap_dir), "capture_layers": list(inst.get("capture_layers", []))}

    # 0. all-sham file, so captures have an arms_sha to salt with
    zero = empty_tables(I, n_arms)
    info["zero_sha"] = write_arms_file(arms_dir / "gate_zero", layers, labels={}, recipe={"gate": "zero"}, **zero)
    client.load_arms(str(arms_dir / "gate_zero.safetensors"))

    # 1. d26 from the stream means of sets A vs B (vector capture); d' from C vs B
    texts = json.loads((DATA / "plumbing_texts.json").read_text())
    can_vec = int(inst.get("vec_rows", 0)) > 0 and L26 in inst.get("capture_layers", [])
    if can_vec:
        means = {}
        for s in ("A", "B", "C"):
            f = request_fields(0, arms_sha=info["zero_sha"], capture="vec", cap_from=4, cap_to=132, tag=f"set{s}")
            dump = _capture(client, texts[s], [f], cap_dir / f"vec_{s}_{run}.safetensors")
            arr, meta = load_dump(dump["path"])
            means[s] = vec_means(arr, meta, L26, tag=f"set{s}").mean(0)
            info[f"vec_rows_{s}"] = int(dump["vec_rows"])
        d26, dprime = _unit(means["A"] - means["B"]), _unit(means["C"] - means["B"])
        info["direction_source"] = "vec capture, sets A/B/C at output 26"
    else:
        rng = np.random.default_rng(26)
        d26, dprime = _unit(rng.standard_normal(5120)), _unit(rng.standard_normal(5120))
        info["direction_source"] = "seeded random (vector capture not available on this boot)"
    rng = np.random.default_rng(1028)
    c = rng.standard_normal(5120)
    c = _unit(c - (c @ d26) * d26)
    for name, v in (("d26", d26), ("dprime", dprime), ("c", c)):
        np.save(Path(out_dir, f"{name}.npy"), v.astype(np.float32))
    info["cos_d26_dprime"] = float(d26 @ dprime)

    # 2. measure-only arms (1 = d26, 2 = d', 3 = c, all slots) on the 20 generic floor prompts -> mu, sd per slot
    t = empty_tables(I, n_arms)
    for arm, v in ((1, d26), (2, dprime), (3, c)):
        for i in range(I):
            set_slot(t, arm, i, "measure", d=v)
    info["measure_sha"] = write_arms_file(arms_dir / "gate_measure", layers,
                                          labels={1: "measure d26", 2: "measure d'", 3: "measure c"},
                                          recipe={"gate": "measure"}, **t)
    client.load_arms(str(arms_dir / "gate_measure.safetensors"))
    generic = [p["ids"] for p in floor_prompts() if p["kind"] == "generic"]
    fl = [request_fields(a, arms_sha=info["measure_sha"], capture="coef", cap_from=4, tag=f"m{a}") for a in (1, 2, 3)]
    dump = _capture(client, generic, fl, cap_dir / f"coef_measure_{run}.safetensors")
    stats = {a: mu_from_capture(dump["path"], a, f"m{a}") for a in (1, 2, 3)}
    info["mu"] = {k: stats[a]["mu"] for a, k in ((1, "d26"), (2, "dprime"), (3, "c"))}
    info["sd"] = {k: stats[a]["sd"] for a, k in ((1, "d26"), (2, "dprime"), (3, "c"))}
    info["mu_rows"] = {k: stats[a]["n"] for a, k in ((1, "d26"), (2, "dprime"), (3, "c"))}
    info["mu_dropped_nonfinite_rows"] = {k: stats[a]["n_dropped_nonfinite_requests"]
                                         for a, k in ((1, "d26"), (2, "dprime"), (3, "c"))}

    # 3. the 9 gate arms
    mu = lambda L: info["mu"]["d26"][str(L)]  # noqa: E731
    sd26 = float(np.mean(info["sd"]["d26"][str(L26)]))
    sdp = float(np.mean(info["sd"]["dprime"][str(L26)]))
    t = empty_tables(I, n_arms)
    for i, L in enumerate(layers):
        set_slot(t, 1, i, "measure", d=d26, mu=mu(L))
        set_slot(t, 3, i, "project", d=d26, mu=mu(L))
        if 22 <= L <= 34:
            set_slot(t, 7, i, "project", d=d26, mu=mu(L))
    s26 = slot[L26]
    set_slot(t, 2, s26, "project", d=d26, mu=mu(L26))
    set_slot(t, 4, s26, "add", d=d26, alpha=alpha_sd * sd26)
    set_slot(t, 5, s26, "shift", d=d26, c=c, mu=mu(L26))
    set_slot(t, 6, s26, "project_mc", d=d26, mu=mu(L26))
    set_slot(t, 8, s26, "add", d=dprime, alpha=alpha_sd * sdp)
    set_slot(t, 9, s26, "project", d=d26, mu=mu(L26))
    for L in (27, 30, 34):
        set_slot(t, 9, slot[L], "measure", d=d26, mu=mu(L))
    labels = {1: "measure d26 all", 2: "project {26}", 3: "project all", 4: f"add {alpha_sd} sd {{26}}",
              5: "matched shift {26} along c", 6: "mean-coupled project {26}", 7: "project {22..34}",
              8: f"add {alpha_sd} sd {{26}} along d'", 9: "project {26} + measure 27/30/34"}
    info["gate_path"] = str(arms_dir / "gate_arms.safetensors")
    info["gate_sha"] = write_arms_file(arms_dir / "gate_arms", layers, labels=labels,
                                       recipe={"gate": "arms", "alpha_sd": alpha_sd}, **t)
    info["load"] = client.load_arms(info["gate_path"])
    info["alpha"] = {"4": alpha_sd * sd26, "8": alpha_sd * sdp, "alpha_sd": alpha_sd}
    # arm 4's content in slot 8, for the reload gate
    t8 = {k: v.copy() for k, v in t.items()}
    for k in ("p", "e", "mu", "a", "g"):
        t8[k][8] = t[k][4]
    t8["mc"][8] = t["mc"][4]
    info["reload_path"] = str(arms_dir / "gate_arms_slot8_is_4.safetensors")
    info["reload_sha"] = write_arms_file(arms_dir / "gate_arms_slot8_is_4", layers, labels={**labels, 8: labels[4]},
                                         recipe={"gate": "reload"}, **t8)
    Path(out_dir, "arms.json").write_text(json.dumps(info, indent=1, default=_np))
    return write(out_dir, "build", {"pass": True, **{k: v for k, v in info.items() if k not in ("mu", "sd")}})


# ----------------------------------------------------------------------------------------------------------------------
# sham vs native
# ----------------------------------------------------------------------------------------------------------------------

def gate_sham_native(client, floor: dict, out_dir: str) -> dict:
    """The 40 floor prompts as sham vs native run (a). Pass/fail per kind (generic, agent, all) as the rule codes it;
    the per-sub-kind numbers (code, prose, agent) are information. Also proves the sham went through the wrapper:
    ``tokens_seen[0]`` equals the prompt tokens (retried requests counted again) and nothing was edited."""
    info = arms_info(out_dir)
    prompts = floor_prompts()
    ids, kinds = [p["ids"] for p in prompts], [p["kind"] for p in prompts]
    subs = [sub_kind(p["id"]) for p in prompts]
    native = load_runs(floor["runs"])["a"]
    s0 = reset_counted(client)
    sham = client.score(ids, fields_for(info, 0), concurrency=len(ids))
    s1 = client.stats()
    cmp = compare(native, sham, kinds)
    verdict = within_floor(cmp, floor_reference(floor))
    seen = int(np.asarray(s1["tokens_seen"])[0])
    expected = int(sum(len(x) * (1 + r["nan_retries"]) for x, r in zip(ids, sham)))
    edited_any = int(np.asarray(s1["tokens_edited"]).sum())
    invalid = s1["invalid_arm_requests"] - s0["invalid_arm_requests"]
    counts_ok = seen == expected and edited_any == 0 and invalid == 0
    return write(out_dir, "sham", {"pass": bool(verdict["ok"] and counts_ok), "compare": cmp, "verdict": verdict,
                                   "reference": floor_reference(floor), "counts_ok": counts_ok,
                                   "compare_sub_kinds": compare(native, sham, subs),
                                   "floor_sub_kinds": floor.get("sub_kinds"),
                                   "tokens_seen_0": seen, "tokens_expected": expected,
                                   "tokens_edited_any": edited_any,
                                   "nonfinite_0": np.asarray(s1["nonfinite"])[0].tolist(),
                                   "nan_retries": {p["id"]: r["nan_retries"] for p, r in zip(prompts, sham)
                                                   if r["nan_retries"]},
                                   "invalid_arm_requests": invalid})


# ----------------------------------------------------------------------------------------------------------------------
# positive control in every path (+ decode vs prefill on the same generations)
# ----------------------------------------------------------------------------------------------------------------------

PW_DECODE = 512  # max_tokens of the 16 decoding requests in the PIECEWISE case


def _cases() -> list[dict]:
    """Requests per execution path: prompt ids, max_tokens, and for the PIECEWISE case's late 600-token prompt the
    ``tokens_seen`` count to wait for (the 16 prompts prefilled plus two decode steps), so it joins a decode step."""
    out = []
    for path, cs in (("full_overlap", (1, 8, 16)), ("full", (17, 64, 256))):
        for c in cs:
            out.append({"path": path, "c": c, "reqs": [{"ids": p, "n": N_GEN} for p in short_prompts(c, seed=c)]})
    pw = [{"ids": p, "n": PW_DECODE} for p in short_prompts(16, seed=99)]
    pw.append({"ids": short_prompts(1, 600, seed=7)[0], "n": N_GEN, "after_seen": 16 * SHORT + 2 * 16})
    out.append({"path": "piecewise", "c": 17, "reqs": pw})
    agent = next(p["ids"] for p in floor_prompts() if p["kind"] == "agent" and len(p["ids"]) >= 4000)
    out.append({"path": "eager", "c": 1, "reqs": [{"ids": agent[:4000], "n": N_GEN}]})
    return out


def _generate(client, reqs: list[dict], fields: dict, arm: int) -> tuple[list[dict], list[dict]]:
    """Generate all requests at once; a request with ``after_seen`` is sent once ``tokens_seen[arm]`` reaches it.
    Returns the generations and, per request, the counter value when it was sent (None if sent at once)."""
    def one(r):
        sent = None
        if r.get("after_seen"):
            t0 = time.time()
            while True:
                sent = int(client.stats()["tokens_seen"][arm])
                if sent >= r["after_seen"] or time.time() - t0 > 120:
                    break
                time.sleep(0.05)
        return client.generate(r["ids"], fields, r["n"])["gen"][0], {"seen_at_send": sent}

    with ThreadPoolExecutor(len(reqs)) as ex:
        res = list(ex.map(one, reqs))
    return [g for g, _ in res], [w for _, w in res]


def _run_paths(client, info: dict, out_dir: str) -> list[dict]:
    """Generate every case under arm 4 and arm 0 with exact counter checks; cached in ``paths_raw.json`` for this
    engine boot and gate arms only."""
    cache = Path(out_dir, "paths_raw.json")
    key = {"boot": client.installed().get("boot"), "gate_sha": info["gate_sha"]}
    if cache.exists():
        old = json.loads(cache.read_text())
        if isinstance(old, dict) and old.get("key") == key:
            return old["rows"]
    s26 = info["slot26"]
    rows = []
    for case in _cases():
        row = {"path": case["path"], "c": case["c"], "P": [len(r["ids"]) for r in case["reqs"]],
               "N": [r["n"] for r in case["reqs"]], "ids": [r["ids"] for r in case["reqs"]]}
        for arm in (4, 0):
            reset_counted(client)
            gen, sent = _generate(client, case["reqs"], fields_for(info, arm), arm)
            d = dev(client.stats())
            row[f"gen{arm}"] = gen
            row[f"seen{arm}"] = int(d["tokens_seen"][arm])
            row[f"edited{arm}_26"] = int(d["tokens_edited"][arm][s26])
            row[f"edited{arm}_all"] = int(d["tokens_edited"][arm].sum())
            row[f"cast_violations{arm}"] = int(d["cast_violations"][arm].sum())
            row[f"nonfinite{arm}"] = int(d["nonfinite"][arm].sum())
            if case["path"] == "piecewise":  # the late prompt must join while the 16 others still decode
                late, decoding = sent[-1]["seen_at_send"], 16 * (SHORT + PW_DECODE - 1)
                row[f"late_seen_at_send{arm}"] = late
                row[f"late_mid_decode{arm}"] = late is not None and case["reqs"][-1]["after_seen"] <= late < decoding
        row["expect_seen"] = int(sum(p + n - 1 for p, n in zip(row["P"], row["N"])))
        row["expect_edited"] = int(sum(p + n - 2 for p, n in zip(row["P"], row["N"])))
        rows.append(row)
        print(f"  {case['path']} c={case['c']}: edited4_26={row['edited4_26']} expect={row['expect_edited']}",
              flush=True)
    cache.write_text(json.dumps({"key": key, "rows": rows}))
    return rows


def gate_positive_control(client, floor: dict, out_dir: str) -> dict:
    info = arms_info(out_dir)
    ref = floor_reference(floor)["generic"]
    rows = _run_paths(client, info, out_dir)
    res, ok = [], True
    for r in rows:
        div = [next((j for j, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
               for a, b in zip(r["gen4"], r["gen0"])]
        frac8 = float(np.mean([j is not None and j < 8 for j in div]))
        seqs = [p + g["ids"] for p, g in zip(r["ids"], r["gen0"])]
        tf4 = client.score(seqs, fields_for(info, 4))
        tf0 = client.score(seqs, fields_for(info, 0))
        med = compare(tf4, tf0, ["generic"] * len(seqs))["all"]["median"]
        row = {"path": r["path"], "c": r["c"], "edited4_26": r["edited4_26"], "expect_edited": r["expect_edited"],
               "seen4": r["seen4"], "seen0": r["seen0"], "expect_seen": r["expect_seen"],
               "edited0_all": r["edited0_all"], "diverge_within_8": frac8, "tf_median_absdelta": med,
               "tf_ratio_to_floor": med / max(ref["median"], 1e-12)}
        row["counts_ok"] = (r["edited4_26"] == r["expect_edited"] and r["seen4"] == r["expect_seen"]
                            and r["seen0"] == r["expect_seen"] and r["edited0_all"] == 0)
        row["path_ok"] = r["path"] != "piecewise" or (r["late_mid_decode4"] and r["late_mid_decode0"])
        row["nonfinite"] = r["nonfinite4"] + r["nonfinite0"]
        row["cast_violations4"] = r["cast_violations4"]
        row["ok"] = row["counts_ok"] and row["path_ok"] and frac8 >= 0.9 and med >= 10 * ref["median"]
        ok &= row["ok"]
        res.append(row)
    return write(out_dir, "positive", {"pass": bool(ok), "alpha": info["alpha"], "floor_generic": ref, "rows": res})


def _gen_lp(rows_P: list[int], gens: list[dict], tf: list[dict]) -> np.ndarray:
    """Teacher-forced logprobs of the generated tokens j >= 1 (token j sits at prompt index P + j; j = 0 was sampled
    from the prefill step), concatenated over requests."""
    return np.concatenate([t["lp"][P: P + len(g["ids"]) - 1] for P, g, t in zip(rows_P, gens, tf)])


def gate_decode_prefill(client, floor: dict, out_dir: str) -> dict:
    """Decode-time logprobs vs teacher-forced ones, same text, same arm. Rule (fixed before any run; arm 4's text is
    off-distribution, so arm 0's text is not its noise baseline): with ``med4 = median|decode4 - tf4a|`` on arm-4
    text, ``med_tf4 = median|tf4a - tf4b|`` (two teacher-forced runs of that text) and ``eff = median|tf4a - tf0|``
    (the same text scored as sham, i.e. what a missing decode edit would give), pass iff
    ``med4 <= 1.5 * max(med0, med_tf4) + 0.01`` and ``med4 <= 0.1 * eff``, in every path."""
    info = arms_info(out_dir)
    rows = _run_paths(client, info, out_dir)
    f0, f4 = fields_for(info, 0), fields_for(info, 4)
    res, ok = [], True
    for r in rows:
        seq0 = [p + g["ids"] for p, g in zip(r["ids"], r["gen0"])]
        seq4 = [p + g["ids"] for p, g in zip(r["ids"], r["gen4"])]
        dec0 = np.concatenate([np.asarray(g["lp"][1:]) for g in r["gen0"]])
        dec4 = np.concatenate([np.asarray(g["lp"][1:]) for g in r["gen4"]])
        tf0 = _gen_lp(r["P"], r["gen0"], client.score(seq0, f0))
        tf4a = _gen_lp(r["P"], r["gen4"], client.score(seq4, f4))
        tf4b = _gen_lp(r["P"], r["gen4"], client.score(seq4, f4))
        tf40 = _gen_lp(r["P"], r["gen4"], client.score(seq4, f0))
        med0, med4 = float(np.median(np.abs(dec0 - tf0))), float(np.median(np.abs(dec4 - tf4a)))
        med_tf4, eff = float(np.median(np.abs(tf4a - tf4b))), float(np.median(np.abs(tf4a - tf40)))
        bound = 1.5 * max(med0, med_tf4) + 0.01
        row = {"path": r["path"], "c": r["c"], "median_arm4": med4, "median_arm0": med0, "median_tf4_vs_tf4": med_tf4,
               "effect_tf4_vs_sham": eff, "noise_bound": bound,
               "ok": med4 <= bound and med4 <= 0.1 * eff}
        ok &= row["ok"]
        res.append(row)
    return write(out_dir, "decode", {"pass": bool(ok), "rows": res})


# ----------------------------------------------------------------------------------------------------------------------
# arm isolation
# ----------------------------------------------------------------------------------------------------------------------

def gate_arm_isolation(client, floor: dict, out_dir: str) -> dict:
    """Mixed batch (16 sham + 16 arm 4) vs single-arm batches. Each request is judged against its own prompt's native
    floor (``prompt_floors``): the pooled generic floor sits between code (~3e-4) and prose (~0.06) prompts, so prose
    twins would fail on noise alone. Pooled verdicts are information. Counters exact per arm, n = 8 children
    included; retried requests count again."""
    info = arms_info(out_dir)
    s26 = info["slot26"]
    ref = floor_reference(floor)
    pfl = prompt_floors(floor)
    gp = [p for p in floor_prompts() if p["kind"] == "generic"][:16]
    ids = [p["ids"] for p in gp]
    f0, f4 = fields_for(info, 0), fields_for(info, 4)
    s0 = reset_counted(client)
    with ThreadPoolExecutor(32) as ex:  # one batch of 32: 16 sham + 16 arm 4, interleaved
        futs = [ex.submit(client.score, [x], f, 32) for x in ids for f in (f0, f4)]
        mixed = [fu.result()[0] for fu in futs]
    d = dev(client.stats())
    m0, m4 = mixed[0::2], mixed[1::2]
    seen0 = int(sum(len(x) * (1 + r["nan_retries"]) for x, r in zip(ids, m0)))
    seen4 = int(sum(len(x) * (1 + r["nan_retries"]) for x, r in zip(ids, m4)))
    edit4 = int(sum((len(x) - 1) * (1 + r["nan_retries"]) for x, r in zip(ids, m4)))
    counts = {"seen0": int(d["tokens_seen"][0]), "seen4": int(d["tokens_seen"][4]), "expect_seen0": seen0,
              "expect_seen4": seen4, "edited4_26": int(d["tokens_edited"][4][s26]), "expect_edited": edit4,
              "edited0_all": int(d["tokens_edited"][0].sum()),
              "edited_other": int(np.delete(d["tokens_edited"], [4], axis=0).sum())}
    single0 = client.score(ids, f0, concurrency=16)
    single4 = client.score(ids, f4, concurrency=16)
    kinds = ["generic"] * len(ids)
    v0 = within_floor(compare(m0, single0, kinds), ref)
    v4 = within_floor(compare(m4, single4, kinds), ref)
    per_req = []
    for arm, mm, ss in ((0, m0, single0), (4, m4, single4)):
        for p, a, b in zip(gp, mm, ss):
            c = compare([a], [b], ["x"])["x"]
            per_req.append({"id": p["id"], "arm": arm, "median": c["median"], "p95": c["p95"],
                            "top1_agree": c["top1_agree"], "floor": pfl[p["id"]],
                            "ok": within_prompt_floor(c, pfl[p["id"]])})

    # n = 8 children: 2 requests under arm 4, 2 sham; prompt shorter than one KV block, so no prefix hit is possible
    short = short_prompts(4, 12, seed=8)
    N, n = 16, 8
    reset_counted(client)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda a: client.generate(a[0], a[1], N, n=n), zip(short, (f4, f4, f0, f0))))
    s2 = client.stats()
    d2 = dev(s2)
    nkids = {"seen4": int(d2["tokens_seen"][4]), "seen0": int(d2["tokens_seen"][0]),
             "expect_seen": 2 * n * (12 + N - 1), "edited4_26": int(d2["tokens_edited"][4][s26]),
             "expect_edited": 2 * n * (12 + N - 2)}
    invalid = s2["invalid_arm_requests"] - s0["invalid_arm_requests"]
    counts_ok = (counts["seen0"] == seen0 and counts["seen4"] == seen4 and counts["edited4_26"] == edit4
                 and counts["edited0_all"] == 0 and nkids["seen4"] == nkids["seen0"] == nkids["expect_seen"]
                 and nkids["edited4_26"] == nkids["expect_edited"] and invalid == 0)
    ok = counts_ok and all(r["ok"] for r in per_req)
    return write(out_dir, "isolation", {"pass": bool(ok), "counts_ok": counts_ok, "counts": counts, "n8": nkids,
                                        "invalid_arm_requests": invalid, "per_request": per_req,
                                        "pooled_mixed_vs_single_arm0": v0, "pooled_mixed_vs_single_arm4": v4})


# ----------------------------------------------------------------------------------------------------------------------
# prefix cache
# ----------------------------------------------------------------------------------------------------------------------

def gate_prefix_cache(client, floor: dict, out_dir: str) -> dict:
    """vLLM 0.31 skips reading the prefix cache whenever ``prompt_logprobs`` is set, so the cache-hit request is
    compared on 32 greedy decode logprobs (hit vs cold run of the same tokens), not on its 300 prompt positions."""
    info = arms_info(out_dir)
    ref = floor_reference(floor)
    agent = next(p["ids"] for p in floor_prompts() if p["kind"] == "agent" and len(p["ids"]) >= 6300)
    P = [0, random.randrange(1000, 100000)] + agent[1:5999]  # fresh second token: no stale blocks from earlier runs
    P300 = P + agent[5999:6299]
    f2 = fields_for(info, 2)

    def step(ids, fields, n_tok):
        reset_counted(client)
        out = client.generate(ids, fields, n_tok, unique=False)
        return out, dev(client.stats())

    _, d1 = step(P, f2, 1)
    hit, d2 = step(P300, f2, 32)
    cold = client.generate(P300, f2, 32)  # fresh salt: full recompute
    _, d3 = step(P, fields_for(info, 4), 1)
    _, d4 = step(P300, fields_for(info, 0), 1)
    g_hit, g_cold = hit["gen"][0], cold["gen"][0]
    same = next((j for j, (x, y) in enumerate(zip(g_hit["ids"], g_cold["ids"])) if x != y), len(g_hit["ids"]))
    dl = np.abs(np.asarray(g_hit["lp"][1:same]) - np.asarray(g_cold["lp"][1:same])) if same > 1 else np.zeros(1)
    res = {"len_P": len(P), "len_P300": len(P300), "cold_seen2": int(d1["tokens_seen"][2]),
           "hit_seen2": int(d2["tokens_seen"][2]), "hit_seen2_prompt": int(d2["tokens_seen"][2]) - 31,
           "arm4_seen4": int(d3["tokens_seen"][4]), "sham_seen0": int(d4["tokens_seen"][0]),
           "hit_cached_tokens": ((hit.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
           "decode_common_prefix": same, "decode_median_absdelta": float(np.median(dl)),
           "floor_agent_median": ref["agent"]["median"]}
    res["checks"] = {"cold_full": res["cold_seen2"] == len(P),
                     "hit": res["hit_seen2_prompt"] < 0.25 * len(P300),
                     "hit_matches_cold": res["decode_median_absdelta"] <= 1.5 * ref["agent"]["median"],
                     "arm4_full": res["arm4_seen4"] == len(P), "sham_full": res["sham_seen0"] == len(P300)}
    return write(out_dir, "prefix", {"pass": all(res["checks"].values()), **res})


# ----------------------------------------------------------------------------------------------------------------------
# arm reload
# ----------------------------------------------------------------------------------------------------------------------

def gate_arm_reload(client, floor: dict, out_dir: str) -> dict:
    info = arms_info(out_dir)
    ref = floor_reference(floor)
    ids = [p["ids"] for p in floor_prompts() if p["kind"] == "generic"]
    kinds = ["generic"] * len(ids)
    loads = [client.load_arms(info["gate_path"])]
    r1 = client.score(ids, request_fields(8, arms_sha=info["gate_sha"]))
    r4 = client.score(ids, request_fields(4, arms_sha=info["gate_sha"]))
    loads.append(client.load_arms(info["reload_path"]))
    r2 = client.score(ids, request_fields(8, arms_sha=info["reload_sha"]))
    loads.append(client.load_arms(info["gate_path"]))
    r3 = client.score(ids, request_fields(8, arms_sha=info["gate_sha"]))
    c21, c24, c31 = compare(r2, r1, kinds), compare(r2, r4, kinds), compare(r3, r1, kinds)
    checks = {"r2_differs": c21["all"]["median"] > 10 * ref["generic"]["median"],
              "r2_is_arm4": within_floor(c24, ref)["ok"], "r3_is_r1": within_floor(c31, ref)["ok"],
              "shas": [x["arms_sha"] for x in loads] == [info["gate_sha"], info["reload_sha"], info["gate_sha"]]}
    return write(out_dir, "reload", {"pass": all(checks.values()), "checks": checks, "r2_vs_r1": c21,
                                     "r2_vs_arm4": c24, "r3_vs_r1": c31,
                                     "checksums": [x.get("checksums") for x in loads]})


# ----------------------------------------------------------------------------------------------------------------------
# projection exact
# ----------------------------------------------------------------------------------------------------------------------

def gate_projection_exact(client, out_dir: str) -> dict:
    """Arms 3, 5, 6 on the 20 generic prompts with coef capture. Rule (fixed 28 Oct before any run; the bf16 cast
    shrinks small per-element steps, so an sd threshold can fail on correct code where a direction has little
    variance but large elements, e.g. outputs 34-38):
    - every edited slot: ``cast_violations == 0`` (every token within the bf16 rounding bound of ``ops.py``, a
      direction-independent statement of "exact"), non-finite 0 in stats and capture, stats ``max_err`` equal to the
      capture's, capture rows aligned with device positions;
    - output 26 only: the original thresholds, ``|c' - target| <= 0.05 sd`` at 100% and ``<= 0.02 sd`` at >= 99.9%;
    - every other slot: the same sd quantiles recorded, not gated."""
    from safetensors.numpy import load_file

    info = arms_info(out_dir)
    t = load_file(info["gate_path"])
    ep = np.einsum("aisd,aisd->ais", t["p"].astype(np.float64), t["e"].astype(np.float64))
    ids = [p["ids"] for p in floor_prompts() if p["kind"] == "generic"]
    layers = info["layers"]
    reset_counted(client)
    # concurrency 1 (decided 28 Oct 06:30 CET, before this gate's first run): the base model returns NaN on some
    # batched prefill requests (native 17/40 batched, 0/40 one by one; build saw 7/150), and a failed attempt's NaN
    # rows stay in the ring, which would fail "non-finite = 0" for a reason unrelated to the edit. At c=1 any NaN
    # retry fails the gate (write()), and every threshold below is unchanged.
    for arm in (3, 5, 6):
        client.score(ids, request_fields(arm, arms_sha=info["gate_sha"], capture="coef", tag=f"x{arm}"), concurrency=1)
    st = dev(client.stats())
    dump = client.dump(str(Path(info["capture_dir"]) / f"coef_projection_{int(time.time())}.safetensors"))
    arr, meta = load_dump(dump["path"])  # raises if capture rows and device positions disagree
    rows, ok = [], True
    for arm in (3, 5, 6):
        x = coef_rows(arr, meta, arm, f"x{arm}", min_pos=1)  # [n, I, 3, 4]
        for i, L in enumerate(layers):
            a, g = float(t["a"][arm, i]), float(t["g"][arm, i])
            if a == 0 and g == 0:
                continue
            c, cp, mu = x[:, i, 0], x[:, i, 1], t["mu"][arm, i].astype(np.float64)
            delta = (c.mean(1, keepdims=True) - mu.mean()) if t["mc"][arm] else c - mu
            target = c - a * delta * ep[arm, i] + g * ep[arm, i]
            sd = np.asarray(info["sd"]["d26"][str(L)])
            err = np.abs(cp - target) / sd  # in sd units, per stream
            worst = err.max(1)
            row = {"arm": arm, "layer": L, "n": int(len(worst)), "edited": int(st["tokens_edited"][arm][i]),
                   "max_err_sd": float(worst.max()), "q999_err_sd": float(np.quantile(worst, 0.999)),
                   "median_err_sd": float(np.median(worst)), "frac_le_0.02sd": float((worst <= 0.02).mean()),
                   "cast_violations": int(st["cast_violations"][arm][i]),
                   "nonfinite_stats": int(st["nonfinite"][arm][i]),
                   "nonfinite_capture": int((~np.isfinite(x[:, i, :2])).sum()),
                   "max_err_stats": float(st["max_err"][arm][i]),
                   "max_err_capture": float(np.abs(cp - target).max())}
            row["exact_ok"] = (row["edited"] > 0 and row["cast_violations"] == 0 and row["nonfinite_stats"] == 0
                               and row["nonfinite_capture"] == 0
                               and abs(row["max_err_stats"] - row["max_err_capture"])
                               <= 1e-3 * max(1.0, row["max_err_capture"]))
            row["sd_ok"] = row["max_err_sd"] <= 0.05 and row["frac_le_0.02sd"] >= 0.999
            row["ok"] = row["exact_ok"] and (row["sd_ok"] or L != L26)
            ok &= row["ok"]
            rows.append(row)
    return write(out_dir, "projection", {"pass": bool(ok), "dump": dump["path"],
                                         "coef_pos_dev_checked": "coef_pos_dev" in arr, "rows": rows})


# ----------------------------------------------------------------------------------------------------------------------
# info (numbers for the record; no pass/fail)
# ----------------------------------------------------------------------------------------------------------------------

def _w(c, mu, a, g, mc):
    """The kernel's per-stream step ``w = g - a * delta`` from captured coefficients ``c`` [n, 4]."""
    delta = (c.mean(1, keepdims=True) - mu.mean()) if mc else c - mu
    return g - a * delta


def gate_info(client, out_dir: str) -> dict:
    """Information only: arm 9 persistence (fraction of the A-vs-B gap left at 27, 30, 34 after
    removal at 26), per-stream displacement |w| for arms 2, 5, 6 at 26, Var(c')/Var(c) and the same for the fold
    ``sum_k pre_k c_k``; plus the cross-engine check data (step 9): sham vector captures at outputs 25, 26, 27 with
    per-prompt tags and their token ids, saved under ``$DSV41_XCHECK_DIR/<gates folder>/``."""
    from safetensors.numpy import load_file
    from safetensors.torch import load_file as load_torch, save_file

    info = arms_info(out_dir)
    sha, layers = info["gate_sha"], info["layers"]
    slot = {L: i for i, L in enumerate(layers)}
    t = load_file(info["gate_path"])
    texts = json.loads((DATA / "plumbing_texts.json").read_text())
    generic = [p["ids"] for p in floor_prompts() if p["kind"] == "generic"]
    run = int(time.time())
    client.wait_idle()
    for arm in (1, 9):
        for s in ("A", "B"):
            client.score(texts[s], request_fields(arm, arms_sha=sha, capture="coef", cap_from=4, tag=f"p{arm}{s}"),
                         concurrency=len(texts[s]))
    for arm in (2, 5, 6):
        client.score(generic, request_fields(arm, arms_sha=sha, capture="coef", cap_from=4, tag=f"d{arm}"))
    dump = client.dump(str(Path(info["capture_dir"]) / f"coef_info_{run}.safetensors"))
    arr, meta = load_dump(dump["path"])
    res = {"dump": dump["path"], "persistence": {}, "displacement": {}, "var_ratio": {}}
    res["nonfinite_requests_dropped"] = int(len(nonfinite_requests(arr)))  # information only; see tools/arms.py
    mean_c = lambda arm, s, L, k: coef_rows(arr, meta, arm, f"p{arm}{s}", finite_requests_only=True)[  # noqa: E731
        :, slot[L], k].mean()
    for L in (26, 27, 30, 34):
        if L not in slot:
            continue
        k = 1 if L == L26 else 0  # at 26 arm 9 removes, so read c' there; downstream the pre-edit coefficient
        g1 = mean_c(1, "A", L, k) - mean_c(1, "B", L, k)
        g9 = mean_c(9, "A", L, k) - mean_c(9, "B", L, k)
        res["persistence"][str(L)] = {"gap_arm1": float(g1), "gap_arm9": float(g9),
                                      "fraction_left": float(g9 / g1) if g1 else None}
    s26 = slot[L26]
    for arm in (2, 5, 6):
        x = coef_rows(arr, meta, arm, f"d{arm}", finite_requests_only=True)
        c, cp, pre = x[:, s26, 0], x[:, s26, 1], x[:, s26, 2]
        w = np.abs(_w(c, t["mu"][arm, s26].astype(np.float64), float(t["a"][arm, s26]), float(t["g"][arm, s26]),
                      int(t["mc"][arm])))
        res["displacement"][str(arm)] = {"n": int(len(w)), "median": np.median(w, 0).tolist(),
                                         "p95": np.percentile(w, 95, axis=0).tolist(), "mean": w.mean(0).tolist()}
        res["var_ratio"][str(arm)] = {"per_stream": (cp.var(0) / c.var(0)).tolist(),
                                      "fold": float((pre * cp).sum(1).var() / (pre * c).sum(1).var())}

    # cross-engine check data: sham, per-prompt tags, outputs 25/26/27 only
    xdir = XCHECK_DIR / Path(out_dir).name
    n_x = int(os.environ.get("DSV41_XCHECK_PROMPTS", 30))  # prompts per set (the mock run uses fewer)
    xdir.mkdir(parents=True, exist_ok=True)
    keep = [L for L in (25, 26, 27) if L in info.get("capture_layers", [])]
    if keep and int(client.installed().get("vec_rows", 0)) > 0:
        jobs = [(s, i, ids) for s in ("A", "B", "C") for i, ids in enumerate(texts[s][:n_x])]
        with ThreadPoolExecutor(32) as ex:  # arm 0 (sham) is all zero in every arms file
            list(ex.map(lambda j: client.score([j[2]], request_fields(0, arms_sha=sha, capture="vec", cap_from=4,
                                                                      cap_to=132, tag=f"{j[0]}{j[1]:02d}"), 32),
                        jobs))
        vd = client.dump(str(Path(info["capture_dir"]) / f"vec_xcheck_{run}.safetensors"))
        tens = load_torch(vd["path"])
        side = json.loads(Path(vd["path"] + ".json").read_text())
        cols = [side["capture_layers"].index(L) for L in keep]
        out = {k: v for k, v in tens.items() if not k.startswith("coef")}
        out["vec"] = tens["vec"][:, cols].contiguous()
        save_file(out, str(xdir / "vec_25_26_27.safetensors"))
        (xdir / "vec_25_26_27.safetensors.json").write_text(json.dumps({**side, "capture_layers": keep}, indent=1))
        (xdir / "token_ids.json").write_text(json.dumps({"tags": "<set><prompt index>", "cap_from": 4, "cap_to": 132,
                                                         "sets": {s: texts[s][:n_x] for s in ("A", "B", "C")},
                                                         "tokenizer_sha256": texts.get("tokenizer_sha256")}))
        res["xcheck"] = {"dir": str(xdir), "rows": int(tens["vec"].shape[0]), "layers": keep}
    else:
        res["xcheck"] = {"skipped": "vector capture not available on this boot"}
    return write(out_dir, "info", {"pass": True, **res})


# ----------------------------------------------------------------------------------------------------------------------
# retry gates (written 28 Oct ~11:30 CET, before any run of them; the reading table is in docs/GATES.md)
#
# They re-measure the floors behind three failures of the first GPU boots (28 Oct). No old gate is changed or replaced: the old
# gates above run in the same suite, and each new result carries the old one next to it. Extra arms live in a second
# arms file (arms 0-9 byte-identical in content to the gate file, 10-15 new); every new gate reloads the gate file at
# its end, so later gates find the state they expect.
# ----------------------------------------------------------------------------------------------------------------------

# Arms 10-13 of the extended file: adds along d26 at output 26 with arm 4's p/e bits, only g differs; sizes in the mean
# sd of d26 at 26 (arm 4 = 8). 10 and 11 are the weak arms of the repeat gate (10 is gated; 11 only if 10 is not
# sensitive); 11-13 are the decode-effect candidates. In the reload twin, slots 14 and 15 hold arms 10's and 11's content.
EXT_ADD_SD = {10: 0.5, 11: 1.0, 12: 2.0, 13: 4.0}
WEAK_ARMS = (10, 11)
# Follow-up only, not the pre-written rule (28 Oct retry: the 0.5 and 1 sd adds moved logprobs by < 10 F, so the weak
# arm was untested): DSV41_WEAK_ADD_SD="2,4" re-sizes arms 10 and 11. Unset = the pre-written sizes above.
if os.environ.get("DSV41_WEAK_ADD_SD"):
    EXT_ADD_SD.update(zip(WEAK_ARMS, map(float, os.environ["DSV41_WEAK_ADD_SD"].split(","))))
STRONG_ARMS = (4, 8)
MEDIUM_ARMS = (11, 12, 13)
EXT_RELOAD = {14: 10, 15: 11}
REPEAT_Q = 0.95       # repeat floor: this quantile of the arm's own repeat distances (numpy's default, linear)
SENS_X = 10.0         # a weak arm must move logprobs by >= 10x the pooled generic floor (the positive control's factor)
LEAK_FRAC = 0.1       # reading only (docs/GATES.md): sham drift / arm effect at or above this reads as a leak
BF16_REL = 2.0 ** -8  # half a bf16 step, relative to the value
SD_MAX, SD_FRAC, FRAC_MIN = 0.05, 0.02, 0.999  # the original projection thresholds, unchanged
EFF_MIN = 0.5         # decode effect: the 0.1 * effect cap counts only where the effect on the decoded text is >= this


def ext_arms(out_dir: str) -> dict:
    """The extended arms file (gate arms + arms 10-13) and its reload twin (slots 14/15 = arms 10/11's content),
    written once per gates folder and gate sha (``arms_ext.json``). Loads nothing."""
    from safetensors.numpy import load_file

    info = arms_info(out_dir)
    p = Path(out_dir, "arms_ext.json")
    if p.exists():
        ext = json.loads(p.read_text())
        if ext.get("gate_sha") == info["gate_sha"] and Path(ext["path"]).exists() and Path(ext["reload_path"]).exists():
            return ext
    t = {k: v.copy() for k, v in load_file(info["gate_path"]).items()}
    layers = [int(x) for x in t.pop("layers")]
    n_arms, s26 = t["a"].shape[0], info["slot26"]
    if n_arms < 16:
        raise ValueError(f"the extended gate arms need 16 arms, the engine has {n_arms}")
    for k in range(10, 16):
        if any(np.any(t[name][k] != 0) for name in ("p", "e", "mu", "a", "g", "mc")):
            raise ValueError(f"gate file arm {k} is not empty; the extended arms would overwrite it")
    sd26 = float(np.mean(info["sd"]["d26"][str(L26)]))
    for k, ksd in EXT_ADD_SD.items():
        t["p"][k, s26], t["e"][k, s26] = t["p"][4, s26], t["e"][4, s26]
        t["g"][k, s26] = ksd * sd26
    side = json.loads(Path(info["gate_path"]).with_suffix(".json").read_text())
    labels = {int(k): v for k, v in side.get("labels", {}).items()}
    labels.update({k: f"add {ksd} sd {{26}}" for k, ksd in EXT_ADD_SD.items()})
    arms_dir = ARMS_DIR / Path(out_dir).name
    arms_dir.mkdir(parents=True, exist_ok=True)
    ext = {"gate_sha": info["gate_sha"], "sd26": sd26, "add_sd": {str(k): v for k, v in EXT_ADD_SD.items()},
           "alpha": {str(k): v * sd26 for k, v in EXT_ADD_SD.items()},
           "path": str(arms_dir / "gate_arms_ext.safetensors"),
           "sha": write_arms_file(arms_dir / "gate_arms_ext", layers, labels=labels,
                                  recipe={"gate": "ext", "add_sd": EXT_ADD_SD, "from": info["gate_sha"]}, **t)}
    t2 = {k: v.copy() for k, v in t.items()}
    for slot, src in EXT_RELOAD.items():
        for name in ("p", "e", "mu", "a", "g", "mc"):
            t2[name][slot] = t[name][src]
    ext["reload_path"] = str(arms_dir / "gate_arms_ext_reload.safetensors")
    ext["reload_sha"] = write_arms_file(arms_dir / "gate_arms_ext_reload", layers,
                                        labels={**labels, **{s: labels[k] for s, k in EXT_RELOAD.items()}},
                                        recipe={"gate": "ext_reload", "slots": EXT_RELOAD}, **t2)
    p.write_text(json.dumps(ext, indent=1))
    return ext


def req_medians(xs: list[dict], ys: list[dict]) -> np.ndarray:
    """Per request, the median over its positions of ``|lp_x - lp_y|``."""
    return np.array([float(np.median(np.abs(np.asarray(a["lp"]) - np.asarray(b["lp"])))) for a, b in zip(xs, ys)])


def repeat_rule(rep, cross, floor_median: float, q: float = REPEAT_Q) -> dict:
    """``median(cross) <= quantile(rep, q) + floor_median`` (the repeat-floor rule), with diagnostics."""
    rep, cross = np.asarray(rep, np.float64), np.asarray(cross, np.float64)
    rq = float(np.quantile(rep, q))
    stat = float(np.median(cross))
    return {"cross_median": stat, "rep_median": float(np.median(rep)), f"rep_q{int(q * 100)}": rq,
            "floor_median": floor_median, "limit": rq + floor_median, "ok": bool(stat <= rq + floor_median),
            "cross_over_rep_median": stat / max(float(np.median(rep)), 1e-12),
            "frac_cross_above_rep_q": float((cross > rq).mean()), "n_rep": int(rep.size), "n_cross": int(cross.size)}


def original_isolation(mm: list[dict], ss: list[dict], prompts: list[dict], pfl: dict) -> list[dict]:
    """The old isolation rule per request: ``mm[i]`` vs ``ss[i]`` within prompt i's own native floor."""
    out = []
    for p, a, b in zip(prompts, mm, ss):
        c = compare([a], [b], ["x"])["x"]
        out.append({"id": p["id"], "median": c["median"], "p95": c["p95"], "top1_agree": c["top1_agree"],
                    "floor": pfl[p["id"]], "ok": within_prompt_floor(c, pfl[p["id"]]),
                    "nan_retries": [a.get("nan_retries", 0), b.get("nan_retries", 0)]})
    return out


def _mixed_batch(client, ids: list[list[int]], f0: dict, fk: dict, k: int, s26: int) -> dict:
    """The isolation gate's mixed batch (sham and arm k interleaved, one batch) with its exact counter checks."""
    s0 = reset_counted(client)
    n = 2 * len(ids)
    with ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(client.score, [x], f, n) for x in ids for f in (f0, fk)]
        res = [fu.result()[0] for fu in futs]
    s1 = client.stats()
    d = dev(s1)
    m0, mk = res[0::2], res[1::2]
    exp0 = int(sum(len(x) * (1 + r["nan_retries"]) for x, r in zip(ids, m0)))
    expk = int(sum(len(x) * (1 + r["nan_retries"]) for x, r in zip(ids, mk)))
    expe = int(sum((len(x) - 1) * (1 + r["nan_retries"]) for x, r in zip(ids, mk)))
    c = {"seen0": int(d["tokens_seen"][0]), "seen_k": int(d["tokens_seen"][k]), "expect_seen0": exp0,
         "expect_seen_k": expk, "edited_k_26": int(d["tokens_edited"][k][s26]),
         "edited_k_all": int(d["tokens_edited"][k].sum()), "expect_edited": expe,
         "edited_other": int(np.delete(d["tokens_edited"], [k], axis=0).sum()),
         "invalid_arm_requests": int(s1["invalid_arm_requests"] - s0["invalid_arm_requests"]),
         "nan_retried_requests": int(sum(bool(r["nan_retries"]) for r in res))}
    ok = (c["seen0"] == exp0 and c["seen_k"] == expk and c["edited_k_26"] == expe and c["edited_k_all"] == expe
          and c["edited_other"] == 0 and c["invalid_arm_requests"] == 0)
    return {"sham": m0, "arm": mk, "counts": c, "counts_ok": bool(ok)}


def gate_repeat_floor(client, floor: dict, out_dir: str) -> dict:
    """Repeat floor for the isolation and reload failures of 28 Oct (arm 4, an 8 sd add at 26: 0.436 and 0.389 per
    the eager isolation and graph reload gates, against a sham floor of 0.013). Distance between two runs of one
    request: the median over its positions of ``|Δ logprob|``. ``F`` = the pooled generic native floor median.

    Part A (isolation), the 16 isolation prompts. For arms 4 and 8 (strong, 8 sd) and 10, 11 (weak, 0.5 and 1 sd adds
    along d26 at 26): the isolation gate's mixed batch (16 sham + 16 arm k, one batch of 32) twice (M1, M2) and arm k
    alone at concurrency 16 once (S); sham alone at concurrency 16 once (S0, shared by all arms).
    ``rep`` = d(M1, M2) per request (16 values), ``cross`` = d(M1, S) and d(M2, S) (32 values).
    Part B (reload), the 20 generic prompts at concurrency 64 like the reload gate: arm 4 in its own slot twice, then the
    reload file's slot 8 (arm 4's content) twice. ``rep`` = the two same-slot pairs (40 values), ``cross`` = slot 8 vs
    slot 4, first with first and second with second (40 values). Weak reload: arms 10 and 11 in their own slots vs the
    same content in slots 14 and 15 (reload twin), plus sham.

    Pass rule (fixed 28 Oct ~11:30 CET, before any run):
    - strong arm (A: arms 4 and 8; B: slot 8 vs slot 4): ``median(cross) <= quantile(rep, 0.95) + F``;
    - weak arm: arm 10 is gated, or arm 11 if arm 10 alone vs sham alone moves the pooled median |Δ| by less than
      ``10 F`` (not sensitive); neither sensitive = fail. The gated weak arm must pass the ORIGINAL isolation rule
      (every M1 request vs S within its own prompt's native floor) and the ORIGINAL reload rule (reloaded slot vs own
      slot ``within_floor`` of the pooled native floor, and reloaded slot vs sham median > ``10 F``);
    - every sham request of the M1 batches of arm 4 and of the gated weak arm within its own prompt's native floor
      (the old isolation rule); M2 shams and the other arms' shams are recorded, not gated;
    - exact counters in every mixed batch (sham and arm k seen = prompt tokens incl. NaN retries, arm k edited at 26
      only, nothing else edited, 0 invalid arms) and the load shas in order.
    A leak between arms shows up as the weak arm failing or a gated sham missing its floor; a strong arm whose cross
    distance exceeds its own repeat noise means the edit depends on its batch neighbours. ``reading`` applies the
    pre-written table (docs/GATES.md); the old results are in ``old``."""
    t_start = time.time()
    info = arms_info(out_dir)
    ext = ext_arms(out_dir)
    s26 = info["slot26"]
    ref = floor_reference(floor)
    F = float(ref["generic"]["median"])
    pfl = prompt_floors(floor)
    generic = [p for p in floor_prompts() if p["kind"] == "generic"]
    gp = generic[:16]
    ids = [p["ids"] for p in gp]
    ids20, k16, k20 = [p["ids"] for p in generic], ["generic"] * len(gp), ["generic"] * len(generic)
    fe = lambda k: request_fields(k, arms_sha=ext["sha"])  # noqa: E731
    loads, part_a, raw = [], {}, {}
    try:
        loads.append(client.load_arms(ext["path"]))
        S0 = client.score(ids, fe(0), concurrency=16)
        for k in STRONG_ARMS + WEAK_ARMS:
            M = [_mixed_batch(client, ids, fe(0), fe(k), k, s26) for _ in range(2)]
            S = client.score(ids, fe(k), concurrency=16)
            raw[k] = (M, S)
        loads.append(client.load_arms(info["gate_path"]))
        R4 = [client.score(ids20, request_fields(4, arms_sha=info["gate_sha"])) for _ in range(2)]
        loads.append(client.load_arms(info["reload_path"]))
        R2 = [client.score(ids20, request_fields(8, arms_sha=info["reload_sha"])) for _ in range(2)]
        loads.append(client.load_arms(ext["path"]))
        W = {k: client.score(ids20, fe(k)) for k in WEAK_ARMS}
        S0_20 = client.score(ids20, fe(0))
        loads.append(client.load_arms(ext["reload_path"]))
        WR = {k: client.score(ids20, request_fields(s, arms_sha=ext["reload_sha"])) for s, k in EXT_RELOAD.items()}
    finally:
        loads.append(client.load_arms(info["gate_path"]))  # later gates expect the gate arms
    shas_ok = [x["arms_sha"] for x in loads] == [ext["sha"], info["gate_sha"], info["reload_sha"], ext["sha"],
                                                  ext["reload_sha"], info["gate_sha"]]

    for k, (M, S) in raw.items():
        rep = req_medians(M[0]["arm"], M[1]["arm"])
        c1, c2 = req_medians(M[0]["arm"], S), req_medians(M[1]["arm"], S)
        eff_i = req_medians(S, S0)  # arm k alone vs sham alone, per request (the size a full leak would have)
        sham1 = original_isolation(M[0]["sham"], S0, gp, pfl)
        for r, drift, e in zip(sham1, req_medians(M[0]["sham"], S0), eff_i):
            r["leak_fraction"] = float(drift / max(e, 1e-12))
        row = {"arm": k, "add_sd": 8.0 if k in STRONG_ARMS else EXT_ADD_SD[k],
               "counts": [m["counts"] for m in M], "counts_ok": all(m["counts_ok"] for m in M),
               "repeat": repeat_rule(rep, np.concatenate([c1, c2]), F),
               "paired_cross1_gt_rep": int((c1 > rep).sum()), "n_requests": len(ids),
               "per_request": [{"id": p["id"], "rep": float(a), "cross1": float(b), "cross2": float(c),
                                "effect_vs_sham": float(e)} for p, a, b, c, e in zip(gp, rep, c1, c2, eff_i)],
               "original_rule": original_isolation(M[0]["arm"], S, gp, pfl),
               "pooled_m1_vs_alone": within_floor(compare(M[0]["arm"], S, k16), ref),
               "effect_vs_sham": compare(S, S0, k16)["all"]["median"],
               "sham_m1": sham1, "sham_m2": original_isolation(M[1]["sham"], S0, gp, pfl)}
        row["original_rule_ok"] = all(r["ok"] for r in row["original_rule"])
        row["sensitive"] = bool(row["effect_vs_sham"] >= SENS_X * F)
        row["sham_m1_ok"] = all(r["ok"] for r in sham1)
        row["sham_m2_ok"] = all(r["ok"] for r in row["sham_m2"])
        part_a[str(k)] = row
    weak = next((k for k in WEAK_ARMS if part_a[str(k)]["sensitive"]), None)

    rep_b = np.concatenate([req_medians(R4[0], R4[1]), req_medians(R2[0], R2[1])])
    cross_b = np.concatenate([req_medians(R2[0], R4[0]), req_medians(R2[1], R4[1])])
    part_b = {"strong": repeat_rule(rep_b, cross_b, F),
              "old_r2_is_arm4": within_floor(compare(R2[0], R4[0], k20), ref),
              "r2_vs_arm4": compare(R2[0], R4[0], k20)["all"], "r4_vs_r4": compare(R4[0], R4[1], k20)["all"],
              "r2_vs_r2": compare(R2[0], R2[1], k20)["all"], "weak": {}}
    for s, k in EXT_RELOAD.items():
        c_own, c_sham = compare(WR[k], W[k], k20), compare(WR[k], S0_20, k20)
        part_b["weak"][str(k)] = {"reloaded_slot": s, "vs_own_slot": c_own["all"], "vs_sham": c_sham["all"],
                                  "original_rule_ok": within_floor(c_own, ref)["ok"],
                                  "differs_from_sham": bool(c_sham["all"]["median"] > SENS_X * F)}

    strong_a_ok = all(part_a[str(k)]["repeat"]["ok"] for k in STRONG_ARMS)
    counts_ok = all(r["counts_ok"] for r in part_a.values())
    weak_a_ok = weak is not None and part_a[str(weak)]["original_rule_ok"]
    sham_ok = part_a["4"]["sham_m1_ok"] and weak is not None and part_a[str(weak)]["sham_m1_ok"]
    weak_b = part_b["weak"].get(str(weak)) if weak is not None else None
    weak_b_ok = bool(weak_b and weak_b["original_rule_ok"] and weak_b["differs_from_sham"])
    checks = {"isolation_strong_within_own_repeat": strong_a_ok, "isolation_weak_original_rule": weak_a_ok,
              "isolation_gated_shams_within_floor": sham_ok, "isolation_counters_exact": counts_ok,
              "reload_strong_within_own_repeat": part_b["strong"]["ok"], "reload_weak_original_rule": weak_b_ok,
              "load_shas": shas_ok}
    res = {"pass": all(checks.values()), "isolation_pass": strong_a_ok and weak_a_ok and sham_ok and counts_ok,
           "reload_pass": part_b["strong"]["ok"] and weak_b_ok and shas_ok, "checks": checks, "gated_weak_arm": weak,
           "floor_generic": ref["generic"], "F": F, "arms_ext": ext, "part_a": part_a, "part_b": part_b,
           "old": _old_results(out_dir, ("isolation", "reload")), "seconds": time.time() - t_start}
    res["reading"] = _reading_repeat(res)
    return write(out_dir, "repeat", res)


def _old_results(out_dir: str, names: tuple[str, ...]) -> dict:
    """The old gates' verdicts in this folder (``pass`` plus the headline numbers), for the side-by-side report."""
    out = {}
    for n in names:
        p = Path(out_dir, f"{n}.json")
        if not p.exists():
            out[n] = None
            continue
        r = json.loads(p.read_text())
        keep = {"pass": r.get("pass"), "error": r.get("error")}
        if n == "isolation" and "per_request" in r:
            a4 = [x["median"] for x in r["per_request"] if x["arm"] == 4]
            keep.update({"arm4_median_of_medians": float(np.median(a4)) if a4 else None,
                         "n_requests_outside_floor": sum(not x["ok"] for x in r["per_request"]),
                         "counts_ok": r.get("counts_ok")})
        if n == "reload" and "checks" in r:
            keep.update({"checks": r["checks"], "r2_vs_arm4_median": r["r2_vs_arm4"]["all"]["median"]})
        if n == "projection" and "rows" in r:
            keep["rows_26"] = [{k: x[k] for k in ("arm", "max_err_sd", "q999_err_sd", "frac_le_0.02sd", "exact_ok",
                                                   "sd_ok")} for x in r["rows"] if x["layer"] == L26]
        if n == "decode" and "rows" in r:
            keep["rows"] = r["rows"]
        out[n] = keep
    return out


def _reading_repeat(res: dict) -> list[str]:
    """The pre-written reading (docs/GATES.md) of a repeat-gate result."""
    out, a, b = [], res["part_a"], res["part_b"]
    if not res["checks"]["isolation_counters_exact"] or not res["checks"]["load_shas"]:
        out.append("REAL BUG: counters or load shas wrong")
    for k in STRONG_ARMS:
        r = a[str(k)]["repeat"]
        out.append(f"arm {k} isolation: " + ("cross-batch distance within its own repeat noise (artefact of a noisy "
                                             "strong arm)" if r["ok"] else "REAL BUG candidate: cross-batch distance "
                                             "exceeds the arm's own repeat noise (edit depends on batch neighbours)")
                   + (f"; cross/rep median ratio {r['cross_over_rep_median']:.2f} > 2: explanation only partial"
                      if r["ok"] and r["cross_over_rep_median"] > 2 else ""))
    w = res["gated_weak_arm"]
    if w is None:
        out.append("INCONCLUSIVE: no weak arm moved logprobs by 10x the floor; leaks at small sizes are untested")
    else:
        ra = a[str(w)]
        if ra["original_rule_ok"]:
            out.append(f"weak arm {w}: passes the original isolation rule (no leak at its size)")
        elif ra["repeat"]["ok"]:
            out.append(f"weak arm {w}: REAL BUG by the rule (fails the original isolation rule); its cross distance is "
                       "within its own repeat noise, so edit-amplified noise at this size is the alternative reading")
        else:
            out.append(f"weak arm {w}: REAL BUG (fails the original rule and exceeds its own repeat noise)")
    for k in sorted({4, w} - {None}):
        for r in a[str(k)]["sham_m1"]:
            if not r["ok"]:
                out.append(f"sham {r['id']} in the arm-{k} mixed batch misses its floor: " +
                           ("REAL LEAK (drift >= 0.1 of the arm's effect)" if r["leak_fraction"] >= LEAK_FRAC else
                            "noise-level miss (drift < 0.1 of the arm's effect); the gate fails as written; a human decision"))
    out.append("reload: " + ("slot 8 vs slot 4 within arm 4's own repeat noise" if b["strong"]["ok"] else
                             "REAL BUG candidate: slot 8 vs slot 4 exceeds arm 4's own repeat noise"))
    if w is not None:
        wb = b["weak"][str(w)]
        out.append(f"weak reload (arm {w} -> slot {wb['reloaded_slot']}): " +
                   ("passes the original reload rule" if wb["original_rule_ok"] and wb["differs_from_sham"] else
                    "REAL BUG: fails the original reload rule"))
    return out


# --- projection outliers -------------------------------------------------------------------------------------------

def _groups(rows: list[int], limit: int) -> list[list[int]]:
    """Consecutive prompt indices grouped so each group's rows fit ``limit`` (a longer prompt goes alone)."""
    out, cur, n = [], [], 0
    for j, r in enumerate(rows):
        if cur and n + r > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(j)
        n += r
    return out + ([cur] if cur else [])


def _last_index(req: np.ndarray, pos: np.ndarray) -> dict[int, int]:
    """``(req, pos)`` key -> row of its last occurrence (a recomputed request gets fresh rows; keep the last)."""
    keys = req.astype(np.int64) * (1 << 32) + pos.astype(np.int64)
    return {int(k): i for i, k in enumerate(keys)}


def read_dump_norms(path: str, layers: list[int], chunk: int = 4096) -> tuple[dict, dict, dict[int, np.ndarray]]:
    """A capture dump without materializing ``vec``: every other tensor as numpy, plus per vector row the norm of the
    pre-edit stream mean ``|mean_s h_s|`` at each of ``layers`` that was captured. Raises like ``load_dump`` when
    ``coef_pos_dev`` differs from ``coef_pos``."""
    from safetensors import safe_open

    meta = json.loads(Path(path + ".json").read_text())
    norms: dict[int, np.ndarray] = {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        arr = {k: f.get_tensor(k).numpy() for k in keys if k != "vec"}
        if "vec" in keys:
            sl = f.get_slice("vec")
            m = int(sl.get_shape()[0])
            for L in layers:
                if L not in meta.get("capture_layers", []):
                    continue
                j = meta["capture_layers"].index(L)
                parts = [sl[r: r + chunk, j: j + 1, :].float().norm(dim=-1).reshape(-1).numpy()
                         for r in range(0, m, chunk)]
                norms[L] = np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)
    if "coef_pos_dev" in arr and not np.array_equal(arr["coef_pos_dev"], arr["coef_pos"]):
        bad = int((arr["coef_pos_dev"] != arr["coef_pos"]).sum())
        raise ValueError(f"{path}: {bad} coef rows have a device position that differs from the host row map")
    return arr, meta, norms


def outlier_stats(err_sd: np.ndarray, sd: np.ndarray, pos: np.ndarray, prompt: list[str],
                  hnorm: np.ndarray | None = None, top: int = 20) -> dict:
    """Per-token coefficient errors of one (arm, slot): ``err_sd [n, 4]`` (per stream, in that stream's sd), ``sd [4]``,
    positions, prompt ids, and ``hnorm [n]`` (pre-edit |mean_s h_s|; NaN where no vector row matched; None where the
    slot has no vector capture). The rule (``rule_ok``, only with ``hnorm``): no (token, stream) above 0.05 sd whose
    bf16 bound ``2^-8 |h| / sd_s`` is <= 0.05 sd, every token has an ``hnorm``, and >= 99.9% of tokens within 0.02 sd."""
    err_sd, sd, pos = np.asarray(err_sd, np.float64), np.asarray(sd, np.float64), np.asarray(pos)
    n = int(len(err_sd))
    worst, wstream = err_sd.max(1), err_sd.argmax(1)
    flagged = worst > SD_MAX
    out = {"n": n, "max_err_sd": float(worst.max()) if n else 0.0,
           "q999_err_sd": float(np.quantile(worst, 0.999)) if n else 0.0,
           "frac_le_0.02sd": float((worst <= SD_FRAC).mean()) if n else 1.0, "n_gt_0.05sd": int(flagged.sum()),
           "frac_gt_0.05sd": float(flagged.mean()) if n else 0.0,
           "old_rule_ok": bool(n and worst.max() <= SD_MAX and (worst <= SD_FRAC).mean() >= FRAC_MIN)}
    fp = pos[flagged]
    out["flagged_positions"] = {"n_pos_lt_4": int((fp < 4).sum()), "n_pos_lt_64": int((fp < 64).sum()),
                                "median": float(np.median(fp)) if fp.size else None,
                                "distinct_prompts": len({prompt[i] for i in np.flatnonzero(flagged)})}
    bound = None
    if hnorm is not None:
        hnorm = np.asarray(hnorm, np.float64)
        have = np.isfinite(hnorm)
        bound = BF16_REL * hnorm[:, None] / sd[None, :]  # [n, 4], NaN where no vector row
        unexplained = (err_sd > SD_MAX) & ~(bound > SD_MAX)  # a missing norm cannot explain an error
        hs = np.sort(hnorm[have])
        thr = float(np.quantile(hs, 0.999)) if hs.size else np.inf
        in_top = have & (hnorm >= thr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(err_sd > 0, err_sd / bound, 0.0)
        out.update({"norm_coverage": float(have.mean()) if n else 1.0, "hnorm_median": float(np.median(hs)) if hs.size
                    else None, "hnorm_q999": thr if hs.size else None, "hnorm_max": float(hs[-1]) if hs.size else None,
                    "n_unexplained": int(unexplained.sum()),
                    "n_tokens_unexplained": int(unexplained.any(1).sum()),
                    "n_flagged_in_top_0.1pct_norm": int((flagged & in_top).sum()),
                    "all_flagged_in_top_0.1pct_norm": bool(np.all(in_top[flagged])) if flagged.any() else None,
                    "max_err_over_bound": float(np.nanmax(ratio)) if n else 0.0,
                    "n_tokens_bound_gt_0.05sd": int((bound > SD_MAX).any(1).sum())})
        out["rule_ok"] = bool(out["n_unexplained"] == 0 and have.all() and out["frac_le_0.02sd"] >= FRAC_MIN)
    order = np.argsort(-worst, kind="stable")[:top]
    tops = []
    for i in order:
        e = {"prompt": prompt[i], "pos": int(pos[i]), "worst_stream": int(wstream[i]),
             "err_sd": [float(x) for x in err_sd[i]], "flagged": bool(flagged[i])}
        if hnorm is not None:
            h = float(hnorm[i])
            e.update({"hnorm": h if np.isfinite(h) else None,
                      "hnorm_pct": float(np.searchsorted(hs, h, side="right") / max(hs.size, 1)) if np.isfinite(h)
                      else None,
                      "bound_sd": [float(x) for x in bound[i]],
                      "explained": bool(not np.any((err_sd[i] > SD_MAX) & ~(bound[i] > SD_MAX)))})
        tops.append(e)
    out["top"] = tops
    return out


def gate_projection_outliers(client, out_dir: str) -> dict:
    """Where the projection errors above 0.05 sd sit (eager boot: max 0.71/0.68/0.45 sd at output 26 for arms
    3/5/6, while >= 99.9% stayed within 0.02 sd). Arms 3, 5, 6 on the 20 generic prompts at concurrency 1, as
    Projection exact, but with vector capture (which also writes the coefficient rows) from position 1, dumped in
    groups of prompts that fit the vector ring. Per token: the coefficient error after the edit per stream in sd (as
    Projection exact), its position, and |h| = the norm of the pre-edit stream mean at that slot (the only norm the
    engine captures; per-stream norms are not captured). Per (arm, slot): top 20 tokens, fraction above 0.05 sd,
    whether every such token is in the top 0.1% of |h|; |h| only at slots with vector capture
    (12, 15, 18, 22, 25, 26, 27, 30, 34), positions everywhere.

    Pass rule (fixed 28 Oct ~11:30 CET, before any run), at output 26 for arms 3, 5, 6 (the scope of the old sd rule):
    the exact part of Projection exact (edited > 0, cast violations 0, non-finite 0 in stats and capture, stats max_err
    = capture max_err, rows aligned); every coefficient row has a vector row (no ring overflow); errors above 0.05 sd
    only on (token, stream) cells whose bf16 bound ``2^-8 * |h| / sd_s`` exceeds 0.05 sd; and >= 99.9% of tokens
    within 0.02 sd. Other slots are recorded. A NaN retry at concurrency 1 fails it (``write``). The old rule's result
    on this capture is ``old_rule_ok`` per row; the old gate's own result is in ``old``.
    Reading: an error above the bf16 bound of its token = a real bug candidate; errors only on tokens whose bound
    exceeds 0.05 sd = the 0.05 sd maximum is below bf16 resolution there (gate artefact)."""
    from safetensors.numpy import load_file

    t_start = time.time()
    info = arms_info(out_dir)
    t = load_file(info["gate_path"])
    ep = np.einsum("aisd,aisd->ais", t["p"].astype(np.float64), t["e"].astype(np.float64))
    layers = info["layers"]
    generic = [p for p in floor_prompts() if p["kind"] == "generic"]
    gp = generic[: int(os.environ.get("DSV41_OUTLIER_PROMPTS", len(generic)))]  # the mock run uses fewer
    inst = client.installed()
    vec_rows = int(inst.get("vec_rows", 0))
    cap_layers = list(inst.get("capture_layers", []))
    if vec_rows <= 0 or L26 not in cap_layers:
        return write(out_dir, "outliers", {"pass": False, "error": "vector capture at output 26 not available"})
    limit = int(os.environ.get("DSV41_OUTLIER_GROUP_ROWS", int(0.9 * vec_rows)))
    groups = _groups([len(p["ids"]) - 1 for p in gp], limit)
    vec_layers = [L for L in layers if L in cap_layers]
    reset_counted(client)
    data: dict[int, dict] = {}
    dumps, run = [], int(time.time())
    for arm in (3, 5, 6):
        parts = []
        for gi, g in enumerate(groups):
            for j in g:  # concurrency 1, as Projection exact (decided 06:30 CET for the batched-NaN problem)
                client.score([gp[j]["ids"]], request_fields(arm, arms_sha=info["gate_sha"], capture="vec", cap_from=1,
                                                            tag=f"o{arm}_{j:02d}"), concurrency=1)
            dump = client.dump(str(Path(info["capture_dir"]) / f"vec_outliers_{run}_a{arm}_g{gi}.safetensors"))
            arr, meta, norms = read_dump_norms(dump["path"], vec_layers)
            dumps.append({"path": dump["path"], "coef_rows": dump.get("coef_rows"), "vec_rows": dump.get("vec_rows"),
                          "coef_overflow": meta.get("coef_overflow", 0), "vec_overflow": meta.get("vec_overflow", 0)})
            sel = np.flatnonzero((arr["coef_arm"] == arm) & (arr["coef_pos"] >= 1))
            last_c = _last_index(arr["coef_req"][sel], arr["coef_pos"][sel])
            rows = sel[sorted(last_c.values())]
            vlast = _last_index(arr["vec_req"], arr["vec_pos"]) if "vec_req" in arr else {}
            keys = arr["coef_req"][rows].astype(np.int64) * (1 << 32) + arr["coef_pos"][rows].astype(np.int64)
            vidx = np.array([vlast.get(int(k), -1) for k in keys], np.int64)
            tags = meta["tags"]
            parts.append({"coef": arr["coef"][rows].astype(np.float64), "pos": arr["coef_pos"][rows],
                          "prompt": [gp[int(tags[i].split("_")[1])]["id"] for i in arr["coef_tag"][rows]],
                          "hnorm": {L: (np.where(vidx >= 0, v[np.maximum(vidx, 0)], np.nan) if v.size
                                        else np.full(len(rows), np.nan)) for L, v in norms.items()},
                          "n_dup_rows": int(len(sel) - len(rows))})
            Path(dump["path"]).unlink(missing_ok=True)  # vector dumps are ~1-3 GB; the sidecar JSON stays
        data[arm] = {"coef": np.concatenate([p["coef"] for p in parts]),
                     "pos": np.concatenate([p["pos"] for p in parts]),
                     "prompt": [x for p in parts for x in p["prompt"]],
                     "hnorm": {L: np.concatenate([p["hnorm"][L] for p in parts]) for L in parts[0]["hnorm"]},
                     "n_dup_rows": sum(p["n_dup_rows"] for p in parts)}
    st = dev(client.stats())
    rows_out, ok, npz = [], True, {}
    for arm in (3, 5, 6):
        x = data[arm]
        worst_all = []
        for i, L in enumerate(layers):
            a, g = float(t["a"][arm, i]), float(t["g"][arm, i])
            if a == 0 and g == 0:
                continue
            c, cp, mu = x["coef"][:, i, 0], x["coef"][:, i, 1], t["mu"][arm, i].astype(np.float64)
            delta = (c.mean(1, keepdims=True) - mu.mean()) if t["mc"][arm] else c - mu
            target = c - a * delta * ep[arm, i] + g * ep[arm, i]
            sd = np.asarray(info["sd"]["d26"][str(L)], np.float64)
            err = np.abs(cp - target)
            row = {"arm": arm, "layer": L, "edited": int(st["tokens_edited"][arm][i]),
                   "cast_violations": int(st["cast_violations"][arm][i]),
                   "nonfinite_stats": int(st["nonfinite"][arm][i]),
                   "nonfinite_capture": int((~np.isfinite(x["coef"][:, i, :2])).sum()),
                   "max_err_stats": float(st["max_err"][arm][i]), "max_err_capture": float(err.max()),
                   **outlier_stats(err / sd, sd, x["pos"], x["prompt"], x["hnorm"].get(L))}
            row["exact_ok"] = bool(row["edited"] > 0 and row["cast_violations"] == 0 and row["nonfinite_stats"] == 0
                                   and row["nonfinite_capture"] == 0
                                   and abs(row["max_err_stats"] - row["max_err_capture"])
                                   <= 1e-3 * max(1.0, row["max_err_capture"]))
            row["gated"] = L == L26
            if row["gated"]:
                row["ok"] = bool(row["exact_ok"] and row.get("rule_ok", False))
                ok &= row["ok"]
                npz[f"a{arm}_err_sd_26"] = (err / sd).astype(np.float32)
                npz[f"a{arm}_hnorm_26"] = x["hnorm"].get(L, np.full(len(err), np.nan)).astype(np.float32)
                npz[f"a{arm}_pos"] = x["pos"].astype(np.int32)
            worst_all.append((err / sd).max(1).astype(np.float32))
            rows_out.append(row)
        if arm == 3:  # worst-stream error at every edited slot [n, 26], for position plots afterwards
            npz["a3_worst_err_sd_all_slots"] = np.stack(worst_all, 1)
            hn = [x["hnorm"][L] for L in vec_layers if L in x["hnorm"]]
            if hn:
                npz["a3_hnorm_vec_layers"] = np.stack(hn, 1).astype(np.float32)
    overflow = sum(d["coef_overflow"] + d["vec_overflow"] for d in dumps)
    ok &= overflow == 0
    np.savez_compressed(Path(out_dir, "outliers_tokens.npz"), **npz)
    return write(out_dir, "outliers", {"pass": bool(ok), "rows": rows_out, "dumps": dumps, "overflow": overflow,
                                       "n_prompts": len(gp), "groups": groups, "vec_layers": vec_layers,
                                       "dup_rows": {str(a): data[a]["n_dup_rows"] for a in data},
                                       "old": _old_results(out_dir, ("projection",)),
                                       "seconds": time.time() - t_start})


# --- decode effect -------------------------------------------------------------------------------------------------

def _gen_one_arm(client, case: dict, fields: dict, arm: int, s26: int) -> dict:
    """One path case generated under ``arm`` with the counter checks of the positive control (one arm only)."""
    reset_counted(client)
    gen, sent = _generate(client, case["reqs"], fields, arm)
    d = dev(client.stats())
    P, N = [len(r["ids"]) for r in case["reqs"]], [r["n"] for r in case["reqs"]]
    row = {"gen": gen, "seen": int(d["tokens_seen"][arm]), "edited_26": int(d["tokens_edited"][arm][s26]),
           "edited_all": int(d["tokens_edited"][arm].sum()), "expect_seen": int(sum(p + n - 1 for p, n in zip(P, N))),
           "expect_edited": int(sum(p + n - 2 for p, n in zip(P, N))), "path_ok": True}
    if case["path"] == "piecewise":  # the late prompt must join while the 16 others still decode
        late, decoding = sent[-1]["seen_at_send"], 16 * (SHORT + PW_DECODE - 1)
        row["late_seen_at_send"] = late
        row["path_ok"] = late is not None and case["reqs"][-1]["after_seen"] <= late < decoding
    row["counts_ok"] = (row["seen"] == row["expect_seen"] and row["edited_26"] == row["expect_edited"]
                        and row["edited_all"] == row["edited_26"])
    return row


def gate_decode_effect(client, floor: dict, out_dir: str) -> dict:
    """Decode vs prefill for an arm whose effect on its own decoded text is large. Decode matches prefill failed only
    its effect cap (``med4 <= 0.1 * effect``): on arm 4's own greedy text the effect was 0.006-0.19, so the cap was
    10% of a small number, while the noise bound passed everywhere.

    Medium arm, chosen before the measurement: candidates arms 11, 12, 13 (adds of 1, 2, 4 sd along d26 at 26); each
    generates 64 greedy tokens on 8 calibration prompts (``short_prompts(8, seed=1028)``, one batch); the candidate
    with the largest calibration effect (median |tf_arm - tf_sham| on its own generated tokens j >= 1; ties to the
    weaker arm) is the medium arm. It then runs every path case of the positive control (same prompts, same
    lengths); arm 0's generations are the positive control's (cached for this boot).

    Pass rule (fixed 28 Oct ~11:30 CET, before any run), in every path row, with the old decode gate's quantities for
    the medium arm m: ``eff_m >= 0.5`` (testable), ``med_m <= 1.5 * max(med0, med_tf_m) + 0.01`` (noise bound,
    unchanged), ``med_m <= 0.1 * eff_m`` (effect cap, unchanged), exact counters for arm m (edited at 26 =
    prompt + generated - 2, seen = prompt + generated - 1, nothing edited at other slots), and the PIECEWISE late
    prompt sent mid-decode. A path row with ``eff_m < 0.5`` makes the gate inconclusive (pass false). Everything is also
    reported relative to the pooled generic floor; the old decode rows are in ``old``.
    Reading: cap fails where eff_m >= 0.5 = the decode path loses part of the edit (real bug); noise bound fails = the
    decode path differs from prefill beyond noise (real bug); all pass = the old cap failure was a small-effect
    artefact."""
    t_start = time.time()
    info = arms_info(out_dir)
    ext = ext_arms(out_dir)
    s26 = info["slot26"]
    F = float(floor_reference(floor)["generic"]["median"])
    rows0 = _run_paths(client, info, out_dir)  # arm 0's (and arm 4's) generations of this boot (cached)
    fe = lambda k: request_fields(k, arms_sha=ext["sha"])  # noqa: E731
    res_rows, cal = [], {}
    try:
        client.load_arms(ext["path"])
        cal_prompts = short_prompts(8, seed=1028)
        P = [len(p) for p in cal_prompts]
        for k in MEDIUM_ARMS:
            gen, _ = _generate(client, [{"ids": p, "n": N_GEN} for p in cal_prompts], fe(k), k)
            seqs = [p + g["ids"] for p, g in zip(cal_prompts, gen)]
            ta, tz = _gen_lp(P, gen, client.score(seqs, fe(k))), _gen_lp(P, gen, client.score(seqs, fe(0)))
            cal[k] = float(np.median(np.abs(ta - tz)))
        m = max(MEDIUM_ARMS, key=lambda k: (cal[k], -k))
        for case, r0 in zip(_cases(), rows0):
            if [r["ids"] for r in case["reqs"]] != r0["ids"]:
                raise RuntimeError(f"path case {case['path']} c={case['c']} differs from the cached generations")
            g = _gen_one_arm(client, case, fe(m), m, s26)
            Pc = r0["P"]
            seqm = [p + x["ids"] for p, x in zip(r0["ids"], g["gen"])]
            seq0 = [p + x["ids"] for p, x in zip(r0["ids"], r0["gen0"])]
            decm = np.concatenate([np.asarray(x["lp"][1:]) for x in g["gen"]])
            dec0 = np.concatenate([np.asarray(x["lp"][1:]) for x in r0["gen0"]])
            tf0 = _gen_lp(Pc, r0["gen0"], client.score(seq0, fe(0)))
            tfa = _gen_lp(Pc, g["gen"], client.score(seqm, fe(m)))
            tfb = _gen_lp(Pc, g["gen"], client.score(seqm, fe(m)))
            tfz = _gen_lp(Pc, g["gen"], client.score(seqm, fe(0)))
            med_m, med0 = float(np.median(np.abs(decm - tfa))), float(np.median(np.abs(dec0 - tf0)))
            med_tf, eff = float(np.median(np.abs(tfa - tfb))), float(np.median(np.abs(tfa - tfz)))
            bound = 1.5 * max(med0, med_tf) + 0.01
            row = {"path": case["path"], "c": case["c"], "arm": m, "median_arm": med_m, "median_arm0": med0,
                   "median_tf_vs_tf": med_tf, "effect_tf_vs_sham": eff, "noise_bound": bound,
                   "testable": eff >= EFF_MIN, "noise_ok": med_m <= bound, "cap_ok": med_m <= 0.1 * eff,
                   "counts_ok": g["counts_ok"], "path_ok": g["path_ok"],
                   **{k: g[k] for k in ("seen", "expect_seen", "edited_26", "expect_edited", "edited_all")},
                   "median_arm_over_floor": med_m / max(F, 1e-12), "effect_over_floor": eff / max(F, 1e-12),
                   "median_arm_over_effect": med_m / max(eff, 1e-12)}
            row["ok"] = bool(row["testable"] and row["noise_ok"] and row["cap_ok"] and row["counts_ok"]
                             and row["path_ok"])
            res_rows.append(row)
    finally:
        client.load_arms(info["gate_path"])  # later gates expect the gate arms
    inconclusive = [f"{r['path']} c={r['c']}" for r in res_rows if not r["testable"]]
    real = [f"{r['path']} c={r['c']}" for r in res_rows if r["testable"] and not (r["noise_ok"] and r["cap_ok"])]
    reading = ([f"REAL BUG candidate in {real}: decode differs from prefill beyond noise or loses part of the edit"]
               if real else []) + ([f"INCONCLUSIVE in {inconclusive}: effect on the decoded text below {EFF_MIN}"]
                                   if inconclusive else [])
    if not real and not inconclusive and all(r["counts_ok"] and r["path_ok"] for r in res_rows):
        reading = ["the decode path carries the edit as prefill does in every path (old cap failure = small-effect "
                   "artefact)"]
    return write(out_dir, "decode_effect", {"pass": all(r["ok"] for r in res_rows), "medium_arm": m,
                                            "medium_add_sd": EXT_ADD_SD[m], "calibration_effect": cal,
                                            "floor_generic_median": F, "rows": res_rows, "inconclusive": inconclusive,
                                            "reading": reading, "arms_ext": ext,
                                            "old": _old_results(out_dir, ("decode",)),
                                            "seconds": time.time() - t_start})


# ----------------------------------------------------------------------------------------------------------------------
# proxy
# ----------------------------------------------------------------------------------------------------------------------

def gate_proxy(client, proxy_url: str, out_dir: str) -> dict:
    """Control routes 404, key required, salts must match a loaded arms sha of this boot, and a global-arm block
    passes only its own arm (503 otherwise). Start the proxy after the counted gates (see the module docstring)."""
    keyed = EngineClient(proxy_url)
    anon = EngineClient(proxy_url, key_file="none")
    sha = arms_info(out_dir)["gate_sha"] if Path(out_dir, "arms.json").exists() else "x"
    tiny = {"model": "dsv41", "prompt": [0, 5, 223], "max_tokens": 2, "temperature": 0.0}
    probes = {
        "GET /collective_rpc": ("GET", "/collective_rpc", None, 404),
        "POST /collective_rpc": ("POST", "/collective_rpc", {"method": "dsv41_installed"}, 404),
        "POST /reset_prefix_cache": ("POST", "/reset_prefix_cache", None, 404),
        "GET /metrics": ("GET", "/metrics", None, 404),
        "GET /v1/models": ("GET", "/v1/models", None, 200),
        "GET /health": ("GET", "/health", None, 200),
        "completion": ("POST", "/v1/completions", tiny, 200),
        "arm 1 without salt": ("POST", "/v1/completions", {**tiny, "vllm_xargs": {"arm": 1}}, 400),
        "arm 1 with salt": ("POST", "/v1/completions", {**tiny, **request_fields(1, arms_sha=sha)}, 200),
        "arm 1 window 5.. with its salt": ("POST", "/v1/completions",
                                           {**tiny, **request_fields(1, 5, arms_sha=sha)}, 200),
        "arm 1 with a wrong salt": ("POST", "/v1/completions",
                                    {**tiny, "vllm_xargs": {"arm": 1}, "cache_salt": "0" * 24}, 400),
        "arm 1 with arm 2's salt": ("POST", "/v1/completions",
                                    {**tiny, "vllm_xargs": {"arm": 1}, **{"cache_salt": request_fields(
                                        2, arms_sha=sha)["cache_salt"]}}, 400),
        "arm 1 salted with an unloaded sha": ("POST", "/v1/completions",
                                              {**tiny, **request_fields(1, arms_sha="f" * 64)}, 400),
        "capture": ("POST", "/v1/completions", {**tiny, **request_fields(0, arms_sha=sha, capture="coef")}, 400),
    }
    res, ok = {}, True

    def probe(name, m, path, body, want):
        nonlocal ok
        status, _ = keyed.request(m, path, body, ok=())
        res[name] = {"status": status, "want": want, "ok": status == want}
        ok &= status == want

    for name, args in probes.items():
        probe(name, *args)
    status, _ = anon.request("POST", "/v1/completions", tiny, ok=())
    res["completion without key"] = {"status": status, "want": 401, "ok": status == 401}
    ok &= status == 401
    # global-arm block: only arm 4's own traffic passes (no counted traffic runs now)
    try:
        client.set_global_arm(4)
        probe("sham during global arm 4", "POST", "/v1/completions", tiny, 503)
        probe("arm 4 during global arm 4", "POST", "/v1/completions", {**tiny, **request_fields(4, arms_sha=sha)},
              200)
    finally:
        client.set_global_arm(-1)
    probe("sham after the block", "POST", "/v1/completions", tiny, 200)
    return write(out_dir, "proxy", {"pass": bool(ok), "url": proxy_url, "probes": res})


def latest_floor() -> str:
    found = sorted((LOG_DIR / "engine").glob("floor_native_*.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no floor_native_*.json in {LOG_DIR / 'engine'}; run tools/client.py floor first")
    return str(found[-1])


def main() -> None:
    """``--only NAME[,NAME]``, ``--floor PATH``, ``--out DIR``."""
    ap = argparse.ArgumentParser(description="engine gate suite (docs/GATES.md)")
    ap.add_argument("--only", default=",".join(g for g in GATES if g != "proxy"),
                    help=f"comma list from {GATES} (default: all but proxy)")
    ap.add_argument("--floor", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--url", default=None, help="engine (default $DSV41_ENGINE_URL or 127.0.0.1:8101)")
    ap.add_argument("--proxy-url", default="http://127.0.0.1:8100")
    ap.add_argument("--alpha-sd", type=float, default=8.0, help="add size in sd (default 8, at most 32)")
    ap.add_argument("--keep-going", action="store_true", help="run later gates after a failure")
    ap.add_argument("--engine-log", default=None, help="engine serve log (required by the install check)")
    ap.add_argument("--edit-layers", default=os.environ.get("DSV41_EDIT_LAYERS", DEFAULT_EDIT_LAYERS),
                    help="installed outputs the install check expects (the engine's DSV41_EDIT_LAYERS)")
    ap.add_argument("--mhc-overlap", default=os.environ.get("DSV41_MHC_OVERLAP", "1"), choices=("0", "1"),
                    help="the engine's DSV41_MHC_OVERLAP (0 = side stream patched off)")
    args = ap.parse_args()
    only = [g for g in args.only.split(",") if g]
    bad = set(only) - set(GATES)
    if bad:
        raise SystemExit(f"unknown gates {sorted(bad)}")
    missing = [v for v, p in (("DSV41_ARMS_DIR", ARMS_DIR), ("DSV41_CAPTURE_DIR", CAPTURE_DIR), ("DSV41_LOG_DIR", LOG_DIR),
                              ("DSV41_XCHECK_DIR", XCHECK_DIR)) if p is None]
    if missing:
        raise SystemExit(f"set DSV41_ROOT (or {', '.join(missing)}); see .env.example")
    for name in ("floor_prompts.json", "plumbing_texts.json"):
        if not (DATA / name).exists():
            raise SystemExit(f"{DATA / name} missing: build it with tools/build_data.py (or set DSV41_DATA_DIR)")
    out = args.out or str(LOG_DIR / "engine" / f"gates_{time.strftime('%m%d_%H%M')}")
    Path(out).mkdir(parents=True, exist_ok=True)
    floor_path = args.floor or latest_floor()
    floor = json.loads(Path(floor_path).read_text())
    check_floor_prompts(floor)
    global _CLIENT
    client = _CLIENT = EngineClient(args.url)
    summary = {"out": out, "floor": floor_path, "results": {}}
    if "build" not in only and Path(out, "arms.json").exists():
        summary["gate_arms"] = ensure_gate_arms(client, out)
    run = {"install": lambda: gate_install(client, out, args.engine_log, args.edit_layers, args.mhc_overlap == "1"),
           "build": lambda: build_gate_arms(client, out, args.alpha_sd),
           "sham": lambda: gate_sham_native(client, floor, out),
           "positive": lambda: gate_positive_control(client, floor, out),
           "decode": lambda: gate_decode_prefill(client, floor, out),
           "isolation": lambda: gate_arm_isolation(client, floor, out),
           "prefix": lambda: gate_prefix_cache(client, floor, out),
           "reload": lambda: gate_arm_reload(client, floor, out),
           "projection": lambda: gate_projection_exact(client, out),
           "info": lambda: gate_info(client, out),
           "repeat": lambda: gate_repeat_floor(client, floor, out),
           "outliers": lambda: gate_projection_outliers(client, out),
           "decode_effect": lambda: gate_decode_effect(client, floor, out),
           "proxy": lambda: gate_proxy(client, args.proxy_url, out)}
    for g in [g for g in GATES if g in only]:
        try:
            r = run[g]()
            summary["results"][g] = bool(r.get("pass"))
        except Exception as e:  # a crash is a failure with a record
            write(out, g, {"pass": False, "error": f"{type(e).__name__}: {e}"})
            summary["results"][g] = False
        Path(out, "summary.json").write_text(json.dumps(summary, indent=1))
        if not summary["results"][g] and not args.keep_going:
            break
    print(json.dumps(summary, indent=1))
    sys.exit(0 if all(summary["results"].values()) else 1)


if __name__ == "__main__":
    main()
