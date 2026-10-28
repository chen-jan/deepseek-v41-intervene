"""CPU tests for the retry gates of tools/gates.py: repeat rule, projection outlier rule, dump reader,
extended arms file, grouping, and the pre-written reading. numpy + safetensors (+ torch for the dump reader); no
engine. Runs under pytest or as ``python tests/test_gates_retry.py``. The HTTP plumbing of the three gates is
covered by ``tests/run_mock_gates.sh``.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from tools import arms, gates  # noqa: E402


def test_repeat_rule():
    rng = np.random.default_rng(0)
    rep = np.abs(rng.normal(0.4, 0.1, 16))
    same = np.abs(rng.normal(0.4, 0.1, 32))
    r = gates.repeat_rule(rep, same, 0.013)
    assert r["ok"] and r["n_rep"] == 16 and r["n_cross"] == 32
    assert abs(r["limit"] - (np.quantile(rep, 0.95) + 0.013)) < 1e-12
    assert not gates.repeat_rule(rep, same * 3, 0.013)["ok"]  # composition-dependent: far above own repeat noise
    # deterministic repeats (rep all 0): only the sham floor is left, so a 0.4 cross distance fails
    assert not gates.repeat_rule(np.zeros(16), same, 0.013)["ok"]
    assert gates.repeat_rule(np.zeros(16), np.full(32, 0.013), 0.013)["ok"]  # boundary is inclusive


def _tokens(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    sd = np.array([22.0, 7.0, 28.0, 4.5])
    hnorm = rng.uniform(80, 200, n)  # bf16 bound 2^-8 |h| / sd: 0.07-0.18 sd at stream 3, < 0.04 sd at stream 2
    err = np.abs(rng.normal(0, 0.002, (n, 4)))
    pos = np.tile(np.arange(1, 1001), n // 1000 + 1)[:n]
    prompt = [f"p{i // 1000:02d}" for i in range(n)]
    return err, sd, pos, prompt, hnorm


def test_outlier_rule_explained_and_not():
    err, sd, pos, prompt, hnorm = _tokens()
    # huge-norm tokens near the start with large errors in the low-variance stream: explained by bf16
    big = np.array([5, 1005, 2010])
    hnorm[big] = 3000.0  # bound at stream 3 = 3000/256/4.5 = 2.6 sd
    err[big, 3] = [0.7, 0.4, 0.2]
    s = gates.outlier_stats(err, sd, pos, prompt, hnorm)
    assert s["rule_ok"] and s["n_gt_0.05sd"] == 3 and s["n_unexplained"] == 0 and not s["old_rule_ok"]
    assert s["all_flagged_in_top_0.1pct_norm"] is True and s["n_flagged_in_top_0.1pct_norm"] == 3
    assert s["top"][0]["pos"] == pos[5] and s["top"][0]["explained"] and abs(s["max_err_sd"] - 0.7) < 1e-12
    assert len(s["top"]) == 20 and s["flagged_positions"]["distinct_prompts"] == 3
    # an error above 0.05 sd on a normal-norm token, in a stream where bf16 cannot produce it: unexplained
    err[7, 2] = 0.08  # bound at stream 2 = 150/256/28 = 0.02 sd
    s2 = gates.outlier_stats(err, sd, pos, prompt, hnorm)
    assert not s2["rule_ok"] and s2["n_unexplained"] == 1 and s2["all_flagged_in_top_0.1pct_norm"] is False
    # a missing norm cannot explain an error
    err2, hn2 = err.copy(), hnorm.copy()
    err2[7, 2] = 0.0
    hn2[5] = np.nan
    s3 = gates.outlier_stats(err2, sd, pos, prompt, hn2)
    assert not s3["rule_ok"] and s3["n_unexplained"] == 1 and s3["norm_coverage"] < 1
    # the 99.9%-within-0.02 sd part is kept: 0.2% of tokens at 0.03 sd fails it even with every error explained
    err3 = err2.copy()
    err3[100:140, 0] = 0.03
    s4 = gates.outlier_stats(err3, sd, pos, prompt, hnorm)
    assert s4["n_unexplained"] == 0 and s4["frac_le_0.02sd"] < 0.999 and not s4["rule_ok"]
    # no vector capture at the slot: positions only, no rule
    s5 = gates.outlier_stats(err3, sd, pos, prompt, None)
    assert "rule_ok" not in s5 and "hnorm" not in s5["top"][0]


def test_groups_and_last_index():
    assert gates._groups([5, 5, 5], 10) == [[0, 1], [2]]
    assert gates._groups([20, 3], 10) == [[0], [1]]
    assert gates._groups([519, 639, 759], 1500) == [[0, 1], [2]]
    import subprocess

    with tempfile.TemporaryDirectory() as d:  # synthetic floor prompts (same lengths as real ones)
        subprocess.run([sys.executable, str(ENGINE / "tools" / "build_data.py"), "--synthetic", "--out", d], check=True,
                       capture_output=True)
        old, gates.DATA = gates.DATA, Path(d)
        try:
            rows = [len(p["ids"]) - 1 for p in gates.floor_prompts() if p["kind"] == "generic"]
        finally:
            gates.DATA = old
    g = gates._groups(rows, int(0.9 * 32768))
    assert sum(len(x) for x in g) == 20 and all(sum(rows[j] for j in x) <= int(0.9 * 32768) for x in g)
    idx = gates._last_index(np.array([0, 0, 1, 0]), np.array([3, 4, 3, 3]))
    assert idx[(0 << 32) + 3] == 3 and idx[(1 << 32) + 3] == 2 and len(idx) == 3


def test_read_dump_norms():
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        m, n, I, cap = 5000, 5000, 3, [12, 26, 30]
        rng = np.random.default_rng(2)
        vec = torch.from_numpy(rng.standard_normal((m, len(cap), 5120)).astype(np.float32)).to(torch.bfloat16)
        pos = np.arange(n, dtype=np.int32)
        P = lambda: torch.from_numpy(pos.copy())  # noqa: E731 (safetensors refuses shared storage)
        t = {"coef": torch.zeros((n, I, 3, 4)), "coef_pos": P(), "coef_pos_dev": P(),
             "coef_arm": torch.zeros(n, dtype=torch.int32), "coef_req": torch.zeros(n, dtype=torch.int32),
             "coef_tag": torch.zeros(n, dtype=torch.int32), "vec": vec, "vec_pos": P(),
             "vec_arm": torch.zeros(m, dtype=torch.int32), "vec_req": torch.zeros(m, dtype=torch.int32),
             "vec_tag": torch.zeros(m, dtype=torch.int32)}
        path = str(Path(tmp, "d.safetensors"))
        save_file(t, path)
        Path(path + ".json").write_text(json.dumps({"capture_layers": cap, "tags": ["x"], "req_ids": ["r"]}))
        arr, meta, norms = gates.read_dump_norms(path, [26, 30, 31], chunk=2000)  # 3 chunks, 31 not captured
        assert set(norms) == {26, 30} and "vec" not in arr and arr["coef"].shape == (n, I, 3, 4)
        want = vec[:, 1].float().norm(dim=-1).numpy()
        assert np.allclose(norms[26], want, rtol=1e-6)
        bad = dict(t, coef_pos_dev=torch.from_numpy(pos + 1))
        Path(path).unlink()
        save_file(bad, path)
        try:
            gates.read_dump_norms(path, [26])
        except ValueError:
            pass
        else:
            raise AssertionError("misaligned capture rows must raise")


def _gate_folder(tmp: Path):
    """A gates folder with a 16-arm gate file shaped like build_gate_arms' (arms 1-9 set, 10-15 empty)."""
    layers = list(range(12, 13)) + list(range(14, 39))
    I, s26 = len(layers), layers.index(26)
    rng = np.random.default_rng(3)
    d, c = rng.standard_normal(5120), rng.standard_normal(5120)
    t = arms.empty_tables(I, 16)
    for i in range(I):
        arms.set_slot(t, 1, i, "measure", d=d, mu=0.5)
        arms.set_slot(t, 3, i, "project", d=d, mu=0.5)
    arms.set_slot(t, 2, s26, "project", d=d, mu=0.5)
    arms.set_slot(t, 4, s26, "add", d=d, alpha=8 * 15.0)
    arms.set_slot(t, 5, s26, "shift", d=d, c=c, mu=0.5)
    arms.set_slot(t, 6, s26, "project_mc", d=d, mu=0.5)
    arms.set_slot(t, 8, s26, "add", d=c, alpha=8 * 2.0)
    labels = {k: f"arm{k}" for k in range(1, 10)}
    sha = arms.write_arms_file(tmp / "arms" / "gate_arms", layers, labels=labels, recipe={}, **t)
    info = {"layers": layers, "slot26": s26, "gate_path": str(tmp / "arms" / "gate_arms.safetensors"),
            "gate_sha": sha, "sd": {"d26": {"26": [10.0, 20.0, 15.0, 15.0]}}}
    out = tmp / "gates_x"
    out.mkdir()
    (out / "arms.json").write_text(json.dumps(info))
    return out, t, s26


def test_ext_arms():
    from safetensors.numpy import load_file

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out, t, s26 = _gate_folder(tmp)
        old = gates.ARMS_DIR
        gates.ARMS_DIR = tmp / "arms_ext"
        try:
            ext = gates.ext_arms(str(out))
            assert gates.ext_arms(str(out)) == ext  # written once per folder and gate sha
        finally:
            gates.ARMS_DIR = old
        e, r = load_file(ext["path"]), load_file(ext["reload_path"])
        for k in range(10):  # arms 0-9 unchanged
            assert all(np.array_equal(e[n][k], t[n][k]) for n in ("p", "e", "mu", "a", "g", "mc"))
        for k, ksd in gates.EXT_ADD_SD.items():
            assert np.array_equal(e["p"][k, s26], t["p"][4, s26]) and np.array_equal(e["e"][k, s26], t["e"][4, s26])
            assert np.isclose(e["g"][k, s26], ksd * 15.0) and e["a"][k].sum() == 0 and e["mu"][k].sum() == 0
            assert np.count_nonzero(e["g"][k]) == 1  # output 26 only
        assert not np.any(e["g"][14:]) and not np.any(e["p"][14:])
        for s, k in gates.EXT_RELOAD.items():
            assert all(np.array_equal(r[n][s], e[n][k]) for n in ("p", "e", "mu", "a", "g", "mc"))
        assert ext["sha"] != ext["reload_sha"] and ext["gate_sha"] == json.loads((out / "arms.json").read_text())[
            "gate_sha"]
        assert np.isclose(ext["alpha"]["10"], 7.5)


def _res(strong_ok=True, weak=10, weak_ok=True, sham_frac=None, reload_ok=True, ratio=1.0):
    rr = {"ok": strong_ok, "cross_over_rep_median": ratio}
    sham = [{"id": "generic_code_04", "ok": sham_frac is None, "leak_fraction": sham_frac or 0.0}]
    a = {str(k): {"repeat": rr, "original_rule_ok": weak_ok, "sham_m1": sham} for k in (4, 8, 10, 11)}
    b = {"strong": {"ok": reload_ok}, "weak": {str(k): {"reloaded_slot": k + 4, "original_rule_ok": weak_ok,
                                                        "differs_from_sham": True} for k in (10, 11)}}
    return {"part_a": a, "part_b": b, "gated_weak_arm": weak,
            "checks": {"isolation_counters_exact": True, "load_shas": True}}


def test_reading_repeat():
    good = gates._reading_repeat(_res())
    assert not any("REAL" in x or "INCONCLUSIVE" in x for x in good), good
    assert any("REAL BUG candidate" in x for x in gates._reading_repeat(_res(strong_ok=False)))
    assert any("explanation only partial" in x for x in gates._reading_repeat(_res(ratio=3.0)))
    assert any("INCONCLUSIVE" in x for x in gates._reading_repeat(_res(weak=None)))
    assert any("REAL LEAK" in x for x in gates._reading_repeat(_res(sham_frac=0.3)))
    assert any("noise-level miss" in x for x in gates._reading_repeat(_res(sham_frac=0.002)))
    assert any("REAL BUG" in x and "weak" in x for x in gates._reading_repeat(_res(weak_ok=False)))


def test_old_gates_unchanged_in_order():
    old = ("install", "build", "sham", "positive", "decode", "isolation", "prefix", "reload", "projection", "info")
    assert gates.GATES[:len(old)] == old and gates.GATES[-1] == "proxy"
    assert set(gates.GATES[len(old):-1]) == {"repeat", "outliers", "decode_effect"}


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
