"""CPU tests: request fields and salts, arms files, mu and controls, floor comparisons, the proxy.

Needs numpy, safetensors and aiohttp, no torch or vLLM. Runs under pytest or as
``python tests/test_tools.py`` (no pytest needed).
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from dsv41_intervene import MAX_EDIT_TO  # noqa: E402
from tools import arms, client, proxy  # noqa: E402

SHA = "ab" * 32


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def test_request_fields_sham_and_arm():
    assert arms.request_fields(arms_sha="") == {"vllm_xargs": {}}
    assert arms.request_fields(0, arms_sha=SHA) == {"vllm_xargs": {}}
    f = arms.request_fields(3, 10, 500, arms_sha=SHA)
    assert f["vllm_xargs"] == {"arm": 3, "edit_from": 10, "edit_to": 500}
    assert f["cache_salt"] == hashlib.sha256(f"{SHA}|3|10|500".encode()).hexdigest()[:24]
    assert arms.request_fields(3, arms_sha=SHA)["cache_salt"] == arms.salt(SHA, 3, 1, MAX_EDIT_TO)
    assert arms.request_fields(3, arms_sha=SHA) == arms.request_fields(3, arms_sha=SHA)  # deterministic
    assert arms.request_fields(3, arms_sha=SHA)["cache_salt"] != arms.request_fields(4, arms_sha=SHA)["cache_salt"]
    assert arms.request_fields(3, arms_sha=SHA)["cache_salt"] != arms.request_fields(3, arms_sha="cd" * 32)["cache_salt"]
    for bad in ({"arm": -1}, {"edit_from": 5, "edit_to": 5}, {"edit_to": MAX_EDIT_TO + 1}):
        raises(ValueError, arms.request_fields, arms_sha=SHA, **bad)
    raises(ValueError, arms.request_fields, 2, arms_sha="")


def test_request_fields_capture():
    a = arms.request_fields(1, arms_sha=SHA, capture="coef", cap_from=4, cap_to=99, tag="t")
    b = arms.request_fields(1, arms_sha=SHA, capture="coef", cap_from=4, cap_to=99, tag="t")
    assert a["vllm_xargs"] == {"arm": 1, "capture": "coef", "cap_from": 4, "cap_to": 99, "tag": "t"}
    assert a["cache_salt"] != b["cache_salt"]  # fresh nonce each time
    c = arms.request_fields(0, arms_sha=SHA, capture="vec", cap_pos=[9, 3, 3], nonce="n1")
    assert c["vllm_xargs"]["cap_pos"] == [3, 9]
    assert c["cache_salt"] == hashlib.sha256(f"{SHA}|0|1|{MAX_EDIT_TO}|cap|n1".encode()).hexdigest()[:24]
    raises(ValueError, arms.request_fields, arms_sha=SHA, capture="all")
    raises(ValueError, arms.request_fields, arms_sha=SHA, capture="vec", cap_pos=list(range(5000)))
    for v in a["vllm_xargs"].values():  # vllm_xargs value types (INTERFACE.md section 0 item 6)
        assert type(v) in (int, str, list)
    assert len(a["cache_salt"]) == 24 and all(ch in "0123456789abcdef" for ch in a["cache_salt"])


def _tables(I=3, A=4):
    t = arms.empty_tables(I, A)
    rng = np.random.default_rng(0)
    d, c = rng.standard_normal(5120), rng.standard_normal(5120)
    arms.set_slot(t, 1, 0, "project", d=d, mu=[1, 2, 3, 4], lam=0.5)
    arms.set_slot(t, 2, 1, "project_mc", d=d, mu=0.25)
    arms.set_slot(t, 3, 2, "shift", d=d, c=c, mu=0.0)
    arms.set_slot(t, 3, 0, "add", d=d, alpha=2.0)
    arms.set_slot(t, 1, 2, "measure", d=d)
    return t, d, c


def test_set_slot_semantics():
    t, d, c = _tables()
    du = d / np.linalg.norm(d)
    assert np.allclose(t["p"][1, 0], du) and np.allclose(t["e"][1, 0], du) and t["a"][1, 0] == 0.5
    assert np.allclose(t["mu"][1, 0], [1, 2, 3, 4]) and t["mc"][1] == 0
    assert t["mc"][2] == 1 and np.allclose(t["mu"][2, 1], 0.25)
    assert np.allclose(t["e"][3, 2], c / np.linalg.norm(c)) and np.allclose(t["p"][3, 2], du)
    assert t["g"][3, 0] == 2.0 and t["a"][3, 0] == 0
    assert np.all(t["e"][1, 2] == 0) and t["a"][1, 2] == 0 and t["g"][1, 2] == 0
    raises(ValueError, arms.set_slot, t, 1, 0, "project_mc", d=np.random.default_rng(1).standard_normal((4, 5120)))
    raises(ValueError, arms.set_slot, t, 1, 0, "steer", d=d)


def test_write_arms_file_roundtrip():
    from safetensors.numpy import load_file

    t, _, _ = _tables()
    with tempfile.TemporaryDirectory() as tmp:
        sha = arms.write_arms_file(f"{tmp}/x", [12, 26, 30], labels={1: "secret"}, recipe={"r": 1}, **t)
        f = Path(tmp, "x.safetensors")
        assert sha == hashlib.sha256(f.read_bytes()).hexdigest()
        back = load_file(str(f))
        assert set(back) == {"layers", "p", "e", "mu", "a", "g", "mc"}
        assert back["layers"].dtype == np.int32 and back["mc"].dtype == np.int32 and back["p"].dtype == np.float32
        assert back["p"].shape == (4, 3, 4, 5120) and list(back["layers"]) == [12, 26, 30]
        side = json.loads(Path(tmp, "x.json").read_text())
        assert side["sha256"] == sha and side["labels"] == {"1": "secret"}
        bad = {k: v.copy() for k, v in t.items()}
        bad["a"][0, 0] = 1.0
        raises(ValueError, arms.write_arms_file, f"{tmp}/y", [12, 26, 30], labels={}, recipe={}, **bad)
        bad = {k: v.copy() for k, v in t.items()}
        bad["p"][1, 0, 0, 0] = np.nan
        raises(ValueError, arms.write_arms_file, f"{tmp}/y", [12, 26, 30], labels={}, recipe={}, **bad)
        raises(ValueError, arms.write_arms_file, f"{tmp}/y", [26, 12, 30], labels={}, recipe={}, **t)


def test_build_from_recipe():
    from safetensors.numpy import load_file

    with tempfile.TemporaryDirectory() as tmp:
        d = np.random.default_rng(2).standard_normal(5120)
        np.save(f"{tmp}/d.npy", d)
        Path(tmp, "mu.json").write_text(json.dumps({"mu": {"26": [0.1] * 4, "30": [0.2] * 4},
                                                    "sd": {"26": [2.0] * 4, "30": [1.0] * 4}}))
        recipe = {"layers": [26, 30], "n_arms": 4, "arms": [
            {"arm": 1, "label": "p", "slots": [{"layers": "all", "kind": "project", "d": f"{tmp}/d.npy",
                                                "mu": f"{tmp}/mu.json", "lambda": 1.0}]},
            {"arm": 2, "label": "add", "slots": [{"layers": [26], "kind": "add", "d": f"{tmp}/d.npy",
                                                  "mu": f"{tmp}/mu.json", "alpha_sd": 8}]}]}
        sha = arms.build_from_recipe(recipe, f"{tmp}/r")
        t = load_file(f"{tmp}/r.safetensors")
        assert sha and np.allclose(t["mu"][1, 1], 0.2) and t["a"][1].tolist() == [1.0, 1.0]
        assert t["g"][2].tolist() == [16.0, 0.0]


def _fake_dump(tmp, n=500, layers=(26, 30)):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(3)
    I = len(layers)
    coef = rng.normal(5.0, 2.0, (n, I, 3, 4)).astype(np.float32)
    coef[:, 1, 0] = rng.standard_cauchy((n, 4)).astype(np.float32) + 1.0  # heavy tail at 30 -> median
    pos = np.tile(np.arange(n // 2, dtype=np.int32), 2)
    arm = np.repeat(np.array([1, 2], np.int32), n // 2)
    tag = np.zeros(n, np.int32)
    vec = rng.standard_normal((n, 1, 5120)).astype(np.float32)
    save_file({"coef": coef, "coef_pos": pos, "coef_arm": arm, "coef_req": tag, "coef_tag": tag,
               "vec": vec, "vec_pos": pos, "vec_arm": arm, "vec_req": tag, "vec_tag": tag}, f"{tmp}/cap.safetensors")
    Path(f"{tmp}/cap.safetensors.json").write_text(json.dumps({"layers": list(layers), "capture_layers": [26],
                                                               "tags": ["g"], "req_ids": ["r0"]}))
    return coef, pos, arm


def test_mu_from_capture_and_gap():
    with tempfile.TemporaryDirectory() as tmp:
        coef, pos, arm = _fake_dump(tmp)
        m = arms.mu_from_capture(f"{tmp}/cap.safetensors", 1, tag="g", min_pos=4)
        sel = (arm == 1) & (pos >= 4)
        assert m["n"] == int(sel.sum())
        assert np.allclose(m["mu"]["26"], coef[sel, 0, 0].astype(np.float64).mean(0))
        assert np.allclose(m["mu"]["30"], np.median(coef[sel, 1, 0].astype(np.float64), 0))
        assert np.allclose(m["sd"]["26"], coef[sel, 0, 0].astype(np.float64).std(0, ddof=1))
        g = arms.gap_report({"26": [1.0] * 4}, {"26": [0.0, 0.0, 0.0, 0.9]}, {"26": [1.0] * 4})
        assert g["flagged_layers"] == [26]
        assert arms.gap_report({"26": [0.1] * 4}, {"26": [0.0] * 4}, {"26": [1.0] * 4})["flagged_layers"] == []


def test_control_sampler():
    rng = np.random.default_rng(4)
    scales = np.linspace(3, 0.1, 300)
    basis = np.linalg.qr(rng.standard_normal((5120, 300)))[0]
    states = (rng.standard_normal((2000, 300)) * scales) @ basis.T
    d = basis[:, 40] + 0.1 * basis[:, 200]
    vecs, info = arms.control_sampler(states, d, n=6, seed=0, top_pcs=200)
    x = states - states.mean(0)
    cov = x.T @ x / (len(x) - 1)
    dn = d / np.linalg.norm(d)
    vd = dn @ cov @ dn
    assert vecs.shape == (6, 5120) and np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5)
    assert info["max_abs_cos_d"] < 1e-5
    for v in vecs.astype(np.float64):
        assert abs((v @ cov @ v) / vd - 1) <= 0.25 + 1e-6
    again, _ = arms.control_sampler(states, d, n=6, seed=0, top_pcs=200)
    assert np.array_equal(vecs, again)  # seeded


def test_parse_prompt_logprobs_and_compare():
    ids = [0, 5, 7, 9]
    plp = [None, {"5": {"logprob": -0.1, "rank": 1}}, {"7": {"logprob": -2.0, "rank": 3}, "8": {"logprob": -0.5, "rank": 1}},
           {"9": {"logprob": -0.3, "rank": 1}}]
    r = client.parse_prompt_logprobs(ids, plp)
    assert r["lp"].tolist() == [-0.1, -2.0, -0.3] and r["top1"].tolist() == [5, 8, 9]
    assert r["hit"].tolist() == [True, False, True]
    raises(client.EngineError, client.parse_prompt_logprobs, ids[:3], plp)
    x = [{"lp": np.zeros(100), "top1": np.arange(100)}, {"lp": np.zeros(50), "top1": np.arange(50)}]
    y = [{"lp": np.full(100, 0.1), "top1": np.arange(100)}, {"lp": np.zeros(50), "top1": np.arange(50) + (np.arange(50) < 5)}]
    c = client.compare(x, y, ["generic", "agent"])
    assert c["generic"]["n_pos"] == 100 and abs(c["generic"]["median"] - 0.1) < 1e-12
    assert c["agent"]["top1_agree"] == 0.9 and c["all"]["n_prompts"] == 2
    floor = {"a_vs_b": c, "a_vs_c": c}
    ref = client.floor_reference(floor)
    assert client.within_floor(c, ref)["ok"]
    worse = {k: dict(v, median=v["median"] * 2 + 1) for k, v in c.items()}
    assert not client.within_floor(worse, ref)["ok"]


def test_save_load_runs():
    runs = {"a": [{"lp": np.arange(3.0), "top1": np.arange(3)}, {"lp": np.arange(5.0), "top1": np.arange(5)}]}
    with tempfile.TemporaryDirectory() as tmp:
        client.save_runs(f"{tmp}/r.npz", runs, ["p0", "p1"])
        back = client.load_runs(f"{tmp}/r.npz")
        assert [x["lp"].tolist() for x in back["a"]] == [[0, 1, 2], [0, 1, 2, 3, 4]]


def test_fresh_salt_unique():
    f = {"vllm_xargs": {"arm": 2}, "cache_salt": "s"}
    a, b = client.fresh_salt(f), client.fresh_salt(f)
    assert a["cache_salt"] != b["cache_salt"] and a["vllm_xargs"] == f["vllm_xargs"] and f["cache_salt"] == "s"
    assert len(a["cache_salt"]) == 24


def test_check_completion():
    ok = proxy.check_completion
    assert ok({"prompt": "x"}) is None
    assert ok({"vllm_xargs": {"arm": 0}}) is None
    assert ok({"vllm_xargs": {"arm": 3}, "cache_salt": "abc"}) is None  # no salt check without shas (tests only)
    for bad in ({"vllm_xargs": {"arm": 3}}, {"vllm_xargs": {"arm": "0"}}, {"vllm_xargs": {"arm": 0.0}},
                {"vllm_xargs": {"arm": 1}, "cache_salt": ""}, {"vllm_xargs": {"capture": "coef"}},
                {"vllm_xargs": {"arm": 2, "capture": "vec"}, "cache_salt": "x"}, {"vllm_xargs": [1]}, [1]):
        assert ok(bad)[0] == 400, bad
    # salts must match a loaded sha for this arm and window
    shas = {SHA, "cd" * 32}
    good = arms.request_fields(3, 5, 900, arms_sha="cd" * 32)
    assert ok({"prompt": [1], **good}, shas) is None
    assert ok({"prompt": [1], **arms.request_fields(3, arms_sha=SHA)}, shas) is None
    for bad in ({**good, "vllm_xargs": {**good["vllm_xargs"], "arm": 4}},  # another arm under this salt
                {**good, "vllm_xargs": {"arm": 3, "edit_from": 6, "edit_to": 900}},  # another window
                arms.request_fields(3, arms_sha="ef" * 32),  # sha not loaded in this boot
                {"vllm_xargs": {}, "cache_salt": "0" * 24},  # sham under a made-up salt
                {**good, "vllm_xargs": {"arm": 3, "edit_from": "5", "edit_to": 900}}):
        assert ok(bad, shas)[0] == 400, bad
    # global-arm block: only that arm passes
    assert ok({"prompt": [1]}, shas, global_arm=3)[0] == 503
    assert ok({"prompt": [1], **good}, shas, global_arm=3) is None
    assert ok({"prompt": [1], **arms.request_fields(2, arms_sha=SHA)}, shas, global_arm=3)[0] == 503
    assert ok({"prompt": [1]}, shas, global_arm=0) is None


def test_state_files():
    import os

    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("DSV41_ARMS_DIR")
        os.environ["DSV41_ARMS_DIR"] = d
        try:
            arms.record_loaded_sha("b1", SHA)
            arms.record_loaded_sha("b0", "cd" * 32)
            arms.record_loaded_sha("b1", "ef" * 32)
            arms.record_global_arm("b1", 4)
            st = proxy.EngineState("b1")
            assert st.shas() == {SHA, "ef" * 32} and st.global_arm() == 4
            arms.record_global_arm("b1", -1)
            assert st.global_arm() == -1
            assert proxy.EngineState("b2").shas() == set() and proxy.EngineState("b2").global_arm() == -1
        finally:
            os.environ.pop("DSV41_ARMS_DIR") if old is None else os.environ.__setitem__("DSV41_ARMS_DIR", old)


def test_metrics_parse_and_capture_n_guard():
    c = client.EngineClient("http://127.0.0.1:9", key_file="none")
    text = ('# HELP vllm:num_requests_running x\nvllm:num_requests_running{engine="0",model_name="dsv41"} 2.0\n'
            'vllm:num_requests_running{engine="1",model_name="dsv41"} 1.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="dsv41"} 0.0\n'
            'vllm:num_requests_waiting_by_reason{reason="capacity"} 5.0\n')
    c.request = lambda *a, **k: (200, text)
    assert c.metrics() == {"running": 3.0, "waiting": 0.0}
    raises(client.EngineError, c.wait_idle, 0.0)
    raises(ValueError, c.generate, [0, 1], arms.request_fields(0, arms_sha=SHA, capture="coef"), 4, n=2)


def test_load_dump_rejects_misaligned_rows():
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "dump.safetensors")
        t = {"coef": torch.zeros(3, 2, 3, 4), "coef_pos": torch.tensor([1, 2, 3], dtype=torch.int32),
             "coef_pos_dev": torch.tensor([1, 2, 3], dtype=torch.int32), "vec": torch.zeros(0, 1, 8).bfloat16()}
        save_file(t, path)
        Path(path + ".json").write_text("{}")
        arms.load_dump(path)
        t["coef_pos_dev"] = torch.tensor([1, 3, 2], dtype=torch.int32)
        save_file(t, path)
        raises(ValueError, arms.load_dump, path)


def test_engine_auth_middleware():
    import os
    from types import SimpleNamespace

    sys.path.insert(0, str(ENGINE))
    from dsv41_intervene import auth

    async def call_next(request):
        return "passed"

    def req(path, hdr=None):
        return SimpleNamespace(scope={"path": path}, headers={"authorization": hdr} if hdr else {})

    async def run():
        out = {}
        for name, r in {"health": req("/health"), "rpc_nokey": req("/collective_rpc"),
                        "rpc_key": req("/collective_rpc", "Bearer s3cret"), "metrics_bad": req("/metrics", "Bearer x"),
                        "v1_key": req("/v1/completions", "Bearer s3cret"),
                        "non_ascii": req("/collective_rpc", "Bearer s3cr\u00e9t\u20ac")}.items():
            res = await auth.require_key(r, call_next)
            out[name] = res if isinstance(res, str) else res.status_code
        return out

    old = os.environ.get("VLLM_API_KEY")
    os.environ["VLLM_API_KEY"] = "s3cret"
    try:
        out = asyncio.run(run())
        os.environ.pop("VLLM_API_KEY")
        closed = asyncio.run(auth.require_key(req("/v1/models", "Bearer "), call_next))
    finally:
        if old is not None:
            os.environ["VLLM_API_KEY"] = old
    assert out == {"health": "passed", "rpc_nokey": 401, "rpc_key": "passed", "metrics_bad": 401, "v1_key": "passed",
                   "non_ascii": 401}, out
    assert closed.status_code == 401  # no key configured: everything but /health is closed


def test_proxy_routes():
    """Proxy against a fake upstream: allowed routes pass with body and key unchanged, the rest is 404/400/401."""
    from aiohttp import ClientSession, web

    seen = []

    async def upstream_handler(request):
        seen.append((request.method, request.path, request.headers.get("Authorization"), await request.read()))
        if request.path == "/v1/completions":
            return web.json_response({"ok": True, "n": len(await request.read())})
        if request.path == "/collective_rpc":  # the proxy reads the engine boot at startup
            return web.json_response({"results": [{"rank": r, "boot": "b-up"} for r in (1, 0, 2, 3)]})
        return web.Response(text="up:" + request.path)

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / arms.LOADED_SHAS).write_text(f"b1 {SHA}\nb0 {'cd' * 32}\n")
    good = arms.request_fields(2, arms_sha=SHA)
    stale = arms.request_fields(2, arms_sha="cd" * 32)

    async def run():
        up = web.Application()
        up.router.add_route("*", "/{tail:.*}", upstream_handler)
        r1 = web.AppRunner(up)
        await r1.setup()
        s1 = web.TCPSite(r1, "127.0.0.1", 0)
        await s1.start()
        up_port = s1._server.sockets[0].getsockname()[1]
        r2 = web.AppRunner(proxy.make_app(f"127.0.0.1:{up_port}", key="k3y", state=proxy.EngineState("b1", root)))
        await r2.setup()
        s2 = web.TCPSite(r2, "127.0.0.1", 0)
        await s2.start()
        base = f"http://127.0.0.1:{s2._server.sockets[0].getsockname()[1]}"
        auth = {"Authorization": "Bearer k3y"}
        out = {}
        async with ClientSession() as s:
            big = {"prompt": list(range(40000)), "max_tokens": 1}
            async with s.post(base + "/v1/completions", json=big, headers=auth) as r:
                out["completion"] = (r.status, await r.json())
            for name, (m, path, body, hdr) in {
                "models": ("GET", "/v1/models", None, auth), "health_nokey": ("GET", "/health", None, {}),
                "tokenize": ("POST", "/tokenize", {"prompt": "x"}, auth),
                "rpc": ("POST", "/collective_rpc", {"method": "x"}, auth), "rpc_get": ("GET", "/collective_rpc", None, auth),
                "reset": ("POST", "/reset_prefix_cache", None, auth), "metrics": ("GET", "/metrics", None, auth),
                "models_post": ("POST", "/v1/models", None, auth), "chat": ("POST", "/v1/chat/completions", {}, auth),
                "nokey": ("POST", "/v1/completions", {"prompt": "x"}, {}),
                "badkey": ("POST", "/v1/completions", {"prompt": "x"}, {"Authorization": "Bearer nope"}),
                "arm_nosalt": ("POST", "/v1/completions", {"vllm_xargs": {"arm": 2}}, auth),
                "capture": ("POST", "/v1/completions", {"vllm_xargs": {"capture": "coef"}, "cache_salt": "x"}, auth),
                "arm_salt": ("POST", "/v1/completions", {"prompt": [1], **good}, auth),
                "arm_madeup_salt": ("POST", "/v1/completions", {"vllm_xargs": {"arm": 2}, "cache_salt": "x"}, auth),
                "arm_stale_boot_salt": ("POST", "/v1/completions", {"prompt": [1], **stale}, auth),
                "nonascii_key": ("POST", "/v1/completions", {"prompt": "x"}, {"Authorization": "Bearer k\u00e9y"}),
            }.items():
                async with s.request(m, base + path, json=body, headers=hdr) as r:
                    out[name] = r.status
            (root / arms.GLOBAL_ARM).write_text("b1 2\n")  # a global-arm block for arm 2
            for name, body in {"g_sham": {"prompt": [1]}, "g_arm2": {"prompt": [1], **good}}.items():
                async with s.post(base + "/v1/completions", json=body, headers=auth) as r:
                    out[name] = r.status
        await r2.cleanup()
        # without a given state the proxy reads the boot from the engine (rank 0) at startup
        import os

        old_dir = os.environ.get("DSV41_ARMS_DIR")
        os.environ["DSV41_ARMS_DIR"] = str(root)
        try:
            app = proxy.make_app(f"127.0.0.1:{up_port}", key="k3y")
            r3 = web.AppRunner(app)
            await r3.setup()
            out["boot"] = app["state"].boot
            await r3.cleanup()
        finally:
            os.environ.pop("DSV41_ARMS_DIR") if old_dir is None else os.environ.__setitem__("DSV41_ARMS_DIR", old_dir)
        await r1.cleanup()
        return out

    out = asyncio.run(run())
    tmp.cleanup()
    assert out["completion"][0] == 200
    want = {"models": 200, "health_nokey": 200, "tokenize": 200, "rpc": 404, "rpc_get": 404, "reset": 404,
            "metrics": 404, "models_post": 404, "chat": 404, "nokey": 401, "badkey": 401, "arm_nosalt": 400,
            "capture": 400, "arm_salt": 200, "arm_madeup_salt": 400, "arm_stale_boot_salt": 400,
            "nonascii_key": 401, "g_sham": 503, "g_arm2": 200, "boot": "b-up"}
    assert {k: out[k] for k in want} == want, out
    paths = [p for _, p, _, _ in seen]
    assert paths.count("/collective_rpc") == 1 and "/metrics" not in paths  # the one startup read
    first = seen[0]
    assert first[2] == "Bearer k3y" and json.loads(first[3])["prompt"][-1] == 39999  # key and body unchanged


def test_build_data_files():
    """Shapes and token counts of tools/build_data.py output (synthetic ids)."""
    import subprocess

    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ENGINE / "tools" / "build_data.py"), "--synthetic", "--out", d], check=True,
                       capture_output=True)
        texts = json.loads((Path(d) / "plumbing_texts.json").read_text())
        floor = json.loads((Path(d) / "floor_prompts.json").read_text())
    assert texts["tokenizer_sha256"] == floor["tokenizer_sha256"]
    for s in ("A", "B", "C"):
        assert len(texts[s]) == 30 and all(len(x) == 400 and x[0] == 0 for x in texts[s])
    gen = [p for p in floor["prompts"] if p["kind"] == "generic"]
    agent = [p for p in floor["prompts"] if p["kind"] == "agent"]
    assert len(gen) == 20 and len(agent) == 20 and len({p["id"] for p in floor["prompts"]}) == 40
    assert all(500 <= len(p["ids"]) <= 3000 for p in gen) and sum(len(p["ids"]) <= 1000 for p in gen) >= 6
    assert all(6000 <= len(p["ids"]) <= 30000 for p in agent)
    assert all(p["ids"][0] == 0 for p in floor["prompts"])


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
