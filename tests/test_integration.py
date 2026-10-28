"""CPU end to end across the three parts. No vLLM or GPU needed.

tools/arms.py writes an arms file and the request fields -> ``EditExt.dsv41_load_arms`` -> ``ArmModelState`` (stub
vLLM base) fills ``tok`` and capture rows for a prefill step and a decode step -> ``ops.edit_stack_`` (torch path)
at every slot -> ``dsv41_stats`` and ``dsv41_capture_dump`` -> tools/arms.py reads the dump (``coef_rows``,
``mu_from_capture``, ``vec_means``) and the Projection exact gate's arithmetic holds on it.
"""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)

from dsv41_intervene import HC, HIDDEN, PAD  # noqa: E402
from dsv41_intervene import arms, ops, rpc  # noqa: E402
from dsv41_intervene.capture import CaptureRings  # noqa: E402
from tools import arms as tarms  # noqa: E402

LAYERS = [12, 26, 27]
CAPS = [26]
A = 6
N_PROMPT = 12
EDIT_FROM_ADD = 5
ALPHA = 3.0


class _StubBase:
    def __init__(self, vllm_config, model, encoder_cache, device):
        self.max_num_reqs = 8

    def add_request(self, req_index, new_req_data):
        pass

    def prepare_inputs(self, input_batch, req_states):
        return {}

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        return {}


class _Staging:
    def copy_to_gpu(self, x, out=None):
        return out.copy_(torch.from_numpy(x))


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """A rank-0 CPU engine: EditState with real capture rings, ArmModelState on a stub base, EditExt."""
    mod = types.ModuleType("vllm.models.deepseek_v41.nvidia.model_state")
    mod.DeepseekV41ModelState = _StubBase
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v41", "vllm.models.deepseek_v41.nvidia"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(arms, "_MODEL_STATE_CLS", None)
    cpu = torch.device("cpu")
    st = arms.alloc_state(LAYERS, CAPS, A, 256, 0, cpu, CaptureRings(len(LAYERS), CAPS, 1024, 64, cpu, True),
                          _Staging())
    monkeypatch.setattr(arms, "STATE", st)
    monkeypatch.setenv("DSV41_CAPTURE_DIR", str(tmp_path / "captures"))
    ms = arms.make_model_state_cls()(None, None, None, cpu)
    return st, ms, rpc.EditExt()


def _write_arms(tmp_path, rng):
    """Arms 1-5 via tools/arms.py: project @26, add @27 (from 5), measure everywhere, shift @26, mean-coupled @26."""
    d = _unit(rng.standard_normal((HC, HIDDEN)))
    dm = _unit(rng.standard_normal(HIDDEN))
    c = rng.standard_normal((HC, HIDDEN))
    c = _unit(c - (c * d).sum(-1, keepdims=True) * d)
    mu = rng.standard_normal(HC)
    t = tarms.empty_tables(len(LAYERS), A)
    tarms.set_slot(t, 1, 1, "project", d=d, mu=mu)
    tarms.set_slot(t, 2, 2, "add", d=d, alpha=ALPHA)
    for slot in range(len(LAYERS)):
        tarms.set_slot(t, 3, slot, "measure", d=d)
    tarms.set_slot(t, 4, 1, "shift", d=d, c=c, mu=mu)
    tarms.set_slot(t, 5, 1, "project_mc", d=dm, mu=mu)
    labels = {1: "project", 2: "add", 3: "measure", 4: "shift", 5: "project_mc"}
    sha = tarms.write_arms_file(str(tmp_path / "arms" / "it"), LAYERS, **t, labels=labels, recipe={"test": 1})
    return str(tmp_path / "arms" / "it.safetensors"), sha, t


def _step(st, ms, reqs, start, cnt, t_pad, rng):
    """One engine step: prepare_inputs, then the edit at every slot on a fresh random stack per slot.
    Returns ``{slot: (h_before, h_after)}``."""
    n_tok = int(sum(cnt))
    batch = SimpleNamespace(req_ids=[r for r, _ in reqs], num_reqs=len(reqs),
                            idx_mapping_np=np.asarray([i for _, i in reqs], dtype=np.intp),
                            num_scheduled_tokens=np.asarray(cnt, dtype=np.int32), num_tokens=n_tok,
                            num_tokens_after_padding=t_pad, num_computed_tokens_np=np.asarray(start, dtype=np.int32))
    ms.prepare_inputs(batch, None)
    pos = torch.zeros(t_pad, dtype=torch.int64)
    pos[:n_tok] = torch.from_numpy(np.concatenate([s + np.arange(n) for s, n in zip(start, cnt)]))
    pre = torch.rand(t_pad, HC)
    out = {}
    for slot in range(len(LAYERS)):
        h = torch.from_numpy(rng.standard_normal((t_pad, HC, HIDDEN)).astype(np.float32)).to(torch.bfloat16)
        post = h.clone()
        ops.edit_stack_(post, pos, pre, slot, st.vslot(slot), st)
        out[slot] = (h, post)
    return out


def test_end_to_end(engine, tmp_path):
    st, ms, ext = engine
    rng = np.random.default_rng(0)
    path, sha, t = _write_arms(tmp_path, rng)
    loaded = ext.dsv41_load_arms(path, sha)
    assert loaded["arms_sha"] == sha and loaded["rank"] == 0 and loaded["layers"] == LAYERS
    for k in ("p", "e", "mu", "a", "g", "mc"):
        assert loaded["checksums"][k] == pytest.approx(float(t[k].astype(np.float64).sum()), rel=1e-6, abs=1e-6)
    ep = np.einsum("aisd,aisd->ais", t["p"].astype(np.float64), t["e"].astype(np.float64))
    np.testing.assert_allclose(st.ep.numpy(), ep, atol=1e-6)

    # requests: arm k in slot k, coef capture of every position (vec for the measure arm), tag "r<k>"
    fields = {}
    for k in range(A):
        kw = {"edit_from": EDIT_FROM_ADD} if k == 2 else {}
        fields[k] = tarms.request_fields(k, arms_sha=sha, capture="vec" if k == 3 else "coef", tag=f"r{k}", **kw)
        assert "cache_salt" in fields[k] and len(fields[k]["cache_salt"]) == 24
        ms.add_request(k, SimpleNamespace(req_id=f"q{k}",
                                          sampling_params=SimpleNamespace(extra_args=fields[k]["vllm_xargs"])))
    assert tarms.request_fields(0, arms_sha=sha) == {"vllm_xargs": {}}  # plain sham: no salt, no xargs
    assert st.invalid_arm_requests == 0 and st.extra.get("invalid_capture_requests", 0) == 0
    reqs = [(f"q{k}", k) for k in range(A)]
    ext.dsv41_stats("1")
    prefill = _step(st, ms, reqs, [0] * A, [N_PROMPT] * A, A * N_PROMPT + 8, rng)
    decode = _step(st, ms, reqs[::-1], [N_PROMPT] * A, [1] * A, A + 2, rng)  # batch order reversed

    s = ext.dsv41_stats("0")
    seen, edited = np.asarray(s["tokens_seen"]), np.asarray(s["tokens_edited"])
    n = N_PROMPT + 1
    assert seen.tolist() == [n] * A
    want = np.zeros((A, len(LAYERS)), np.int64)
    want[1, 1] = want[4, 1] = want[5, 1] = n - 1  # position 0 never edited
    want[2, 2] = n - EDIT_FROM_ADD
    assert edited.tolist() == want.tolist()
    assert np.asarray(s["nonfinite"]).sum() == 0 and np.asarray(s["cast_violations"]).sum() == 0
    assert s["coef_rows_used"] == A * n and s["vec_rows_used"] == n and s["coef_overflow"] == 0
    # sham, measure-only and padding rows are bit-identical at every slot
    for out, order, cnt in ((prefill, list(range(A)), N_PROMPT), (decode, list(range(A))[::-1], 1)):
        arm = np.repeat(order, cnt)
        for slot, (h, post) in out.items():
            n_real = len(arm)
            assert torch.equal(post[n_real:], h[n_real:])
            for k in (0, 3):
                rows = torch.from_numpy(np.flatnonzero(arm == k))
                assert torch.equal(post[rows], h[rows])

    dump = ext.dsv41_capture_dump(str(tmp_path / "captures" / "it.safetensors"))
    assert dump["coef_rows"] == A * n and dump["vec_rows"] == n
    arr, meta = tarms.load_dump(dump["path"])  # also checks coef_pos_dev (kernel) == coef_pos (host map)
    assert "coef_pos_dev" in arr and arr["coef_pos_dev"].size == A * n
    assert meta["layers"] == LAYERS and meta["capture_layers"] == CAPS and meta["arms_sha"] == sha
    assert sorted(meta["tags"]) == [f"r{k}" for k in range(A)]
    for k in range(A):
        sel = arr["coef_arm"] == k
        assert sorted(arr["coef_pos"][sel].tolist()) == list(range(n))

    mu = t["mu"].astype(np.float64)
    x1 = tarms.coef_rows(arr, meta, 1, "r1", min_pos=1)  # project @26: c' = mu up to the bf16 cast
    assert len(x1) == n - 1 and np.abs(x1[:, 1, 1] - mu[1, 1]).max() < 0.05
    np.testing.assert_array_equal(x1[:, 0, 0], x1[:, 0, 1])  # other slots untouched
    x2 = tarms.coef_rows(arr, meta, 2, "r2", min_pos=EDIT_FROM_ADD)  # add @27 from position 5
    assert len(x2) == n - EDIT_FROM_ADD and np.abs(x2[:, 2, 1] - x2[:, 2, 0] - ALPHA).max() < 0.05
    early = arr["coef"][(arr["coef_arm"] == 2) & (arr["coef_pos"] < EDIT_FROM_ADD)]
    np.testing.assert_array_equal(early[:, 2, 0], early[:, 2, 1])
    x4 = tarms.coef_rows(arr, meta, 4, "r4", min_pos=1)  # matched shift: d-coefficient unchanged
    assert np.abs(x4[:, 1, 1] - x4[:, 1, 0]).max() < 0.05
    x5 = tarms.coef_rows(arr, meta, 5, "r5", min_pos=1)  # mean-coupled: mean over streams -> mean(mu)
    assert np.abs(x5[:, 1, 1].mean(1) - mu[5, 1].mean()).max() < 0.05

    # the Projection exact gate's arithmetic (tools/gates.py) on this dump; its stats-vs-capture check holds
    max_err = np.asarray(s["max_err"])
    for k in (1, 4, 5):
        x = tarms.coef_rows(arr, meta, k, f"r{k}", min_pos=1)
        c, cp = x[:, 1, 0], x[:, 1, 1]
        a, g = float(t["a"][k, 1]), float(t["g"][k, 1])
        delta = (c.mean(1, keepdims=True) - mu[k, 1].mean()) if t["mc"][k] else c - mu[k, 1]
        target = c - a * delta * ep[k, 1] + g * ep[k, 1]
        err = np.abs(cp - target).max()
        assert err < 0.05 and abs(max_err[k, 1] - err) <= 1e-3 * max(1.0, err), (k, err, max_err[k, 1])

    # mu from the measure arm's capture, and its stream-mean vectors at 26 (pre-edit, bf16)
    m = tarms.mu_from_capture(dump["path"], arm=3, tag="r3", min_pos=4)
    x3 = tarms.coef_rows(arr, meta, 3, "r3", min_pos=4)
    np.testing.assert_allclose(m["mu"]["26"], x3[:, 1, 0].mean(0), rtol=1e-6)
    v = tarms.vec_means(arr, meta, 26, arm=3, tag="r3", min_pos=0)
    assert v.shape == (n, HIDDEN)
    order = np.argsort(arr["vec_pos"])
    first = np.repeat(range(A), N_PROMPT) == 3
    h26 = prefill[1][0][: A * N_PROMPT][torch.from_numpy(first)].float().mean(1)
    np.testing.assert_allclose(v[order][:N_PROMPT], h26.to(torch.bfloat16).double().numpy(), atol=1e-2)
    assert ext.dsv41_stats("0")["coef_rows_used"] == 0  # the dump reset the rings
