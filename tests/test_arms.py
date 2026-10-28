"""CPU tests: parse_xargs, token_columns (idx_mapping, padding -1, global arm), arms-file checks,
ArmModelState on a stub vLLM base class, and the RPC methods on a CPU EditState. No vLLM or GPU needed."""

import hashlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from dsv41_intervene import HC, HIDDEN, MAX_CAP_POS, MAX_EDIT_TO, PAD, TOK_ARM, TOK_CROW, TOK_P0, TOK_P1, TOK_VROW
from dsv41_intervene import arms
from dsv41_intervene import parse_layers

INSTALLED = parse_layers("12,14-38")
CAPS = parse_layers("12,15,18,22,25,26,27,30,34")
A = 4


class FakeRings:
    """Stand-in for capture.CaptureRings: assigns consecutive rows to the tokens a spec selects."""

    def __init__(self, enabled=True):
        self.coef = torch.zeros(64 if enabled else 0, len(INSTALLED), 3, 4)
        self.vec = torch.zeros(8 if enabled else 0, len(CAPS), 8)
        self.next_c = self.next_v = 0
        self.calls = []
        self.dumped = None

    def assign(self, specs, start, cnt, arm, out_crow, out_vrow):
        self.calls.append((list(specs), start.copy(), cnt.copy(), arm.copy()))
        t = 0
        for spec, s, c in zip(specs, start, cnt):
            for pos in range(int(s), int(s) + int(c)):
                if spec is not None and (pos in spec.pos if spec.pos is not None else spec.lo <= pos < spec.hi):
                    out_crow[t] = self.next_c
                    self.next_c += 1
                    if spec.kind == "vec":
                        out_vrow[t] = self.next_v
                        self.next_v += 1
                t += 1

    def stats(self):
        return {"coef_rows_used": self.next_c, "vec_rows_used": self.next_v, "coef_overflow": 0, "vec_overflow": 0}

    def dump(self, path, meta):
        self.dumped = (path, meta)
        return {"path": path, "coef_rows": self.next_c, "vec_rows": self.next_v}


class FakeStaging:
    def copy_to_gpu(self, x, out=None):
        return out.copy_(torch.from_numpy(x))


def cpu_state(rank=0, max_t=256, n_arms=A):
    return arms.alloc_state(INSTALLED, CAPS, n_arms, max_t, rank, torch.device("cpu"), FakeRings(rank == 0),
                            FakeStaging())


# ---------------------------------------------------------------------------------------------------------------------
# parse_xargs
# ---------------------------------------------------------------------------------------------------------------------

def test_parse_defaults():
    for xa in (None, {}, {"unrelated": "x"}, "not a dict"):
        r = arms.parse_xargs(xa, A)
        assert (r.arm, r.p0, r.p1, r.cap, r.valid, r.cap_valid) == (0, 1, MAX_EDIT_TO, None, True, True)


def test_parse_arm_and_window():
    r = arms.parse_xargs({"arm": 3, "edit_from": 120, "edit_to": 4000}, A)
    assert (r.arm, r.p0, r.p1, r.valid) == (3, 120, 4000, True)
    r = arms.parse_xargs({"arm": np.int64(2), "edit_from": 0, "edit_to": 0}, A)  # empty window is allowed
    assert (r.arm, r.p0, r.p1, r.valid) == (2, 0, 0, True)


@pytest.mark.parametrize("xa", [
    {"arm": A}, {"arm": -1}, {"arm": "3"}, {"arm": 3.0}, {"arm": True}, {"arm": [1]},
    {"arm": 1, "edit_from": 10, "edit_to": 5}, {"arm": 1, "edit_from": -1}, {"arm": 1, "edit_to": 2**31},
    {"arm": 1, "edit_from": "5"}, {"arm": 1, "edit_to": 1.5},
])
def test_parse_invalid_is_plain_sham(xa):
    r = arms.parse_xargs({**xa, "capture": "coef"}, A)
    assert (r.arm, r.p0, r.p1, r.cap, r.valid) == (0, 1, MAX_EDIT_TO, None, False)


def test_parse_capture_range_and_positions():
    r = arms.parse_xargs({"arm": 1, "capture": "coef", "cap_from": 4, "cap_to": 90, "tag": "t1"}, A, "req-a")
    c = r.cap
    assert (c.kind, c.lo, c.hi, c.pos, c.tag, c.req_id) == ("coef", 4, 90, None, "t1", "req-a")
    r = arms.parse_xargs({"capture": "coef"}, A)
    assert (r.cap.lo, r.cap.hi, r.cap.tag) == (0, MAX_EDIT_TO, "")
    r = arms.parse_xargs({"capture": "vec", "cap_pos": [30, 7, 7, 12], "cap_from": 100}, A)
    assert r.cap.kind == "vec" and r.cap.pos.tolist() == [7, 12, 30] and r.cap.pos.dtype == np.int64


@pytest.mark.parametrize("xa", [
    {"capture": "all"}, {"capture": 1}, {"capture": "coef", "tag": 5}, {"capture": "vec", "cap_pos": []},
    {"capture": "vec", "cap_pos": list(range(MAX_CAP_POS + 1))}, {"capture": "vec", "cap_pos": [1, "2"]},
    {"capture": "vec", "cap_pos": 5}, {"capture": "coef", "cap_from": 9, "cap_to": 3},
    {"capture": "coef", "cap_from": -1},
])
def test_parse_bad_capture_keeps_arm(xa):
    r = arms.parse_xargs({"arm": 2, "edit_from": 5, **xa}, A)
    assert (r.arm, r.p0, r.valid, r.cap, r.cap_valid) == (2, 5, True, None, False)


# ---------------------------------------------------------------------------------------------------------------------
# token_columns
# ---------------------------------------------------------------------------------------------------------------------

def _req(n=8):
    return {"arm": np.arange(n, dtype=np.int32) % A, "p0": np.arange(n, dtype=np.int32) + 10,
            "p1": np.arange(n, dtype=np.int32) + 1000}


def test_token_columns_mapping_and_padding():
    req = _req()
    idx = np.array([5, 2, 7], dtype=np.intp)  # batch order != slot order
    cnt = np.array([3, 1, 2], dtype=np.int32)  # a prefill chunk and two decodes
    host = arms.token_columns(req, idx, cnt, 8, -1)
    assert host.shape == (8, 5) and host.dtype == np.int32
    assert host[:6, TOK_ARM].tolist() == [1, 1, 1, 2, 3, 3]
    assert host[:6, TOK_P0].tolist() == [15, 15, 15, 12, 17, 17]
    assert host[:6, TOK_P1].tolist() == [1005, 1005, 1005, 1002, 1007, 1007]
    assert (host[:, [TOK_CROW, TOK_VROW]] == PAD).all() and (host[6:] == PAD).all()


def test_token_columns_global_arm_and_valid_mask():
    req = _req()
    idx = np.array([0, 1, 2], dtype=np.intp)
    cnt = np.array([2, 2, 1], dtype=np.int32)
    host = arms.token_columns(req, idx, cnt, 5, 3)
    assert host[:, TOK_ARM].tolist() == [3] * 5 and host[:, TOK_P0].tolist() == [10, 10, 11, 11, 12]
    host = arms.token_columns(req, idx, cnt, 6, 3, valid=np.array([True, False, True]))
    assert host[:, TOK_ARM].tolist() == [3, 3, PAD, PAD, 3, PAD]
    assert (host[2:4] == PAD).all() and (host[5] == PAD).all()


# ---------------------------------------------------------------------------------------------------------------------
# arms files and load_arms
# ---------------------------------------------------------------------------------------------------------------------

def _tables(n_arms=A, seed=0):
    g = torch.Generator().manual_seed(seed)
    I = len(INSTALLED)
    t = {"layers": torch.tensor(INSTALLED, dtype=torch.int32),
         "p": torch.randn(n_arms, I, HC, HIDDEN, generator=g), "e": torch.randn(n_arms, I, HC, HIDDEN, generator=g),
         "mu": torch.randn(n_arms, I, HC, generator=g), "a": torch.rand(n_arms, I, generator=g),
         "g": torch.rand(n_arms, I, generator=g), "mc": torch.tensor([0, 1, 0, 1][:n_arms], dtype=torch.int32)}
    for k in ("p", "e", "mu", "a", "g", "mc"):
        t[k][0] = 0
    for k in torch.nonzero(t["mc"] == 1).flatten().tolist():  # mean-coupled arms carry one direction on all streams
        for name in ("p", "e"):
            t[name][k] = t[name][k][:, :1].expand_as(t[name][k])
    return t


def _write(tmp_path, tables, name="arms"):
    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tables.items()}, str(path))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_arms_copies_and_checksums(tmp_path):
    st = cpu_state()
    t = _tables()
    path, sha = _write(tmp_path, t)
    out = arms.load_arms(st, path, sha.upper())
    assert out["arms_sha"] == sha == st.arms_sha
    for k in ("p", "e", "mu", "a", "g", "mc"):
        assert torch.equal(getattr(st, k), t[k])
        assert out["checksums"][k] == pytest.approx(float(t[k].double().sum()), rel=0, abs=1e-6)
    ep = (t["e"].double() * t["p"].double()).sum(-1).float()
    assert torch.equal(st.ep, ep) and st.ep[0].abs().sum() == 0


@pytest.mark.parametrize("bad", ["sha", "layers", "arm0", "nan", "mc", "mc_streams", "extra", "dtype", "shape",
                                 "n_arms"])
def test_load_arms_rejects_and_keeps_old_tables(tmp_path, bad):
    st = cpu_state()
    good, sha = _write(tmp_path, _tables(seed=1), "good")
    arms.load_arms(st, good, sha)
    before = {k: getattr(st, k).clone() for k in ("p", "e", "mu", "a", "g", "mc", "ep")}
    t = _tables(seed=2)
    if bad == "layers":
        t["layers"] = t["layers"].flip(0)
    elif bad == "arm0":
        t["g"][0, 3] = 0.5
    elif bad == "nan":
        t["e"][2, 1, 0, 7] = float("nan")
    elif bad == "mc":
        t["mc"][1] = 2
    elif bad == "mc_streams":  # arm 1 is mean-coupled; a different write direction on stream 2 is refused
        t["e"][1, 4, 2] += 1.0
    elif bad == "extra":
        t["labels"] = torch.zeros(1)
    elif bad == "dtype":
        t["p"] = t["p"].to(torch.bfloat16)
    elif bad == "shape":
        t["mu"] = t["mu"][:, :, :2]
    elif bad == "n_arms":
        t = _tables(n_arms=A - 1, seed=2)
    path, sha_bad = _write(tmp_path, t, "bad")
    with pytest.raises(ValueError):
        arms.load_arms(st, path, sha if bad == "sha" else sha_bad)
    for k, v in before.items():
        assert torch.equal(getattr(st, k), v)
    assert st.arms_sha == sha


def test_alloc_state_shapes():
    st = cpu_state(max_t=64)
    I = len(INSTALLED)
    assert st.n_inst == I == 26 and st.layers.tolist() == INSTALLED
    assert st.p.shape == (A, I, HC, HIDDEN) and st.p.dtype == torch.float32 and st.mc.dtype == torch.int32
    assert st.tok.shape == (64, 5) and (st.tok == PAD).all() and st.tok.dtype == torch.int32
    assert st.tokens_seen.shape == (A,) and st.tokens_edited.shape == (A, I) and st.nonfinite.dtype == torch.int64
    assert st.cast_violations.shape == (A, I) and st.cast_violations.dtype == torch.int64 and st.boot
    assert st.vslot(0) == 0 and st.vslot(1) == -1 and st.vslot(INSTALLED.index(34)) == len(CAPS) - 1
    st.tokens_edited += 3
    st.max_err += 1
    st.reset_counters()
    assert all(int(t.abs().sum()) == 0 for t in st.counters().values())


# ---------------------------------------------------------------------------------------------------------------------
# ArmModelState on a stub base class
# ---------------------------------------------------------------------------------------------------------------------

class _StubBase:
    def __init__(self, vllm_config, model, encoder_cache, device):
        self.max_num_reqs = 8
        self.base_calls = []

    def add_request(self, req_index, new_req_data):
        self.base_calls.append(("add", req_index))

    def prepare_inputs(self, input_batch, req_states):
        return {"lookback_token_ids": "from-base"}

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        return {"dummy": num_tokens}


@pytest.fixture
def model_state(monkeypatch):
    mod = types.ModuleType("vllm.models.deepseek_v41.nvidia.model_state")
    mod.DeepseekV41ModelState = _StubBase
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v41", "vllm.models.deepseek_v41.nvidia"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(arms, "_MODEL_STATE_CLS", None)

    def make(rank=0):
        monkeypatch.setattr(arms, "STATE", cpu_state(rank=rank, max_t=32))
        return arms.make_model_state_cls()(None, None, None, torch.device("cpu"))
    return make


def _new_req(req_id, extra):
    return SimpleNamespace(req_id=req_id, sampling_params=SimpleNamespace(extra_args=extra))


def _batch(req_ids, idx, cnt, start, t_pad):
    cnt = np.asarray(cnt, dtype=np.int32)
    return SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids), idx_mapping_np=np.asarray(idx, dtype=np.intp),
                           num_scheduled_tokens=cnt, num_tokens=int(cnt.sum()), num_tokens_after_padding=t_pad,
                           num_computed_tokens_np=np.asarray(start, dtype=np.int32))


def test_model_state_fills_tok_per_request(model_state):
    ms = model_state()
    st = arms.STATE
    ms.add_request(3, _new_req("r-sham", None))
    ms.add_request(1, _new_req("r-arm2", {"arm": 2, "edit_from": 5, "edit_to": 9}))
    ms.add_request(6, _new_req("r-bad", {"arm": 99}))
    assert st.invalid_arm_requests == 1 and ms.base_calls == [("add", 3), ("add", 1), ("add", 6)]
    st.tok.fill_(7)  # stale rows from an earlier step must be overwritten up to t_pad
    out = ms.prepare_inputs(_batch(["r-arm2", "r-sham", "r-bad"], [1, 3, 6], [4, 1, 1], [0, 50, 9], 8), None)
    assert out == {"lookback_token_ids": "from-base"}
    tok = st.tok[:8]
    assert tok[:, TOK_ARM].tolist() == [2, 2, 2, 2, 0, 0, PAD, PAD]
    assert tok[:6, TOK_P0].tolist() == [5, 5, 5, 5, 1, 1] and tok[:6, TOK_P1].tolist()[:4] == [9] * 4
    assert (tok[:, [TOK_CROW, TOK_VROW]] == PAD).all() and (tok[6:] == PAD).all()
    assert (st.tok[8:] == 7).all()  # rows past t_pad are never read by the kernel
    assert st.capture.calls == []  # no capture request: assign is skipped


def test_model_state_dummy_batches_and_global_arm(model_state):
    ms = model_state()
    st = arms.STATE
    ms.add_request(0, _new_req("real", {"arm": 1}))
    # make_dummy-style batch: slots 0..n-1 with random ids -> nothing edited or counted
    ms.prepare_inputs(_batch(["req_0_x", "req_1_y"], [0, 1], [3, 3], [0, 0], 6), None)
    assert (st.tok[:6] == PAD).all()
    st.global_arm = 3
    ms.prepare_inputs(_batch(["real"], [0], [2], [10], 4), None)
    assert st.tok[:4, TOK_ARM].tolist() == [3, 3, PAD, PAD]
    st.tok.fill_(0)
    assert ms.prepare_dummy_inputs(2, 4) == {"dummy": 4} and (st.tok == PAD).all()


def test_model_state_capture_rows_rank0_only(model_state):
    for rank in (0, 1):
        ms = model_state(rank)
        st = arms.STATE
        ms.add_request(2, _new_req("cap", {"arm": 1, "capture": "vec", "cap_pos": [11, 13], "tag": "x"}))
        ms.add_request(4, _new_req("plain", None))
        ms.prepare_inputs(_batch(["plain", "cap"], [4, 2], [2, 4], [0, 10], 8), None)
        crow, vrow = st.tok[:8, TOK_CROW].tolist(), st.tok[:8, TOK_VROW].tolist()
        if rank == 0:
            (specs, start, cnt, arm), = st.capture.calls
            assert specs[0] is None and specs[1].tag == "x" and specs[1].req_id == "cap"
            assert start.tolist() == [0, 10] and cnt.tolist() == [2, 4] and arm.tolist() == [0, 1]
            assert crow == [PAD, PAD, PAD, 0, PAD, 1, PAD, PAD] and vrow == crow
        else:
            assert st.capture.calls == [] and crow == vrow == [PAD] * 8


def test_model_state_rejects_oversize_batch(model_state):
    ms = model_state()
    ms.add_request(0, _new_req("r", None))
    with pytest.raises(AssertionError):
        ms.prepare_inputs(_batch(["r"], [0], [40], [0], 40), None)  # MAX_T is 32 here


# ---------------------------------------------------------------------------------------------------------------------
# RPC methods (EditExt) on a CPU state
# ---------------------------------------------------------------------------------------------------------------------

class _Layer(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.fuse_mhc_all_reduce = False
        self.mhc_stream = object()
        self.use_sequence_parallel = False
        self.ffn = SimpleNamespace(experts=SimpleNamespace(moe_config=SimpleNamespace(use_deferred_moe_finalize=False)))
        if wrapped:
            self.forward = lambda *a, **k: None


class DeepseekV4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_mhc_all_reduce = False
        self.layers = torch.nn.ModuleList(_Layer(i - 1 in INSTALLED) for i in range(40))


@pytest.fixture
def ext(monkeypatch, tmp_path):
    from dsv41_intervene import rpc

    monkeypatch.setattr(arms, "STATE", cpu_state())
    monkeypatch.setenv("DSV41_CAPTURE_DIR", str(tmp_path / "captures"))
    e = rpc.EditExt()
    wrapper = torch.nn.Module()
    wrapper.model = DeepseekV4Model()
    e.model_runner = SimpleNamespace(model=wrapper)
    e.vllm_config = SimpleNamespace(speculative_config=None)
    return e


def test_rpc_installed(ext):
    out = ext.dsv41_installed()
    assert out["installed"] and out["rank"] == 0 and out["layers"] == INSTALLED and out["wrapped"] == INSTALLED
    assert out["fuse_mhc_all_reduce"] is False and out["mhc_stream"] is True and out["sequence_parallel"] is False
    assert out["model_fuse_mhc_all_reduce"] is False and out["deferred_finalize"] == [] and out["boot"]
    assert out["all_reduce_mhc_built"] in (None, False)  # None: no vLLM in this test
    assert out["spec_decode"] is False and out["n_arms"] == A and out["n_inst"] == 26 and out["arms_sha"] == ""
    assert out["coef_rows"] == 64 and out["capture_layers"] == CAPS


def test_rpc_installed_reports_deferred_finalize(ext):
    model = ext.model_runner.model.model
    model.layers[20].ffn.experts.moe_config.use_deferred_moe_finalize = True
    model.fuse_mhc_all_reduce = True
    out = ext.dsv41_installed()
    assert out["deferred_finalize"] == [20] and out["model_fuse_mhc_all_reduce"] is True


def test_rpc_stats_reset_and_global_arm(ext):
    st = arms.STATE
    st.tokens_seen[0] = 5
    st.tokens_edited[2, 3] = 7
    st.invalid_arm_requests = 2
    out = ext.dsv41_stats("1")
    assert out["tokens_seen"][0] == 5 and out["tokens_edited"][2][3] == 7 and out["invalid_arm_requests"] == 2
    assert len(out["max_err"]) == A and len(out["max_err"][0]) == 26 and out["coef_rows_used"] == 0
    again = ext.dsv41_stats("0")
    assert again["tokens_seen"][0] == 0 and again["tokens_edited"][2][3] == 0 and again["invalid_arm_requests"] == 2
    with pytest.raises(ValueError):
        ext.dsv41_stats("yes")
    assert ext.dsv41_set_global_arm("3")["global_arm"] == 3 == st.global_arm
    assert ext.dsv41_set_global_arm("-1")["global_arm"] == -1
    for bad in ("4", "-2"):
        with pytest.raises(ValueError):
            ext.dsv41_set_global_arm(bad)


def test_rpc_stats_nonfinite_is_json_safe(ext):
    import json
    import math

    st = arms.STATE
    st.sum_abs_delta[A - 1, 13] = float("nan")
    st.max_err[1, 2] = float("inf")
    st.sum_abs_delta[2, 0] = 3.5
    out = ext.dsv41_stats("1")
    json.dumps(out, allow_nan=False)  # the API server must be able to serialize it
    assert out["sum_abs_delta"][A - 1][13] == -1.0 and out["max_err"][1][2] == -1.0 and out["sum_abs_delta"][2][0] == 3.5
    assert out["nonfinite_counter_cells"] == {"sum_abs_delta": 1, "max_err": 1}
    again = ext.dsv41_stats("0")
    assert not math.isnan(again["sum_abs_delta"][A - 1][13]) and again["nonfinite_counter_cells"] == {"sum_abs_delta": 0,
                                                                                                    "max_err": 0}


def test_rpc_load_arms_and_capture_dump(ext, tmp_path):
    path, sha = _write(tmp_path, _tables())
    out = ext.dsv41_load_arms(path, sha)
    assert out["arms_sha"] == sha and set(out["checksums"]) == {"p", "e", "mu", "a", "g", "mc"}
    root = tmp_path / "captures"
    for bad in (str(tmp_path / "elsewhere.safetensors"), str(root), str(root / ".." / "x.safetensors")):
        with pytest.raises(ValueError):
            ext.dsv41_capture_dump(bad)
    dump = ext.dsv41_capture_dump(str(root / "run1" / "c.safetensors"))
    assert dump["path"].endswith("c.safetensors") and dump["rank"] == 0 and root.joinpath("run1").is_dir()
    meta = arms.STATE.capture.dumped[1]
    assert meta["arms_sha"] == sha and meta["layers"] == INSTALLED and meta["capture_layers"] == CAPS
    (root / "run1" / "c.safetensors").write_bytes(b"")
    with pytest.raises(FileExistsError):
        ext.dsv41_capture_dump(str(root / "run1" / "c.safetensors"))


def test_rpc_capture_dump_other_ranks_noop(ext):
    arms.STATE.rank = 2
    out = ext.dsv41_capture_dump("/anywhere")
    assert out == {"rank": 2, "layers": INSTALLED, "arms_sha": "", "boot": arms.STATE.boot, "path": None,
                   "coef_rows": 0, "vec_rows": 0}


def test_model_state_with_real_capture_rings_and_dump(model_state, monkeypatch, tmp_path):
    """Contract check against core's CaptureRings: rows land in tok, the host map and the dump."""
    from dsv41_intervene import rpc
    from dsv41_intervene.capture import CaptureRings

    ms = model_state()
    st = arms.STATE
    st.capture = CaptureRings(st.n_inst, CAPS, 16, 4, torch.device("cpu"), enabled=True)
    ms.add_request(5, _new_req("cap", {"arm": 2, "capture": "vec", "cap_from": 3, "cap_to": 6, "tag": "g1"}))
    ms.add_request(1, _new_req("coef", {"capture": "coef", "cap_pos": [0, 2]}))
    ms.prepare_inputs(_batch(["cap", "coef"], [5, 1], [5, 3], [0, 0], 8), None)
    crow, vrow = st.tok[:8, TOK_CROW].tolist(), st.tok[:8, TOK_VROW].tolist()
    assert crow[:5] == [PAD, PAD, PAD, 0, 1] and crow[5:] == [2, PAD, 3]
    assert vrow == [PAD, PAD, PAD, 0, 1, PAD, PAD, PAD]
    assert st.capture.stats() == {"coef_rows_used": 4, "vec_rows_used": 2, "coef_overflow": 0, "vec_overflow": 0}
    monkeypatch.setenv("DSV41_CAPTURE_DIR", str(tmp_path))
    e = rpc.EditExt()
    out = e.dsv41_capture_dump(str(tmp_path / "d.safetensors"))
    assert out["coef_rows"] == 4 and out["vec_rows"] == 2 and (tmp_path / "d.safetensors.json").exists()
    from safetensors.torch import load_file
    d = load_file(str(tmp_path / "d.safetensors"))
    assert d["coef_pos"].tolist() == [3, 4, 0, 2] and d["coef_arm"].tolist() == [2, 2, 0, 0]
    assert d["vec_pos"].tolist() == [3, 4] and d["vec"].shape == (2, len(CAPS), HIDDEN)
