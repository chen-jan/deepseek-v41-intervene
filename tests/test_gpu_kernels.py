"""1-GPU gates: Triton vs torch at T in {1, 16, 17, 600, 4096}; CUDA-graph capture + in-place buffer
rewrite + replay (both impls); materialize path (mhc_post_tilelang + mhc_pre_delayed_tilelang) vs mhc_shifted_post_pre;
the layer wrapper end to end with a stand-in layer. Needs < 3 GB on one GPU.

Run on one free GPU of the node: ``CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_gpu_kernels.py``. Uses the
real layer-26/27 mHC parameters when ``$DSV41_MODEL_DIR`` holds the checkpoint, random ones otherwise.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dsv41_intervene import HC, HIDDEN, PAD  # noqa: E402
from dsv41_intervene import ops  # noqa: E402
from test_edit_math import BIG, fake_state, finish_tables, table  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one GPU")

H = HIDDEN
A, I = 8, 4
MAX_T = int(os.environ.get("DSV41_TEST_MAX_T", "4096"))  # lower it to shrink the GPU footprint
CAP = [1, 3]  # installed slots 0..3; vector capture at slots 1 and 3


def make_state(seed, max_t=4096):
    gen = torch.Generator().manual_seed(seed)
    st = fake_state(A, I, max_t, capture_layers=CAP, coef_rows=8192, vec_rows=4096, device="cuda")
    load_tables(st, gen)
    return st


def load_tables(st, gen):
    for slot in range(I):
        tab, _ = table(A, gen)
        for name in ("p", "e", "mu", "a", "g"):
            getattr(st, name)[:, slot].copy_(tab[name])
        st.mc.copy_(tab["mc"])
    finish_tables(st)


def fill_tok(st, T, seed, n_real=None):
    """Random arms of every kind, windows, coef capture on even rows, vector capture on every 4th row."""
    gen = torch.Generator().manual_seed(seed)
    n_real = T if n_real is None else n_real
    tok = torch.full((T, 5), PAD, dtype=torch.int32)
    r = torch.arange(n_real)
    tok[:n_real, 0] = torch.randint(-1, 7, (n_real,), generator=gen).int()
    tok[:n_real, 1] = torch.randint(0, 8, (n_real,), generator=gen).int()
    tok[:n_real, 2] = torch.where(r % 5 == 0, torch.randint(0, 600, (n_real,), generator=gen), BIG).int()
    tok[:n_real, 3] = torch.where(r % 2 == 0, r, -1).int()
    tok[:n_real, 4] = torch.where(r % 4 == 0, r // 4, -1).int()
    st.tok[:T].copy_(tok)
    pos = torch.randint(0, 3000, (T,), generator=gen)
    pos[: min(T, 3)] = torch.arange(min(T, 3))  # positions 0, 1, 2 present
    return pos.cuda()


def rand_inputs(T, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(T, HC, H, device="cuda", generator=gen).to(torch.bfloat16)
    pre = torch.rand(T, HC, device="cuda", generator=gen)
    return h, pre


def max_ulps(a, b):
    """Largest |a - b| in bf16 units in the last place of max(|a|, |b|)."""
    a, b = a.float(), b.float()
    m = torch.maximum(a.abs(), b.abs()).clamp(min=2.0**-100)
    ulp = torch.exp2(torch.floor(torch.log2(m)) - 7)
    return ((a - b).abs() / ulp).max().item()


def counters(st):
    return {n: getattr(st, n).clone() for n in ("tokens_seen", "tokens_edited", "sum_abs_delta", "max_err",
                                                "nonfinite", "cast_violations")}


def assert_counters(c1, c2):
    for n in ("tokens_seen", "tokens_edited", "nonfinite", "cast_violations"):
        assert torch.equal(c1[n], c2[n]), n
    assert c1["cast_violations"].sum() == 0  # every edit within the bf16 rounding bound (ops.CAST_BOUND)
    torch.testing.assert_close(c1["sum_abs_delta"], c2["sum_abs_delta"], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(c1["max_err"], c2["max_err"], rtol=0, atol=2e-3)


@pytest.mark.parametrize("T", [1, 16, 17, 600, 4096])
def test_triton_matches_torch(T):
    if T > MAX_T:
        pytest.skip(f"T={T} > DSV41_TEST_MAX_T")
    assert ops.IMPL == "triton"
    st_t, st_r = make_state(1), make_state(1)
    pos = fill_tok(st_t, T, seed=T)
    fill_tok(st_r, T, seed=T)
    h, pre = rand_inputs(T, seed=T)
    post_t, post_r = h.clone(), h.clone()
    for slot in range(I):
        ops.edit_stack_(post_t, pos, pre, slot, st_t.vslot(slot), st_t)
        ops.edit_stack_torch_(post_r, pos, pre, slot, st_r.vslot(slot), st_r)
        torch.cuda.synchronize()
        assert max_ulps(post_t, post_r) <= 1, f"slot {slot}"
        post_r.copy_(post_t)  # next slot starts from the same input
    assert_counters(counters(st_t), counters(st_r))
    tok = st_t.tok[:T].long().cpu()
    valid = tok[:, 0] >= 0
    assert st_t.tokens_seen.sum().item() == valid.sum().item()
    assert st_t.tokens_edited.sum().item() > 0 or T == 1
    rows = tok[valid & (tok[:, 3] >= 0), 3]
    assert torch.equal(st_t.capture.cpos[rows].cpu(), pos.cpu()[valid & (tok[:, 3] >= 0)].int())
    assert torch.equal(st_t.capture.cpos[rows], st_r.capture.cpos[rows])
    ct, cr = st_t.capture.coef[rows], st_r.capture.coef[rows]
    torch.testing.assert_close(ct[:, :, 0], cr[:, :, 0], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(ct[:, :, 1], cr[:, :, 1], rtol=1e-5, atol=2e-3)
    assert torch.equal(ct[:, :, 2], cr[:, :, 2])
    vrows = tok[valid & (tok[:, 4] >= 0), 4]
    assert max_ulps(st_t.capture.vec[vrows], st_r.capture.vec[vrows]) <= 1
    # padding (arm -1) rows untouched
    pad = (~valid).nonzero().flatten().cuda()
    assert torch.equal(post_t[pad], h[pad])


@pytest.mark.parametrize("impl", ["triton", "torch"])
def test_graph_replay_sees_rewritten_buffers(impl):
    fn = ops.edit_stack_ if impl == "triton" else ops.edit_stack_torch_
    T, n_real = 32, 20
    st, ref = make_state(2, max_t=64), make_state(2, max_t=64)
    post = torch.empty(T, HC, H, dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros(T, dtype=torch.int64, device="cuda")
    pre = torch.zeros(T, HC, device="cuda")

    def step(s, p):
        for slot in range(I):
            fn(p, pos, pre, slot, s.vslot(slot), s)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up (and compile) outside the capture
        step(st, post)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for name in counters(st):
        getattr(st, name).zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(st, post)

    outs = []
    for rnd in range(2):
        if rnd == 1:  # rewrite tables and per-token rows in place, as load_arms / prepare_inputs do
            for s in (st, ref):
                load_tables(s, torch.Generator().manual_seed(99))
        h, pre_v = rand_inputs(T, seed=10 + rnd)
        for s in (st, ref):
            pos.copy_(fill_tok(s, T, seed=20 + rnd, n_real=n_real))
        pre.copy_(pre_v)
        post.copy_(h)
        graph.replay()
        want = h.clone()
        step(ref, want)
        torch.cuda.synchronize()
        assert torch.equal(post, want)
        assert torch.equal(post[n_real:], h[n_real:])
        outs.append((h, post.clone()))
    assert_counters(counters(st), counters(ref))
    assert st.tokens_edited.sum() > 0
    # round 2 really used the new tables: the old tables give a different result
    old = make_state(2, max_t=64)
    fill_tok(old, T, seed=21, n_real=n_real)
    h2, out2 = outs[1]
    again = h2.clone()
    step(old, again)
    assert not torch.equal(again, out2)


def _hc_params():
    """Layer-26 FFN mixes (to make realistic carried mixes) and layer-27 attention params, from the checkpoint if
    ``$DSV41_MODEL_DIR`` has it, else random. Returns (params, source)."""
    base = os.environ.get("DSV41_MODEL_DIR", "")
    index = os.path.join(base, "model.safetensors.index.json")
    names = {"fn26": "layers.26.hc_ffn_fn", "scale26": "layers.26.hc_ffn_scale", "base26": "layers.26.hc_ffn_base",
             "w26": "layers.26.ffn_norm.weight", "fn27": "layers.27.hc_attn_fn",
             "scale27": "layers.27.hc_attn_scale", "base27": "layers.27.hc_attn_base",
             "w27": "layers.27.attn_norm.weight"}
    cfg = {"rms_norm_eps": 1e-6, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20}
    if os.path.exists(index):
        from safetensors import safe_open

        wmap = json.load(open(index))["weight_map"]
        out = {}
        for key, name in names.items():
            with safe_open(os.path.join(base, wmap[name]), "pt") as f:
                out[key] = f.get_tensor(name)
        hf = json.load(open(os.path.join(base, "config.json")))
        hf = hf.get("text_config", hf)  # the VL checkpoint keeps the decoder config under text_config
        cfg.update({k: v for k, v in hf.items() if k in cfg})
        source = "checkpoint"
    else:
        gen = torch.Generator().manual_seed(7)
        mix = HC * (HC + 2)
        out = {}
        for lay in ("26", "27"):
            out["fn" + lay] = torch.randn(mix, HC * H, generator=gen) * 1e-2
            out["scale" + lay] = torch.tensor([0.5, 0.5, 0.5])
            out["base" + lay] = torch.randn(mix, generator=gen) * 0.1
            out["w" + lay] = torch.ones(H)
        source = "random"
    dev = {k: (v.bfloat16() if k.startswith("w") else v.float()).cuda().contiguous() for k, v in out.items()}
    return dev, cfg, source


def _mix_args(prm, cfg, lay):
    eps, hc_eps = cfg["rms_norm_eps"], cfg["hc_eps"]
    return (prm["fn" + lay], prm["scale" + lay], prm["base" + lay], eps, hc_eps, hc_eps, 2.0,
            cfg["hc_sinkhorn_iters"])


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


def _boundary_inputs(T, prm, cfg):
    """Carried state at the layer-26 -> 27 boundary: x (FFN output), residual, post/res mixes and pre-mix."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang

    gen = torch.Generator(device="cuda").manual_seed(T)
    residual = torch.randn(T, HC, H, device="cuda", generator=gen).to(torch.bfloat16)
    x = (torch.randn(T, H, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    pre0 = torch.full((T, HC), 1.0 / HC, device="cuda")
    post_mix, res_mix, _, pre_mix = mhc_pre_delayed_tilelang(
        residual, *_mix_args(prm, cfg, "26"), pre_mix=pre0, norm_weight=prm["w26"], norm_eps=cfg["rms_norm_eps"])
    return x, residual, post_mix, res_mix, pre_mix


@pytest.mark.parametrize("T", [8, 16, 17, 600])
def test_materialize_equals_fused(T):
    if T > MAX_T:
        pytest.skip(f"T={T} > DSV41_TEST_MAX_T")
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    from vllm.models.deepseek_v41.nvidia.ops.mhc import (
        MHC_OVERLAP_MAX_TOKENS,
        mhc_pre_delayed_overlap,
        mhc_shifted_post_pre,
    )

    prm, cfg, source = _hc_params()
    print(f"mHC params: {source}")
    x, residual, post_mix, res_mix, pre_mix = _boundary_inputs(T, prm, cfg)
    kw = dict(pre_mix=pre_mix, norm_weight=prm["w27"], norm_eps=cfg["rms_norm_eps"])
    # materialize path (what the wrapper does, with the edit as a no-op)
    post = mhc_post_tilelang(x, residual, post_mix, res_mix)
    mat = mhc_pre_delayed_tilelang(post, *_mix_args(prm, cfg, "27"), **kw)
    # fused paths: TileLang with aux (capture_aux=True) and the default dispatch (Mega mHC where supported)
    for capture_aux in (True, False):
        res_f, *mix_f, aux = mhc_shifted_post_pre(x, residual, post_mix, res_mix, *_mix_args(prm, cfg, "27"),
                                                  capture_aux=capture_aux, stream=None, reduce_results=False,
                                                  **kw)
        torch.cuda.synchronize()
        assert max_ulps(res_f, post) <= 1, f"stack, capture_aux={capture_aux}"
        for name, a, b in zip(("post_mix", "res_mix", "layer_input", "next_pre_mix"), mix_f, mat):
            assert a.shape == b.shape and _rel(a, b) <= 1e-2, (name, capture_aux, _rel(a, b))
        if capture_aux:
            cos = torch.nn.functional.cosine_similarity(aux.float(), post.float().mean(1), dim=-1)
            assert cos.min().item() > 0.9999
    # the side-stream (overlap) variant that FULL graphs of <= 16 tokens pick inside the wrapped layer
    if T <= MHC_OVERLAP_MAX_TOKENS:
        side = torch.cuda.Stream()
        ov = mhc_pre_delayed_overlap(post, *_mix_args(prm, cfg, "27"), stream=side, **kw)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for name, a, b in zip(("post_mix", "res_mix", "layer_input", "next_pre_mix"), ov, mat):
            assert a.shape == b.shape and _rel(a, b) <= 1e-2, (name, "overlap", _rel(a, b))


def test_wrapper_materializes_edits_and_calls_original():
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

    from dsv41_intervene import arms, patch

    T = 24
    prm, cfg, _ = _hc_params()
    x, residual, post_mix, res_mix, pre_mix = _boundary_inputs(T, prm, cfg)
    st = make_state(3, max_t=64)
    pos = torch.arange(T, device="cuda")
    st.tok[:T, 0] = 2  # project at every slot
    st.tok[:T, 1] = 1
    st.tok[:T, 2] = BIG
    st.tok[:T, 3:] = PAD
    seen = {}

    def orig(x_, positions, input_ids, pre_mix_, post_mix_, res_mix_, residual_, engram_hashes, engram_mask, *,
             capture_previous_aux=False, mega_gate_metadata=None):
        seen.update(x=x_.clone(), pre_mix=pre_mix_, rest=(post_mix_, res_mix_, residual_), aux=capture_previous_aux)
        return x_, x_, post_mix_, res_mix_, pre_mix_, None

    prev, arms.STATE = getattr(arms, "STATE", None), st
    try:
        fwd = patch._make_forward(1, st.vslot(1), orig)
        out = fwd(x, pos, None, pre_mix, post_mix, res_mix, residual, None, None, capture_previous_aux=True,
                  mega_gate_metadata=None)
    finally:
        arms.STATE = prev
    want = mhc_post_tilelang(x, residual, post_mix, res_mix)
    ref = make_state(3, max_t=64)
    ref.tok.copy_(st.tok)
    ops.edit_stack_(want, pos, pre_mix, 1, ref.vslot(1), ref)
    torch.cuda.synchronize()
    assert torch.equal(seen["x"], want) and seen["pre_mix"] is pre_mix
    assert seen["rest"] == (None, None, None) and seen["aux"] is False
    assert st.tokens_edited[2, 1] == T - 1 and st.tokens_edited.sum() == T - 1
    torch.testing.assert_close(out[5].float(), want.float().mean(1), rtol=1e-2, atol=1e-2)
