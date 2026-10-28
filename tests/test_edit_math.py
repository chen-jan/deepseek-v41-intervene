"""CPU gate "Edit math": torch reference on random [T, 4, 5120] (docs/GATES.md). No vLLM import.

Also checks the torch stack path (``edit_stack_torch_``) on a stand-in ``EditState``: counters, capture rows, padding,
and the capture rings' host side (``assign``, ``dump``).
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dsv41_intervene import HC, HIDDEN, PAD, TOK_COLS  # noqa: E402
from dsv41_intervene.capture import CapSpec, CaptureRings  # noqa: E402
from dsv41_intervene.ops import edit_ref, edit_stack_torch_  # noqa: E402

H = HIDDEN
BIG = 2**31 - 1


def unit(*shape, gen):
    v = torch.randn(*shape, H, generator=gen, dtype=torch.float64)
    return (v / v.norm(dim=-1, keepdim=True)).float()


def orth(v, d):
    """Unit vectors orthogonal to ``d`` (both [..., H])."""
    v = v.double() - (v.double() * d.double()).sum(-1, keepdim=True) * d.double()
    return (v / v.norm(dim=-1, keepdim=True)).float()


def fake_state(n_arms, n_inst, max_t, capture_layers=(), installed=None, coef_rows=64, vec_rows=32,
               device="cpu"):
    """Stand-in for ``arms.EditState`` with the INTERFACE.md section 3 fields (tables all zero, tok all -1)."""
    dev = torch.device(device)
    A, I = n_arms, n_inst
    z = dict(dtype=torch.float32, device=dev)
    installed = list(installed if installed is not None else range(I))
    cap_layers = list(capture_layers)
    return SimpleNamespace(
        installed=installed, capture_layers=cap_layers, n_arms=A, max_t=max_t, rank=0, device=dev,
        layers=torch.tensor(installed, dtype=torch.int32, device=dev),
        p=torch.zeros(A, I, HC, H, **z), e=torch.zeros(A, I, HC, H, **z), mu=torch.zeros(A, I, HC, **z),
        a=torch.zeros(A, I, **z), g=torch.zeros(A, I, **z), ep=torch.zeros(A, I, HC, **z),
        mc=torch.zeros(A, dtype=torch.int32, device=dev),
        tok=torch.full((max_t, TOK_COLS), PAD, dtype=torch.int32, device=dev),
        tokens_seen=torch.zeros(A, dtype=torch.int64, device=dev),
        tokens_edited=torch.zeros(A, I, dtype=torch.int64, device=dev),
        sum_abs_delta=torch.zeros(A, I, **z), max_err=torch.zeros(A, I, **z),
        nonfinite=torch.zeros(A, I, dtype=torch.int64, device=dev),
        cast_violations=torch.zeros(A, I, dtype=torch.int64, device=dev),
        capture=CaptureRings(I, cap_layers, coef_rows, vec_rows, dev, True),
        vslot=lambda s: cap_layers.index(installed[s]) if installed[s] in cap_layers else -1,
    )


def finish_tables(st):
    """What ``load_arms`` does after the copy: recompute ``ep``."""
    st.ep.copy_((st.e * st.p).sum(-1))


def table(n_arms, gen):
    """Per-slot tables [A, ...] with arm 0 sham and arms 1..6 = measure-only, project, mean-coupled project,
    matched shift, add, half-strength project."""
    p = torch.zeros(n_arms, HC, H)
    e = torch.zeros(n_arms, HC, H)
    mu = torch.zeros(n_arms, HC)
    a = torch.zeros(n_arms)
    g = torch.zeros(n_arms)
    mc = torch.zeros(n_arms, dtype=torch.int32)
    d = unit(HC, gen=gen)  # a different direction per stream
    dm = unit(1, gen=gen).expand(HC, H)  # mean-space direction copied to all 4
    c = orth(unit(HC, gen=gen), d)
    m = torch.randn(HC, generator=gen)
    p[1] = d                                                   # measure-only
    p[2], e[2], mu[2], a[2] = d, d, m, 1.0                     # project, per stream
    p[3], e[3], mu[3], a[3], mc[3] = dm, dm, m, 1.0, 1         # project, mean-coupled
    p[4], e[4], mu[4], a[4] = d, c, m, 1.0                     # matched shift
    e[5], g[5] = d, 8.0                                        # add
    p[6], e[6], mu[6], a[6] = d, d, m, 0.5                     # lambda = 0.5
    return dict(p=p, e=e, mu=mu, a=a, g=g, mc=mc), dict(d=d, dm=dm, c=c, m=m)


def batch(T, arm, gen, p0=1, p1=BIG):
    h = torch.randn(T, HC, H, generator=gen).to(torch.bfloat16)
    pos = torch.arange(T)
    arm = torch.full((T,), arm) if isinstance(arm, int) else arm
    return h, pos, arm, torch.full((T,), p0), torch.full((T,), p1)


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def coef64(h, p):
    return (h.double() * p.double()).sum(-1)


def sd_of(h, p):
    return coef64(h, p).std(0)


def test_sham_and_measure_only_are_identity(gen):
    tab, _ = table(8, gen)
    for k in (0, 1):
        h, pos, arm, p0, p1 = batch(64, k, gen)
        out, r = edit_ref(h, pos, arm, p0, p1, **tab)
        assert torch.equal(out, h) and not r["edited"].any()


def test_project_sets_coefficient_to_mu(gen):
    tab, v = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 2, gen)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    sd = sd_of(h, v["d"])
    assert r["edited"][1:].all() and not r["edited"][0]
    # fp32 coefficient of the pre-cast result is mu
    pre = h.double() - r["delta"].double()[..., None] * v["d"].double()
    err = (coef64(pre, v["d"]) - v["m"].double())[1:].abs()
    assert (err <= 1e-5 * sd).all(), err.max()
    torch.testing.assert_close(r["target"][1:], v["m"].expand(63, HC), rtol=0, atol=1e-5)
    # After the bf16 cast: h is on the bf16 grid, so a per-element step below half a grid spacing is lost, which
    # shrinks the edit systematically (error grows with |delta|). On N(0, 1) states this reaches ~0.035 sd.
    assert ((coef64(out, v["d"]) - v["m"].double())[1:].abs() <= 0.05 * sd).all()
    assert torch.allclose(r["c_post"].double(), coef64(out, v["d"]), atol=1e-4)


def test_mean_coupled_keeps_stream_differences(gen):
    tab, v = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 3, gen)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    dm = v["dm"]
    sd = sd_of(h, dm)
    c = coef64(h, dm)
    pre = h.double() - r["delta"].double()[..., None] * dm.double()
    cn = coef64(pre, dm)
    assert ((cn.mean(1) - v["m"].double().mean())[1:].abs() <= 1e-5 * sd.mean()).all()
    diff = (cn - cn[:, :1]) - (c - c[:, :1])
    assert (diff[1:].abs() <= 1e-5 * sd.mean()).all()
    assert torch.equal(r["delta"][:, 0:1].expand(-1, HC), r["delta"])


def test_matched_shift_same_displacement_and_unchanged_d(gen):
    tab, v = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 4, gen)
    out_s, r_s = edit_ref(h, pos, arm, p0, p1, **tab)
    _, r_p = edit_ref(h, pos, torch.full((64,), 2), p0, p1, **tab)
    sd = sd_of(h, v["d"])
    torch.testing.assert_close(r_s["delta"], r_p["delta"], rtol=0, atol=0)
    pre_s = h.double() - r_s["delta"].double()[..., None] * v["c"].double()
    pre_p = h.double() - r_p["delta"].double()[..., None] * v["d"].double()
    # d coefficient unchanged under the shift
    assert ((coef64(pre_s, v["d"]) - coef64(h, v["d"]))[1:].abs() <= 1e-5 * sd).all()
    # equal displacement norm per stream (before the cast; after it, bf16 rounding of h dominates)
    disp_s = (pre_s - h.double()).norm(dim=-1)[1:]
    disp_p = (pre_p - h.double()).norm(dim=-1)[1:]
    torch.testing.assert_close(disp_s, disp_p, rtol=1e-5, atol=1e-6)
    assert ((coef64(out_s, v["d"]) - coef64(h, v["d"]))[1:].abs() <= 0.02 * sd).all()


def test_add_moves_coefficient_by_g(gen):
    tab, v = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 5, gen)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    gain = (coef64(out, v["d"]) - coef64(h, v["d"]))[1:]
    assert ((gain - 8.0).abs() <= 0.02 * sd_of(h, v["d"])).all()
    assert r["edited"][1:].all()


def test_lambda_scales_removal(gen):
    tab, v = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 6, gen)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    c = coef64(h, v["d"])
    pre = h.double() - 0.5 * r["delta"].double()[..., None] * v["d"].double()
    want = c - 0.5 * (c - v["m"].double())
    assert ((coef64(pre, v["d"]) - want)[1:].abs() <= 1e-5 * sd_of(h, v["d"])).all()


def test_windows_and_position_zero(gen):
    tab, _ = table(8, gen)
    h, pos, arm, p0, p1 = batch(64, 2, gen, p0=5, p1=20)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    want = (pos >= 5) & (pos < 20)
    assert torch.equal(r["edited"], want)
    assert torch.equal(out[~want], h[~want]) and not torch.equal(out[want], h[want])
    h, pos, arm, p0, p1 = batch(8, 5, gen, p0=0)  # p0 = 0 still never edits position 0
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    assert not r["edited"][0] and r["edited"][1:].all() and torch.equal(out[0], h[0])


def test_padding_untouched(gen):
    tab, _ = table(8, gen)
    h, pos, _, p0, p1 = batch(16, 0, gen)
    arm = torch.full((16,), PAD)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    assert torch.equal(out, h) and not r["edited"].any()


def test_mixed_batch_rows_equal_single_arm_runs(gen):
    tab, _ = table(8, gen)
    T = 96
    h, pos, _, _, _ = batch(T, 0, gen)
    arm = torch.randint(-1, 7, (T,), generator=gen)
    p0 = torch.randint(0, 40, (T,), generator=gen)
    p1 = p0 + torch.randint(0, 60, (T,), generator=gen)
    out, r = edit_ref(h, pos, arm, p0, p1, **tab)
    for k in range(-1, 7):
        rows = (arm == k).nonzero().flatten()
        if rows.numel():
            o1, r1 = edit_ref(h[rows], pos[rows], arm[rows], p0[rows], p1[rows], **tab)
            assert torch.equal(out[rows], o1) and torch.equal(r["edited"][rows], r1["edited"])


def _load(st, slot, tab):
    for name in ("p", "e", "mu", "a", "g"):
        getattr(st, name)[:, slot].copy_(tab[name])
    st.mc.copy_(tab["mc"])


def test_stack_path_counters_capture_padding(gen):
    A, I, T, Tpad = 8, 3, 40, 48
    st = fake_state(A, I, max_t=64, capture_layers=[1], coef_rows=200, vec_rows=50)
    tabs = []
    for slot in range(I):
        tab, _ = table(A, gen)
        if slot == 2:
            tab["a"][6] = 0.0  # arm 6 off at slot 2
        tabs.append(tab)
        _load(st, slot, tab)
    finish_tables(st)
    h0 = torch.randn(Tpad, HC, H, generator=gen).to(torch.bfloat16)
    pos = torch.cat([torch.arange(20), torch.arange(100, 120), torch.zeros(Tpad - T, dtype=torch.int64)])
    arm = torch.randint(0, 7, (Tpad,), generator=gen)
    arm[T:] = PAD
    p0 = torch.where(torch.arange(Tpad) % 3 == 0, 10, 1)
    p1 = torch.full((Tpad,), BIG)
    st.tok[:Tpad, 0], st.tok[:Tpad, 1], st.tok[:Tpad, 2] = arm.int(), p0.int(), p1.int()
    st.tok[:Tpad, 3] = torch.where(torch.arange(Tpad) % 2 == 0, torch.arange(Tpad), -1).int()
    st.tok[:Tpad, 4] = torch.where(torch.arange(Tpad) % 4 == 0, torch.arange(Tpad) // 4, -1).int()
    st.tok[T:, 3:] = PAD
    pre_mix = torch.rand(Tpad, HC)
    post = h0.clone()
    want = h0.clone()
    refs = []
    for slot in range(I):
        before = want.clone()
        want, r = edit_ref(want, pos, arm, p0, p1, **tabs[slot])
        refs.append((before, r))
        edit_stack_torch_(post, pos, pre_mix, slot, st.vslot(slot), st)
    assert torch.equal(post, want)
    assert torch.equal(post[T:], h0[T:])
    seen = torch.bincount(arm[:T], minlength=A)
    assert torch.equal(st.tokens_seen, seen)
    for slot, (before, r) in enumerate(refs):
        ed = r["edited"]
        assert not ed[T:].any()
        for k in range(A):
            m = ed & (arm == k)
            assert st.tokens_edited[k, slot] == m.sum()
            torch.testing.assert_close(st.sum_abs_delta[k, slot], r["delta"][m].abs().sum(), rtol=1e-5, atol=1e-5)
            err = (r["c_post"] - r["target"]).abs().amax(-1)[m]
            torch.testing.assert_close(st.max_err[k, slot], err.max() if m.any() else torch.tensor(0.0),
                                       rtol=1e-4, atol=1e-6)
        crow = st.tok[:T, 3].long()
        rows = (crow >= 0).nonzero().flatten()
        cb = st.capture.coef[crow[rows], slot]
        assert torch.equal(cb[:, 0], r["c"][rows]) and torch.equal(cb[:, 1], r["c_post"][rows])
        assert torch.equal(cb[:, 2], pre_mix[rows])
        if st.vslot(slot) >= 0:
            vrow = st.tok[:T, 4].long()
            vr = (vrow >= 0).nonzero().flatten()
            mean = (before.float()[vr].sum(1) / HC).to(torch.bfloat16)
            assert torch.equal(st.capture.vec[vrow[vr], 0], mean)
    assert st.max_err.max() < 0.05 and st.nonfinite.sum() == 0
    assert st.cast_violations.sum() == 0  # every edited token within the bf16 rounding bound
    assert st.tokens_edited[0].sum() == 0 and st.tokens_edited[1].sum() == 0  # sham, measure-only
    crow = st.tok[:T, 3].long()
    rows = (crow >= 0).nonzero().flatten()
    assert torch.equal(st.capture.cpos[crow[rows]], pos[rows].int())  # device positions of the capture rows


def test_nonfinite_counted(gen):
    st = fake_state(8, 1, max_t=8)
    tab, _ = table(8, gen)
    _load(st, 0, tab)
    finish_tables(st)
    post = torch.randn(8, HC, H, generator=gen).to(torch.bfloat16)
    post[3, 1, 7] = float("nan")
    st.tok[:8, 0] = 2
    st.tok[:8, 1] = 1
    st.tok[:8, 2] = BIG
    edit_stack_torch_(post, torch.arange(8), torch.zeros(8, HC), 0, -1, st)
    assert st.nonfinite[2, 0] == 1 and st.tokens_edited[2, 0] == 7


def test_nonfinite_sham_rows_checked_at_first_and_last_slot(gen):
    """Sham rows compute only at the first and last installed slot, where a NaN state is counted under arm 0."""
    st = fake_state(8, 3, max_t=8)
    for slot in range(3):
        _load(st, slot, table(8, gen)[0])
    finish_tables(st)
    st.tok[:4, 0] = 0
    st.tok[:4, 1] = 1
    st.tok[:4, 2] = BIG
    for slot in range(3):
        post = torch.randn(4, HC, H, generator=gen).to(torch.bfloat16)
        post[2, 3, 11] = float("inf")
        edit_stack_torch_(post, torch.arange(4), torch.zeros(4, HC), slot, -1, st)
    assert st.nonfinite[0].tolist() == [1, 0, 1] and st.tokens_edited.sum() == 0


def test_cast_bound_flags_a_wrong_target(gen):
    """``cast_violations`` stays 0 for correct edits and counts every edited token when the target is off."""
    st = fake_state(8, 1, max_t=16)
    tab, _ = table(8, gen)
    _load(st, 0, tab)
    finish_tables(st)
    st.tok[:16, 0] = torch.tensor([2, 3, 4, 5, 6] * 3 + [2]).int()
    st.tok[:16, 1] = 1
    st.tok[:16, 2] = BIG
    pos = torch.arange(16)
    h = torch.randn(16, HC, H, generator=gen).to(torch.bfloat16)
    edit_stack_torch_(h.clone(), pos, torch.zeros(16, HC), 0, -1, st)
    assert st.cast_violations.sum() == 0 and st.tokens_edited.sum() == 15
    st.ep[2] += 5.0  # a stale ep: the target no longer matches what the edit does
    edit_stack_torch_(h.clone(), pos, torch.zeros(16, HC), 0, -1, st)
    assert st.cast_violations[2, 0] == 3 and st.cast_violations.sum() == 3


def test_cast_bound_holds_on_bf16_rounding(gen):
    """The bound holds from steps far below half a bf16 step (edit lost) to steps far above it."""
    p = unit(1, gen=gen).expand(HC, H).contiguous()
    h = (torch.randn(64, HC, H, generator=gen) * 50).to(torch.bfloat16)
    for w in (1e-3, 0.1, 3.0, 300.0):
        hf = h.float()
        hn = (hf + w * p).to(torch.bfloat16)
        cp = (hn.float() * p).sum(-1)
        target = (hf * p).sum(-1) + w * (p * p).sum(-1)
        bound = 0.004 * (p.abs() * torch.maximum(hf.abs(), hn.float().abs())).sum(-1)
        assert ((cp - target).abs() <= bound).all(), w


def test_capture_assign_and_dump(tmp_path):
    rings = CaptureRings(2, [5], coef_rows=10, vec_rows=3, device=torch.device("cpu"), enabled=True)
    specs = [CapSpec("coef", 2, 5, None, "t-a", "req-a"), None,
             CapSpec("vec", 0, BIG, np.array([0, 3, 4, 9]), "t-b", "req-b")]
    start, cnt, arm = np.array([0, 7, 2]), np.array([6, 2, 4]), np.array([1, 0, 2])
    crow, vrow = np.zeros(12, np.int32), np.zeros(12, np.int32)
    rings.assign(specs, start, cnt, arm, crow, vrow)
    # batch row 0: positions 0..5, capture 2,3,4; row 2: positions 2..5, capture 3,4
    assert crow.tolist() == [-1, -1, 0, 1, 2, -1, -1, -1, -1, 3, 4, -1]
    assert vrow.tolist() == [-1] * 9 + [0, 1, -1]
    rings.assign([CapSpec("vec", 0, BIG, None, "t-c", "req-c")], np.array([0]), np.array([8]), np.array([3]),
                 np.zeros(8, np.int32), np.zeros(8, np.int32))
    s = rings.stats()
    assert s == {"coef_rows_used": 10, "vec_rows_used": 3, "coef_overflow": 3, "vec_overflow": 7}
    rings.coef[:10] = torch.arange(10, dtype=torch.float32)[:, None, None, None]
    out = rings.dump(str(tmp_path / "cap.safetensors"), {"layers": [12, 14], "arms_sha": "x"})
    assert out["coef_rows"] == 10 and out["vec_rows"] == 3
    from safetensors.torch import load_file

    t = load_file(str(tmp_path / "cap.safetensors"))
    side = json.load(open(tmp_path / "cap.safetensors.json"))
    assert t["coef"].shape == (10, 2, 3, HC) and t["vec"].shape == (3, 1, H)
    assert t["coef_pos"].tolist() == [2, 3, 4, 3, 4, 0, 1, 2, 3, 4]
    assert t["coef_arm"].tolist() == [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
    assert [side["req_ids"][i] for i in t["coef_req"].tolist()][:5] == ["req-a"] * 3 + ["req-b"] * 2
    assert [side["tags"][i] for i in t["vec_tag"].tolist()] == ["t-b", "t-b", "t-c"]
    assert side["coef_overflow"] == 3 and side["capture_layers"] == [5]
    assert rings.stats()["coef_rows_used"] == 0 and rings.req_ids == []


def test_disabled_rings_assign_nothing():
    rings = CaptureRings(26, [26], coef_rows=10, vec_rows=10, device=torch.device("cpu"), enabled=False)
    assert rings.coef.shape[0] == 0 and rings.vec.shape[0] == 0 and rings.coef_buf.shape[0] == 1
    crow = np.zeros(4, np.int32)
    rings.assign([CapSpec("vec", 0, BIG, None, "", "r")], np.array([0]), np.array([4]), np.array([0]), crow,
                 np.zeros(4, np.int32))
    assert (crow == -1).all()


def _interp_check() -> None:
    """Triton kernel under ``TRITON_INTERPRET=1`` (CPU) vs the torch path: outputs, counters, capture rows."""
    from dsv41_intervene.ops import edit_stack_triton_

    gen = torch.Generator().manual_seed(5)
    A, I, T = 8, 3, 24
    sts = [fake_state(A, I, max_t=32, capture_layers=[1], coef_rows=64, vec_rows=16) for _ in range(2)]
    for slot in range(I):
        tab, _ = table(A, gen)
        for st in sts:
            _load(st, slot, tab)
    h = torch.randn(T, HC, H, generator=gen).to(torch.bfloat16)
    pos = torch.arange(T)
    pre = torch.rand(T, HC, generator=gen)
    arm = torch.randint(-1, 7, (T,), generator=gen).int()
    for st in sts:
        finish_tables(st)
        st.tok[:T, 0] = arm
        st.tok[:T, 1] = torch.where(torch.arange(T) % 3 == 0, 6, 1).int()
        st.tok[:T, 2] = BIG
        st.tok[:T, 3] = torch.where(torch.arange(T) % 2 == 0, torch.arange(T), -1).int()
        st.tok[:T, 4] = torch.where(torch.arange(T) % 3 == 0, torch.arange(T) // 3, -1).int()
    post_k, post_r = h.clone(), h.clone()
    for slot in range(I):
        edit_stack_triton_(post_k, pos, pre, slot, sts[0].vslot(slot), sts[0])
        edit_stack_torch_(post_r, pos, pre, slot, sts[1].vslot(slot), sts[1])
        d = (post_k.float() - post_r.float()).abs()
        ulp = torch.exp2(torch.floor(torch.log2(torch.maximum(post_k.float().abs(), post_r.float().abs())
                                                 .clamp(min=2.0**-100))) - 7)
        assert (d <= ulp).all(), f"slot {slot}: {(d / ulp).max()} ulp"
        post_r.copy_(post_k)
    k, r = sts
    for name in ("tokens_seen", "tokens_edited", "nonfinite"):
        assert torch.equal(getattr(k, name), getattr(r, name)), name
    # cast_violations is not compared: the interpreter truncates the bf16 cast (up to 1 step, the bound is 1/2)
    rows_k = k.tok[:T, 3][(arm >= 0) & (k.tok[:T, 3] >= 0)].long()
    assert torch.equal(k.capture.cpos[rows_k], pos[(arm >= 0) & (k.tok[:T, 3] >= 0)].int())
    torch.testing.assert_close(k.sum_abs_delta, r.sum_abs_delta, rtol=1e-5, atol=1e-4)
    # The interpreter truncates fp32 -> bf16 casts (the GPU rounds to nearest even), so c' and max_err carry a
    # small truncation bias here; tests/test_gpu_kernels.py compares them tightly.
    assert torch.equal(k.max_err == 0, r.max_err == 0) and k.max_err.max() < 0.15
    rows = k.tok[:T, 3][(arm >= 0) & (k.tok[:T, 3] >= 0)].long()
    ck, cr = k.capture.coef[rows], r.capture.coef[rows]
    torch.testing.assert_close(ck[:, :, 0], cr[:, :, 0], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(ck[:, :, 1], cr[:, :, 1], rtol=0, atol=0.15)
    assert torch.equal(ck[:, :, 2], cr[:, :, 2])
    vrows = k.tok[:T, 4][(arm >= 0) & (k.tok[:T, 4] >= 0)].long()
    assert (k.capture.vec[vrows].float() - r.capture.vec[vrows].float()).abs().max() <= 0.02
    assert k.tokens_edited.sum() > 0
    print("interp ok", k.tokens_edited.sum().item(), "edited token-slots")


def test_triton_kernel_in_interpreter():
    """Runs ``_interp_check`` in a subprocess with ``TRITON_INTERPRET=1`` (it must be set before Triton loads)."""
    pytest.importorskip("triton")
    import subprocess

    pylib = os.path.dirname(os.path.dirname(pytest.__file__))  # the subprocess imports this file, hence pytest
    env = {**os.environ, "TRITON_INTERPRET": "1", "DSV41_EDIT_IMPL": "triton",
           "PYTHONPATH": os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), pylib]))}
    out = subprocess.run([sys.executable, __file__, "--interp"], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and "interp ok" in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]


if __name__ == "__main__" and "--interp" in sys.argv:
    _interp_check()
