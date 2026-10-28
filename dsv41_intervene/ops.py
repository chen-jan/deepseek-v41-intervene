"""The edit operation: one Triton kernel, grid = tokens, plus a torch version with the same semantics.

Must import without vLLM; Triton is imported lazily. Semantics: docs/INTERFACE.md section 3 ("The edit").

Per token t at installed slot ``slot`` (arm ``k = tok[t, ARM]``, ``c_s = h_s . p_s`` in fp32):

    delta_s = c_s - mu_s                                (mc = 0)
    delta_s = mean_s(c) - mean_s(mu)                    (mc = 1)
    on      = k >= 0 and pos > 0 and p0 <= pos < p1 and (a != 0 or g != 0)
    h_s    <- bf16(h_s + (g - a * delta_s) * e_s)       (only where on)

A token row "computes" when it is edited or captured (``tok[t, CROW] >= 0``, or ``tok[t, VROW] >= 0`` at a vector
layer), and every real row computes at the first and the last installed slot; every other row exits after the
``tokens_seen`` count, so sham costs about one launch per slot plus two row reads. ``nonfinite`` counts computing rows
whose ``c`` or ``c'`` is not finite (so a NaN that reaches sham traffic shows up at the first or last slot).

``cast_violations`` counts edited rows where some stream misses its target by more than the bf16 rounding bound
``|c'_s - target_s| <= CAST_BOUND * sum_i |p_si| * max(|h_si|, |h'_si|)`` (half a bf16 step is at most 2**-8 of the
value, i.e. 0.0039216 of the rounded value; the rest of 0.004 covers the fp32 sums). It holds for any direction, so it
is the Projection exact gate's test at every slot.
"""

import os

import torch

from . import HC, HIDDEN, PAD, TOK_ARM, TOK_COLS, TOK_CROW, TOK_P0, TOK_P1, TOK_VROW

IMPL = os.environ.get("DSV41_EDIT_IMPL", "triton")
if IMPL not in ("triton", "torch"):
    raise ValueError(f"DSV41_EDIT_IMPL must be 'triton' or 'torch', not {IMPL!r}")

_BLOCK = 1024  # hidden-dim tile; HIDDEN % _BLOCK == 0
_TORCH_CHUNK = 512  # tokens per chunk in the torch path (bounds its fp32 temporaries)
_FMAX = 3.4e38  # |x| <= _FMAX is false for inf and NaN
CAST_BOUND = 0.004  # bf16 rounding bound factor (module docstring)

try:  # Triton ships with torch on CUDA builds; the torch path and the CPU tests do not need it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit(do_not_specialize=["pre_stride", "slot", "vslot", "n_inst", "n_cap"])
    def _edit_kernel(post_ptr, pos_ptr, pre_ptr, pre_stride, tok_ptr,
                     p_ptr, e_ptr, mu_ptr, a_ptr, g_ptr, ep_ptr, mc_ptr,
                     seen_ptr, edited_ptr, sad_ptr, maxerr_ptr, nonfin_ptr, viol_ptr, coef_ptr, cpos_ptr, vec_ptr,
                     slot, vslot, n_inst, n_cap,
                     NHC: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr, NCOL: tl.constexpr,
                     CAST: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        row = tok_ptr + t * NCOL
        arm = tl.load(row + 0)
        p0 = tl.load(row + 1)
        p1 = tl.load(row + 2)
        crow = tl.load(row + 3)
        vrow = tl.load(row + 4)
        pos = tl.load(pos_ptr + t)
        valid = arm >= 0
        k = tl.maximum(arm, 0)
        if valid & (slot == 0):
            tl.atomic_add(seen_ptr + k, tl.full([], 1, tl.int64))
        ks = k * n_inst + slot
        a = tl.load(a_ptr + ks)
        g = tl.load(g_ptr + ks)
        on = valid & (pos > 0) & (pos >= p0) & (pos < p1) & ((a != 0.0) | (g != 0.0))
        cap = valid & (crow >= 0)
        vcap = valid & (vslot >= 0) & (vrow >= 0)
        chk = valid & ((slot == 0) | (slot == n_inst - 1))
        if on | cap | vcap | chk:
            s = tl.arange(0, NHC)
            cols = tl.arange(0, BLOCK)
            tab = (ks * NHC + s)[:, None] * H
            base = t * (NHC * H) + s[:, None] * H
            vbase = (vrow.to(tl.int64) * n_cap + vslot) * H
            acc = tl.zeros([NHC, BLOCK], tl.float32)
            for j in range(0, H, BLOCK):
                hv = tl.load(post_ptr + base + j + cols[None, :]).to(tl.float32)
                acc += hv * tl.load(p_ptr + tab + j + cols[None, :])
                if vcap:
                    m = tl.sum(hv, axis=0) / NHC
                    tl.store(vec_ptr + vbase + j + cols, m.to(tl.bfloat16, fp_downcast_rounding="rtne"))
            c = tl.sum(acc, axis=1)
            mu = tl.load(mu_ptr + ks * NHC + s)
            mc = tl.load(mc_ptr + k)
            delta = tl.where(mc != 0, tl.sum(c, axis=0) / NHC - tl.sum(mu, axis=0) / NHC, c - mu)
            w = g - a * delta
            cp = c
            if on:
                acc2 = tl.zeros([NHC, BLOCK], tl.float32)
                bacc = tl.zeros([NHC, BLOCK], tl.float32)
                for j in range(0, H, BLOCK):
                    off = base + j + cols[None, :]
                    pv = tl.load(p_ptr + tab + j + cols[None, :])
                    hv = tl.load(post_ptr + off).to(tl.float32)
                    hn = hv + w[:, None] * tl.load(e_ptr + tab + j + cols[None, :])
                    hn = hn.to(tl.bfloat16, fp_downcast_rounding="rtne")
                    tl.store(post_ptr + off, hn)
                    hf = hn.to(tl.float32)
                    acc2 += hf * pv
                    bacc += tl.abs(pv) * tl.maximum(tl.abs(hv), tl.abs(hf))
                cp = tl.sum(acc2, axis=1)
                target = c + w * tl.load(ep_ptr + ks * NHC + s)
                dev = tl.abs(cp - target)
                err = tl.max(dev, axis=0)
                tl.atomic_add(edited_ptr + ks, tl.full([], 1, tl.int64))
                tl.atomic_add(sad_ptr + ks, tl.sum(tl.abs(delta), axis=0))
                tl.atomic_max(maxerr_ptr + ks, tl.where(err <= 3.4e38, err, 0.0))
                if tl.sum((dev > CAST * tl.sum(bacc, axis=1)).to(tl.int32), axis=0) > 0:
                    tl.atomic_add(viol_ptr + ks, tl.full([], 1, tl.int64))
            bad = tl.sum(((tl.abs(c) <= 3.4e38) & (tl.abs(cp) <= 3.4e38)).to(tl.int32), axis=0) < NHC
            if bad:
                tl.atomic_add(nonfin_ptr + ks, tl.full([], 1, tl.int64))
            if cap:
                cb = coef_ptr + (crow.to(tl.int64) * n_inst + slot) * (3 * NHC)
                tl.store(cb + s, c)
                tl.store(cb + NHC + s, cp)
                tl.store(cb + 2 * NHC + s, tl.load(pre_ptr + t * pre_stride + s))
                if slot == 0:
                    tl.store(cpos_ptr + crow.to(tl.int64), pos.to(tl.int32))


def edit_stack_(post: torch.Tensor, positions: torch.Tensor, pre_mix: torch.Tensor, slot: int, vslot: int,
                st) -> None:
    """Edit ``post`` ([T, 4, 5120] bf16, contiguous) in place at installed slot ``slot``.

    Reads per-token arm / window / capture rows from ``st.tok[:T]`` and positions from ``positions`` ([T] int64);
    updates ``st`` counters; writes capture rows into ``st.capture`` (coef always, vec when ``vslot >= 0``).
    ``pre_mix`` is the carried pre-mix [T, 4] fp32, stored in coef capture rows. Launches the same kernels whatever
    the arm values are (no host sync). Dispatches to the Triton kernel or ``edit_stack_torch_`` per ``IMPL``.
    """
    if IMPL == "torch" or not post.is_cuda:
        edit_stack_torch_(post, positions, pre_mix, slot, vslot, st)
    else:
        edit_stack_triton_(post, positions, pre_mix, slot, vslot, st)


def edit_stack_triton_(post: torch.Tensor, positions: torch.Tensor, pre_mix: torch.Tensor, slot: int, vslot: int,
                       st) -> None:
    """One launch of the Triton kernel, grid = ``T`` (also runs under ``TRITON_INTERPRET=1`` on CPU tensors)."""
    T = post.shape[0]
    if T == 0:
        return
    assert post.is_contiguous() and post.dtype == torch.bfloat16 and post.shape[1:] == (HC, HIDDEN)
    assert pre_mix.dtype == torch.float32 and pre_mix.stride(1) == 1 and T <= st.tok.shape[0]
    cap = st.capture
    _edit_kernel[(T,)](
        post, positions, pre_mix, pre_mix.stride(0), st.tok,
        st.p, st.e, st.mu, st.a, st.g, st.ep, st.mc,
        st.tokens_seen, st.tokens_edited, st.sum_abs_delta, st.max_err, st.nonfinite, st.cast_violations,
        cap.coef_buf, cap.cpos_buf, cap.vec_buf,
        slot, vslot, st.p.shape[1], cap.vec_buf.shape[1],
        NHC=HC, H=HIDDEN, BLOCK=_BLOCK, NCOL=TOK_COLS, CAST=CAST_BOUND, num_warps=8,
    )


def _edit_rows(h, pos, arm, p0, p1, P, E, mu, a, g, ep, mc) -> dict[str, torch.Tensor]:
    """The edit on gathered per-row tables (``P``, ``E`` [n, 4, H] fp32; ``mu``, ``ep`` [n, 4]; ``a``, ``g``,
    ``mc`` [n]). Returns the new bf16 rows (unedited rows unchanged) and the per-row quantities."""
    hf = h.float()
    c = (hf * P).sum(-1)
    delta = torch.where((mc != 0)[:, None], (c.mean(-1) - mu.mean(-1))[:, None].expand_as(c), c - mu)
    w = g[:, None] - a[:, None] * delta
    valid = arm >= 0
    on = valid & (pos > 0) & (pos >= p0) & (pos < p1) & ((a != 0) | (g != 0))
    hn = (hf + w[..., None] * E).to(torch.bfloat16)
    out = torch.where(on[:, None, None], hn, h)
    c_post = torch.where(on[:, None], (out.float() * P).sum(-1), c)
    bound = CAST_BOUND * (P.abs() * torch.maximum(hf.abs(), hn.float().abs())).sum(-1)
    return {"h": out, "hf": hf, "c": c, "c_post": c_post, "target": c + w * ep, "delta": delta, "on": on,
            "valid": valid, "bound": bound}


def edit_stack_torch_(post: torch.Tensor, positions: torch.Tensor, pre_mix: torch.Tensor, slot: int, vslot: int,
                      st) -> None:
    """Torch version with the kernel's semantics. Graph safe: fixed launches per static ``T``, no host sync.
    Uncaptured rows write to the rings' trash row. Works on CPU too (tests)."""
    T = post.shape[0]
    cap = st.capture
    n_inst = st.p.shape[1]
    chk = slot in (0, n_inst - 1)  # every real row is checked for non-finite values at the first and last slot
    coef_view = cap.coef_buf[:, slot]
    vec_view = cap.vec_buf[:, max(vslot, 0)]
    for t0 in range(0, T, _TORCH_CHUNK):
        t1 = min(t0 + _TORCH_CHUNK, T)
        h = post[t0:t1]
        tok = st.tok[t0:t1]
        arm = tok[:, TOK_ARM].long()
        k = arm.clamp(min=0)
        r = _edit_rows(h, positions[t0:t1], arm, tok[:, TOK_P0], tok[:, TOK_P1], st.p[k, slot], st.e[k, slot],
                       st.mu[k, slot], st.a[k, slot], st.g[k, slot], st.ep[k, slot], st.mc[k])
        valid, on = r["valid"], r["on"]
        crow = tok[:, TOK_CROW].long()
        vrow = tok[:, TOK_VROW].long()
        capm = valid & (crow >= 0)
        vcap = valid & (vrow >= 0) & (vslot >= 0)
        if vslot >= 0:
            vec_view.index_put_((torch.where(vcap, vrow, cap.vec_trash),),
                                (r["hf"].sum(1) / HC).to(torch.bfloat16))
        h.copy_(r["h"])
        flat = k * n_inst + slot
        if slot == 0:
            st.tokens_seen.index_add_(0, k, valid.long())
        st.tokens_edited.view(-1).index_add_(0, flat, on.long())
        st.sum_abs_delta.view(-1).index_add_(0, flat, torch.where(on, r["delta"].abs().sum(-1), 0.0))
        dev = (r["c_post"] - r["target"]).abs()
        err = dev.amax(-1)
        err = torch.where(on & (err <= _FMAX), err, 0.0)
        st.max_err.view(-1).scatter_reduce_(0, flat, err, reduce="amax")
        st.cast_violations.view(-1).index_add_(0, flat, (on & (dev > r["bound"]).any(-1)).long())
        fin = (r["c"].abs() <= _FMAX).all(-1) & (r["c_post"].abs() <= _FMAX).all(-1)
        st.nonfinite.view(-1).index_add_(0, flat, ((on | capm | vcap | (valid & chk)) & ~fin).long())
        vals = torch.stack([r["c"], r["c_post"], pre_mix[t0:t1].float()], dim=1)
        rows = torch.where(capm, crow, cap.coef_trash)
        coef_view.index_put_((rows,), vals)
        if slot == 0:
            cap.cpos_buf.index_put_((rows,), positions[t0:t1].to(torch.int32))


def edit_ref(h: torch.Tensor, pos: torch.Tensor, arm: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
             p: torch.Tensor, e: torch.Tensor, mu: torch.Tensor, a: torch.Tensor, g: torch.Tensor,
             mc: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure reference for one slot, for the CPU tests.

    Args: ``h`` [T, 4, H] bf16; ``pos``, ``arm``, ``p0``, ``p1`` [T] int; per-arm tables at this slot ``p``, ``e``
    [A, 4, H] fp32, ``mu`` [A, 4], ``a``, ``g`` [A], ``mc`` [A]. Arm -1 = padding (untouched, uncounted).
    Returns the edited copy of ``h`` and ``{"c": [T,4], "c_post": [T,4], "target": [T,4], "delta": [T,4],
    "bound": [T,4] (bf16 rounding bound), "edited": [T] bool}``.
    """
    arm = arm.long()
    k = arm.clamp(min=0)
    ep = (e * p).sum(-1)
    r = _edit_rows(h, pos.long(), arm, p0.long(), p1.long(), p[k], e[k], mu[k], a[k], g[k], ep[k], mc[k])
    return r["h"], {"c": r["c"], "c_post": r["c_post"], "target": r["target"], "delta": r["delta"],
                    "bound": r["bound"], "edited": r["on"]}


def warm_kernels(st) -> None:
    """Compile the Triton kernel once at a few token counts (1, 17, 1024) on scratch inputs, outside any capture,
    then zero the counters and capture rings it touched."""
    if IMPL != "triton" or st.device.type != "cuda":
        return
    st.tok.fill_(PAD)
    for T in sorted({min(n, st.tok.shape[0]) for n in (1, 17, 1024)}):
        post = torch.zeros(T, HC, HIDDEN, dtype=torch.bfloat16, device=st.device)
        pos = torch.zeros(T, dtype=torch.int64, device=st.device)
        pre = torch.zeros(T, HC, dtype=torch.float32, device=st.device)
        for vslot in (-1, 0):
            edit_stack_(post, pos, pre, 0, vslot, st)
    torch.cuda.synchronize(st.device)
    for name in ("tokens_seen", "tokens_edited", "sum_abs_delta", "max_err", "nonfinite", "cast_violations"):
        getattr(st, name).zero_()
    st.capture.reset()
