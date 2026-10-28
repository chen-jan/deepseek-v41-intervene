"""Engine-side arms: the per-rank ``EditState``, xargs parsing, ``ArmModelState``, ``load_arms``.

Shapes and field names: docs/INTERFACE.md sections 3-5 and 7. vLLM is imported lazily (CPU tests import this module).
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import (DEFAULT_COEF_ROWS, DEFAULT_N_ARMS, DEFAULT_VEC_ROWS, HC, HIDDEN, MAX_CAP_POS, MAX_EDIT_TO, PAD,
               TOK_ARM, TOK_COLS, TOK_CROW, TOK_P0, TOK_P1, TOK_VROW, capture_layers, edit_layers, env_int)
from .capture import CapSpec, CaptureRings

STATE: "EditState | None" = None  # one per worker process (= per TP rank); set by init_state

ARMS_KEYS = ("layers", "p", "e", "mu", "a", "g", "mc")
CAPTURE_KINDS = ("coef", "vec")


@dataclass
class EditState:
    """All GPU buffers the edit kernel reads or writes. Allocated once, before profiling and graph capture."""

    installed: list[int]
    capture_layers: list[int]
    n_arms: int
    max_t: int
    rank: int
    device: torch.device
    # arms table [A, I, ...] fp32 (mc int32 [A]); written only by load_arms
    layers: torch.Tensor
    p: torch.Tensor
    e: torch.Tensor
    mu: torch.Tensor
    a: torch.Tensor
    g: torch.Tensor
    ep: torch.Tensor
    mc: torch.Tensor
    # per-token buffer [MAX_T, 5] int32 (TOK_* columns) and its host staging pool
    tok: torch.Tensor
    tok_staging: Any
    # device counters
    tokens_seen: torch.Tensor
    tokens_edited: torch.Tensor
    sum_abs_delta: torch.Tensor
    max_err: torch.Tensor
    nonfinite: torch.Tensor
    cast_violations: torch.Tensor
    capture: CaptureRings
    # host state
    invalid_arm_requests: int = 0
    global_arm: int = -1
    arms_sha: str = ""
    extra: dict = field(default_factory=dict)
    # identifies this server process (rank 0's value names the boot; tools key caches and proxy state by it)
    boot: str = field(default_factory=lambda: f"{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}")

    @property
    def n_inst(self) -> int:
        return len(self.installed)

    def vslot(self, slot: int) -> int:
        """Index of installed slot ``slot`` in ``capture_layers``, or -1."""
        L = self.installed[slot]
        return self.capture_layers.index(L) if L in self.capture_layers else -1

    def counters(self) -> dict[str, torch.Tensor]:
        """The device counters by name (INTERFACE.md section 3)."""
        return {"tokens_seen": self.tokens_seen, "tokens_edited": self.tokens_edited,
                "sum_abs_delta": self.sum_abs_delta, "max_err": self.max_err, "nonfinite": self.nonfinite,
                "cast_violations": self.cast_violations}

    def reset_counters(self) -> None:
        """Zero the device counters in place."""
        for t in self.counters().values():
            t.zero_()


def alloc_state(installed: list[int], capture_layers: list[int], n_arms: int, max_t: int, rank: int,
                device: torch.device, capture: CaptureRings, tok_staging: Any) -> EditState:
    """Allocate every table, buffer and counter of an ``EditState`` on ``device`` (arms all zero, ``tok`` all -1)."""
    if n_arms < 2:
        raise ValueError(f"DSV41_N_ARMS must be >= 2 (arm 0 is sham), got {n_arms}")
    A, I = n_arms, len(installed)
    f32 = dict(dtype=torch.float32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    return EditState(
        installed=list(installed), capture_layers=list(capture_layers), n_arms=A, max_t=max_t, rank=rank,
        device=device,
        layers=torch.tensor(installed, dtype=torch.int32, device=device),
        p=torch.zeros(A, I, HC, HIDDEN, **f32), e=torch.zeros(A, I, HC, HIDDEN, **f32),
        mu=torch.zeros(A, I, HC, **f32), a=torch.zeros(A, I, **f32), g=torch.zeros(A, I, **f32),
        ep=torch.zeros(A, I, HC, **f32), mc=torch.zeros(A, dtype=torch.int32, device=device),
        tok=torch.full((max_t, TOK_COLS), PAD, dtype=torch.int32, device=device), tok_staging=tok_staging,
        tokens_seen=torch.zeros(A, **i64), tokens_edited=torch.zeros(A, I, **i64),
        sum_abs_delta=torch.zeros(A, I, **f32), max_err=torch.zeros(A, I, **f32),
        nonfinite=torch.zeros(A, I, **i64), cast_violations=torch.zeros(A, I, **i64), capture=capture)


def init_state(vllm_config, device: torch.device | None = None) -> EditState:
    """Allocate ``STATE`` from the ``DSV41_*`` env vars (INTERFACE.md section 2) on the current CUDA device.

    Asserts ``torch.cuda.current_device()`` matches ``device``. Capture rings are real on TP rank 0 only. Arms tables
    start all zero (every arm is sham until ``load_arms``). A second call returns the existing state.
    """
    global STATE
    if STATE is not None:
        return STATE
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

    cur = torch.cuda.current_device()
    device = torch.device("cuda", cur) if device is None else torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", cur)
    assert device == torch.device("cuda", cur), f"EditState device {device} is not the current device cuda:{cur}"
    installed = edit_layers()
    caps = capture_layers(installed)
    max_t = vllm_config.scheduler_config.max_num_batched_tokens
    rank = get_tensor_model_parallel_rank()
    rings = CaptureRings(len(installed), caps, env_int("DSV41_COEF_ROWS", DEFAULT_COEF_ROWS),
                         env_int("DSV41_VEC_ROWS", DEFAULT_VEC_ROWS), device, enabled=rank == 0)
    STATE = alloc_state(installed, caps, env_int("DSV41_N_ARMS", DEFAULT_N_ARMS), max_t, rank, device, rings,
                        UvaBufferPool((max_t, TOK_COLS), torch.int32))
    return STATE


@dataclass
class ReqArm:
    """Parsed per-request fields."""

    arm: int = 0
    p0: int = 1
    p1: int = 2**31 - 1
    cap: CapSpec | None = None
    valid: bool = True
    cap_valid: bool = True


def _int(v: Any, lo: int, hi: int) -> int | None:
    """``v`` as an int in ``[lo, hi]``, else None (bools and floats are rejected)."""
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        return None
    v = int(v)
    return v if lo <= v <= hi else None


def _parse_capture(xa: dict, req_id: str) -> CapSpec | None:
    """The capture fields of ``xa`` as a ``CapSpec``; raises ValueError on a malformed request."""
    kind = xa["capture"]
    if kind not in CAPTURE_KINDS:
        raise ValueError(f"capture must be one of {CAPTURE_KINDS}")
    tag = xa.get("tag", "")
    if not isinstance(tag, str):
        raise ValueError("tag must be a string")
    pos = None
    lo, hi = 0, MAX_EDIT_TO
    if "cap_pos" in xa:
        raw = xa["cap_pos"]
        if not isinstance(raw, (list, tuple)) or not 0 < len(raw) <= MAX_CAP_POS:
            raise ValueError(f"cap_pos must be a list of 1..{MAX_CAP_POS} positions")
        vals = [_int(v, 0, MAX_EDIT_TO - 1) for v in raw]
        if any(v is None for v in vals):
            raise ValueError("cap_pos entries must be ints in [0, 2**31-1)")
        pos = np.unique(np.asarray(vals, dtype=np.int64))
    else:
        lo = _int(xa.get("cap_from", 0), 0, MAX_EDIT_TO)
        hi = _int(xa.get("cap_to", MAX_EDIT_TO), 0, MAX_EDIT_TO)
        if lo is None or hi is None or lo > hi:
            raise ValueError("cap_from/cap_to must be ints with 0 <= cap_from <= cap_to <= 2**31-1")
    return CapSpec(kind=kind, lo=lo, hi=hi, pos=pos, tag=tag, req_id=req_id)


def parse_xargs(extra_args: dict | None, n_arms: int, req_id: str = "") -> ReqArm:
    """Parse ``vllm_xargs`` (INTERFACE.md section 4). Invalid arm -> arm 0 with ``valid=False``; bad capture
    fields -> no capture. Never raises on client input.

    A bad edit window (``edit_from``/``edit_to`` not ints in ``[0, 2**31-1]`` or ``edit_from > edit_to``) also makes
    the request invalid: it runs as plain sham with no capture, so a typo never edits more positions than asked.
    Unknown keys are ignored.
    """
    xa = extra_args if isinstance(extra_args, dict) else {}
    arm = _int(xa.get("arm", 0), 0, n_arms - 1)
    p0 = _int(xa.get("edit_from", 1), 0, MAX_EDIT_TO)
    p1 = _int(xa.get("edit_to", MAX_EDIT_TO), 0, MAX_EDIT_TO)
    if arm is None or p0 is None or p1 is None or p0 > p1:
        return ReqArm(valid=False)
    r = ReqArm(arm=arm, p0=p0, p1=p1)
    if "capture" in xa:
        try:
            r.cap = _parse_capture(xa, req_id)
        except ValueError:
            r.cap_valid = False
    return r


def batch_arms(req: dict[str, np.ndarray], idx: np.ndarray, global_arm: int,
               valid: np.ndarray | None = None) -> np.ndarray:
    """Effective arm per batch row: the request's arm, or ``global_arm`` when it is >= 0; -1 where not ``valid``."""
    arm = req["arm"][idx] if global_arm < 0 else np.full(len(idx), global_arm, dtype=np.int32)
    return arm if valid is None else np.where(valid, arm, PAD).astype(np.int32)


def token_columns(req: dict[str, np.ndarray], idx: np.ndarray, cnt: np.ndarray, t_pad: int,
                  global_arm: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Host ``[t_pad, 5]`` int32 block for one step: columns ARM/P0/P1 = ``np.repeat(req[...][idx], cnt)``
    (ARM = ``global_arm`` for real tokens when ``global_arm >= 0``), CROW/VROW = -1, padding rows all -1.

    ``valid`` (bool per batch row, optional) marks rows that belong to a known request; other rows (dummy batches)
    get -1 in every column.
    """
    n_tok = int(cnt.sum())
    assert n_tok <= t_pad, f"{n_tok} scheduled tokens > padded size {t_pad}"
    host = np.full((t_pad, TOK_COLS), PAD, dtype=np.int32)
    p0, p1 = req["p0"][idx], req["p1"][idx]
    if valid is not None:
        p0, p1 = np.where(valid, p0, PAD), np.where(valid, p1, PAD)
    host[:n_tok, TOK_ARM] = np.repeat(batch_arms(req, idx, global_arm, valid), cnt)
    host[:n_tok, TOK_P0] = np.repeat(p0, cnt)
    host[:n_tok, TOK_P1] = np.repeat(p1, cnt)
    return host


def _sha256_file(path: str) -> tuple[str, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest(), data


def check_arms(tables: dict[str, torch.Tensor], installed: list[int], n_arms: int) -> None:
    """Raise ValueError unless ``tables`` is a valid arms file for this engine (INTERFACE.md section 7)."""
    if set(tables) != set(ARMS_KEYS):
        raise ValueError(f"arms file keys {sorted(tables)} != {sorted(ARMS_KEYS)}")
    A, I = n_arms, len(installed)
    want = {"layers": ((I,), torch.int32), "p": ((A, I, HC, HIDDEN), torch.float32),
            "e": ((A, I, HC, HIDDEN), torch.float32), "mu": ((A, I, HC), torch.float32),
            "a": ((A, I), torch.float32), "g": ((A, I), torch.float32), "mc": ((A,), torch.int32)}
    for k, (shape, dtype) in want.items():
        t = tables[k]
        if tuple(t.shape) != shape or t.dtype != dtype:
            raise ValueError(f"arms[{k}] is {tuple(t.shape)} {t.dtype}, want {shape} {dtype}")
    if tables["layers"].tolist() != list(installed):
        raise ValueError(f"arms layers {tables['layers'].tolist()} != installed {list(installed)}")
    for k in ("p", "e", "mu", "a", "g"):
        if not bool(torch.isfinite(tables[k]).all()):
            raise ValueError(f"arms[{k}] has non-finite values")
    if not bool(((tables["mc"] == 0) | (tables["mc"] == 1)).all()):
        raise ValueError("arms[mc] must be 0 or 1")
    for k in torch.nonzero(tables["mc"] == 1).flatten().tolist():  # mean-coupled: one direction on all 4 streams
        for name in ("p", "e"):
            x = tables[name][k]
            if not torch.equal(x, x[:, :1].expand_as(x)):
                raise ValueError(f"arm {k} is mean-coupled but arms[{name}][{k}] differs between streams")
    for k in ("p", "e", "mu", "a", "g", "mc"):
        if bool(tables[k][0].any()):
            raise ValueError(f"arm 0 (sham) must be all zero, arms[{k}][0] is not")


def load_arms(st: EditState, path: str, sha256: str) -> dict:
    """Verify and copy an arms file into the tables (INTERFACE.md section 7). Returns ``{"arms_sha", "checksums"}``.

    Checks the sha256, keys, shapes, dtypes, ``layers == installed``, finite values, ``mc`` in {0, 1}, one direction
    on all 4 streams for mean-coupled arms, and an all-zero arm 0 before touching the tables; a failed check leaves
    the loaded arms as they were. ``ep = sum_h e*p`` is computed in float64 on the host. Checksums are float64 sums of
    the device tables after the copy.
    """
    from safetensors.torch import load as st_load

    got, data = _sha256_file(path)
    if got != sha256.strip().lower():
        raise ValueError(f"sha256 mismatch for {path}: file {got}, expected {sha256}")
    tables = st_load(data)
    check_arms(tables, st.installed, st.n_arms)
    ep = (tables["e"].double() * tables["p"].double()).sum(-1).float()
    cuda = st.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(st.device)
    for k in ("p", "e", "mu", "a", "g", "mc"):
        getattr(st, k).copy_(tables[k])
    st.ep.copy_(ep)
    if cuda:
        torch.cuda.synchronize(st.device)
    st.arms_sha = got
    return {"arms_sha": got, "checksums": checksums(st)}


def checksums(st: EditState) -> dict[str, float]:
    """float64 sums of the arms tables on the device (the client compares them across ranks)."""
    return {k: float(getattr(st, k).double().sum().item()) for k in ("p", "e", "mu", "a", "g", "mc")}


def make_model_state_cls():
    """Return ``ArmModelState``, a subclass of vLLM's ``DeepseekV41ModelState`` (built lazily so this module imports
    without vLLM). ``patch.apply`` installs it through ``get_model_state_cls``."""
    global _MODEL_STATE_CLS
    if _MODEL_STATE_CLS is not None:
        return _MODEL_STATE_CLS
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    class ArmModelState(DeepseekV41ModelState):
        """DeepseekV41ModelState plus per-request arms: fills ``STATE.tok`` before every step (graph replays too)."""

        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            assert STATE is not None, "arms.init_state must run in DeepseekV4Model.__init__ before the model state"
            n = self.max_num_reqs
            self.req = {"arm": np.zeros(n, dtype=np.int32), "p0": np.ones(n, dtype=np.int32),
                        "p1": np.full(n, MAX_EDIT_TO, dtype=np.int32)}
            self.req_cap: list[CapSpec | None] = [None] * n
            self.req_id: list[str] = [""] * n

        def add_request(self, req_index, new_req_data) -> None:
            super().add_request(req_index, new_req_data)
            st = STATE
            sp = new_req_data.sampling_params
            r = parse_xargs(getattr(sp, "extra_args", None), st.n_arms, new_req_data.req_id)
            if not r.valid:
                st.invalid_arm_requests += 1
            if not r.cap_valid:
                st.extra["invalid_capture_requests"] = st.extra.get("invalid_capture_requests", 0) + 1
            self.req["arm"][req_index] = r.arm
            self.req["p0"][req_index] = r.p0
            self.req["p1"][req_index] = r.p1
            self.req_cap[req_index] = r.cap
            self.req_id[req_index] = new_req_data.req_id

        def prepare_inputs(self, input_batch, req_states):
            model_inputs = super().prepare_inputs(input_batch, req_states)
            st = STATE
            n = input_batch.num_reqs
            t_pad = input_batch.num_tokens_after_padding
            assert t_pad <= st.max_t, f"num_tokens_after_padding {t_pad} > MAX_T {st.max_t}"
            idx = input_batch.idx_mapping_np[:n]
            cnt = input_batch.num_scheduled_tokens[:n]
            n_tok = int(cnt.sum())
            assert n_tok == input_batch.num_tokens, f"scheduled {n_tok} != num_tokens {input_batch.num_tokens}"
            # Dummy batches (profiling, warmup) reuse slot indices with random ids: they never match a request.
            req_id = self.req_id
            valid = np.fromiter((req_id[j] == r for j, r in zip(idx.tolist(), input_batch.req_ids)), bool, n)
            host = token_columns(self.req, idx, cnt, t_pad, st.global_arm, valid)
            if st.rank == 0:
                specs = [self.req_cap[j] if v else None for j, v in zip(idx.tolist(), valid.tolist())]
                if any(s is not None for s in specs):
                    crow = np.full(n_tok, PAD, dtype=np.int32)
                    vrow = np.full(n_tok, PAD, dtype=np.int32)
                    st.capture.assign(specs, input_batch.num_computed_tokens_np[:n], cnt,
                                      batch_arms(self.req, idx, st.global_arm, valid), crow, vrow)
                    host[:n_tok, TOK_CROW] = crow
                    host[:n_tok, TOK_VROW] = vrow
            st.tok_staging.copy_to_gpu(host, out=st.tok[:t_pad])
            return model_inputs

        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
            STATE.tok.fill_(PAD)
            return model_inputs

    _MODEL_STATE_CLS = ArmModelState
    return ArmModelState


_MODEL_STATE_CLS = None
