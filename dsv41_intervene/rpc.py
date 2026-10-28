"""Worker extension with the ``dsv41_*`` collective RPC methods.

Launch with ``--worker-extension-cls dsv41_intervene.rpc.EditExt``. Importing this module calls
``plugin.register()``, which patches the model when ``DSV41_ENGINE=1``. Args are strings; every method returns a dict
with ``"rank"`` (docs/INTERFACE.md section 8). Every attribute of ``EditExt`` carries the ``dsv41_`` prefix: vLLM asserts
that the extension adds no name the worker class already has (``v1/worker/worker_base.py``).
"""

import os
import time

import torch

from . import env_path
from . import arms, ops
from .plugin import register

register()


def _state() -> "arms.EditState":
    st = arms.STATE
    if st is None:
        raise RuntimeError("dsv41 engine is not installed in this worker (DSV41_ENGINE=1 and DeepSeek V4.1 needed)")
    return st


def _sync(st: "arms.EditState") -> None:
    """Wait for every in-flight step (async scheduling) before reading or resetting device state."""
    if st.device.type == "cuda":
        torch.cuda.synchronize(st.device)


def _base(st: "arms.EditState") -> dict:
    return {"rank": st.rank, "layers": list(st.installed), "arms_sha": st.arms_sha, "boot": st.boot}


def _deferred_finalize(model) -> list[int] | None:
    """Layers whose MoE leaves its top-k finalize to a fused all-reduce (must be none), or None if no layer exposes
    ``ffn.experts.moe_config``."""
    found, out = False, []
    for i, layer in enumerate(model.layers):
        cfg = getattr(getattr(getattr(layer, "ffn", None), "experts", None), "moe_config", None)
        if cfg is not None:
            found = True
            if bool(getattr(cfg, "use_deferred_moe_finalize", False)):
                out.append(i)
    return out if found else None


def _all_reduce_mhc_built() -> bool | None:
    """Whether the fused all-reduce + mHC kernel was built in this process (must be False); None without vLLM."""
    try:
        from vllm.models.deepseek_v41.nvidia.ops import mhc
    except ImportError:
        return None
    return getattr(mhc, "_all_reduce_mhc", None) is not None


def _decoder_model(model) -> torch.nn.Module | None:
    """The ``DeepseekV4Model`` inside the loaded model (LLM or VL wrapper), or None."""
    if not isinstance(model, torch.nn.Module):
        model = getattr(model, "model", None)  # a runner-side wrapper around the module
    if not isinstance(model, torch.nn.Module):
        return None
    for m in model.modules():
        if type(m).__name__ == "DeepseekV4Model":
            return m
    return None


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root and path != root


class EditExt:
    """Mixed into the vLLM worker class; methods run on every rank between steps."""

    def dsv41_installed(self) -> dict:
        """Installed layers, capture layers, fusion flag (must be False), mhc_stream present, N_ARMS, impl, SP and
        spec-decode flags, buffer sizes, arms_sha, boot. ``wrapped`` lists the installed outputs whose successor layer
        carries the patched ``forward``. Direct fusion evidence: ``model_fuse_mhc_all_reduce`` (False),
        ``deferred_finalize`` (layers that defer the MoE finalize, ``[]``) and ``all_reduce_mhc_built`` (False)."""
        st = arms.STATE
        if st is None:
            return {"rank": getattr(self, "rank", -1), "installed": False, "layers": [], "arms_sha": ""}
        model = _decoder_model(getattr(self.model_runner, "model", None))
        # Layer flags are None when the decoder cannot be found (reported, never guessed).
        flags = dict.fromkeys(("wrapped", "fuse_mhc_all_reduce", "mhc_stream", "sequence_parallel",
                               "model_fuse_mhc_all_reduce", "deferred_finalize"))
        if model is not None:
            succ = [model.layers[L + 1] for L in st.installed]
            flags.update(
                wrapped=[L for L, m in zip(st.installed, succ) if "forward" in vars(m)],
                fuse_mhc_all_reduce=any(bool(getattr(m, "fuse_mhc_all_reduce", True)) for m in succ),
                mhc_stream=all(getattr(m, "mhc_stream", None) is not None for m in succ),
                sequence_parallel=any(bool(getattr(m, "use_sequence_parallel", True)) for m in succ),
                model_fuse_mhc_all_reduce=getattr(model, "fuse_mhc_all_reduce", None),
                deferred_finalize=_deferred_finalize(model))
        cap = st.capture
        return {
            **_base(st),
            "installed": True,
            "capture_layers": list(st.capture_layers),
            **flags,
            "all_reduce_mhc_built": _all_reduce_mhc_built(),
            "spec_decode": self.vllm_config.speculative_config is not None,
            "n_arms": st.n_arms,
            "n_inst": st.n_inst,
            "impl": ops.IMPL,
            "max_t": st.max_t,
            "coef_rows": int(cap.coef.shape[0]),
            "vec_rows": int(cap.vec.shape[0]),
            "global_arm": st.global_arm,
        }

    def dsv41_load_arms(self, path: str, sha256: str) -> dict:
        """``arms.load_arms`` on this rank; returns ``arms_sha`` and float64 checksums."""
        st = _state()
        out = arms.load_arms(st, path, sha256)
        return {**_base(st), **out}

    def dsv41_stats(self, reset: str = "0") -> dict:
        """Counters as lists plus host counters and capture usage; ``reset="1"`` zeroes the device counters after
        reading (host counters such as ``invalid_arm_requests`` keep counting)."""
        if reset not in ("0", "1"):
            raise ValueError(f'reset must be "0" or "1", got {reset!r}')
        st = _state()
        _sync(st)
        # JSON cannot carry NaN/inf, and one non-finite edited row (the base model's batched-prefill NaN) makes
        # sum_abs_delta NaN until the next reset, which turned every later stats call into a 400 (first GPU boot, 28 Oct 2026).
        # Non-finite cells go out as -1.0 (no real counter is negative); "nonfinite" already counts those rows.
        cnt = st.counters()
        out = {**_base(st), **{k: (torch.nan_to_num(t, nan=-1.0, posinf=-1.0, neginf=-1.0) if t.is_floating_point()
                                   else t).tolist() for k, t in cnt.items()},
               "nonfinite_counter_cells": {k: int((~torch.isfinite(t)).sum()) for k, t in cnt.items()
                                           if t.is_floating_point()}}
        if reset == "1":
            st.reset_counters()
            _sync(st)
        out.update(st.capture.stats())
        out.update(invalid_arm_requests=st.invalid_arm_requests,
                   invalid_capture_requests=st.extra.get("invalid_capture_requests", 0),
                   global_arm=st.global_arm, t=round(time.time(), 3))
        return out

    def dsv41_capture_dump(self, path: str) -> dict:
        """Rank 0: dump the capture rings to ``path`` (must be under ``DSV41_CAPTURE_DIR``). Other ranks: no-op."""
        st = _state()
        if st.rank != 0:
            return {**_base(st), "path": None, "coef_rows": 0, "vec_rows": 0}
        root = env_path("DSV41_CAPTURE_DIR", "captures")
        if not _inside(path, root):
            raise ValueError(f"capture dump path must be inside DSV41_CAPTURE_DIR ({root})")
        if os.path.exists(path) or os.path.exists(path + ".json"):
            raise FileExistsError(f"{path} exists; capture dumps never overwrite")
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        _sync(st)
        meta = {"layers": list(st.installed), "capture_layers": list(st.capture_layers), "arms_sha": st.arms_sha,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        return {**_base(st), **st.capture.dump(path, meta)}

    def dsv41_set_global_arm(self, k: str) -> dict:
        """Set the global arm (``"-1"`` returns to per-request arms); read by ``prepare_inputs`` every step."""
        st = _state()
        arm = int(k)
        if not -1 <= arm < st.n_arms:
            raise ValueError(f"global arm must be in [-1, {st.n_arms - 1}], got {arm}")
        st.global_arm = arm
        return {**_base(st), "global_arm": st.global_arm}
