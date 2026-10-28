"""Monkeypatches applied in each worker before the model is built.

- fusion off: ``model.supports_mhc_all_reduce`` -> False, so ``x`` is a reduced tensor at every boundary;
- ``DSV41_MHC_OVERLAP=0`` (fallback when only the FULL-overlap graph path fails): also
  ``model.supports_mhc_overlap`` -> False, so no layer gets the mHC side stream;
- ``DeepseekV4Model.__init__``: allocate ``arms.STATE`` and wrap ``layers[L+1].forward`` for every installed L;
- ``get_model_state_cls`` on both DeepSeek V4.1 classes -> ``arms.ArmModelState``.

See docs/INTERFACE.md sections 0, 3 and 5. Graph safety: the wrapper has no Python branch on arm state, and every
buffer the kernel reads is allocated before CUDA-graph capture and changed only in place.
"""

import inspect
import os
from collections.abc import Callable

import torch

_applied = False
_orig_init: Callable | None = None

# DeepseekV4DecoderLayer.forward as checked in vLLM 0.31 (model.py:446-462); the wrapper relies on this order.
_LAYER_PARAMS = ["x", "positions", "input_ids", "pre_mix", "post_mix", "res_mix", "residual", "engram_hashes",
                 "engram_mask", "capture_previous_aux", "mega_gate_metadata"]


def apply() -> None:
    """Install all patches once per process (idempotent). Raises if the DeepSeek V4.1 module layout differs."""
    global _applied, _orig_init
    if _applied:
        return
    import vllm.models.deepseek_v41.nvidia.model as M
    import vllm.models.deepseek_v41.nvidia.vl_model as VL

    from . import arms

    for mod, names in ((M, ("supports_mhc_all_reduce", "mhc_post_tilelang", "DeepseekV4Model",
                            "DeepseekV4DecoderLayer", "DeepseekV41LLMForCausalLM")),
                       (VL, ("DeepseekV41ForCausalLM",))):
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            raise RuntimeError(f"dsv41_intervene: {mod.__name__} lacks {missing}; vLLM layout changed")
    params = list(inspect.signature(M.DeepseekV4DecoderLayer.forward).parameters)[1:]
    if params != _LAYER_PARAMS:
        raise RuntimeError(f"dsv41_intervene: DeepseekV4DecoderLayer.forward signature changed: {params}")

    M.supports_mhc_all_reduce = lambda vllm_config: False
    if os.environ.get("DSV41_MHC_OVERLAP", "1") == "0":
        if not hasattr(M, "supports_mhc_overlap"):
            raise RuntimeError("dsv41_intervene: model.supports_mhc_overlap missing; vLLM layout changed")
        M.supports_mhc_overlap = lambda vllm_config: False
    _orig_init = M.DeepseekV4Model.__init__
    M.DeepseekV4Model.__init__ = _model_init
    state_cls = arms.make_model_state_cls()
    M.DeepseekV41LLMForCausalLM.get_model_state_cls = staticmethod(lambda: state_cls)
    VL.DeepseekV41ForCausalLM.get_model_state_cls = staticmethod(lambda: state_cls)
    _applied = True


def _model_init(self, *, vllm_config, prefix: str = "") -> None:
    """Patched ``DeepseekV4Model.__init__``: original init, asserts (no fusion, no sequence parallel, no spec decode,
    PP = 1, no micro-batching, no context parallel), ``arms.init_state``, wrapper install on ``layers[L+1]`` for each
    installed L, ``ops.warm_kernels``."""
    from vllm.distributed import get_pp_group
    from vllm.logger import init_logger

    from . import arms, ops

    _orig_init(self, vllm_config=vllm_config, prefix=prefix)
    assert not self.fuse_mhc_all_reduce, "dsv41_intervene: fused mHC all-reduce is still on"
    assert not self.use_sequence_parallel, "dsv41_intervene: sequence parallel is not supported"
    assert vllm_config.speculative_config is None, "dsv41_intervene: speculative decoding is not supported"
    assert get_pp_group().world_size == 1, "dsv41_intervene: pipeline parallel is not supported"
    # With DBO / ubatching or context parallel a layer sees a slice of the batch, but the kernel reads tok[0:T].
    pc = vllm_config.parallel_config
    assert not pc.use_ubatching, "dsv41_intervene: micro-batching (DBO / ubatch) is not supported"
    assert pc.prefill_context_parallel_size == 1 and pc.decode_context_parallel_size == 1, \
        "dsv41_intervene: context parallel is not supported"
    st = arms.init_state(vllm_config, torch.device("cuda", torch.cuda.current_device()))
    arms.STATE = st
    for slot, L in enumerate(st.installed):
        succ = self.layers[L + 1]
        assert self.start_layer <= L and L + 1 < self.end_layer, f"output {L} has no local successor"
        assert succ.engram is None, f"successor of output {L} is an Engram layer"
        succ.forward = _make_forward(slot, st.vslot(slot), succ.forward)
    ops.warm_kernels(st)
    # Under "vllm." so vLLM's handler prints it: a logger outside the "vllm" tree has no handler and drops INFO
    # (found on the first GPU boot: the line never reached the serve log, which the install gate greps).
    init_logger("vllm.dsv41_intervene.patch").info(
        "dsv41_intervene: fusion off, %d edit slots on outputs %s (capture %s), %d arms, impl %s, mhc_stream %s",
        len(st.installed), st.installed, st.capture_layers, st.n_arms, ops.IMPL,
        self.layers[st.installed[0] + 1].mhc_stream is not None)


def _make_forward(slot: int, vslot: int, orig: Callable) -> Callable:
    """Return a ``DeepseekV4DecoderLayer.forward`` replacement for the successor of installed slot ``slot``.

    Materializes the previous layer's output stack with ``mhc_post_tilelang``, edits it in place with
    ``ops.edit_stack_(post, positions, pre_mix, slot, vslot, arms.STATE)``, then calls ``orig`` with ``residual=None``
    and the carried ``pre_mix``. ``vslot`` is the index into the capture layers, or -1.
    """
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

    from . import arms, ops

    def forward(x, positions, input_ids, pre_mix=None, post_mix=None, res_mix=None, residual=None,
                engram_hashes=None, engram_mask=None, *, capture_previous_aux=False, mega_gate_metadata=None):
        # Holds because fusion is off: the previous layer reduced and finalized its own output.
        assert residual is not None and isinstance(x, torch.Tensor)
        post = mhc_post_tilelang(x, residual, post_mix, res_mix)  # output stack of layer L, [T, 4, 5120] bf16
        ops.edit_stack_(post, positions, pre_mix, slot, vslot, arms.STATE)
        # residual=None with a 3-D x takes the plain mhc_pre branch with the carried pre_mix; it picks the
        # side-stream (overlap) variant by itself inside FULL graphs of <= 16 tokens.
        out = orig(post, positions, input_ids, pre_mix, None, None, None, engram_hashes, engram_mask,
                   capture_previous_aux=False, mega_gate_metadata=mega_gate_metadata)
        if capture_previous_aux:  # static per layer (aux layers are fixed at init)
            out = (*out[:5], post.mean(dim=1))
        return out

    forward.dsv41_slot = slot
    return forward
