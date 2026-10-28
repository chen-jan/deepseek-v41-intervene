"""Dry install check against the real vLLM 0.31. CPU only: no model, no GPU, no server.

Each case runs in a fresh interpreter, because ``patch.apply()`` changes vLLM module globals for the whole process:
- with ``DSV41_ENGINE=1``, importing ``dsv41_intervene.rpc`` (what ``--worker-extension-cls`` does) patches the model
  module, installs ``ArmModelState``, and ``EditExt`` passes vLLM's worker-extension clash check;
- without it, the same import patches nothing (the plain server shares the venv);
- with ``DSV41_MHC_OVERLAP=0`` as well, the side-stream switch is patched off too (fallback);
- the ``serve_engine.sh`` flags parse with vLLM's own ``vllm serve`` parser.

Run: ``python -m pytest -q tests/test_install_dry.py`` (skipped without vLLM), or
``DSV41_ENGINE=1 python tests/test_install_dry.py engine`` for one case with a printout.
"""

import importlib.util
import json
import os
import subprocess
import sys

ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RPC_METHODS = {"dsv41_installed", "dsv41_load_arms", "dsv41_stats", "dsv41_capture_dump", "dsv41_set_global_arm"}
SERVE_FLAGS = ["--host", "127.0.0.1", "--port", "8101", "--served-model-name", "dsv41", "--tensor-parallel-size", "4",
               "--gpu-memory-utilization", "0.92", "--max-model-len", "131072", "--enable-prefix-caching",
               "--max-num-seqs", "512", "--max-num-batched-tokens", "32768",
               "--worker-extension-cls", "dsv41_intervene.rpc.EditExt", "--enable-prompt-tokens-details",
               "--middleware", "dsv41_intervene.auth.require_key"]
EAGER_FLAGS = ["--enforce-eager", "--kernel-config", '{"enable_flashinfer_autotune": false}']

def _engine_case() -> dict:
    import inspect

    import vllm.models.deepseek_v41.nvidia.model as M
    import vllm.models.deepseek_v41.nvidia.vl_model as VL
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.worker.gpu_worker import Worker

    orig_supports, orig_init = M.supports_mhc_all_reduce, M.DeepseekV4Model.__init__
    orig_overlap = M.supports_mhc_overlap
    layer_params = list(inspect.signature(M.DeepseekV4DecoderLayer.forward).parameters)[1:]
    ext = resolve_obj_by_qualname("dsv41_intervene.rpc.EditExt")  # as worker_base.py does; runs plugin.register()
    from dsv41_intervene import arms, ops, patch

    assert patch._applied and M.supports_mhc_all_reduce is not orig_supports
    assert M.supports_mhc_all_reduce(None) is False
    if os.environ.get("DSV41_MHC_OVERLAP") == "0":
        assert M.supports_mhc_overlap is not orig_overlap and M.supports_mhc_overlap(None) is False
    else:
        assert M.supports_mhc_overlap is orig_overlap
    assert M.DeepseekV4Model.__init__ is patch._model_init and patch._orig_init is orig_init
    init_params = inspect.signature(orig_init).parameters
    assert list(init_params) == ["self", "vllm_config", "prefix"], list(init_params)
    assert init_params["vllm_config"].kind is inspect.Parameter.KEYWORD_ONLY
    cls_llm = M.DeepseekV41LLMForCausalLM.get_model_state_cls()
    cls_vl = VL.DeepseekV41ForCausalLM.get_model_state_cls()
    assert cls_llm is cls_vl and issubclass(cls_llm, DeepseekV41ModelState) and cls_llm.__name__ == "ArmModelState"
    for name in ("__init__", "add_request", "prepare_inputs", "prepare_dummy_inputs"):
        got = list(inspect.signature(getattr(cls_llm, name)).parameters)
        want = list(inspect.signature(getattr(DeepseekV41ModelState, name)).parameters)
        assert got == want, (name, got, want)
    # vLLM's own clash check (v1/worker/worker_base.py) before it appends the extension to Worker.__bases__
    clashes = [a for a in dir(ext) if not a.startswith("__") and hasattr(Worker, a)]
    assert not clashes, clashes
    calls = {a for a in dir(ext) if not a.startswith("__") and callable(getattr(ext, a))}
    assert calls == RPC_METHODS, calls
    fwd = patch._make_forward(0, -1, lambda *a, **k: None)
    assert list(inspect.signature(fwd).parameters) == layer_params, layer_params
    patch.apply()  # idempotent
    assert M.DeepseekV4Model.__init__ is patch._model_init
    # names the patch, the arms and the GPU tests import lazily
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank  # noqa: F401
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang  # noqa: F401
    from vllm.models.deepseek_v41.nvidia.ops.mhc import (  # noqa: F401
        MHC_OVERLAP_MAX_TOKENS,
        mhc_pre_delayed_overlap,
        mhc_shifted_post_pre,
    )
    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool  # noqa: F401

    assert ops.IMPL == "triton" and ops.triton is not None
    assert arms.STATE is None  # allocated only inside DeepseekV4Model.__init__
    # the guards in patch._model_init and the install check's fusion evidence read these vLLM names
    from vllm.config import ParallelConfig

    pc = ParallelConfig()
    assert pc.use_ubatching is False and pc.prefill_context_parallel_size == 1 == pc.decode_context_parallel_size
    from dsv41_intervene import rpc

    assert rpc._all_reduce_mhc_built() is False  # module imported, fused kernel not built
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    assert isinstance(getattr(FusedMoEConfig, "use_deferred_moe_finalize", None), property)

    class _Uninstalled(ext):
        rank, model_runner = 3, None

    assert _Uninstalled().dsv41_installed() == {"rank": 3, "installed": False, "layers": [], "arms_sha": ""}
    return {"layer_params": layer_params, "rpc": sorted(calls), "overlap_max_tokens": MHC_OVERLAP_MAX_TOKENS}


def _plain_case() -> dict:
    import vllm.models.deepseek_v41.nvidia.model as M

    orig_supports, orig_init = M.supports_mhc_all_reduce, M.DeepseekV4Model.__init__
    orig_state = M.DeepseekV41LLMForCausalLM.get_model_state_cls()
    import dsv41_intervene.rpc  # noqa: F401
    from dsv41_intervene import patch

    assert not patch._applied
    assert M.supports_mhc_all_reduce is orig_supports and M.DeepseekV4Model.__init__ is orig_init
    assert M.DeepseekV41LLMForCausalLM.get_model_state_cls() is orig_state
    return {"state_cls": orig_state.__name__}


def _flags_case() -> dict:
    import vllm.platforms
    from vllm.platforms.cuda import CudaPlatform

    vllm.platforms.current_platform = CudaPlatform()  # a CPU node detects no device; the parser needs one
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    out = {}
    for name, extra in (("graph", []), ("eager", EAGER_FLAGS)):
        args = make_arg_parser(FlexibleArgumentParser()).parse_args(["/model", *SERVE_FLAGS, *extra])
        ea = AsyncEngineArgs.from_cli_args(args)
        assert ea.worker_extension_cls == "dsv41_intervene.rpc.EditExt" and ea.tensor_parallel_size == 4
        assert ea.max_num_batched_tokens == 32768 and ea.enable_prefix_caching is True
        assert args.enable_prompt_tokens_details is True and args.port == 8101 and args.host == "127.0.0.1"
        assert args.middleware == ["dsv41_intervene.auth.require_key"]
        if name == "eager":
            kc = ea.kernel_config
            autotune = kc.get("enable_flashinfer_autotune") if isinstance(kc, dict) else kc.enable_flashinfer_autotune
            assert ea.enforce_eager is True and autotune is False, kc
        out[name] = "ok"
    # vLLM's own loader (entrypoints/serve/middleware/register.py) must take it as an http function middleware
    import importlib
    import inspect

    mod, _, obj = args.middleware[0].rpartition(".")
    assert inspect.iscoroutinefunction(getattr(importlib.import_module(mod), obj))
    return out


CASES = {"engine": _engine_case, "engine_no_overlap": _engine_case, "plain": _plain_case, "flags": _flags_case}


def _run(case: str) -> dict:
    env = {**os.environ, "PYTHONPATH": ENGINE_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("DSV41_ENGINE", None)
    env.pop("DSV41_MHC_OVERLAP", None)
    if case.startswith("engine"):
        env["DSV41_ENGINE"] = "1"
    if case == "engine_no_overlap":
        env["DSV41_MHC_OVERLAP"] = "0"
    proc = subprocess.run([sys.executable, __file__, case], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


if __name__ != "__main__":
    import pytest

    @pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="needs vLLM")
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_dry_install(case):
        print(case, _run(case))


if __name__ == "__main__":  # one case in this interpreter (pytest is not on the plain venv's path)
    print(json.dumps(CASES[sys.argv[1]]()))
