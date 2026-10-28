#!/bin/bash
# Build the one venv this repo uses: plain vLLM 0.31.0 (torch 2.13 cu130) + FlashInfer's prebuilt kernel cache.
# The plain server and the edit engine share it. Never `pip install` this repo into it: the engine is put on
# PYTHONPATH only by scripts/serve_engine.sh, so the plain server can never be patched (see docs/INTERFACE.md).
#
#   bash scripts/build_env.sh            # needs DSV41_ROOT (venv goes to $DSV41_VENV, default $DSV41_ROOT/envs/vllm031)
#
# Offline GPU hosts: download the two flashinfer_jit_cache*-0.7.0.post1+cu130 wheels elsewhere and point
# DSV41_FLASHINFER_WHEELS at the folder that holds them.
set -euo pipefail
. "$(dirname "$0")/common.sh" || exit 2
UV=${UV:-uv}
command -v "$UV" >/dev/null || { echo "error: uv not found (set UV=/path/to/uv)" >&2; exit 2; }
[ -d "$DSV41_VENV" ] || "$UV" venv --python 3.12 "$DSV41_VENV"
. "$DSV41_VENV/bin/activate"
echo "[$(date "+%F %T")] vllm"
"$UV" pip install "vllm==0.31.0" --torch-backend=cu130
echo "[$(date "+%F %T")] flashinfer-jit-cache"
if [ -n "${DSV41_FLASHINFER_WHEELS:-}" ]; then
  "$UV" pip install --no-deps "$DSV41_FLASHINFER_WHEELS"/flashinfer_jit_cache*.whl
else
  # The measured env had both flashinfer_jit_cache and flashinfer_jit_cache_sm100a (B200 kernels), 0.7.0.post1+cu130.
  "$UV" pip install "flashinfer-jit-cache==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/cu130 \
    || echo "WARN: flashinfer-jit-cache not installed; FlashInfer will JIT-compile its kernels with nvcc (slow first boot)"
  "$UV" pip install "flashinfer-jit-cache-sm100a==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/cu130 \
    || echo "WARN: flashinfer-jit-cache-sm100a not installed (only matters on B200)"
fi
# fastsafetensors was in the measured env, but do not use it as the loader: it stages a whole 101.5 GB Engram shard
# on GPU 0 and runs out of memory. pytest is for the tests only.
"$UV" pip install fastsafetensors pytest
python - <<"PY"
import importlib.metadata as md
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm.general_plugins:", [e.name for e in md.entry_points(group="vllm.general_plugins")], "(should be [])")
PY
echo "[$(date "+%F %T")] build done"
