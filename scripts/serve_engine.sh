#!/bin/bash
# The edit engine: the plain server's flags (scripts/serve_plain.sh) plus additions that stay outside vLLM's compile
# hash, so the autotune/compile caches of the plain server stay valid:
#   DSV41_ENGINE=1 (turns the patch on), VLLM_SERVER_DEV_MODE=1 (the /collective_rpc route),
#   --worker-extension-cls dsv41_intervene.rpc.EditExt (the dsv41_* RPC methods; importing it applies the patch),
#   --middleware dsv41_intervene.auth.require_key (the API key on every route but /health: vLLM itself guards only
#   /v1..., and the dev routes would otherwise be open to every process that can reach the port),
#   --host 127.0.0.1 --port 8101 (control routes stay local; tools/proxy.py can serve other machines on :8100).
#
# Run inside your 4-GPU allocation (examples/slurm/ has a template). Give the step at least 32 CPUs: with 24, vLLM's
# NUMA binding gave TP rank 0 16 CPUs and left ranks 1-3 sharing 8, and loading took ~2x longer.
#
# Fallbacks (set by the caller):
#   DSV41_EAGER=1          one execution path, no CUDA graphs, autotune off (~10% of native throughput)
#   DSV41_EDIT_IMPL=triton the Triton edit kernel instead of the torch op. Default here is torch: every GPU boot so far
#                          used torch, and the Triton kernel differed from it by up to 4 ulp on GPU (test asks <= 1)
#   DSV41_MHC_OVERLAP=0    no mHC side stream (if only the FULL-overlap graph path misbehaves)
#   VLLM_USE_BREAKABLE_CUDAGRAPH=0  (if only the PIECEWISE path misbehaves)
set -uo pipefail
. "$(dirname "$0")/common.sh" || exit 2
need DSV41_MODEL_DIR || exit 2
[ -s "$DSV41_KEY_FILE" ] || { echo "no key at $DSV41_KEY_FILE (run scripts/make_key.sh)"; exit 2; }
export VLLM_API_KEY=$(cat "$DSV41_KEY_FILE")
# The engine package comes from this checkout via PYTHONPATH (never pip-installed into the shared venv).
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1
export DSV41_ENGINE=1 VLLM_SERVER_DEV_MODE=1 DSV41_EDIT_LAYERS=${DSV41_EDIT_LAYERS:-12,14-38} \
       DSV41_N_ARMS=${DSV41_N_ARMS:-16} DSV41_CAPTURE_LAYERS=${DSV41_CAPTURE_LAYERS:-12,15,18,22,25,26,27,30,34} \
       DSV41_COEF_ROWS=${DSV41_COEF_ROWS:-1048576} DSV41_VEC_ROWS=${DSV41_VEC_ROWS:-32768} \
       DSV41_EDIT_IMPL=${DSV41_EDIT_IMPL:-torch}
mkdir -p "$DSV41_CAPTURE_DIR"
EXTRA=()
if [ "${DSV41_EAGER:-0}" = 1 ]; then
  EXTRA=(--enforce-eager --kernel-config '{"enable_flashinfer_autotune": false}')
fi
PORT=${PORT:-8101}
OUT=$DSV41_LOG_DIR/engine; mkdir -p "$OUT"
LOG=$OUT/serve_engine_$(date +%m%d_%H%M).log
echo "[$(date "+%F %T")] engine on $(hostname) 127.0.0.1:$PORT, layers $DSV41_EDIT_LAYERS, arms $DSV41_N_ARMS," \
     "impl $DSV41_EDIT_IMPL, eager ${DSV41_EAGER:-0}, mhc_overlap ${DSV41_MHC_OVERLAP:-1}, log $LOG"
exec vllm serve "$DSV41_MODEL_DIR" --host 127.0.0.1 --port "$PORT" --served-model-name dsv41 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.92 --max-model-len 131072 \
  --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 32768 \
  --worker-extension-cls dsv41_intervene.rpc.EditExt --enable-prompt-tokens-details \
  --middleware dsv41_intervene.auth.require_key "${EXTRA[@]}" > "$LOG" 2>&1
