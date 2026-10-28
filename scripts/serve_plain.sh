#!/bin/bash
# Plain DeepSeek V4.1 Flash server: vLLM 0.31, TP4 on 4 GPUs, nothing hooked into the forward pass.
# Run it inside your 4-GPU allocation (for Slurm see examples/slurm/). It listens on $HOST:$PORT (default
# 0.0.0.0:8100) and needs the key file (scripts/make_key.sh); vLLM reads the key from VLLM_API_KEY, so it is not on
# the command line.
#
# Flags (measured on 4x B200):
# - max-num-seqs 512: KV was 3.9% full with 256 x 9.7k tokens; agent contexts reach ~50k, so 512 x 50k still fits.
# - max-num-batched-tokens 32768: at 256 concurrent, a 16k cap made prefill the bottleneck (5.8k vs 13.6k tok/s decode).
# - --enable-prompt-tokens-details: per-call cached-token counts. An API-server flag, outside the compile hash.
# Do not set VLLM_ENGINE_READY_TIMEOUT_S: it is a torch.compile hash factor and forces a recompile.
set -uo pipefail
. "$(dirname "$0")/common.sh" || exit 2
need DSV41_MODEL_DIR || exit 2
[ -s "$DSV41_KEY_FILE" ] || { echo "no key at $DSV41_KEY_FILE (run scripts/make_key.sh)"; exit 2; }
export VLLM_API_KEY=$(cat "$DSV41_KEY_FILE")
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8100}
OUT=$DSV41_LOG_DIR/serve; mkdir -p "$OUT"
LOG=$OUT/serve_plain_$(date +%m%d_%H%M).log
echo "[$(date "+%F %T")] serving on $(hostname):$PORT, log $LOG"
exec vllm serve "$DSV41_MODEL_DIR" --host "$HOST" --port "$PORT" --served-model-name dsv41 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.92 --max-model-len 131072 \
  --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 32768 --enable-prompt-tokens-details \
  "$@" > "$LOG" 2>&1
