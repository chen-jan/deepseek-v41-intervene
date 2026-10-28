#!/bin/bash
# Throughput of plain serving configs: for each config, start `vllm serve`, wait for /health, run `vllm bench serve`
# (random 8,192-token prompts -> 1,500 forced output tokens) at several concurrencies, stop the server.
# Run inside your 4-GPU allocation, with nothing else on the GPUs. Results: $DSV41_LOG_DIR/bench_configs/.
#
#   bash scripts/bench_configs.sh                       # CONFIGS="tp4", CONC="256 128 64 16 1"
#   CONFIGS="tp4 tp4_ep" bash scripts/bench_configs.sh  # tp4_ep and tp4_dspark3 were never measured
#
# For the server as scripts/serve_plain.sh runs it (and for the engine), use tools/bench.sh against a running server.
set -uo pipefail
. "$(dirname "$0")/common.sh" || exit 2
need DSV41_MODEL_DIR || exit 2
OUT=$DSV41_LOG_DIR/bench_configs; mkdir -p "$OUT"
PORT=${PORT:-8100}
# Default safetensors loader on purpose: fastsafetensors stages a whole 101.5 GB Engram shard on GPU 0 and OOMs.
COMMON="--tensor-parallel-size 4 --gpu-memory-utilization 0.92 --max-model-len 32768 --enable-prefix-caching --max-num-seqs 256 --max-num-batched-tokens 16384 --served-model-name dsv41 --host 127.0.0.1 --port $PORT"
declare -A CFG=(
  [tp4]=""
  [tp4_ep]="--enable-expert-parallel"
  [tp4_dspark3]="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":3}"
)
healthy() { python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; }
for name in ${CONFIGS:-tp4}; do
  echo "[$(date "+%F %T")] === $name: ${CFG[$name]}"
  t0=$SECONDS
  vllm serve "$DSV41_MODEL_DIR" $COMMON ${CFG[$name]} > "$OUT/serve_$name.log" 2>&1 &
  SP=$!
  until healthy; do
    sleep 10
    if ! kill -0 $SP 2>/dev/null; then echo "server $name died during boot"; tail -20 "$OUT/serve_$name.log"; continue 2; fi
    if [ $((SECONDS - t0)) -gt 2400 ]; then echo "server $name boot > 40 min"; kill $SP; continue 2; fi
  done
  echo "[$(date "+%F %T")] $name up in $((SECONDS - t0)) s"
  for c in ${CONC:-256 128 64 16 1}; do   # highest load first: the max-throughput number arrives first
    n=$(( c < 4 ? 8 : 2 * c ))
    vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --model dsv41 --tokenizer "$DSV41_MODEL_DIR" \
      --dataset-name random --random-input-len 8192 --random-output-len 1500 --ignore-eos --num-prompts $n \
      --max-concurrency $c --save-result --result-dir "$OUT" --result-filename "${name}_c${c}.json" \
      > "$OUT/bench_${name}_c${c}.log" 2>&1 \
      || { echo "bench $name c=$c failed"; tail -5 "$OUT/bench_${name}_c${c}.log"; }
    grep -E "Output token throughput|Mean TTFT|Mean TPOT" "$OUT/bench_${name}_c${c}.log" | sed "s/^/  $name c=$c /"
  done
  kill $SP; wait $SP 2>/dev/null; sleep 20
done
echo "[$(date "+%F %T")] BENCH DONE"
