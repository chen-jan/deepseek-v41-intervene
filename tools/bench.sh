#!/bin/bash
# `vllm bench serve` against a running server: native (the plain server, :8100), sham and arms (the engine, :8101,
# --extra-body from tools/arms.py request_fields). Concurrency 1/16/64/128/256, random 8192 -> 1500 tokens,
# --num-prompts 2c (8 for c=1). OPENAI_API_KEY is read from $DSV41_KEY_FILE inside this script and never printed.
# Run it on the GPU node (inside your allocation), e.g.
#   bash tools/bench.sh native                       # :8100, no extra body
#   bash tools/bench.sh sham                         # :8101, arm 0
#   ARM=3 ARMS_SHA=<sha> bash tools/bench.sh arm     # :8101, one arm
#   ARM=7 ARMS_SHA=<sha> bash tools/bench.sh mix     # 64 sham + 64 arm 7
# The random dataset with the default --seed 0 sends the same first N prompts in every run and prefix caching is on,
# so later rungs of a ladder hit blocks cached by earlier ones (the native baseline: 56-60% hits at c=128/256). To
# match native's cache history: run `sham` FIRST on a fresh engine with the native ladder; `arm` runs the ladder
# 1 16 64 128 under its own salt (cold for that salt); `mix` uses --seed 7 for both halves (cold for both).
# Results: $DSV41_LOG_DIR/engine/bench_<name>_c<c>.{json,log}; a one-line summary per run on stdout.
set -uo pipefail
. "$(dirname "$0")/../scripts/common.sh" || exit 2
need DSV41_MODEL_DIR || exit 2
MODE=${1:?usage: bench.sh native|sham|arm|mix}
OUT=$DSV41_LOG_DIR/engine; mkdir -p "$OUT"
TAG=${TAG:-$(date +%m%d_%H%M)}
case $MODE in
  native) PORT=${PORT:-8100}; CONC=${CONC:-1 16 64 128 256} ;;
  sham) PORT=${PORT:-8101}; CONC=${CONC:-1 16 64 128 256} ;;
  arm) PORT=${PORT:-8101}; CONC=${CONC:-1 16 64 128}; : "${ARM:?ARM=<k> needed}" "${ARMS_SHA:?ARMS_SHA needed}" ;;
  mix) PORT=${PORT:-8101}; CONC=64; SEED=${SEED:-7}; : "${ARM:?ARM=<k> needed}" "${ARMS_SHA:?ARMS_SHA needed}" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
python -c "import urllib.request as u; assert u.urlopen('http://127.0.0.1:$PORT/health', timeout=5).status == 200" \
  || { echo "server on :$PORT not healthy"; exit 1; }
fields() {  # request fields for arm $1 as one JSON object (vllm_xargs + cache_salt), from tools/arms.py
  python "$REPO/tools/arms.py" fields --arm "$1" --arms-sha "${ARMS_SHA:-}"
}
[ -s "$DSV41_KEY_FILE" ] || { echo "no key at $DSV41_KEY_FILE"; exit 2; }
export OPENAI_API_KEY=$(cat "$DSV41_KEY_FILE")

run() {  # name concurrency extra_body_json
  local name=$1 c=$2 body=$3 n
  n=$(( c < 4 ? 8 : 2 * c ))
  local extra=(); [ "$body" != "{}" ] && extra=(--extra-body "$body")
  vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model dsv41 --tokenizer "$DSV41_MODEL_DIR" \
    --dataset-name random --random-input-len 8192 --random-output-len 1500 --ignore-eos --num-prompts $n \
    --seed ${SEED:-0} \
    --max-concurrency $c "${extra[@]}" --save-result --result-dir $OUT --result-filename ${name}_c${c}.json \
    > $OUT/${name}_c${c}.log 2>&1 || { echo "bench $name c=$c failed"; tail -5 $OUT/${name}_c${c}.log; }
  grep -E "Output token throughput|Mean TTFT|Mean TPOT" $OUT/${name}_c${c}.log | sed "s/^/  $name c=$c /"
}

echo "[$(date "+%F %T")] bench $MODE on :$PORT, conc $CONC, tag $TAG"
case $MODE in
  native) for c in $CONC; do run bench_native_$TAG $c "{}"; done ;;
  sham) for c in $CONC; do run bench_sham_$TAG $c "{}"; done ;;
  arm) B=$(fields $ARM); for c in $CONC; do run bench_arm${ARM}_$TAG $c "$B"; done ;;
  mix) B=$(fields $ARM)
       run bench_mix_sham_$TAG 64 "{}" & P1=$!
       run bench_mix_arm${ARM}_$TAG 64 "$B" & P2=$!
       wait $P1 $P2 ;;
esac
echo "[$(date "+%F %T")] BENCH DONE"
