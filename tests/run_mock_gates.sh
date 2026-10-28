#!/bin/bash
# Dry run of the whole gate suite (tools/gates.py) against tests/mock_engine.py and tools/proxy.py, on CPU.
# No GPU, no model, no real key: synthetic token ids (tools/build_data.py --synthetic) and a throwaway key file.
# It checks the gate code and the HTTP/RPC plumbing, not the engine's numerics.
#
#   bash tests/run_mock_gates.sh          # PYTHON=... to pick an interpreter (needs numpy, torch, safetensors, aiohttp)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
R=$(mktemp -d "${TMPDIR:-/tmp}/dsv41_gates_mock.XXXXXX")
mkdir -p "$R/capture" "$R/arms" "$R/logs/engine"
head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$R/key"; chmod 600 "$R/key"
export PYTHONDONTWRITEBYTECODE=1 DSV41_ROOT=$R DSV41_KEY_FILE=$R/key DSV41_CAPTURE_DIR=$R/capture \
  DSV41_ARMS_DIR=$R/arms DSV41_XCHECK_DIR=$R/xcheck DSV41_LOG_DIR=$R/logs DSV41_DATA_DIR=$R/data \
  DSV41_XCHECK_PROMPTS=3 DSV41_OUTLIER_PROMPTS=3 DSV41_OUTLIER_GROUP_ROWS=1500  # small vec captures; 2 dumps per arm
$PY tools/build_data.py --synthetic --out "$R/data" > "$R/data.out" || { cat "$R/data.out"; exit 1; }
$PY tests/mock_engine.py --port 18101 --key-file "$R/key" --capture-dir "$R/capture" > "$R/mock.log" 2>&1 & M=$!
X=
trap 'kill $M $X 2>/dev/null' EXIT
sleep 3
$PY tools/client.py --url http://127.0.0.1:18101 floor --data "$R/data/floor_prompts.json" --out "$R/floor_mock.json" \
  > "$R/floor.out" 2>&1 || cat "$R/floor.out"
echo "dsv41_intervene: fusion off, 26 edit slots on outputs [12, 14, ...] (mock log)" > "$R/engine.log"
# as on a GPU boot: every gate but proxy first, then start the proxy, then --only proxy
$PY tools/gates.py --url http://127.0.0.1:18101 --floor "$R/floor_mock.json" --engine-log "$R/engine.log" \
  --out "$R/gates" --keep-going ${GATES_ARGS:-}
rc1=$?
$PY tools/proxy.py --listen 127.0.0.1:18100 --upstream 127.0.0.1:18101 --key-file "$R/key" > "$R/proxy.log" 2>&1 & X=$!
sleep 3
$PY tools/gates.py --url http://127.0.0.1:18101 --proxy-url http://127.0.0.1:18100 --floor "$R/floor_mock.json" \
  --out "$R/gates" --only proxy
rc=$(( rc1 || $? ))
for f in "$R"/gates/*.json; do $PY - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if isinstance(r, dict) and "gate" in r:
    print(r["gate"], r.get("pass"), r.get("error", ""))
PY
done
rm -rf "$R/arms" "$R/capture" "$R/xcheck"  # big files; the gate JSONs stay
echo "run folder $R rc=$rc"
exit $rc
