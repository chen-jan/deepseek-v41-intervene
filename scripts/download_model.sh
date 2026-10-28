#!/bin/bash
# Download the DeepSeek V4.1 Flash checkpoint (476 GiB, 48 safetensors shards) to $DSV41_MODEL_DIR.
# Put it on fast local disk (NVMe) of the GPU node if you can: loading is disk- and page-cache-bound.
#
#   bash scripts/download_model.sh       # needs DSV41_ROOT, DSV41_MODEL_DIR; uses the `hf` CLI from the venv
#   ENCODING_ONLY=1 bash scripts/download_model.sh   # only encoding/, tokenizer and config files (a few MB),
#                                                    # e.g. on a laptop for the provider tests
set -euo pipefail
. "$(dirname "$0")/common.sh" || exit 2
need DSV41_MODEL_DIR || exit 2
REPO_ID=${DSV41_HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
HF=${HF:-hf}
command -v "$HF" >/dev/null || { echo "error: hf CLI not found (pip install huggingface_hub, or set HF=...)" >&2; exit 2; }
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$DSV41_MODEL_DIR"
echo "[$(date "+%F %T")] start on $(hostname), free: $(df -h "$DSV41_MODEL_DIR" | tail -1)"
t0=$SECONDS
if [ "${ENCODING_ONLY:-0}" = 1 ]; then
  "$HF" download "$REPO_ID" --include "encoding/*" --include "tokenizer*" --include "config.json" \
    --local-dir "$DSV41_MODEL_DIR"
else
  "$HF" download "$REPO_ID" --local-dir "$DSV41_MODEL_DIR" --max-workers 16
fi
echo "[$(date "+%F %T")] done in $((SECONDS - t0)) s; $(du -sh "$DSV41_MODEL_DIR" | cut -f1)"
