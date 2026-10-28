# Source me (do not run me). Shared setup for every script in this repo:
#   1. loads <repo>/.env if it exists (copy .env.example to .env and fill it in);
#   2. sources $DSV41_SETUP_HOOK if set (your site's `module load cuda/...` lines);
#   3. puts every cache and temp dir under $DSV41_CACHE, never under $HOME;
#   4. activates the venv at $DSV41_VENV if it exists.
# Scripts source it as `. scripts/common.sh || exit 2`, then call `need VAR ... || exit 2`.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi

need() {  # need VAR [VAR ...]: stop with a clear message when a variable is unset or empty
  local v
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then
      echo "error: $v is not set. Put it in $REPO/.env (see .env.example) or export it." >&2
      return 2
    fi
  done
}

need DSV41_ROOT || return 2

if [ -n "${DSV41_SETUP_HOOK:-}" ]; then
  # shellcheck disable=SC1090
  . "$DSV41_SETUP_HOOK"
fi
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
fi

export DSV41_CACHE="${DSV41_CACHE:-$DSV41_ROOT/cache}"
export DSV41_VENV="${DSV41_VENV:-$DSV41_ROOT/envs/vllm031}"
export DSV41_LOG_DIR="${DSV41_LOG_DIR:-$DSV41_ROOT/logs}"
export DSV41_KEY_FILE="${DSV41_KEY_FILE:-$DSV41_ROOT/.serve_key}"
export DSV41_ARMS_DIR="${DSV41_ARMS_DIR:-$DSV41_ROOT/arms}"
export DSV41_CAPTURE_DIR="${DSV41_CAPTURE_DIR:-$DSV41_ROOT/captures}"

C="$DSV41_CACHE"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$C/uv}" PIP_CACHE_DIR="$C/pip" XDG_CACHE_HOME="$C/xdg" \
       TRITON_CACHE_DIR="$C/triton" VLLM_CACHE_ROOT="$C/vllm" TORCHINDUCTOR_CACHE_DIR="$C/inductor" \
       FLASHINFER_WORKSPACE_BASE="$C/flashinfer" HF_HOME="${HF_HOME:-$C/hf}" CUDA_CACHE_PATH="$C/xdg/nv" \
       TMPDIR="$C/tmp" TILELANG_CACHE_DIR="$C/tilelang" TORCH_EXTENSIONS_DIR="$C/torch_extensions" \
       NUMBA_CACHE_DIR="$C/numba" MPLCONFIGDIR="$C/xdg/matplotlib"
mkdir -p "$TMPDIR" "$TILELANG_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$DSV41_LOG_DIR"
unset C

# As in every measured run: FlashInfer sampler off, spawned workers, unbuffered Python.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1

if [ -f "$DSV41_VENV/bin/activate" ]; then
  deactivate 2>/dev/null || true
  # shellcheck disable=SC1091
  . "$DSV41_VENV/bin/activate"
fi
