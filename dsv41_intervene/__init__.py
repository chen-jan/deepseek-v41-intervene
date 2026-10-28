"""Causal edits (project / matched shift / add) at DeepSeek V4.1 mHC layer outputs, per request, CUDA-graph safe.

Contract: ``docs/INTERFACE.md``. Shared constants live here; this module must import without vLLM or torch.
"""

import os

HC = 4
HIDDEN = 5120
MAX_EDIT_TO = 2**31 - 1
MAX_CAP_POS = 4096
PAD = -1

# Columns of EditState.tok ([MAX_T, 5] int32).
TOK_ARM, TOK_P0, TOK_P1, TOK_CROW, TOK_VROW = range(5)
TOK_COLS = 5

DEFAULT_EDIT_LAYERS = "12,14-38"
DEFAULT_CAPTURE_LAYERS = "12,15,18,22,25,26,27,30,34"
DEFAULT_N_ARMS = 16
DEFAULT_COEF_ROWS = 1 << 20
DEFAULT_VEC_ROWS = 1 << 15
# Successors of outputs 0 and 13 are Engram layers; output 39 has no successor.
FORBIDDEN_LAYERS = frozenset({0, 13})
NUM_LAYERS = 40


def parse_layers(spec: str) -> list[int]:
    """Parse ``"12,14-38"`` into a sorted, de-duplicated list of 0-based output layers."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def env_int(name: str, default: int) -> int:
    """Integer env var with a default."""
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def env_path(var: str, sub: str, required: bool = True) -> str | None:
    """``$var``, else ``$DSV41_ROOT/<sub>``. When neither is set: a clear error, or None if not ``required``."""
    val = os.environ.get(var)
    if val:
        return val
    root = os.environ.get("DSV41_ROOT")
    if root:
        return os.path.join(root, sub)
    if required:
        raise RuntimeError(f"set {var} (or DSV41_ROOT, which defaults it to $DSV41_ROOT/{sub}); see .env.example")
    return None


def edit_layers() -> list[int]:
    """Installed output layers from ``DSV41_EDIT_LAYERS``, validated."""
    layers = parse_layers(os.environ.get("DSV41_EDIT_LAYERS", DEFAULT_EDIT_LAYERS))
    bad = [L for L in layers if L in FORBIDDEN_LAYERS or not 0 <= L < NUM_LAYERS - 1]
    if bad:
        raise ValueError(f"DSV41_EDIT_LAYERS has outputs that cannot take the edit: {bad}")
    return layers


def capture_layers(installed: list[int]) -> list[int]:
    """Vector-capture layers from ``DSV41_CAPTURE_LAYERS``; each must be installed."""
    layers = parse_layers(os.environ.get("DSV41_CAPTURE_LAYERS", DEFAULT_CAPTURE_LAYERS))
    missing = sorted(set(layers) - set(installed))
    if missing:
        raise ValueError(f"DSV41_CAPTURE_LAYERS not installed: {missing}")
    return layers
