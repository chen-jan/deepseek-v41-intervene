"""API-key check on every route of the engine's HTTP server except ``/health``.

vLLM 0.31 checks ``VLLM_API_KEY`` only on paths under ``/v1``, ``/v2``, ``/inference`` and ``/cohere``
(``entrypoints/serve/middleware/authenticate.py``). With ``VLLM_SERVER_DEV_MODE=1`` the dev routes
(``/collective_rpc``, ``/reset_prefix_cache``, ``/sleep``, ...) and ``/metrics`` would then be open to every process on
the node, and on a shared node other users' jobs share its loopback. ``scripts/serve_engine.sh`` therefore adds
``--middleware dsv41_intervene.auth.require_key`` (a frontend flag, outside vLLM's config hash).

Fails closed: without ``VLLM_API_KEY`` only ``/health`` answers. Imports nothing from vLLM or torch.
"""

import hmac
import os

OPEN_PATHS = frozenset({"/health"})


def _authorized(header: str | None, key: str) -> bool:
    if not key or not header:
        return False
    try:
        got = header.encode("latin-1")  # starlette decodes header bytes as latin-1
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(got, f"Bearer {key}".encode("utf-8"))


async def require_key(request, call_next):
    """Starlette ``http`` middleware: 401 unless the path is open or ``Authorization: Bearer $VLLM_API_KEY``."""
    if request.scope.get("path", "") in OPEN_PATHS or _authorized(request.headers.get("authorization"),
                                                                     os.environ.get("VLLM_API_KEY", "")):
        return await call_next(request)
    from starlette.responses import JSONResponse

    return JSONResponse({"error": "Unauthorized"}, status_code=401)
