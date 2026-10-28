"""aiohttp proxy ``0.0.0.0:8100 -> 127.0.0.1:8101``. Rules: docs/INTERFACE.md section 4.

Forwards only the four routes below, with the client's headers and body unchanged, and streams the answer back. Every
route except ``/health`` needs the server's API key (read from ``DSV41_KEY_FILE`` here, never printed). Completions:
- 400 with any ``vllm_xargs.capture``, a non-int ``arm``, or ``arm != 0`` without ``cache_salt``;
- 400 when a ``cache_salt`` is not ``request_fields(arm, edit_from, edit_to, arms_sha=S)``'s salt for some arms sha S
  that ``tools/client.py load_arms`` recorded for this engine boot (``$DSV41_ARMS_DIR/LOADED_SHAS``), so no client can
  write or reuse KV under another arm's salt or a stale sha;
- 503 for every arm but the global one while ``tools/client.py set_global_arm`` has set a global arm k >= 0 for this
  boot (``$DSV41_ARMS_DIR/GLOBAL_ARM``; block protocol: unsalted sham traffic pauses in non-sham blocks).
Everything else gets 404. The engine boot is read once at startup (``dsv41_installed`` on rank 0), so restart the
proxy after every engine reboot; until then every salted request is refused.

    python tools/proxy.py --listen 0.0.0.0:8100 --upstream 127.0.0.1:8101
"""

import argparse
import hmac
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsv41_intervene import MAX_EDIT_TO, env_path  # noqa: E402
from tools.arms import GLOBAL_ARM, LOADED_SHAS, arms_dir, read_state, salt  # noqa: E402

ALLOWED = {("POST", "/v1/completions"), ("GET", "/v1/models"), ("GET", "/health"), ("POST", "/tokenize")}
OPEN = {("GET", "/health")}
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "upgrade", "te", "trailer",
       "proxy-authorization", "proxy-connection", "content-encoding"}


def check_completion(body, shas: set[str] | None = None, global_arm: int = -1) -> tuple[int, str] | None:
    """``(status, reason)`` for a completion body the proxy must refuse, else None. ``shas`` are the arms shas loaded
    in this engine boot (None skips the salt check; tests only); ``global_arm`` is the engine's global arm (-1 = off)."""
    if not isinstance(body, dict):
        return 400, "body must be a JSON object"
    xargs = body.get("vllm_xargs")
    xargs = {} if xargs is None else xargs
    if not isinstance(xargs, dict):
        return 400, "vllm_xargs must be an object"
    if "capture" in xargs:
        return 400, "capture requests are not accepted on this port"
    arm = xargs.get("arm", 0)
    if type(arm) is not int:
        return 400, "arm must be an int"
    salt_in = body.get("cache_salt")
    if arm != 0 and not salt_in:
        return 400, "a request with arm != 0 needs its cache_salt"
    if salt_in is not None and shas is not None:
        p0, p1 = xargs.get("edit_from", 1), xargs.get("edit_to", MAX_EDIT_TO)
        if type(p0) is not int or type(p1) is not int:
            return 400, "edit_from / edit_to must be ints"
        if not any(hmac.compare_digest(str(salt_in), salt(s, arm, p0, p1)) for s in shas):
            return 400, "cache_salt does not match this arm and window under a loaded arms file"
    if global_arm >= 0 and arm != global_arm:
        return 503, "the engine is in a global-arm block; only that block's traffic is accepted"
    return None


class EngineState:
    """Loaded arms shas and the global arm for one engine boot, re-read when the state files change."""

    def __init__(self, boot: str, root: Path | None = None):
        self.boot, self.root = boot, root or arms_dir()
        self._cache: dict[str, tuple] = {}

    def _values(self, name: str) -> list[str]:
        path = self.root / name
        try:
            st = path.stat()
            key = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            key = None
        hit = self._cache.get(name)
        if hit is None or hit[0] != key:
            hit = (key, read_state(path, self.boot) if key else [])
            self._cache[name] = hit
        return hit[1]

    def shas(self) -> set[str]:
        return set(self._values(LOADED_SHAS))

    def global_arm(self) -> int:
        vals = self._values(GLOBAL_ARM)
        return int(vals[-1]) if vals else -1


def _authorized(header: str, want: str) -> bool:
    try:
        return hmac.compare_digest(header.encode("utf-8", "surrogateescape"), want.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def make_app(upstream: str, key: str | None = None, state: EngineState | None = None):
    """Build the ``aiohttp.web.Application`` (streaming passthrough of allowed routes, 404 otherwise). Without
    ``state`` the proxy reads the engine boot from ``upstream`` at startup."""
    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

    base = upstream if upstream.startswith("http") else f"http://{upstream}"
    want = f"Bearer {key}" if key else None

    async def on_startup(app):
        # aiohttp's default connector caps a session at 100 connections; the engine runs up to 512 sequences
        app["session"] = ClientSession(connector=TCPConnector(limit=0),
                                       timeout=ClientTimeout(total=None, sock_read=None), auto_decompress=False)
        if app.get("state") is None:
            hdr = {"Authorization": want} if want else {}
            async with app["session"].post(base + "/collective_rpc", json={"method": "dsv41_installed"},
                                           headers=hdr) as r:
                res = (await r.json())["results"]
            boot = sorted(res, key=lambda x: x.get("rank", 0))[0].get("boot")
            if not boot:
                raise RuntimeError("engine reports no boot id (dsv41_installed)")
            app["state"] = EngineState(boot)
            print(f"proxy: engine boot {boot}, state files in {app['state'].root}", flush=True)

    async def on_cleanup(app):
        await app["session"].close()

    async def handle(request: web.Request) -> web.StreamResponse:
        route = (request.method, request.path)
        if route not in ALLOWED:
            return web.json_response({"error": "not found"}, status=404)
        if want and route not in OPEN and not _authorized(request.headers.get("Authorization", ""), want):
            return web.json_response({"error": "unauthorized"}, status=401)
        raw = await request.read()
        if route == ("POST", "/v1/completions"):
            st = request.app["state"]
            try:
                err = check_completion(json.loads(raw), st.shas(), st.global_arm())
            except ValueError:
                err = (400, "body is not JSON")
            if err:
                return web.json_response({"error": {"message": err[1], "code": err[0]}}, status=err[0])
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        async with request.app["session"].request(request.method, base + request.path_qs, data=raw,
                                                  headers=headers) as up:
            resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items()
                                                                 if k.lower() not in HOP})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    app = web.Application(client_max_size=256 * 2**20)
    app["state"] = state
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def main() -> None:
    """``--listen HOST:PORT --upstream HOST:PORT``."""
    from aiohttp import web

    ap = argparse.ArgumentParser(description="engine proxy (only /v1 traffic reaches vLLM)")
    ap.add_argument("--listen", default="0.0.0.0:8100")
    ap.add_argument("--upstream", default="127.0.0.1:8101")
    ap.add_argument("--key-file", default=env_path("DSV41_KEY_FILE", ".serve_key", required=False),
                    help="'none' = no key check (tests only)")
    args = ap.parse_args()
    if not args.key_file:
        ap.error("set DSV41_KEY_FILE or DSV41_ROOT, or pass --key-file (none = no key check, tests only)")
    key = None if args.key_file == "none" else Path(args.key_file).read_text().strip()
    host, _, port = args.listen.rpartition(":")
    web.run_app(make_app(args.upstream, key), host=host, port=int(port), access_log=None)


if __name__ == "__main__":
    main()
