"""Raw-completions provider for DeepSeek V4.1 Flash on a self-hosted vLLM (OpenAI-compatible /v1/completions).

Every prompt is rendered by DeepSeek's OFFICIAL ``encoding.py`` (shipped with the checkpoint, loaded from a configurable
path, sha256-checked, never re-implemented here) in thinking mode with a numeric reasoning effort, and sent as a raw
completion. Model output is parsed with the official ``parse_message_from_completion_text``. So fresh runs,
continuations from a cut point mid-reasoning (prefill), and sentence-level rejection resampling (``RejectionProvider``)
all condition on exactly the string the official chat encoding defines.

Interface: a drop-in for ``FireworksCompletionsProvider`` (same constructor kwargs the harness passes, same prefill
protocol, same history format), so ``RejectionProvider`` works unchanged::

    messages[-1] == {"role": "assistant", "reasoning_content": "<partial CoT>", "_prefill": True}

is an incomplete turn: the prompt is ``encode(messages[:-1])`` (which ends with the official generation prompt
``<｜Assistant｜><think>``) followed by the partial text verbatim. ``reasoning`` is accepted for ``reasoning_content``.
A stub with non-empty ``content`` is a finished chain of thought plus a partial answer: it is rendered with the official
``wo_eos=True`` form (reasoning + ``</think>`` + content). On success the stub is replaced by the completed message.

Configuration (constructor kwargs win over environment variables):
    DSV41_BASE_URL       http(s)://host:port of the vLLM server (``/v1`` suffix optional)
    LLM_UDS / VLLM_UDS   unix socket bridged to the server (sandboxes); wins over DSV41_BASE_URL. A ``uds=`` or
                         ``base_url=`` kwarg overrides both
    DSV41_API_KEY        bearer key (VLLM_API_KEY accepted as fallback; "none" = server without --api-key). Required
                         once a server is configured. Read at construction, then removed from os.environ; never logged
    DSV41_MODEL          served model name on the wire (VLLM_MODEL accepted as fallback; default: the ``model`` arg)
    DSV41_REASONING_EFFORT  used when the harness passes reasoning_effort=None (default "high")
    DSV41_TEMPERATURE / DSV41_TOP_P  used when the harness passes None (default 1.0 / 1.0, DeepSeek's recommendation)
    DSV41_SEED           base seed; call i gets sha256(f"{seed}:{run_id}:{i}") (recorded)
    DSV41_RUN_ID         run id mixed into the seeds (default: a fresh random id per provider object; must be unique
                         per run and per continuation rep when set)
    DSV41_MAX_TOKENS     cap on completion tokens (default: all context left)
    DSV41_MAX_MODEL_LEN  server context length (default: read from GET /v1/models, else 131072)
    DSV41_CONTEXT_MARGIN tokens left free below max_model_len (default 16) so a small server/local count drift is
                         diagnosed (strict checks) instead of turning every call into a context error
    DSV41_PROMPT_MODE    "text" (default; prompt string + add_special_tokens=false) or "token_ids" (local tokenizer)
    DSV41_ENCODING_PY    path to the official encoding.py (or its directory); default: searched next to this file
    DSV41_ALLOW_ENCODING_MISMATCH  1 = accept an encoding.py whose sha256 is not the checkpoint's (default: refuse)
    DSV41_TOKENIZER_JSON path to the checkpoint's tokenizer.json ("none" disables; then counts use POST /tokenize)
    DSV41_CALL_LOG       JSONL file receiving one record per completion call
    DSV41_TIMEOUT        per-request timeout in seconds (default 7200; non-streaming, so it bounds a whole generation)
    DSV41_RETRY_FOR_S    how long connection errors, 408/429/5xx and aborted generations are retried (default 3600 s);
                         DSV41_RETRY_BASE first backoff (default 2 s, doubling, capped at 120 s);
                         DSV41_MAX_ATTEMPTS optional hard cap on attempts (default: none)
    DSV41_MALFORMED_RETRIES  resample a finished completion the official parser rejects up to N times (default 2)
    DSV41_STRICT_CHECKS  1 (default): raise FidelityError when the server's prompt token ids show a double BOS or
                         differ from the pre-count tokenization; 0 records only
    DSV41_VLLM_XARGS     JSON object sent as ``vllm_xargs`` in every completion (causal-intervention engine: arm id and
                         edit window from ``tools/arms.py:request_fields``; opaque ids only). Default: not sent
    DSV41_CACHE_SALT     string sent as ``cache_salt`` in every completion (the per-arm salt from the same helper)
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
import random
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

try:  # inside an agent harness that defines these types (an agent-interp-envs style ``BaseProvider``)
    from agent_interp_envs.providers.base import BaseProvider
    from agent_interp_envs.types import LLMResponse, ToolCall, ToolResult
except ImportError:  # standalone: minimal stand-ins with the fields this module uses
    from dataclasses import dataclass as _dataclass

    class BaseProvider:  # noqa: D101
        def add_message(self, message: dict) -> None:
            self.messages.append(message)

    @_dataclass
    class ToolCall:  # noqa: D101
        id: str
        name: str
        arguments: str

    @_dataclass
    class ToolResult:  # noqa: D101
        id: str
        content: str

    @_dataclass
    class LLMResponse:  # noqa: D101
        reasoning: str | None = None
        response: str | None = None
        tool_calls: list | None = None

HERE = Path(__file__).resolve().parent
# Repo layout: provider/ next to official_encoding/ (gitignored; the checkpoint's encoding/ folder and tokenizer.json).
DEFAULT_ENCODING_PY = HERE.parent / "official_encoding" / "encoding" / "encoding.py"
DEFAULT_TOKENIZER_JSON = HERE.parent / "official_encoding" / "tokenizer.json"
_MODEL_DIR = Path(os.environ["DSV41_MODEL_DIR"]) if os.environ.get("DSV41_MODEL_DIR") else None
# Searched in this order when no path is configured: next to this file first (an image that copies the provider and
# its official_encoding/ folder next to the harness providers), then the checkpoint ($DSV41_MODEL_DIR), then the repo
# layout.
ENCODING_SEARCH = (HERE / "official_encoding" / "encoding" / "encoding.py", HERE / "official_encoding" / "encoding.py",
                   *((_MODEL_DIR / "encoding" / "encoding.py",) if _MODEL_DIR else ()), DEFAULT_ENCODING_PY)
TOKENIZER_SEARCH = (HERE / "official_encoding" / "tokenizer.json", HERE / "tokenizer.json",
                    *((_MODEL_DIR / "tokenizer.json",) if _MODEL_DIR else ()), DEFAULT_TOKENIZER_JSON)
# sha256 of encoding.py in DeepSeek-V4.1-Flash-1017 (checked against the checkpoint copy on 28 Oct 2026).
OFFICIAL_ENCODING_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"

PREFILL_KEY = "_prefill"  # same key as fireworks_completions_provider / rejection_provider
META_KEY = "_dsv41"       # provenance on produced assistant messages (seeded calls, parse failures); never rendered
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_CONTEXT_MARGIN = 16
DEFAULT_MALFORMED_RETRIES = 2
DEFAULT_RETRY_FOR_S = 3600.0
DEFAULT_TEMPERATURE = 1.0  # DeepSeek V4.1 model card: temperature 1.0, top_p 1.0
DEFAULT_TOP_P = 1.0


class ContextLengthError(RuntimeError):
    """No room left for a completion. The message contains "maximum context length", which the harness's
    ``is_context_limit_error`` recognises, so the episode ends censored instead of crashing."""


class PromptRenderError(ValueError):
    """The official encoding refuses the conversation (e.g. the literal image placeholder in a tool output) or the
    prompt is not valid UTF-8 (a lone surrogate). No further turn is possible; the message says "maximum context", so
    the harness censors the episode like a context overflow. The real cause is the first part of the message."""


class ServerError(RuntimeError):
    """Non-retryable HTTP error from the server (4xx other than context length), or a model the server does not serve."""


class APIConnectionError(ConnectionError):
    """The server stayed unreachable (or kept failing with 408/429/5xx/aborted generations) for the whole retry budget.
    The class name is one of the fleet driver's connection-failure markers, so a crashed run is resumed."""


class FidelityError(RuntimeError):
    """The server tokenized the prompt differently from the pre-count (double BOS, or other ids): the sample would not
    come from the official encoding's token sequence. Raised after the call record is written (strict checks)."""


class _DiscardedSample(RuntimeError):
    """A 200 response that is not a model sample (finish_reason abort/error): retried like a transport failure."""


# ----------------------------------------------------------------------------------------------------------------------
# Official encoding + tokenizer loading
# ----------------------------------------------------------------------------------------------------------------------

_LOAD_LOCK = threading.Lock()
_TOKENIZERS: dict[str, Any] = {}


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _xarg_ok(v: Any) -> bool:
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def engine_fields(env=os.environ) -> tuple[dict | None, str | None]:
    """``(vllm_xargs, cache_salt)`` from DSV41_VLLM_XARGS / DSV41_CACHE_SALT, validated like vLLM 0.31 does (values
    str/int/float or flat lists of them; salt non-empty, <= 128 chars, no ``@ / \\`` or NUL). Unset or empty -> None.
    A bad value raises ValueError at construction rather than a 400 on every call."""
    xargs = salt = None
    raw = env.get("DSV41_VLLM_XARGS")
    if raw:
        try:
            xargs = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"DSV41_VLLM_XARGS is not valid JSON: {e}") from None
        if not isinstance(xargs, dict) or not all(
                isinstance(k, str) and (_xarg_ok(v) or (isinstance(v, list) and all(_xarg_ok(x) for x in v)))
                for k, v in xargs.items()):
            raise ValueError("DSV41_VLLM_XARGS must be a JSON object of str/int/float values or lists of them")
    raw = env.get("DSV41_CACHE_SALT")
    if raw:
        if len(raw) > 128 or any(c in raw for c in "@/\\\0"):
            raise ValueError("DSV41_CACHE_SALT must be at most 128 chars without '@', '/', '\\' or NUL")
        salt = raw
    return xargs, salt


def _find_encoding(path: str | os.PathLike | None) -> Path:
    raw = path or os.environ.get("DSV41_ENCODING_PY")
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            p = p / "encoding.py"
        if not p.is_file():
            raise FileNotFoundError(f"DSV41_ENCODING_PY / encoding_path {p} does not exist")
        return p
    for p in ENCODING_SEARCH:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "the official DeepSeek V4.1 encoding.py was not found (searched: " + ", ".join(map(str, ENCODING_SEARCH))
        + "). Set DSV41_ENCODING_PY to the checkpoint's encoding/encoding.py, or copy the official_encoding/ folder "
          "next to this provider file.")


def load_official_encoding(path: str | os.PathLike | None = None, allow_mismatch: bool | None = None):
    """Import DeepSeek's encoding.py from ``path`` (file or directory) under a private module name.

    The file's sha256 must equal the checkpoint's (OFFICIAL_ENCODING_SHA256) unless ``allow_mismatch`` (or
    DSV41_ALLOW_ENCODING_MISMATCH=1). Bytecode writing is disabled for the import so the checkpoint copy's directory is
    never touched.
    """
    p = _find_encoding(path).resolve()
    if allow_mismatch is None:
        allow_mismatch = _truthy(os.environ.get("DSV41_ALLOW_ENCODING_MISMATCH", "0"))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha != OFFICIAL_ENCODING_SHA256 and not allow_mismatch:
        raise ImportError(f"{p} has sha256 {sha}, not the DeepSeek-V4.1-Flash checkpoint's encoding.py "
                          f"({OFFICIAL_ENCODING_SHA256}); set DSV41_ALLOW_ENCODING_MISMATCH=1 to load it anyway")
    name = "_dsv41_official_encoding_" + hashlib.sha1(f"{p}:{sha}".encode()).hexdigest()[:10]
    with _LOAD_LOCK:
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(name, p)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load the official DeepSeek V4.1 encoding from {p}")
            mod = importlib.util.module_from_spec(spec)
            old = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.dont_write_bytecode = old
            for attr in ("encode_messages", "parse_message_from_completion_text", "bos_token", "eos_token",
                         "thinking_start_token", "thinking_end_token", "ASSISTANT_SP_TOKEN", "dsml_token",
                         "tool_calls_block_name", "REASONING_EFFORT_MAPPINGS", "DEFAULT_REASONING_EFFORT"):
                if not hasattr(mod, attr):
                    raise ImportError(f"{p} is not the DeepSeek V4.1 encoding (missing {attr})")
            mod.__dsv41_path__ = str(p)
            mod.__dsv41_sha256__ = sha
            sys.modules[name] = mod
    return mod


def load_tokenizer(path: str | os.PathLike | None = None):
    """The checkpoint's tokenizer.json via the ``tokenizers`` library, or None when unavailable/disabled."""
    raw = path if path is not None else os.environ.get("DSV41_TOKENIZER_JSON")
    if isinstance(raw, str) and raw.lower() in ("", "none", "off"):
        return None
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            p = p / "tokenizer.json"
    else:
        p = next((c for c in TOKENIZER_SEARCH if c.is_file()), TOKENIZER_SEARCH[0])
    if not p.exists():
        return None
    key = str(p.resolve())
    with _LOAD_LOCK:
        if key not in _TOKENIZERS:
            try:
                from tokenizers import Tokenizer
            except ImportError:
                return None
            _TOKENIZERS[key] = Tokenizer.from_file(key)
    return _TOKENIZERS[key]


# ----------------------------------------------------------------------------------------------------------------------
# API key (read once, removed from the environment so the agent's subprocesses never inherit it)
# ----------------------------------------------------------------------------------------------------------------------

_ENV_KEY_NAMES = ("DSV41_API_KEY", "VLLM_API_KEY")
_env_key_cache: str | None = None


def _take_env_api_key() -> str | None:
    """DSV41_API_KEY (or VLLM_API_KEY) from os.environ, then removed from it. Cached in this module so later providers
    in the same process (continuation reps, external runners) still find it."""
    global _env_key_cache
    for name in _ENV_KEY_NAMES:
        v = os.environ.get(name)
        if v:
            _env_key_cache = v
            break
    for name in _ENV_KEY_NAMES:
        os.environ.pop(name, None)
    return _env_key_cache


# ----------------------------------------------------------------------------------------------------------------------
# Message mapping (harness / OpenAI chat format -> official encoding input)
# ----------------------------------------------------------------------------------------------------------------------

def _system_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):  # the official system branch formats a str; join text parts like vLLM does
        return "\n\n".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return content


def to_official_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Map harness messages to the official encoder's message dicts.

    Only the fields the official encoder defines are passed (role, content, reasoning_content, tool_calls with
    name/arguments/namespace, tool_call_id; tools on a system message), so provider-private keys (``_prefill``,
    ``_dsv41``) or other providers' extras cannot leak into the render. ``reasoning`` (vLLM/OpenRouter spelling) is
    accepted for ``reasoning_content``. ``content=None`` becomes "" (deliberate: for a tool message the official encoder
    would print the Python repr ``None``; the harness's ToolResult.content is a str anyway).

    Tools go on messages[0] when it is a system message; otherwise an empty system message carrying them is inserted at
    index 0. For every harness shape (system first, or no system message) this equals vLLM 0.31's own DeepSeek V4.1
    chat path (tokenizers/deepseek_v41.py). vLLM instead attaches tools to the first system message anywhere; that
    would move the tool section when a mid-conversation system message appears after a system-less start, so earlier
    turns would re-render differently from how they were sampled. Index 0 keeps every render a prefix of the next.
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            o = {"role": "system", "content": _system_text(m.get("content"))}
        elif role == "user":
            content = m.get("content")
            o = {"role": "user", "content": "" if content is None else content}
        elif role == "assistant":
            rc = m.get("reasoning_content")
            if rc is None:
                rc = m.get("reasoning")
            o = {"role": "assistant", "content": m.get("content") or "", "reasoning_content": rc or ""}
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    f = {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")}
                    if fn.get("namespace") is not None:
                        f["namespace"] = fn["namespace"]
                    call = {"id": tc.get("id") or "", "type": "function", "function": f}
                    if tc.get("namespace") is not None:
                        call["namespace"] = tc["namespace"]
                    calls.append(call)
                o["tool_calls"] = calls
        elif role == "tool":
            content = m.get("content")
            o = {"role": "tool", "tool_call_id": m.get("tool_call_id") or "", "content": "" if content is None else content}
        else:
            raise ValueError(f"Unsupported role in messages: {role!r}")
        out.append(o)
    if tools:
        if out and out[0]["role"] == "system":
            out[0]["tools"] = tools
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


def normalize_effort(effort: Any, official) -> str | int:
    """low/high/max or an int in [1, 100] (numeric strings allowed); None -> the official default ("high")."""
    if effort is None:
        return official.DEFAULT_REASONING_EFFORT
    if isinstance(effort, str):
        e = effort.strip().lower()
        if e.lstrip("-").isdigit():
            effort = int(e)
        elif e in official.REASONING_EFFORT_MAPPINGS:
            return e
        else:
            raise ValueError(
                f"reasoning_effort {effort!r} is not defined by the official V4.1 encoding: use "
                f"{sorted(official.REASONING_EFFORT_MAPPINGS)} (= {official.REASONING_EFFORT_MAPPINGS}) or an int 1-100 "
                "(some runners map 'medium' to 62)")
    if type(effort) is int and 1 <= effort <= 100:
        return effort
    raise ValueError(f"reasoning_effort must be low/high/max or an int in [1, 100], got {effort!r}")


def effort_budget(effort: str | int, official) -> int:
    return official.REASONING_EFFORT_MAPPINGS[effort] if isinstance(effort, str) else effort


def derive_seed(base_seed: int, call_index: int, run_id: str | None = None) -> int:
    """Distinct, reproducible per-call seed. The run id keeps two runs (or two continuation reps from the same cut)
    that share DSV41_SEED from sending identical requests with identical seeds; the call index keeps retries and
    rejection resamples from replaying the same sample."""
    key = f"{base_seed}:{call_index}" if run_id is None else f"{base_seed}:{run_id}:{call_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def _next_call_index(messages: list[dict], run_id: str) -> int:
    """Continue the call counter of a resumed run (same run id): one past the highest index stored on its messages."""
    used = [m[META_KEY]["call_index"] for m in messages
            if isinstance(m.get(META_KEY), dict) and m[META_KEY].get("run_id") == run_id
            and isinstance(m[META_KEY].get("call_index"), int)]
    return max(used) + 1 if used else 0


def _new_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex[:16]}"


def _completion_problem(raw: bytes) -> str | None:
    """None for a real sample (finish_reason stop/length), else why the 200 response is not one."""
    try:
        finish = json.loads(raw)["choices"][0].get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return "unparseable completion response"
    return None if finish in ("stop", "length") else f"finish_reason={finish}"


# ----------------------------------------------------------------------------------------------------------------------
# Transport (stdlib only: works in sandbox images without httpx/openai extras)
# ----------------------------------------------------------------------------------------------------------------------

class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._uds_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._uds_path)
        self.sock = sock


class _Transport:
    """POST/GET JSON to the vLLM server over TCP or a unix socket. The API key lives only here."""

    def __init__(self, base_url: str | None, uds: str | None, api_key: str | None, timeout: float):
        self.uds = uds or None
        self.timeout = timeout
        self._api_key = api_key
        if self.uds:
            self.scheme, self.host, self.port, self.prefix = "http", "localhost", None, ""
        elif base_url:
            u = urlsplit(base_url.rstrip("/"))
            if u.scheme not in ("http", "https") or not u.hostname:
                raise ValueError(f"DSV41_BASE_URL must look like http://host:port, got {base_url!r}")
            self.scheme, self.host, self.port = u.scheme, u.hostname, u.port
            prefix = u.path.rstrip("/")
            self.prefix = prefix[: -len("/v1")] if prefix.endswith("/v1") else prefix
        else:
            raise ValueError("no server configured: set DSV41_BASE_URL (http://host:port) or LLM_UDS (unix socket)")

    def describe(self) -> str:
        return f"unix:{self.uds}" if self.uds else f"{self.scheme}://{self.host}:{self.port or ''}{self.prefix}"

    def _conn(self) -> http.client.HTTPConnection:
        if self.uds:
            return _UnixHTTPConnection(self.uds, self.timeout)
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn = self._conn()
        try:
            conn.request(method, self.prefix + path, body=data, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def __repr__(self) -> str:  # never show the key
        return f"_Transport({self.describe()})"


_CONTEXT_MARKERS = ("maximum context length", "context length", "context_length", "max_model_len", "too long")


# ----------------------------------------------------------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------------------------------------------------------

class DeepSeekV41VLLMProvider(BaseProvider):
    """DeepSeek V4.1 Flash via raw completions on vLLM, rendered with the official encoding.py.

    Interchangeable with FireworksCompletionsProvider (same constructor, prefill protocol, history format), so it can
    be wrapped by RejectionProvider directly: ``RejectionProvider(DeepSeekV41VLLMProvider(...), judge, ...)``.
    """

    def __init__(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | int | None = None,
        *,
        max_tokens: int | None = None,
        seed: int | None = None,
        run_id: str | None = None,
        base_url: str | None = None,
        uds: str | None = None,
        api_key: str | None = None,
        max_model_len: int | None = None,
        prompt_mode: str | None = None,
        encoding_path: str | os.PathLike | None = None,
        tokenizer_path: str | os.PathLike | None = None,
        call_log_path: str | os.PathLike | None = None,
        drop_thinking: bool = True,
        timeout: float | None = None,
        malformed_retries: int | None = None,
        strict_checks: bool | None = None,
        verbose: bool = True,
    ) -> None:
        env = os.environ
        self.enc = load_official_encoding(encoding_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model = model
        self.wire_model = env.get("DSV41_MODEL") or env.get("VLLM_MODEL") or model
        self.messages = messages
        self.tools = tools
        if reasoning_effort is None:
            reasoning_effort = env.get("DSV41_REASONING_EFFORT") or None
        self.reasoning_effort = normalize_effort(reasoning_effort, self.enc)
        self.budget = effort_budget(self.reasoning_effort, self.enc)
        self.temperature = float(temperature if temperature is not None
                                 else env.get("DSV41_TEMPERATURE", DEFAULT_TEMPERATURE))
        self.top_p = float(top_p if top_p is not None else env.get("DSV41_TOP_P", DEFAULT_TOP_P))
        if max_tokens is None and env.get("DSV41_MAX_TOKENS"):
            max_tokens = int(env["DSV41_MAX_TOKENS"])
        self.max_tokens = max_tokens
        if seed is None and env.get("DSV41_SEED"):
            seed = int(env["DSV41_SEED"])
        self.seed = seed
        self.run_id = str(run_id if run_id is not None else (env.get("DSV41_RUN_ID") or uuid.uuid4().hex[:16]))
        if max_model_len is None and env.get("DSV41_MAX_MODEL_LEN"):
            max_model_len = int(env["DSV41_MAX_MODEL_LEN"])
        self.max_model_len = max_model_len  # None: discovered from the server on first use
        self.context_margin = int(env.get("DSV41_CONTEXT_MARGIN", DEFAULT_CONTEXT_MARGIN))
        self.prompt_mode = (prompt_mode or env.get("DSV41_PROMPT_MODE") or "text").lower()
        if self.prompt_mode not in ("text", "token_ids"):
            raise ValueError("prompt_mode must be 'text' or 'token_ids'")
        if self.prompt_mode == "token_ids" and self.tokenizer is None:
            raise ValueError("prompt_mode=token_ids needs the local tokenizer (tokenizers + tokenizer.json)")
        self.drop_thinking = drop_thinking
        self.timeout = float(timeout if timeout is not None else env.get("DSV41_TIMEOUT", 7200))
        self.retry_for_s = float(env.get("DSV41_RETRY_FOR_S", DEFAULT_RETRY_FOR_S))
        self.max_attempts = int(env["DSV41_MAX_ATTEMPTS"]) if env.get("DSV41_MAX_ATTEMPTS") else None
        self.retry_base = float(env.get("DSV41_RETRY_BASE", 2.0))
        self.malformed_retries = int(malformed_retries if malformed_retries is not None
                                     else env.get("DSV41_MALFORMED_RETRIES", DEFAULT_MALFORMED_RETRIES))
        self.strict_checks = (strict_checks if strict_checks is not None
                              else _truthy(env.get("DSV41_STRICT_CHECKS", "1")))
        log = call_log_path if call_log_path is not None else env.get("DSV41_CALL_LOG")
        self.call_log_path = Path(log) if log else None
        self.verbose = verbose
        self.vllm_xargs, self.cache_salt = engine_fields(env)

        # Server: explicit kwargs (uds over base_url) win over the environment (socket over URL).
        if uds is not None or base_url is not None:
            self._uds, self._base_url = uds, base_url
        else:
            self._uds = env.get("LLM_UDS") or env.get("VLLM_UDS")
            self._base_url = env.get("DSV41_BASE_URL")
        env_key = _take_env_api_key()
        key = api_key if api_key is not None else env_key
        if key is not None and key.strip().lower() == "none":
            key = ""  # explicit: the server runs without --api-key
        self._transport: _Transport | None = None
        self._api_key_for_lazy_transport = key
        if self._base_url or self._uds:
            self._get_transport()  # fail at construction on a bad URL or a missing key

        self.bos_id = self._special_id(self.enc.bos_token, 0)
        self.eos_id = self._special_id(self.enc.eos_token, 1)
        self.calls: list[dict] = []  # one record per completion call (see _finish_record)
        self.last_prompt: str | None = None
        self._call_index = _next_call_index(messages, self.run_id)
        # (completed message object, prefill stub) for the most recent prefill-consuming turn, so revert_last_turn can
        # restore the stub (same contract as FireworksCompletionsProvider; external judges may read it too).
        self._last_prefill: tuple[dict, dict] | None = None

    # ------------------------------------------------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def generation_prompt(self) -> str:
        return self.enc.ASSISTANT_SP_TOKEN + self.enc.thinking_start_token

    def _split_prefill(self) -> tuple[list[dict], dict | None]:
        if self.messages and self.messages[-1].get(PREFILL_KEY):
            stub = self.messages[-1]
            if stub.get("role") != "assistant":
                raise ValueError("Prefill stub must be an assistant message")
            if stub.get("tool_calls"):
                raise ValueError("a prefill stub cannot carry tool_calls: give the partial reasoning (and optionally the "
                                 "start of the answer as content)")
            return self.messages[:-1], stub
        return self.messages, None

    @staticmethod
    def _check_utf8(text: str) -> None:
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise PromptRenderError(
                f"the prompt is not valid UTF-8 ({e}); no further turn is possible, so the episode ends like a maximum "
                "context length overflow (censored)") from e

    def _encode_official(self, official_messages: list[dict]) -> str:
        try:
            text = self.enc.encode_messages(official_messages, thinking_mode="thinking",
                                            drop_thinking=self.drop_thinking, reasoning_effort=self.reasoning_effort)
        except ValueError as e:
            if "image special token" not in str(e):
                raise
            raise PromptRenderError(
                f"the official DeepSeek V4.1 encoding cannot render this conversation: {e} No further turn is "
                "possible, so the episode ends like a maximum context length overflow (censored)") from e
        self._check_utf8(text)
        return text

    def encode(self, history: list[dict]) -> str:
        """Official render of completed turns (thinking mode, this provider's effort), generation prompt included."""
        return self._encode_official(to_official_messages(history, self.tools))

    def _stub_prefill(self, history: list[dict], stub: dict, base: str) -> str:
        """Text appended after ``base`` for a prefill stub: the partial reasoning verbatim, or (non-empty content) the
        official wo_eos render of the partial turn, i.e. reasoning + </think> + content."""
        rc = stub.get("reasoning_content")
        if rc is None:
            rc = stub.get("reasoning")
        rc = rc or ""
        content = stub.get("content")
        if not content:
            return rc
        msgs = to_official_messages(history, self.tools)
        msgs.append({"role": "assistant", "reasoning_content": rc, "content": content, "wo_eos": True})
        full = self._encode_official(msgs)
        if not full.startswith(base):
            raise ValueError("the official wo_eos render of the prefill stub does not extend the conversation render")
        return full[len(base):]

    def render_parts(self, history: list[dict] | None = None, prefill: str | None = None) -> tuple[str, str]:
        """(official render ending in the generation prompt, prefill text appended after it)."""
        stub = None
        if history is None:
            history, stub = self._split_prefill()
        base = self.encode(history)
        if not base.endswith(self.generation_prompt):
            raise ValueError("the conversation must end with a user or tool turn: the official render ends with "
                             f"{base[-60:]!r}, not the generation prompt {self.generation_prompt!r}")
        if prefill is None:
            prefill = self._stub_prefill(history, stub, base) if stub is not None else ""
        self._check_utf8(prefill)
        return base, prefill

    def render(self) -> str:
        """The exact prompt text the next invoke() sends (official render + any prefill)."""
        base, prefill = self.render_parts()
        return base + prefill

    # ------------------------------------------------------------------------------------------------------------------
    # Tokens
    # ------------------------------------------------------------------------------------------------------------------

    def _special_id(self, token: str, default: int) -> int:
        if self.tokenizer is not None:
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                return tid
        return default

    def tokenize(self, text: str) -> list[int]:
        """Token ids exactly as the server tokenizes a text prompt with add_special_tokens=false."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        status, raw = self._http("POST", "/tokenize",
                                 {"model": self.wire_model, "prompt": text, "add_special_tokens": False})
        out = json.loads(raw)
        if "max_model_len" in out and self.max_model_len is None:
            self.max_model_len = int(out["max_model_len"])
        return list(out["tokens"])

    def _ensure_max_model_len(self) -> int:
        """max_model_len from GET /v1/models (once). Also refuses a wire model name the server does not serve."""
        if self.max_model_len is None:
            try:
                _, raw = self._http("GET", "/v1/models")
                cards = [c for c in (json.loads(raw).get("data") or []) if isinstance(c, dict)]
            except (ServerError, ValueError, AttributeError):  # discovery is best effort; outages still raise
                cards = []
            if cards:
                served = [c.get("id") for c in cards]
                if self.wire_model not in served:
                    raise ServerError(f"model {self.wire_model!r} is not served at {self._get_transport().describe()} "
                                      f"(served: {served}); set DSV41_MODEL or VLLM_MODEL to the served name")
                card = next(c for c in cards if c.get("id") == self.wire_model)
                if card.get("max_model_len"):
                    self.max_model_len = int(card["max_model_len"])
            if self.max_model_len is None:
                self.max_model_len = DEFAULT_MAX_MODEL_LEN
        return self.max_model_len

    # ------------------------------------------------------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------------------------------------------------------

    def _get_transport(self) -> _Transport:
        if self._transport is None:
            if not (self._base_url or self._uds):
                raise ValueError("no server configured: set DSV41_BASE_URL (http://host:port) or LLM_UDS (unix socket)")
            if self._api_key_for_lazy_transport is None:
                raise ValueError("no API key for the vLLM server: set DSV41_API_KEY (or VLLM_API_KEY), or "
                                 "DSV41_API_KEY=none for a server started without --api-key")
            self._transport = _Transport(self._base_url, self._uds, self._api_key_for_lazy_transport, self.timeout)
            self._api_key_for_lazy_transport = None
        return self._transport

    def _http(self, method: str, path: str, body: dict | None = None, *, failures: list | None = None,
              validate: Callable[[bytes], str | None] | None = None) -> tuple[int, bytes]:
        """Request with retries on connection errors, 408/429/5xx and responses ``validate`` rejects, for up to
        ``retry_for_s`` seconds (failed-attempt time + backoff). Raises ContextLengthError / ServerError at once, and
        APIConnectionError when the budget runs out. Each failed attempt is appended to ``failures``."""
        transport = self._get_transport()
        spent, attempt = 0.0, 0
        while True:
            attempt += 1
            self._last_attempts = attempt
            t0 = time.monotonic()
            try:
                status, raw = transport.request(method, path, body)
            except (OSError, http.client.HTTPException) as e:  # refused, reset, timeout, socket missing
                last, reason = e, type(e).__name__
                status, raw = None, b""
            took = time.monotonic() - t0
            if status is not None:
                if status < 400:
                    problem = validate(raw) if validate is not None else None
                    if problem is None:
                        return status, raw
                    last, reason = _DiscardedSample(f"vLLM returned {problem} (not a model sample)"), problem
                else:
                    text = raw.decode("utf-8", "replace")
                    if status == 400 and any(m in text.lower() for m in _CONTEXT_MARKERS):
                        raise ContextLengthError(f"vLLM rejected the request (maximum context length): {text[:500]}")
                    if status == 413:
                        raise ContextLengthError(f"request too large (HTTP 413, maximum context length): {text[:300]}")
                    if status not in (408, 429) and status < 500:
                        hint = " (check DSV41_API_KEY)" if status == 401 else ""
                        raise ServerError(f"vLLM HTTP {status}{hint}: {text[:500]}")
                    last, reason = ServerError(f"vLLM HTTP {status}: {text[:300]}"), f"HTTP {status}"
            spent += took
            if failures is not None:
                failures.append({"attempt": attempt, "reason": reason, "elapsed_s": round(took, 3)})
            wait = min(120.0, self.retry_base * 2 ** min(attempt - 1, 20)) * (0.5 + random.random() / 2)
            if (self.max_attempts is not None and attempt >= self.max_attempts) or spent + wait > self.retry_for_s:
                raise APIConnectionError(
                    f"vLLM at {transport.describe()} still failing after {attempt} attempts and {spent:.0f}s "
                    f"(retry budget DSV41_RETRY_FOR_S={self.retry_for_s:.0f}s): {type(last).__name__}: {last}")
            if self.verbose:
                print(f"[provider-retry] {type(last).__name__} attempt={attempt} sleep={wait:.1f}s "
                      f"elapsed={spent:.0f}s/{self.retry_for_s:.0f}s {str(last)[:120]}", flush=True)
            time.sleep(wait)
            spent += wait

    # ------------------------------------------------------------------------------------------------------------------
    # Completion + parsing
    # ------------------------------------------------------------------------------------------------------------------

    def _write_record(self, rec: dict) -> None:
        self.calls.append(rec)
        if self.call_log_path is not None:
            self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.call_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _complete(self, base: str, prefill: str) -> dict:
        """One raw completion for prompt base+prefill. Returns the (unfinished) call record."""
        prompt = base + prefill
        self.last_prompt = prompt
        call_index = self._call_index
        self._call_index += 1
        prompt_ids = self.tokenize(prompt)  # local tokenizer, or the server's /tokenize when none is available
        n_prompt = len(prompt_ids)
        ctx = self._ensure_max_model_len()
        room = ctx - n_prompt - self.context_margin
        if room < 1:
            raise ContextLengthError(
                f"maximum context length {ctx} reached: the prompt has {n_prompt} tokens, no room to generate "
                f"(margin {self.context_margin})")
        max_tokens = room if self.max_tokens is None else min(self.max_tokens, room)
        seed = derive_seed(self.seed, call_index, self.run_id) if self.seed is not None else None
        body: dict[str, Any] = {
            "model": self.wire_model,
            "prompt": prompt_ids if self.prompt_mode == "token_ids" else prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "stream": False,
            # The rendered prompt already starts with BOS: the server must not add another.
            "add_special_tokens": False,
            # Keep every generated token in the text; EOS is stripped by vLLM and re-attached below.
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "include_stop_str_in_output": False,
            "stop_token_ids": [self.eos_id],
            "return_token_ids": True,
        }
        if seed is not None:
            body["seed"] = seed
        if self.vllm_xargs is not None:
            body["vllm_xargs"] = dict(self.vllm_xargs)
        if self.cache_salt is not None:
            body["cache_salt"] = self.cache_salt
        rec: dict[str, Any] = {
            "t": round(time.time(), 3),
            "call_index": call_index,
            "model": self.wire_model,
            "server": self._get_transport().describe(),
            "prompt_mode": self.prompt_mode,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_budget": self.budget,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": seed,
            "seed_run_id": self.run_id,
            "max_tokens": max_tokens,
            "max_model_len": ctx,
            "prompt_tokens_local": n_prompt,  # counted before sending (local tokenizer or /tokenize)
            "prompt_ids_source": "local" if self.tokenizer is not None else "server_tokenize",
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prefill_chars": len(prefill),
            "encoding_sha256": self.enc.__dsv41_sha256__,
            "vllm_xargs": self.vllm_xargs,
            "cache_salt": self.cache_salt,
        }
        failures: list[dict] = []
        t0 = time.time()
        try:
            _, raw = self._http("POST", "/v1/completions", body, failures=failures, validate=_completion_problem)
        except Exception as e:  # keep a diagnostic record of the failed call, then let the error through
            rec.update(error=f"{type(e).__name__}: {e}"[:500], failed_attempts=failures,
                       http_attempts=getattr(self, "_last_attempts", 1))
            self._write_record(rec)
            raise
        latency = time.time() - t0
        resp = json.loads(raw)
        choice = resp["choices"][0]
        text = choice.get("text") or ""
        finish = choice.get("finish_reason")
        stop_reason = choice.get("stop_reason")
        token_ids = choice.get("token_ids")
        server_prompt_ids = choice.get("prompt_token_ids") or resp.get("prompt_token_ids")
        usage = resp.get("usage") or {}
        eos = self.enc.eos_token
        ended_with_eos = finish == "stop" and (stop_reason is None or stop_reason == self.eos_id)
        if token_ids:
            ended_with_eos = ended_with_eos or token_ids[-1] == self.eos_id
        completion_tokens = usage.get("completion_tokens", len(token_ids) if token_ids else None)
        rec.update({
            "prompt_tokens": usage.get("prompt_tokens", n_prompt),
            "cached_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
            "completion_tokens": completion_tokens,
            "finish_reason": finish,
            "stop_reason": stop_reason,
            "ended_with_eos": ended_with_eos,
            # finish_reason=length with the whole remaining context as max_tokens: the context ran out mid-turn.
            "context_exhausted": bool(finish == "length" and max_tokens == room
                                      and (completion_tokens is None or completion_tokens >= max_tokens)),
            "latency_s": round(latency, 3),
            "http_attempts": getattr(self, "_last_attempts", 1),
            "failed_attempts": failures,  # timeouts, refused connections, 5xx, aborted generations (each discarded)
            "completion_text": text,
            "completion_token_ids": token_ids,
        })
        if server_prompt_ids is not None:
            rec["server_prompt_bos_ok"] = (len(server_prompt_ids) > 1 and server_prompt_ids[0] == self.bos_id
                                           and server_prompt_ids[1] != self.bos_id)
            rec["server_prompt_ids_match_local"] = list(server_prompt_ids) == list(prompt_ids)
        rec["_text_for_parse"] = text + (eos if ended_with_eos and not text.endswith(eos) else "")
        return rec

    def _parse_lenient(self, full: str) -> dict | None:
        """The official parser on ``full`` with special-token MENTIONS in the reasoning masked.

        Used only when the strict parse already got through the structure and failed on its final check that BOS, EOS,
        <think>, </think> and ｜DSML｜ do not occur in reasoning or content, e.g. the agent's reasoning quotes a chat
        template or tokenizer file it just read. Masking happens only inside the reasoning span (before the first
        </think>), where none of these tokens is structural. The content keeps the strict check: markers there (V4
        tags, a second thinking block) are a malformed turn. Returns None if the masked text still fails to parse.
        """
        enc = self.enc
        i = full.find(enc.thinking_end_token)
        if i < 0 or "\ue000" in full:
            return None
        marks = [enc.bos_token, enc.eos_token, enc.thinking_start_token, enc.dsml_token]
        reasoning = full[:i]
        masked = reasoning
        for k, t in enumerate(marks):
            masked = masked.replace(t, f"\ue000{k}\ue001")
        try:
            parsed = enc.parse_message_from_completion_text(masked + full[i:], thinking_mode="thinking")
        except Exception:  # noqa: BLE001
            return None
        rc = parsed["reasoning_content"]
        for k, t in enumerate(marks):
            rc = rc.replace(f"\ue000{k}\ue001", t)
        if rc != reasoning:
            return None
        parsed["reasoning_content"] = rc
        return parsed

    def _parse(self, prefill: str, text_for_parse: str) -> tuple[dict, bool, str | None]:
        """Official parse of prefill+completion. Returns (assistant message, parse_ok, error)."""
        message, info = self.parse_completion(prefill, text_for_parse)
        return message, info["parse_ok"], info["parse_error"]

    def parse_completion(self, prefill: str, text_for_parse: str) -> tuple[dict, dict]:
        """Official parse of prefill+completion. Returns (assistant message, parse info)."""
        enc = self.enc
        full = prefill + text_for_parse
        info: dict[str, Any] = {"parse_ok": False, "parse_error": None, "parse_lenient": False,
                                "strict_parse_error": None, "n_invalid_args": 0, "fallback_has_dsml": False}
        try:
            parsed = enc.parse_message_from_completion_text(full, thinking_mode="thinking")
        except Exception as e:  # noqa: BLE001 - the official parser raises AssertionError/ValueError on bad format
            err = f"{type(e).__name__}: {e}"[:300]
            parsed = self._parse_lenient(full) if str(e).startswith("Unexpected special token") else None
            if parsed is None:
                info["parse_error"] = err
                return self._fallback(full, info), info
            info.update(parse_lenient=True, strict_parse_error=err)
        message = {"role": "assistant", "content": parsed["content"], "reasoning_content": parsed["reasoning_content"]}
        calls, bad = [], []
        for tc in parsed.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            try:
                if not isinstance(json.loads(args), dict):
                    bad.append(args)
            except ValueError:
                bad.append(args)
            call = {"id": _new_tool_call_id(), "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": args}}
            if tc.get("namespace") is not None:
                call["namespace"] = tc["namespace"]
            calls.append(call)
        if bad:
            # string="false" with a value that is not JSON: the official decoder pastes it in unchecked. The harness
            # could not execute it, and the official re-render of such a turn shows a different call.
            info.update(n_invalid_args=len(bad), parse_lenient=False, strict_parse_error=None,
                        parse_error=f"invalid JSON arguments in {len(bad)} tool call(s): {bad[0][:200]}")
            return self._fallback(full, info), info
        if calls:
            message["tool_calls"] = calls
        info["parse_ok"] = True
        return message, info

    def _fallback(self, full: str, info: dict) -> dict:
        """A turn for a completion the official parser rejects: the reasoning, NO content and no tool calls.

        The text after </think> (a malformed or truncated tool call, a cut-off answer) is never returned as content:
        an environment that treats a no-tool-call turn as the final report then records an empty report,
        and the harness's episode_end marks the run ended_empty_turn (excluded) instead of grading garbage. The dropped
        text is kept on the message under META_KEY and in the call record (completion_text)."""
        enc = self.enc
        body = full[: -len(enc.eos_token)] if full.endswith(enc.eos_token) else full
        if enc.thinking_end_token in body:
            reasoning, rest = body.split(enc.thinking_end_token, 1)
        else:  # cut off (or malformed) mid-reasoning
            reasoning, rest = body, ""
        info["fallback_has_dsml"] = enc.dsml_token in body
        message = {"role": "assistant", "content": None, "reasoning_content": reasoning}
        if rest:
            message["_unparsed_content"] = rest  # moved under META_KEY by _attach_meta
        return message

    def _finish_record(self, rec: dict, history: list[dict], base: str, prefill: str, message: dict,
                       info: dict) -> None:
        text_for_parse = rec.pop("_text_for_parse")
        rec.update(info)
        rec["n_tool_calls"] = len(message.get("tool_calls") or [])
        # Does the official render of the stored turn reproduce exactly what was sampled? (False e.g. when the model
        # wrote a non-canonical JSON value that re-renders differently; the next prompt then shows the canonical form.)
        rec["render_roundtrip_exact"] = None
        if info["parse_ok"] and rec["ended_with_eos"]:
            try:
                rec["render_roundtrip_exact"] = (self.enc.encode_messages(
                    to_official_messages(history + [message], self.tools), thinking_mode="thinking",
                    drop_thinking=self.drop_thinking, reasoning_effort=self.reasoning_effort)
                    == base + prefill + text_for_parse)
            except Exception:  # noqa: BLE001
                rec["render_roundtrip_exact"] = False
        # Is the cut a canonical token boundary for what the model then wrote? (False: re-tokenizing prefill+completion
        # merges tokens across the cut, i.e. the model continued from a token split it would not normally see.)
        rec["prefill_boundary_canonical"] = None
        if prefill:
            try:
                p_ids = self.tokenize(prefill)
                f_ids = self.tokenize(prefill + rec["completion_text"])
                rec["prefill_boundary_canonical"] = f_ids[: len(p_ids)] == p_ids
            except Exception:  # noqa: BLE001 - a diagnostic; never fail a finished call over it
                pass
        self._write_record(rec)
        if self.verbose:
            print(f"[provider-usage] t={time.time():.0f} model={self.model} in={rec['prompt_tokens']} "
                  f"cache_read={rec['cached_tokens'] or 0} cache_write=0 out={rec['completion_tokens'] or 0} "
                  f"rl_in_rem=None rl_out_rem=None finish={rec['finish_reason']} latency={rec['latency_s']}s "
                  f"parse_ok={info['parse_ok']} effort={self.reasoning_effort} budget={self.budget}", flush=True)
        if self.strict_checks and (rec.get("server_prompt_bos_ok") is False
                                   or rec.get("server_prompt_ids_match_local") is False):
            raise FidelityError(
                f"the server tokenized the prompt differently from the pre-count (server_prompt_bos_ok="
                f"{rec.get('server_prompt_bos_ok')}, server_prompt_ids_match_local="
                f"{rec.get('server_prompt_ids_match_local')}, prompt_tokens={rec['prompt_tokens']} vs local "
                f"{rec['prompt_tokens_local']}); the sample is not from the official token sequence. Check the "
                "server's tokenizer / add_special_tokens handling (DSV41_STRICT_CHECKS=0 records only)")

    def _attach_meta(self, message: dict, rec: dict, info: dict) -> None:
        meta: dict[str, Any] = {}
        unparsed = message.pop("_unparsed_content", None)
        if self.seed is not None or not info["parse_ok"] or info["parse_lenient"]:
            meta = {"call_index": rec["call_index"], "run_id": self.run_id}
            if rec.get("seed") is not None:
                meta["seed"] = rec["seed"]
            if not info["parse_ok"]:
                meta.update(parse_ok=False, parse_error=info["parse_error"], finish_reason=rec["finish_reason"])
                if unparsed is not None:
                    meta["unparsed_content"] = unparsed
            if info["parse_lenient"]:
                meta["parse_lenient"] = True
        if meta:
            message[META_KEY] = meta

    def invoke(self) -> LLMResponse:
        """Generate the next assistant turn (continuing any prefill stub).

        A finished completion the official parser rejects (malformed DSML, invalid JSON arguments) is resampled up to
        ``malformed_retries`` times. If it still fails, or the turn was cut by the DSV41_MAX_TOKENS cap, it comes back
        as reasoning only (content None, no tool calls). A turn cut off because the context ran out raises
        ContextLengthError, so the harness censors the episode.
        """
        history, stub = self._split_prefill()
        base, prefill = self.render_parts()
        for attempt in range(self.malformed_retries + 1):
            rec = self._complete(base, prefill)
            message, info = self.parse_completion(prefill, rec["_text_for_parse"])
            rec["malformed_retry"] = attempt
            self._finish_record(rec, history, base, prefill, message, info)
            if info["parse_ok"]:
                break
            if rec["finish_reason"] == "length":
                if rec["context_exhausted"]:
                    raise ContextLengthError(
                        f"maximum context length {rec['max_model_len']} reached during generation: prompt "
                        f"{rec['prompt_tokens']} + completion {rec['completion_tokens']} tokens, cut off before the "
                        "turn was complete (finish_reason=length)")
                break  # cut by the DSV41_MAX_TOKENS cap: not a format error, so not resampled
        self._attach_meta(message, rec, info)

        if stub is not None:
            self.messages[-1] = message
            self._last_prefill = (message, stub)
        else:
            self.messages.append(message)
            self._last_prefill = None

        tool_calls = [ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                      for tc in message.get("tool_calls") or []]
        return LLMResponse(
            reasoning=message.get("reasoning_content"),
            response=message.get("content"),
            tool_calls=tool_calls or None,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # History management (FireworksCompletionsProvider semantics)
    # ------------------------------------------------------------------------------------------------------------------

    def add_tool_result(self, tool_result: ToolResult) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_result.id, "content": tool_result.content})

    def revert_last_turn(self) -> None:
        """Remove the last assistant turn; if it consumed a prefill stub, restore the stub."""
        if not self.messages:
            return
        popped = self.messages.pop()
        if self._last_prefill is not None:
            completed, stub = self._last_prefill
            if popped is completed:
                self.messages.append(stub)
            self._last_prefill = None

    def set_prefill(self, reasoning: str) -> None:
        """Set (or replace) a trailing prefill stub: the next invoke() continues this partial chain of thought."""
        stub = {"role": "assistant", "reasoning_content": reasoning, PREFILL_KEY: True}
        if self.messages and self.messages[-1].get(PREFILL_KEY):
            self.messages[-1] = stub
        else:
            self.messages.append(stub)

    def clear_prefill(self) -> None:
        if self.messages and self.messages[-1].get(PREFILL_KEY):
            self.messages.pop()

    def dump_history(self, include_prompt: bool = False) -> str:
        """JSON of the message history (what checkpoints store; unchanged format).

        include_prompt=True instead returns {"messages", "rendered_prompt" (exactly what the next invoke() sends),
        "last_prompt" (what the last call sent), "calls" (per-call records)} for debugging.
        """
        if not include_prompt:
            return json.dumps(self.messages, indent=2)
        try:
            rendered = self.render()
        except ValueError as e:  # history ends in an assistant turn, or the official encoding refuses it
            rendered = f"<not renderable: {e}>"
        return json.dumps({"messages": self.messages, "rendered_prompt": rendered, "last_prompt": self.last_prompt,
                           "encoding": {"path": self.enc.__dsv41_path__, "sha256": self.enc.__dsv41_sha256__},
                           "calls": self.calls}, indent=2, ensure_ascii=False)

    def print_history(self) -> None:
        try:
            from agent_interp_envs.providers.fireworks_provider import FireworksProvider
        except ImportError:  # standalone
            print(self.dump_history())
            return
        FireworksProvider.print_history(self)

    def __repr__(self) -> str:
        server = self._transport.describe() if self._transport else (self._uds or self._base_url)
        return (f"DeepSeekV41VLLMProvider(model={self.model!r}, server={server!r}, effort={self.reasoning_effort!r}, "
                f"T={self.temperature}, top_p={self.top_p})")


_HARNESS_ONLY_KNOBS = ("provider_preferences", "reasoning_summary", "mock_script")


def create_provider(
    model: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float | None = None,
    top_p: float | None = None,
    reasoning_effort: str | int | None = None,
    rejection: dict | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Build the provider and, when ``rejection`` is given (harness config shape: judge_model, criterion, optional
    max_retries_per_boundary / max_generations_per_turn / log_path), wrap it in the harness's RejectionProvider with
    its SentenceJudge -- the same wrapping the harness's create_provider does for fireworks_completions.

    Accepts ``**checkpoint.provider_kwargs(config)`` directly (``provider`` must then be "dsv41_vllm"; the harness's
    OpenRouter/OpenAI/mock-only knobs are ignored). Like the harness factory, the API key leaves os.environ here."""
    if provider not in (None, "dsv41_vllm"):
        raise ValueError(f"this factory builds the dsv41_vllm provider, not {provider!r}")
    for k in _HARNESS_ONLY_KNOBS:
        kwargs.pop(k, None)
    built: BaseProvider = DeepSeekV41VLLMProvider(model, messages, tools, temperature=temperature, top_p=top_p,
                                                  reasoning_effort=reasoning_effort, **kwargs)
    if rejection and rejection.get("enabled") is not False:
        from agent_interp_envs.cot_judge import SentenceJudge
        from agent_interp_envs.providers.rejection_provider import RejectionProvider

        judge = SentenceJudge(model=rejection["judge_model"], criterion=rejection["criterion"])
        rkw = {k: rejection[k] for k in ("max_retries_per_boundary", "max_generations_per_turn", "log_path")
               if k in rejection}
        built = RejectionProvider(built, judge, **rkw)
    return built
