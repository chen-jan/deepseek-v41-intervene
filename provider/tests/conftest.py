"""Shared fixtures: import paths, the official encoding, the local tokenizer, synthetic agent conversations, and a
mock vLLM server (TCP or unix socket) that tokenizes prompts like vLLM does."""

import copy
import http.server
import json
import os
import socketserver
import sys
import threading
from pathlib import Path

import pytest

sys.dont_write_bytecode = True  # never leave .pyc files in the harness checkout or the checkpoint copy

PROVIDER_DIR = Path(__file__).resolve().parents[1]
# Optional: the source folder of an agent harness with an ``agent_interp_envs`` package. Without it, the tests that
# exercise the harness integration are skipped (see pytest_collection_modifyitems below).
HARNESS_SRC = os.environ.get("HARNESS_SRC")
# The checkpoint's encoding/ folder and tokenizer.json: $DSV41_OFFICIAL_DIR, else $DSV41_MODEL_DIR, else
# <repo>/official_encoding (gitignored). `ENCODING_ONLY=1 bash scripts/download_model.sh` fetches just these files.
OFFICIAL_DIR = Path(os.environ.get("DSV41_OFFICIAL_DIR") or os.environ.get("DSV41_MODEL_DIR")
                    or PROVIDER_DIR.parent / "official_encoding")
for p in (str(PROVIDER_DIR), HARNESS_SRC):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import deepseek_v41_vllm_provider as dsp  # noqa: E402

# The tests clear DSV41_* variables, so point the provider's default search at OFFICIAL_DIR first.
dsp.ENCODING_SEARCH = (OFFICIAL_DIR / "encoding" / "encoding.py", *dsp.ENCODING_SEARCH)
dsp.TOKENIZER_SEARCH = (OFFICIAL_DIR / "tokenizer.json", *dsp.TOKENIZER_SEARCH)

try:
    from agent_interp_envs.tool_calling import EXECUTE_COMMAND_TOOL  # noqa: E402
    HAVE_HARNESS = True
except ImportError:  # a shell tool of the same shape
    HAVE_HARNESS = False
    EXECUTE_COMMAND_TOOL = {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a shell command and return its output and exit code.",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string", "description": "The command to run"}},
                           "required": ["command"]},
        },
    }
HAVE_OFFICIAL = (OFFICIAL_DIR / "encoding" / "encoding.py").is_file() and (OFFICIAL_DIR / "tokenizer.json").is_file()


def pytest_collection_modifyitems(config, items):
    """Skip everything without the official encoding, and the harness-integration tests without the harness."""
    import inspect

    for item in items:
        if not HAVE_OFFICIAL:
            item.add_marker(pytest.mark.skip(reason=f"needs the checkpoint's encoding/ and tokenizer.json in {OFFICIAL_DIR}"
                                                    " (set DSV41_OFFICIAL_DIR or DSV41_MODEL_DIR)"))
            continue
        fn = getattr(item, "function", None)
        try:
            src = inspect.getsource(fn) if fn is not None else ""
        except (OSError, TypeError):
            src = ""
        if not HAVE_HARNESS and "agent_interp_envs" in src:
            item.add_marker(pytest.mark.skip(reason="needs an agent harness with agent_interp_envs (set HARNESS_SRC)"))

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit the final answer with a confidence and a list of touched files.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "files": {"type": "array", "items": {"type": "string"}},
                "final": {"type": "boolean"},
            },
            "required": ["answer"],
        },
    },
}
TOOLS = [EXECUTE_COMMAND_TOOL, SUBMIT_TOOL]

SYSTEM = "You are a coding agent working in /workspace. Use the execute_command tool."
TASK = "Fix the failing test in dulwich/objects.py. The issue: `Tree.add` mis-sorts names containing '/'."


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def conversations():
    """Harness-format conversations (as the agent loop builds them), each ending in a user or tool turn."""
    c1 = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "", "reasoning_content": "Let me look at the repository first.",
         "tool_calls": [call("c1", "execute_command", {"command": "ls -la && git log --oneline | head -5"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 8\ndrwxr-xr-x dulwich\n3f2a1b0 Initial commit\n"},
        {"role": "assistant", "content": "Now the tests.", "reasoning_content": "The log shows one commit.\n\nI should run the tests.",
         "tool_calls": [call("c2", "execute_command", {"command": "python -m pytest -x -q 2>&1 | tail -20"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "FAILED tests/test_objects.py::TreeTests::test_add - AssertionError"},
    ]
    # Parallel tool calls with results arriving out of order (the official encoder sorts them by call order),
    # unicode, JSON-looking strings, non-string arguments, and a harness re-prompt user message.
    c2 = copy.deepcopy(c1[:4]) + [
        {"role": "assistant", "content": "", "reasoning_content": "Two things at once: grep and cat — ünïcödé ✓.",
         "tool_calls": [call("p1", "execute_command", {"command": "grep -rn 'def add' dulwich/objects.py"}),
                        call("p2", "execute_command", {"command": "echo '{\"a\": [1, 2]}' | python -m json.tool"})]},
        {"role": "tool", "tool_call_id": "p2", "content": "{\n    \"a\": [\n        1,\n        2\n    ]\n}"},
        {"role": "tool", "tool_call_id": "p1", "content": "412:    def add(self, name, mode, hexsha):"},
        {"role": "assistant", "content": "Submitting.", "reasoning_content": "Done; submit with a confidence.",
         "tool_calls": [call("s1", "submit", {"answer": "patched Tree.add", "confidence": 0.9,
                                              "files": ["dulwich/objects.py"], "final": True})]},
        {"role": "tool", "tool_call_id": "s1", "content": "Error: submit is only allowed after the tests pass."},
        {"role": "user", "content": "Please call execute_command to continue, or reply with your final report."},
    ]
    # No system message (tools must still be rendered), an assistant turn without reasoning or tool calls.
    c3 = [
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "I will inspect the code.", "reasoning_content": ""},
        {"role": "user", "content": "Go ahead."},
        {"role": "assistant", "content": None, "reasoning_content": "Start with ls.",
         "tool_calls": [call("x1", "execute_command", {"command": "ls"})]},
        {"role": "tool", "tool_call_id": "x1", "content": ""},
    ]
    # Long-ish multi-turn agent loop.
    c4 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    for i in range(6):
        c4 += [
            {"role": "assistant", "content": "", "reasoning_content": f"Step {i}: check file {i}.\n\nThen continue.",
             "tool_calls": [call(f"k{i}", "execute_command", {"command": f"sed -n '{i * 10},{i * 10 + 9}p' dulwich/objects.py"})]},
            {"role": "tool", "tool_call_id": f"k{i}", "content": f"line {i * 10}\n<not a tag>\n\tindented\n"},
        ]
    return {"basic": c1, "parallel_unicode_nonstring": c2, "no_system": c3, "long": c4}


def attach_tools(messages, tools):
    """Official input format written directly (README: tools go on the system message), without the provider."""
    msgs = copy.deepcopy(messages)
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["tools"] = tools
        return msgs
    return [{"role": "system", "content": "", "tools": tools}] + msgs


@pytest.fixture(scope="session")
def enc():
    return dsp.load_official_encoding()


@pytest.fixture(scope="session")
def tok():
    t = dsp.load_tokenizer()
    assert t is not None, "tokenizers + tokenizer.json are needed for the tests"
    return t


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("DSV41_") or k in ("LLM_UDS", "VLLM_UDS", "VLLM_API_KEY", "VLLM_MODEL"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DSV41_RETRY_BASE", "0.01")
    monkeypatch.setattr(dsp, "_env_key_cache", None)  # the provider caches (and scrubs) an env key per process


def make_provider(messages, tools=TOOLS, **kw):
    kw.setdefault("verbose", False)
    return dsp.DeepSeekV41VLLMProvider("dsv41", copy.deepcopy(messages), tools, **kw)


# ----------------------------------------------------------------------------------------------------------------------
# Mock vLLM server
# ----------------------------------------------------------------------------------------------------------------------

class MockVLLM:
    """Minimal vLLM /v1/completions, /tokenize, /v1/models. Prompts are tokenized with the real tokenizer; with
    add_special_tokens=true the mock prepends BOS (the worst case a server can do), so a double BOS is visible."""

    def __init__(self, tok, uds_path=None, max_model_len=131072, api_key="sk-test-SECRET-123"):
        self.tok = tok
        self.uds_path = uds_path
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.requests = []  # (method, path, headers, body)
        self.script = []    # completions to return, in order: str text (EOS appended) or dict overrides
        self.fail_first = []  # HTTP statuses to return before serving normally
        self.lock = threading.Lock()

    # -- behaviour --
    def server_tokenize(self, prompt, add_special_tokens):
        if isinstance(prompt, list):
            return list(prompt)
        ids = self.tok.encode(prompt, add_special_tokens=False).ids
        return ([0] + ids) if add_special_tokens else ids

    def handle(self, method, path, headers, body):
        with self.lock:
            self.requests.append((method, path, dict(headers), body))
            if self.fail_first:
                status = self.fail_first.pop(0)
                return status, {"error": {"message": "Service Unavailable", "code": status}}
        if self.api_key and headers.get("Authorization") != f"Bearer {self.api_key}":
            return 401, {"error": {"message": "Unauthorized"}}
        if path == "/v1/models":
            return 200, {"data": [{"id": "dsv41", "max_model_len": self.max_model_len}]}
        if path == "/tokenize":
            ids = self.server_tokenize(body["prompt"], body.get("add_special_tokens", True))
            return 200, {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids}
        if path == "/v1/completions":
            prompt_ids = self.server_tokenize(body["prompt"], body.get("add_special_tokens", True))
            if len(prompt_ids) + body.get("max_tokens", 16) > self.max_model_len:
                return 400, {"error": {"message": f"This model's maximum context length is {self.max_model_len} tokens. "
                                                  f"However, you requested {len(prompt_ids) + body['max_tokens']} tokens.",
                                       "type": "BadRequestError", "code": 400}}
            with self.lock:
                item = self.script.pop(0) if self.script else "Nothing to do.</think>Done."
            if isinstance(item, str):
                item = {"text": item}
            text = item["text"]
            finish = item.get("finish_reason", "stop")
            gen_ids = self.tok.encode(text, add_special_tokens=False).ids
            if finish == "stop":
                gen_ids = gen_ids + [1]  # EOS is in token_ids but stripped from text, as vLLM does
            choice = {"index": 0, "text": text, "finish_reason": finish, "stop_reason": None, "logprobs": None}
            if body.get("return_token_ids"):
                choice["token_ids"] = gen_ids
                choice["prompt_token_ids"] = prompt_ids
            return 200, {"id": "cmpl-x", "object": "text_completion", "model": body["model"], "choices": [choice],
                         "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(gen_ids),
                                   "total_tokens": len(prompt_ids) + len(gen_ids)}}
        return 404, {"error": {"message": "not found"}}

    @property
    def completion_bodies(self):
        return [b for (m, p, h, b) in self.requests if p == "/v1/completions" and b is not None]

    # -- plumbing --
    def _handler(self):
        mock = self

        class H(http.server.BaseHTTPRequestHandler):
            def _do(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                status, payload = mock.handle(method, self.path, self.headers, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self._do("POST")

            def do_GET(self):
                self._do("GET")

            def log_message(self, *a):
                pass

            def address_string(self):
                return "mock"

        return H

    def __enter__(self):
        if self.uds_path:
            class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
                daemon_threads = True
            self.server = UnixHTTPServer(str(self.uds_path), self._handler())
            self.url = None
        else:
            self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
            self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def mock_server(tok):
    with MockVLLM(tok) as m:
        yield m
