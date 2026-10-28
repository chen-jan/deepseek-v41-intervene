# Provider: DeepSeek V4.1 Flash for agent harnesses (official encoding, raw vLLM completions)

`provider/deepseek_v41_vllm_provider.py` turns a chat history (OpenAI-style messages with tool calls) into the exact
prompt string DeepSeek's chat format defines, sends it to a vLLM server as a raw completion, and parses the answer
back. On every turn it:

1. renders the conversation with DeepSeek's **official** `encoding.py`, the copy shipped with the checkpoint. It loads
   that file from a configurable path, refuses a file whose sha256 is not the checkpoint's
   (`13231c59…599e2`), and never reimplements it. Rendering uses thinking mode and a numeric `Reasoning Effort` line;
2. appends any prefill (a partial chain of thought) verbatim after the official generation prompt
   `<｜Assistant｜><think>`;
3. POSTs the string to `{base}/v1/completions` with `add_special_tokens=false`, so the server adds no second BOS (or
   sends the token ids from the checkpoint tokenizer, `DSV41_PROMPT_MODE=token_ids`);
4. parses the completion with the official `parse_message_from_completion_text` (V4.1 `<｜DSML｜ calls>` tags);
5. records tokens, finish reason, latency and fidelity checks for every call, and fails loudly when the server
   tokenized the prompt differently.

That covers fresh agent runs, continuations from a cut point mid-reasoning, and sentence-level rejection resampling,
all conditioned on exactly the string the official chat encoding defines.

The class follows the interface of an agent-interp-envs style harness provider (`invoke()`, `add_tool_result()`,
`revert_last_turn()`, `set_prefill()`, `dump_history()`, a `_prefill` stub as the last message). It runs standalone
too: without the harness installed it uses small stand-in types for `LLMResponse`, `ToolCall` and `ToolResult`.
Rejection wrapping (`create_provider(..., rejection=...)`) needs the harness.

## Files it needs

The checkpoint's `encoding/encoding.py` and `tokenizer.json` (and the `tokenizers` package for local token counts;
without it every turn also calls `POST /tokenize`). It looks for them in this order: `DSV41_ENCODING_PY` /
`DSV41_TOKENIZER_JSON`, then `provider/official_encoding/`, then `$DSV41_MODEL_DIR`, then `<repo>/official_encoding/`.
`ENCODING_ONLY=1 bash scripts/download_model.sh` downloads just those files (a few MB).

## Use

```python
import sys; sys.path.insert(0, "provider")
from deepseek_v41_vllm_provider import DeepSeekV41VLLMProvider

TOOLS = [{"type": "function", "function": {"name": "execute_command", "description": "Run a shell command.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
p = DeepSeekV41VLLMProvider("dsv41", [{"role": "system", "content": SYS}, {"role": "user", "content": TASK}], TOOLS,
                            reasoning_effort="high", base_url="http://127.0.0.1:8100")   # key from DSV41_API_KEY
r = p.invoke()                        # r.reasoning, r.response, r.tool_calls
print(p.render())                     # the next prompt, exactly as it will be sent

# Continue from a cut point: the history up to the cut turn plus the reasoning prefix as a stub.
p = DeepSeekV41VLLMProvider("dsv41", history + [{"role": "assistant", "reasoning_content": r[:cut], "_prefill": True}],
                            TOOLS, reasoning_effort=62)
```

Prefill stubs:
- `reasoning_content` (or `reasoning`) is the partial chain of thought, appended verbatim.
- A stub with non-empty `content` is a finished chain of thought plus the start of the answer. It is rendered the
  official way for a turn prefix, `wo_eos=True` (reasoning + `</think>` + content), and the model continues the
  answer. An empty `content` is ignored, so the stub stays a mid-reasoning cut.
- A stub with `tool_calls` is refused.

**With the edit engine.** Set `DSV41_VLLM_XARGS` and `DSV41_CACHE_SALT` to the values `tools/arms.py fields` prints
for your arm (or pass them from `request_fields()`), and point `DSV41_BASE_URL` at the engine or its proxy.

## Configuration (environment; constructor keyword arguments override)

| Variable | Meaning | Default |
|---|---|---|
| `DSV41_BASE_URL` | `http://host:port` of the vLLM server (`/v1` suffix accepted) | - |
| `LLM_UDS` (or `VLLM_UDS`) | unix socket bridged to the server (sandboxes); wins over the URL | - |
| `DSV41_API_KEY` (or `VLLM_API_KEY`) | bearer key, **required** once a server is configured (`none` = server without `--api-key`). Read at construction, then removed from `os.environ`, so the agent's subprocesses never inherit it. Never in records, `dump_history`, `repr` or printed lines | - |
| `DSV41_MODEL` (or `VLLM_MODEL`) | served model name on the wire | the `model` argument |
| `DSV41_REASONING_EFFORT` | used when the caller passes `None`: `low` (50), `high` (75), `max` (100) or an int 1-100. `medium` is rejected (the official encoding does not define it) | `high` |
| `DSV41_TEMPERATURE`, `DSV41_TOP_P` | used when the caller passes `None` | 1.0, 1.0 (model card) |
| `DSV41_SEED`, `DSV41_RUN_ID` | base seed; call *i* gets `sha256(f"{seed}:{run_id}:{i}")`. The run id must be unique per run and per continuation rep | unset; random run id |
| `DSV41_MAX_TOKENS` | cap on completion tokens | all context left |
| `DSV41_MAX_MODEL_LEN` | context length (set it to skip one `GET /v1/models` per provider object) | from the server, else 131072 |
| `DSV41_CONTEXT_MARGIN` | tokens left free below the context length | 16 |
| `DSV41_PROMPT_MODE` | `text` or `token_ids` | `text` |
| `DSV41_ENCODING_PY`, `DSV41_TOKENIZER_JSON` | the official files (`none` disables the tokenizer) | searched (above) |
| `DSV41_ALLOW_ENCODING_MISMATCH` | `1` loads an `encoding.py` with another sha256 | refuse |
| `DSV41_CALL_LOG` | JSONL file, one record per completion call | unset |
| `DSV41_TIMEOUT` | per-request timeout, s (non-streaming: it bounds a whole generation; raise it on the eager engine) | 7200 |
| `DSV41_RETRY_FOR_S`, `DSV41_RETRY_BASE`, `DSV41_MAX_ATTEMPTS` | retry budget for connection errors, 408/429/5xx and aborted generations; first backoff; optional cap | 3600 s, 2 s, none |
| `DSV41_MALFORMED_RETRIES` | resample a finished completion the official parser rejects, up to N times | 2 |
| `DSV41_STRICT_CHECKS` | `1` raises `FidelityError` on a double BOS or a server tokenization that differs from the pre-count | 1 |
| `DSV41_VLLM_XARGS` | JSON object sent as `vllm_xargs` in every completion (edit engine) | not sent |
| `DSV41_CACHE_SALT` | string sent as `cache_salt` in every completion (edit engine) | not sent |

## What is sent

```json
{"model": "dsv41", "prompt": "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (…)\n\n…<｜Assistant｜><think>[prefill]",
 "max_tokens": "min(cap, max_model_len - prompt_tokens - margin)", "temperature": 1.0, "top_p": 1.0, "n": 1,
 "stream": false, "add_special_tokens": false, "skip_special_tokens": false, "spaces_between_special_tokens": false,
 "include_stop_str_in_output": false, "stop_token_ids": [1], "return_token_ids": true, "seed": "(if set)"}
```

`add_special_tokens` defaults to true in vLLM 0.31, so it is set explicitly. vLLM strips EOS from the returned text,
so the provider re-attaches it when `finish_reason == "stop"` and `stop_reason` is null or 1, then calls the official
parser.

## When something goes wrong

| Situation | What the provider does |
|---|---|
| The official parser rejects a finished turn (malformed DSML, text after the calls block, ...) | Resamples up to `DSV41_MALFORMED_RETRIES` times. A turn that still fails comes back as reasoning only (no content, no tool calls); the dropped text is kept under `_dsv41.unparsed_content` and in the call record. |
| The reasoning merely mentions BOS, EOS, `<think>` or `｜DSML｜` before a well-formed call | Re-runs the official parser with those mentions in the reasoning masked, keeps the recovered turn (`parse_lenient=true`). |
| The context runs out mid-turn | Records the call, raises `ContextLengthError` (message contains "maximum context length", so harnesses censor the episode). |
| A turn is cut by `DSV41_MAX_TOKENS` | Not resampled; returned as reasoning only. |
| The official encoding refuses the conversation, or the prompt is not UTF-8 | `PromptRenderError` before any request. |
| Connection errors, timeouts, 408, 429, 5xx, or `finish_reason` abort | Retried for up to `DSV41_RETRY_FOR_S`, each failed attempt recorded; then `APIConnectionError`. |
| The server's prompt ids show a double BOS or differ from the pre-count | Writes the record, then raises `FidelityError`. |
| Other 4xx (401, 404, unknown model) | `ServerError`, not retried. |

Each call record (`provider.calls`, `DSV41_CALL_LOG`) holds the request settings, token counts, finish reason,
latency, attempts, prompt hash, completion text and ids, parse results, and four fidelity flags:
`server_prompt_bos_ok`, `server_prompt_ids_match_local`, `render_roundtrip_exact`, `prefill_boundary_canonical`.

## Live check against the real server (28 Oct 2026, plain vLLM 0.31 on 4x B200)

98 live completions, all passed: server prompt ids = local tokenizer ids (98/98), exactly one BOS (98/98), no
retries, no lenient parses, no format resamples. All calls that ended on EOS parsed and re-rendered byte for byte. A
two-turn agent exchange with a fake shell tool parsed both turns. Prefill continuations (including a 13.9k-character
cut of a max-effort trace) went on without a new `<think>`. Effort low vs max: mean reasoning tokens 4,018 vs 6,798.
Throughput through the provider: 32 parallel requests decoding 512 tokens each, 2,875 tok/s in aggregate (104 tok/s
per sequence); one sequence 181 tok/s.

Observations (model behaviour, not provider bugs): first agent turns often have very short reasoning (4-24 tokens),
and about 1 sample in 8 closes `</think>` as its first token (empty reasoning), at high and at max effort.
`cached_tokens` stays null unless the server runs with `--enable-prompt-tokens-details` (the serve scripts set it).

## Limits

1. **Cut points must be canonical token boundaries** for an exact continuation. The prompt text is re-tokenized, so a
   cut that splits what the tokenizer would merge (`file.` + `\n\nThe`, or a cut ending in a space) continues from a
   token split the model rarely saw. `prefill_boundary_canonical` flags this per call. A token-id prefill path is not
   implemented.
2. **Format resampling conditions on well-formed output.** Report the per-run count of `parse_ok=false` calls next to
   any rate.
3. **`render_roundtrip_exact=false`** happens when the model writes a non-string parameter in non-canonical JSON
   (`0.90`, `1e5`, compact arrays, ...). The next prompt shows the canonical form.
4. **Seeds** make a call reproducible only for the same batch composition (vLLM is not batch-invariant by default).
5. Strings like `<｜end▁of▁sentence｜>` or `</think>` inside tool outputs tokenize as special tokens. This is inherent
   to the string-based official encoding and the same on vLLM's chat path.
6. Tools go on `messages[0]` when it is a system message, otherwise on an inserted empty system message at index 0
   (equal to vLLM 0.31's chat path for every usual shape).
7. Tool content `None` renders as an empty tool result; the official encoder would print `None`.

## Tests

```bash
cd provider && DSV41_OFFICIAL_DIR=/path/with/encoding_and_tokenizer python -m pytest -q
```

The suite needs the official `encoding/` folder and `tokenizer.json` (it skips everything without them). Tests that
exercise a harness integration are skipped unless `HARNESS_SRC` points at a harness source folder with an
`agent_interp_envs` package. One test runs DeepSeek's own encoding examples and needs the checkpoint's
`inference/examples` folder; it skips without it. Without the harness and the examples: 497 passed, 13 skipped. With
both (`HARNESS_SRC` = the `src/` folder of the public
[agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) main branch, plus its dependencies): 510 passed
(28 Oct 2026).
