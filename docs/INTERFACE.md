# Engine interface (DeepSeek V4.1 Flash causal edits, vLLM 0.31)

The field-level contract of the edit engine: names, shapes, env vars, request fields, capture format, arms file and
RPC methods. "Edit at layer L" always means the output stack of 0-based layer L, at the input boundary of
`layers[L+1]`. Paths are relative to the repo root.

## 0. Checked against the vLLM 0.31.0 source (28 Oct 2026)

Package: `vllm` 0.31.0 as installed by `scripts/build_env.sh`. Line numbers refer to that version.

**Design points that follow from the vLLM source**
1. **Module path.** The model is `vllm/models/deepseek_v41/nvidia/model.py` (`vllm.models.deepseek_v41.nvidia.model`),
   not under `vllm/model_executor/models/`. The mHC kernels are in `vllm/model_executor/kernels/mhc/tilelang.py`; the
   dispatch ops in `vllm/models/deepseek_v41/nvidia/ops/mhc.py`.
2. **Plugin vs worker extension.** The patch loads through `--worker-extension-cls`, not a global plugin. A
   `vllm.general_plugins` entry point exists but is a no-op unless `DSV41_ENGINE=1`, because `load_general_plugins()` runs in every worker before the extension class is resolved and before the model is built
   (`v1/worker/worker_base.py:267-269, 289-312`), and it loads **every** entry point in the venv unless `VLLM_PLUGINS` is
   set (`plugins/__init__.py:77-91`). An installed plugin would therefore also load into the plain server, which
   uses the same venv. So:
   - `plugin.register()` does nothing unless `DSV41_ENGINE=1`.
   - **Do not `pip install` this package into the shared venv.** `scripts/serve_engine.sh` puts the repo on `PYTHONPATH` and passes
     `--worker-extension-cls dsv41_intervene.rpc.EditExt`; importing `rpc` calls `plugin.register()`. The entry point
     in `pyproject.toml` is declared but used only if the package is ever installed into an engine-only venv.
   - Both `worker_extension_cls` and `DSV41_*` envs are outside vLLM's config hash, so the autotune cache stays warm.
3. **Host positions.** No per-request token counter is needed: `InputBatch.num_computed_tokens_np`
   (`[num_reqs]`, batch order, gathered through `idx_mapping_np` at `model_runner.py:1461`) gives each request's first
   scheduled position on the host. The kernel masks windows with the device `positions` tensor it already gets, so
   host positions are used only to assign capture rows.
4. **Padding value.** Padding (and dummy/capture runs) has per-token arm id **-1**: never
   edited, never counted. So `tokens_seen[0]` counts real sham tokens exactly.
5. **`remove_request` takes `req_id: str`**, not an index (`model_states/default.py:84`). Per-request host state is
   keyed by `req_index` and overwritten by the next `add_request`; nothing to do on removal.
6. **`vllm_xargs` value types** are `str | int | float | list[str | int | float]` (`completion/protocol.py:226`):
   no bools, no nested dicts, ints within int64 (`sampling_params.py:683-701`). All fields below respect that.
7. **`cache_salt`** must be a non-empty string of at most 128 chars without `@ / \` or NUL
   (`entrypoints/generate/base/protocol.py:33-47`). Salts here are 24 hex chars; the capture nonce goes inside the hash.

**Other facts the engine relies on**
- `supports_mhc_all_reduce` is imported by name into `model.py` (line 108) and read at `DeepseekV4Model.__init__`
  (line 649), so `M.supports_mhc_all_reduce = lambda vllm_config: False` turns fusion off. With fusion off, `wo_b` and
  the MoE reduce their own outputs (`model.py:318-328`), so `x` is a reduced `torch.Tensor` at every boundary.
- `mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix) -> Tensor[T, 4, 5120] bf16`, a fresh `empty_like`
  (`tilelang.py:791-801`). Imported by name into `model.py:28-31`.
- `mhc_pre_delayed_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
  hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6)` returns
  `(post_mix [T,4,1], res_mix [T,4,4], layer_input [T,5120] bf16, next_pre_mix [T,4] fp32)` (`tilelang.py:74-115`).
  `mhc_pre_delayed_overlap` has the same positional args plus `*, stream, layer_input=None`, and its side stream waits
  on the current stream before reading `residual` (`ops/mhc.py:98-190`), so an in-place edit on the current stream is
  seen.
- `DeepseekV4DecoderLayer.forward(x, positions, input_ids, pre_mix=None, post_mix=None, res_mix=None, residual=None,
  engram_hashes=None, engram_mask=None, *, capture_previous_aux=False, mega_gate_metadata=None)` returns
  `(x, residual, post_mix, res_mix, pre_mix, previous_aux)` (`model.py:446-617`). Its `residual is None` branch with a
  3-D `x` (`model.py:494-508`) runs `mhc_pre` with the carried `pre_mix` and picks the overlap variant by itself. The
  model loop calls `layer(...)` (`model.py:917-929`), so an instance-attribute `forward` is used.
- `get_model_state_cls` is a staticmethod on `DeepseekV41LLMForCausalLM` (`model.py:1388`) and on the VL wrapper
  `DeepseekV41ForCausalLM` (`vl_model.py:275`); `init_model_state` calls `model.get_model_state_cls()`
  (`v1/worker/gpu/model_states/__init__.py:32`). Patch both.
- `DeepseekV41ModelState(DefaultModelState)` (`nvidia/model_state.py`): `add_request(req_index, new_req_data)`,
  `prepare_inputs(input_batch, req_states) -> dict`, `prepare_dummy_inputs(num_reqs, num_tokens) -> dict`. It stages
  per-request data with `UvaBufferPool(size, dtype).copy_to_gpu(np_array, out=gpu_view)`; copy that pattern.
- `InputBatch` (`v1/worker/gpu/input_batch.py:39-124`) has `num_reqs`, `idx_mapping_np`, `num_scheduled_tokens`
  (np, batch order), `num_tokens`, `num_tokens_after_padding`, `num_computed_tokens_np`, `positions` (GPU,
  `[num_tokens_after_padding]`).
- `NewRequestData.sampling_params.extra_args` is the request's `vllm_xargs` (`completion/protocol.py:354`).
- `/collective_rpc` takes `{"method", "args": [str], "kwargs": {str: str}, "timeout"}` and answers
  `{"results": [one per rank]}`; a result that is not a dict or list is turned into a string
  (`entrypoints/serve/dev/rpc/api_router.py:23-55`). RPC methods therefore return dicts.

## 1. Files

| Path | What it is |
|---|---|
| `dsv41_intervene/__init__.py` | constants, `parse_layers()`, `env_path()`; imports without vLLM or torch |
| `dsv41_intervene/patch.py` | fusion off, model-init hook, layer wrapper, model-state hook |
| `dsv41_intervene/ops.py` | `edit_stack_` Triton kernel, torch version with the same semantics, `warm_kernels` |
| `dsv41_intervene/capture.py` | capture rings, host row map, dump |
| `dsv41_intervene/arms.py` | `EditState` (all device buffers), `init_state`, xargs parsing, `ArmModelState`, `load_arms` |
| `dsv41_intervene/rpc.py` | `EditExt` worker extension (`dsv41_*` RPC methods) |
| `dsv41_intervene/plugin.py` | `register()` entry point, gated by `DSV41_ENGINE=1` |
| `dsv41_intervene/auth.py` | API-key middleware for every route but `/health` |
| `scripts/serve_engine.sh` | launch |
| `tools/arms.py` | build arms files, mu, control sampler, `request_fields()` and salts |
| `tools/client.py` | HTTP client: completions, `score`, native floor, RPC wrappers, CLI |
| `tools/proxy.py` | aiohttp proxy `0.0.0.0:8100 -> 127.0.0.1:8101` |
| `tools/gates.py` | the gate suite (`docs/GATES.md`) |
| `tools/bench.sh` | `vllm bench serve` runs (native, sham, arms) |
| `tools/build_data.py` | builds `data/*.json` (section 9) |
| `provider/deepseek_v41_vllm_provider.py` | `DSV41_VLLM_XARGS` / `DSV41_CACHE_SALT` passthrough into the `/v1/completions` body |
| `tests/` | CPU tests, the CPU mock engine, 1-GPU kernel tests |

`ops.py` and the torch reference must import without vLLM (CPU tests run without it); `tools/*` import nothing from
`dsv41_intervene` except `dsv41_intervene/__init__.py`.

## 2. Constants and env vars (`dsv41_intervene/__init__.py`)

| Env var | Default | Read by | Meaning |
|---|---|---|---|
| `DSV41_ENGINE` | unset | plugin | must be `1` or nothing is patched |
| `DSV41_EDIT_LAYERS` | `12,14-38` | arms.init_state | installed 0-based output layers (26 slots); never 0, 13 or >= 39 |
| `DSV41_N_ARMS` | `16` | arms.init_state | arm slots; arm 0 = sham, all zero |
| `DSV41_CAPTURE_LAYERS` | `12,15,18,22,25,26,27,30,34` | arms.init_state | vector-capture layers; each must be installed |
| `DSV41_COEF_ROWS` | `1048576` | capture | coefficient ring rows (rank 0) |
| `DSV41_VEC_ROWS` | `32768` | capture | vector ring rows (rank 0) |
| `DSV41_EDIT_IMPL` | `triton` in code, `torch` in `serve_engine.sh` | ops | `triton` or `torch` |
| `DSV41_MHC_OVERLAP` | `1` | patch.apply, gates | `0` also patches `supports_mhc_overlap` off (no mHC side stream; fallback if only the FULL-overlap path fails) |
| `DSV41_EAGER` | `0` | serve_engine.sh | `1` adds `--enforce-eager` and autotune off (fallback) |
| `DSV41_CAPTURE_DIR` | `$DSV41_ROOT/captures` | rpc, gates | dump folder (node-local disk is best); dump paths must be inside it |
| `DSV41_VLLM_XARGS` | unset | provider | JSON object sent as `vllm_xargs` |
| `DSV41_CACHE_SALT` | unset | provider | string sent as `cache_salt` |
| `DSV41_ENGINE_URL` | `http://127.0.0.1:8101` | tools | engine (control routes live here only) |
| `DSV41_KEY_FILE` | `$DSV41_ROOT/.serve_key` | servers, tools | read inside the script or tool, never printed |
| `DSV41_ARMS_DIR` | `$DSV41_ROOT/arms` | tools, proxy | arms files, `LOADED_SHAS`, `GLOBAL_ARM` |
| `DSV41_DATA_DIR` | `<repo>/data` | gates | test prompts (section 9) |

Fixed: `HC = 4`, `HIDDEN = 5120`, `MAX_EDIT_TO = 2**31 - 1`, `MAX_CAP_POS = 4096`, `PAD = -1`.
`MAX_T = vllm_config.scheduler_config.max_num_batched_tokens` (32768 with the serve flags); assert
`num_tokens_after_padding <= MAX_T` in `prepare_inputs`.

**Ports.** vLLM: `127.0.0.1:8101` with `VLLM_SERVER_DEV_MODE=1`. Proxy: `0.0.0.0:8100` (what other machines and
sandboxes use). Control tools run on the GPU node against `:8101`.

## 3. Shared device state: `arms.EditState` (one per rank, module global `arms.STATE`)

Allocated once by `arms.init_state(vllm_config, device)`, called from the patched `DeepseekV4Model.__init__` before
profiling and graph capture. Changed afterwards only in place (`copy_`, `fill_`, `zero_`). `A = N_ARMS`, `I = N_INST`.

**Arms table** (written only by `load_arms`; every kernel reads it through these fixed tensors):

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| `layers` | `[I]` (host list too: `st.installed`) | int32 | installed output layers, ascending |
| `p` | `[A, I, 4, 5120]` | fp32 | read direction per stream (d; a mean-space d is copied to all 4) |
| `e` | `[A, I, 4, 5120]` | fp32 | write direction per stream (d for project, c for matched shift and add-of-c) |
| `mu` | `[A, I, 4]` | fp32 | per-stream reference coefficient |
| `a` | `[A, I]` | fp32 | removal strength lambda (0 = off at this slot) |
| `g` | `[A, I]` | fp32 | add amount alpha (0 = off) |
| `ep` | `[A, I, 4]` | fp32 | `sum_h e*p` per stream, computed by `load_arms` |
| `mc` | `[A]` | int32 | 1 = mean-coupled delta, 0 = per-stream (default) |

A slot is edited for a token iff `a != 0 or g != 0` there. Measure-only = `p = d`, `a = g = 0` (coefficients are
still captured and counted in `tokens_seen`, never in `tokens_edited`).

**The edit** (`c_s = h_s . p_s` in fp32):
```
delta_s = c_s - mu_s                                   (mc = 0)
delta_s = mean_s(c) - mean_s(mu)   for every s         (mc = 1)
edit iff arm >= 0 and pos > 0 and p0 <= pos < p1 and (a != 0 or g != 0):
h_s <- bf16(h_s - a * delta_s * e_s + g * e_s)
target_s = c_s - a * delta_s * ep_s + g * ep_s ;  err_s = |c'_s - target_s|  (c'_s from the bf16 result)
```

**Per-token buffer** (written each step by `ArmModelState.prepare_inputs`; read by the kernel):

| Name | Shape | dtype | Columns |
|---|---|---|---|
| `tok` | `[MAX_T, 5]` | int32 | `TOK_ARM=0, TOK_P0=1, TOK_P1=2, TOK_CROW=3, TOK_VROW=4`; all `-1` for padding |
| `tok_staging` | `UvaBufferPool((MAX_T, 5), torch.int32)` | host | staging for `copy_to_gpu(np, out=tok[:Tpad])` |

**How the wrapper finds the batch's arm ids inside CUDA graphs.** `tok` is a persistent buffer allocated before
capture. The wrapper passes the whole `st.tok` and the static `T = post.shape[0]` (the padded size of the captured
graph); the kernel reads rows `0..T-1`. `prepare_inputs` runs in Python before every step, also before replays, and
fills `tok[:num_tokens_after_padding]` in place, padding rows with -1. `prepare_dummy_inputs` sets `tok.fill_(-1)`, so
profiling and capture run with nothing edited. Positions come from the layer's own `positions` argument (device), not
from `tok`.

**Counters** (device, updated by the kernel; read and reset by `dsv41_stats`):

| Name | Shape | dtype | Rule |
|---|---|---|---|
| `tokens_seen` | `[A]` | int64 | +1 per token with `arm >= 0`, counted at slot 0 only |
| `tokens_edited` | `[A, I]` | int64 | +1 per edited token |
| `sum_abs_delta` | `[A, I]` | fp32 | `+ sum_s |delta_s|` per edited token |
| `max_err` | `[A, I]` | fp32 | atomic max of `max_s err_s` over edited tokens |
| `nonfinite` | `[A, I]` | int64 | +1 per token (arm >= 0) whose `c` or `c'` is not finite (rows the kernel computes: edited, captured, and every real row at the first and last installed slot) |
| `cast_violations` | `[A, I]` | int64 | +1 per edited token where some stream has `\|c'_s - target_s\| > 0.004 * sum_i \|p_si\| max(\|h_si\|, \|h'_si\|)` (the bf16 rounding bound; `ops.CAST_BOUND`) |

Host counters on `EditState`: `invalid_arm_requests` (int), `global_arm` (int, -1 = per request), `arms_sha` (str,
`""` until the first load). Capture overflow counters live on `CaptureRings`.

## 4. Per-request fields

**`vllm_xargs` keys** (all optional; parsed by `arms.parse_xargs`):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `arm` | int | 0 | arm slot `0..A-1`; anything else -> arm 0 and `invalid_arm_requests += 1` |
| `edit_from` | int | 1 | edit window start (absolute position, inclusive); pos 0 is never edited |
| `edit_to` | int | `2**31-1` | edit window end (exclusive) |
| `capture` | str | absent | `"coef"` or `"vec"` (`vec` also writes coef rows) |
| `cap_from`, `cap_to` | int | 0, `2**31-1` | positions to capture, `[cap_from, cap_to)` |
| `cap_pos` | list[int] | absent | explicit positions (<= 4096); overrides `cap_from/cap_to` |
| `tag` | str | `""` | private label, stored only in the capture map (tools only, never from sandboxes) |

**`cache_salt` rule** (built only by `tools/arms.py:request_fields`; the provider and every tool use it):
- arm 0, no capture: no salt (all sham traffic, plain agent runs included, shares one cache pool);
- arm != 0: `sha256(f"{arms_sha}|{arm}|{edit_from}|{edit_to}").hexdigest()[:24]`;
- any capture request: `sha256(f"{arms_sha}|{arm}|{edit_from}|{edit_to}|cap|{nonce}").hexdigest()[:24]` with a fresh
  random nonce (forces a full recompute so every position is captured).

`request_fields(arm=0, edit_from=1, edit_to=2**31-1, *, arms_sha, capture=None, cap_from=None, cap_to=None,
cap_pos=None, tag=None, nonce=None) -> {"vllm_xargs": {...}, "cache_salt": "..."}` (no `cache_salt` key for plain
sham; `vllm_xargs` omits defaults). Arm ids and salts are opaque; labels live only in the arms sidecar JSON.

**Provider passthrough.** If `DSV41_VLLM_XARGS` is set (JSON object) it goes into the body as `vllm_xargs`; if
`DSV41_CACHE_SALT` is set it goes in as `cache_salt`. Default off. A sandbox launcher that cleans the environment
(for example Apptainer `--cleanenv`) must pass both explicitly, from `request_fields()`.

**Proxy rules** (`tools/proxy.py`). Forward only `POST /v1/completions`, `GET /v1/models`, `GET /health`,
`POST /tokenize`, with the client's headers (API key) and body unchanged. Answer 400 to a completion whose
`vllm_xargs.arm` is not 0 and that has no `cache_salt`, and 400 to any completion carrying `vllm_xargs.capture`
(captures go to `:8101` from the GPU node only). 400 when a `cache_salt` is not `request_fields(arm, edit_from,
edit_to, arms_sha=S)`'s salt for an arms sha S loaded in this engine boot (`$DSV41_ARMS_DIR/LOADED_SHAS`, written by
`tools/client.py load_arms`). 503 for every arm but k while `tools/client.py set_global_arm(k)` holds k >= 0
(`$DSV41_ARMS_DIR/GLOBAL_ARM`). 404 for everything else. The proxy reads the engine boot once at startup: restart it
after every engine reboot. Upstream connections are unlimited (aiohttp's default cap is 100).

## 5. Engine-side classes and the per-step path

**`arms.ArmModelState(DeepseekV41ModelState)`**
- `__init__(vllm_config, model, encoder_cache, device)`: super, then host arrays of size `max_num_reqs`:
  `req_arm`, `req_p0`, `req_p1` (int32 numpy), `req_cap: list[CapSpec | None]`, `req_id: list[str]`.
- `add_request(req_index, new_req_data)`: super, then `parse_xargs(new_req_data.sampling_params.extra_args, A)` into
  the arrays at `req_index` (the request id is `new_req_data.req_id`).
- `prepare_inputs(input_batch, req_states)`: super; `n = num_reqs`, `idx = idx_mapping_np[:n]`,
  `cnt = num_scheduled_tokens[:n]` (assert `cnt.sum() == num_tokens`), `start = num_computed_tokens_np[:n]`; host
  `[Tpad, 5]` int32 filled with -1; columns 0-2 = `np.repeat(...[idx], cnt)` (arm column = `global_arm` for every real
  token when `global_arm >= 0`); on rank 0, `st.capture.assign(...)` fills columns 3-4; then
  `tok_staging.copy_to_gpu(host, out=st.tok[:Tpad])`. Returns super's dict unchanged.
- `prepare_dummy_inputs(num_reqs, num_tokens)`: super, then `st.tok.fill_(-1)`.

**`patch._make_forward(slot, vslot, orig)`**: `post = mhc_post_tilelang(x, residual, post_mix,
res_mix)`; `ops.edit_stack_(post, positions, pre_mix, slot, vslot, STATE)`; return `orig(post, positions, input_ids,
pre_mix, None, None, None, engram_hashes, engram_mask, capture_previous_aux=False,
mega_gate_metadata=mega_gate_metadata)`. `slot` = index into `installed`; `vslot` = index into capture layers or -1.
No Python branch on arm state inside (graph safety).

**`ops.edit_stack_(post, positions, pre_mix, slot, vslot, st)`**: one launch, grid = `T`. Reads `st.tok[t]`, the
tables at `[arm, slot]`, `positions[t]`; edits `post` in place; updates the counters; when `tok[t, CROW] >= 0` writes
`st.capture.coef[crow, slot] = [c (4), c' (4), pre_mix[t] (4)]`; when `vslot >= 0 and tok[t, VROW] >= 0` writes
`st.capture.vec[vrow, vslot] = mean_s(h_s)` (pre-edit, bf16).

## 6. Capture format (`capture.CaptureRings`, rank 0; other ranks allocate empty rings and never assign rows)

| Device ring | Shape | dtype |
|---|---|---|
| `coef` | `[COEF_ROWS, I, 3, 4]` | fp32 (`[:, :, 0]` = c before, `1` = c' after the bf16 cast, `2` = carried pre-mix) |
| `cpos` | `[COEF_ROWS]` | int32, the device position the kernel saw for the row (written at slot 0) |
| `vec` | `[VEC_ROWS, N_CAP, 5120]` | bf16, pre-edit stream mean at the capture layers |

Host row map (numpy, appended by `assign`): per coef row `pos`, `arm`, `req` (index into `req_ids`), `tag` (index into
`tags`); the same for vec rows. Row allocation is a bump pointer; a full ring increments `coef_overflow` /
`vec_overflow` and assigns -1. Rows are assigned at the step the token is computed (prefix-cache hits are never
captured, hence the unique capture salt).

**Dump** `dsv41_capture_dump(path)` writes `path` (`.safetensors`: `coef [n, I, 3, 4]`, `coef_pos`, `coef_pos_dev`,
`coef_arm`, `coef_req`, `coef_tag` int32 `[n]`, `vec [m, N_CAP, 5120]`, `vec_pos`, `vec_arm`, `vec_req`, `vec_tag` int32 `[m]`) and
`path + ".json"` (`layers`, `capture_layers`, `arms_sha`, `req_ids`, `tags`, overflow counts, time), then resets the
rings. Paths must be under `DSV41_CAPTURE_DIR`.

## 7. Arms file (`tools/arms.py` writes, `arms.load_arms` reads)

`<name>.safetensors` with exactly: `layers` int32 `[I]`, `p` and `e` fp32 `[A, I, 4, 5120]`, `mu` fp32 `[A, I, 4]`,
`a` and `g` fp32 `[A, I]`, `mc` int32 `[A]` (~68 MB at A=16, I=26). `arms_sha` = sha256 hex of the file. Sidecar
`<name>.json`: `{"sha256", "labels": {arm: private label}, "recipe": {...}}`; labels never go on the wire.
`load_arms` checks the sha, shapes, `layers == installed`, arm 0 all zero and finite values, then
`synchronize; copy_; synchronize`, recomputes `ep`, sets `arms_sha`, and returns float64 checksums.

## 8. RPC (`rpc.EditExt`, via `POST :8101/collective_rpc`; args are strings; each rank returns a dict with `"rank"`)

| Method | Args | Returns (besides `rank`) |
|---|---|---|
| `dsv41_installed` | - | `layers`, `capture_layers`, `fuse_mhc_all_reduce` (must be False), `model_fuse_mhc_all_reduce` (False), `deferred_finalize` (`[]`), `all_reduce_mhc_built` (False), `mhc_stream` (bool), `n_arms`, `n_inst`, `impl`, `sequence_parallel`, `spec_decode`, `max_t`, `coef_rows`, `vec_rows`, `arms_sha`, `boot` |
| `dsv41_load_arms` | `path`, `sha256` | `arms_sha`, `checksums` {`p`,`e`,`mu`,`a`,`g`,`mc`: float64 sums} |
| `dsv41_stats` | `reset` (`"0"`/`"1"`) | the counters as nested lists, `invalid_arm_requests`, `global_arm`, `coef_rows_used`, `vec_rows_used`, `coef_overflow`, `vec_overflow`, `arms_sha` |
| `dsv41_capture_dump` | `path` | `path` (None on ranks != 0), `coef_rows`, `vec_rows` |
| `dsv41_set_global_arm` | `k` (`"-1"` = per request) | `global_arm` |

Client rule (`tools/client.py`): every RPC must return 4 results with equal `layers`, `arms_sha` and checksums, or
the tool raises.

## 9. Test data (`data/`, built by `tools/build_data.py`; not in the repo)

- `plumbing_texts.json`: `{"tokenizer_sha256", "A": [[ids], ...30], "B": [...30], "C": [...]}` (A vLLM source, B
  English prose, C agent transcripts; ~400 tokens each).
- `floor_prompts.json`: `{"tokenizer_sha256", "prompts": [{"id", "kind": "generic"|"agent", "ids": [...]}, ...40]}`
  (20 generic of 0.5-3k tokens with ~6 of <= 1000; 20 agent of 6-30k; engine checks only, never claims).

## 10. Details

**RPC results.** Every RPC result carries `rank`, `layers`, `arms_sha` and `boot`, so `tools/client.py` can compare
ranks on any call. `dsv41_installed` also returns `installed` (False, with `rank`/`layers`/`arms_sha` only, when the
patch is not active), `wrapped` (installed outputs whose successor layer holds the patched `forward`), `global_arm`,
and the direct fusion evidence (`model_fuse_mhc_all_reduce`, `deferred_finalize`, `all_reduce_mhc_built`). The layer
flags are `null` when the decoder cannot be found. `dsv41_stats` also returns `invalid_capture_requests`, `t` and
`nonfinite_counter_cells`. `reset="1"` returns the counters **before** zeroing them, and zeroes only the device
counters; the host counters keep counting. Non-finite float counters go out as -1.0 (JSON cannot carry NaN; no real
counter is negative).

**Request parsing.** A bad edit window (`edit_from`/`edit_to` not an int in `[0, 2**31-1]`, or `edit_from >
edit_to`) makes the request invalid, like a bad arm: it runs as plain sham with no capture and counts in
`invalid_arm_requests`. Bools, floats and numeric strings are rejected as ints. A bad capture field drops only the
capture (`invalid_capture_requests += 1`) and keeps the arm. `ArmModelState.prepare_inputs` marks a batch row as real
only when its request id equals the one stored at `add_request`; dummy batches (profiling, warmup) have random ids,
so their rows are all -1.

**Counters and rings.**
- `nonfinite` counts rows the kernel computes: edited rows, captured rows, and every real row (sham included) at the
  first and the last installed slot.
- `CaptureRings` keeps one trash row at the end of each ring (`coef_buf [COEF_ROWS+1, ...]`, `vec_buf [VEC_ROWS+1,
  ...]`); `coef` / `vec` are views without it. Disabled rings (ranks != 0) hold only the trash row. The torch path
  writes uncaptured tokens there.
- The kernel writes each coef row's device position into `cpos` (dumped as `coef_pos_dev`); `tools/arms.py
  load_dump` raises when it differs from the host map's `coef_pos` on any row.
- A preempted and recomputed request gets fresh rows for the same positions: keep the last row per `(req, pos)`.
- `dsv41_capture_dump` refuses a path outside `DSV41_CAPTURE_DIR`, the folder itself, and an existing file. Dumps
  hold `vec` as bf16; `tools/arms.py load_dump` reads them through torch and returns float32.

**Kernel.** `ops.edit_stack_triton_` is the raw kernel launch (it also runs under `TRITON_INTERPRET=1` on CPU
tensors); `edit_stack_` dispatches, and CPU tensors always take the torch path. `warm_kernels` compiles the kernel at
a few sizes before graph capture, then resets `tok`, counters and rings.

**Patch checks.** `patch.apply()` checks the decoder-layer `forward` signature and the module layout and raises if
vLLM changed. `_model_init` asserts fusion off, no sequence parallel, no speculative decoding, PP = 1, no
micro-batching (`parallel_config.use_ubatching`) and no prefill/decode context parallel, then logs one line starting
`dsv41_intervene: fusion off, 26 edit slots on outputs [...]`, which the install gate greps.

**Security.** `:8101` checks the API key on every route but `/health` (`--middleware
dsv41_intervene.auth.require_key`). vLLM itself guards only `/v1...`, and with `VLLM_SERVER_DEV_MODE=1` the dev
routes (`/collective_rpc`, `/reset_prefix_cache`, `/sleep`, ...) would otherwise be open to every process that can
reach the port. The middleware fails closed: without `VLLM_API_KEY` only `/health` answers.

**Client.** `load_arms` / `set_global_arm` record state for the proxy; `wait_idle()` (via `/metrics`) runs before every
counted block; `generate` refuses `capture` with `n > 1` (children share the salt, so only the first child computes
and captures the full prompt blocks).

**bf16 cast.** The edit lands on a state that is already on the bf16 grid. Any per-element step smaller than half a
grid spacing rounds back to the old value, so the cast shrinks the edit systematically; this is not noise. On N(0, 1)
states with a unit direction and |delta| up to ~4 sd, `|c' - target|` reached 0.035 sd in the CPU test, while the fp32
result before the cast is exact (<= 1e-5 sd). An sd threshold can therefore fail by construction on a low-variance
direction; `cast_violations` (section 3) is the direction-independent test.

**Prefix cache.** vLLM 0.31 sets `skip_reading_prefix_cache` whenever `prompt_logprobs` is set
(`sampling_params.py:559-565`), so a teacher-forced request never hits the cache. The prefix-cache gate checks the hit
with a plain request and compares 32 greedy decode logprobs (hit vs cold).

**NaN on batched requests.** The base model returns NaN logprobs on some batched prefill / `prompt_logprobs` requests,
on the plain server too: vLLM answers 400 "Out of range float values are not JSON compliant: nan" (17/40 and 8/40 in
two batched runs, 0/40 one by one). `client.score` retries up to 8 times and records `nan_retries`. The rows of a
failed attempt stay in the capture rings, so tools drop every row of a request with any non-finite coefficient
(`tools/arms.py nonfinite_requests`) when they estimate mu or the info numbers.
