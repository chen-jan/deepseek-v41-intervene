# Usage and reference

Full reference for the scripts, the engine, arms, captures, gates and known issues. The short version is in the
README.

## Hardware and software

| | Tested | Notes |
|---|---|---|
| GPUs | **4x NVIDIA B200** (183 GB each) | Everything here was measured on this setup. |
| | 4x H200 (141 GB) | **Untested.** Weights take ~80-87 GB per GPU at TP4, so it should fit with a smaller KV cache. The B200-specific FlashInfer kernel cache (`sm100a`) is not used there. |
| Host memory | 700 GiB for the job | The two Engram tables (layers 1 and 14, two 101.5 GB shards) are offloaded to CPU: ~285 GB of shared memory, ~70 GB per rank. |
| CPUs | 32 for the server step | With 24, rank 0 got 16 CPUs and ranks 1-3 shared 8; loading took about twice as long. |
| Disk | 476 GiB checkpoint | Local NVMe if you can. Page cache of the Engram shards can starve a reboot: `tools/dropcache.py` drops it. |
| Software | Python 3.12, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.31.0, FlashInfer 0.7.0.post1 (+ `flashinfer_jit_cache`, `flashinfer_jit_cache_sm100a`), tilelang 0.1.12, triton 3.7.1, transformers 5.17.0 | `scripts/build_env.sh` pins vLLM and FlashInfer; the rest comes with vLLM. |

The model: 40 layers, hidden size 5120, 4 residual streams (mHC), 384 experts with top-6 routing (MXFP4), FP8
attention, Engram tables at layers 1 and 14.

## Install

```bash
cp .env.example .env              # set DSV41_ROOT (work folder) and DSV41_MODEL_DIR (checkpoint); paths only
bash scripts/build_env.sh         # venv at $DSV41_ROOT/envs/vllm031 (needs uv)
bash scripts/download_model.sh    # 476 GiB into $DSV41_MODEL_DIR (ENCODING_ONLY=1: just the chat encoding files)
bash scripts/make_key.sh          # API key file, mode 0600, never printed
```

Every script sources `scripts/common.sh`. It reads `.env`, puts every cache (pip, uv, Triton, inductor, FlashInfer,
TileLang, Hugging Face, tmp) under `$DSV41_CACHE`, and stops with a clear message if a required variable is missing.
`DSV41_SETUP_HOOK` can point at a file your site needs sourced first (for example a `module load cuda` line).
`examples/slurm/serve.sbatch` is a Slurm template with placeholders.

## Serve the plain model

```bash
bash scripts/serve_plain.sh       # 0.0.0.0:8100, model name "dsv41", key from $DSV41_KEY_FILE
```

Flags: TP 4, `--gpu-memory-utilization 0.92`, `--max-model-len 131072`, prefix caching, `--max-num-seqs 512`,
`--max-num-batched-tokens 32768`, `--enable-prompt-tokens-details`. Extra arguments are passed to `vllm serve`.

The first boot takes about 27 minutes (weight load ~160 s, compile, and FlashInfer autotuning); a reboot with warm
caches takes ~8 min to load plus ~9 min of autotuning (autotune runs on every graph boot).

**Throughput** (output tok/s, `vllm bench serve`, random 8,192-token prompts, 1,500 output tokens; `tools/bench.sh
native`):

| Concurrency | 1 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `serve_plain.sh` flags | 175.7 | 1,826.9 | 3,804.9 | 5,345.7 (TTFT 1.9 s, TPOT 22.3 ms) | 6,694.0 |
| `bench_configs.sh` flags (32k context, 256 seqs, 16k batched tokens) | 185.3 | 1,969.7 | 3,994.8 | 6,873.0 | 5,848.4 (prefill-bound) |

Pure decode with 256 running requests reached 13,592 tok/s; prefill ~60k tok/s. `scripts/bench_configs.sh` also has
expert-parallel and speculative-decoding (MTP) configurations; those were never measured.

## Serve the edit engine

Stop the plain server first (both need all 4 GPUs), then:

```bash
bash scripts/serve_engine.sh      # 127.0.0.1:8101 only; tools/proxy.py serves other machines on :8100
```

It runs the plain server's flags plus `PYTHONPATH=<repo>`, `DSV41_ENGINE=1`, `VLLM_SERVER_DEV_MODE=1`,
`--worker-extension-cls dsv41_intervene.rpc.EditExt` and `--middleware dsv41_intervene.auth.require_key`. These stay
outside vLLM's compile hash, so the plain server's compile and autotune caches stay valid. Never `pip install` this
repo into the shared venv: the plugin would then load into the plain server too (it is a no-op without
`DSV41_ENGINE=1`, but keep it out).

Settings (environment):

| Variable | Default | Meaning |
|---|---|---|
| `DSV41_EDIT_LAYERS` | `12,14-38` | layer outputs (0-based) that can be edited: 26 slots. Never 0, 13 or 39. |
| `DSV41_N_ARMS` | `16` | arm slots; arm 0 is the sham |
| `DSV41_CAPTURE_LAYERS` | `12,15,18,22,25,26,27,30,34` | layers with stream-vector capture |
| `DSV41_COEF_ROWS`, `DSV41_VEC_ROWS` | `1048576`, `32768` | capture ring sizes (rank 0) |
| `DSV41_EDIT_IMPL` | `torch` | `torch` op or `triton` kernel (Triton differed from torch by up to 4 ulp on GPU; every boot so far used torch) |
| `DSV41_MHC_OVERLAP` | `1` | `0` turns off the mHC side stream (fallback if only that graph path misbehaves) |
| `DSV41_EAGER` | `0` | `1` = no CUDA graphs, no autotune; one execution path, ~10% of native throughput |

How it works, in one paragraph: vLLM normally fuses each layer's mHC output step with the next all-reduce. The patch
turns that fusion off, so every layer output exists as a `[tokens, 4, 5120]` bf16 stack. It wraps
`layers[L+1].forward` for each edit slot L: the wrapper materializes the stack, edits it in place with one kernel that
reads each token's arm from a persistent buffer, and calls the original layer. A per-step hook fills that buffer from
the requests' `vllm_xargs`, so the same captured CUDA graph serves every mix of arms. Details:
`docs/INTERFACE.md`.

**Engine throughput** (sham traffic, same benchmark as above): **46.9% of native at concurrency 128** (2,506 tok/s;
44-47% from c = 1 to 256). Eager mode: 10.2% at c = 128. Graph boots took ~30 minutes to become healthy (load, graph
capture, ~9-11 min of autotuning).

## Arms and requests

**The edit.** For stream s of a token, with read direction `p`, write direction `e`, reference `mu`, removal
strength `a` and add amount `g`: `c_s = h_s . p_s`, `delta_s = c_s - mu_s`, then
`h_s <- bf16(h_s - a * delta_s * e_s + g * e_s)`. With `e = p = d`, `a = 1`: project `d` out to its mean. With
`e = c` (another direction): a matched shift. `a = 0, g > 0`: add. `a = g = 0`: measure only. A mean-coupled arm uses
one `delta` for all 4 streams. Position 0 (BOS) is never edited.

**Arms file.** Build one from a recipe and load it into the running engine:

```bash
python tools/arms.py build --recipe recipe.json --out $DSV41_ARMS_DIR/myarms   # myarms.safetensors + .json
python tools/client.py load $DSV41_ARMS_DIR/myarms.safetensors                 # all 4 ranks; prints arms_sha
```

```json
{"layers": [12, 14, 15, "... all 26 installed slots"], "n_arms": 16,
 "arms": [{"arm": 1, "label": "remove d at 26", "slots": [{"layers": [26], "kind": "project", "d": "d.npy", "mu": "mu.json"}]},
          {"arm": 2, "label": "add 4 sd of d at 26", "slots": [{"layers": [26], "kind": "add", "d": "d.npy", "mu": "mu.json", "alpha_sd": 4}]}]}
```

Kinds: `measure`, `project`, `project_mc` (mean-coupled), `shift` (needs `c`), `add`. Directions are `.npy` files
(`[5120]`, `[4, 5120]` or per slot), `.npz` keyed by layer, or inline lists. `mu.json` comes from
`tools/arms.py mu` on a coefficient capture. The file is ~68 MB at 16 arms and 26 slots; its sha256 is the
`arms_sha`. Labels stay in the sidecar JSON and never go on the wire. Loading new arms replaces all of them.

**Per-request fields.** A completion request picks its arm with `vllm_xargs` and must carry the matching
`cache_salt`, so prefix-cache blocks never mix between arms:

```bash
python tools/arms.py fields --arms-sha <sha> --arm 1 [--edit-from 1 --edit-to N]
# -> {"vllm_xargs": {"arm": 1}, "cache_salt": "<24 hex>"}  (put both into the /v1/completions body)
```

`cache_salt = sha256(f"{arms_sha}|{arm}|{edit_from}|{edit_to}")[:24]`; plain sham requests carry no salt. Optional
`vllm_xargs`: `edit_from`/`edit_to` (absolute token positions, `[from, to)`), `capture` (`"coef"` or `"vec"`),
`cap_from`/`cap_to` or `cap_pos`, `tag`. A bad arm or edit window makes the request run as sham (counted in
`invalid_arm_requests`); a bad capture field drops only the capture. Neither returns an error. From Python: `tools.arms.request_fields(...)`. The provider takes them as `DSV41_VLLM_XARGS` and
`DSV41_CACHE_SALT`.

## Captures and RPC

Control goes through vLLM's `/collective_rpc` route on `:8101` (key required). `tools/client.py` wraps it and checks
that all 4 ranks agree:

```bash
python tools/client.py installed            # patch on all ranks, edit slots, fusion off, impl, ring sizes, boot id
python tools/client.py stats [--reset]      # per-arm counters: tokens_seen, tokens_edited, sum_abs_delta, max_err, nonfinite, cast_violations
python tools/client.py dump $DSV41_CAPTURE_DIR/run1.safetensors   # write and clear the capture rings
python tools/client.py global-arm K         # tell the proxy to admit only arm K (-1 = per request)
```

A capture request gets a fresh salt with a random nonce (`request_fields(..., capture="coef")`), so every position is
computed and captured. Coefficient rows hold `c` before, `c'` after the bf16 cast, and the carried pre-mix, per slot
and stream; vector rows hold the pre-edit stream mean at the capture layers (bf16). `tools/arms.py load_dump` reads a
dump. Format: `docs/INTERFACE.md` section 6.

**Proxy.** `python tools/proxy.py --listen 0.0.0.0:8100 --upstream 127.0.0.1:8101` forwards only completions,
models, tokenize and health, refuses captures, refuses a salt that does not match an arms file loaded in this boot,
and answers 404 to every control route. Restart it after every engine reboot. (Never run on GPU yet; tested on CPU.)

## Gates

`tools/gates.py` checks a booted engine: install, build (the gate arms), sham vs native, positive control on every
execution path, decode vs prefill, arm isolation in mixed batches, prefix-cache isolation, arm reload, projection
exactness, info, the retry gates (repeat, outliers, decode effect) and the proxy. Each writes a JSON verdict.
You need your own test prompts first (`tools/build_data.py`; the repo ships none) and a native noise floor from the
plain server (`tools/client.py floor`). The rules, how to run them, and how to read each result are in
`docs/GATES.md`. The rules were fixed before any GPU run; do not loosen one after it fails.

## Known issues

- **Sign-off state:** see `docs/RESULTS_2026-10-28.md` (passed on a follow-up check, not on the pre-written gates).
- **NaN on batched teacher-forced requests, base model too.** Some batched `prompt_logprobs` requests return
  NaN (vLLM answers 400 "Out of range float values are not JSON compliant: nan"): 17/40 and 8/40 in two batched runs,
  0/40 one by one, on the plain server. The client retries up to 8 times and records it; tools drop rows of such
  requests from captures.
- **Agent-length prompts are noisy run to run** even on the plain server (top-1 agreement ~82%, p95 |Δ logprob| ~3.5).
  Teacher-forced distances jump past position 512 (the sparse attention's top-k of 512).
- Fusion is off, and steps over 1,024 tokens run eager. The 53% throughput cost has not been broken down.
- One direction per arm per slot (rank 1). No edits at outputs 0-11, 13, 39. No speculative decoding with the
  engine.
- Do not set `VLLM_ENGINE_READY_TIMEOUT_S`: it is part of vLLM's compile hash and forces a cold compile.
- `fastsafetensors` runs out of memory on the Engram shard; use the default loader.
- Not run on GPU yet: the proxy, arm and mixed-arm throughput benches.

## Tests (no GPU)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements-test.txt
python -m pytest -q                       # engine and tools (GPU tests and the vLLM install check skip)
bash tests/run_mock_gates.sh              # the whole gate suite against a CPU mock engine, synthetic prompts
cd provider && DSV41_OFFICIAL_DIR=/folder/with/encoding_and_tokenizer python -m pytest -q   # encoding/ + tokenizer.json
```

`tests/test_gpu_kernels.py` needs one GPU; `tests/test_install_dry.py` needs vLLM 0.31 installed; the CPU run of the
Triton kernel needs `triton` (in `requirements-test.txt`). The provider tests need the checkpoint's `encoding/` folder
and `tokenizer.json` (`ENCODING_ONLY=1 bash scripts/download_model.sh`); the harness tests also need `HARNESS_SRC`
(`docs/PROVIDER.md`).

## Files and provenance

All code here was written for this project and is under the MIT licence (`LICENSE`), except where the table says
otherwise. Nothing is copied from vLLM, DeepSeek or another project; the engine imports and wraps vLLM 0.31's own
modules (Apache-2.0) at runtime, and the provider loads DeepSeek's `encoding.py` from the checkpoint at runtime.

| Path | What | Origin |
|---|---|---|
| `dsv41_intervene/__init__.py` | constants, layer parsing, env paths | own |
| `dsv41_intervene/patch.py` | the vLLM patch: fusion off, layer wrappers, model-state hook | own; wraps vLLM internals (Apache-2.0) at runtime, no vLLM code copied |
| `dsv41_intervene/ops.py` | edit kernel (Triton) and torch op | own |
| `dsv41_intervene/arms.py` | device state, request parsing, arms loading | own |
| `dsv41_intervene/capture.py` | capture rings and dumps | own |
| `dsv41_intervene/rpc.py` | worker extension (`dsv41_*` RPC) | own |
| `dsv41_intervene/plugin.py` | `vllm.general_plugins` entry point | own |
| `dsv41_intervene/auth.py` | API-key middleware | own |
| `tools/arms.py`, `client.py`, `proxy.py`, `gates.py`, `retry_summary.py`, `build_data.py`, `dropcache.py`, `bench.sh`, `__init__.py` | arms files, client, proxy, gates, data builder, benches | own |
| `scripts/*.sh` | env build, download, key, serve, bench | own (call `uv`, `hf` and `vllm` CLIs) |
| `tests/*` | CPU unit and integration tests, CPU mock engine, GPU kernel tests | own |
| `provider/deepseek_v41_vllm_provider.py` | chat-encoding provider | own; follows the provider interface of [agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) (MIT) and imports it when installed; no code copied. Loads DeepSeek's `encoding.py` at runtime (not included). |
| `provider/tests/*` | provider tests with a mock vLLM server | own; the template test reads DeepSeek's tools prompt text from the official `encoding.py` at runtime instead of copying it |
| `docs/*`, `README.md`, `.env.example`, `examples/slurm/serve.sbatch`, `pyproject.toml`, `pytest.ini`, `requirements-test.txt` | docs and config | own |

**Not included:** the model weights; DeepSeek's `encoding/encoding.py`, `tokenizer.json` and the checkpoint's
`inference/examples` (download them from the model repo, which is MIT-licensed by DeepSeek); any prompts,
transcripts, captures, activations or logs; the agent harness itself.
