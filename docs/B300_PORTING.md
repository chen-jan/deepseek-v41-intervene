# Porting to B300 (sm_103): answers

What we know about moving the DeepSeek V4.1 Flash setup (plain server, edit engine, lens engine) from 4x B200 (sm_100)
to B300 (sm_103). Everything below was checked on 28 Oct 2026 against the B200 setup: its files, its venvs and its
logs. Nothing here has run on a B300. Items marked **unverified** are open. `$DSV41_ROOT` is the work folder
(`.env.example`).

## 1. Compiled kernels and the GPU arch list

**Short answer:** none of our own code is compiled ahead of time, and nothing of ours sets `TORCH_CUDA_ARCH_LIST`
(a grep of this repo and the lens scripts finds no match). Every compiled kernel comes from the vLLM, torch or
FlashInfer wheels, or is JIT-compiled at runtime.

| Part | Where the kernel comes from (B200 log line) | Arch in the binary |
|---|---|---|
| MXFP4 MoE | FlashInfer TRT-LLM: `Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend`, `TrtLlmMxfp4ExpertsModular`. MARLIN (W4A16) is what vLLM picks on H200, according to the lens usage notes. MARLIN never ran on B200. | FlashInfer, see the last row |
| FP8 / NVFP4 DS-MLA attention | vLLM wheel, `vllm/_flashmla_C.abi3.so`. The plain and edit engines log `FLASHMLA_MEGA_ATTN_DSV41` and `nvfp4_ds_mla` KV. The lens engine logs `FLASHMLA_SPARSE_DSV41` and `fp8_ds_mla` KV. | cubins sm_90a + sm_100, no PTX (`cuobjdump`) |
| mHC (4-stream mix) | vLLM `model_executor/layers/mhc.py`, TileLang ops (`mhc_pre_tilelang`, `mhc_post_tilelang`, ...), JIT-compiled by tilelang 0.1.12 into `TILELANG_CACHE_DIR` | built at runtime |
| Engram | vLLM `engram.py`. Each table is offloaded to pinned host memory (section 6). | **unverified**: kernel origin not checked |
| Other vLLM ops | `_C_stable_libtorch` (cubins sm_75 to sm_120 incl. sm_100; PTX sm_80/89/90), `_moe_C_stable_libtorch` (cubins incl. sm_100; PTX sm_80), `_deepselect_C` and `_qutlass_C` (sm_100 only) | as listed |
| torch 2.13.0+cu130 | `torch._C._cuda_getArchFlags()` = `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | no sm_103 entry |
| Our edit op | `DSV41_EDIT_IMPL=torch`, the default in `scripts/serve_engine.sh` and used on every GPU boot: plain torch ops, no custom kernel. `triton` selects `_edit_kernel` in `dsv41_intervene/ops.py`, a Triton JIT kernel. | built at runtime |
| Stack-pre / stream-mean capture (lens) | vllm-metamodels fork (`vllm_lens` 1.1.0.post7): Python hooks only. The tarball holds no `.cu`, no `@triton.jit` and no `load_inline`. | - |
| FlashInfer | `flashinfer-python` 0.7.0.post1, `flashinfer-jit-cache` 0.7.0.post1+cu130 (generic) and `flashinfer-jit-cache-sm100a` (B200 kernels). The autotune cache sits in an arch folder (`.../flashinfer_autotune_cache/0.7.0.post1/100a/`). | sm100a provider |

**What changes for sm_103:**
- **FlashInfer kernel cache.** Install `flashinfer-jit-cache-sm103a==0.7.0.post1` from `https://flashinfer.ai/whl/cu130`.
  The index lists x86_64 and aarch64 wheels (checked 28 Oct). `scripts/build_env.sh` installs only the sm100a one.
- **Autotune.** Expect a full FlashInfer autotune on the first boot, since the cache folder is per arch. It took ~6-11
  min per boot on B200.
- **Triton.** Triton 3.7.1 maps capability 103 to `sm_103a` (`sm_arch_from_capability`). It compiles with its bundled
  `ptxas-blackwell` (CUDA 13.1), which lists sm_103/sm_103a/sm_103f. This is a static check only: no Triton kernel
  has been run on sm_103.
- **vLLM's capability checks.** `flashmla_sparse.py` accepts compute capability major 9 or 10, requires major 10 for
  `nvfp4_ds_mla`, and uses `is_device_capability_family(100)`. A 10.3 device passes these checks on paper.
- **Unverified:**
  - whether the sm_100 cubins in `_flashmla_C`, `_moe_C`, `_deepselect_C` and `_qutlass_C` load on sm_103;
  - which MXFP4 MoE backend vLLM picks on sm_103;
  - whether TileLang 0.1.12 targets sm_103.

  Check the boot log for the `Mxfp4 MoE backend` and `KV cache format` lines.
- **`TORCH_CUDA_ARCH_LIST`** only matters if you build something from source. Nothing in this setup does.

## 2. Image, versions, how to reproduce

There is no Dockerfile. The setup uses uv venvs built by `scripts/build_env.sh` (Python 3.12;
`uv pip install vllm==0.31.0 --torch-backend=cu130`; the FlashInfer jit-cache wheels; `fastsafetensors pytest`).

The two venvs (plain/edit and lens) hold the same versions, read from their `dist-info` folders: torch 2.13.0+cu130,
vllm 0.31.0 (pip wheel), triton 3.7.1, flashinfer-python 0.7.0.post1 + `flashinfer_jit_cache` and
`flashinfer_jit_cache_sm100a` 0.7.0.post1+cu130, tilelang 0.1.12, transformers 5.17.0, nvidia-cuda-runtime 13.0.96,
nvidia-nccl-cu13 2.29.7, nvidia-cutlass-dsl 4.7.1.

The lens venv adds `vllm_lens` 1.1.0.post7, the fork installed `--no-deps` from `vllm_metamodels_dsv41_7ab496a.tar.gz`
(sha256 `c92dd134e7633ea689351f09c3cf5558cb125a591077c509f9ec27f08e0e2707`). Its build script also installs
`tokenizers`, `transformers>=5.16.1,<5.18`, `peft==0.21.*`, `pandas` and `pyarrow`.

**Host setup:** a CUDA 13.0 toolkit on `PATH` for the JIT compilers (`DSV41_SETUP_HOOK`, e.g. `module load cuda/13.0.1`;
the lens env also sets `CC/CXX` to gcc 12.2); `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
Do not set `VLLM_ENGINE_READY_TIMEOUT_S` (part of vLLM's compile hash). For B300, add the sm103a wheel (section 1) and
keep everything else pinned.

## 3. Reference output from the edit/plain server (the request to repeat)

The B200 output is in `docs/reference/engine_b200_sham.json` (two repeats gave identical tokens). Send the same request on B300:

```
POST http://127.0.0.1:8101/v1/completions     # edit engine, sham (or the plain server on :8100); header: API key
{"model": "dsv41", "prompt": [<token ids>], "max_tokens": 50, "temperature": 0.0, "logprobs": 5,
 "ignore_eos": true, "return_tokens_as_token_ids": true}
```

No `vllm_xargs` and no `cache_salt` = arm 0, the sham. Suggested prompt: the 20 ids of the lens reference (section 4),
so both engines see one input. Compare token ids and top-5 logprobs. Two cautions:
- **Noise floor.** Same hardware, same prompts, the plain server's run-to-run noise is median |Δ logprob| 0.013 (generic)
  and 0.059 (agent), with top-1 agreement 0.945 / 0.819 (`docs/RESULTS_2026-10-28.md`). A cross-GPU comparison has no
  floor yet: **unverified**.
- **Engines differ numerically.** The two engines use different attention/KV paths (section 1), so compare like with
  like.

## 4. Lens engine: reference readout (B200, TP4, already produced)

Text `The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is`, ids (BOS prepended)
`[0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102, 294, 10322, 344, 17575, 16, 455, 6102, 294, 14251, 344]`,
positions `[10, 18, 19]`, layers `[22, 26, 30, 34]`, base weights (unmerged).

Stream-mean norms per layer, at positions 10 / 18 / 19:

| Layer | pos 10 | pos 18 | pos 19 |
|---|---|---|---|
| 22 | 61.14 | 65.49 | 51.30 |
| 26 | 92.46 | 88.81 | 87.08 |
| 30 | 102.07 | 112.92 | 120.38 |
| 34 | 134.23 | 162.19 | 158.17 |

Release lens (layer 26), greedy bullets:
- pos 10: `* France is Paris, and the capital of Italy is Rome. Therefore, the capital`
- pos 19: `* Rome (or possibly Brussels). The capital of Italy is Rome. Therefore, the`

The full output is in two files, shared separately (not in this repo):
- `reference.json` holds the ids, the norms, the first 8 values of each vector, and the release-lens and multilayer-lens
  (22/30/34) greedy bullets;
- `means.pt` holds the `[4 layers, 3 pos, 5120]` tensor.

## 5. Lens engine: several layers in one pass

Yes, several layers come out of one pass. One `generate` call captures every requested layer at every requested
position. Per request, `SamplingParams(max_tokens=1, temperature=0, extra_args={"output_residual_stream": [22, 26, 30],
"capture_positions": [...], "lens_early_exit": True})`, after `set_mhc_capture` mode `"mean"`. The result is
`o.activations["residual_stream"]` with shape `[n_layers, n_pos, 5120]`. Driver code: `VllmBackend.capture` (in
`planb/ar_smoke.py`) and `lens_read.run`, which sends prompts in chunks of 400k tokens.

On B200 (TP4), 34,094 positions at layers 22/26/30 took 67 s in one pass. Layer 34 took 73 s in a second pass, only
because it was added later.

`set_lens(adapter_dir)` (the fork's `merge_lora`) merges a lens adapter in 0.2 s; `set_lens(None)` restores the base.
Always capture with the base restored.

The edit engine also captures pre-edit stream means, at `DSV41_CAPTURE_LAYERS`, in one pass (`capture: "vec"`).

## 6. Lens engine: 4 vs 8 GPUs, host RAM

**GPUs.** We ran TP4 on 4x B200, eager, with:
- `gpu_mem 0.85`, `max_model_len 16384`;
- `FLASHMLA_SPARSE_DSV41` attention;
- custom all-reduce off.

Measured on that run: 80.43 GiB of weights per GPU, 61.47 GiB of KV cache (2.34M tokens), 156.5 GB used per GPU. The
lens authors' default is TP8, sized for 8x H200 or 8x B200. We never ran TP8, and nothing has run on B300 (whether TP4
fits there is **unverified**).

**Host RAM.** vLLM resolves `EngramConfig(cpu_offload=True)`, so each Engram table is offloaded to pinned host memory.
At TP4 every rank logs two tables of 23.60 GiB each: 47.2 GiB per rank, 188.8 GiB pinned in total. The lens notes give
11.8 GiB per rank per table at TP8, so about the same total. The edit-engine notes measured ~285 GB of shared memory
(~70 GB per rank).

Our B200 job had a 700 GiB memory limit (`--mem=700G` in `examples/slurm/serve.sbatch`), and that was enough. The two
Engram shards, `model-00047/48-of-00048.safetensors` (101.5 GB each), fill ~203 GB of page cache. That page cache counts
against the job: drop it with `tools/dropcache.py <the two shards>` before a reboot. On B200, ranks 1-3 loaded in
950-1,020 s without the drop and in 486-490 s with it.

## 7. Arms file: your own direction per layer over a band, plus mu

**Shapes** (`tools/arms.py`, `dsv41_intervene/ops.py`). The tables are fp32: `p`, `e` `[A, I, 4, 5120]`; `mu`
`[A, I, 4]`; `a`, `g` `[A, I]`; `mc` `[A]` int32. `A` = 16 arms and `I` = the 26 installed slots (`12,14-38`); the
file's `layers` must equal the booted slots.

**Directions.** `d` is `[5120]` (stream-mean space, copied to all 4 streams) or `[4, 5120]` (one row per stream). Each
stream row is normalized to unit length. Per token and stream, `c_s = h_s · p_s`, and the edit is
`h_s <- bf16(h_s + (g - a (c_s - mu_s)) e_s)`. So `mu` is in coefficient units: the mean of `h_s · unit(d_s)`, one
value per stream.

**Recipe inputs.** A recipe `d` is an inline list, a `.npy` of shape `[5120]`, `[4,5120]`, `[26,5120]` or
`[26,4,5120]` (first axis = all 26 slots), or a `.npz` keyed by layer (`"22"`, `"23"`, ...).

`mu` can be a scalar, `[4]`, or a `mu.json` with `{"mu": {"<layer>": [4 floats]}, "sd": {"<layer>": [4 floats]}}`.
That is the output of `tools/arms.py mu` on a coefficient capture, so it carries `sd` too; `alpha_sd` needs that `sd`.

Kinds: `project` (`e = p = d`, `a = lambda`), `add` (`g = alpha` or `alpha_sd * mean(sd)`), `shift` (needs `c`),
`project_mc` (same direction on all 4 streams), `measure`.

```python
import json, numpy as np
band = range(22, 31)                                    # outputs 22..30
np.savez("dirs.npz", **{str(L): my_dir[L] for L in band})   # my_dir[L]: [5120] or [4, 5120]
json.dump({"mu": {str(L): my_mu[L] for L in band}, "sd": {str(L): my_sd[L] for L in band}}, open("mu_mine.json", "w"))
json.dump({"layers": [12] + list(range(14, 39)), "n_arms": 16, "arms": [
  {"arm": 1, "label": "project out d, 22-30", "slots": [{"layers": list(band), "kind": "project", "d": "dirs.npz", "mu": "mu_mine.json"}]},
  {"arm": 2, "label": "add 4 sd of d, 22-30", "slots": [{"layers": list(band), "kind": "add", "d": "dirs.npz", "mu": "mu_mine.json", "alpha_sd": 4}]}]},
  open("recipe.json", "w"))
# python tools/arms.py build --recipe recipe.json --out $DSV41_ARMS_DIR/mine   -> mine.safetensors (+ .json), prints sha
# python tools/client.py load $DSV41_ARMS_DIR/mine.safetensors
# per request: python tools/arms.py fields --arms-sha <sha> --arm 1  -> {"vllm_xargs": {"arm": 1}, "cache_salt": ...}
```

**Rules:** arm 0 stays all zero (the sham). The code forbids edits at outputs 0, 13 and 39; only 12 and 14-38 are
installed by default. Position 0 is never edited. Each arm holds one direction per slot.

## 8. Decode-effect test: what counts as "the edit works"

All rules come from `tools/gates.py` and `docs/GATES.md`, and were fixed before any GPU run.

- **Sham:** arm 0 must stay within the native floor per kind: median <= 1.5x, p95 <= 2x and top-1 >= floor - 1 point.
  The counters must be exact.
- **Positive control:** an 8 sd add at output 26 on every execution path (FULL with the side stream at c = 1/8/16,
  FULL at 17/64/256, PIECEWISE, eager). Pass: exact counters, >= 90% of greedy generations diverge within 8 tokens,
  teacher-forced median >= 10x the generic floor.
- **Decode test (`decode`, arm 4).** `med4` = median |decode logprob - teacher-forced logprob| on arm-4 text. It passes
  iff, on every path, `med4 <= 1.5 max(med0, med_tf4) + 0.01` (noise bound) and `med4 <= 0.1 * effect` (effect cap),
  where `effect` = median |tf_arm - tf_sham| on that text.
- **Decode-effect test (`decode_effect`).** The medium arm is whichever of the 1, 2 and 4 sd adds at output 26 has the
  largest effect on 8 calibration prompts. Each path row needs `effect >= 0.5` (else "inconclusive", which counts
  as a fail), the same noise bound and 0.1-effect cap, exact counters, and the PIECEWISE late prompt joining
  mid-decode. Reading: a cap failure with `effect >= 0.5` means the decode path loses part of the edit (a real bug).
- **B200 result.** Decode was inconclusive. A 4 sd add moved its own decoded text by only 0.15-0.32 (< 0.5). The noise
  bound held, except one random miss. Sign-off came from follow-up batch checks (`docs/RESULTS_2026-10-28.md`).

## 9. Lens and edit engine on one node, one after the other

Yes. On the same 4x B200 job they ran one after the other, with the same node-local weights folder, separate venvs
with the same versions, and the same cache root. The edit engine (graph mode) stopped, and the lens session booted 37 s later. Both need all 4 GPUs, and the lens
backend refuses to boot while any GPU holds > 2,000 MiB.

Boot times from the logs:
- **Edit engine, graph mode, warm caches:** 15.8 min to `Application startup complete`. Weights took 70 s on rank 0
  and ~404 s on ranks 1-3; graph capture took 31 s; autotune took ~7 min.
- **Lens engine, eager:** 468 s. Weights took 15 s, and autotune took ~6 min with the cached configs.

## Established vs. still to check (for anyone running this on new hardware)

**Established on 4×B200 (28 Oct 2026)**
- Edit engine: sham arm is bit-exact against the plain server; projection and positive-control gates pass at layer 26;
  edits are per request, so sham / l26 / multi-layer arms share one batch. Runtime is flat in the number of edited
  layers (the edit step runs at every installed slot inside one CUDA graph).
- Same-hardware noise floor for the edit gates: median 0.013 (sham vs sham).
- Lens fork: stream means at several layers come out of one early-exit pass (22/26/30 together, 34 alone, TP4).
- Lens and edit engine share one node one after the other (same weights directory, same CUDA/torch/vLLM versions).

**Known problems / open**
- Lens-fork capture in very large batched calls (~400k tokens per call) corrupts some vectors: for byte-identical
  prefixes, the worst 24 pairs have median cosine 0.42 batched vs 0.976 captured one prompt per call, and 15 of 34,094
  rows came out all-NaN (all finite when recaptured singly). Until fixed: capture in small batches (≈ the engine's
  max_num_batched_tokens) or one transcript per call, and drop non-finite rows.
- Edit engine: run-to-run noise of a strong arm under a different batch mix is ~0.4 (reload/isolation gates), so
  compare arms only against a sham run in the same wave; the decode-effect gate was inconclusive on B200 (effect
  0.15-0.32 against a 0.5 bar).
- Removing a direction at many layers (13-layer band, 22-layer wide) broke the model's matched controls (0% valid
  commands); single-layer and 4-layer (22/26/30/34) settings are the ones with working controls.

**Checklist on a new GPU type (before any experiment)**
1. Boot the plain server and the edit engine; record which MoE / attention backends vLLM picked (boot log).
2. Reference request (below) on both: token ids identical, greedy text identical for ≥ 30 tokens, top-5 logprobs
   within ~0.05 nats of the B200 reference.
3. Edit-engine gates (`tools/gates.py --only install,build,sham,positive,decode,isolation,prefix,reload,projection`):
   sham bit-exact, positive control passes, noise floor near 0.013.
4. Lens fork: reference readout (below): stream-mean cosine ≥ 0.99 vs B200 at each layer, lens text sensible;
   batched vs single-prompt capture of the same prompts agree (cosine ≥ 0.99) before any large readout run.
