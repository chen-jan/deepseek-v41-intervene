# deepseek-v41-intervene

Serve DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash-1017`) with vLLM 0.31 on 4 GPUs, and edit its residual
stream per request inside the normal batched server (CUDA graphs on).

- **Serving:** environment build, download, plain server, benchmarks. Measured on 4x B200.
- **Edit engine:** each request names an *arm*. An arm can remove a direction, shift it, add it, or only measure it,
  at any of 26 layer outputs. Arm 0 is a sham that runs the same code and changes nothing, so edited and unedited
  requests share one batch. It also captures per-token coefficients and vectors, counts every edit, and comes with a
  gate suite.
- **Provider:** renders agent conversations with DeepSeek's official chat encoding and sends them as raw completions.

## Status (28 Oct 2026)

| | |
|---|---|
| Plain serving | works; 5,346 output tok/s at concurrency 128 |
| Engine speed | 47% of native at concurrency 128 (graph mode); eager mode 10% |
| Passed | install, sham = native, positive control on all 8 execution paths, prefix-cache isolation, projection outliers (only bf16 rounding on one huge-norm token), proxy |
| Failed as pre-written, then explained by a follow-up check | arm isolation and reload: an edit's own result shifts with batch *size*; mixing with unedited neighbours adds nothing beyond repeat noise, and the edit keeps 98.6-99.8% of its effect, on short prompts and on 17k-token agent transcripts |
| Still open | decode-effect gate inconclusive (test edits barely change decoded text); validate with your own direction before relying on results |

Details and numbers: `docs/RESULTS_2026-10-28.md`.

## Quick start

```bash
cp .env.example .env              # set DSV41_ROOT and DSV41_MODEL_DIR (paths only)
bash scripts/build_env.sh         # venv (needs uv)
bash scripts/download_model.sh    # 476 GiB checkpoint
bash scripts/make_key.sh          # API key file
bash scripts/serve_plain.sh       # plain server on :8100
bash scripts/serve_engine.sh      # or: edit engine on 127.0.0.1:8101 (+ python tools/proxy.py for :8100)
```

Arms, per-request fields, captures, gates and known issues: `docs/USAGE.md`. Engine internals: `docs/INTERFACE.md`.
Gate rules: `docs/GATES.md`. Provider: `docs/PROVIDER.md`. CPU tests: `python -m pytest -q` and
`bash tests/run_mock_gates.sh`.
Multi-layer edits, captures and O-lens readouts (lens tool not yet run end to end from this repo): `docs/MULTILAYER.md`.
- Porting to new GPUs (B300 / sm_103): what is compiled where, versions, reference checks, known problems: [docs/B300_PORTING.md](docs/B300_PORTING.md)

## Related work

The per-request design (a vLLM plugin where each request says what to edit or read) follows UK AISI's
[vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens). We used Celeste's
[vllm-metamodels](https://github.com/ceselder/vllm-metamodels), a faster fork of vllm-lens, for our lens runs; this
repo adds what that engine does not support on DeepSeek V4.1: edits at middle layers, edits during generation with
CUDA graphs on, and removing or shifting a direction. No code is copied from either project.

## Licence

MIT (`LICENSE`) for all code here. Nothing is copied from vLLM or DeepSeek: the engine wraps vLLM's modules at
runtime, and the provider loads DeepSeek's `encoding.py` from the checkpoint. File-by-file provenance:
`docs/USAGE.md`.
