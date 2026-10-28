# Gates: what they check and how to read them

`tools/gates.py` runs these checks against a booted engine (`scripts/serve_engine.sh`, `127.0.0.1:8101`). Each gate
writes one JSON file into the gates folder (default `$DSV41_LOG_DIR/engine/gates_<MMDD_HHMM>/`) and prints one
`[gate] <name>: PASS` or `FAIL` line. `summary.json` collects the verdicts. A failure stops the suite unless you pass
`--keep-going`. A crash counts as a failure, with the error in the gate's JSON.

The rules below were fixed before any GPU run. Do not loosen a rule after it fails. The only allowed change is to
re-measure a floor that was measured wrongly, once, and to keep the old result next to the new one. The retry gates
at the end of this page are that re-measurement for three rules.

## Before the gates

1. **Test prompts.** Build `data/` once with `tools/build_data.py` from your own text (code, prose, rendered agent
   transcripts). See the docstring of that file. The repo ships no prompts.
2. **Native floor.** With the plain server up (`scripts/serve_plain.sh`, `:8100`), measure the run-to-run noise of
   the unedited model on the floor prompts:

   ```bash
   python tools/client.py --url http://127.0.0.1:8100 floor --out $DSV41_LOG_DIR/engine/floor_native_$(date +%m%d_%H%M).json
   ```

   This scores the 40 floor prompts three times: (a) all at once, (b) again in shuffled order, (c) one by one. The
   floor per kind (generic, agent) is the worse of (a vs b) and (a vs c). The gates pick up the newest
   `floor_native_*.json` automatically (`--floor` to choose one). Never name an engine floor `floor_native_*`.
3. **Boot the engine** (stop the plain server first: both need all 4 GPUs). Wait for `/health` to return 200.
4. **Run the gates.** Nothing else may talk to `:8101` while they run: the counters must be exact.

   ```bash
   python tools/gates.py --engine-log $(ls -t $DSV41_LOG_DIR/engine/serve_engine_*.log | head -1) --keep-going
   ```

   After a fallback boot, pass the same settings: `--edit-layers ...` and `--mhc-overlap 0` if you changed them.
   Rerun single gates with `--only a,b --out <same folder>`. The gate arms are reloaded automatically after a reboot.
5. **Proxy last.** Start `tools/proxy.py` only after the counted gates, then `python tools/gates.py --only proxy --out
   <same folder>`.

`python tools/retry_summary.py <gates folder>` prints the headline numbers of every gate side by side.

## Units and words

- **Arm.** A row of the arms table: what to edit, at which layer outputs, how strongly. Arm 0 is the sham: all zero,
  so it runs the full wrapper but changes nothing. The gate arms file has 9 arms (below).
- **Distance between two runs.** Teacher-forced scoring (`prompt_logprobs`) of the same token ids twice; per position
  `|Δ logprob|` of the actual next token. Reported as median, p95 and top-1 agreement over positions.
- **Floor rule.** A comparison is "within the floor" when its median is <= 1.5x the floor median, its p95 <= 2x the
  floor p95, and its top-1 agreement >= the floor's minus 1 point, for every kind.
- **sd.** The standard deviation of the coefficient `h . d26` over generic prompt tokens, per layer output.
- **Paths.** vLLM runs a step through one of several execution paths: FULL CUDA graphs with the mHC side stream
  (steps of <= 16 tokens), FULL graphs, PIECEWISE graphs (mixed prefill + decode steps), and eager (steps > 1,024
  tokens). The positive control and decode gates cover each of them.

## Gate arms (built by the `build` gate)

`d26` is the unit difference of mean output-26 states between set A (code) and set B (prose), from a vector capture.
`d'` is the same for set C (agent text) vs B. `c` is a seeded random unit vector orthogonal to `d26`. The `mu` and
`sd` per slot come from a coefficient capture under measure-only arms on the 20 generic prompts.

| Arm | What |
|---|---|
| 1 | measure only along `d26` at every slot |
| 2 | project out `d26` at output 26 |
| 3 | project out `d26` at every slot |
| 4 | add 8 sd of `d26` at output 26 (the positive control) |
| 5 | matched shift at 26: remove the `d26` deviation and write it along `c` |
| 6 | mean-coupled projection at 26 (one delta for all 4 streams) |
| 7 | project out `d26` at outputs 22-34 |
| 8 | add 8 sd of `d'` at 26 |
| 9 | project at 26, measure at 27, 30, 34 (how much of the removed gap comes back downstream) |

## The gates

| Gate | Pass rule |
|---|---|
| `install` | All 4 TP ranks report the patch installed, the expected slots wrapped (default 26 slots: outputs 12 and 14-38), fusion off (layer flag, model flag, no deferred MoE finalize, fused kernel never built), mHC side stream present (absent with `DSV41_MHC_OVERLAP=0`), no sequence parallel, no speculative decoding. The serve log has the `dsv41_intervene: fusion off` line and neither "Lamport all-reduce fused" nor "finalize fused". |
| `build` | Builds and loads the gate arms (above). Passes when it finishes. |
| `sham` | The 40 floor prompts under arm 0 vs the native run (a): within the native floor per kind (generic, agent). Also `tokens_seen[0]` equals the prompt tokens (NaN retries counted again), nothing edited, no invalid arms. Code/prose/agent sub-kinds are recorded, not gated. |
| `positive` | Arm 4 vs arm 0 on every path: FULL with side stream (c = 1, 8, 16), FULL (17, 64, 256), PIECEWISE (16 requests decoding 512 tokens plus one 600-token prompt sent once decoding is under way), eager (one 4,000-token prompt). Exact counters (`tokens_edited[4]` at output 26 = prompt + generated tokens - 2), >= 90% of greedy generations diverge within 8 tokens, teacher-forced median >= 10x the generic floor. |
| `decode` | Decode-time logprobs vs teacher-forced logprobs of the same arm-4 text. With `med4 = median|decode4 - tf4|`, `med_tf4` = two teacher-forced runs of that text, `med0` the same for arm 0, and `effect = median|tf4 - tf0|` on arm-4 text: pass iff `med4 <= 1.5 max(med0, med_tf4) + 0.01` (noise bound) **and** `med4 <= 0.1 effect` (effect cap), on every path. |
| `isolation` | A mixed batch (16 sham + 16 arm-4 requests) vs each arm alone. Every request within its own prompt's native floor (the worse of a-vs-b and a-vs-c on that prompt, with minimums 0.005 median, 0.05 p95, 0.99 top-1). Exact per-arm counters, including `n = 8` children. |
| `prefix` | A prefix-cache hit under an arm vs a cold run: the cold run computes every token, the hit recomputes < 25% of the prompt, 32 greedy decode logprobs of hit and cold match within 1.5x the agent floor, and other arms and the sham never reuse the arm's blocks. (vLLM 0.31 never reads the prefix cache for a `prompt_logprobs` request, so the hit is checked on decode logprobs.) |
| `reload` | Load arms, reload with arm 4's content in slot 8, reload the original: slot 8 changes as expected, matches arm 4 within the floor, and goes back. Shas in order. |
| `projection` | Arms 3, 5, 6 with coefficient capture, at concurrency 1 (batched requests can hit the base model's NaN issue). At every edited slot: `cast_violations == 0`, no non-finite values, `max_err` from the counters equals the capture's, capture rows aligned with the kernel's positions. At output 26 only: error `<= 0.05 sd` on every token and `<= 0.02 sd` on >= 99.9% of tokens. Other slots are recorded. |
| `info` | No pass/fail. Persistence of arm 9's removal at 27/30/34, displacement norms, variance ratios, and vector captures at outputs 25-27 saved to `$DSV41_XCHECK_DIR` for a cross-engine comparison. |
| `proxy` | Control routes 404 through the proxy, the key is required, salts must match a loaded arms sha of this boot, captures are refused, and a global-arm block passes only its own arm (503 otherwise). |

**NaN retries.** Every gate JSON records how many teacher-forced requests the client had to retry because the
server returned NaN logprobs. A retry at concurrency 1 fails the gate: one-by-one requests never hit the NaN issue on
the plain server.

**Throughput.** Not a gate in the script. Run `tools/bench.sh native` on the plain server and `tools/bench.sh sham` on
a fresh engine. The bar we used: engine sham at c = 128 >= 40% of native output tok/s.

**bf16 rounding bound.** The edited stack is written back as bf16, so a projection can land short of its target by
the rounding of each element. Round-to-nearest-even bounds that per element by `2^-8 / (1 - 2^-8) |h_i|`
(0.0039216 `|h_i|`). The kernel counts a token in `cast_violations` when any stream's coefficient error exceeds
`0.004 sum_i |p_i| max(|h_before_i|, |h_after_i|)`. Correct code gives 0 at every slot and scale. This holds for any
direction, unlike an sd threshold, which can fail on correct code where a direction varies little but has large
elements.

## Retry gates

Three of the first GPU boots' failures (see `docs/RESULTS_2026-10-28.md`) may come from floors that do not fit the
arm, not from bugs. The retry gates re-measure those floors. They run after `info` and never replace an old gate: the
old gates run unchanged in the same suite, and each new result carries the old one next to it (`old`).

| Old failure | Hypothesis | New gate |
|---|---|---|
| Isolation: arm 4 mixed vs alone, median 0.436 vs a sham floor of 0.013 | An 8 sd add makes outputs chaotic, so tiny batch-composition rounding differences are amplified. The right floor for an edited arm is the same arm repeated. | `repeat`, part A |
| Reload: slot 8 with arm 4's content vs arm 4, median 0.389 | Same: two runs of arm 4 differ by about 0.4 anyway | `repeat`, part B |
| Projection at 26: max error 0.71 / 0.68 / 0.45 sd vs 0.05 sd (99.9% part passed) | The maxima sit on a few huge-norm tokens where bf16 cannot hold the coefficient to 0.05 sd | `outliers` |
| Decode effect cap failed in 5 of 8 path rows (noise bound passed in all) | On arm 4's own greedy text the effect is only 0.006-0.19, so the cap is 10% of a small number | `decode_effect` |

Extra arms live in a second arms file: arms 0-9 as in the gate file, arms 10-13 add 0.5, 1, 2 and 4 sd along `d26` at
output 26. The reload twin puts arms 10 and 11 into slots 14 and 15.

- **`repeat`.** Part A, on the 16 isolation prompts: the mixed batch twice (M1, M2), the arm alone once, sham alone
  once, for arms 4, 8 (strong) and 10, 11 (weak). Part B, on the 20 generic prompts: arm 4 in its own slot twice, the
  reloaded slot 8 twice, and the weak arms vs their reloaded slots. Distance = per-request median `|Δ logprob|`.
  Pass: for each strong arm, `median(cross) <= quantile(own repeats, 0.95) + F` (F = pooled generic native floor
  median); the gated weak arm (10, or 11 if 10 moves logprobs by < 10 F) passes the original isolation and reload
  rules; every sham in the M1 batches of arm 4 and the weak arm is inside its own prompt's floor; counters exact.
- **`outliers`.** Arms 3, 5, 6 on the generic prompts at concurrency 1, with vector capture so each token's norm is
  known. Pass at output 26: every error above 0.05 sd sits on a token whose bf16 bound `2^-8 |h| / sd` exceeds 0.05
  sd, >= 99.9% of tokens within 0.02 sd, and the exact part of `projection` passes.
- **`decode_effect`.** Picks a medium arm (1, 2 or 4 sd add, the one with the largest calibration effect, chosen
  before the measurement) and reruns the decode rule on every path. Pass: effect on its own text >= 0.5 on every
  path row, noise bound and 0.1-effect cap pass, counters exact, PIECEWISE late prompt joins mid-decode.

### How to read the retry gates (written before any retry data)

"Real bug": do not trust the engine; debug. "Gate artefact": the old failure is explained by the measured floor; the
old FAIL stays on record next to it. Each gate JSON carries `reading` lines that apply this table.

| Result | Reading |
|---|---|
| Strong arms within their own repeat noise; the weak arm passes the original isolation rule and is sensitive; gated shams inside their floors; counters exact | Gate artefact. 8 sd adds are not a fine-grained readout; read them only for large effects. |
| As above, but cross/repeat median ratio > 2 | Passes by the rule, but only a partial explanation. Report it. |
| A strong arm's cross median > `q95(repeats) + F` | Real bug candidate: the edit depends on what else is in the batch. |
| The gated weak arm fails the original isolation rule | Real bug (a leak at projection size). |
| Neither weak arm moves logprobs by >= 10 F | Inconclusive: a leak at small sizes is untested. |
| A gated sham misses its floor and its drift is >= 0.1 of the arm's own effect | Real leak. |
| A gated sham misses its floor but its drift is < 0.1 of the arm's effect | Noise-level miss. The gate fails as written; it is not evidence of a leak. |
| Counters not exact, or load shas out of order | Real bug. |
| Reload within arm 4's repeat noise and the weak reload passes the original rule | Gate artefact. |
| Reload beyond arm 4's repeat noise, or the weak reload fails | Real bug in `load_arms` or the slot tables. |
| Every projection error above 0.05 sd is on a token whose bf16 bound exceeds 0.05 sd | Gate artefact (below bf16 resolution on those tokens). |
| Any error above 0.05 sd on a token whose bound is <= 0.05 sd | Real bug candidate. Check that token. (The norm is that of the stream mean; one stream could still be larger.) |
| Fraction within 0.02 sd < 99.9% at output 26 | Real failure. |
| Cast violations, non-finite values, stats vs capture mismatch, misaligned rows, ring overflow | Real bug. |
| Decode: effect >= 0.5 and the cap fails on some path | Real bug: that path loses part of the edit (for example a stale buffer in a graph replay). |
| Decode: noise bound fails on some path | Real bug. |
| Decode: effect < 0.5 on some path row | Inconclusive for that path; the gate fails. |
| `install`, `sham` or `positive` fails | Problem with this boot. Stop; do not read the retry gates. |
