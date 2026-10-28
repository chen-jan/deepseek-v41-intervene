# Multi-layer edits, captures and lens readouts

Editing several layers with one arm, capturing several layers in one request, and reading captured vectors at
several layers with an O-lens. The engine needs nothing new for any of this: every arm already spans all installed
slots, and every capture request already records all of them.

## Multi-layer edits

**An arm covers every slot.** An arms file holds, per arm, one entry for each of the 26 installed slots (outputs 12
and 14-38, fixed at boot by `DSV41_EDIT_LAYERS`), each with its own read direction `p`, write direction `e`, `mu`,
removal strength `a` and add amount `g` (`docs/INTERFACE.md` section 3). An arm that edits outputs 22, 26, 30 and 34
is an arm with entries at those four slots and zeros elsewhere; a slot with `a = g = 0` is a no-op for that arm.

Limits: one direction per arm per slot (rank 1); 15 arms besides the sham per file, and loading a file replaces all
of them; one `[edit_from, edit_to)` window per request, applied at every slot of the arm; no edits at outputs 0-11,
13 or 39.

**Layer sets we used (28 Oct 2026).**

| Name | Outputs | Slots |
|---|---|---|
| `l26` | 26 | 1 |
| `l4` | 22, 26, 30, 34 | 4 |
| `band` | 22-34 | 13 |
| `wide` | 12, 14-34 | 22 |

Outputs 35-38 stayed out of every set. Later that day `band` and `wide` were dropped in favour of `l4` (see the risk
below). Why edit more than one layer at all: a removal at one layer partly comes back downstream. After removing a
direction at output 26, 3.4% of its gap was back at 27, 17% at 30 and 31% at 34 (`docs/RESULTS_2026-10-28.md`, Info).

**Arm tables with explicit layer lists.** Write every slot entry's layers as a list, never as a setting name, so the
table itself says what was edited. A recipe for `tools/arms.py build` (directions as `.npz` files keyed by layer,
`"22"`, `"26"`, ...; `mu.json` from `tools/arms.py mu`; `layers` at the top is the full installed list that
`tools/client.py installed` prints):

```json
{"layers": [12, 14, 15, 16, "...", 38], "n_arms": 16,
 "arms": [
  {"arm": 1, "label": "remove d at 26",
   "slots": [{"layers": [26], "kind": "project", "d": "d.npz", "mu": "mu_d.json"}]},
  {"arm": 2, "label": "remove d at 22/26/30/34",
   "slots": [{"layers": [22, 26, 30, 34], "kind": "project", "d": "d.npz", "mu": "mu_d.json"}]},
  {"arm": 3, "label": "matched shift of d at 22/26/30/34",
   "slots": [{"layers": [22, 26, 30, 34], "kind": "shift", "d": "d.npz", "c": "c.npz", "mu": "mu_d.json"}]},
  {"arm": 4, "label": "random projection at 22/26/30/34",
   "slots": [{"layers": [22, 26, 30, 34], "kind": "project", "d": "r.npz", "mu": "mu_r.json"}]},
  {"arm": 5, "label": "add 2 sd of d at 22/26/30/34",
   "slots": [{"layers": [22, 26, 30, 34], "kind": "add", "d": "d.npz", "mu": "mu_d.json", "alpha_sd": 2}]}]}
```

- `mu` at each slot must be measured along that slot's exact `p`: capture coefficients with a measure-only arm on the
  same layers (`"kind": "measure"`), then `tools/arms.py mu` (mean; median from output 30 on; positions >= 4;
  requests with any non-finite value dropped). Measure on the kind of tokens you will edit.
- `alpha_sd` is per layer: k times that layer's own sd from `mu.json`.
- Controls go on exactly the same layer list as the edit they control. `tools/arms.py controls` draws one layer at
  a time; collect its output into an `.npz` keyed by layer. If a control cannot be drawn at some layer, drop that
  layer from the edit too, or report the mismatch.

**Filling slots without their own direction.** Stream-mean vectors exist only at the capture layers
(`DSV41_CAPTURE_LAYERS`, default 12, 15, 18, 22, 25, 26, 27, 30, 34), so a direction built from captures exists only
there. For `band` and `wide` we filled the slots in between with this rule:

- a slot that is a capture layer uses its own direction;
- any other slot borrows the direction of the nearest capture layer (a tie goes to the lower one), but only if the
  directions at the capture layers just below and just above the slot have cosine >= 0.5;
- for a control direction (matched shift, random projection) the cosine test runs on the direction it controls, so a
  control covers the same slots as its edit;
- nothing is extrapolated: a slot below the lowest or above the highest capture layer gets no direction;
- a slot that fails stays empty (a no-op), and the arm's layer list leaves it out.

```python
import numpy as np

def fill_slots(vecs: dict[int, np.ndarray], slots: list[int], gate: dict[int, np.ndarray] | None = None,
               min_cos: float = 0.5) -> dict[int, int]:
    """{slot: capture layer whose direction it uses}; ``vecs`` = {capture layer: direction}; ``gate`` = the
    directions the cosine test runs on (default ``vecs``; for a control, the directions it controls)."""
    gate = vecs if gate is None else gate
    have, src = sorted(vecs), {}
    for L in slots:
        if L in vecs:
            src[L] = L
            continue
        lo = max((x for x in have if x < L), default=None)
        hi = min((x for x in have if x > L), default=None)
        if lo is None or hi is None:
            continue
        a, b = gate[lo], gate[hi]
        if a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) >= min_cos:
            src[L] = lo if L - lo <= hi - L else hi
    return src
```

**Runtime does not depend on how many layers an arm edits.** The CUDA graphs are captured at boot with the edit step
at every installed slot, and every forward runs that step at all 26 slots for every token, whatever the arms say.
With the torch op (`DSV41_EDIT_IMPL=torch`, the default in `scripts/serve_engine.sh` and what every boot so far used)
a slot does the same arithmetic whether it edits or not. With the Triton kernel an unedited row exits early, so each
edited slot adds one read and write of that token's `[4, 5120]` stack. So an arm that edits 1, 4 or 13 layers runs at
the sham's speed (with Triton, nearly so); the cost, about 47% of native throughput, comes from having the slots at
all. This follows from the design; per-arm throughput was not benchmarked.

**Risk: many-layer edits can break the model.** On 28 Oct, in agent runs scored by whether the model's next shell
command parsed, every control arm at `band` (13 layers) and at `wide`
(22 layers), matched shifts and random projections alike, gave **0% valid commands** (no command within 4,096
generated tokens), and so did the removal of one of the two directions we tested. The other direction's removals at
those settings therefore had no working control, and their specificity was shown only at the single layer 26. What we
do since:

- Check damage first, for every arm including the controls: for example, the share of runs with a valid action must
  not fall more than 10 points below the sham's. An arm that fails this says nothing about its direction.
- Prefer few layers (we moved to `l4`), and compare an edit only with controls on the same layer list.

## Multi-layer capture

One capture request records every layer at once:

- **Coefficients** (`capture` `"coef"` or `"vec"`) at all installed slots: `c` before the edit, `c'` after the bf16
  cast, and the carried pre-mix, per stream, along the request arm's `p` at each slot. Arm 0 reads nothing, so its `c`
  is 0. To read your own direction at, say, 22/26/30/34, load an arms file with a measure-only arm there.
- **Vectors** (`capture="vec"`) at the capture layers only: the stream mean before that slot's own edit (edits at
  earlier slots have already reached it), stored bf16. Vectors at another layer need a reboot with
  `DSV41_CAPTURE_LAYERS` set (each must be an installed slot).

```python
import os
from tools.client import EngineClient

c = EngineClient()  # :8101 on the GPU node, arms file loaded (the capture salt needs its sha)
dump = os.path.join(os.environ["DSV41_CAPTURE_DIR"], "ml_run1.safetensors")  # a new file in the engine's folder
res = c.capture_layers(prompts, [22, 26, 30, 34], dump, arm=1, cap_from=4)  # prompts: token-id lists, BOS first
res["prompts"][0]["vec"][26]   # [n, 5120] stream means at output 26
res["prompts"][0]["coef"][30]  # [n, 3, 4]: c, c', pre-mix per stream at output 30
```

From the shell: `examples/multilayer_capture.py` (writes an `.npz` and a JSON summary). What the helper does:

- checks the layers against `dsv41_installed` (installed slots; capture layers for vectors) and the ring room
  (1,048,576 coefficient rows and 32,768 vector rows by default);
- refuses to start while the rings already hold rows: the dump takes every row and clears the rings, so run it on an
  engine nobody else is capturing on;
- sends one teacher-forced request per prompt, each with its own tag (concurrency 1 by default, which avoided the
  batched-NaN problem in our runs; `cap_pos` takes at most 4,096 positions per prompt), then one dump;
- per prompt keeps the last request with its tag (a NaN retry) and the last row per position, flags positions with
  non-finite values (`nonfinite_pos`) and layers where `c` is 0 on every row (`zero_coef_layers`: the arm has no
  direction there). `tools/arms.py layer_capture` does the same split on any dump offline.

Checked by hand against the CPU mock engine (`tests/mock_engine.py`), not on GPU.

## Multi-layer lens readouts

`tools/lens_read.py` captures stream means at several layers and reads them with an O-lens: the release lens at one
layer and/or a multilayer lens at several. It runs on the lens engine, not on this repo's edit engine: the
vllm-metamodels DeepSeek V4.1 build (see "Related work" in the README; our runs used build `7ab496a`) with its usage
modules `olens_deepseek_v41_flash_usage` and `dsv41_engine`, in that build's own venv. None of it is included here;
the tool imports it from `--lens-engine-dir`. It takes the 4 GPUs, so stop the edit engine first.

**Not run end to end from this repo.** The 28 Oct readouts came from a private harness with the same maths; this
version was only run on CPU with a stub engine (`--check` validates inputs and lens folders without a GPU).

What it does:

1. **Capture**, base weights: one early-exit forward per transcript gives the stream mean (the mean of the 4 mHC
   streams) at every requested layer and position. Saved bf16 as `<out>/h.pt` `[N, layers, 5120]` with the layer
   list and the shas of the positions and inputs; a rerun reuses it only when all three match.
2. **Release lens** (default layer 26): the captured vector goes in at the marker of the lens's prompt through the
   usage module's `steer_embed(v, marker, ALPHA)`.
3. **Multilayer lens** (its input heads, as it was trained): `v <- v / |v|`, `u = W_L v + b_L`, and `steer_embed`
   puts `alpha * u / |u|` at the marker of the prompt tagged with layer L. With both lenses, 512 random positions are
   also read with the multilayer head at 26, to compare the two lenses on the same vectors.
4. **Sampling**: per position, greedy plus 4 samples at T = 1 (top-p 1), each one bullet, constrained by the usage
   module's bullet grammar; the run stops if any request's injection did not land.
5. **NaN guard**: a non-finite vector is written as `{"i": i, "nonfinite": true}` and never read (15 rows of our
   first 28 Oct capture were all-NaN). Readouts append to `readouts_<stage>.jsonl` (`{"i", "greedy", "samples"}`);
   a rerun skips rows already there and refuses if that stage's lens or settings changed.

```bash
python tools/lens_read.py --check --positions pos.json --token-ids prompts.json --layers 22,26,30,34 \
    --release-lens <release lens dir> --multilayer-lens <multilayer lens dir> --out <out dir>
python tools/lens_read.py --positions pos.json --token-ids prompts.json --layers 22,26,30,34 \
    --release-lens <dir> --multilayer-lens <dir> --lens-engine-dir <dir> --model-dir "$DSV41_MODEL_DIR" --out <dir>
```

Inputs: `--positions` (`{"reads": [{"key", "pos", ...}]}`; each read keeps its row index `i`) and either
`--token-ids` (`{"<key>": [ids]}` or JSONL `{"key", "ids"}`) or `--items` (key list + `.npz` of concatenated ids and
offsets). Every prompt starts with BOS (id 0). The paths can also come from `DSV41_MODEL_DIR`,
`DSV41_LENS_ENGINE_DIR`, `DSV41_RELEASE_LENS` and `DSV41_MULTILAYER_LENS`. Lens weights are not part of this repo.

**Which positions to read.** The tool reads whatever positions it is given. Ours came from a fixed rule on pairs of
agent transcripts that differ only in a short piece of text (one version with it, twins without), per transcript:

- *context*: punctuation or newline tokens (every non-space character is Unicode punctuation, or the token holds a
  newline) in the last tool output before the cut point, i.e. from the last user-turn token to the end of the prompt.
  At most 80: first all within 60 tokens of the first token where the versions differ, then evenly spaced ones from
  the rest;
- *generation prompt*: the two tokens at the end of the prompt (`<｜Assistant｜><think>`);
- *null*: 40 punctuation or newline tokens evenly spaced before that tool output, where the versions are
  token-identical, so readout differences there measure engine and lens noise;
- *thinking*: every one of the first 32 thinking tokens of each continuation.

Positions are chosen on the version with the text and mapped to its twins by token alignment (difflib matching
blocks); a position inside the differing text has no twin and is read on that version only.
