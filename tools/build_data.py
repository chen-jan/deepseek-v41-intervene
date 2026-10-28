"""Build the token-id test data the gates need: ``plumbing_texts.json`` and ``floor_prompts.json`` (docs/INTERFACE.md
section 9). The repo ships no prompts; build them from your own text, or synthetic ids for the CPU mock run::

    python tools/build_data.py --out data --tokenizer $DSV41_MODEL_DIR/tokenizer.json \\
        --code path/to/some/python/repo --prose notes1.md notes2.md --agent run1.txt run2.txt run3.txt
    python tools/build_data.py --out data --synthetic        # random ids, same shapes (mock engine / CPU tests only)

- Set A (code) and set B (prose): 30 windows of 400 tokens each. The gates build a test direction from the mean
  difference of A and B at output 26, and a second one from C vs B.
- Set C and the 20 agent floor prompts: long agent transcripts, already rendered to text with the model's chat format
  (for example with ``provider/deepseek_v41_vllm_provider.py``'s ``render()``). Each agent prompt is a prefix of one
  transcript, 6,000-30,000 tokens long; the prefix-cache gate needs one of at least 6,300 tokens, the eager-path case
  one of at least 4,000.
- 20 generic floor prompts of 520-3,000 tokens (6 of at most 1,000 tokens, for PIECEWISE-sized steps), alternating
  code and prose.

Every prompt is a list of token ids starting with BOS (id 0). Engine checks only: these prompts never back a claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

BOS = 0
CHUNK = 400  # tokens per plumbing chunk, BOS included
N_SET = 30
GENERIC_LENS = [520, 640, 760, 880, 960, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050, 2200, 2350, 2500, 2650,
                2800, 2900, 3000]
N_AGENT = 20
AGENT_RANGE = (6000, 30000)
VOCAB = 129280  # DeepSeek V4.1 vocabulary size; synthetic ids stay well inside it


def windows(ids: list[int], n: int, size: int, skip: int = 0) -> list[list[int]]:
    """``n`` evenly spaced non-overlapping windows of ``size`` ids (after ``skip``), each prefixed with BOS."""
    body = size - 1
    span = len(ids) - skip
    if span < n * body:
        raise SystemExit(f"need {n * body} tokens for {n} windows, have {span}: give more text")
    step = span // n
    return [[BOS] + ids[skip + i * step: skip + i * step + body] for i in range(n)]


def spread(pools: list[list[int]], n: int, size: int) -> list[list[int]]:
    """``n`` windows of ``size`` spread over several sources (more from the longer ones)."""
    pools = [p for p in pools if len(p) >= size]
    if not pools:
        raise SystemExit(f"no source has {size} tokens")
    total = sum(len(p) for p in pools)
    per = [max(1, round(n * len(p) / total)) for p in pools]
    while sum(per) > n:
        per[per.index(max(per))] -= 1
    while sum(per) < n:
        per[per.index(min(per))] += 1
    return [w for p, k in zip(pools, per) if k for w in windows(p, k, size)]


def agent_prompts(runs: dict[str, list[int]]) -> list[dict]:
    """20 prefixes of the transcripts, lengths evenly spaced over what the transcripts allow in ``AGENT_RANGE``."""
    lo, hi = AGENT_RANGE
    usable = {k: v for k, v in runs.items() if len(v) + 1 >= lo}
    if not usable:
        raise SystemExit(f"agent prompts need transcripts of at least {lo} tokens")
    top = min(hi, max(len(v) + 1 for v in usable.values()))
    if top < 6300:
        raise SystemExit("the prefix-cache gate needs one agent transcript of at least 6,300 tokens")
    lens = [round(lo + i * (top - lo) / (N_AGENT - 1)) for i in range(N_AGENT)]
    names = sorted(usable)
    out = []
    for i, n in enumerate(lens):
        fits = [k for k in names if len(usable[k]) + 1 >= n]
        name = fits[i % len(fits)]
        out.append({"id": f"agent_{name}_{i:02d}", "kind": "agent", "ids": [BOS] + usable[name][: n - 1]})
    return out


def generic_prompts(code: list[list[int]], prose: list[list[int]]) -> list[dict]:
    out = []
    for i, n in enumerate(GENERIC_LENS):
        pool, label = (code, "code") if i % 2 == 0 else (prose, "prose")
        fits = [p for p in pool if len(p) >= n - 1]
        if not fits:
            raise SystemExit(f"generic {label} prompt {i} needs a source of {n - 1} tokens")
        src = fits[(i // 2) % len(fits)]
        off = (i * 997) % max(1, len(src) - n + 1)
        out.append({"id": f"generic_{label}_{i:02d}", "kind": "generic", "ids": [BOS] + src[off: off + n - 1]})
    return out


def synthetic(seed: int) -> tuple[dict, dict]:
    """Random ids with the real data's shapes; sets A, B and C use disjoint id ranges so their means differ."""
    import random

    rng = random.Random(seed)
    draw = lambda lo, hi, n: [rng.randrange(lo, hi) for _ in range(n)]  # noqa: E731
    code = [draw(100, 40000, 9000) for _ in range(4)]
    prose = [draw(40000, 80000, 9000) for _ in range(4)]
    agent = {f"syn{k}": draw(80000, VOCAB, 32000) for k in range(4)}
    texts = {"tokenizer_sha256": "synthetic", "chunk": CHUNK, "sources": {"A": "synthetic", "B": "synthetic",
                                                                          "C": "synthetic"},
             "A": spread(code, N_SET, CHUNK), "B": spread(prose, N_SET, CHUNK),
             "C": spread(list(agent.values()), N_SET, CHUNK)}
    floor = {"tokenizer_sha256": "synthetic", "prompts": generic_prompts(code, prose) + agent_prompts(agent)}
    return texts, floor


def files(paths: list[str], suffixes: tuple[str, ...] | None = None) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file() and (suffixes is None or q.suffix in suffixes))
        else:
            out.append(p)
    return out


def from_text(args) -> tuple[dict, dict]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    tok_sha = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    ids = lambda path: tok.encode(path.read_text(errors="replace"), add_special_tokens=False).ids  # noqa: E731
    code_files, prose_files, agent_files = files(args.code, (".py",)), files(args.prose), files(args.agent)
    if not (code_files and prose_files and agent_files):
        raise SystemExit("give --code, --prose and --agent sources")
    code, prose = [ids(f) for f in code_files], [ids(f) for f in prose_files]
    runs = {f.stem: ids(f) for f in agent_files}
    texts = {"tokenizer_sha256": tok_sha, "chunk": CHUNK,
             "sources": {"A": [f.name for f in code_files], "B": [f.name for f in prose_files], "C": sorted(runs)},
             "A": spread(code, N_SET, CHUNK), "B": spread(prose, N_SET, CHUNK),
             "C": spread(list(runs.values()), N_SET, CHUNK)}
    floor = {"tokenizer_sha256": tok_sha, "prompts": generic_prompts(code, prose) + agent_prompts(runs)}
    return texts, floor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--synthetic", action="store_true", help="random ids (mock engine and CPU tests only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", help="the checkpoint's tokenizer.json")
    ap.add_argument("--code", nargs="*", default=[], help="Python files or folders (set A, generic code prompts)")
    ap.add_argument("--prose", nargs="*", default=[], help="English text files (set B, generic prose prompts)")
    ap.add_argument("--agent", nargs="*", default=[], help="rendered agent transcripts (set C, agent prompts)")
    args = ap.parse_args()
    if args.synthetic:
        texts, floor = synthetic(args.seed)
    else:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required unless --synthetic")
        texts, floor = from_text(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plumbing_texts.json").write_text(json.dumps(texts))
    (out / "floor_prompts.json").write_text(json.dumps(floor))
    lens = lambda xs: [len(x) for x in xs]  # noqa: E731
    print(json.dumps({"out": str(out), "A": len(texts["A"]), "B": len(texts["B"]), "C": len(texts["C"]),
                      "generic": lens([p["ids"] for p in floor["prompts"] if p["kind"] == "generic"]),
                      "agent": lens([p["ids"] for p in floor["prompts"] if p["kind"] == "agent"])}))


if __name__ == "__main__":
    main()
