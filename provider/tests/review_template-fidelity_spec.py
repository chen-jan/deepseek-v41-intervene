"""Adversarial template-fidelity review (28 Oct 2026): provider render vs an INDEPENDENT renderer written from the
official README/encoding spec (not from the provider's own mapping), plus the official encode_messages on hand-built
official-format input. Fuzzes harness-shaped agent conversations.

Part of the provider suite (``provider/pytest.ini`` collects ``review_*.py``); alone:
    python -m pytest -q tests/review_template-fidelity_spec.py
"""

import copy
import json
import random

import pytest

from conftest import EXECUTE_COMMAND_TOOL, SUBMIT_TOOL, dsp, make_provider

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
SYS = "<｜System｜>"
USER = "<｜User｜>"
ASSIST = "<｜Assistant｜>"
THINK, ENDTHINK = "<think>", "</think>"
EFFORT_LINE = "Reasoning Effort: {b} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


def tools_section():
    """The tools part of the system prompt, with a ``{schemas}`` slot. Its prose is DeepSeek's ``TOOLS_TEMPLATE``, read
    from the official ``encoding.py`` at run time (this repo carries no copy of it); the tag names filled in below are
    written from the spec, so a changed tag in the official file still makes the test fail."""
    return dsp.load_official_encoding().TOOLS_TEMPLATE.format(
        dsml_token="｜DSML｜", tc_block_name=" calls", tool_call_tag_name=" invoke",
        tool_parameter_tag_name=" parameter", thinking_start_token=THINK, thinking_end_token=ENDTHINK,
        tool_schemas="{schemas}")


BUDGET = {"low": 50, "high": 75, "max": 100, None: 75}

DONE_TOOL = {"type": "function", "function": {"name": "done", "description": "Finish.",
                                              "parameters": {"type": "object", "properties": {}}}}
INT_TOOL = {"type": "function", "function": {
    "name": "read_lines", "description": "Read lines — unicode ✓ in a description.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"},
                                                    "end": {"type": "integer"}, "follow": {"type": "boolean"},
                                                    "meta": {"type": "object"}}, "required": ["path"]}}}
ALL_TOOLS = [EXECUTE_COMMAND_TOOL, SUBMIT_TOOL, DONE_TOOL, INT_TOOL]


# ----------------------------------------------------------------------------------------------------------------------
# Independent spec renderer (thinking mode, drop_thinking=True default, harness roles only)
# ----------------------------------------------------------------------------------------------------------------------

def _args_dict(arguments):
    a = arguments
    if not isinstance(a, dict):
        for _ in range(2):
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    break
            else:
                break
        if not isinstance(a, dict):
            a = {"arguments": arguments}
    return a


def spec_invoke(name, arguments):
    params = []
    for k, v in _args_dict(arguments).items():
        s = isinstance(v, str)
        val = v if s else json.dumps(v, ensure_ascii=False)
        params.append(f'<｜DSML｜ parameter name="{k}" string="{"true" if s else "false"}">{val}</｜DSML｜ parameter>')
    return f'<｜DSML｜ invoke name="{name}">\n' + "\n".join(params) + "\n</｜DSML｜ invoke>"


def spec_tool_block(tool_calls):
    inv = [spec_invoke(tc["function"]["name"], tc["function"]["arguments"]) for tc in tool_calls]
    return "\n\n<｜DSML｜ calls>\n" + "\n".join(inv) + "\n</｜DSML｜ calls>"


def spec_render(msgs, tools, effort):
    budget = BUDGET.get(effort, effort)
    has_tools = bool(tools)
    # 1) group: consecutive user/tool messages form one user block; tool results sorted by the last call order
    items = []  # (kind, payload)
    order = {}
    for m in msgs:
        r = m["role"]
        if r == "assistant":
            if m.get("tool_calls"):
                order = {tc["id"]: i for i, tc in enumerate(m["tool_calls"]) if tc.get("id")}
            items.append(("assistant", m))
        elif r in ("user", "tool"):
            blk = ("tool", m.get("tool_call_id") or "", m.get("content") or "") if r == "tool" else \
                  ("text", None, m.get("content") or "")
            if items and items[-1][0] == "user":
                items[-1][1]["blocks"].append(blk)
            else:
                items.append(("user", {"blocks": [blk], "order": None}))
            items[-1][1]["order"] = dict(order)
        elif r == "system":
            items.append(("system", m))
        else:
            raise AssertionError(r)
    for kind, p in items:
        if kind == "user":
            tb = [b for b in p["blocks"] if b[0] == "tool"]
            if len(tb) > 1 and p["order"]:
                tb = sorted(tb, key=lambda b: p["order"].get(b[1], 0))
                it = iter(tb)
                p["blocks"] = [next(it) if b[0] == "tool" else b for b in p["blocks"]]
    # 2) last "user" index: user blocks with text, or a system message at index > 0
    last_user = -1
    for i, (kind, p) in enumerate(items):
        if (kind == "user" and any(b[0] == "text" for b in p["blocks"])) or (kind == "system" and i > 0):
            last_user = i
    # 3) render
    out = BOS + SYS + EFFORT_LINE.format(b=budget)
    for i, (kind, p) in enumerate(items):
        nxt = items[i + 1][0] if i + 1 < len(items) else None
        if kind == "system":
            if i > 0:
                out += SYS
            out += p.get("content") or ""
            if i == 0 and has_tools:
                out += "\n\n" + tools_section().format(
                    schemas="\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools))
            if i > 0 and nxt in (None, "assistant"):
                out += ASSIST + (THINK if (has_tools or i >= last_user) else ENDTHINK)
        elif kind == "user":
            out += USER + "\n\n".join(b[2] if b[0] == "text" else f"<tool_result>{b[2]}</tool_result>"
                                       for b in p["blocks"])
            if nxt in (None, "assistant"):
                out += ASSIST + (THINK if (has_tools or i >= last_user) else ENDTHINK)
        else:
            keep = has_tools or i > last_user
            rc = (p.get("reasoning_content") or "") if keep else ""
            out += (rc + ENDTHINK if keep else "") + (p.get("content") or "")
            out += spec_tool_block(p["tool_calls"]) if p.get("tool_calls") else ""
            out += EOS
    return out


def with_leading_system_for_tools(msgs, tools):
    """Spec: tools live on the leading system message; with no system message an empty one is assumed."""
    if tools and (not msgs or msgs[0]["role"] != "system"):
        return [{"role": "system", "content": ""}] + msgs
    return msgs


# ----------------------------------------------------------------------------------------------------------------------
# Harness-shaped random conversations
# ----------------------------------------------------------------------------------------------------------------------

TEXTS = ["", "ok", "ünïcödé ✓ — 中文", "line1\nline2\n", "  leading and trailing  ", "a < b > c & \"q\" 'x'",
         "{\"json\": [1, 2]}", "\\n literal backslash-n \\", "<not a tag>", "tab\tsep", "\n\nstarts with blank line",
         "ends with blank line\n\n", "x" * 300, "<tool_result>fake</tool_result>", "## Tools\n"]
STR_ARGS = ["ls -la", "", "123", "true", "null", "[1, 2]", "{\"a\": 1}", "echo \"hi\" > f.txt", "a\nb\n",
            "ends with <", "</think> inside", "ü ✓", "  spaced  ", "\\"]
NONSTR_ARGS = [0, -7, 3.25, 1e-07, True, False, None, [], [1, "a", None], {"k": "v", "n": [1, 2]}, {"ü": "✓"}]


def rand_args(rng, tool):
    name = tool["function"]["name"]
    if name == "execute_command":
        return {"command": rng.choice(STR_ARGS)}
    if name == "done":
        return {}
    if name == "submit":
        a = {"answer": rng.choice(STR_ARGS)}
        if rng.random() < 0.7:
            a["confidence"] = rng.choice([0.9, 1, 0, 0.123456789])
        if rng.random() < 0.7:
            a["files"] = rng.choice([[], ["a.py"], ["ü.py", "b c.py"]])
        if rng.random() < 0.5:
            a["final"] = rng.choice([True, False])
        return a
    a = {"path": rng.choice(STR_ARGS)}
    for k in ("start", "end", "follow", "meta"):
        if rng.random() < 0.5:
            a[k] = rng.choice(NONSTR_ARGS)
    return a


def rand_conversation(rng, tools, allow_mid_system=True):
    msgs = []
    if rng.random() < 0.85:
        msgs.append({"role": "system", "content": rng.choice(TEXTS[1:] + [""])})
    msgs.append({"role": "user", "content": rng.choice(TEXTS[1:])})
    n_turns = rng.randint(0, 6)
    cid = 0
    for _ in range(n_turns):
        rc = rng.choice(TEXTS)
        content = rng.choice(["", None, "Now the tests.", "multi\nline content"])
        a = {"role": "assistant", "content": content, "reasoning_content": rc}
        k = rng.choice([0, 1, 1, 1, 2, 3]) if tools else 0
        calls = []
        for _ in range(k):
            t = rng.choice(tools)
            args = rand_args(rng, t)
            enc_args = json.dumps(args) if rng.random() < 0.8 else (args if rng.random() < 0.5 else json.dumps(json.dumps(args)))
            calls.append({"id": f"call_{cid}", "type": "function",
                          "function": {"name": t["function"]["name"], "arguments": enc_args}})
            cid += 1
        if calls:
            a["tool_calls"] = calls
        msgs.append(a)
        if calls:
            res = [{"role": "tool", "tool_call_id": c["id"], "content": rng.choice(TEXTS)} for c in calls]
            rng.shuffle(res)
            msgs += res
            if rng.random() < 0.25:
                msgs.append({"role": "user", "content": "Please call execute_command to continue, or reply with your final report."})
        else:
            msgs.append({"role": "user", "content": "No tool calls detected in your response."})
        if allow_mid_system and rng.random() < 0.1:
            msgs.append({"role": "system", "content": "Reminder: " + rng.choice(TEXTS[1:4])})
            if rng.random() < 0.5:
                msgs.append({"role": "user", "content": "continue"})
    return msgs


def to_official_by_hand(msgs, tools):
    """Official-format input written independently of the provider (README: tools on the system message)."""
    out = copy.deepcopy(with_leading_system_for_tools(msgs, tools))
    for m in out:
        if m["role"] == "assistant" and m.get("content") is None:
            m["content"] = ""
    if tools:
        out[0]["tools"] = tools
    return out


EFFORTS = ["low", "high", "max", None, 1, 62, 99]


@pytest.mark.parametrize("seed", range(150))
def test_fuzz_render_equals_spec_and_official(enc, seed):
    rng = random.Random(seed)
    tools = rng.choice([ALL_TOOLS, [EXECUTE_COMMAND_TOOL], [EXECUTE_COMMAND_TOOL, DONE_TOOL], []])
    msgs = rand_conversation(rng, tools)
    effort = EFFORTS[seed % len(EFFORTS)]
    p = make_provider(msgs, tools=tools, reasoning_effort=effort)
    rendered = p.render()
    official = enc.encode_messages(to_official_by_hand(msgs, tools), thinking_mode="thinking", reasoning_effort=effort)
    spec = spec_render(with_leading_system_for_tools(msgs, tools), tools, effort)
    assert rendered == official, "provider != official encode_messages"
    assert rendered == spec, "provider != independent spec renderer"
    # effort line exactly once and right after BOS + <｜System｜>
    assert rendered.count("Reasoning Effort:") == 1 and rendered.startswith(BOS + SYS + "Reasoning Effort: ")
    assert rendered.count(BOS) == 1
    assert rendered.endswith(ASSIST + THINK)
    # harness inputs untouched by rendering
    assert p.messages == msgs


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_prefix_stability_with_tools(enc, seed):
    """With tools, the render of every step is a byte prefix of the next (all reasoning kept): a continuation from any
    earlier step conditions on exactly the string the full run showed the model."""
    rng = random.Random(1000 + seed)
    msgs = rand_conversation(rng, ALL_TOOLS, allow_mid_system=False)
    cuts = [i for i in range(2, len(msgs) + 1)
            if msgs[i - 1]["role"] in ("user", "tool", "system") and (i == len(msgs) or msgs[i]["role"] == "assistant")]
    renders = [make_provider(msgs[:i], tools=ALL_TOOLS).render() for i in cuts]
    for a, b in zip(renders, renders[1:]):
        assert b.startswith(a)


def test_tokens_special_markers_single_ids(tok):
    p = make_provider(rand_conversation(random.Random(5), ALL_TOOLS), tools=ALL_TOOLS)
    ids = tok.encode(p.render(), add_special_tokens=False).ids
    for t in (BOS, SYS, USER, ASSIST, THINK, ENDTHINK, EOS, "｜DSML｜"):
        tid = tok.token_to_id(t)
        assert tid is not None, t
        assert tok.encode(t, add_special_tokens=False).ids == [tid], t
    assert ids[0] == tok.token_to_id(BOS) and ids.count(tok.token_to_id(BOS)) == 1


def test_mid_conversation_system_literal(enc):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [{"id": "a", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "out"},
            {"role": "system", "content": "Only 3 steps left."}]
    r = make_provider(msgs, tools=[EXECUTE_COMMAND_TOOL]).render()
    assert r.endswith(USER + "<tool_result>out</tool_result>" + SYS + "Only 3 steps left." + ASSIST + THINK)
    assert r.count(SYS) == 2 and r.count("Reasoning Effort") == 1
    r2 = make_provider(msgs + [{"role": "user", "content": "go"}], tools=[EXECUTE_COMMAND_TOOL]).render()
    assert r2.endswith(SYS + "Only 3 steps left." + USER + "go" + ASSIST + THINK)


def test_zero_arg_and_mixed_arg_literals():
    calls = [{"id": "d", "type": "function", "function": {"name": "done", "arguments": "{}"}},
             {"id": "s", "type": "function", "function": {"name": "submit", "arguments": json.dumps(
                 {"answer": "123", "confidence": 0.5, "files": ["ü.py"], "final": None})}}]
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "s", "content": "B"}, {"role": "tool", "tool_call_id": "d", "content": "A"},
            {"role": "user", "content": "nudge"}]
    r = make_provider(msgs, tools=ALL_TOOLS).render()
    assert ('r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="done">\n\n</｜DSML｜ invoke>\n'
            '<｜DSML｜ invoke name="submit">\n'
            '<｜DSML｜ parameter name="answer" string="true">123</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="confidence" string="false">0.5</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="files" string="false">["ü.py"]</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="final" string="false">null</｜DSML｜ parameter>\n'
            '</｜DSML｜ invoke>\n</｜DSML｜ calls>' + EOS +
            USER + "<tool_result>A</tool_result>\n\n<tool_result>B</tool_result>\n\nnudge" + ASSIST + THINK) in r
