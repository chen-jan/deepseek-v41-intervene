"""Adversarial template-fidelity review (28 Oct 2026), part 2: sample -> official parse -> stored message -> next
official render, through the mock vLLM server, plus edge shapes where the provider's render is NOT the official one.

Tests marked xfail(strict=True) asserted the official/expected behaviour and failed: each was a finding. After the
review fixes those markers were removed; tests that asserted a reviewed behaviour now assert the decision.
    python -m pytest -q -rxX tests/review_template-fidelity_roundtrip.py
"""

import copy
import importlib
import json
import random

import pytest

from conftest import EXECUTE_COMMAND_TOOL, make_provider

S = importlib.import_module("review_template-fidelity_spec")
EOS, ASSIST, THINK = S.EOS, S.ASSIST, S.THINK


class TR:
    def __init__(self, id, content):
        self.id, self.name, self.content = id, "x", content


def _sample_text(rc, content, calls):
    return rc + "</think>" + (content or "") + (S.spec_tool_block(calls) if calls else "")


@pytest.mark.parametrize("seed", range(120))
def test_fuzz_sample_parse_rerender_is_lossless(enc, mock_server, seed):
    """A canonical official-format sample (with or without a prefill cut) is stored so that the next prompt contains
    exactly the sampled bytes, and the harness gets JSON-equal arguments."""
    rng = random.Random(5000 + seed)
    tools = S.ALL_TOOLS
    hist = S.rand_conversation(rng, tools)
    rc = rng.choice([t for t in S.TEXTS])
    content = rng.choice(["", "Now the tests.", "ü ✓\n"])
    calls, want_args = [], []
    for i in range(rng.choice([0, 1, 1, 2, 3])):
        t = rng.choice(tools)
        a = S.rand_args(rng, t)
        want_args.append(a)
        calls.append({"id": f"z{i}", "type": "function", "function": {"name": t["function"]["name"], "arguments": json.dumps(a)}})
    text = _sample_text(rc, content, calls)
    p = make_provider(hist, tools=tools, base_url=mock_server.url, api_key=mock_server.api_key)
    cut = rng.randint(0, len(rc)) if (rc and rng.random() < 0.5) else 0
    if cut:
        p.set_prefill(rc[:cut])
    base = make_provider(hist, tools=tools).render()
    mock_server.script.append(text[cut:])
    resp = p.invoke()
    rec = p.calls[-1]
    assert mock_server.completion_bodies[-1]["prompt"] == base + rc[:cut]
    assert rec["parse_ok"] and rec["render_roundtrip_exact"] is True
    m = p.messages[-1]
    assert m["reasoning_content"] == rc and m["content"] == content and not m.get("_prefill")
    assert [tc.name for tc in resp.tool_calls or []] == [c["function"]["name"] for c in calls]
    assert [json.loads(tc.arguments) for tc in resp.tool_calls or []] == want_args
    # next turn: the prompt extends exactly what was sampled
    if resp.tool_calls:
        for tc in reversed(resp.tool_calls):  # out of order on purpose
            p.add_tool_result(TR(tc.id, "res"))
    else:
        p.add_message({"role": "user", "content": "continue"})
    assert p.render().startswith(base + text + EOS)


# --- what re-renders differently from what was sampled (render_roundtrip_exact=False), beyond README item 3 --------------

INV = '<｜DSML｜ invoke name="{n}">\n{p}\n</｜DSML｜ invoke>'
PAR = '<｜DSML｜ parameter name="{k}" string="{s}">{v}</｜DSML｜ parameter>'


def _tc_text(name, params):
    return "r</think>\n\n<｜DSML｜ calls>\n" + INV.format(n=name, p="\n".join(PAR.format(k=k, s=s, v=v) for k, s, v in params)) \
        + "\n</｜DSML｜ calls>"


NONCANON = {
    "float_trailing_zero": ("submit", [("answer", "true", "a"), ("confidence", "false", "0.90")]),
    "exp_notation": ("submit", [("answer", "true", "a"), ("confidence", "false", "1e5")]),
    "ascii_escape_in_json": ("submit", [("answer", "true", "a"), ("files", "false", '["\\u00fc.py"]')]),
    "string_as_json_string": ("execute_command", [("command", "false", '"ls"')]),
    "trailing_newline_nonstring": ("submit", [("answer", "true", "a"), ("confidence", "false", "1\n")]),
}


@pytest.mark.parametrize("case", list(NONCANON))
def test_noncanonical_outputs_rerender_differently(mock_server, case):
    name, params = NONCANON[case]
    p = make_provider([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tools=S.ALL_TOOLS,
                      base_url=mock_server.url, api_key=mock_server.api_key)
    mock_server.script.append(_tc_text(name, params))
    p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] is True and rec["render_roundtrip_exact"] is False  # counted, as README says


def test_zero_arg_call_single_newline_parses_but_rerenders_with_blank_line(mock_server):
    """Official render of a zero-argument call is invoke>\\n\\n</invoke; the parser also accepts a single \\n (via
    Python's `$`), and the stored turn then re-renders with the blank line the model did not write."""
    p = make_provider([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tools=S.ALL_TOOLS,
                      base_url=mock_server.url, api_key=mock_server.api_key)
    mock_server.script.append('r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="done">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    resp = p.invoke()
    assert p.calls[-1]["parse_ok"] and resp.tool_calls[0].name == "done" and resp.tool_calls[0].arguments == "{}"
    assert p.calls[-1]["render_roundtrip_exact"] is False
    p.add_tool_result(TR(resp.tool_calls[0].id, "ok"))
    assert '<｜DSML｜ invoke name="done">\n\n</｜DSML｜ invoke>' in p.render()


def test_string_false_with_bare_text_is_rejected_as_malformed(enc, mock_server):
    """string="false" with non-JSON text. The official parser pastes the value into the arguments unchecked
    ('{"command": ls -la}'), and the official re-render of such a stored turn would show a single `arguments` STRING
    parameter holding the broken JSON -- a different call than the one the model wrote. FIXED: the provider validates
    the arguments as JSON and treats the turn as malformed (parse_ok False, n_invalid_args 1, no tool call, no content),
    so history never holds that call."""
    p = make_provider([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tools=[EXECUTE_COMMAND_TOOL],
                      base_url=mock_server.url, api_key=mock_server.api_key, malformed_retries=0)
    mock_server.script.append(_tc_text("execute_command", [("command", "false", "ls -la")]))
    resp = p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] is False and rec["n_invalid_args"] == 1 and rec["n_tool_calls"] == 0
    assert "invalid JSON arguments" in rec["parse_error"] and rec["fallback_has_dsml"] is True
    assert resp.tool_calls is None and resp.response is None
    # what the official encoder would have shown had the call been kept (the reviewed failure)
    bad = BASE + [{"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": ls -la}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "x"}]
    off = copy.deepcopy(bad)
    off[0]["tools"] = [EXECUTE_COMMAND_TOOL]
    assert '<｜DSML｜ parameter name="arguments" string="true">{"command": ls -la}' in enc.encode_messages(off, thinking_mode="thinking")
    p.add_message({"role": "user", "content": "Please use the tool."})
    assert 'name="arguments"' not in p.render()


# --- shapes where the provider render != official encode_messages on the same message ----------------------------------

BASE = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
ONE_CALL = {"role": "assistant", "content": "", "reasoning_content": "r",
            "tool_calls": [{"id": "a", "type": "function", "function": {"name": "execute_command", "arguments": '{"command": "ls"}'}}]}


def test_tool_content_none_renders_empty_deliberate_deviation(enc):
    """FINDING (kept as a documented deviation): tool content None renders <tool_result></tool_result>; the official
    encoder formats the Python repr, <tool_result>None</tool_result>. The harness's ToolResult.content is a str, so this
    never occurs in runs; '' is the meaningful rendering of "no output"."""
    msgs = BASE + [ONE_CALL, {"role": "tool", "tool_call_id": "a", "content": None}]
    off = copy.deepcopy(msgs)
    off[0]["tools"] = [EXECUTE_COMMAND_TOOL]
    prov = make_provider(msgs, tools=[EXECUTE_COMMAND_TOOL]).render()
    official = enc.encode_messages(off, thinking_mode="thinking")
    assert "<tool_result></tool_result>" in prov and "<tool_result>None</tool_result>" in official
    assert prov == official.replace("<tool_result>None</tool_result>", "<tool_result></tool_result>")


def test_function_level_namespace_kept(enc):
    """FIXED: function-level namespace (an official input form) is passed to the encoder."""
    tool = copy.deepcopy(EXECUTE_COMMAND_TOOL)
    tool["namespace"] = "sh"
    call = copy.deepcopy(ONE_CALL)
    call["tool_calls"][0]["function"]["namespace"] = "sh"
    msgs = BASE + [call, {"role": "tool", "tool_call_id": "a", "content": "x"}]
    off = copy.deepcopy(msgs)
    off[0]["tools"] = [tool]
    assert make_provider(msgs, tools=[tool]).render() == enc.encode_messages(off, thinking_mode="thinking")


def test_user_then_system_tools_placement_differs_from_vllm_chat_path(enc):
    """vLLM 0.31 tokenizers/deepseek_v41.py:64-68 attaches request tools to the FIRST system message anywhere; the
    provider attaches only to messages[0] and otherwise inserts an empty system message at index 0."""
    msgs = [{"role": "user", "content": "U"}, {"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    vllm_style = copy.deepcopy(msgs)
    vllm_style[1]["tools"] = [EXECUTE_COMMAND_TOOL]
    prov = make_provider(msgs, tools=[EXECUTE_COMMAND_TOOL]).render()
    assert prov != enc.encode_messages(vllm_style, thinking_mode="thinking")
    assert prov.startswith(S.BOS + S.SYS + S.EFFORT_LINE.format(b=75) + "\n\n## Tools")


def test_no_system_tools_placement_vs_official_load_cases(enc):
    """Official load_cases() puts case tools on messages[0] whatever its role; on a user-first conversation the user
    branch never renders tools, so the official test-case path would show NO tool section. The provider (like vLLM)
    inserts an empty system message instead. Harness envs that drop an empty system prompt (palisades_bounty) hit this."""
    msgs = [{"role": "user", "content": "U"}]
    load_cases_style = copy.deepcopy(msgs)
    load_cases_style[0]["tools"] = [EXECUTE_COMMAND_TOOL]
    lc = enc.encode_messages(load_cases_style, thinking_mode="thinking")
    prov = make_provider(msgs, tools=[EXECUTE_COMMAND_TOOL]).render()
    assert "## Tools" not in lc and "## Tools" in prov
    assert prov.startswith(S.BOS + S.SYS + S.EFFORT_LINE.format(b=75) + "\n\n## Tools")  # effort line + blank line


def test_prefill_stub_with_reasoning_alias_is_continued():
    """FIXED: a prefill stub spelled with the `reasoning` alias (accepted for history messages) is continued."""
    p = make_provider(BASE)
    p.messages.append({"role": "assistant", "reasoning": "partial thought", "_prefill": True})
    assert p.render().endswith(ASSIST + THINK + "partial thought")


def test_prefill_stub_with_content_matches_official_wo_eos(enc):
    """FIXED: a stub with content is rendered the official way for a turn prefix with content (wo_eos=True, which
    closes </think>)."""
    p = make_provider(BASE, tools=[EXECUTE_COMMAND_TOOL])
    p.messages.append({"role": "assistant", "reasoning_content": "done thinking", "content": "Answer: ", "_prefill": True})
    off = copy.deepcopy(BASE) + [{"role": "assistant", "reasoning_content": "done thinking", "content": "Answer: ", "wo_eos": True}]
    off[0]["tools"] = [EXECUTE_COMMAND_TOOL]
    assert p.render() == enc.encode_messages(off, thinking_mode="thinking")


def test_reasoning_only_prefill_equals_official_partial_turn(enc):
    """Positive control for the prefill rule: base + partial reasoning is a byte prefix of the official render of the
    finished turn and of the official wo_eos render (which adds </think>), for tools and no-tools."""
    for tools in ([EXECUTE_COMMAND_TOOL], []):
        p = make_provider(BASE, tools=tools)
        p.set_prefill("half a tho")
        prompt = p.render()
        off = copy.deepcopy(BASE) + [{"role": "assistant", "reasoning_content": "half a thought.", "content": "", "wo_eos": True}]
        if tools:
            off[0]["tools"] = tools
        assert enc.encode_messages(off, thinking_mode="thinking").startswith(prompt)


@pytest.mark.parametrize("where", ["tool", "reasoning"])
def test_image_placeholder_in_tool_output_or_reasoning_crashes_render(where):
    """The official encoder raises on the literal image placeholder in any message text. FIXED: the provider re-raises
    it as PromptRenderError (a ValueError whose message the harness classifies as censored, see the adversarial
    review's test_obs_image_placeholder_in_tool_output_crashes_the_episode)."""
    msgs = BASE + [copy.deepcopy(ONE_CALL), {"role": "tool", "tool_call_id": "a", "content": "x"}]
    if where == "tool":
        msgs[-1]["content"] = "tokenizer.json: \"<｜deepseek_image｜>\": 128815"
    else:
        msgs[2]["reasoning_content"] = "the token <｜deepseek_image｜> is special"
    with pytest.raises(ValueError, match="image") as ei:
        make_provider(msgs, tools=[EXECUTE_COMMAND_TOOL]).render()
    assert isinstance(ei.value, S.dsp.PromptRenderError)


def test_effort_aliases_render_identically_to_ints(enc):
    for alias, n in (("low", 50), ("high", 75), ("max", 100), ("HIGH", 75), (" max ", 100), ("062", 62)):
        assert make_provider(BASE, reasoning_effort=alias).render() == make_provider(BASE, reasoning_effort=n).render()
