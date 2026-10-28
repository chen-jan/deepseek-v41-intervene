"""(d) Parsing official renders of tool calls round-trips (and malformed output gives the harness a turn without tool
calls, which its retry logic handles)."""

import json

import pytest

from conftest import TOOLS, attach_tools, call, conversations, make_provider

HISTORY = conversations()["basic"]

TURNS = {
    "single_call": {"role": "assistant", "content": "", "reasoning_content": "Run the tests.",
                    "tool_calls": [call("a", "execute_command", {"command": "pytest -q"})]},
    "content_and_call": {"role": "assistant", "content": "Checking the diff now.",
                         "reasoning_content": "  spaces kept  \n\n- a list\n- of steps",
                         "tool_calls": [call("a", "execute_command", {"command": "git show HEAD --stat"})]},
    "parallel_calls": {"role": "assistant", "content": "", "reasoning_content": "Two at once.",
                       "tool_calls": [call("a", "execute_command", {"command": "ls"}),
                                      call("b", "execute_command", {"command": "cat setup.py | head -3"})]},
    "nonstring_args": {"role": "assistant", "content": "", "reasoning_content": "Submit.",
                       "tool_calls": [call("a", "submit", {"answer": "x < y && \"quoted\"", "confidence": 0.75,
                                                           "files": ["a.py", "b/c.py"], "final": False})]},
    "multiline_unicode_arg": {"role": "assistant", "content": "", "reasoning_content": "Write a file — ✓.",
                              "tool_calls": [call("a", "execute_command",
                                                  {"command": "cat > fix.py <<'EOF'\ndef f(x):\n    return x  # ü\nEOF"})]},
    "final_report": {"role": "assistant", "content": "I fixed the bug in Tree.add; all tests pass.",
                     "reasoning_content": "All tests pass; write the report."},
    "empty_reasoning": {"role": "assistant", "content": "", "reasoning_content": "",
                        "tool_calls": [call("a", "execute_command", {"command": "true"})]},
}


def _completion(enc, history, turn, effort="high"):
    """The completion the model would have produced for ``turn``: official render with the turn minus without it."""
    p = make_provider(history, reasoning_effort=effort)
    base = p.render()
    full = enc.encode_messages(attach_tools(history + [turn], TOOLS), thinking_mode="thinking", reasoning_effort=effort)
    assert full.startswith(base)
    return p, base, full, full[len(base):]


@pytest.mark.parametrize("name", list(TURNS))
def test_parse_roundtrips_official_render(enc, name):
    turn = TURNS[name]
    p, base, full, completion = _completion(enc, HISTORY, turn)
    msg, ok, err = p._parse("", completion)
    assert ok, err
    assert msg["reasoning_content"] == turn["reasoning_content"]
    assert msg["content"] == (turn["content"] or "")
    got = msg.get("tool_calls") or []
    want = turn.get("tool_calls") or []
    assert [c["function"]["name"] for c in got] == [c["function"]["name"] for c in want]
    assert [json.loads(c["function"]["arguments"]) for c in got] == [json.loads(c["function"]["arguments"]) for c in want]
    assert all(c["id"].startswith("chatcmpl-tool-") for c in got)
    # Re-rendering the parsed message reproduces the sampled string byte for byte.
    assert enc.encode_messages(attach_tools(HISTORY + [msg], TOOLS), thinking_mode="thinking") == full


@pytest.mark.parametrize("name", ["single_call", "nonstring_args", "final_report"])
def test_invoke_parses_server_completion_without_eos_in_text(enc, mock_server, name):
    """vLLM strips EOS from the text (finish_reason=stop, stop_reason=None); the provider re-attaches it."""
    turn = TURNS[name]
    p, base, full, completion = _completion(enc, HISTORY, turn)
    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key)
    assert completion.endswith(enc.eos_token)
    mock_server.script.append(completion[: -len(enc.eos_token)])
    resp = p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] and rec["ended_with_eos"] and rec["render_roundtrip_exact"] is True
    assert resp.reasoning == turn["reasoning_content"]
    assert (resp.tool_calls is None) == (not turn.get("tool_calls"))


def test_official_parser_example_from_the_official_tests(enc):
    p = make_provider(HISTORY)
    text = ('  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
            '</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>')
    msg, ok, _ = p._parse("", text)
    assert ok and msg["reasoning_content"] == "  reason  " and msg["content"] == "summary"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"query": "value", "limit": 2}


def test_noncanonical_json_value_is_flagged_not_roundtrip(enc, mock_server):
    """string="false" values re-render as json.dumps(...) (", " separators); a compact [1,2] does not round-trip."""
    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key)
    mock_server.script.append('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="submit">\n'
                              '<｜DSML｜ parameter name="files" string="false">["a","b"]</｜DSML｜ parameter>\n'
                              '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    resp = p.invoke()
    assert json.loads(resp.tool_calls[0].arguments) == {"files": ["a", "b"]}
    assert p.calls[-1]["parse_ok"] and p.calls[-1]["render_roundtrip_exact"] is False


MALFORMED = {
    # V4-style tags (what the V4 template teaches): the official V4.1 parser does not accept them.
    "v4_dsml_tags": ('ok</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="execute_command">\n<｜DSML｜parameter '
                     'name="command" string="true">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>', "stop"),
    "missing_blank_line_before_calls": ('ok</think><｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n'
                                        '<｜DSML｜ parameter name="command" string="true">ls</｜DSML｜ parameter>\n'
                                        '</｜DSML｜ invoke>\n</｜DSML｜ calls>', "stop"),
    "truncated_mid_reasoning": ("Let me think about the sort key and whether", "length"),
    "truncated_mid_call": ('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ par', "length"),
    "tool_call_inside_reasoning": ('think\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ '
                                   'parameter name="command" string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
                                   '</｜DSML｜ calls>', "stop"),
}


@pytest.mark.parametrize("name", list(MALFORMED))
def test_malformed_output_gives_no_tool_call_turn(mock_server, name):
    text, finish = MALFORMED[name]
    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key, malformed_retries=0)
    mock_server.script.append({"text": text, "finish_reason": finish})
    resp = p.invoke()
    rec = p.calls[-1]
    assert not rec["parse_ok"] and rec["parse_error"]
    assert resp.tool_calls is None
    assert rec["ended_with_eos"] == (finish == "stop")
    # Never hand malformed or truncated text back as content (review fix): reasoning only; the dropped text is kept.
    assert resp.response is None and p.messages[-1]["content"] is None
    if "</think>" not in text:
        assert resp.reasoning == text  # reasoning-only, as the Fireworks provider does
    else:
        assert p.messages[-1]["_dsv41"]["unparsed_content"] == text.split("</think>", 1)[1]
    assert rec["fallback_has_dsml"] == ("｜DSML｜" in text)
    # The stored message is still renderable (the harness may keep it after its retries run out).
    p.add_message({"role": "user", "content": "Please use the tool."})
    assert p.render().endswith("<｜Assistant｜><think>")


def test_harness_retry_logic_recovers_from_malformed_turn(mock_server):
    """The harness's own validate_and_get_command: malformed -> revert + retry -> valid command."""
    from agent_interp_envs.tool_calling import validate_and_get_command

    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key, malformed_retries=0)
    mock_server.script += [
        MALFORMED["v4_dsml_tags"][0],
        'fine</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter name="command" '
        'string="true">git status</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>',
    ]
    result = validate_and_get_command(p, allowed_tools={"execute_command"})
    assert result.success and result.command == "git status"
    assert len(p.messages) == len(HISTORY) + 1 and p.messages[-1]["reasoning_content"] == "fine"
    assert [c["parse_ok"] for c in p.calls] == [False, True]


def test_malformed_retries_option_resamples_inside_provider(mock_server):
    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key, malformed_retries=2)
    mock_server.script += [MALFORMED["v4_dsml_tags"][0], MALFORMED["missing_blank_line_before_calls"][0],
                           "done</think>Final report."]
    resp = p.invoke()
    assert resp.response == "Final report." and [c["malformed_retry"] for c in p.calls] == [0, 1, 2]
    assert len(p.messages) == len(HISTORY) + 1
