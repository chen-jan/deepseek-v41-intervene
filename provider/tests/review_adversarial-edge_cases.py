"""Adversarial edge-case review of deepseek_v41_vllm_provider.py (28 Oct 2026).

Convention: a test named ``test_issue_*`` asserted the behaviour the provider SHOULD have and was marked
``xfail(strict=True)``. ``test_obs_*`` tests asserted the reviewed behaviour and print evidence. ``test_ok_*`` tests are
edge cases that behave correctly. After the review fixes the xfail markers are gone and each
``test_obs_*`` test asserts the decision taken (FIXED or KEPT, stated in its docstring).

Run:  python -m pytest -q -rxs tests/review_adversarial-edge_cases.py
"""

import copy
import json
import os
import pickle
import re

import pytest

from conftest import TOOLS, MockVLLM, attach_tools, call, conversations, dsp, make_provider

HISTORY = conversations()["basic"]
CALL = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter name="command" '
        'string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
# a harness environment's malformed-reply pattern (a reply matching this is re-prompted; anything else is the final report)
HARNESS_MALFORMED_RE = re.compile(r"default_api|execute_command\s*\{|<call[:_]|```tool_code")


def _prov(mock, history=HISTORY, **kw):
    return make_provider(history, base_url=mock.url, api_key=mock.api_key, **kw)


def _n_prompt(tok, history=HISTORY):
    return len(tok.encode(make_provider(history).render(), add_special_tokens=False).ids)


M = dsp.DEFAULT_CONTEXT_MARGIN  # tokens the provider keeps free below max_model_len
# 8 tokens (as vLLM returns when max_tokens=8 runs out mid-reasoning): "Let me look at the failing test first"
CUT_REASONING = "Let me look at the failing test first"


# ----------------------------------------------------------------------------------------------------------------------
# 1. EOS missing / max_tokens hit
# ----------------------------------------------------------------------------------------------------------------------

def test_obs_context_exhausted_mid_reasoning_is_a_normal_no_tool_call_turn(tok):
    """max_tokens defaults to ALL remaining context. When the context fills mid-reasoning, the server returns
    finish_reason=length. It used to come back as a reasoning-only turn, so treat_no_tool_call_as_done envs
    ended the episode as a completed run with an empty report. FIXED: ContextLengthError after the call is
    recorded (context_exhausted=True), so the harness censors the episode; no malformed retry is spent on it."""
    from agent_interp_envs.runs import is_context_limit_error
    from agent_interp_envs.tool_calling import validate_and_get_command

    n = _n_prompt(tok)
    with MockVLLM(tok, max_model_len=n + 8 + M) as m:
        p = _prov(m)
        m.script.append({"text": CUT_REASONING, "finish_reason": "length"})
        with pytest.raises(dsp.ContextLengthError) as ei:
            validate_and_get_command(p, allowed_tools={"execute_command"}, treat_no_tool_call_as_done=True)
        rec = p.calls[-1]
        print("\nEVIDENCE context-exhausted:", {k: rec[k] for k in ("max_tokens", "max_model_len", "prompt_tokens",
                                                                    "finish_reason", "context_exhausted", "parse_ok")})
        assert len(tok.encode(CUT_REASONING, add_special_tokens=False).ids) == 8 == rec["completion_tokens"]
        assert rec["max_tokens"] == 8 and rec["finish_reason"] == "length" and rec["context_exhausted"] is True
        assert is_context_limit_error(ei.value) and len(p.calls) == 1 and p.messages == HISTORY


def test_issue_context_exhausted_during_generation_should_censor(tok):
    from agent_interp_envs.runs import is_context_limit_error

    n = _n_prompt(tok)
    with MockVLLM(tok, max_model_len=n + 8 + M) as m:
        p = _prov(m)
        m.script.append({"text": CUT_REASONING, "finish_reason": "length"})
        with pytest.raises(Exception) as ei:
            p.invoke()
        assert is_context_limit_error(ei.value) and "during generation" in str(ei.value)


def test_cap_truncation_is_not_context_exhaustion(tok):
    """A turn cut by the DSV41_MAX_TOKENS cap (not the context) is returned without content, not resampled."""
    with MockVLLM(tok) as m:
        p = _prov(m, max_tokens=8)
        m.script.append({"text": "ok</think>I fixed the bug by", "finish_reason": "length"})
        resp = p.invoke()
        rec = p.calls[-1]
        assert rec["context_exhausted"] is False and len(p.calls) == 1 and rec["parse_ok"] is False
        assert resp.response is None and p.messages[-1]["_dsv41"]["unparsed_content"] == "I fixed the bug by"


def test_obs_truncated_inside_dsml_block_becomes_final_report_text(tok):
    """Truncated after </think> inside the tool-call block: content used to be the raw DSML, graded as the final report
    by a treat-no-tool-call-as-done environment. FIXED: at the context limit this is censored
    (ContextLengthError); under a DSV41_MAX_TOKENS cap the turn has no content (the DSML is kept in _dsv41)."""
    text = ('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n'
            '<｜DSML｜ parameter name="command" string="true">python -m pytest')
    n = _n_prompt(tok)
    with MockVLLM(tok, max_model_len=n + 20 + M) as m:
        p = _prov(m)
        m.script.append({"text": text, "finish_reason": "length"})
        with pytest.raises(dsp.ContextLengthError):
            p.invoke()
    with MockVLLM(tok) as m:
        p = _prov(m, max_tokens=20)
        m.script.append({"text": text, "finish_reason": "length"})
        resp = p.invoke()
        print("\nEVIDENCE truncated-in-DSML response:", repr(resp.response))
        assert resp.tool_calls is None and resp.response is None and p.calls[-1]["fallback_has_dsml"] is True
        assert "python -m pytest" in p.messages[-1]["_dsv41"]["unparsed_content"]


def test_obs_length_stop_right_after_calls_block_executes_call_and_history_gains_unsampled_eos(enc, tok):
    """The official parser accepts a tool-call turn with no EOS (stop_token None is allowed after </｜DSML｜ calls>).
    The call executes, and the next prompt renders an EOS the model never produced. Benign but undetected:
    render_roundtrip_exact is None (only checked when ended_with_eos)."""
    n = _n_prompt(tok)
    completion = "ok</think>" + CALL
    k = len(tok.encode(completion, add_special_tokens=False).ids)
    with MockVLLM(tok, max_model_len=n + k + M) as m:
        p = _prov(m)
        m.script.append({"text": completion, "finish_reason": "length"})
        resp = p.invoke()
        rec = p.calls[-1]
        assert rec["finish_reason"] == "length" and rec["ended_with_eos"] is False and rec["context_exhausted"]
        assert rec["parse_ok"] is True and resp.tool_calls and resp.tool_calls[0].name == "execute_command"
        assert rec["render_roundtrip_exact"] is None
        p.add_tool_result(type("TR", (), {"id": resp.tool_calls[0].id, "content": "x"})())
        assert (completion + enc.eos_token) in p.render()


# ----------------------------------------------------------------------------------------------------------------------
# 2. Malformed / partial DSML
# ----------------------------------------------------------------------------------------------------------------------

def test_obs_string_false_with_non_json_value_parses_ok_but_args_are_invalid_json_and_history_is_rewritten(enc, mock_server):
    """Model mislabels a string as string="false". The official parser does not validate JSON (arguments
    '{"command": ls -la}'); the turn used to be parse_ok=True and, once kept, re-rendered as a call the model never
    wrote. FIXED: the arguments are validated; such a turn is malformed (resampled, here twice, then returned without
    tool calls or content), and the next prompt never shows the rewritten call."""
    bad = ('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter name="command" '
           'string="false">ls -la</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    p = _prov(mock_server)
    mock_server.script += [bad] * 3
    resp = p.invoke()
    rec = p.calls[-1]
    print("\nEVIDENCE", rec["parse_error"])
    assert [c["n_invalid_args"] for c in p.calls] == [1, 1, 1] and rec["parse_ok"] is False
    assert resp.tool_calls is None and resp.response is None
    p.add_message({"role": "user", "content": "Please use the tool."})
    rendered = p.render()
    assert 'name="arguments"' not in rendered and 'string="false">ls -la<' not in rendered


@pytest.mark.parametrize("reasoning", [
    "The repo's chat template wraps reasoning in <think> tags; I should just run ls.",
    "Tool calls use the <｜DSML｜ calls> block; let me run ls.",
    "The file mentions <｜end▁of▁sentence｜> as EOS. Run ls.",
])
def test_obs_well_formed_call_rejected_when_reasoning_mentions_a_marker(mock_server, reasoning):
    """A well-formed tool call whose reasoning merely MENTIONS <think>, ｜DSML｜, EOS/BOS (e.g. after cat-ing a template
    or tokenizer file) fails the official parser's final special-token assertion; the turn used to be dropped and its
    DSML block returned as the final report. FIXED: when only that assertion fails, the official parser is re-run with
    the mentions in the REASONING masked; the call is recovered (parse_lenient=True, the strict error kept), and the
    stored turn re-renders byte for byte."""
    p = _prov(mock_server)
    mock_server.script.append(reasoning + "</think>" + CALL)
    resp = p.invoke()
    rec = p.calls[-1]
    print("\nEVIDENCE", rec["strict_parse_error"], "| reasoning:", repr(resp.reasoning[:60]))
    assert rec["parse_ok"] is True and rec["parse_lenient"] is True and len(p.calls) == 1
    assert "Unexpected special token" in rec["strict_parse_error"] and rec["render_roundtrip_exact"] is True
    assert resp.reasoning == reasoning and resp.tool_calls[0].name == "execute_command"
    assert p.messages[-1]["_dsv41"]["parse_lenient"] is True


def test_markers_in_content_stay_strict(mock_server):
    """The lenient path covers the reasoning only: V4-style DSML tags or a second </think> after the first one are a
    malformed turn, not prose (otherwise a V4 tool block would become a final report)."""
    p = _prov(mock_server, malformed_retries=0)
    mock_server.script.append('ok</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="execute_command">\n'
                              '<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n'
                              '</｜DSML｜tool_calls>')
    resp = p.invoke()
    assert p.calls[-1]["parse_ok"] is False and p.calls[-1]["parse_lenient"] is False and resp.response is None


@pytest.mark.parametrize("name,text", [
    ("duplicate_param", 'ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n'
                        '<｜DSML｜ parameter name="command" string="true">ls</｜DSML｜ parameter>\n'
                        '<｜DSML｜ parameter name="command" string="true">pwd</｜DSML｜ parameter>\n'
                        '</｜DSML｜ invoke>\n</｜DSML｜ calls>'),
    ("string_attr_capitalised", 'ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n'
                                '<｜DSML｜ parameter name="command" string="True">ls</｜DSML｜ parameter>\n'
                                '</｜DSML｜ invoke>\n</｜DSML｜ calls>'),
    ("text_after_calls_block", "ok</think>" + CALL + "\nDone."),
    ("missing_calls_close", 'ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n'
                            '<｜DSML｜ parameter name="command" string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>'),
    ("second_think_close_in_content", "ok</think>Answer: use </think> carefully."),
])
def test_ok_malformed_variants_give_no_tool_call_and_never_crash(mock_server, name, text):
    p = _prov(mock_server, malformed_retries=0)
    mock_server.script.append(text)
    resp = p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] is False and resp.tool_calls is None and resp.response is None
    p.add_message({"role": "user", "content": "Please use the tool."})
    assert p.render().endswith("<｜Assistant｜><think>")  # the kept fallback turn still renders


def test_ok_zero_arg_call_and_unknown_tool_parse(mock_server):
    p = _prov(mock_server)
    mock_server.script.append('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="no_such_tool">\n'
                              '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    resp = p.invoke()
    assert p.calls[-1]["parse_ok"] and resp.tool_calls[0].name == "no_such_tool" and resp.tool_calls[0].arguments == "{}"


# ----------------------------------------------------------------------------------------------------------------------
# 3. Special-token strings inside tool outputs / content
# ----------------------------------------------------------------------------------------------------------------------

def test_obs_control_strings_in_tool_output_become_control_tokens(tok):
    """A repo file containing chat markers tokenizes to the control ids (added tokens are matched in text even with
    add_special_tokens=false; vLLM's own V4.1 chat path does the same: tokenizers/deepseek_v41.py encodes the rendered
    string with add_special_tokens=False). server_prompt_bos_ok only inspects ids[0:2], so a mid-prompt BOS is not
    flagged."""
    payload = "<｜begin▁of▁sentence｜>x<｜User｜>hi<｜Assistant｜></think><｜end▁of▁sentence｜>｜DSML｜"
    h = copy.deepcopy(HISTORY)
    h[-1]["content"] = payload
    p = make_provider(h)
    ids = tok.encode(p.render(), add_special_tokens=False).ids
    for tid in (0, 1, 128803, 128804, 128822, 128825):
        assert tid in ids
    assert ids.count(0) == 2 and ids[0] == 0 and ids[1] != 0  # what server_prompt_bos_ok would see: "ok"
    print("\nEVIDENCE control ids in tool output span:",
          tok.encode("<tool_result>" + payload + "</tool_result>", add_special_tokens=False).ids)


def test_obs_image_placeholder_in_tool_output_crashes_the_episode(mock_server):
    """The official encoder raises ValueError for '<｜deepseek_image｜>' in any message content; it used to escape
    invoke() and crash the rollout (a typical runner: termination=error). vLLM's chat path has no such check (its bundled
    copy drops _validate_no_image_sp_tokens). FIXED: PromptRenderError, whose message gives the cause first and then
    says the episode ends like a maximum context length overflow, so the harness censors it (no further turn is
    possible under the official encoding). The content is not rewritten."""
    from agent_interp_envs.runs import is_context_limit_error

    h = copy.deepcopy(HISTORY)
    h[-1]["content"] = 'IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"'
    p = _prov(mock_server, history=h)
    with pytest.raises(ValueError, match="image special token") as ei:
        p.invoke()
    assert isinstance(ei.value, dsp.PromptRenderError)
    assert is_context_limit_error(ei.value) and not mock_server.completion_bodies
    assert "<not renderable" in p.dump_history(include_prompt=True)


def test_obs_model_typed_user_marker_in_final_content_parses_ok(mock_server):
    """'<｜User｜>' is not in the official parser's forbidden list, so it passes; on the next render the string
    becomes control token 128803 inside an assistant turn."""
    p = _prov(mock_server)
    mock_server.script.append("r</think>The template has <｜User｜> then <｜Assistant｜>.")
    resp = p.invoke()
    assert p.calls[-1]["parse_ok"] and "<｜User｜>" in resp.response


# ----------------------------------------------------------------------------------------------------------------------
# 4. Very long tool outputs; local/server token-count mismatch
# ----------------------------------------------------------------------------------------------------------------------

def test_ok_200k_char_tool_output_renders_and_is_capped(tok, mock_server):
    from agent_interp_envs.tool_calling import MAX_TOOL_OUTPUT_CHARS, truncate_output

    out = truncate_output(("ünï 漢字 line\t" * 40000), MAX_TOOL_OUTPUT_CHARS)
    h = copy.deepcopy(HISTORY)
    h[-1]["content"] = out
    p = _prov(mock_server, history=h)
    mock_server.script.append("r</think>" + CALL)
    resp = p.invoke()
    rec = p.calls[-1]
    assert resp.tool_calls and rec["server_prompt_ids_match_local"] is True
    assert rec["max_tokens"] == 131072 - rec["prompt_tokens_local"] - M
    print("\nEVIDENCE 200k-char tool output -> prompt tokens", rec["prompt_tokens_local"])


class _OffByOneServer(MockVLLM):
    """Server tokenizer yields one more token than the local tokenizer (any wrapper/version drift)."""

    def server_tokenize(self, prompt, add_special_tokens):
        ids = super().server_tokenize(prompt, add_special_tokens)
        return ids if isinstance(prompt, list) else ids + [220]


def test_issue_one_token_count_drift_censors_every_call(tok):
    """max_tokens used to be ctx - n_local exactly: a server counting one token more turned every request into a
    context-length 400 (every episode censored at step 0, no call record). FIXED: a margin (DSV41_CONTEXT_MARGIN,
    default 16) keeps the call valid, the record shows server_prompt_ids_match_local=False, and the strict check fails
    loudly (FidelityError, not a censor). With DSV41_STRICT_CHECKS=0 the call is recorded and kept."""
    with _OffByOneServer(tok) as m:
        p = _prov(m)
        m.script.append("r</think>" + CALL)
        try:
            with pytest.raises(dsp.FidelityError):
                p.invoke()
        finally:
            print("\nEVIDENCE drift: calls recorded =", len(p.calls))
        assert p.calls[-1]["server_prompt_ids_match_local"] is False and p.messages == HISTORY
        q = _prov(m, strict_checks=False)
        m.script.append("r</think>" + CALL)
        assert q.invoke().tool_calls and q.calls[-1]["server_prompt_ids_match_local"] is False


def test_server_context_400_leaves_a_diagnostic_record(tok):
    """A context-length 400 from the server (counts drifted by more than the margin) is still censored, but the failed
    call now leaves a record with the local count, so the cause is visible."""
    from agent_interp_envs.runs import is_context_limit_error

    class _FarOffServer(MockVLLM):
        def server_tokenize(self, prompt, add_special_tokens):
            ids = MockVLLM.server_tokenize(self, prompt, add_special_tokens)
            return ids if isinstance(prompt, list) else ids + [220] * 100

    with _FarOffServer(tok) as m:
        p = _prov(m)
        with pytest.raises(dsp.ContextLengthError) as ei:
            p.invoke()
        assert is_context_limit_error(ei.value)
        rec = p.calls[-1]
        assert "maximum context length" in rec["error"] and rec["prompt_tokens_local"] == _n_prompt(tok)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Unicode
# ----------------------------------------------------------------------------------------------------------------------

def test_ok_unicode_roundtrip_in_reasoning_args_and_tool_output(enc, mock_server):
    weird = "naïve é 漢字 🙂‍↔️ ​ RTL:‮abc‬ NUL:\x00 CRLF:\r\n tab:\t"
    h = copy.deepcopy(HISTORY)
    h[-1]["content"] = weird
    p = _prov(mock_server, history=h)
    text = (weird + "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"execute_command\">\n<｜DSML｜ parameter "
            "name=\"command\" string=\"true\">echo '" + weird + "'</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    mock_server.script.append(text)
    resp = p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] and rec["render_roundtrip_exact"] is True and rec["server_prompt_ids_match_local"] is True
    assert json.loads(resp.tool_calls[0].arguments)["command"] == "echo '" + weird + "'"
    assert resp.reasoning == weird


def test_obs_lone_surrogate_in_tool_output_crashes_before_any_request(mock_server):
    """Unreachable through run_command (decode(errors="replace")), but any tool that returns str from a
    surrogateescape decode used to crash the episode (TypeError from tokenizers). FIXED: PromptRenderError before any
    request, censored like the image placeholder."""
    from agent_interp_envs.runs import is_context_limit_error

    h = copy.deepcopy(HISTORY)
    h[-1]["content"] = "bad byte: \udcff"
    p = _prov(mock_server, history=h)
    with pytest.raises(Exception) as ei:
        p.invoke()
    print("\nEVIDENCE lone surrogate ->", type(ei.value).__name__, str(ei.value)[:80])
    assert not mock_server.completion_bodies
    assert isinstance(ei.value, dsp.PromptRenderError) and is_context_limit_error(ei.value)


# ----------------------------------------------------------------------------------------------------------------------
# 6. Empty reasoning, empty completion, multiple tool calls
# ----------------------------------------------------------------------------------------------------------------------

def test_ok_empty_reasoning_roundtrips(mock_server):
    p = _prov(mock_server)
    mock_server.script.append("</think>" + CALL)
    resp = p.invoke()
    rec = p.calls[-1]
    assert rec["parse_ok"] and rec["render_roundtrip_exact"] is True and resp.reasoning == "" and resp.tool_calls


def test_obs_immediate_eos_is_an_empty_final_report(mock_server):
    p = _prov(mock_server, malformed_retries=0)
    mock_server.script.append("")
    resp = p.invoke()
    assert p.calls[-1]["ended_with_eos"] and not p.calls[-1]["parse_ok"]
    assert resp.reasoning == "" and resp.response is None and resp.tool_calls is None


def test_ok_parallel_calls_harness_path_renders_official(enc, mock_server):
    """3x parallel calls -> harness gives up, adds one error tool result per call; the next prompt is the official
    render (results sorted by call order)."""
    from agent_interp_envs.tool_calling import validate_and_get_command

    two = ('two</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter name="command" '
           'string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="execute_command">\n'
           '<｜DSML｜ parameter name="command" string="true">pwd</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    p = _prov(mock_server)
    mock_server.script += [two, two, two]
    result = validate_and_get_command(p, allowed_tools={"execute_command"})
    assert result.error_type == "parallel_tool_calls"
    ids = [c["id"] for c in p.messages[-3]["tool_calls"]]
    assert len(set(ids)) == 2 and [m["tool_call_id"] for m in p.messages[-2:]] == ids
    assert p.calls[-1]["render_roundtrip_exact"] is True
    assert p.render() == enc.encode_messages(attach_tools(p.messages, TOOLS), thinking_mode="thinking")


# ----------------------------------------------------------------------------------------------------------------------
# 7. Effort values
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, "high"), ("low", "low"), (" HIGH ", "high"), ("max", "max"), (1, 1), (100, 100), ("62", 62), (" 37 ", 37),
    ("６２", 62),  # full-width digits are accepted as 62
])
def test_ok_effort_accepted(enc, value, expected):
    assert dsp.normalize_effort(value, enc) == expected
    p = make_provider(HISTORY, reasoning_effort=value)
    budget = {"low": 50, "high": 75, "max": 100}.get(expected, expected)
    assert p.render().startswith(f"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100")


@pytest.mark.parametrize("value", [0, 101, -5, "-5", "+62", 75.0, "75.0", True, False, "medium", "xhigh", "none", "",
                                   "²"])
def test_ok_effort_rejected(enc, value):
    with pytest.raises(ValueError):
        dsp.normalize_effort(value, enc)


# ----------------------------------------------------------------------------------------------------------------------
# 8. Double BOS
# ----------------------------------------------------------------------------------------------------------------------

def test_ok_double_bos_guards(tok, mock_server):
    for mode in ("text", "token_ids"):
        p = _prov(mock_server, prompt_mode=mode)
        mock_server.script.append("r</think>" + CALL)
        p.invoke()
        body = mock_server.completion_bodies[-1]
        sent = body["prompt"] if mode == "token_ids" else tok.encode(body["prompt"], add_special_tokens=False).ids
        assert sent[0] == 0 and sent[1] != 0 and p.calls[-1]["server_prompt_bos_ok"] is True
        assert body["add_special_tokens"] is False


# ----------------------------------------------------------------------------------------------------------------------
# 9. Secret handling
# ----------------------------------------------------------------------------------------------------------------------

def test_ok_key_absent_from_prints_records_errors_and_repr(tok, capsys, tmp_path):
    key = "sk-REVIEW-SECRET-777"
    with MockVLLM(tok, api_key=key) as m:
        m.fail_first += [503, 500]
        log = tmp_path / "calls.jsonl"
        p = make_provider(HISTORY, base_url=m.url, api_key=key, verbose=True, call_log_path=log)
        m.script.append("r</think>" + CALL)
        p.invoke()
        bad = make_provider(HISTORY, base_url=m.url, api_key=key + "x", verbose=True)
        with pytest.raises(dsp.ServerError) as e401:
            bad.invoke()
    dead = make_provider(HISTORY, base_url="http://127.0.0.1:9", api_key=key, verbose=True)
    dead.max_attempts = 2
    with pytest.raises(ConnectionError) as econn:
        dead.invoke()
    out = capsys.readouterr()
    blob = (out.out + out.err + log.read_text() + p.dump_history(include_prompt=True) + json.dumps(p.calls)
            + repr(p) + repr(p._transport) + str(e401.value) + str(econn.value) + repr(dead))
    assert key not in blob and "[provider-retry]" in out.out


def test_obs_key_is_reachable_via_object_state_and_pickle(mock_server):
    """Not printed anywhere. The provider itself cannot be pickled/deep-copied (it holds the encoding module), so the
    key is reachable only through object state (vars / a pickled transport)."""
    key = mock_server.api_key
    p = _prov(mock_server)
    with pytest.raises(TypeError):
        pickle.dumps(p)
    assert key in pickle.dumps(p._transport).decode("latin-1")
    assert vars(p._transport)["_api_key"] == key


def test_obs_module_create_provider_does_not_scrub_env_key(monkeypatch, mock_server):
    """README claimed 'The harness scrubs *_API_KEY from the environment after construction', but only the harness's
    create_provider does, so direct construction left DSV41_API_KEY in os.environ for an unhardened agent shell.
    FIXED: every construction reads DSV41_API_KEY / VLLM_API_KEY once and removes both from os.environ; the key is
    cached in the module so later providers in the same process (continuation reps) still work."""
    monkeypatch.setenv("DSV41_API_KEY", mock_server.api_key)
    monkeypatch.setenv("VLLM_API_KEY", "sk-other-SECRET")
    monkeypatch.setenv("DSV41_BASE_URL", mock_server.url)
    dsp.create_provider("dsv41", copy.deepcopy(HISTORY), TOOLS, verbose=False)
    assert "DSV41_API_KEY" not in os.environ and "VLLM_API_KEY" not in os.environ
    second = dsp.DeepSeekV41VLLMProvider("dsv41", copy.deepcopy(HISTORY), TOOLS, verbose=False)
    mock_server.script.append("r</think>" + CALL)
    assert second.invoke().tool_calls
