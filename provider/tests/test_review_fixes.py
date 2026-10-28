"""Tests for review fixes that the review test files (``review_*.py``) do not exercise end to end."""

import copy

import pytest

from conftest import TOOLS, MockVLLM, attach_tools, conversations, dsp, make_provider

HISTORY = conversations()["basic"]
CALL = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter name="command" '
        'string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def _prov(mock, history=HISTORY, **kw):
    return make_provider(history, base_url=mock.url, api_key=mock.api_key, **kw)


def test_content_prefill_through_invoke_and_revert(enc, mock_server):
    """A stub with content continues the answer: the prompt is the official wo_eos render, the completed message holds
    reasoning + content prefix + continuation, and revert restores the stub."""
    stub = {"role": "assistant", "reasoning_content": "Tests pass now.", "content": "Final report: I ", "_prefill": True}
    p = _prov(mock_server, history=HISTORY + [stub])
    mock_server.script.append("fixed Tree.add.")
    resp = p.invoke()
    off = attach_tools(HISTORY, TOOLS) + [{"role": "assistant", "reasoning_content": "Tests pass now.",
                                           "content": "Final report: I ", "wo_eos": True}]
    assert mock_server.completion_bodies[-1]["prompt"] == enc.encode_messages(off, thinking_mode="thinking")
    assert resp.reasoning == "Tests pass now." and resp.response == "Final report: I fixed Tree.add."
    assert p.calls[-1]["parse_ok"] and p.calls[-1]["render_roundtrip_exact"] is True
    p.revert_last_turn()
    assert p.messages[-1] == stub and len(p.messages) == len(HISTORY) + 1


def test_stub_with_empty_content_is_still_open_reasoning():
    """A stub copied from a finished tool-call turn carries content "": it stays a mid-reasoning cut (no </think>)."""
    p = make_provider(HISTORY + [{"role": "assistant", "reasoning_content": "half", "content": "", "_prefill": True}])
    assert p.render().endswith("<｜Assistant｜><think>half")


def test_stub_with_tool_calls_is_refused():
    p = make_provider(HISTORY + [dict(HISTORY[2], _prefill=True)])
    with pytest.raises(ValueError, match="cannot carry tool_calls"):
        p.render()


class _BosAddingServer(MockVLLM):
    """A server (or tokenizer wrapper) that ignores add_special_tokens=false and prepends BOS."""

    def server_tokenize(self, prompt, add_special_tokens):
        return MockVLLM.server_tokenize(self, prompt, True)


def test_strict_check_fails_on_double_bos_after_recording(tok):
    with _BosAddingServer(tok) as m:
        p = _prov(m)
        m.script.append("r</think>" + CALL)
        with pytest.raises(dsp.FidelityError, match="server_prompt_bos_ok=False"):
            p.invoke()
        assert p.calls[-1]["server_prompt_bos_ok"] is False and p.messages == HISTORY


def test_timeouts_count_against_the_retry_budget(tok, monkeypatch):
    """A request that times out spends its duration from the budget, so a hung server is not retried for hours."""
    import time

    class Hung(MockVLLM):
        def handle(self, method, path, headers, body):
            if path == "/v1/completions":
                time.sleep(0.4)
            return super().handle(method, path, headers, body)

    monkeypatch.setenv("DSV41_RETRY_FOR_S", "1.0")
    with Hung(tok) as m:
        p = _prov(m, timeout=0.3, max_model_len=131072)
        with pytest.raises(dsp.APIConnectionError):
            p.invoke()
        fails = p.calls[-1]["failed_attempts"]
        assert fails and all(f["reason"] == "TimeoutError" for f in fails) and len(fails) <= 4


def test_max_attempts_env_caps_attempts(mock_server, monkeypatch):
    monkeypatch.setenv("DSV41_MAX_ATTEMPTS", "2")
    p = _prov(mock_server, max_model_len=131072)
    mock_server.fail_first += [503] * 5
    with pytest.raises(dsp.APIConnectionError, match="after 2 attempts"):
        p.invoke()


def test_meta_is_never_rendered_and_dump_keeps_it(mock_server):
    """_dsv41 provenance (seeded call / parse failure) rides along in messages.json but never reaches the prompt."""
    p = _prov(mock_server, seed=3, malformed_retries=0)
    mock_server.script.append("r</think>" + CALL)
    p.invoke()
    m = p.messages[-1]
    assert m["_dsv41"]["call_index"] == 0 and m["_dsv41"]["run_id"] == p.run_id
    p.add_tool_result(type("TR", (), {"id": m["tool_calls"][0]["id"], "name": "x", "content": "o"})())
    clean = copy.deepcopy(p.messages)
    clean[-2].pop("_dsv41")
    assert p.render() == make_provider(clean).render() and "_dsv41" in p.dump_history()


def test_kwarg_uds_beats_kwarg_base_url(tmp_path, mock_server):
    p = make_provider(HISTORY, uds=str(tmp_path / "s.sock"), base_url=mock_server.url, api_key="k")
    assert p._get_transport().describe() == f"unix:{tmp_path / 's.sock'}"
