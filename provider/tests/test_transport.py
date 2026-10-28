"""(e) Transport against a mocked vLLM: no double BOS (add_special_tokens=false or token ids), sampling/stop params,
HTTP and unix-socket paths, retries, the context cap, per-call records, and the key never leaking."""

import json

import pytest

from conftest import MockVLLM, conversations, dsp, make_provider

HISTORY = conversations()["basic"]
CALL_TEXT = ('ok</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="execute_command">\n<｜DSML｜ parameter '
             'name="command" string="true">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def _provider(mock, **kw):
    return make_provider(HISTORY, base_url=mock.url, api_key=mock.api_key, **kw)


def test_text_mode_sends_add_special_tokens_false_and_no_double_bos(tok, mock_server):
    p = _provider(mock_server)
    mock_server.script.append(CALL_TEXT)
    p.invoke()
    body = mock_server.completion_bodies[-1]
    assert body["add_special_tokens"] is False
    assert body["skip_special_tokens"] is False and body["spaces_between_special_tokens"] is False
    assert body["stop_token_ids"] == [1] and body["model"] == "dsv41" and body["n"] == 1
    assert isinstance(body["prompt"], str) and body["prompt"].startswith("<｜begin▁of▁sentence｜><｜System｜>")
    assert body["prompt"].count("<｜begin▁of▁sentence｜>") == 1
    local = tok.encode(body["prompt"], add_special_tokens=False).ids
    assert local[0] == 0 and local.count(0) == 1
    # What the (BOS-adding) mock server actually tokenized: one BOS. The provider checks this from prompt_token_ids.
    rec = p.calls[-1]
    assert rec["server_prompt_bos_ok"] is True and rec["server_prompt_ids_match_local"] is True
    assert mock_server.server_tokenize(body["prompt"], True)[:2] == [0, 0]  # what add_special_tokens=true would do
    # tokenizer.json itself has no BOS template (post-processor is ByteLevel only), so even add_special_tokens=true
    # adds nothing locally; the explicit false guards against servers/tokenizer wrappers that do add one.
    assert tok.encode(body["prompt"], add_special_tokens=True).ids == local


def test_token_ids_mode_sends_local_ids_with_single_bos(tok, mock_server):
    p = _provider(mock_server, prompt_mode="token_ids")
    expected = tok.encode(p.render(), add_special_tokens=False).ids
    mock_server.script.append(CALL_TEXT)
    p.invoke()
    body = mock_server.completion_bodies[-1]
    assert body["prompt"] == expected and body["prompt"][0] == 0 and body["prompt"].count(0) == 1
    assert p.calls[-1]["server_prompt_bos_ok"] is True


def test_sampling_params_seed_and_records(mock_server, tmp_path):
    log = tmp_path / "calls.jsonl"
    p = _provider(mock_server, seed=123, call_log_path=log)
    mock_server.script += [CALL_TEXT, CALL_TEXT]
    p.invoke()
    p.messages.append({"role": "tool", "tool_call_id": p.messages[-1]["tool_calls"][0]["id"], "content": "x"})
    p.invoke()
    b1, b2 = mock_server.completion_bodies
    assert b1["temperature"] == 1.0 and b1["top_p"] == 1.0  # DeepSeek's recommended defaults when config gives None
    assert b1["seed"] == dsp.derive_seed(123, 0, p.run_id) and b2["seed"] == dsp.derive_seed(123, 1, p.run_id)
    assert b1["seed"] != b2["seed"]
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(recs) == 2 == len(p.calls)
    for r, b in zip(recs, (b1, b2)):
        assert r["seed"] == b["seed"] and r["finish_reason"] == "stop" and r["ended_with_eos"]
        assert r["prompt_tokens"] == r["prompt_tokens_local"] > 500
        assert r["completion_tokens"] > 0 and r["latency_s"] >= 0 and r["reasoning_budget"] == 75
        assert r["max_tokens"] == 131072 - r["prompt_tokens"] - dsp.DEFAULT_CONTEXT_MARGIN
        assert r["seed_run_id"] == p.run_id and r["failed_attempts"] == []
        assert r["completion_token_ids"][-1] == 1 and r["encoding_sha256"] == dsp.OFFICIAL_ENCODING_SHA256
    assert recs[1]["prompt_tokens"] > recs[0]["prompt_tokens"]
    q = _provider(mock_server, temperature=0.6, top_p=0.95)
    mock_server.script.append(CALL_TEXT)
    q.invoke()
    assert mock_server.completion_bodies[-1]["temperature"] == 0.6 and "seed" not in mock_server.completion_bodies[-1]


def test_unix_socket_transport(tok, tmp_path, monkeypatch):
    sock = tmp_path / "llm.sock"
    with MockVLLM(tok, uds_path=sock) as m:
        monkeypatch.setenv("LLM_UDS", str(sock))
        monkeypatch.setenv("DSV41_API_KEY", m.api_key)
        monkeypatch.setenv("DSV41_BASE_URL", "http://unused.invalid:1")  # the socket wins, as in the sandbox
        p = make_provider(HISTORY)
        m.script.append(CALL_TEXT)
        resp = p.invoke()
        assert resp.tool_calls[0].name == "execute_command"
        assert p.calls[-1]["server"] == f"unix:{sock}"
        assert m.requests[-1][2]["Authorization"] == f"Bearer {m.api_key}"


def test_base_url_with_v1_suffix_and_env_config(mock_server, monkeypatch):
    monkeypatch.setenv("DSV41_BASE_URL", mock_server.url + "/v1/")
    monkeypatch.setenv("DSV41_API_KEY", mock_server.api_key)
    monkeypatch.setenv("DSV41_MODEL", "dsv41")
    monkeypatch.setenv("DSV41_MAX_TOKENS", "512")
    monkeypatch.setenv("DSV41_SEED", "5")
    p = dsp.DeepSeekV41VLLMProvider("deepseek-v4.1-flash", list(HISTORY), [], verbose=False)
    mock_server.script.append("r</think>c")
    p.invoke()
    assert mock_server.requests[-1][1] == "/v1/completions"
    body = mock_server.completion_bodies[-1]
    assert body["model"] == "dsv41" and body["max_tokens"] == 512 and body["seed"] == dsp.derive_seed(5, 0, p.run_id)


def test_retries_transient_errors(mock_server):
    p = _provider(mock_server, max_model_len=131072)  # no /v1/models discovery: the failures hit the completion
    mock_server.fail_first += [503, 429]
    mock_server.script.append(CALL_TEXT)
    resp = p.invoke()
    assert resp.tool_calls and p.calls[-1]["http_attempts"] == 3


def test_unauthorized_is_not_retried_and_key_never_leaks(mock_server, capsys):
    p = make_provider(HISTORY, base_url=mock_server.url, api_key="wrong-key-SECRET-xyz", verbose=True)
    with pytest.raises(dsp.ServerError, match="401"):
        p.invoke()
    assert len(mock_server.completion_bodies) == 1
    good = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key, verbose=True)
    mock_server.script.append(CALL_TEXT)
    good.invoke()
    out = capsys.readouterr().out
    blob = out + good.dump_history(include_prompt=True) + json.dumps(good.calls) + repr(good) + repr(p)
    assert mock_server.api_key not in blob and "wrong-key-SECRET-xyz" not in blob
    assert "[provider-usage]" in out


def test_context_cap_and_censor_compatible_error(tok):
    from agent_interp_envs.runs import is_context_limit_error

    n = len(tok.encode(make_provider(HISTORY).render(), add_special_tokens=False).ids)
    margin = dsp.DEFAULT_CONTEXT_MARGIN
    with MockVLLM(tok, max_model_len=n + 100) as m:
        p = make_provider(HISTORY, base_url=m.url, api_key=m.api_key)  # max_model_len discovered via /v1/models
        m.script.append("r</think>c")
        p.invoke()
        assert m.completion_bodies[-1]["max_tokens"] == 100 - margin and p.calls[-1]["max_model_len"] == n + 100
    with MockVLLM(tok, max_model_len=n + margin) as m:  # only the margin left: censored before sending
        p = make_provider(HISTORY, base_url=m.url, api_key=m.api_key)
        with pytest.raises(dsp.ContextLengthError) as ei:
            p.invoke()
        assert is_context_limit_error(ei.value) and not m.completion_bodies
    with MockVLLM(tok, max_model_len=n - 50) as m:  # provider believes there is room; the server's 400 is mapped
        p = make_provider(HISTORY, base_url=m.url, api_key=m.api_key, max_model_len=n + 1000)
        with pytest.raises(dsp.ContextLengthError) as ei:
            p.invoke()
        assert is_context_limit_error(ei.value) and "maximum context length" in str(ei.value)
        assert "error" in p.calls[-1] and p.calls[-1]["prompt_tokens_local"] == n  # diagnostic record of the 400


def test_server_tokenize_fallback_without_local_tokenizer(tok, mock_server):
    p = make_provider(HISTORY, base_url=mock_server.url, api_key=mock_server.api_key, tokenizer_path="none")
    assert p.tokenizer is None
    mock_server.script.append(CALL_TEXT)
    p.invoke()
    tok_req = [b for (_, path, _, b) in mock_server.requests if path == "/tokenize"][-1]
    assert tok_req["add_special_tokens"] is False
    n = len(tok.encode(p.last_prompt, add_special_tokens=False).ids)
    assert p.calls[-1]["prompt_tokens_local"] == n and p.calls[-1]["server_prompt_bos_ok"] is True


def test_dump_history_format(mock_server):
    p = _provider(mock_server)
    mock_server.script.append(CALL_TEXT)
    p.invoke()
    assert json.loads(p.dump_history()) == p.messages  # unchanged checkpoint format
    p.add_tool_result(type("TR", (), {"id": "x", "name": "execute_command", "content": "out"})())
    dbg = json.loads(p.dump_history(include_prompt=True))
    assert dbg["rendered_prompt"] == p.render() and dbg["last_prompt"] == p.last_prompt
    assert dbg["encoding"]["sha256"] == dsp.OFFICIAL_ENCODING_SHA256 and len(dbg["calls"]) == 1


def test_missing_server_config_is_a_clear_error():
    p = make_provider(HISTORY)
    with pytest.raises(ValueError, match="DSV41_BASE_URL"):
        p.invoke()


def test_engine_xargs_and_cache_salt_passthrough(mock_server, monkeypatch):
    p = _provider(mock_server)  # default: neither field on the wire
    mock_server.script.append(CALL_TEXT)
    p.invoke()
    assert "vllm_xargs" not in mock_server.completion_bodies[-1] and "cache_salt" not in mock_server.completion_bodies[-1]
    assert p.calls[-1]["vllm_xargs"] is None and p.calls[-1]["cache_salt"] is None
    xargs = {"arm": 3, "edit_from": 120, "edit_to": 2**31 - 1}
    monkeypatch.setenv("DSV41_VLLM_XARGS", json.dumps(xargs))
    monkeypatch.setenv("DSV41_CACHE_SALT", "0123456789abcdef01234567")
    q = _provider(mock_server)
    mock_server.script += [CALL_TEXT, CALL_TEXT]
    q.invoke()
    q.messages.append({"role": "tool", "tool_call_id": q.messages[-1]["tool_calls"][0]["id"], "content": "x"})
    q.invoke()  # every turn of the episode carries the same arm and salt
    for body in mock_server.completion_bodies[-2:]:
        assert body["vllm_xargs"] == xargs and body["cache_salt"] == "0123456789abcdef01234567"
    assert q.calls[-1]["vllm_xargs"] == xargs and q.calls[-1]["cache_salt"] == "0123456789abcdef01234567"
    for name, bad in [("DSV41_VLLM_XARGS", "[1, 2]"), ("DSV41_VLLM_XARGS", '{"arm": true}'),
                      ("DSV41_VLLM_XARGS", '{"arm": {"k": 1}}'), ("DSV41_VLLM_XARGS", "{arm: 1}"),
                      ("DSV41_CACHE_SALT", "a/b"), ("DSV41_CACHE_SALT", "x" * 129)]:
        with monkeypatch.context() as m:
            m.setenv(name, bad)
            with pytest.raises(ValueError, match=name):
                _provider(mock_server)
