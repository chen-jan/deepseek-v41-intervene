"""(b) The provider's rendered prompt equals the official encode_messages output byte for byte, for several synthetic
agent conversations and efforts low/high/max plus integers."""

import copy

import pytest

from conftest import TOOLS, attach_tools, conversations, dsp, make_provider

EFFORTS = ["low", "high", "max", 1, 37, 62, 100, None]
BUDGET = {"low": 50, "high": 75, "max": 100, None: 75}
CONVS = conversations()


def _effort_line(budget):
    return (f"<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the "
            "reasoning)\n\n")


@pytest.mark.parametrize("effort", EFFORTS, ids=str)
@pytest.mark.parametrize("name", list(CONVS))
def test_render_equals_official(enc, name, effort):
    messages = CONVS[name]
    p = make_provider(messages, reasoning_effort=effort)
    official = enc.encode_messages(attach_tools(messages, TOOLS), thinking_mode="thinking", reasoning_effort=effort)
    rendered = p.render()
    assert rendered == official
    budget = BUDGET.get(effort, effort)
    assert rendered.startswith(enc.bos_token + _effort_line(budget))
    assert rendered.count(enc.bos_token) == 1
    assert rendered.endswith("<｜Assistant｜><think>")
    # V4.1 markers: spaced DSML, tools section present, no V4 tags anywhere.
    assert '<｜DSML｜ invoke name="execute_command">' in rendered
    assert "<｜DSML｜tool_calls>" not in rendered and "<｜DSML｜invoke" not in rendered
    assert "## Tools" in rendered
    # Thinking is kept for every earlier turn when tools are present (official rule).
    for m in messages:
        if m["role"] == "assistant" and m.get("reasoning_content"):
            assert m["reasoning_content"] + "</think>" in rendered


@pytest.mark.parametrize("name", list(CONVS))
def test_rendered_prompt_tokenizes_with_single_bos(tok, name):
    ids = tok.encode(make_provider(CONVS[name]).render(), add_special_tokens=False).ids
    assert ids[0] == 0 and ids.count(0) == 1
    assert tok.id_to_token(ids[-1]) == "<think>" and tok.id_to_token(ids[-2]) == "<｜Assistant｜>"


def test_no_tools_conversation_matches_official_drop_thinking(enc):
    """Without tools the official default drops reasoning before the last user turn; the provider follows it."""
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "r1"},
        {"role": "user", "content": "Q2"},
    ]
    p = make_provider(messages, tools=[], reasoning_effort="max")
    official = enc.encode_messages(copy.deepcopy(messages), thinking_mode="thinking", reasoning_effort="max")
    assert p.render() == official
    assert "r1" not in official and "<｜Assistant｜></think>A1" in official


def test_mapping_ignores_private_keys_and_accepts_reasoning_alias(enc):
    messages = copy.deepcopy(CONVS["basic"])
    variant = copy.deepcopy(messages)
    variant[2]["reasoning"] = variant[2].pop("reasoning_content")  # vLLM / OpenRouter spelling
    variant[2]["reasoning_details"] = [{"text": "ignored"}]
    variant[4]["_debug"] = {"x": 1}
    assert make_provider(variant).render() == make_provider(messages).render()


def test_rejects_efforts_the_official_encoding_does_not_define():
    for bad in ("medium", "xhigh", "none", 0, 101, 62.0, True):
        with pytest.raises(ValueError):
            make_provider(CONVS["basic"], reasoning_effort=bad)
    assert make_provider(CONVS["basic"], reasoning_effort="62").budget == 62  # numeric strings from CLI overrides


def test_env_effort_used_when_harness_passes_none(monkeypatch):
    monkeypatch.setenv("DSV41_REASONING_EFFORT", "low")
    assert make_provider(CONVS["basic"]).render().startswith(dsp.load_official_encoding().bos_token + _effort_line(50))


def test_render_refuses_conversation_ending_with_assistant():
    msgs = CONVS["basic"][:3]  # ends with a completed assistant turn: the official render has no generation prompt
    with pytest.raises(ValueError, match="must end with a user or tool turn"):
        make_provider(msgs).render()
