"""(c) Prefill: a partial chain of thought is appended verbatim after the official generation prompt, the result is a
prefix of the official render of the completed turn, and the prefill protocol (stub, revert, RejectionProvider) works."""

import copy
import json

import pytest

from conftest import TOOLS, attach_tools, call, conversations, make_provider

CONVS = conversations()
PARTIAL = "The assertion says \"b/c sorts before b.txt\". Maybe the sort key is wrong?\n\nI"


@pytest.mark.parametrize("effort", ["low", "high", "max", 42])
@pytest.mark.parametrize("name", list(CONVS))
def test_prefill_render(enc, name, effort):
    history = CONVS[name]
    p = make_provider(history, reasoning_effort=effort)
    p.set_prefill(PARTIAL)
    rendered = p.render()
    official = enc.encode_messages(attach_tools(history, TOOLS), thinking_mode="thinking", reasoning_effort=effort)
    assert rendered == official + PARTIAL
    assert rendered.endswith("<｜Assistant｜><think>" + PARTIAL)
    assert official.endswith("<｜Assistant｜><think>")  # the official generation prompt


@pytest.mark.parametrize("tools", [TOOLS, []], ids=["tools", "no_tools"])
def test_prefill_prompt_is_prefix_of_official_completed_turn(enc, tools):
    """encode(history + [completed turn]) == encode(history) + reasoning + '</think>' + rest + EOS, so sampling after
    a prefill continues exactly the string the official template would show for the finished turn."""
    history = CONVS["basic"] if tools else [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    rest = " will read the sort key first; I'll fix the bug myself."
    done = {"role": "assistant", "content": "", "reasoning_content": PARTIAL + rest,
            "tool_calls": [call("z", "execute_command", {"command": "sed -n 400,430p dulwich/objects.py"})]}
    p = make_provider(history, tools=tools)
    p.set_prefill(PARTIAL)
    prompt = p.render()
    full = enc.encode_messages(attach_tools(history + [done], tools) if tools else history + [done],
                               thinking_mode="thinking")
    assert full.startswith(prompt)
    assert full[len(prompt):].startswith(rest + "</think>\n\n<｜DSML｜ calls>")
    assert full.endswith("</｜DSML｜ calls><｜end▁of▁sentence｜>")


def test_invoke_with_prefill_sends_exact_prompt_and_stores_full_reasoning(enc, mock_server):
    history = CONVS["basic"]
    p = make_provider(history, base_url=mock_server.url, api_key=mock_server.api_key)
    p.set_prefill(PARTIAL)
    expected_prompt = p.render()
    cont = (" will fix it myself.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"execute_command\">\n"
            "<｜DSML｜ parameter name=\"command\" string=\"true\">git diff</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    mock_server.script.append(cont)
    resp = p.invoke()
    assert mock_server.completion_bodies[-1]["prompt"] == expected_prompt
    assert resp.reasoning == PARTIAL + " will fix it myself."
    assert resp.tool_calls[0].name == "execute_command" and json.loads(resp.tool_calls[0].arguments) == {"command": "git diff"}
    assert not p.messages[-1].get("_prefill") and p.messages[-1]["reasoning_content"] == resp.reasoning
    assert p.calls[-1]["prefill_chars"] == len(PARTIAL) and p.calls[-1]["parse_ok"]
    assert p.calls[-1]["render_roundtrip_exact"] is True
    assert p.calls[-1]["prefill_boundary_canonical"] in (True, False)
    # The next prompt (after a tool result) extends the previous one: prefix caching stays valid.
    p.add_tool_result(type("TR", (), {"id": resp.tool_calls[0].id, "name": "execute_command", "content": "diff"})())
    assert p.render().startswith(expected_prompt + " will fix it myself.</think>")
    # revert restores the stub so a retry regenerates from the same partial chain of thought.
    p.messages.pop()
    p.revert_last_turn()
    assert p.messages[-1] == {"role": "assistant", "reasoning_content": PARTIAL, "_prefill": True}
    p.clear_prefill()
    assert p.messages == history


class _FlagJudge:
    """Stand-in SentenceJudge: flags sentences containing a word."""

    def __init__(self, word):
        self.word = word
        self.seen = []

    def judge(self, sentences):
        self.seen.append(list(sentences))
        return [i for i, s in enumerate(sentences) if self.word in s]


def test_rejection_provider_resamples_from_sentence_boundary(mock_server, tmp_path):
    from agent_interp_envs.providers.rejection_provider import RejectionProvider

    history = CONVS["basic"]
    inner = make_provider(history, base_url=mock_server.url, api_key=mock_server.api_key, seed=7)
    rp = RejectionProvider(inner, _FlagJudge("test"), log_path=str(tmp_path / "rej.jsonl"))
    tc = ("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"execute_command\">\n<｜DSML｜ parameter name=\"command\" "
          "string=\"true\">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    mock_server.script += [
        "I see the sort key. This looks like a test fixture bug. So I will skip it.</think>" + tc,
        " I will read objects.py next.</think>" + tc,
    ]
    resp = rp.invoke()
    bodies = mock_server.completion_bodies
    base = make_provider(history).render()
    assert bodies[0]["prompt"] == base
    assert bodies[1]["prompt"] == base + "I see the sort key."  # resumed right before the flagged sentence
    assert resp.reasoning == "I see the sort key. I will read objects.py next."
    assert inner.messages[-1]["reasoning_content"] == resp.reasoning and len(inner.messages) == len(history) + 1
    assert bodies[0]["seed"] != bodies[1]["seed"]  # distinct per-call seeds, so resampling is not a replay
    recs = [json.loads(line) for line in (tmp_path / "rej.jsonl").read_text().splitlines()]
    assert [r["action"] for r in recs] == ["reject_extend", "accept"]


def test_prefill_from_existing_run_cut(enc, mock_server):
    """Arm-style continuation: history of a source run + reasoning cut at a character offset (an external runner's shape)."""
    src = copy.deepcopy(CONVS["long"])
    reasoning = src[-2]["reasoning_content"]
    history, cut = src[:-2], len("Step 5: check file 5.")
    p = make_provider(history + [{"role": "assistant", "reasoning_content": reasoning[:cut], "_prefill": True}],
                      base_url=mock_server.url, api_key=mock_server.api_key)
    mock_server.script.append("\n\nDone.</think>All good.")
    resp = p.invoke()
    assert mock_server.completion_bodies[-1]["prompt"] == enc.encode_messages(
        attach_tools(history, TOOLS), thinking_mode="thinking") + reasoning[:cut]
    assert resp.reasoning == reasoning[:cut] + "\n\nDone." and resp.response == "All good." and resp.tool_calls is None


def test_create_provider_wraps_rejection_like_the_harness(monkeypatch):
    import agent_interp_envs.cot_judge as cj
    from agent_interp_envs.providers.rejection_provider import RejectionProvider

    from conftest import dsp

    class DummyJudge:
        def __init__(self, model, criterion):
            self.model, self.criterion = model, criterion

    monkeypatch.setattr(cj, "SentenceJudge", DummyJudge)
    rp = dsp.create_provider("dsv41", copy.deepcopy(CONVS["basic"]), TOOLS, reasoning_effort="max", verbose=False,
                             rejection={"judge_model": "j", "criterion": "c", "max_generations_per_turn": 5})
    assert isinstance(rp, RejectionProvider) and isinstance(rp.inner, dsp.DeepSeekV41VLLMProvider)
    assert rp.max_generations_per_turn == 5 and rp.inner.budget == 100 and rp.judge.criterion == "c"
    plain = dsp.create_provider("dsv41", copy.deepcopy(CONVS["basic"]), TOOLS, rejection={"enabled": False}, verbose=False)
    assert isinstance(plain, dsp.DeepSeekV41VLLMProvider)


def test_print_history_uses_harness_format(capsys):
    pytest.importorskip("agent_interp_envs", reason="the harness format needs agent_interp_envs (set HARNESS_SRC)")
    make_provider(CONVS["basic"]).print_history()
    out = capsys.readouterr().out
    assert "SYSTEM PROMPT" in out and "TOOL CALL" in out and "ls -la" in out
