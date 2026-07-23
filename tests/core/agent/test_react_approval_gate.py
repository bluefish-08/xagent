"""Tool-approval gate in the ReAct segment loop (#767/#768/#769 core).

Drives the gate directly with fakes to pin:
- risk metadata coercion (read_only -> SAFE, unannotated -> EXECUTE);
- policy gating: never never gates; on_dangerous gates EXECUTE; strict adds WRITE;
- pause: a gated call sets waiting_for_user, is NOT executed, stays pending;
- decision precedence: session cache > one-shot > policy default;
- approve / approve_session / deny semantics and args-hash cache specificity;
- resume parsing: approve / approve_session / free-text-as-deny (fail closed);
- checkpoint round-trip of the gate state.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.core.agent.concurrency_harness import (
    FakeRuntime,
    FakeTool,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.tools.adapters.vibe.base import ToolMetadata, ToolRisk


def _make_pattern(policy: str):
    pattern = make_react(
        parallel=False,
        approval_policy=policy,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )
    pattern.task_text = "t"
    return pattern


# --- #767 risk metadata coercion -------------------------------------------


def test_read_only_coerces_to_safe() -> None:
    assert ToolMetadata(name="r", read_only=True).risk is ToolRisk.SAFE


def test_unannotated_defaults_to_execute() -> None:
    assert ToolMetadata(name="u").risk is ToolRisk.EXECUTE


def test_explicit_write_is_kept() -> None:
    assert ToolMetadata(name="w", risk=ToolRisk.WRITE).risk is ToolRisk.WRITE


# --- policy gating ----------------------------------------------------------


async def test_never_policy_runs_dangerous_tool() -> None:
    tools = [FakeTool("danger", risk="execute")]
    pattern = _make_pattern("never")
    pattern.pending_tool_calls = [make_tool_call("danger")]
    context = RecordingContext()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=FakeRuntime()
    )

    assert result is None  # loop drained, tool ran
    assert [r["tool_name"] for r in context.tool_results] == ["danger"]


async def test_on_dangerous_pauses_execute_tool() -> None:
    tools = [FakeTool("danger", risk="execute")]
    pattern = _make_pattern("on_dangerous")
    call = make_tool_call("danger")
    pattern.pending_tool_calls = [call]
    context = RecordingContext()
    runtime = FakeRuntime()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=runtime
    )

    assert result is not None and result["status"] == "waiting_for_user"
    assert result["message_type"] == "approval"
    # Tool did NOT run and the call stays pending for re-entry.
    assert context.tool_results == []
    assert pattern.pending_tool_calls == [call]
    assert pattern.waiting_for_user_request["kind"] == "approval"
    # An approval question with action_cards was emitted to the user.
    sent = runtime.events_of("send_message")
    assert sent and sent[0]["message_type"] == "approval"
    assert sent[0]["metadata"]["interactions"][0]["type"] == "action_cards"


async def test_on_dangerous_allows_safe_tool() -> None:
    tools = [FakeTool("reader", read_only=True)]  # -> SAFE
    pattern = _make_pattern("on_dangerous")
    pattern.pending_tool_calls = [make_tool_call("reader")]
    context = RecordingContext()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=FakeRuntime()
    )

    assert result is None
    assert [r["tool_name"] for r in context.tool_results] == ["reader"]


async def test_strict_pauses_write_but_on_dangerous_does_not() -> None:
    write_tool = FakeTool("writer", risk="write")
    call = make_tool_call("writer")

    lenient = _make_pattern("on_dangerous")
    lenient.pending_tool_calls = [dict(call)]
    ctx = RecordingContext()
    assert (
        await lenient._execute_pending_tool_calls(
            context=ctx, tools=[write_tool], llm=None, runtime=FakeRuntime()
        )
        is None
    )  # WRITE not gated under on_dangerous

    strict = _make_pattern("strict")
    strict.pending_tool_calls = [dict(call)]
    result = await strict._execute_pending_tool_calls(
        context=RecordingContext(), tools=[write_tool], llm=None, runtime=FakeRuntime()
    )
    assert result is not None and result["status"] == "waiting_for_user"


# --- decision application (one-shot / session cache / deny) ------------------


async def test_one_shot_approve_runs_and_is_consumed() -> None:
    tools = [FakeTool("danger", risk="execute")]
    pattern = _make_pattern("on_dangerous")
    call = make_tool_call("danger")
    pattern.pending_tool_calls = [call]
    pattern.pending_approval_decisions[call["id"]] = {"decision": "approve"}
    context = RecordingContext()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=FakeRuntime()
    )

    assert result is None
    assert [r["tool_name"] for r in context.tool_results] == ["danger"]
    # one-shot consumed; not cached for the session.
    assert pattern.pending_approval_decisions == {}
    assert pattern.session_approved == {}


async def test_one_shot_deny_backfills_denied_result() -> None:
    tools = [FakeTool("danger", risk="execute")]
    pattern = _make_pattern("on_dangerous")
    call = make_tool_call("danger")
    pattern.pending_tool_calls = [call]
    pattern.pending_approval_decisions[call["id"]] = {
        "decision": "deny",
        "hint": "too risky",
    }
    context = RecordingContext()

    result = await pattern._execute_pending_tool_calls(
        context=context, tools=tools, llm=None, runtime=FakeRuntime()
    )

    assert result is None
    assert len(context.tool_results) == 1
    denied = context.tool_results[0]["result"]
    assert denied["status"] == "denied_by_user"
    assert "too risky" in denied["message"]
    assert tools[0].calls == []  # tool never executed


async def test_approve_session_caches_exact_args_only() -> None:
    tools = [FakeTool("danger", risk="execute")]
    pattern = _make_pattern("on_dangerous")
    call = make_tool_call("danger", {"cmd": "ls"})
    pattern.pending_tool_calls = [call]
    pattern.pending_approval_decisions[call["id"]] = {"decision": "approve_session"}

    await pattern._execute_pending_tool_calls(
        context=RecordingContext(), tools=tools, llm=None, runtime=FakeRuntime()
    )
    assert len(pattern.session_approved) == 1

    # Same args -> cache hit -> allowed without asking.
    same = make_tool_call("danger", {"cmd": "ls"})
    assert pattern._resolve_approval(same, tools) == ("allow", "")
    # Different args -> not cached -> must ask.
    other = make_tool_call("danger", {"cmd": "rm -rf /"})
    assert pattern._resolve_approval(other, tools) == ("ask", "")


# --- resume decision parsing (fail closed) ----------------------------------


def _ctx_with_user_reply(text: str) -> RecordingContext:
    context = RecordingContext()
    context.messages.append(SimpleNamespace(role="user", content=text, metadata={}))
    return context


def test_record_decision_parses_approve_variants() -> None:
    for text, expected in [
        ("Tool approval: approve", "approve"),
        ("Tool approval: approve_session", "approve_session"),
        ("批准", "approve"),
        ("too dangerous, do not run", "deny"),  # free text -> deny (fail closed)
        ("Tool approval: deny", "deny"),
    ]:
        pattern = _make_pattern("on_dangerous")
        pattern.waiting_for_user_request = {
            "kind": "approval",
            "tool_call_id": "call_x",
        }
        pattern._record_approval_decision(context=_ctx_with_user_reply(text))
        assert pattern.pending_approval_decisions["call_x"]["decision"] == expected, (
            text
        )


def test_free_text_deny_carries_hint() -> None:
    pattern = _make_pattern("on_dangerous")
    pattern.waiting_for_user_request = {"kind": "approval", "tool_call_id": "c"}
    pattern._record_approval_decision(
        context=_ctx_with_user_reply("please avoid touching prod")
    )
    decision = pattern.pending_approval_decisions["c"]
    assert decision["decision"] == "deny"
    assert decision["hint"] == "please avoid touching prod"


# --- checkpoint round-trip --------------------------------------------------


def test_gate_state_round_trips() -> None:
    pattern = _make_pattern("strict")
    pattern.session_approved = {"danger:abc": True}
    pattern.pending_approval_decisions = {"c1": {"decision": "approve", "hint": ""}}
    state = pattern.get_state()

    restored = _make_pattern("never")
    restored.load_state(state)
    assert restored.approval_policy == "strict"
    assert restored.session_approved == {"danger:abc": True}
    assert restored.pending_approval_decisions == {
        "c1": {"decision": "approve", "hint": ""}
    }
