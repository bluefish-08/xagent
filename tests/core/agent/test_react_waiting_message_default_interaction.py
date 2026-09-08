"""A suspended ReAct run always publishes at least one answerable field.

xagent#1528 declared an empty ``interactions`` list a legitimate shape and
left "no controls means answer in free text" to the reader. No reader
implements it, so a model that emits ``ask_user_question`` with an empty or
missing ``interactions`` -- or that suspends through ``send_message``, which
has no ``interactions`` parameter at all -- produced a ``waiting_for_user``
message the frontend rendered as ordinary assistant prose, with nothing to
answer it with.

``ReActPattern._send_waiting_message`` is the one place every suspending
path publishes through, and it substitutes a ``text_input`` field when the
list is empty. These cells pin that on all three paths and pin the reverse:
a model that did supply interactions gets exactly its own, untouched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    async def chat(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)


def _control_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


async def _run(name: str, arguments: dict[str, Any]) -> tuple[Any, PatternRuntime]:
    llm = FakeLLM(responses=[_control_call(name, arguments)])
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-waiting-default")
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)
    assert result["status"] == "waiting_for_user"
    return result, runtime


def _published_interactions(runtime: PatternRuntime) -> list[dict[str, Any]]:
    waiting = [
        payload
        for payload in runtime.outbound_messages
        if payload.get("expect_response")
    ]
    assert len(waiting) == 1
    return waiting[0]["metadata"]["interactions"]


DEFAULT_FIELD = {
    "type": "text_input",
    "field": "response",
    "label": "Your response",
    "placeholder": "Type your answer",
    "multiline": True,
}


@pytest.mark.asyncio
async def test_ask_user_question_with_an_empty_interactions_list() -> None:
    result, runtime = await _run(
        "ask_user_question", {"message": "Which one?", "interactions": []}
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert result["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_ask_user_question_with_the_interactions_key_omitted() -> None:
    """The shape Kimi K2.5 was observed producing: ``interactions`` is in the
    schema's ``required`` list and the model omits it anyway."""

    result, runtime = await _run("ask_user_question", {"message": "Which one?"})
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert result["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interactions",
    ["pick one", {"type": "text_input", "field": "city"}, 7],
    ids=["string", "dict", "int"],
)
async def test_ask_user_question_with_a_non_list_interactions(
    interactions: Any,
) -> None:
    """``_normalize_ask_user_interactions`` already flattens every non-list to
    ``[]``; the default field is what keeps that from reaching the user as an
    unanswerable message."""

    _, runtime = await _run(
        "ask_user_question", {"message": "Which one?", "interactions": interactions}
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_that_expects_a_response() -> None:
    """``send_message`` has no ``interactions`` parameter, so its suspending
    branch had no ``interactions`` metadata key at all before this guard."""

    result, runtime = await _run(
        "send_message",
        {
            "message": "Here is my plan. Shall I proceed?",
            "message_type": "question",
            "expect_response": True,
        },
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert result["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_pause_carries_the_field_into_the_structured_row() -> None:
    """A task with ``interaction_protocol_version`` set reads its controls off
    the interaction row, which is built from ``clarification_draft`` -- not off
    the chat message. Publishing the field only in the outbound metadata would
    leave that surface empty, so the waiting request has to carry it too.

    The draft must still attribute itself to ``send_message``: the field is
    substituted by the engine, it does not turn the call into an
    ``ask_user_question``.
    """

    result, _ = await _run(
        "send_message",
        {
            "message": "Here is my plan. Shall I proceed?",
            "message_type": "question",
            "expect_response": True,
        },
    )
    draft = result["clarification_draft"]
    assert draft is not None
    assert draft.source == "send_message"
    assert list(draft.interactions) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_a_hidden_waiting_message_keeps_its_visible_flag() -> None:
    """``visible`` is passed through, not overridden: the fallback adds a
    field, it does not decide whether the message is shown."""

    _, runtime = await _run(
        "send_message",
        {
            "message": "Quiet question",
            "expect_response": True,
            "visible": False,
        },
    )
    waiting = [p for p in runtime.outbound_messages if p.get("expect_response")]
    assert len(waiting) == 1
    assert waiting[0]["visible"] is False
    assert waiting[0]["metadata"]["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_tool_requested_pause_with_no_interactions() -> None:
    """The third suspending path: a tool that reports ``waiting_for_user``
    without offering any interaction of its own."""

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tool-wait")
    context = ExecutionContext()
    context.add_user_message("Ask")

    outcome = await pattern._pause_for_tool_results(
        waiting_pairs=[
            ({"id": "call_a", "name": "tool_a"}, {"message": "Need a value"})
        ],
        context=context,
        runtime=runtime,
    )

    assert outcome["status"] == "waiting_for_user"
    assert outcome["interactions"] == [DEFAULT_FIELD]
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert pattern.waiting_for_user_request is not None
    assert pattern.waiting_for_user_request["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_without_expect_response_stays_field_free() -> None:
    """The fallback must not leak onto messages that do not suspend the run."""

    llm = FakeLLM(
        responses=[
            _control_call(
                "send_message",
                {"message": "Working on it.", "message_type": "progress"},
            ),
            _control_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Done.",
                    "outcome": "completed",
                },
            ),
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    runtime = PatternRuntime(execution_id="exec-no-wait")
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert runtime.outbound_messages
    for payload in runtime.outbound_messages:
        assert not payload.get("expect_response")
        assert "interactions" not in payload["metadata"]


@pytest.mark.asyncio
async def test_model_supplied_interactions_are_not_augmented() -> None:
    """The reverse assertion: a well-formed question keeps exactly its own
    fields, with no default appended."""

    supplied = [
        {
            "type": "select_one",
            "field": "city",
            "label": "City",
            "options": [{"label": "Paris", "value": "paris"}],
        }
    ]
    result, runtime = await _run(
        "ask_user_question", {"message": "Which city?", "interactions": supplied}
    )
    published = _published_interactions(runtime)
    assert [item["field"] for item in published] == ["city"]
    assert published[0]["type"] == "select_one"
    assert result["interactions"] == published


@pytest.mark.asyncio
async def test_default_field_passes_the_write_side_validator() -> None:
    """The injected field has to survive the same admissibility rules the
    model-supplied ones do, or the substitution buys nothing."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    _, runtime = await _run("ask_user_question", {"message": "Which one?"})
    parsed = AskUserQuestionArgs.model_validate(
        {"message": "Which one?", "interactions": _published_interactions(runtime)}
    )
    validate_v1_write_payload(parsed)
