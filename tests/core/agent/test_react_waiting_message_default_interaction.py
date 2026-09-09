"""A suspended ReAct run always publishes at least one answerable field.

xagent#1528 declared an empty ``interactions`` list a legitimate shape and
left "no controls means answer in free text" to the reader. No reader
implements it, so a model that emits ``ask_user_question`` with an empty or
missing ``interactions`` -- or that suspends through ``send_message``, which
has no ``interactions`` parameter at all -- produced a ``waiting_for_user``
message the frontend rendered as ordinary assistant prose, with nothing to
answer it with.

``ReActPattern._send_waiting_message`` is the one place every suspending
path publishes through, and it substitutes a ``text_input`` field whenever
nothing in the list is answerable -- an empty list, but equally a list whose
every entry the render surface would drop or render with nothing to pick.
These cells pin that on all three paths and pin the reverse: a model that
did supply an answerable field gets exactly its own, untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
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
@pytest.mark.parametrize(
    "picker_type", ["select_one", "select_multiple", "action_cards"]
)
async def test_a_picker_whose_options_were_all_blank_is_replaced(
    picker_type: str,
) -> None:
    """The list is non-empty and still unanswerable.

    ``_normalize_ask_user_interactions`` drops blank options but keeps the
    interaction itself, so a picker whose every option was blank arrives here
    as an entry with ``options == []`` -- a control with nothing to select,
    which is the same dead end an empty list is. An emptiness test lets it
    through; the answerability test replaces it.
    """

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": picker_type,
                    "field": "choice",
                    "label": "Choice",
                    "options": [{"label": "  ", "value": ""}],
                }
            ],
        },
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_a_picker_whose_options_are_not_a_list_is_replaced() -> None:
    """``_normalize_ask_user_interactions`` warns about a non-list ``options``
    and then leaves it exactly as the model wrote it. A truthy non-list is
    still nothing the render surface can iterate, so answerability has to test
    the type, not just truthiness."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": "select_one",
                    "field": "choice",
                    "label": "Choice",
                    "options": "auto",
                }
            ],
        },
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interaction", "expected_type"),
    [
        ({"type": "confirm", "field": "ok", "label": "Proceed?"}, "confirm"),
        ({"type": "file_upload", "field": "doc", "label": "Upload"}, "file_upload"),
        ({"type": "number_input", "field": "n", "label": "How many?"}, "number_input"),
        ({"type": "text_input", "field": "note", "label": "Note"}, "text_input"),
        ({"type": "text", "field": "note", "label": "Note"}, "text_input"),
        ({"type": "boolean", "field": "ok", "label": "Proceed?"}, "confirm"),
    ],
    ids=[
        "confirm",
        "file_upload",
        "number_input",
        "text_input",
        "alias_text",
        "alias_boolean",
    ],
)
async def test_a_type_that_needs_no_options_is_left_alone(
    interaction: dict[str, Any],
    expected_type: str,
) -> None:
    """The answerability test must not fire on the types that never render
    options -- replacing one of those would throw away the model's real
    question. The last two are aliases the schema ``enum`` never offers:
    ``_normalize_ask_user_interactions`` maps them onto a canonical name, so
    the answerability check, the transcript builder and the write-side
    validator -- none of which know the aliases -- all see the same seven."""

    _, runtime = await _run(
        "ask_user_question", {"message": "Well?", "interactions": [interaction]}
    )
    published = _published_interactions(runtime)
    # Field name too: a replaced alias would land on the default field, whose
    # type is itself ``text_input`` and would satisfy the type check alone.
    assert [(item["type"], item["field"]) for item in published] == [
        (expected_type, interaction["field"])
    ]


@pytest.mark.asyncio
async def test_one_answerable_field_keeps_the_whole_list() -> None:
    """The substitution is all-or-nothing: one usable control is enough, and
    the unusable one beside it is published as the model wrote it rather than
    being pruned. Pruning is the write side's decision, not this one's."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which city?",
            "interactions": [
                {"type": "select_one", "field": "empty", "label": "E", "options": []},
                {
                    "type": "select_one",
                    "field": "city",
                    "label": "City",
                    "options": [{"label": "Paris", "value": "paris"}],
                },
            ],
        },
    )
    assert [item["field"] for item in _published_interactions(runtime)] == [
        "empty",
        "city",
    ]


@pytest.mark.asyncio
async def test_each_substitution_publishes_its_own_copy() -> None:
    """The default lives in a module-level dict. Publishing it by reference
    would let any reader that edits what it was handed corrupt every later
    waiting message in the process."""

    _, first = await _run("ask_user_question", {"message": "One?"})
    published = _published_interactions(first)
    published[0]["field"] = "mutated"

    _, second = await _run("ask_user_question", {"message": "Two?"})
    assert _published_interactions(second) == [DEFAULT_FIELD]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interaction_type",
    [
        "something_new",
        "",
        "connect_apps",
        ["select_one"],
        {"type": "select_one"},
        None,
        7,
    ],
    ids=["unknown", "blank", "connect_apps", "list", "dict", "none", "int"],
)
async def test_a_type_the_render_surface_cannot_use_is_replaced(
    interaction_type: Any,
) -> None:
    """``normalizeInteractions`` drops anything outside its whitelist and a
    list it empties renders no form, so an unrecognized type is the very dead
    end this substitution exists to prevent. Nothing upstream refuses one: the
    schema ``enum`` is a prompt, the tool arguments are raw ``json.loads``, and
    ``validate_v1_write_payload`` has no production caller. A non-``str`` type
    additionally used to raise ``TypeError`` on the frozenset lookup and take
    the whole run down with it.

    ``connect_apps`` is the one the frontend does keep and still cannot answer:
    a form whose fields are all live widgets renders no Submit button at all
    (``isConnectAppsOnly``, ``clarification-form.tsx``). Replacing it costs
    nothing -- a widget beside any real field keeps the whole list, because
    one answerable entry is enough."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Well?",
            "interactions": [
                {"type": interaction_type, "field": "x", "label": "X"},
            ],
        },
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_a_model_supplied_list_is_published_as_copies() -> None:
    """The published list is stored on the waiting request, the tool-call
    record and the result dict. One policy for both branches: the pass-through
    one must not alias items the caller still holds either, which is what
    ``_pause_for_tool_results`` does -- it keeps the same items on each
    per-tool request entry."""

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-waiting-copies")
    supplied = [
        {
            "type": "select_one",
            "field": "city",
            "label": "City",
            "options": [{"label": "Paris", "value": "paris"}],
        }
    ]
    _, published = await pattern._send_waiting_message(
        runtime=runtime,
        message="Which city?",
        message_type="question",
        interactions=supplied,
        metadata={},
    )
    assert published == supplied
    assert published[0] is not supplied[0]


class _ResumableTool:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(name="gate", description="gate")

    def resume_user_interaction(self, *, interaction_id: str, response: str) -> None:
        pass


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ([dict(DEFAULT_FIELD)], "hello"),
        (
            [{"type": "text_input", "field": "note", "label": "Your response"}],
            "Your response: hello",
        ),
    ],
    ids=["substituted", "model_supplied_same_label"],
)
def test_the_substituted_label_is_stripped_before_the_resume_callback(
    published: list[dict[str, Any]], expected: str
) -> None:
    """``clarification-form.tsx`` submits "<label>: <value>". The substituted
    field's label is engine-invented, so passing the prefixed string on would
    hand a tool's ``resume_user_interaction`` something the user never typed.
    A model-supplied field that happens to share the label keeps its prefix --
    that label is the model's own words and the tool may rely on it."""

    pattern = ReActPattern()
    pattern._queue_tool_interaction_responses(
        waiting_request={
            "kind": "tool_waiting_for_user",
            "interactions": published,
            "requests": [
                {"tool_name": "gate", "tool_call_id": "call_1"},
            ],
        },
        response="Your response: hello",
        tools=[_ResumableTool()],
    )

    assert [
        item["response"] for item in pattern.pending_tool_interaction_responses
    ] == [expected]


@pytest.mark.asyncio
async def test_a_live_widget_survives_beside_an_answerable_field() -> None:
    """Replacing an unanswerable list must not become "drop every widget".
    ``connect_apps`` is unanswerable alone but is a real control the model may
    want shown next to a question, and the substitution is all-or-nothing."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Connect an app, then tell me which one.",
            "interactions": [
                {"type": "connect_apps", "field": "apps", "label": "Connect"},
                {"type": "text_input", "field": "which", "label": "Which one?"},
            ],
        },
    )
    assert [item["type"] for item in _published_interactions(runtime)] == [
        "connect_apps",
        "text_input",
    ]


@pytest.mark.asyncio
async def test_a_normalized_alias_passes_the_write_side_validator() -> None:
    """The alias mapping is what keeps the four readers of a published type
    agreeing. Without it the engine would bless ``{"type": "text"}`` as
    answerable while this validator rejected it as an unsupported type and
    the transcript builder rendered no line for it at all."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Note?",
            "interactions": [{"type": "text", "field": "note", "label": "Note"}],
        },
    )
    published = _published_interactions(runtime)
    assert published[0]["type"] == "text_input"
    assert published[0]["field"] == "note"
    validate_v1_write_payload(
        AskUserQuestionArgs.model_validate(
            {"message": "Note?", "interactions": published}
        )
    )
