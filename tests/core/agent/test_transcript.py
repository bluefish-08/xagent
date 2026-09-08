from xagent.core.agent.transcript import (
    build_assistant_transcript_content,
    normalize_transcript_messages,
)
from xagent.core.context_ref import CONTEXT_REFS_KEY, ContextReference
from xagent.core.tools.adapters.vibe.interaction_types import (
    DEFAULT_WAITING_INTERACTION,
)


def test_the_engines_substituted_field_leaves_no_trace_in_the_transcript() -> None:
    """Every ``waiting_for_user`` turn now carries at least one field, and a
    ``send_message`` pause carries the substituted one. Rendering it would
    append a form block to the persisted content of every such turn -- text
    the user reads back on reload and the model replays as context -- saying
    only what the prose already said.
    """

    assert (
        build_assistant_transcript_content(
            "Shall I proceed?", [dict(DEFAULT_WAITING_INTERACTION)]
        )
        == "Shall I proceed?"
    )


def test_a_model_supplied_field_beside_the_substituted_one_still_renders() -> None:
    """Only the substituted field is skipped. Suppressing the whole block
    whenever it appears would lose real questions."""

    content = build_assistant_transcript_content(
        "Pick one",
        [
            dict(DEFAULT_WAITING_INTERACTION),
            {"type": "text_input", "field": "city", "label": "City"},
        ],
    )
    assert "Your response" not in content
    assert "- City: text input" in content


def _image_reference() -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-id",
            "filename": "diagram.png",
            "mime_type": "image/png",
        },
        metadata={"source": "user_upload"},
    )


def test_normalize_transcript_retains_refs_only_message_and_alias() -> None:
    reference = _image_reference()

    normalized = normalize_transcript_messages(
        [
            {
                "role": "user",
                "content": "",
                "context_refs": [reference.durable_dict()],
            }
        ]
    )

    assert normalized == [
        {
            "role": "user",
            "content": "",
            CONTEXT_REFS_KEY: [reference.durable_dict()],
        }
    ]


def test_normalize_transcript_filters_malformed_refs_without_losing_text() -> None:
    normalized = normalize_transcript_messages(
        [
            {
                "role": "user",
                "content": "Keep this text",
                CONTEXT_REFS_KEY: [{"type": "image"}],
            },
            {
                "role": "user",
                "content": "",
                CONTEXT_REFS_KEY: [{"type": "image"}],
            },
        ]
    )

    assert normalized == [{"role": "user", "content": "Keep this text"}]
