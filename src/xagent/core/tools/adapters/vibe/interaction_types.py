"""The interaction types the ask_user_question render surface implements,
plus two more constants about that surface -- one subset of these types, one
literal field -- that the engine, the transcript builder and the web layer
all have to agree on.

Kept in a module of its own rather than beside ``InteractionArg`` in
``ask_user_tool``, which is where the model that carries the field lives:
importing ``ask_user_tool`` pulls in the whole tool-registration chain and
with it 61 ``xagent.web`` modules, and one of this list's three consumers
(``core/agent/pattern/react/react.py``) imports nothing from ``xagent.web``
today. A list of seven strings must not be what changes that. This module
imports nothing, so any consumer can take it.

Ordered, not a set: two of the three consumers render it into text a model
reads -- the ``ask_user_question`` JSON-Schema enum and
``InteractionArg.type``'s own description -- and an unstable order would
change the prompt from run to run.

The three consumers, all of which used to carry their own copy of these
seven names:

* ``InteractionArg.type``'s description (``ask_user_tool.py``)
* the ``ask_user_question`` JSON-Schema enum (``react.py``)
* the write side's admissibility set (``_V1_INTERACTION_TYPES``,
  ``web/services/task_interaction_service.py``)
"""

INTERACTION_TYPES: tuple[str, ...] = (
    "select_one",
    "select_multiple",
    "text_input",
    "file_upload",
    "confirm",
    "number_input",
    "action_cards",
)

# Every ``type`` the frontend's ``normalizeInteractions`` keeps
# (``frontend/src/contexts/app-context-chat.tsx``): the seven above, the
# ``connect_apps`` live widget it renders but the tool schema does not offer,
# and the aliases it maps onto the canonical names. It drops anything else,
# and a list it empties renders no form at all -- so this set, not
# ``INTERACTION_TYPES``, is what the engine's answerability check reads.
RENDERABLE_INTERACTION_TYPES: frozenset[str] = frozenset(INTERACTION_TYPES) | {
    "connect_apps",
    "input",
    "text",
    "textarea",
    "string",
    "file",
    "upload",
    "number",
    "integer",
    "boolean",
}

# The subset whose whole purpose is picking from a supplied list, so one of
# them carrying no options is a control nobody can answer. Shared so the
# write-side admissibility rule and the engine's answerability check cannot
# drift apart.
TYPES_REQUIRING_OPTIONS: frozenset[str] = frozenset(
    {"select_one", "select_multiple", "action_cards"}
)

# The free-text field the engine substitutes when a suspending message would
# otherwise carry nothing answerable. Copy before publishing -- readers keep
# what they are handed, and this dict is shared.
DEFAULT_WAITING_INTERACTION: dict[str, object] = {
    "type": "text_input",
    "field": "response",
    "label": "Your response",
    "placeholder": "Type your answer",
    "multiline": True,
}


def is_default_waiting_interaction(interaction: object) -> bool:
    """Whether this published item is the substituted field, not a real one.

    Equality against the literal above, so it only recognizes the shape the
    engine itself publishes: an interaction that has been round-tripped
    through a model dump or arrives from a client carries other keys and is
    rendered like any other field.
    """

    return interaction == DEFAULT_WAITING_INTERACTION
