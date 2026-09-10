"""The interaction types the ask_user_question render surface implements,
plus three more things about that surface -- the off-contract aliases it
accepts, one subset of these types, and the free-text field the engine
substitutes (with its translations and the check that recognizes it) -- that
the engine, the transcript builder and the web layer all have to agree on.

Kept in a module of its own rather than beside ``InteractionArg`` in
``ask_user_tool``, which is where the model that carries the field lives:
importing ``ask_user_tool`` pulls in the whole tool-registration chain and
with it 61 ``xagent.web`` modules, and one of this list's three consumers
(``core/agent/pattern/react/react.py``) imports nothing from ``xagent.web``
today. A list of seven strings must not be what changes that. This module
imports only the standard library, so any consumer can take it.

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

from collections.abc import Mapping
from types import MappingProxyType

INTERACTION_TYPES: tuple[str, ...] = (
    "select_one",
    "select_multiple",
    "text_input",
    "file_upload",
    "confirm",
    "number_input",
    "action_cards",
)

# The off-contract type names the frontend's ``normalizeInteractions``
# (``frontend/src/contexts/app-context-chat.tsx``) maps onto the seven above.
# Applied engine-side too, so one alias cannot mean a rendered field to the
# frontend and an unsupported type to the write-side validator, the transcript
# builder, or the answerability check -- each of which knows only the seven.
INTERACTION_TYPE_ALIASES: dict[str, str] = {
    "input": "text_input",
    "text": "text_input",
    "textarea": "text_input",
    "string": "text_input",
    "file": "file_upload",
    "upload": "file_upload",
    "number": "number_input",
    "integer": "number_input",
    "boolean": "confirm",
}

# The subset whose whole purpose is picking from a supplied list, so one of
# them carrying no options is a control nobody can answer. Shared so the
# write-side admissibility rule and the engine's answerability check cannot
# drift apart.
TYPES_REQUIRING_OPTIONS: frozenset[str] = frozenset(
    {"select_one", "select_multiple", "action_cards"}
)

# The free-text field the engine substitutes when a suspending message would
# otherwise carry nothing answerable, in English -- the copy used whenever no
# output language is pinned. Read-only: publishers copy it, and an in-place
# edit here would change every later substitution in the process.
DEFAULT_WAITING_INTERACTION: Mapping[str, object] = MappingProxyType(
    {
        "type": "text_input",
        "field": "response",
        "label": "Your response",
        "placeholder": "Type your answer",
        "multiline": True,
    }
)

_SIMPLIFIED = ("您的回复", "请输入您的回答")
_TRADITIONAL = ("您的回覆", "請輸入您的回答")

# Translations of that label and placeholder, keyed by the canonical language
# labels ``effective_output_language`` returns (``language.py``). Only Chinese:
# English and Chinese are the two locales the product ships UI copy in
# (``frontend/src/i18n/locales``), and Simplified and Traditional are kept apart
# because ``language.py`` treats them as different output languages. Any
# language absent here keeps the English copy above.
_DEFAULT_WAITING_TRANSLATIONS: dict[str, tuple[str, str]] = {
    "Chinese": _SIMPLIFIED,
    "Simplified Chinese": _SIMPLIFIED,
    "Mandarin Chinese": _SIMPLIFIED,
    "Traditional Chinese": _TRADITIONAL,
    "Cantonese": _TRADITIONAL,
}


def default_waiting_interaction(output_language: str = "") -> dict[str, object]:
    """Return a fresh substituted field, localized to a pinned output language."""

    translated = _DEFAULT_WAITING_TRANSLATIONS.get(output_language)
    if translated is None:
        return dict(DEFAULT_WAITING_INTERACTION)
    label, placeholder = translated
    return {**DEFAULT_WAITING_INTERACTION, "label": label, "placeholder": placeholder}


# Compared without ``field``: the substituted field is deduplicated against
# whatever fields are already in the list it joins, so its name is not fixed.
_DEFAULT_WAITING_VARIANTS: tuple[dict[str, object], ...] = tuple(
    {key: value for key, value in variant.items() if key != "field"}
    for variant in (
        dict(DEFAULT_WAITING_INTERACTION),
        *(
            default_waiting_interaction(language)
            for language in _DEFAULT_WAITING_TRANSLATIONS
        ),
    )
)


def is_default_waiting_interaction(interaction: object) -> bool:
    """Whether this published item is the substituted field, not a real one.

    Equality against every localized variant the substitution can publish,
    generated from the same table it publishes from, so the two cannot drift
    and a run resumed under a different pinned language still recognizes the
    field it published earlier. ``field`` is excluded from the comparison: the
    substituted field is deduplicated against the fields already in the list
    it joins, so it can be published as ``response_2``.

    Equality is a heuristic, not proof of origin: a model-supplied free-text
    field that happens to carry this exact copy is indistinguishable from the
    substituted one. Three of the four callers only skip rendering it -- one
    transcript line, one channel bullet, one replayed history row -- and the
    published form is unaffected. The fourth (``_queue_tool_interaction_
    responses``, ``react.py``) strips the ``"<label>: "`` prefix off the
    answer handed to a tool's resume callback, so a false match there costs
    that tool the label it chose.
    """

    if not isinstance(interaction, dict):
        return False
    without_field = {key: value for key, value in interaction.items() if key != "field"}
    return any(without_field == variant for variant in _DEFAULT_WAITING_VARIANTS)
