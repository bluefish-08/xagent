"""Shared filter that turns raw values into a ``prompt_context`` value."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..schemas.agent import PromptContextFieldList
from .file_turn import build_prompt_context_block

_PRIMITIVES = (str, int, float, bool)
_DECLARATIONS: TypeAdapter[Any] = TypeAdapter(PromptContextFieldList)


def _declaration_limits(declarations: Any) -> dict[str, int]:
    try:
        validated = _DECLARATIONS.validate_python(declarations, strict=True)
    except ValidationError:
        return {}
    return {item.name: item.max_length for item in validated}


def build_prompt_context_value(
    values: Any, declarations: Any, *, signed: bool
) -> dict[str, Any] | None:
    """Filter raw values into a prompt_context value, or None if none survive."""
    if not isinstance(values, Mapping):
        return None
    limits = None if signed else _declaration_limits(declarations)
    kept: dict[str, Any] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, _PRIMITIVES):
            continue
        if limits is not None:
            if name not in limits:
                continue
            try:
                length = len(str(value))
            except ValueError:
                continue
            if length > limits[name]:
                continue
        kept[name] = value
    if not kept:
        return None
    result = {"data": kept, "signed": signed}
    return result if build_prompt_context_block(result) is not None else None
