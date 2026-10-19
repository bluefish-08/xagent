"""Shared prompt-context value filter for every entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .file_turn import build_prompt_context_block

_PRIMITIVES = (str, int, float, bool)


def _declaration_limits(declarations: Any) -> dict[str, int]:
    if not isinstance(declarations, Sequence) or isinstance(declarations, (str, bytes)):
        return {}
    limits: dict[str, int] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            continue
        name = declaration.get("name")
        max_length = declaration.get("max_length")
        if not isinstance(name, str):
            continue
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            continue
        if 1 <= max_length <= 1024:
            limits[name] = max_length
    return limits


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
            if len(str(value)) > limits[name]:
                continue
        kept[name] = value
    if not kept:
        return None
    result = {"data": kept, "signed": signed}
    return result if build_prompt_context_block(result) is not None else None
