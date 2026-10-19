"""Unit matrix for the shared prompt-context value filter."""

import pytest

from xagent.web.services.prompt_context import build_prompt_context_value

DECLARATIONS = [
    {"name": "account", "max_length": 8},
    {"name": "tier", "max_length": 8},
    {"name": "exact", "max_length": 5},
]


def test_unsigned_keeps_declared_primitive_values() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "tier": 3, "exact": "abcde"},
        DECLARATIONS,
        signed=False,
    ) == {"data": {"account": "acme", "tier": 3, "exact": "abcde"}, "signed": False}


def test_unsigned_drops_undeclared_keys() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "other": "x"}, DECLARATIONS, signed=False
    ) == {"data": {"account": "acme"}, "signed": False}


def test_unsigned_drops_over_length_values() -> None:
    assert build_prompt_context_value(
        {"account": "acme", "exact": "abcdef"}, DECLARATIONS, signed=False
    ) == {"data": {"account": "acme"}, "signed": False}


def test_unsigned_measures_non_string_primitives_by_str_length() -> None:
    assert (
        build_prompt_context_value({"tier": 123456789}, DECLARATIONS, signed=False)
        is None
    )


def test_unsigned_drops_none_and_containers() -> None:
    assert (
        build_prompt_context_value(
            {"account": None, "tier": [1], "exact": {"a": 1}},
            DECLARATIONS,
            signed=False,
        )
        is None
    )


def test_signed_keeps_undeclared_and_ignores_length_limits() -> None:
    values = {"undeclared": "x" * 4000, "none": None, "list": [1]}
    assert build_prompt_context_value(values, [], signed=True) == {
        "data": {"undeclared": "x" * 4000},
        "signed": True,
    }


def test_signed_keeps_only_primitive_values() -> None:
    assert build_prompt_context_value(
        {"a": "x", "b": True, "c": 1.5, "d": None, "e": [1]},
        None,
        signed=True,
    ) == {"data": {"a": "x", "b": True, "c": 1.5}, "signed": True}


def test_empty_and_non_mapping_values_return_none() -> None:
    assert build_prompt_context_value({}, DECLARATIONS, signed=False) is None
    assert build_prompt_context_value(None, DECLARATIONS, signed=False) is None
    assert build_prompt_context_value(["a"], DECLARATIONS, signed=False) is None


def test_malformed_declarations_declare_nothing() -> None:
    values = {"account": "acme"}
    for declarations in (
        None,
        "account",
        [{"name": "account"}],
        [{"name": "account", "max_length": "8"}],
        [{"name": "account", "max_length": 0}],
        [{"name": "account", "max_length": True}],
        [{"name": "account", "max_length": 8.0}],
        [{"max_length": 8}],
        [None],
        [None, {"name": "account", "max_length": 8}],
        [{"name": "account", "max_length": 8}, {"name": "account", "max_length": 9}],
        [{"name": f"f{i}", "max_length": 8} for i in range(32)]
        + [{"name": "account", "max_length": 8}],
    ):
        assert build_prompt_context_value(values, declarations, signed=False) is None, (
            declarations
        )


def test_invalid_declared_name_admits_nothing() -> None:
    assert (
        build_prompt_context_value(
            {"bad name": "a"}, [{"name": "bad name", "max_length": 4}], signed=False
        )
        is None
    )


def test_unsigned_drops_integer_too_large_to_stringify() -> None:
    assert build_prompt_context_value(
        {"big": 10**5000, "account": "acme"},
        [{"name": "big", "max_length": 1024}, {"name": "account", "max_length": 8}],
        signed=False,
    ) == {"data": {"account": "acme"}, "signed": False}


def test_over_total_cap_returns_none() -> None:
    assert build_prompt_context_value({"a": "x" * 4096}, [], signed=True) is None
    assert build_prompt_context_value({"a": "x" * 4000}, [], signed=True) == {
        "data": {"a": "x" * 4000},
        "signed": True,
    }


def test_nan_returns_none() -> None:
    assert build_prompt_context_value({"a": float("nan")}, [], signed=True) is None


@pytest.mark.parametrize("value", ["text", 1, 0, -2.5, True, False])
def test_signed_keeps_each_primitive_kind(value: str | int | float | bool) -> None:
    assert build_prompt_context_value({"a": value}, [], signed=True) == {
        "data": {"a": value},
        "signed": True,
    }
