"""The fallback sites actually reach default_image_abilities.

test_default_abilities.py covers the pure function; without these, deleting any
`or default_image_abilities(...)` wiring leaves the suite green.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.model.image.adapter import get_image_model_instance


def _db_model(**overrides: Any) -> SimpleNamespace:
    row = {
        "model_provider": "dashscope",
        "model_name": "qwen-image-edit",
        "api_key": "test-key",
        "base_url": None,
        "abilities": None,
        "timeout": 300.0,
        "max_retries": 3,
    }
    row.update(overrides)
    return SimpleNamespace(**row)


def test_adapter_infers_abilities_for_a_null_row() -> None:
    model = get_image_model_instance(_db_model())
    assert model.abilities == ["generate", "edit"]


def test_adapter_leaves_a_generate_only_name_alone() -> None:
    model = get_image_model_instance(_db_model(model_name="wanx-v1"))
    assert model.abilities == ["generate"]


def test_adapter_never_overrides_a_declared_list() -> None:
    model = get_image_model_instance(
        _db_model(model_name="qwen-image-edit", abilities=["generate"])
    )
    assert model.abilities == ["generate"]


def _service_abilities(provider: str, model_name: str) -> list[str]:
    from xagent.web.services import model_service

    db_model = SimpleNamespace(
        id=1,
        model_id="img-1",
        category="image",
        is_active=True,
        model_provider=provider,
        model_name=model_name,
        api_key="test-key",
        base_url="https://example.invalid",
        abilities=None,
        description=None,
    )

    class _Query:
        def filter(self, *_args: Any) -> "_Query":
            return self

        def all(self) -> list[Any]:
            return [db_model]

    class _Session:
        def query(self, *_args: Any) -> _Query:
            return _Query()

    models = model_service.get_image_models(_Session())
    return [m.abilities for m in models.values()][0]


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [("qwen-image-edit", ["generate", "edit"]), ("wanx-v1", ["generate"])],
)
def test_model_service_infers_abilities_for_a_null_row(
    model_name: str, expected: list[str]
) -> None:
    assert _service_abilities("dashscope", model_name) == expected


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "gpt-image-1"),
        ("xinference", "sd-3.5"),
        ("dashscope", "qwen-image-edit"),
        ("dashscope", "wanx-v1"),
    ],
)
def test_both_paths_agree_on_a_null_row(provider: str, model_name: str) -> None:
    """A row disagreeing across the two paths is the bug this helper exists to stop.

    An openai row read one way as generate-only and the other way as editable makes
    get_default_image_edit_model hand back a model _get_edit_model then rejects,
    silently swapping in a different one.
    """
    adapter_model = get_image_model_instance(
        _db_model(model_provider=provider, model_name=model_name)
    )
    assert adapter_model.abilities == _service_abilities(provider, model_name)
