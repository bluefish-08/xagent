"""Tests for default_image_abilities."""

import pytest

from xagent.core.model.image.base import default_image_abilities

GENERATE = ["generate"]
BOTH = ["generate", "edit"]


@pytest.mark.parametrize(
    ("provider", "model_name", "fallback", "expected"),
    [
        ("dashscope", "qwen-image-edit", GENERATE, BOTH),
        ("dashscope", "wanx-v1", GENERATE, GENERATE),
        ("gemini", "gemini-3-pro-image-preview", GENERATE, BOTH),
        ("gemini", "gemini-2.5-flash-image", GENERATE, GENERATE),
        ("  DashScope  ", "qwen-image-edit", GENERATE, BOTH),
        # Providers outside the name-inferred set keep their call site's default,
        # including one whose name would otherwise match.
        ("openai", "gpt-image-1", BOTH, BOTH),
        ("xinference", "my-image-edit", BOTH, BOTH),
        ("xinference", "my-image-edit", GENERATE, GENERATE),
        ("unknown-provider", "something-edit", GENERATE, GENERATE),
    ],
)
def test_default_image_abilities(
    provider: str, model_name: str, fallback: list[str], expected: list[str]
) -> None:
    assert default_image_abilities(provider, model_name, fallback) == expected
