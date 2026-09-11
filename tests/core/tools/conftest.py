"""Shared fixtures for the vibe agent-tool tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_resolvable_default_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the delegation model fallback yield nothing by default.

    ``get_configured_defaults`` ends in ``create_llm_from_env()``, so a test
    asserting the no-valid-model exit would otherwise pass only on a machine
    with no provider key exported -- and would issue a real request on one
    that has it. Autouse so a new test cannot reintroduce that hazard by
    forgetting to ask; a test that wants a resolvable default overrides this
    with its own monkeypatch.
    """
    from xagent.web.services.llm_utils import UserAwareModelStorage

    monkeypatch.setattr(
        UserAwareModelStorage,
        "get_configured_defaults",
        lambda self, *args, **kwargs: (None, None, None, None),
    )
