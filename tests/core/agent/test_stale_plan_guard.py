from __future__ import annotations

from types import SimpleNamespace

from xagent.core.agent.pattern.react.stale_plan_guard import (
    build_stale_plan_refusal,
    tool_needs_fresh_plan,
)


def _tool(hint: str | None) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(mcp_write_hint=hint))


def test_an_undeclared_mcp_tool_enrolls() -> None:
    # Most connectors ship no annotations at all, so "undeclared" is the
    # common case and has to count as a write - the opposite of the
    # duplicate guard next door, which exempts it.
    assert tool_needs_fresh_plan(_tool("undeclared")) is True


def test_a_destructive_mcp_tool_enrolls() -> None:
    assert tool_needs_fresh_plan(_tool("destructive")) is True


def test_a_server_declared_read_only_tool_is_exempt() -> None:
    assert tool_needs_fresh_plan(_tool("read_only")) is False


def test_a_non_mcp_tool_is_exempt() -> None:
    # None means no MCP declaration at all: an internal tool, which this
    # guard leaves alone.
    assert tool_needs_fresh_plan(_tool(None)) is False
    assert tool_needs_fresh_plan(SimpleNamespace()) is False


def test_a_sandbox_wrapper_forwarding_only_metadata_still_enrolls() -> None:
    wrapper = SimpleNamespace(metadata=SimpleNamespace(mcp_write_hint="undeclared"))

    assert tool_needs_fresh_plan(wrapper) is True


def test_the_refusal_names_the_tool_and_does_not_claim_success() -> None:
    refusal = build_stale_plan_refusal("mcp_LinkedIn_create_post")

    assert refusal["success"] is False
    assert refusal["status"] == "refused_stale_plan"
    assert "mcp_LinkedIn_create_post was not called" in refusal["error"]
