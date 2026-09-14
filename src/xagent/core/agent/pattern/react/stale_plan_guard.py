"""Refuse an outside write ordered by a plan the user has since overtaken.

A DAG plan runs to completion once generated, and a reply to a step's own
question is forwarded into that step rather than replanned -- so a step
planned under an earlier message keeps executing under the intent it was
planned with, not the one the user has since expressed. That is how a
publish survived the user picking Cancel on the confirmation that preceded
it.

Enrollment fails CLOSED: every MCP tool except one whose server explicitly
claims ``read_only`` counts, because most connectors omit annotations
entirely and an undeclared tool may well write. This is the
confirmation-style consumer ``MCPWriteHint``'s docstring prescribes, and the
mirror image of the same-turn duplicate guard next door, which fails open
because suppressing an unannotated poll would hand the model stale data.

The declaration is read off ``tool.metadata`` rather than the adapter: the
sandbox wrapper forwards only ``.metadata``, so reading the adapter would
exempt every sandboxed MCP server.
"""

from __future__ import annotations

from typing import Any

# Spelled rather than imported from the MCP adapter: this package is the
# common ReAct contract and must not depend on a concrete tool adapter.
_READ_ONLY_HINT = "read_only"

STALE_PLAN_REFUSED_STATUS = "refused_stale_plan"


def tool_needs_fresh_plan(tool: Any) -> bool:
    """Whether ``tool`` may only run under a plan newer than the last message.

    ``None`` means the tool carries no MCP declaration at all -- an internal
    tool, which this guard leaves alone. Everything else enrolls unless the
    server's own annotations claim read-only, and that claim is taken as a
    routing hint, never as a security fact.
    """
    metadata = getattr(tool, "metadata", None)
    if metadata is None:
        return False
    hint = getattr(metadata, "mcp_write_hint", None)
    if hint is None:
        return False
    return str(hint) != _READ_ONLY_HINT


def build_stale_plan_refusal(tool_name: str) -> dict[str, Any]:
    """Build the model-facing envelope for a refused stale-plan write."""
    return {
        "success": False,
        "status": STALE_PLAN_REFUSED_STATUS,
        "tool_name": tool_name,
        "error": (
            f"{tool_name} was not called. This step was planned before the "
            "user's latest message, and an outside call may not run on an "
            "intent the user has had the chance to change since. Re-read what "
            "the user last said and decide again from there."
        ),
    }
