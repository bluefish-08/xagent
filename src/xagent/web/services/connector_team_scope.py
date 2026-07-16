"""Optional application hooks for team-owned MCP and Custom API connectors.

Standalone xagent keeps connectors user-owned.  A multi-tenant application can
install these hooks to materialize team access in the existing per-user
junction tables without teaching xagent about the application's team tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ConnectorType = Literal["mcp", "custom_api"]
ConnectorSharedHook = Callable[[Any, int, ConnectorType, int], None]
ConnectorRenamedHook = Callable[[Any, int, ConnectorType, int, str, str], None]


@dataclass(frozen=True)
class ConnectorDeleteDecision:
    team_owned: bool = False
    authorized: bool = False
    delete_definition: bool = False


ConnectorDeletedHook = Callable[[Any, int, ConnectorType, int], ConnectorDeleteDecision]

ConnectorVisibilityHook = Callable[[Any, int], dict[str, set[int]]]
CredentialPolicyHook = Callable[[ConnectorType, str | None], bool]

_connector_shared_hook: ConnectorSharedHook | None = None
_connector_deleted_hook: ConnectorDeletedHook | None = None
_connector_renamed_hook: ConnectorRenamedHook | None = None
_connector_visibility_hook: ConnectorVisibilityHook | None = None
_credential_policy_hook: CredentialPolicyHook | None = None


def set_connector_team_hooks(
    *,
    shared: ConnectorSharedHook | None = None,
    deleted: ConnectorDeletedHook | None = None,
    renamed: ConnectorRenamedHook | None = None,
    visibility: ConnectorVisibilityHook | None = None,
    credential_policy: CredentialPolicyHook | None = None,
) -> None:
    """Install application-owned connector lifecycle hooks."""

    global _connector_shared_hook, _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook, _credential_policy_hook
    _connector_shared_hook = shared
    _connector_deleted_hook = deleted
    _connector_renamed_hook = renamed
    _connector_visibility_hook = visibility
    _credential_policy_hook = credential_policy


def visible_team_connector_ids(db: Any, user_id: int) -> dict[str, set[int]]:
    """Team-shared connector ids visible to user; empty when no hook/standalone."""
    if _connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    return _connector_visibility_hook(db, int(user_id))


def credential_sharing_allowed(connector_type: ConnectorType, transport: str | None) -> bool:
    """Whether a connector may share the owner's credential with the team.
    Single policy point. Standalone (no hook) -> False. Flip to re-enable later."""
    if _credential_policy_hook is None:
        return False
    return bool(_credential_policy_hook(connector_type, transport))


def share_connector_with_team(
    db: Any, user_id: int, connector_type: ConnectorType, connector_id: int
) -> None:
    """Notify the application that a user connected or created a connector."""

    if _connector_shared_hook is not None:
        _connector_shared_hook(db, user_id, connector_type, connector_id)


def delete_team_connector(
    db: Any, user_id: int, connector_type: ConnectorType, connector_id: int
) -> ConnectorDeleteDecision:
    """Remove team ownership before a global delete.

    Returns whether the application recognized the connector as team-owned.
    """

    if _connector_deleted_hook is None:
        return ConnectorDeleteDecision()
    return _connector_deleted_hook(db, user_id, connector_type, connector_id)


def rename_team_connector(
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    old_name: str,
    new_name: str,
) -> None:
    """Keep application-owned connector selectors aligned after a rename."""

    if _connector_renamed_hook is not None and old_name != new_name:
        _connector_renamed_hook(db, user_id, connector_type, connector_id, old_name, new_name)
