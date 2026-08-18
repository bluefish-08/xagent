"""Cache-reused tool configs re-derive the governing agent's team (#1281).

``WebToolConfig._connector_team_id`` was written once at construction while
the instance is cached with its ``AgentService`` by task, so a governing
agent re-homed mid-session kept resolving connector visibility against the
team captured on the first turn. These tests drive the real cache-hit return
path of ``get_agent_for_task``; the visibility consequence of the refresh
itself is pinned in ``tests/web/tools/test_mcp_team_visibility.py``.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import TaskStatus
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)
from xagent.web.tools.config import WebToolConfig

TASK_ID = 42
OWNER_ID = 1


def _snapshot(team_id: Optional[int], *, with_agent: bool = True) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=TASK_ID,
            user_id=OWNER_ID,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=OWNER_ID, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="governed-agent",
            status=AgentStatus.PUBLISHED,
            instructions=None,
            team_id=team_id,
        )
        if with_agent
        else None,
        agent_config=None,
        excluded_agent_id=7,
    )


def _cached_manager(built_team_id: Optional[int]) -> tuple[Any, Any, Any]:
    """A manager whose cache already holds an agent built for ``built_team_id``."""
    manager = AgentServiceManager()
    # A real config, not a mock: the sync is guarded by hasattr, and a
    # MagicMock would satisfy that guard whether or not the setter exists.
    tool_config = WebToolConfig(
        db=None,
        request=None,
        user_id=OWNER_ID,
        connector_team_id=built_team_id,
    )
    agent = MagicMock()
    agent.tool_config = tool_config
    manager._agents[TASK_ID] = agent
    manager._agent_owner_ids[TASK_ID] = OWNER_ID
    manager._agent_scope_fingerprints[TASK_ID] = None
    return manager, agent, tool_config


async def _reuse(manager: AgentServiceManager, snapshot: Any, **extra: Any) -> Any:
    return await manager.get_agent_for_task(
        task_id=TASK_ID,
        db=None,
        user=None,
        task_setup_snapshot=snapshot,
        task_owner_user_id=OWNER_ID,
        resolved_execution_scope=None,
        **extra,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("built_team_id", "next_team_id"),
    [(None, 101), (101, 202), (101, None)],
)
async def test_reused_config_adopts_the_new_governing_team(
    built_team_id, next_team_id
) -> None:
    manager, agent, tool_config = _cached_manager(built_team_id)

    reused = await _reuse(manager, _snapshot(next_team_id))

    assert reused is agent
    assert tool_config._connector_team_id == next_team_id
    agent.invalidate_tools.assert_called_once()


@pytest.mark.asyncio
async def test_unchanged_team_does_not_rebuild_tools() -> None:
    manager, agent, tool_config = _cached_manager(101)

    await _reuse(manager, _snapshot(101))

    assert tool_config._connector_team_id == 101
    agent.invalidate_tools.assert_not_called()


@pytest.mark.asyncio
async def test_absent_snapshot_leaves_the_team_alone() -> None:
    """Neither read supplied is no read at all -- the sync may not guess."""
    manager, agent, tool_config = _cached_manager(101)

    await _reuse(manager, None)

    assert tool_config._connector_team_id == 101
    agent.invalidate_tools.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("built_team_id", "read_team_id"),
    [(None, 101), (101, 202), (101, None)],
)
async def test_governing_team_projection_alone_refreshes(
    built_team_id, read_team_id
) -> None:
    """The reply paths pass only the team projection -- no snapshot -- so the
    scalar has to be sufficient on its own."""
    manager, agent, tool_config = _cached_manager(built_team_id)

    await _reuse(manager, None, governing_team_id=read_team_id)

    assert tool_config._connector_team_id == read_team_id
    agent.invalidate_tools.assert_called_once()


@pytest.mark.asyncio
async def test_unread_projection_does_not_override_a_snapshot() -> None:
    """Build callers pass a snapshot and no projection; the default sentinel
    must not be mistaken for an authoritative ``None``."""
    manager, agent, tool_config = _cached_manager(None)

    await _reuse(manager, _snapshot(101))

    assert tool_config._connector_team_id == 101


@pytest.mark.asyncio
async def test_negative_agent_result_clears_the_team() -> None:
    """A present snapshot resolving no agent is an authoritative negative, not
    an absent read: fresh construction derives None from it, so a reused config
    must stop resolving the previous team's grants."""
    manager, agent, tool_config = _cached_manager(101)

    await _reuse(manager, _snapshot(None, with_agent=False))

    assert tool_config._connector_team_id is None
    agent.invalidate_tools.assert_called_once()
