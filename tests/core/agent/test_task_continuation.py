"""
Tests for task continuation functionality in AgentService.
"""

import pytest

from tests.utils.mock_react_llm import MockReactLLM
from xagent.core.agent.service import AgentService


class TestTaskContinuation:
    """Test task continuation functionality."""

    @pytest.mark.asyncio
    async def test_task_continuation_not_implemented_for_nonexistent_agent(self):
        """Test that task continuation returns error for non-existent agents."""
        # Create agent service with ReAct pattern (no DAG)
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=False
        )

        # Try to continue a task that doesn't exist
        result = await service.execute_task(
            "additional task", task_id="nonexistent_task_id"
        )

        # Should return error result with NotImplementedError details
        assert result["success"] is False
        assert "not supported" in result["error"]
        assert "not the original executor" in result["error"]

    @pytest.mark.asyncio
    async def test_task_continuation_with_dag_pattern(self):
        """Test task continuation with DAG pattern."""
        # Create agent service with DAG pattern
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=True
        )

        # Execute initial task
        initial_result = await service.execute_task("initial task")
        assert initial_result["success"] is True

        # Get the task_id from the initial execution
        task_id = initial_result.get("metadata", {}).get("task_id")
        if not task_id:
            # If no task_id in metadata, use the agent's internal tracking
            task_id = getattr(service, "_current_task_id", None)

        assert task_id is not None, "Task ID should be set after initial execution"

        # Continue the task - this will fall back to normal execution since no current plan exists
        continuation_result = await service.execute_task(
            "additional task", task_id=task_id
        )

        # Should execute successfully (fallback to normal execution)
        assert continuation_result["success"] is True
        # Task ID should be preserved
        assert service._current_task_id == task_id

    @pytest.mark.asyncio
    async def test_task_continuation_preserves_task_id_on_success(self):
        """Test that task_id is preserved when execution succeeds."""
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=True
        )

        # First execute a task normally to set _current_task_id
        initial_result = await service.execute_task("initial task")
        assert initial_result["success"] is True

        # Now use the generated task_id for continuation
        task_id = service._current_task_id
        result = await service.execute_task("test task", task_id=task_id)

        # Execution should succeed and task_id should be preserved
        assert result["success"] is True
        assert service._current_task_id == task_id

    @pytest.mark.asyncio
    async def test_normal_execution_without_task_id(self):
        """Test normal execution without task continuation."""
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=False
        )

        # Execute task without task_id
        result = await service.execute_task("test task")

        # Should succeed
        assert result["success"] is True
        assert "output" in result

        # task_id should be generated automatically to enable future continuation
        assert service._current_task_id is not None
        assert service._current_task_id.startswith("task_")

    def test_use_dag_pattern_false_creates_react_pattern(self):
        """Test that use_dag_pattern=False creates ReAct pattern instead of DAG."""
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=False
        )

        # Should have ReAct pattern, not DAG pattern
        assert len(service.patterns) == 1
        pattern = service.patterns[0]
        assert "ReAct" in pattern.__class__.__name__
        assert (
            not hasattr(service, "get_dag_pattern") or service.get_dag_pattern() is None
        )

    def test_use_dag_pattern_true_creates_dag_pattern(self):
        """Test that use_dag_pattern=True creates DAG pattern."""
        llm = MockReactLLM()
        service = AgentService(
            name="test_agent", id="test_agent", llm=llm, use_dag_pattern=True
        )

        # Should have DAG pattern
        assert len(service.patterns) == 1
        dag_pattern = service.get_dag_pattern()
        assert dag_pattern is not None
        assert "DAG" in dag_pattern.__class__.__name__
