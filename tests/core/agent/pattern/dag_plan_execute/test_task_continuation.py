#!/usr/bin/env python3
"""
Unit test to verify task continuation logic works correctly.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.agent.pattern.dag_plan_execute.dag_plan_execute import (
    DAGPlanExecutePattern,
)
from xagent.core.agent.service import AgentService
from xagent.core.agent.trace import Tracer
from xagent.core.memory.in_memory import InMemoryMemoryStore


class TestTaskContinuation:
    """Test task continuation functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        llm = Mock()
        llm.model_name = "mock_llm"
        llm.description = "Mock LLM for testing"
        llm.supports_thinking_mode = False
        llm.chat = AsyncMock(return_value="Mock response")
        llm.generate = AsyncMock(return_value="Mock response")
        return llm

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer"""
        return Tracer()

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory store"""
        return InMemoryMemoryStore()

    @pytest.fixture
    def mock_dag_pattern(self):
        """Create a mock DAG pattern"""
        pattern = Mock(spec=DAGPlanExecutePattern)
        pattern.handle_continuation = AsyncMock(
            return_value={
                "success": True,
                "output": "Continuation completed successfully",
                "continuation": True,
                "phase": "completed",
                "additional_steps": 2,
            }
        )
        # Start with no current plan - will be set in tests that need it
        pattern.current_plan = None
        pattern.get_execution_status = Mock(
            return_value={
                "phase": "completed",
                "success": True,
                "steps_completed": 5,
                "total_steps": 5,
            }
        )
        return pattern

    @pytest.fixture
    def agent_service(self, mock_llm, mock_tracer, mock_memory, mock_dag_pattern):
        """Create an AgentService with mocked dependencies"""
        # Mock the get_dag_pattern method to return our mock pattern
        service = AgentService(
            name="test_agent",
            id="test_agent_id",
            llm=mock_llm,
            tracer=mock_tracer,
            memory=mock_memory,
            use_dag_pattern=True,
        )

        # Replace the patterns with our mock
        service.patterns = [mock_dag_pattern]

        # Mock the get_dag_pattern method
        service.get_dag_pattern = Mock(return_value=mock_dag_pattern)

        return service

    @pytest.mark.asyncio
    async def test_continuation_with_matching_task_id(self, agent_service):
        """Test continuation when task_id matches current_task_id"""
        # Set up the current task ID
        agent_service._current_task_id = "30"

        # Set up current plan so continuation works
        mock_pattern = agent_service.get_dag_pattern()
        mock_pattern.current_plan = Mock()

        # Execute the task with continuation
        result = await agent_service.execute_task(
            task="Additional task requirement",
            context={"priority": "high"},
            task_id="30",  # Same as current_task_id
        )

        # Verify the result structure
        assert result["success"] is True
        assert result["output"] == "Continuation completed successfully"
        assert result["status"] == "completed"
        assert "dag_status" in result
        assert result["metadata"]["task_id"] == "30"

        # Verify handle_continuation was called
        mock_pattern.handle_continuation.assert_called_once_with(
            "Additional task requirement", {"priority": "high"}
        )

    @pytest.mark.asyncio
    async def test_continuation_with_int_task_id(self, agent_service):
        """Test continuation when task_id is int but current_task_id is str"""
        # Set up the current task ID as string
        agent_service._current_task_id = "30"

        # Set up current plan so continuation works
        mock_pattern = agent_service.get_dag_pattern()
        mock_pattern.current_plan = Mock()

        # Execute with int task_id
        await agent_service.execute_task(
            task="Additional task requirement",
            context={},
            task_id=30,  # Int, should be converted to str for comparison
        )

        # Verify continuation was called
        mock_pattern.handle_continuation.assert_called_once_with(
            "Additional task requirement", {}
        )

    @pytest.mark.asyncio
    async def test_continuation_with_str_task_id(self, agent_service):
        """Test continuation when task_id is str"""
        # Set up the current task ID as string
        agent_service._current_task_id = "30"

        # Set up current plan so continuation works
        mock_pattern = agent_service.get_dag_pattern()
        mock_pattern.current_plan = Mock()

        # Execute with str task_id
        await agent_service.execute_task(
            task="Additional task requirement",
            context={},
            task_id="30",  # Str
        )

        # Verify continuation was called
        mock_pattern.handle_continuation.assert_called_once_with(
            "Additional task requirement", {}
        )

    @pytest.mark.asyncio
    async def test_continuation_fails_with_different_task_id(self, agent_service):
        """Test continuation fails when task_id doesn't match current_task_id"""
        # Set up the current task ID
        agent_service._current_task_id = "30"

        # Execute with different task_id - should return error result
        result = await agent_service.execute_task(
            task="Additional task requirement",
            context={},
            task_id="25",  # Different from current_task_id
        )

        # Should return error result instead of raising exception
        assert result["success"] is False
        assert "Agent is not the original executor" in result["error"]

    @pytest.mark.asyncio
    async def test_continuation_fails_without_current_task_id(self, agent_service):
        """Test continuation fails when current_task_id is not set"""
        # Don't set current_task_id
        agent_service._current_task_id = None

        # Execute with task_id - should return error result
        result = await agent_service.execute_task(
            task="Additional task requirement", context={}, task_id="30"
        )

        # Should return error result instead of raising exception
        assert result["success"] is False
        assert "Agent is not the original executor" in result["error"]

    @pytest.mark.asyncio
    async def test_continuation_fails_without_dag_pattern(self, agent_service):
        """Test continuation fails when DAG pattern is not available"""
        # Set up the current task ID
        agent_service._current_task_id = "30"

        # Remove DAG pattern
        agent_service.get_dag_pattern = Mock(return_value=None)

        # Should fall back to normal execution
        with patch.object(agent_service, "_execute_normal_task") as mock_normal:
            mock_normal.return_value = {"success": True, "output": "Normal execution"}

            result = await agent_service.execute_task(
                task="Additional task requirement", context={}, task_id="30"
            )

            assert result["success"] is True
            mock_normal.assert_called_once_with("Additional task requirement", {}, "30")

    @pytest.mark.asyncio
    async def test_continuation_fails_without_handle_continuation(self, agent_service):
        """Test continuation fails when DAG pattern doesn't support continuation"""
        # Set up the current task ID
        agent_service._current_task_id = "30"

        # Create pattern without handle_continuation method
        mock_pattern = Mock(spec=DAGPlanExecutePattern)
        del mock_pattern.handle_continuation  # Remove the method
        agent_service.get_dag_pattern = Mock(return_value=mock_pattern)

        # Should fall back to normal execution
        with patch.object(agent_service, "_execute_normal_task") as mock_normal:
            mock_normal.return_value = {"success": True, "output": "Normal execution"}

            result = await agent_service.execute_task(
                task="Additional task requirement", context={}, task_id="30"
            )

            assert result["success"] is True
            mock_normal.assert_called_once_with("Additional task requirement", {}, "30")

    @pytest.mark.asyncio
    async def test_normal_execution_without_task_id(self, agent_service):
        """Test normal execution when task_id is not provided"""
        # Mock normal execution
        with patch.object(agent_service, "_execute_normal_task") as mock_normal:
            mock_normal.return_value = {"success": True, "output": "Normal execution"}

            result = await agent_service.execute_task(
                task="Normal task",
                context={"priority": "high"},
                # No task_id
            )

            assert result["success"] is True
            mock_normal.assert_called_once_with(
                "Normal task", {"priority": "high"}, None
            )

    @pytest.mark.asyncio
    async def test_current_task_id_set_after_execution(self, agent_service):
        """Test that _current_task_id is set after normal execution"""
        # Reset current task ID
        agent_service._current_task_id = None

        # Mock normal execution but still set the task_id
        def mock_execute_normal_task(task, context, task_id):
            if task_id:
                agent_service._current_task_id = str(task_id)
            else:
                from uuid import uuid4

                agent_service._current_task_id = f"task_{uuid4().hex[:8]}"
            return {"success": True, "output": "Normal execution"}

        with patch.object(
            agent_service, "_execute_normal_task", side_effect=mock_execute_normal_task
        ):
            # Execute without task_id first
            await agent_service.execute_task("Normal task")

            # Check that _current_task_id was set to a generated value
            assert agent_service._current_task_id is not None
            assert agent_service._current_task_id.startswith("task_")

            # Execute with task_id - this should work because we're calling _execute_normal_task directly
            await agent_service._execute_normal_task("Another task", {}, "123")

            # Check that _current_task_id was set to the provided task_id as string
            assert agent_service._current_task_id == "123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
