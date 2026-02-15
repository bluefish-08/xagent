"""
Tests for memory integration with agent patterns.
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from xagent.core.agent.pattern.memory_utils import (
    enhance_goal_with_memory,
    lookup_relevant_memories,
    store_execution_result_memory,
    store_plan_generation_memory,
    store_react_task_memory,
)
from xagent.core.memory import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse


class MockMemoryStore(MemoryStore):
    """Mock memory store for testing."""

    def __init__(self):
        self.memories = []

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.memories.append(note)
        return MemoryResponse(success=True, memory_id=note.id, content=note)

    def get(self, memory_id: str) -> MemoryResponse:
        for memory in self.memories:
            if memory.id == memory_id:
                return MemoryResponse(success=True, content=memory)
        return MemoryResponse(success=False, error="Not found")

    def update(self, note: MemoryNote) -> MemoryResponse:
        for i, memory in enumerate(self.memories):
            if memory.id == note.id:
                self.memories[i] = note
                return MemoryResponse(success=True)
        return MemoryResponse(success=False, error="Not found")

    def delete(self, memory_id: str) -> MemoryResponse:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                del self.memories[i]
                return MemoryResponse(success=True)
        return MemoryResponse(success=False, error="Not found")

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict = None,
        similarity_threshold: Optional[float] = None,
    ) -> list:
        # Simple mock search - return limited memories
        return self.memories[:k]

    def clear(self) -> None:
        self.memories.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        results = self.memories.copy()

        if filters:
            filtered_results = []
            for note in results:
                match = True

                # Category filter
                if "category" in filters and note.category != filters["category"]:
                    match = False

                # Tag filter
                if "tags" in filters:
                    required_tags = filters["tags"]
                    if not all(tag in note.tags for tag in required_tags):
                        match = False

                # Keyword filter
                if "keywords" in filters:
                    required_keywords = filters["keywords"]
                    if not all(
                        keyword in note.keywords for keyword in required_keywords
                    ):
                        match = False

                if match:
                    filtered_results.append(note)

            results = filtered_results

        return results

    def get_stats(self) -> dict[str, Any]:
        total_count = len(self.memories)
        category_counts = {}
        tag_counts = {}

        for note in self.memories:
            # Count by category
            category_counts[note.category] = category_counts.get(note.category, 0) + 1

            # Count tags
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_count": total_count,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "memory_store_type": "mock",
        }


class TestMemoryUtils:
    """Test memory utility functions."""

    def test_enhance_goal_with_memory_no_memories(self):
        """Test goal enhancement with no memories."""
        goal = "Test goal"
        memories = []

        result = enhance_goal_with_memory(goal, memories)

        assert result == goal

    def test_enhance_goal_with_memory_with_memories(self):
        """Test goal enhancement with memories."""
        goal = "Test goal"
        memories = [
            {"content": "Previous similar task succeeded", "context": "context1"},
            {"content": "Another relevant experience", "context": "context2"},
        ]

        result = enhance_goal_with_memory(goal, memories)

        assert "Test goal" in result
        assert "Relevant Context from Previous Experience:" in result
        assert "Previous similar task succeeded" in result
        assert "Another relevant experience" in result

    def test_store_plan_generation_memory_success(self):
        """Test successful plan generation memory storage."""
        memory_store = MockMemoryStore()
        goal = "Test goal"
        steps_count = 5
        plan_id = "test_plan_id"

        result = store_plan_generation_memory(memory_store, goal, steps_count, plan_id)

        assert result is not None
        assert len(memory_store.memories) == 1

        stored_memory = memory_store.memories[0]
        assert "Task Planning Experience: Test goal" in stored_memory.content
        assert "5 steps" in stored_memory.content
        assert "task planning" in stored_memory.keywords
        assert stored_memory.metadata["steps_count"] == 5
        assert stored_memory.metadata["plan_id"] == plan_id

    def test_store_plan_generation_memory_no_plan_id(self):
        """Test plan generation memory storage without plan ID."""
        memory_store = MockMemoryStore()
        goal = "Test goal"
        steps_count = 3

        result = store_plan_generation_memory(memory_store, goal, steps_count)

        assert result is not None
        assert len(memory_store.memories) == 1

        stored_memory = memory_store.memories[0]
        assert stored_memory.metadata["plan_id"] is None

    def test_store_plan_generation_memory_failure(self):
        """Test plan generation memory storage failure."""
        memory_store = MockMemoryStore()
        # Mock add to always fail
        memory_store.add = MagicMock(return_value=MemoryResponse(success=False))

        result = store_plan_generation_memory(memory_store, "test", 5)

        assert result is None

    def test_store_execution_result_memory_success(self):
        """Test successful execution result memory storage."""
        memory_store = MockMemoryStore()
        results = [
            {"step_id": "step1", "status": "completed"},
            {"step_id": "step2", "status": "completed"},
        ]
        plan_id = "test_plan"

        result = store_execution_result_memory(
            memory_store, results, "Test goal", plan_id
        )

        assert result is not None
        assert len(memory_store.memories) == 1

        stored_memory = memory_store.memories[0]
        assert "Goal: Test goal" in stored_memory.content
        assert "2 successful steps out of 2 total steps" in stored_memory.content
        assert "execution" in stored_memory.keywords
        assert stored_memory.metadata["total_steps"] == 2

    def test_store_react_task_memory_success(self):
        """Test successful ReAct task memory storage."""
        memory_store = MockMemoryStore()
        task = "Test task"
        result = {
            "success": True,
            "output": "Task completed successfully",
            "iterations": 3,
            "history": ["step1", "step2", "step3"],
        }

        result_memory_id = store_react_task_memory(memory_store, task, result)

        assert result_memory_id is not None
        assert len(memory_store.memories) == 1

        stored_memory = memory_store.memories[0]
        assert "Task: Test task" in stored_memory.content
        assert "Outcome: Success" in stored_memory.content
        assert "react" in stored_memory.keywords
        assert stored_memory.metadata["success"] is True

    def test_lookup_relevant_memories_success(self):
        """Test successful memory lookup."""
        memory_store = MockMemoryStore()

        # Add test memories
        test_memory = MemoryNote(
            content="Test memory content",
            keywords=["test", "memory"],
            category="test_category",
            metadata={"operation": "test"},
        )
        memory_store.add(test_memory)

        result = lookup_relevant_memories(
            memory_store, "test query", "test_category", 5
        )

        assert len(result) == 1
        assert result[0]["content"] == "Test memory content"
        assert result[0]["keywords"] == ["test", "memory"]
        assert result[0]["metadata"]["operation"] == "test"

    def test_lookup_relevant_memories_no_category(self):
        """Test memory lookup without category filter."""
        memory_store = MockMemoryStore()

        # Add test memories with different categories
        test_memory1 = MemoryNote(
            content="Memory 1",
            keywords=["test1"],
            category="category1",
            metadata={"operation": "test1"},
        )
        test_memory2 = MemoryNote(
            content="Memory 2",
            keywords=["test2"],
            category="category2",
            metadata={"operation": "test2"},
        )
        memory_store.add(test_memory1)
        memory_store.add(test_memory2)

        result = lookup_relevant_memories(memory_store, "test query", None, 5)

        assert len(result) == 2

    def test_lookup_relevant_memories_failure(self):
        """Test memory lookup failure."""
        memory_store = MockMemoryStore()
        # Mock search to raise exception
        memory_store.search = MagicMock(side_effect=Exception("Search failed"))

        result = lookup_relevant_memories(
            memory_store, "test query", "test_category", 5
        )

        assert result == []

    def test_lookup_relevant_memories_limit(self):
        """Test memory lookup with result limit."""
        memory_store = MockMemoryStore()

        # Add multiple test memories
        for i in range(10):
            test_memory = MemoryNote(
                content=f"Memory {i}",
                keywords=[f"test{i}"],
                category="test_category",
                metadata={"operation": f"test{i}"},
            )
            memory_store.add(test_memory)

        # Test with include_general=False to limit to specific category only
        result = lookup_relevant_memories(
            memory_store, "test query", "test_category", include_general=False, limit=3
        )

        assert len(result) == 3


class TestPatternMemoryIntegration:
    """Test memory integration with agent patterns."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store."""
        return MockMemoryStore()

    def test_dag_plan_execute_memory_integration(self, mock_memory_store, tmp_path):
        """Test that DAG plan-execute pattern can be integrated with memory."""
        from xagent.core.agent.pattern.dag_plan_execute.dag_plan_execute import (
            DAGPlanExecutePattern,
        )
        from xagent.core.model.chat.basic.base import BaseLLM
        from xagent.core.workspace import TaskWorkspace

        # Create a mock LLM
        mock_llm = MagicMock(spec=BaseLLM)

        # Create pattern with memory store and workspace (required)
        workspace = TaskWorkspace(id="test_workspace", base_dir=str(tmp_path))
        pattern = DAGPlanExecutePattern(
            llm=mock_llm, memory_store=mock_memory_store, workspace=workspace
        )

        assert pattern.memory_store is mock_memory_store

    def test_react_pattern_memory_integration(self, mock_memory_store):
        """Test that ReAct pattern can be integrated with memory."""
        from xagent.core.agent.pattern.react import ReActPattern
        from xagent.core.model.chat.basic.base import BaseLLM

        # Create a mock LLM
        mock_llm = MagicMock(spec=BaseLLM)

        # Create pattern with memory store
        pattern = ReActPattern(llm=mock_llm, memory_store=mock_memory_store)

        assert pattern.memory_store is mock_memory_store

    def test_dag_step_agent_memory_integration(self, mock_memory_store):
        """Test that DAG step agents can be integrated with shared memory."""
        from xagent.core.agent.pattern.dag_plan_execute.step_agent_factory import (
            StepAgentFactory,
        )
        from xagent.core.agent.trace import Tracer
        from xagent.core.model.chat.basic.base import BaseLLM
        from xagent.core.tools.adapters.vibe import Tool
        from xagent.core.workspace import TaskWorkspace

        # Create a mock LLM
        mock_llm = MagicMock(spec=BaseLLM)

        # Create a mock tool
        mock_tool = MagicMock(spec=Tool)
        mock_tool.metadata.name = "test_tool"

        # Create StepAgentFactory with memory store
        factory = StepAgentFactory(
            llm=mock_llm,
            tracer=Tracer(),
            workspace=TaskWorkspace(id="test"),
            memory_store=mock_memory_store,
        )

        # Create a step agent
        step_agent = factory.create_step_agent("test_step", [mock_tool])

        # Verify the step agent's ReAct pattern has access to the memory store
        react_pattern = step_agent.patterns[0]
        assert hasattr(react_pattern, "memory_store")
        assert react_pattern.memory_store is mock_memory_store

    @pytest.mark.asyncio
    async def test_react_pattern_with_memory_enhancement(self, mock_memory_store):
        """Test ReAct pattern task enhancement with memory."""
        from xagent.core.agent.pattern.react import ReActPattern
        from xagent.core.model.chat.basic.base import BaseLLM
        from xagent.core.tools.adapters.vibe import Tool

        # Add a memory to the store
        test_memory = MemoryNote(
            content="Previous task was completed successfully",
            keywords=["test", "previous"],
            category="react_memory",
            metadata={"operation": "react_task_completion"},
        )
        mock_memory_store.add(test_memory)

        # Create a mock LLM
        mock_llm = MagicMock(spec=BaseLLM)

        # Create pattern with memory store
        pattern = ReActPattern(llm=mock_llm, memory_store=mock_memory_store)

        # Register a mock tool
        mock_tool = MagicMock(spec=Tool)
        mock_tool.metadata.name = "test_tool"

        # The pattern should enhance the task with memory
        # (We can't fully test the run method without proper LLM setup,
        # but we can verify the memory integration is in place)
        assert pattern.memory_store is mock_memory_store
        assert len(mock_memory_store.memories) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
