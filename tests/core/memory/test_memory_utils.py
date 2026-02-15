"""Test cases for memory utility functions."""

from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.memory_utils import (
    enhance_goal_with_memory,
    lookup_relevant_memories,
    store_execution_result_memory,
    store_plan_generation_memory,
    store_react_task_memory,
)
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    store = Mock(spec=MemoryStore)
    return store


@pytest.fixture
def sample_memories():
    """Create sample memory notes for testing."""
    return [
        MemoryNote(
            content="System memory 1: Agent configuration",
            keywords=["config", "agent"],
            category="react_memory",
            metadata={"type": "system"},
        ),
        MemoryNote(
            content="General memory 1: User preferences",
            keywords=["preferences", "user"],
            category="general",
            metadata={"type": "user"},
        ),
        MemoryNote(
            content="System memory 2: Workflow setup",
            keywords=["workflow", "setup"],
            category="react_memory",
            metadata={"type": "system"},
        ),
        MemoryNote(
            content="General memory 2: Project context",
            keywords=["project", "context"],
            category="general",
            metadata={"type": "user"},
        ),
    ]


class TestLookupRelevantMemories:
    """Test cases for lookup_relevant_memories function."""

    def test_lookup_with_category_only(self, mock_memory_store, sample_memories):
        """Test looking up memories from specific category only."""
        # Setup mock to return filtered results
        mock_memory_store.search.return_value = [sample_memories[0], sample_memories[2]]

        result = lookup_relevant_memories(
            mock_memory_store,
            "test query",
            category="react_memory",
            include_general=False,
        )

        # Verify search was called with correct filters
        mock_memory_store.search.assert_called_once_with(
            query="test query",
            k=5,
            filters={"category": "react_memory"},
            similarity_threshold=None,
        )

        # Verify result structure
        assert len(result) == 2
        assert "context" not in result[0]  # Context field should be removed
        assert result[0]["content"] == "System memory 1: Agent configuration"
        assert result[1]["content"] == "System memory 2: Workflow setup"

    def test_lookup_with_general_only(self, mock_memory_store, sample_memories):
        """Test looking up memories from general category only."""
        # Setup mock to return filtered results
        mock_memory_store.search.return_value = [sample_memories[1], sample_memories[3]]

        result = lookup_relevant_memories(
            mock_memory_store, "test query", category=None, include_general=True
        )

        # Verify search was called with correct filters
        mock_memory_store.search.assert_called_once_with(
            query="test query",
            k=5,
            filters={"category": "general"},
            similarity_threshold=None,
        )

        # Verify result structure
        assert len(result) == 2
        assert result[0]["content"] == "General memory 1: User preferences"
        assert result[1]["content"] == "General memory 2: Project context"

    def test_lookup_with_both_categories(self, mock_memory_store, sample_memories):
        """Test looking up memories from both system and general categories."""

        # Setup mock to return different results for different calls
        def mock_search_side_effect(query, k, filters, similarity_threshold=None):
            if filters.get("category") == "react_memory":
                return [sample_memories[0], sample_memories[2]]
            elif filters.get("category") == "general":
                return [sample_memories[1], sample_memories[3]]
            return []

        mock_memory_store.search.side_effect = mock_search_side_effect

        result = lookup_relevant_memories(
            mock_memory_store,
            "test query",
            category="react_memory",
            include_general=True,
        )

        # Verify both searches were called
        assert mock_memory_store.search.call_count == 2

        # Verify deduplication and combination
        assert len(result) == 4
        contents = [r["content"] for r in result]
        assert "System memory 1: Agent configuration" in contents
        assert "General memory 1: User preferences" in contents
        assert "System memory 2: Workflow setup" in contents
        assert "General memory 2: Project context" in contents

    def test_lookup_with_duplicate_content(self, mock_memory_store):
        """Test that duplicate memories are properly deduplicated."""
        duplicate_memory = MemoryNote(
            content="Duplicate content",
            keywords=["test"],
            category="react_memory",
            metadata={"type": "system"},
        )

        # Setup mock to return memories with duplicate content
        mock_memory_store.search.return_value = [
            duplicate_memory,  # From system search
            duplicate_memory,  # From general search
        ]

        result = lookup_relevant_memories(
            mock_memory_store,
            "test query",
            category="react_memory",
            include_general=True,
        )

        # Verify deduplication
        assert len(result) == 1
        assert result[0]["content"] == "Duplicate content"

    def test_lookup_with_limit(self, mock_memory_store, sample_memories):
        """Test that result limit is properly applied."""
        # Setup mock to return more results than limit
        mock_memory_store.search.return_value = sample_memories

        result = lookup_relevant_memories(
            mock_memory_store,
            "test query",
            category="react_memory",
            include_general=True,
            limit=3,
        )

        # Verify limit is applied
        assert len(result) <= 3

    def test_lookup_with_error_handling(self, mock_memory_store):
        """Test error handling in lookup function."""
        # Setup mock to raise exception
        mock_memory_store.search.side_effect = Exception("Database error")

        result = lookup_relevant_memories(
            mock_memory_store, "test query", category="react_memory"
        )

        # Verify empty list is returned on error
        assert result == []

    def test_lookup_with_no_category_and_no_general(self, mock_memory_store):
        """Test lookup when no category specified and general is disabled."""
        result = lookup_relevant_memories(
            mock_memory_store, "test query", category=None, include_general=False
        )

        # Verify no search is performed
        mock_memory_store.search.assert_not_called()
        assert result == []


class TestStorePlanGenerationMemory:
    """Test cases for store_plan_generation_memory function."""

    def test_store_plan_generation_success(self, mock_memory_store):
        """Test successful storage of plan generation memory."""
        goal = "Complete the project"
        steps_count = 5
        plan_id = "plan_123"

        # Setup mock response
        mock_memory_store.add.return_value = Mock(success=True)

        store_plan_generation_memory(mock_memory_store, goal, steps_count, plan_id)

        # Verify memory store was called
        mock_memory_store.add.assert_called_once()

        # Verify the memory note structure
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert "Generated execution plan with 5 steps" in call_args.content
        assert goal in call_args.content
        assert call_args.category == "dag_plan_execute_memory"
        assert "task planning" in call_args.keywords
        assert call_args.metadata["plan_id"] == plan_id
        assert call_args.metadata["steps_count"] == steps_count

    def test_store_plan_generation_error(self, mock_memory_store):
        """Test error handling in store_plan_generation_memory."""
        # Setup mock to raise exception
        mock_memory_store.add.side_effect = Exception("Storage error")

        # Should not raise exception, should log and return
        result = store_plan_generation_memory(mock_memory_store, "goal", 5, "plan_123")

        # Verify no exception was raised
        assert result is None


class TestEnhanceGoalWithMemory:
    """Test cases for enhance_goal_with_memory function."""

    def test_enhance_goal_with_no_memories(self):
        """Test goal enhancement with no memories."""
        goal = "Complete the project"
        memories = []

        result = enhance_goal_with_memory(goal, memories)

        # Verify goal is returned unchanged
        assert result == goal

    def test_enhance_goal_with_memories(self):
        """Test goal enhancement with memories."""
        goal = "Complete the project"
        memories = [
            {
                "content": "User prefers detailed reports",
                "keywords": ["preferences"],
                "metadata": {"type": "user"},
            },
            {
                "content": "Project uses agile methodology",
                "keywords": ["workflow"],
                "metadata": {"type": "system"},
            },
        ]

        result = enhance_goal_with_memory(goal, memories)

        # Verify memories are included in the enhanced goal
        assert "Relevant Context from Previous Experience" in result
        assert "User prefers detailed reports" in result
        assert "Project uses agile methodology" in result
        assert goal in result

    def test_enhance_goal_with_context_field_ignored(self):
        """Test that context field is ignored if present (backward compatibility)."""
        goal = "Complete the project"
        memories = [
            {
                "content": "User prefers detailed reports",
                "context": "This should be ignored",
                "keywords": ["preferences"],
                "metadata": {"type": "user"},
            }
        ]

        result = enhance_goal_with_memory(goal, memories)

        # Verify context field is ignored
        assert "This should be ignored" not in result
        assert "User prefers detailed reports" in result


class TestStoreExecutionResultMemory:
    """Test cases for store_execution_result_memory function."""

    def test_store_execution_result_success(self, mock_memory_store):
        """Test successful storage of execution result memory."""
        plan_id = "plan_123"
        goal = "Test goal"
        # The function expects a list of results
        results = [
            {"step": "setup", "status": "completed"},
            {"step": "execute", "status": "completed"},
        ]

        mock_memory_store.add.return_value = Mock(success=True)

        store_execution_result_memory(mock_memory_store, results, goal, plan_id)

        # Verify memory store was called
        mock_memory_store.add.assert_called_once()

        # Verify the memory note structure
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert "Execution result:" in call_args.content
        assert call_args.category == "execution_memory"
        assert call_args.metadata["total_steps"] == 2
        assert call_args.metadata["successful_steps"] == 2
        assert call_args.metadata["failed_steps"] == 0


class TestStoreReactTaskMemory:
    """Test cases for store_react_task_memory function."""

    def test_store_react_task_success(self, mock_memory_store):
        """Test successful storage of React task memory."""
        task = "Calculate the total cost"
        result = {"output": "The total cost is $1500"}

        mock_memory_store.add.return_value = Mock(success=True)

        store_react_task_memory(mock_memory_store, task, result)

        # Verify memory store was called
        mock_memory_store.add.assert_called_once()

        # Verify the memory note structure
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert "Task:" in call_args.content and "Outcome:" in call_args.content
        assert task in call_args.content
        assert "Calculate the total cost" in call_args.content
        assert call_args.category == "react_memory"
        assert call_args.metadata["task"] == task
