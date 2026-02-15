"""
Comprehensive unit tests for the Agent class and AgentContext.

This module tests the Agent functionality including:
- Agent initialization and properties
- Sub-agent management
- Execution history handling
- Result management
- Query execution details
- AgentContext functionality
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from xagent.core.agent.agent import Agent
from xagent.core.agent.context import AgentContext
from xagent.core.agent.pattern.base import AgentPattern
from xagent.core.memory import MemoryStore
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe import Tool


class MockPattern(AgentPattern):
    """Mock pattern for testing."""

    def __init__(self, name: str):
        self.name = name

    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: List[Tool],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        return {"success": True, "pattern": self.name}


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str):
        self.name = name

    @property
    def metadata(self):
        from xagent.core.tools.adapters.vibe.base import ToolMetadata

        return ToolMetadata(name=self.name, description=f"Mock tool {self.name}")

    async def execute(self, **kwargs) -> Any:
        return {"mock_tool_result": True}


class MockMemoryStore(InMemoryMemoryStore):
    """Mock memory store for testing."""

    def __init__(self):
        super().__init__()

    async def store(self, key: str, value: Any) -> None:
        # Create a MemoryNote for simple key-value storage
        from xagent.core.memory.core import MemoryNote

        note = MemoryNote(content=str(value), metadata={"key": key})
        self.add(note)

    async def retrieve(self, key: str) -> Optional[Any]:
        # Search for notes with this key
        results = self.search(key)
        for note in results:
            if note.metadata.get("key") == key:
                return note.content
        return None


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self):
        self.call_count = 0
        self.responses = []
        self._model_name = "mock_llm"

    @property
    def abilities(self) -> List[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        self.call_count += 1
        return self.responses.pop(0) if self.responses else "Mock LLM response"


class TestAgentContext:
    """Test cases for AgentContext class."""

    def test_agent_context_initialization(self):
        """Test AgentContext initialization with default values."""
        context = AgentContext()

        assert context.task_id is not None
        assert len(context.task_id) > 0
        assert context.user_id is None
        assert context.session_id is None
        assert context.start_time is not None
        assert isinstance(context.start_time, datetime)
        assert context.history == []
        assert context.state == {}

    def test_agent_context_with_custom_values(self):
        """Test AgentContext initialization with custom values."""
        custom_task_id = "custom-task-123"
        custom_user_id = "user-456"
        custom_session_id = "session-789"

        context = AgentContext(
            task_id=custom_task_id, user_id=custom_user_id, session_id=custom_session_id
        )

        assert context.task_id == custom_task_id
        assert context.user_id == custom_user_id
        assert context.session_id == custom_session_id

    def test_agent_context_history_operations(self):
        """Test AgentContext history operations."""
        context = AgentContext()

        # Initially empty
        assert context.history == []

        # Add to history
        context.history.append("First action")
        context.history.append("Second action")

        assert len(context.history) == 2
        assert context.history[0] == "First action"
        assert context.history[1] == "Second action"

    def test_agent_context_state_operations(self):
        """Test AgentContext state operations."""
        context = AgentContext()

        # Initially empty
        assert context.state == {}

        # Add to state
        context.state["key1"] = "value1"
        context.state["key2"] = 42
        context.state["key3"] = {"nested": "value"}

        assert len(context.state) == 3
        assert context.state["key1"] == "value1"
        assert context.state["key2"] == 42
        assert context.state["key3"]["nested"] == "value"

    def test_agent_context_state_modification(self):
        """Test AgentContext state modification."""
        context = AgentContext()

        context.state["test_key"] = "initial_value"
        assert context.state["test_key"] == "initial_value"

        context.state["test_key"] = "modified_value"
        assert context.state["test_key"] == "modified_value"

        del context.state["test_key"]
        assert "test_key" not in context.state


class TestAgent:
    """Test cases for Agent class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory store."""
        return MockMemoryStore()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return [MockTool("test_tool"), MockTool("another_tool")]

    @pytest.fixture
    def mock_patterns(self):
        """Create mock patterns."""
        return [MockPattern("pattern1"), MockPattern("pattern2")]

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MockLLM()

    @pytest.fixture
    def basic_agent(self, mock_patterns, mock_memory, mock_tools):
        """Create a basic agent without LLM."""
        return Agent(
            name="test_agent",
            patterns=mock_patterns,
            memory=mock_memory,
            tools=mock_tools,
        )

    @pytest.fixture
    def agent_with_llm(self, mock_patterns, mock_memory, mock_tools, mock_llm):
        """Create an agent with LLM."""
        return Agent(
            name="agent_with_llm",
            patterns=mock_patterns,
            memory=mock_memory,
            tools=mock_tools,
            llm=mock_llm,
        )

    def test_agent_initialization(
        self, basic_agent, mock_patterns, mock_memory, mock_tools
    ):
        """Test Agent initialization."""
        assert basic_agent.name == "test_agent"
        assert basic_agent.patterns == mock_patterns
        assert basic_agent.memory == mock_memory
        assert basic_agent.tools == mock_tools
        assert basic_agent.llm is None
        assert basic_agent._execution_history is None
        assert basic_agent._final_result is None
        assert basic_agent._step_id is None
        assert basic_agent._sub_agents == {}
        assert basic_agent._parent_agent is None
        assert basic_agent._created_at is not None

    def test_agent_initialization_with_llm(self, agent_with_llm, mock_llm):
        """Test Agent initialization with LLM."""
        assert agent_with_llm.llm == mock_llm

    def test_agent_get_runner(self, basic_agent):
        """Test Agent.get_runner method."""
        runner = basic_agent.get_runner()

        assert runner is not None
        assert runner.agent == basic_agent
        assert runner.context is not None

    def test_agent_created_at_timestamp(self, basic_agent):
        """Test that created_at timestamp is set correctly."""
        assert isinstance(basic_agent._created_at, datetime)
        assert basic_agent._created_at <= datetime.now()

    def test_agent_add_sub_agent(self, basic_agent):
        """Test adding a sub-agent."""
        sub_agent = Agent(
            name="sub_agent", patterns=[], memory=MockMemoryStore(), tools=[]
        )

        basic_agent.add_sub_agent(sub_agent)

        assert "sub_agent" in basic_agent._sub_agents
        assert basic_agent._sub_agents["sub_agent"] == sub_agent
        assert sub_agent._parent_agent == basic_agent

    def test_agent_add_multiple_sub_agents(self, basic_agent):
        """Test adding multiple sub-agents."""
        sub_agent1 = Agent(
            name="sub_agent1", patterns=[], memory=MockMemoryStore(), tools=[]
        )

        sub_agent2 = Agent(
            name="sub_agent2", patterns=[], memory=MockMemoryStore(), tools=[]
        )

        basic_agent.add_sub_agent(sub_agent1)
        basic_agent.add_sub_agent(sub_agent2)

        assert len(basic_agent._sub_agents) == 2
        assert "sub_agent1" in basic_agent._sub_agents
        assert "sub_agent2" in basic_agent._sub_agents
        assert sub_agent1._parent_agent == basic_agent
        assert sub_agent2._parent_agent == basic_agent

    def test_agent_get_sub_agent(self, basic_agent):
        """Test getting a sub-agent by name."""
        sub_agent = Agent(
            name="sub_agent", patterns=[], memory=MockMemoryStore(), tools=[]
        )

        basic_agent.add_sub_agent(sub_agent)

        retrieved = basic_agent.get_sub_agent("sub_agent")
        assert retrieved == sub_agent

        # Test non-existent sub-agent
        non_existent = basic_agent.get_sub_agent("non_existent")
        assert non_existent is None

    def test_agent_has_execution_history(self, basic_agent):
        """Test checking if agent has execution history."""
        # Initially no history
        assert not basic_agent.has_execution_history()

        # Set history
        basic_agent.set_execution_history([{"role": "user", "content": "test"}])
        assert basic_agent.has_execution_history()

    def test_agent_get_execution_history(self, basic_agent):
        """Test getting execution history."""
        # Initially no history
        assert basic_agent.get_execution_history() is None

        # Set and get history
        history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        basic_agent.set_execution_history(history)

        retrieved_history = basic_agent.get_execution_history()
        assert retrieved_history == history
        # Note: The implementation may return the same object, which is acceptable

    def test_agent_set_execution_history(self, basic_agent):
        """Test setting execution history."""
        history = [{"role": "user", "content": "test"}]
        basic_agent.set_execution_history(history)

        assert basic_agent._execution_history == history
        assert basic_agent.has_execution_history()

    def test_agent_get_final_result(self, basic_agent):
        """Test getting final result."""
        # Initially no result
        assert basic_agent.get_final_result() is None

        # Set and get result
        result = {"success": True, "output": "test result"}
        basic_agent.set_final_result(result)

        retrieved_result = basic_agent.get_final_result()
        assert retrieved_result == result

    def test_agent_set_final_result(self, basic_agent):
        """Test setting final result."""
        result = {"success": True, "output": "test result"}
        basic_agent.set_final_result(result)

        assert basic_agent._final_result == result
        assert basic_agent.get_final_result() == result

    def test_agent_step_id(self, basic_agent):
        """Test step ID management."""
        # Initially no step ID
        assert basic_agent._step_id is None

        # Set step ID
        basic_agent._step_id = "step_123"
        assert basic_agent._step_id == "step_123"

    def test_agent_query_execution_details_no_history(self, basic_agent):
        """Test querying execution details when no history exists."""
        result = basic_agent.query_execution_details("What happened?")
        # Should return a coroutine since it's an async method
        import asyncio

        assert asyncio.iscoroutine(result)

    def test_agent_query_execution_details_no_llm(self, basic_agent):
        """Test querying execution details when no LLM is available."""
        basic_agent.set_execution_history([{"role": "user", "content": "test"}])

        result = basic_agent.query_execution_details("What happened?")
        # Should return a coroutine since it's an async method
        import asyncio

        assert asyncio.iscoroutine(result)

    @pytest.mark.asyncio
    async def test_agent_query_execution_details_with_history_and_llm(
        self, agent_with_llm
    ):
        """Test querying execution details with history and LLM."""
        # Set up execution history
        history = [
            {"role": "user", "content": "Analyze this data"},
            {"role": "assistant", "content": "I'll analyze the data for you"},
        ]
        agent_with_llm.set_execution_history(history)

        # Set up final result
        result = {"success": True, "analysis": "Data analysis complete"}
        agent_with_llm.set_final_result(result)

        # Set up mock LLM response
        agent_with_llm.llm.responses = ["The agent analyzed the data successfully"]

        query_result = await agent_with_llm.query_execution_details(
            "What did the agent do?"
        )

        assert query_result == "The agent analyzed the data successfully"
        assert agent_with_llm.llm.call_count == 1

    def test_agent_format_history_for_query(self, basic_agent):
        """Test formatting history for querying."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        formatted = basic_agent._format_history_for_query(history)

        expected = "USER: Hello\nASSISTANT: Hi there!\nUSER: How are you?"
        assert formatted == expected

    def test_agent_format_history_for_query_with_missing_role(self, basic_agent):
        """Test formatting history with missing role."""
        history = [
            {"content": "Hello"},  # Missing role
            {"role": "user", "content": "Hi"},
        ]

        formatted = basic_agent._format_history_for_query(history)

        expected = "UNKNOWN: Hello\nUSER: Hi"
        assert formatted == expected

    def test_agent_format_history_for_query_with_missing_content(self, basic_agent):
        """Test formatting history with missing content."""
        history = [
            {"role": "user"},  # Missing content
            {"role": "assistant", "content": "Hello"},
        ]

        formatted = basic_agent._format_history_for_query(history)

        expected = "USER: \nASSISTANT: Hello"
        assert formatted == expected

    def test_agent_to_dict(self, basic_agent):
        """Test Agent.to_dict method."""
        # Add some sub-agents
        sub_agent = Agent(
            name="sub_agent", patterns=[], memory=MockMemoryStore(), tools=[]
        )
        basic_agent.add_sub_agent(sub_agent)

        # Set some properties
        basic_agent.set_execution_history([{"role": "user", "content": "test"}])
        basic_agent._step_id = "step_123"

        agent_dict = basic_agent.to_dict()

        assert agent_dict["name"] == "test_agent"
        assert agent_dict["patterns"] == ["MockPattern", "MockPattern"]
        assert agent_dict["tools"] == ["test_tool", "another_tool"]
        assert agent_dict["memory_type"] == "MockMemoryStore"
        assert agent_dict["llm_available"] is False
        assert agent_dict["sub_agents"] == ["sub_agent"]
        assert agent_dict["has_execution_history"] is True
        assert "created_at" in agent_dict
        assert agent_dict["step_id"] == "step_123"

    def test_agent_to_dict_with_llm(self, agent_with_llm):
        """Test Agent.to_dict method with LLM."""
        agent_dict = agent_with_llm.to_dict()

        assert agent_dict["llm_available"] is True

    def test_agent_to_dict_empty_sub_agents(self, basic_agent):
        """Test Agent.to_dict method with no sub-agents."""
        agent_dict = basic_agent.to_dict()

        assert agent_dict["sub_agents"] == []

    def test_agent_to_dict_no_execution_history(self, basic_agent):
        """Test Agent.to_dict method with no execution history."""
        agent_dict = basic_agent.to_dict()

        assert agent_dict["has_execution_history"] is False

    def test_agent_to_dict_no_step_id(self, basic_agent):
        """Test Agent.to_dict method with no step ID."""
        agent_dict = basic_agent.to_dict()

        assert agent_dict["step_id"] is None

    def test_agent_nested_relationships(self):
        """Test nested agent relationships."""
        # Create parent agent
        parent = Agent(name="parent", patterns=[], memory=MockMemoryStore(), tools=[])

        # Create child agent
        child = Agent(name="child", patterns=[], memory=MockMemoryStore(), tools=[])

        # Create grandchild agent
        grandchild = Agent(
            name="grandchild", patterns=[], memory=MockMemoryStore(), tools=[]
        )

        # Build hierarchy
        parent.add_sub_agent(child)
        child.add_sub_agent(grandchild)

        # Test relationships
        assert child._parent_agent == parent
        assert grandchild._parent_agent == child
        assert parent.get_sub_agent("child") == child
        assert child.get_sub_agent("grandchild") == grandchild
        assert parent._sub_agents == {"child": child}
        assert child._sub_agents == {"grandchild": grandchild}


if __name__ == "__main__":
    pytest.main([__file__])
