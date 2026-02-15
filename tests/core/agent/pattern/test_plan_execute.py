from typing import Any

import pytest

from xagent.core.agent.context import AgentContext
from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata
from xagent.core.workspace import TaskWorkspace


class MockLLM(BaseLLM):
    def __init__(self):
        self._model_name = "mock_llm"

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        """Mock LLM doesn't support thinking mode"""
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        # Handle JSON mode - return dict instead of string
        if kwargs.get("response_format") == {"type": "json_object"}:
            # Mock LLM response for task analysis in JSON mode
            if any(
                "task execution analyzer" in msg.get("content", "").lower()
                for msg in messages
            ):
                return '{"success": true, "direct_answer": "Weather is sunny in Singapore today", "file_outputs": [], "confidence": "high", "reasoning": "The weather information was successfully retrieved for Singapore."}'
            # Mock LLM response for Chinese task analysis in JSON mode
            elif any("任务分析助手" in msg.get("content", "") for msg in messages):
                return '{"success": true, "direct_answer": "Weather is sunny in Singapore today", "file_outputs": [], "confidence": "high", "reasoning": "The weather information was successfully retrieved for Singapore."}'

        # Mock LLM response for plan generation
        if any("execution plan" in msg.get("content", "").lower() for msg in messages):
            return """{
                "plan": {
                    "goal": "Check weather",
                    "steps": [
                        {
                            "id": "step1",
                            "name": "get_weather",
                            "description": "Check the weather in a city",
                            "tool_names": ["get_weather"],
                            "dependencies": []
                        }
                    ]
                }
            }"""
        # Mock LLM response for goal achievement check
        elif any(
            "goal" in msg.get("content", "").lower()
            and "achieved" in msg.get("content", "").lower()
            for msg in messages
        ):
            return '{"achieved": true, "reason": "Weather information retrieved successfully"}'
        # Mock LLM response for task analysis
        elif any(
            "task execution analyzer" in msg.get("content", "").lower()
            for msg in messages
        ):
            return '{"success": true, "direct_answer": "Weather is sunny in Singapore today", "file_outputs": [], "confidence": "high", "reasoning": "The weather information was successfully retrieved for Singapore."}'
        # Mock LLM response for Chinese task analysis
        elif any("任务分析助手" in msg.get("content", "") for msg in messages):
            return '{"success": true, "direct_answer": "Weather is sunny in Singapore today", "file_outputs": [], "confidence": "high", "reasoning": "The weather information was successfully retrieved for Singapore."}'
        # Mock ReAct response
        else:
            return '{"type": "final_answer", "content": "Weather is sunny in Singapore today"}'


class MockWeatherTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="get_weather", description="Mock weather tool")

    def args_type(self):
        return dict  # Not enforced in mock

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"forecast": "sunny", "city": args.get("city", "Unknown")}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"forecast": "sunny", "city": args.get("city", "Unknown")}

    async def save_state_json(self):
        return {}

    async def load_state_json(self, state: dict[str, Any]):
        pass

    def return_value_as_string(self, value: Any) -> str:
        return str(value)


class DummyMemoryStore(MemoryStore):
    def add(self, note):
        return MemoryResponse(success=True)

    def get(self, note_id: str):
        return MemoryResponse(success=True)

    def update(self, note):
        return MemoryResponse(success=True)

    def delete(self, note_id: str):
        return MemoryResponse(success=True)

    def search(self, query: str, k: int = 5, filters=None, similarity_threshold=None):
        return []

    def clear(self):
        pass

    def list_all(self, filters=None):
        return []

    def get_stats(self):
        return {"total_count": 0, "category_counts": {}, "tag_counts": {}}


@pytest.mark.asyncio
async def test_dag_plan_execute_pattern(tmp_path):
    """Test the new DAG Plan Execute pattern"""
    llm = MockLLM()
    memory = DummyMemoryStore()
    tools = [MockWeatherTool()]
    workspace = TaskWorkspace(id="test_workspace", base_dir=str(tmp_path))
    pattern = DAGPlanExecutePattern(
        llm, max_iterations=1, goal_check_enabled=True, workspace=workspace
    )

    result = await pattern.run(
        task="check today's weather in Singapore",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Verify basic success
    assert result["success"] is True
    assert "output" in result
    assert "history" in result

    # Check execution history structure
    history = result["history"]
    assert len(history) > 0

    # Check that a plan was generated
    first_iteration = history[0]
    assert "plan" in first_iteration
    plan = first_iteration["plan"]
    assert "steps" in plan
    assert len(plan["steps"]) > 0

    # Verify the step structure
    step = plan["steps"][0]
    assert step["name"] == "get_weather"
    assert step["tool_names"] == ["get_weather"]
    assert step["dependencies"] == []


@pytest.mark.asyncio
async def test_dag_plan_execute_pattern_with_dependencies(tmp_path):
    """Test DAG pattern with step dependencies"""

    class MockPlanningLLM(BaseLLM):
        def __init__(self):
            self._model_name = "mock_planning_llm"

        @property
        def abilities(self) -> list[str]:
            return ["chat"]

        @property
        def model_name(self) -> str:
            """Get the model name/identifier."""
            return self._model_name

        @property
        def supports_thinking_mode(self) -> bool:
            """Mock LLM doesn't support thinking mode"""
            return False

        async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
            # Handle JSON mode - return dict instead of string
            if kwargs.get("response_format") == {"type": "json_object"}:
                # Mock LLM response for task analysis in JSON mode
                if any(
                    "task execution analyzer" in msg.get("content", "").lower()
                    for msg in messages
                ):
                    return '{"success": true, "direct_answer": "Detailed weather information for Singapore was retrieved successfully", "file_outputs": [], "confidence": "high", "reasoning": "Both steps completed successfully."}'
                # Mock LLM response for Chinese task analysis in JSON mode
                elif any("任务分析助手" in msg.get("content", "") for msg in messages):
                    return '{"success": true, "direct_answer": "Detailed weather information for Singapore was retrieved successfully", "file_outputs": [], "confidence": "high", "reasoning": "Both steps completed successfully."}'

            if any(
                "execution plan" in msg.get("content", "").lower() for msg in messages
            ):
                return """{
                    "plan": {
                        "goal": "get detailed weather information for Singapore",
                        "steps": [
                            {
                                "id": "step1",
                                "name": "search_location",
                                "description": "Find the location",
                                "tool_names": ["get_weather"],
                                "dependencies": [],
                                "difficulty": "easy"
                            },
                            {
                                "id": "step2",
                                "name": "get_detailed_weather",
                                "description": "Get detailed weather for the location",
                                "tool_names": ["get_weather"],
                                "dependencies": ["step1"],
                                "difficulty": "hard"
                            }
                        ]
                    }
                }"""
            elif any("achieved" in msg.get("content", "").lower() for msg in messages):
                return (
                    '{"achieved": true, "reason": "Both steps completed successfully"}'
                )
            else:
                return (
                    '{"type": "final_answer", "content": "Task completed successfully"}'
                )

    llm = MockPlanningLLM()
    memory = DummyMemoryStore()
    tools = [MockWeatherTool()]
    workspace = TaskWorkspace(id="test_workspace", base_dir=str(tmp_path))
    pattern = DAGPlanExecutePattern(
        llm, max_iterations=1, goal_check_enabled=True, workspace=workspace
    )

    result = await pattern.run(
        task="get detailed weather information for Singapore",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    assert result["success"] is True
    history = result["history"]
    plan = history[0]["plan"]

    # Verify dependency structure
    assert len(plan["steps"]) == 2
    assert plan["steps"][0]["dependencies"] == []
    assert plan["steps"][1]["dependencies"] == ["step1"]


@pytest.mark.asyncio
async def test_dag_pattern_error_handling(tmp_path):
    """Test error handling in DAG pattern"""

    class FailingLLM(BaseLLM):
        def __init__(self):
            self._model_name = "failing_llm"

        @property
        def abilities(self) -> list[str]:
            return ["chat"]

        @property
        def model_name(self) -> str:
            """Get the model name/identifier."""
            return self._model_name

        @property
        def supports_thinking_mode(self) -> bool:
            """Mock LLM doesn't support thinking mode"""
            return False

        async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
            # Return invalid JSON to trigger parsing error
            return "invalid json response"

    llm = FailingLLM()
    memory = DummyMemoryStore()
    tools = [MockWeatherTool()]
    workspace = TaskWorkspace(id="test_workspace", base_dir=str(tmp_path))
    pattern = DAGPlanExecutePattern(llm, max_iterations=1, workspace=workspace)

    result = await pattern.run(
        task="test error handling",
        memory=memory,
        tools=tools,
        context=AgentContext(),
    )

    # Should handle errors gracefully
    assert result["success"] is False
    assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
