"""Test for MockLLM utility."""

import tempfile
import unittest
from typing import Any

from tests.utils.mock_llm import MockLLM
from xagent.core.agent.context import AgentContext
from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.agent.trace import Tracer
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata


class MockTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="test_tool", description="Mock test tool")

    def args_type(self):
        return dict

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"result": "success", "args": args}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"result": "success", "args": args}

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
        return {"total_count": 0}


class TestMockLLM(unittest.IsolatedAsyncioTestCase):
    """Test cases for MockLLM utility."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for workspace
        self.temp_dir = tempfile.mkdtemp()
        self.llm = MockLLM()
        self.memory = DummyMemoryStore()
        self.tools = [MockTool()]

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_mock_llm_plan_generation(self):
        """Test that MockLLM generates valid execution plans."""
        # Test plan generation
        plan_request = [
            {"role": "system", "content": "Generate an execution plan"},
            {
                "role": "user",
                "content": "Goal: analyze data\nGenerate a JSON array of steps for the execution plan.",
            },
        ]

        # Convert tools to the format expected by the LLM
        tools_for_llm = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Mock test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {
                                "type": "string",
                                "description": "Test parameter",
                            }
                        },
                    },
                },
            }
        ]

        response = await self.llm.chat(plan_request, tools=tools_for_llm)

        # Response should be a JSON string
        self.assertIsInstance(response, str)

        # Parse and validate the plan
        import json

        plan_data = json.loads(response)
        self.assertIsInstance(plan_data, dict)
        self.assertIn("plan", plan_data)

        plan = plan_data["plan"]["steps"]
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)

        # Each step should have required fields
        for step in plan:
            self.assertIn("id", step)
            self.assertIn("name", step)
            self.assertIn("description", step)
            self.assertIn("tool_names", step)
            self.assertIn("dependencies", step)

    async def test_mock_llm_goal_check(self):
        """Test that MockLLM performs goal achievement checks."""
        goal_check_request = [
            {"role": "system", "content": "Check if goal is achieved"},
            {
                "role": "user",
                "content": "Goal: analyze data\nCheck if the goal has been achieved.",
            },
        ]

        response = await self.llm.chat(goal_check_request)

        # Response should be a JSON string
        self.assertIsInstance(response, str)

        # Parse and validate the goal check result
        import json

        result = json.loads(response)
        self.assertIn("achieved", result)
        self.assertIn("reason", result)
        self.assertIsInstance(result["achieved"], bool)
        self.assertIsInstance(result["reason"], str)

    async def test_mock_llm_with_dag_pattern(self):
        """Test MockLLM integration with DAG pattern."""
        # Create tracer to capture events
        tracer = Tracer()

        # Create workspace (required)
        from xagent.core.workspace import TaskWorkspace

        workspace = TaskWorkspace(id="test_workspace", base_dir=self.temp_dir)

        # Create pattern with mock LLM
        pattern = DAGPlanExecutePattern(
            llm=self.llm,
            max_iterations=1,
            goal_check_enabled=True,
            tracer=tracer,
            workspace=workspace,
        )

        # Run the pattern
        result = await pattern.run(
            task="analyze data",
            memory=self.memory,
            tools=self.tools,
            context=AgentContext(),
        )

        # Should get a result
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_mock_llm_call_counting(self):
        """Test MockLLM call counting functionality."""
        initial_count = self.llm.get_call_count()
        self.assertEqual(initial_count, 0)

        # Reset call count
        self.llm.reset_call_count()
        self.assertEqual(self.llm.get_call_count(), 0)

    def test_mock_llm_thinking_mode(self):
        """Test MockLLM thinking mode property."""
        self.assertFalse(self.llm.supports_thinking_mode)


if __name__ == "__main__":
    unittest.main()
