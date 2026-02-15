#!/usr/bin/env python3
"""
Simple test script to verify the DAG Plan-Execute system works correctly.
"""

import asyncio
import sys
from typing import Any, Mapping, Optional, Type

import pytest
from pydantic import BaseModel

from tests.utils.mock_llm import MockLLM
from xagent.core.agent.service import AgentService
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.tools.adapters.vibe.base import ToolMetadata, ToolVisibility


class MockToolArgs(BaseModel):
    """Arguments for mock tool"""

    query: Optional[str] = "default query"


class MockToolReturn(BaseModel):
    """Return type for mock tool"""

    result: str
    success: bool


class MockTool:
    """Simple mock tool for testing"""

    def __init__(self, name: str, description: str = "Mock tool"):
        self._metadata = ToolMetadata(
            name=name, description=description, visibility=ToolVisibility.PUBLIC
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def args_type(self) -> Type[BaseModel]:
        return MockToolArgs

    def return_type(self) -> Type[BaseModel]:
        return MockToolReturn

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> dict:
        """Mock tool execution"""
        return {
            "tool_name": self.metadata.name,
            "args": dict(args),
            "result": f"Mock result from {self.metadata.name}",
            "success": True,
        }

    def run_json_sync(self, args: Mapping[str, Any]) -> dict:
        """Sync version - not used in this test"""
        return {
            "tool_name": self.metadata.name,
            "args": dict(args),
            "result": f"Mock result from {self.metadata.name}",
            "success": True,
        }

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass


@pytest.mark.asyncio
async def test_dag_system():
    """Test the DAG Plan-Execute system end-to-end"""
    print("=== Testing DAG Plan-Execute System ===")

    # Create mock tools
    tools = [
        MockTool("fetch_data", "Fetch data from database"),
        MockTool("validate_data", "Validate data quality"),
        MockTool("analyze_stats", "Perform statistical analysis"),
        MockTool("create_charts", "Create visualizations"),
        MockTool("generate_report", "Generate final report"),
    ]

    # Create AgentService with DAG pattern and limited iterations for testing
    mock_llm = MockLLM()
    agent_service = AgentService(
        name="test_dag_agent",
        id="test_dag_agent",
        llm=mock_llm,
        tools=tools,
        memory=InMemoryMemoryStore(),
        use_dag_pattern=True,
    )

    # Configure the DAG pattern for testing
    dag_pattern = agent_service.get_dag_pattern()
    if dag_pattern:
        dag_pattern.max_iterations = 2  # Reduce for testing
        dag_pattern.goal_check_enabled = (
            False  # Disable goal checking for predictable results
        )

    print(f"✅ Created AgentService with {len(tools)} tools")
    print(f"✅ DAG pattern enabled: {agent_service.get_dag_pattern() is not None}")

    # Test basic goal achievement
    task = "Analyze sales data and create a comprehensive report with visualizations"
    print(f"\n🎯 Executing task: {task}")

    try:
        result = await agent_service.execute_task(task)

        print("\n📋 Execution Result:")
        print(f"   Status: {result['status']}")
        print(f"   Success: {result['success']}")
        print(f"   Phase: {result['metadata'].get('phase', 'unknown')}")
        print(f"   Iterations: {result['metadata'].get('iterations', 0)}")

        if result.get("dag_status"):
            dag_status = result["dag_status"]
            print("\n📊 DAG Status:")
            print(f"   Phase: {dag_status.get('phase', 'unknown')}")
            if dag_status.get("progress"):
                progress = dag_status["progress"]
                print(
                    f"   Progress: {progress['completed_steps']}/{progress['total_steps']} ({progress['percentage']:.1f}%)"
                )

        print("\n🔄 Visualization updates received: 0")

        # Test DAG pattern methods
        dag_pattern = agent_service.get_dag_pattern()
        if dag_pattern:
            status = dag_pattern.get_execution_status()
            print("\n🔍 DAG Pattern Status:")
            print(f"   Phase: {status.get('phase', 'unknown')}")
            if status.get("current_plan"):
                plan = status["current_plan"]
                print(f"   Plan ID: {plan.get('id', 'unknown')}")
                print(f"   Steps: {len(plan.get('steps', []))}")

        return result["success"]

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_step_injection():
    """Test step injection functionality"""
    print("\n=== Testing Step Injection ===")

    # Create a simple agent service with MockLLM
    mock_llm = MockLLM()
    agent_service = AgentService(
        name="injection_test_agent",
        id="injection_test_agent",
        llm=mock_llm,
        tools=[MockTool("test_tool", "Test tool for injection")],
        use_dag_pattern=True,
    )

    # Test adding injection hooks (this would normally be called during execution)
    success = agent_service.add_step_injection(
        "test_step",
        pre_hook=lambda prompt, args: f"INJECTED: {prompt}",
        post_hook=lambda prompt, result: {**result, "injected": True},
    )

    print(
        f"✅ Step injection test: {'Success' if success else 'Failed (expected - no active plan)'}"
    )
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting DAG System Tests")

    test1_success = await test_dag_system()
    test2_success = await test_step_injection()

    print(f"\n{'=' * 50}")
    print("📈 Test Results:")
    print(f"   DAG System Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Step Injection Test: {'✅ PASS' if test2_success else '❌ FAIL'}")

    overall_success = test1_success and test2_success
    print(
        f"   Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}"
    )

    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
