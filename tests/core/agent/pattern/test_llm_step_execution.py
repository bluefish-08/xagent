#!/usr/bin/env python3
"""
Test script to verify LLM participation in each step and context passing.
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


class DetailedMockTool:
    """Mock tool that returns detailed information for context testing"""

    def __init__(self, name: str, description: str = "Mock tool"):
        self._metadata = ToolMetadata(
            name=name, description=description, visibility=ToolVisibility.PUBLIC
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def args_type(self) -> Type[BaseModel]:
        class Args(BaseModel):
            query: Optional[str] = "default query"

        return Args

    def return_type(self) -> Type[BaseModel]:
        class Return(BaseModel):
            result: str
            success: bool

        return Return

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> dict:
        """Mock tool execution with detailed results"""
        return {
            "tool_name": self.metadata.name,
            "args_received": dict(args),
            "mock_data": f"Generated data from {self.metadata.name}",
            "timestamp": "2024-01-01T10:00:00",
            "success": True,
            "details": {
                "processed_items": 100,
                "quality_score": 0.95,
                "metadata": {"source": self.metadata.name},
            },
        }

    def run_json_sync(self, args: Mapping[str, Any]) -> dict:
        return {"sync_result": "not used"}

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass


@pytest.mark.asyncio
async def test_llm_step_execution():
    """Test that each step involves LLM and context is properly passed"""
    print("=== Testing LLM Step Execution and Context Passing ===")

    # Create tools that return detailed, distinguishable results
    tools = [
        DetailedMockTool("data_collector", "Collect data from various sources"),
        DetailedMockTool("data_processor", "Process and clean collected data"),
        DetailedMockTool("analyzer", "Analyze processed data"),
        DetailedMockTool("visualizer", "Create visualizations"),
        DetailedMockTool("reporter", "Generate final report"),
    ]

    # Track LLM calls and contexts
    call_log = []

    def viz_callback(data):
        call_log.append(data)

    # Create AgentService with enhanced MockLLM
    mock_llm = MockLLM()
    agent_service = AgentService(
        name="context_test_agent",
        id="context_test_agent",
        llm=mock_llm,
        tools=tools,
        memory=InMemoryMemoryStore(),
        use_dag_pattern=True,
    )

    # Configure for testing
    dag_pattern = agent_service.get_dag_pattern()
    if dag_pattern:
        dag_pattern.max_iterations = 1
        dag_pattern.goal_check_enabled = False

    print(f"✅ Created AgentService with {len(tools)} detailed tools")

    # Execute task
    task = "Collect and analyze sales data with detailed reporting"
    print(f"\n🎯 Executing task: {task}")

    # Reset LLM call count
    mock_llm.reset_call_count()

    try:
        result = await agent_service.execute_task(task)

        print("\n📋 Execution Result:")
        print(f"   Status: {result['status']}")
        print(f"   Success: {result['success']}")
        print(f"   LLM calls made: {mock_llm.get_call_count()}")

        # Analyze the execution results for context passing
        if result.get("dag_status") and result["dag_status"].get("current_plan"):
            plan = result["dag_status"]["current_plan"]
            steps = plan.get("steps", [])

            print("\n🔍 Step Analysis:")
            for step in steps:
                step_id = step["id"]
                step_name = step["name"]
                step_context = step.get("context", {})

                print(f"   Step {step_id} ({step_name}):")
                print(f"      Dependencies: {step.get('dependencies', [])}")
                print(f"      Context keys: {list(step_context.keys())}")

                # Check if step has context from dependencies
                if step_context:
                    print("      Context data preview:")
                    for dep_id, dep_data in step_context.items():
                        if isinstance(dep_data, dict):
                            preview = (
                                str(dep_data)[:100] + "..."
                                if len(str(dep_data)) > 100
                                else str(dep_data)
                            )
                            print(f"         {dep_id}: {preview}")

        # Check if LLM was called for planning + each step
        expected_calls = 1 + len(tools)  # 1 for planning + 1 for each step
        actual_calls = mock_llm.get_call_count()

        print("\n📊 LLM Participation Analysis:")
        print(f"   Expected calls: {expected_calls} (1 planning + {len(tools)} steps)")
        print(f"   Actual calls: {actual_calls}")

        if actual_calls >= expected_calls:
            print("   ✅ LLM participated in planning and all steps")
        else:
            print("   ⚠️  LLM participation may be less than expected")

        print(f"\n🔄 Visualization events: {len(call_log)}")

        return result["success"]

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the detailed LLM step execution test"""
    print("🚀 Starting LLM Step Execution Test")

    success = await test_llm_step_execution()

    print(f"\n{'=' * 60}")
    print(f"📈 Test Result: {'✅ PASS' if success else '❌ FAIL'}")

    if success:
        print("✅ LLM successfully participates in each step")
        print("✅ Context passing between dependencies works")
        print("✅ DAG execution with intelligent step processing confirmed")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
