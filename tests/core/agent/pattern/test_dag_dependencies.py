#!/usr/bin/env python3
"""
Test script to verify that DAG plan-execute steps receive dependency results.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from xagent.core.agent.pattern.dag_plan_execute.models import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from xagent.core.agent.trace import Tracer
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata


class MockDAGLLM(BaseLLM):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []
        self._model_name = "mock_dag_llm"

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
        # Record the call with complete messages
        self.call_history.append(
            {
                "call_number": self.call_count + 1,
                "messages": messages.copy(),
                "kwargs": kwargs,
                "timestamp": datetime.now(),
            }
        )

        print(f"\n=== LLM Call {self.call_count + 1} ===")
        print(f"Messages count: {len(messages)}")

        # Show message progression
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = (
                msg["content"][:150] + "..."
                if len(msg["content"]) > 150
                else msg["content"]
            )
            print(f"  {i + 1}. [{role}] {content}")

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            print(f"Response: {response[:100]}...")
            return response

        response = '{"type": "final_answer", "content": "Task completed successfully"}'
        print(f"Response: {response}")
        return response


class MockSearchTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="web_search", description="Web search tool")

    def args_type(self):
        return dict

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: Dict[str, Any]) -> Any:
        query = args.get("query", "")
        return {
            "results": [
                {
                    "title": "AI News 1",
                    "snippet": "Recent breakthrough in AI technology...",
                },
                {
                    "title": "AI News 2",
                    "snippet": "New AI model achieves state-of-the-art performance...",
                },
            ],
            "query": query,
        }

    def run_json_sync(self, args: Dict[str, Any]) -> Any:
        return {"result": "Search completed"}

    async def save_state_json(self):
        return {}

    async def load_state_json(self, state: Dict[str, Any]):
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


async def test_dag_step_dependencies():
    """Test that DAG steps receive dependency results."""
    print("Testing DAG step dependency result passing...")

    # Create a plan with dependent steps
    plan = ExecutionPlan(
        id="test_plan",
        goal="Test plan with dependencies",
        steps=[
            PlanStep(
                id="step1",
                name="搜索AI新闻",
                description="搜索最近一个月的AI新闻",
                tool_names=["web_search"],
                dependencies=[],
                status=StepStatus.PENDING,
            ),
            PlanStep(
                id="step2",
                name="整理分析搜索结果",
                description="分析和整理搜索到的AI新闻，提取重要信息",
                tool_names=["analysis"],
                dependencies=["step1"],
                status=StepStatus.PENDING,
            ),
        ],
    )

    # Create mock LLM with responses for both steps
    responses = [
        # Step 1 response (web search)
        '{"type": "action", "content": "Searching for AI news", "action": {"tool": "web_search", "parameters": {"query": "最近一个月的AI新闻"}}}',
        # Step 2 response (analysis) - this should receive the search results
        '{"type": "final_answer", "content": "基于搜索结果，我发现了以下重要的AI新闻：\\n1. AI技术取得重大突破\\n2. 新AI模型达到最先进性能"}',
    ]

    llm = MockDAGLLM(responses)
    memory = DummyMemoryStore()
    tracer = Tracer()

    # Create a simple workspace
    from xagent.core.workspace import TaskWorkspace

    workspace = TaskWorkspace(id="test_task")

    # Create plan executor
    from xagent.core.agent.pattern.dag_plan_execute.plan_executor import PlanExecutor

    executor = PlanExecutor(
        llm=llm, tracer=tracer, workspace=workspace, memory_store=memory
    )

    # Create tools
    tools = [MockSearchTool()]
    tool_map = {tool.metadata.name: tool for tool in tools}

    # Execute the plan
    result = await executor.execute_plan(plan, tool_map)

    print("\n=== Plan Execution Result ===")
    print(f"Plan execution completed with {len(result)} step results")

    print("\n=== LLM Call Analysis ===")
    print(f"Total LLM calls: {len(llm.call_history)}")

    # Verify that step2 received step1's results
    step2_call = None
    for call in llm.call_history:
        # Look for the call that has the analysis context
        for msg in call["messages"]:
            if "整理分析搜索结果" in msg.get("content", ""):
                step2_call = call
                break
        if step2_call:
            break

    if step2_call:
        print(f"\n✓ Found step2 LLM call with {len(step2_call['messages'])} messages")

        # Check if dependency results are included
        has_dependency_context = False
        for msg in step2_call["messages"]:
            content = msg.get("content", "")
            if "Previous Step Results" in content or "搜索AI新闻" in content:
                has_dependency_context = True
                print("✓ Found dependency context in message:")
                print(f"  Role: {msg['role']}")
                print(f"  Content preview: {content[:200]}...")
                break

        if has_dependency_context:
            print("✓ Step2 successfully received step1's results as context")
        else:
            print("✗ Step2 did not receive step1's results")

        # Show all messages for step2
        print("\n--- Step2 Messages ---")
        for i, msg in enumerate(step2_call["messages"]):
            role = msg["role"]
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            print(f"  {i + 1}. [{role}] {content}")
    else:
        print("✗ Could not find step2 LLM call")

    # Check step results
    print("\n=== Step Results ===")
    for step_result in result:
        step_id = step_result.get("step_id")
        status = step_result.get("status")
        print(f"Step {step_id}: {status}")

        if status == "completed":
            step_data = step_result.get("result", {})
            if "output" in step_data:
                print(f"  Output: {step_data['output'][:100]}...")


if __name__ == "__main__":
    asyncio.run(test_dag_step_dependencies())
