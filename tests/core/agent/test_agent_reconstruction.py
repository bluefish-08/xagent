"""单元测试：Agent状态重建功能"""

from datetime import datetime
from typing import Any, Dict

import pytest

from tests.utils.mock_llm import MockLLM
from xagent.core.agent.pattern.dag_plan_execute.models import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe import Tool


class MockTool(Tool):
    """Mock tool for testing"""

    def __init__(self, name: str):
        super().__init__(name)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        return {"result": f"Mock {self.name} result"}


class TestAgentReconstruction:
    """测试Agent状态重建功能"""

    @pytest.fixture
    def mock_llm(self):
        """创建mock LLM"""
        return MockLLM()

    @pytest.fixture
    def mock_tool(self):
        """创建mock tool"""
        return MockTool("calculator")

    @pytest.fixture
    def agent_service(self, mock_llm, mock_tool):
        """创建AgentService实例"""
        return AgentService(
            name="test_agent",
            id="test_agent_id",
            llm=mock_llm,
            tools=[mock_tool],
            use_dag_pattern=True,
        )

    @pytest.fixture
    def sample_plan_state(self):
        """创建示例计划状态"""
        steps = [
            {
                "id": "step1",
                "name": "Calculate",
                "description": "Calculate something",
                "tool_name": "calculator",
                "tool_args": {"expression": "2+2"},
                "dependencies": [],
                "status": "completed",
                "result": {"result": "4"},
                "error": None,
                "error_type": None,
                "error_traceback": None,
                "context": {},
                "difficulty": "easy",
            },
            {
                "id": "step2",
                "name": "Analyze",
                "description": "Analyze results",
                "tool_name": "calculator",
                "tool_args": {"expression": "4*2"},
                "dependencies": ["step1"],
                "status": "pending",
                "result": None,
                "error": None,
                "error_type": None,
                "error_traceback": None,
                "context": {},
                "difficulty": "hard",
            },
        ]

        return {
            "id": "test_plan",
            "goal": "Test goal",
            "iteration": 1,
            "created_at": datetime.now().isoformat(),
            "steps": steps,
        }

    @pytest.fixture
    def sample_tracer_events(self):
        """创建示例tracer事件"""
        return [
            {
                "id": "event1",
                "event_type": "task_start_general",
                "task_id": "test_task",
                "step_id": None,
                "timestamp": datetime.now().timestamp(),
                "data": {"goal": "Test goal"},
                "parent_id": None,
            },
            {
                "id": "event2",
                "event_type": "step_end_dag",
                "task_id": "test_task",
                "step_id": "step1",
                "timestamp": datetime.now().timestamp(),
                "data": {"success": True, "result": "4"},
                "parent_id": "event1",
            },
            {
                "id": "event3",
                "event_type": "task_end_general",
                "task_id": "test_task",
                "step_id": None,
                "timestamp": datetime.now().timestamp(),
                "data": {"success": True, "result": "Task completed"},
                "parent_id": None,
            },
        ]

    @pytest.mark.asyncio
    async def test_reconstruct_from_history_basic(self, agent_service):
        """测试基本的重建功能"""
        task_id = "test_task"
        tracer_events = []
        plan_state = None

        # 执行重建
        await agent_service.reconstruct_from_history(task_id, tracer_events, plan_state)

        # 验证任务ID设置正确
        assert agent_service._current_task_id == task_id

    @pytest.mark.asyncio
    async def test_reconstruct_with_dag_pattern(
        self, agent_service, sample_plan_state, sample_tracer_events
    ):
        """测试重建DAG pattern状态"""
        task_id = "test_task"

        # 执行重建
        await agent_service.reconstruct_from_history(
            task_id, sample_tracer_events, sample_plan_state
        )

        # 验证任务ID设置正确
        assert agent_service._current_task_id == task_id

        # 验证DAG pattern重建
        dag_pattern = agent_service.get_dag_pattern()
        assert dag_pattern is not None
        assert dag_pattern.current_plan is not None
        assert dag_pattern.current_plan.id == sample_plan_state["id"]
        assert len(dag_pattern.current_plan.steps) == 2

        # 验证步骤状态
        step1 = dag_pattern.current_plan.get_step_by_id("step1")
        assert step1.status == StepStatus.COMPLETED
        assert step1.result == {"result": "4"}

        step2 = dag_pattern.current_plan.get_step_by_id("step2")
        assert step2.status == StepStatus.PENDING
        assert step2.dependencies == ["step1"]

    @pytest.mark.asyncio
    async def test_reconstruct_context_from_tracer_events(
        self, agent_service, sample_tracer_events
    ):
        """测试从tracer事件重建上下文"""
        task_id = "test_task"

        # 执行重建
        await agent_service.reconstruct_from_history(task_id, sample_tracer_events)

        # 验证内存中存储了成功的结果
        search_results = agent_service.memory.search("Task completed")
        assert len(search_results) >= 1
        assert any("Task completed" in note.content for note in search_results)

    @pytest.mark.asyncio
    async def test_reconstruct_execution_status(
        self, agent_service, sample_plan_state, sample_tracer_events
    ):
        """测试重建执行状态"""
        task_id = "test_task"

        # 执行重建
        await agent_service.reconstruct_from_history(
            task_id, sample_tracer_events, sample_plan_state
        )

        # 验证DAG pattern的执行状态
        dag_pattern = agent_service.get_dag_pattern()
        assert dag_pattern is not None

        # 应该有1个完成的步骤
        assert "step1" in dag_pattern.step_execution_results
        assert (
            dag_pattern.step_execution_results["step1"].final_result["status"]
            == "completed"
        )
        # 检查没有失败步骤
        failed_steps = [
            step_id
            for step_id, result in dag_pattern.step_execution_results.items()
            if result.final_result["status"] != "completed"
        ]
        assert len(failed_steps) == 0

    @pytest.mark.asyncio
    async def test_reconstruct_with_failed_steps(
        self, agent_service, sample_plan_state
    ):
        """测试重建包含失败步骤的状态"""
        task_id = "test_task"

        # 创建包含失败步骤的tracer事件
        tracer_events = [
            {
                "id": "event1",
                "event_type": "step_end_dag",
                "task_id": "test_task",
                "step_id": "step2",
                "timestamp": datetime.now().timestamp(),
                "data": {"success": False, "error": "Step failed"},
                "parent_id": None,
            }
        ]

        # 执行重建
        await agent_service.reconstruct_from_history(
            task_id, tracer_events, sample_plan_state
        )

        # 验证失败步骤被正确记录
        dag_pattern = agent_service.get_dag_pattern()
        assert dag_pattern is not None
        assert "step2" in dag_pattern.step_execution_results
        assert (
            dag_pattern.step_execution_results["step2"].final_result["status"]
            == "failed"
        )

    def test_get_reconstruction_data(self, agent_service, sample_plan_state):
        """测试获取重建数据"""
        # 设置DAG pattern
        dag_pattern = agent_service.get_dag_pattern()
        if dag_pattern:
            # 手动创建一个ExecutionPlan用于测试
            steps = []
            for step_data in sample_plan_state["steps"]:
                step = PlanStep(
                    id=step_data["id"],
                    name=step_data["name"],
                    description=step_data["description"],
                    tool_names=[step_data["tool_name"]],
                    dependencies=step_data.get("dependencies", []),
                    status=StepStatus(step_data["status"]),
                    result=step_data.get("result"),
                    error=step_data.get("error"),
                    error_type=step_data.get("error_type"),
                    error_traceback=step_data.get("error_traceback"),
                    context=step_data.get("context", {}),
                    difficulty=step_data.get("difficulty", "hard"),
                )
                steps.append(step)

            execution_plan = ExecutionPlan(
                id=sample_plan_state["id"],
                goal=sample_plan_state["goal"],
                iteration=sample_plan_state.get("iteration", 1),
                steps=steps,
            )
            dag_pattern.current_plan = execution_plan

        # 设置任务ID
        agent_service._current_task_id = "test_task"

        # 获取重建数据
        data = agent_service.get_reconstruction_data()

        # 验证数据结构
        assert data["task_id"] == "test_task"
        assert data["agent_name"] == "test_agent"
        assert data["patterns"] == 1
        assert "plan_state" in data
        assert "execution_status" in data

    @pytest.mark.asyncio
    async def test_reconstruct_error_handling(self, agent_service):
        """测试重建过程中的错误处理"""
        task_id = "test_task"

        # 创建无效的plan_state
        invalid_plan_state = {
            "id": "test_plan",
            "goal": "Test goal",
            "steps": [
                {
                    "id": "step1",
                    "name": "Invalid Step",
                    "description": "Missing required fields",
                    # 缺少必要的字段
                }
            ],
        }

        # 应该抛出异常
        with pytest.raises(Exception):
            await agent_service.reconstruct_from_history(
                task_id, [], invalid_plan_state
            )

    @pytest.mark.asyncio
    async def test_reconstruct_empty_data(self, agent_service):
        """测试使用空数据进行重建"""
        task_id = "test_task"

        # 使用空数据执行重建
        await agent_service.reconstruct_from_history(task_id, [], None)

        # 应该只设置任务ID，不抛出异常
        assert agent_service._current_task_id == task_id
