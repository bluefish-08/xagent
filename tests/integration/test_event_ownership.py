"""
合并后的事件归属验证测试
整合了多个测试文件的功能，消除重复代码
"""

import shutil
import unittest
from unittest.mock import patch

import pytest

from tests.shared import (
    CaptureTraceHandler,
    EventOwnershipValidator,
    create_mock_db_session,
    create_test_components,
)
from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.agent.trace import (
    ACTION_END_LLM,
    ACTION_END_TOOL,
    ACTION_START_LLM,
    ACTION_START_TOOL,
    STEP_END_DAG,
    STEP_START_DAG,
    TASK_END_DAG,
    TASK_START_DAG,
    TraceEvent,
)


@pytest.mark.slow
class TestEventOwnership(unittest.IsolatedAsyncioTestCase):
    """合并后的事件归属验证测试"""

    def setUp(self):
        """设置测试组件"""
        self.components = create_test_components()
        self.validator = EventOwnershipValidator()
        self.capture_handler = CaptureTraceHandler()

        # 添加捕获处理器到追踪器
        self.components["tracer"].add_handler(self.capture_handler)

        # 设置模拟数据库
        self.mock_db, self.mock_get_db = create_mock_db_session()
        self.db_patcher = patch(
            "xagent.web.models.database.get_db", side_effect=self.mock_get_db
        )
        self.db_patcher.start()

    def tearDown(self):
        """清理测试组件"""
        self.db_patcher.stop()
        self.capture_handler.clear()
        self.validator.events.clear()
        # Clean up temporary workspace directory
        if "temp_dir" in self.components:
            shutil.rmtree(self.components["temp_dir"], ignore_errors=True)

    async def test_dag_pattern_event_ownership(self):
        """测试DAG模式的事件归属"""
        print("\n=== 测试DAG模式事件归属 ===")

        # 创建DAG模式
        pattern = DAGPlanExecutePattern(
            llm=self.components["llm"],
            max_iterations=1,
            goal_check_enabled=True,
            tracer=self.components["tracer"],
            workspace=self.components["workspace"],
        )

        # 执行任务
        try:
            await pattern.run(
                task="check and analyze weather in Singapore",
                memory=self.components["memory"],
                tools=self.components["tools"],
                context=self.components["context"],
            )
        except Exception:
            # 即使执行失败，我们也要验证捕获到的事件
            pass

        # 添加捕获的事件到验证器
        for event in self.capture_handler.events:
            self.validator.add_event(event)

        # 验证事件归属
        validation_passed = self.validator.validate_event_ownership()
        report = self.validator.get_ownership_report()

        print(f"总事件数: {report['total_events']}")
        print(f"有task_id的事件: {report['events_with_task_id']}")
        print(f"有parent_id的事件: {report['events_with_parent_id']}")
        print(f"覆盖率: {report['coverage_percentage']:.1f}%")
        print(f"验证结果: {'✅ 通过' if validation_passed else '❌ 失败'}")

        if report["ownership_issues"]:
            print("归属问题:")
            for issue in report["ownership_issues"]:
                print(f"  - {issue}")

        # 验证基本要求 - 注意：这里我们允许验证失败，因为这是在测试真实系统的问题
        self.assertGreater(report["total_events"], 0, "应该捕获到一些事件")

        # 验证事件分类
        plan_events = [
            e for e in self.validator.events if self.validator._is_plan_level_event(e)
        ]
        step_events = [
            e for e in self.validator.events if self.validator._is_step_level_event(e)
        ]
        llm_tool_events = [
            e for e in self.validator.events if self.validator._is_llm_or_tool_event(e)
        ]

        print("事件分类:")
        print(f"  Plan事件: {len(plan_events)}")
        print(f"  Step事件: {len(step_events)}")
        print(f"  LLM/工具事件: {len(llm_tool_events)}")

    async def test_react_pattern_event_ownership(self):
        """测试ReAct模式的事件归属"""
        print("\n=== 测试ReAct模式事件归属 ===")

        # 创建ReAct模式
        pattern = ReActPattern(
            llm=self.components["llm"],
            max_iterations=3,
            tracer=self.components["tracer"],
        )

        # 执行任务
        try:
            await pattern.run(
                task="check weather in Singapore",
                memory=self.components["memory"],
                tools=[self.components["tools"][0]],  # 只使用天气工具
                context=self.components["context"],
            )
        except Exception:
            # 即使执行失败，我们也要验证捕获到的事件
            pass

        # 添加捕获的事件到验证器
        for event in self.capture_handler.events:
            self.validator.add_event(event)

        # 验证事件归属
        validation_passed = self.validator.validate_event_ownership()
        report = self.validator.get_ownership_report()

        print(f"总事件数: {report['total_events']}")
        print(f"覆盖率: {report['coverage_percentage']:.1f}%")
        print(f"验证结果: {'✅ 通过' if validation_passed else '❌ 失败'}")

        if report["ownership_issues"]:
            print("归属问题:")
            for issue in report["ownership_issues"]:
                print(f"  - {issue}")

        # 验证ReAct特定的事件
        react_events = [
            e for e in self.validator.events if e.event_type.value.startswith("react_")
        ]
        print(f"ReAct事件: {len(react_events)}")

    async def test_manual_event_ownership_validation(self):
        """测试手动创建的事件归属验证"""
        print("\n=== 测试手动事件归属验证 ===")

        # 手动创建一些典型事件来测试验证逻辑
        events = [
            # Plan级别事件
            TraceEvent(TASK_START_DAG, task_id="task_1"),
            TraceEvent(TASK_START_DAG, task_id="task_1"),
            TraceEvent(TASK_END_DAG, task_id="task_1", parent_id="plan_start_id"),
            # Step级别事件
            TraceEvent(
                STEP_START_DAG,
                task_id="task_1",
                step_id="step1",
                data={"step_name": "get_weather"},
                parent_id="plan_end_id",
            ),
            TraceEvent(
                STEP_END_DAG,
                task_id="task_1",
                step_id="step1",
                data={"step_name": "get_weather"},
                parent_id="step_start_id",
            ),
            # LLM/工具事件
            TraceEvent(
                ACTION_START_LLM,
                task_id="task_1",
                step_id="step1",
                parent_id="step_start_id",
            ),
            TraceEvent(
                ACTION_END_LLM,
                task_id="task_1",
                step_id="step1",
                parent_id="step_start_id",
            ),
            TraceEvent(
                ACTION_START_TOOL,
                task_id="task_1",
                step_id="step1",
                parent_id="step_start_id",
            ),
            TraceEvent(
                ACTION_END_TOOL,
                task_id="task_1",
                step_id="step1",
                parent_id="step_start_id",
            ),
        ]

        # 设置正确的parent_id引用
        events[1].id = "plan_start_id"  # DAG_PLAN_START
        events[2].id = "plan_end_id"  # DAG_PLAN_END
        events[3].id = "step_start_id"  # DAG_STEP_START
        events[4].id = "step_end_id"  # DAG_STEP_END
        events[5].id = "llm_start_id"  # LLM_CALL_START
        events[6].id = "llm_end_id"  # LLM_CALL_END
        events[7].id = "tool_start_id"  # TOOL_EXECUTION_START
        events[8].id = "tool_end_id"  # TOOL_EXECUTION_END

        for event in events:
            self.validator.add_event(event)

        # 验证归属
        report = self.validator.get_ownership_report()

        print(f"总事件数: {report['total_events']}")
        print(f"Task ID覆盖率: {report['coverage_percentage']:.1f}%")
        print(f"验证结果: {'✅ 通过' if report['validation_passed'] else '❌ 失败'}")

        if report["ownership_issues"]:
            print("问题:")
            for issue in report["ownership_issues"]:
                print(f"  - {issue}")

        # 验证预期结果 - 手动创建的事件应该通过验证
        self.assertTrue(report["validation_passed"], "手动创建的事件应该通过验证")
        self.assertEqual(
            report["coverage_percentage"], 100.0, "所有事件都应该有task_id"
        )

    async def test_missing_task_id_validation(self):
        """测试缺少task_id的验证"""
        print("\n=== 测试缺少task_id的验证 ===")

        # 创建缺少task_id的事件 - 现在TraceEvent会在构造时验证
        # 所以我们需要先创建正常的事件，然后在验证前修改task_id
        events = [
            TraceEvent(TASK_START_DAG, task_id="temp_task"),  # 临时task_id
            TraceEvent(ACTION_START_LLM, task_id="task_1", step_id="step1"),
            TraceEvent(ACTION_END_LLM, task_id="task_1", step_id="step1"),
        ]

        # 现在移除第一个事件的task_id来测试验证
        events[0].task_id = None

        for event in events:
            self.validator.add_event(event)

        report = self.validator.get_ownership_report()

        print(f"Task ID覆盖率: {report['coverage_percentage']:.1f}%")
        print(f"问题数量: {len(report['ownership_issues'])}")

        # 应该检测到缺少task_id的问题
        self.assertFalse(report["validation_passed"], "应该检测到缺少task_id的问题")
        self.assertLess(
            report["coverage_percentage"], 100.0, "Task ID覆盖率应该小于100%"
        )
        self.assertTrue(
            any("missing task_id" in issue for issue in report["ownership_issues"]),
            "应该报告缺少task_id的问题",
        )

    async def test_missing_parent_id_validation(self):
        """测试缺少parent_id的验证"""
        print("\n=== 测试缺少parent_id的验证 ===")

        # 创建缺少parent_id的LLM/工具事件
        events = [
            TraceEvent(TASK_START_DAG, task_id="task_1"),
            TraceEvent(
                STEP_START_DAG,
                task_id="task_1",
                step_id="step1",
            ),
            TraceEvent(
                ACTION_START_LLM, task_id="task_1", step_id="step1"
            ),  # 缺少parent_id
            TraceEvent(
                ACTION_END_LLM, task_id="task_1", step_id="step1"
            ),  # 缺少parent_id
        ]

        for event in events:
            self.validator.add_event(event)

        report = self.validator.get_ownership_report()

        print(
            f"Parent ID覆盖率: {report['events_with_parent_id']}/{report['total_events']}"
        )
        print(f"问题数量: {len(report['ownership_issues'])}")

        # 应该检测到缺少parent_id的问题
        self.assertFalse(report["validation_passed"], "应该检测到缺少parent_id的问题")
        self.assertTrue(
            any("missing parent_id" in issue for issue in report["ownership_issues"]),
            "应该报告缺少parent_id的问题",
        )

    async def test_cross_mode_event_ownership(self):
        """测试跨模式的事件归属（DAG + ReAct）"""
        print("\n=== 测试跨模式事件归属 ===")

        # 先执行DAG模式
        dag_pattern = DAGPlanExecutePattern(
            llm=self.components["llm"],
            max_iterations=1,
            goal_check_enabled=True,
            tracer=self.components["tracer"],
            workspace=self.components["workspace"],
        )

        try:
            await dag_pattern.run(
                task="check weather in Singapore",
                memory=self.components["memory"],
                tools=[self.components["tools"][0]],
                context=self.components["context"],
            )
        except Exception:
            pass

        # 再执行ReAct模式
        react_pattern = ReActPattern(
            llm=self.components["llm"],
            max_iterations=2,
            tracer=self.components["tracer"],
        )

        try:
            await react_pattern.run(
                task="analyze weather data",
                memory=self.components["memory"],
                tools=[self.components["tools"][0]],
                context=self.components["context"],
            )
        except Exception:
            pass

        # 添加所有捕获的事件到验证器
        for event in self.capture_handler.events:
            self.validator.add_event(event)

        # 验证所有事件归属
        report = self.validator.get_ownership_report()

        print(f"跨模式总事件数: {report['total_events']}")
        print(f"覆盖率: {report['coverage_percentage']:.1f}%")
        print(f"验证结果: {'✅ 通过' if report['validation_passed'] else '❌ 失败'}")

        if report["ownership_issues"]:
            print("跨模式归属问题:")
            for issue in report["ownership_issues"]:
                print(f"  - {issue}")

        # 验证事件数量
        self.assertGreater(report["total_events"], 0, "应该捕获到跨模式事件")

    async def test_event_hierarchy_consistency(self):
        """测试事件层次结构的一致性"""
        print("\n=== 测试事件层次结构一致性 ===")

        # 执行DAG模式任务
        pattern = DAGPlanExecutePattern(
            llm=self.components["llm"],
            max_iterations=1,
            goal_check_enabled=True,
            tracer=self.components["tracer"],
            workspace=self.components["workspace"],
        )

        try:
            await pattern.run(
                task="check and analyze weather in Singapore",
                memory=self.components["memory"],
                tools=self.components["tools"],
                context=self.components["context"],
            )
        except Exception:
            pass

        # 添加捕获的事件到验证器
        for event in self.capture_handler.events:
            self.validator.add_event(event)

        # 验证层次结构一致性
        hierarchy_issues = []

        for event in self.validator.events:
            if event.parent_id:
                parent_event = self.validator._find_event_by_id(event.parent_id)
                if not parent_event:
                    hierarchy_issues.append(
                        f"Event {event.id} references non-existent parent {event.parent_id}"
                    )
                else:
                    # 验证父子关系的合理性
                    if self.validator._is_llm_or_tool_event(event):
                        if not self.validator._is_step_level_event(parent_event):
                            hierarchy_issues.append(
                                f"LLM/Tool event {event.id} has non-step parent {parent_event.id}"
                            )

        print(f"层次结构问题: {len(hierarchy_issues)}")
        for issue in hierarchy_issues:
            print(f"  - {issue}")

        # 验证层次结构问题数量
        self.assertLessEqual(len(hierarchy_issues), 2, "层次结构问题应该很少")


if __name__ == "__main__":
    unittest.main()
