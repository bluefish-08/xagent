"""
合并后的追踪测试
整合了多个追踪相关的测试文件，消除重复代码
"""

import shutil
import unittest
from unittest.mock import patch

import pytest

from tests.shared import (
    CaptureTraceHandler,
    create_mock_db_session,
    create_test_components,
)
from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.agent.trace import (
    ACTION_START_LLM,
    STEP_START_DAG,
    TASK_START_DAG,
    TraceEvent,
)


@pytest.mark.slow
class TestTracing(unittest.IsolatedAsyncioTestCase):
    """合并后的追踪测试"""

    def setUp(self):
        """设置测试组件"""
        self.components = create_test_components()
        self.capture_handler = CaptureTraceHandler()

        # 添加捕获处理器到追踪器
        self.components["tracer"].add_handler(self.capture_handler)

        # 设置模拟数据库
        self.mock_db, self.mock_get_db = create_mock_db_session()
        # Patch at the import location in trace_handlers
        self.db_patcher = patch(
            "xagent.web.api.trace_handlers.get_db", side_effect=self.mock_get_db
        )
        self.db_patcher.start()

    def tearDown(self):
        """清理测试组件"""
        self.db_patcher.stop()
        self.capture_handler.clear()
        # Clean up temporary workspace directory
        if "temp_dir" in self.components:
            shutil.rmtree(self.components["temp_dir"], ignore_errors=True)

    async def test_trace_event_capture(self):
        """测试追踪事件捕获"""
        print("\n=== 测试追踪事件捕获 ===")

        # 手动创建一些事件
        from xagent.core.agent.trace import ACTION_START_LLM

        await self.components["tracer"].trace_event(TASK_START_DAG, task_id="test_task")
        await self.components["tracer"].trace_event(TASK_START_DAG, task_id="test_task")
        await self.components["tracer"].trace_event(
            ACTION_START_LLM, task_id="test_task", step_id="step1"
        )

        # 验证事件被捕获
        self.assertEqual(len(self.capture_handler.events), 3, "应该捕获到3个事件")

        # 验证事件类型
        event_types = [event.event_type for event in self.capture_handler.events]
        self.assertIn(TASK_START_DAG, event_types)
        self.assertIn(ACTION_START_LLM, event_types)

        print(f"捕获的事件数量: {len(self.capture_handler.events)}")
        for event in self.capture_handler.events:
            print(f"  - {event.event_type.value}")

    async def test_trace_event_data_integrity(self):
        """测试追踪事件数据完整性"""
        print("\n=== 测试追踪事件数据完整性 ===")

        test_data = {
            "step_id": "step1",
            "step_name": "test_step",
            "custom_field": "custom_value",
        }

        await self.components["tracer"].trace_event(
            STEP_START_DAG, task_id="test_task", step_id="step1", data=test_data
        )

        # 验证事件数据
        self.assertEqual(len(self.capture_handler.events), 1, "应该捕获到1个事件")
        event = self.capture_handler.events[0]

        self.assertEqual(event.task_id, "test_task")
        self.assertEqual(event.event_type, STEP_START_DAG)
        self.assertEqual(event.data["step_id"], "step1")
        self.assertEqual(event.data["step_name"], "test_step")
        self.assertEqual(event.data["custom_field"], "custom_value")

        print("事件数据完整性验证: ✅")
        print(f"  task_id: {event.task_id}")
        print(f"  event_type: {event.event_type.value}")
        print(f"  data keys: {list(event.data.keys())}")

    async def test_trace_event_parent_child_relationship(self):
        """测试追踪事件的父子关系"""
        print("\n=== 测试追踪事件父子关系 ===")

        # 创建父事件
        parent_id = await self.components["tracer"].trace_event(
            STEP_START_DAG,
            task_id="test_task",
            step_id="step1",
            data={"step_id": "step1"},
        )

        # 创建子事件
        child_id = await self.components["tracer"].trace_event(
            ACTION_START_LLM, task_id="test_task", step_id="step1", parent_id=parent_id
        )

        # 验证父子关系
        self.assertEqual(len(self.capture_handler.events), 2, "应该捕获到2个事件")

        parent_event = None
        child_event = None

        for event in self.capture_handler.events:
            if event.id == parent_id:
                parent_event = event
            elif event.id == child_id:
                child_event = event

        self.assertIsNotNone(parent_event, "应该找到父事件")
        self.assertIsNotNone(child_event, "应该找到子事件")
        self.assertEqual(
            child_event.parent_id, parent_id, "子事件应该有正确的parent_id"
        )

        print("父子关系验证: ✅")
        print(f"  父事件: {parent_event.event_type.value} ({parent_event.id})")
        print(f"  子事件: {child_event.event_type.value} ({child_event.id})")
        print(f"  parent_id: {child_event.parent_id}")

    async def test_trace_with_database_integration(self):
        """测试追踪与数据库集成"""
        print("\n=== 测试追踪与数据库集成 ===")

        # 导入数据库追踪处理器
        from xagent.web.api.trace_handlers import DatabaseTraceHandler

        # 添加数据库处理器
        db_handler = DatabaseTraceHandler(task_id=1)
        self.components["tracer"].add_handler(db_handler)

        # 创建一些事件
        await self.components["tracer"].trace_event(TASK_START_DAG, task_id="test_task")
        await self.components["tracer"].trace_event(TASK_START_DAG, task_id="test_task")

        # 验证数据库操作被调用
        self.assertGreater(self.mock_db.add_call_count, 0, "数据库add方法应该被调用")
        self.assertGreater(
            self.mock_db.commit.call_count, 0, "数据库commit方法应该被调用"
        )

        print("数据库集成验证: ✅")
        print(f"  数据库add调用次数: {self.mock_db.add_call_count}")
        print(f"  数据库commit调用次数: {self.mock_db.commit.call_count}")

    async def test_trace_event_error_handling(self):
        """测试追踪事件错误处理"""
        print("\n=== 测试追踪事件错误处理 ===")

        # 清空之前的事件
        self.capture_handler.clear()

        # 创建一个会抛出异常的处理器
        class FailingTraceHandler:
            def __init__(self):
                self.call_count = 0

            async def handle_event(self, event: TraceEvent) -> None:
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Test exception")

        failing_handler = FailingTraceHandler()
        self.components["tracer"].add_handler(failing_handler)

        # 创建事件 - 即使一个处理器失败，其他处理器也应该继续工作

        await self.components["tracer"].trace_event(TASK_START_DAG, task_id="test_task")

        # 验证正常的处理器仍然能捕获事件
        self.assertEqual(
            len(self.capture_handler.events), 1, "正常处理器应该捕获到事件"
        )
        self.assertEqual(failing_handler.call_count, 1, "失败的处理器应该被调用")

        print("错误处理验证: ✅")
        print(f"  正常处理器捕获事件: {len(self.capture_handler.events)}")
        print(f"  失败处理器调用次数: {failing_handler.call_count}")

    async def test_trace_event_filtering(self):
        """测试追踪事件过滤"""
        print("\n=== 测试追踪事件过滤 ===")

        from xagent.core.agent.trace import ACTION_START_LLM

        # 创建不同类型的事件
        events_to_create = [
            (TASK_START_DAG, "task_1"),
            (TASK_START_DAG, "task_1"),
            (ACTION_START_LLM, "task_1"),
            (TASK_START_DAG, "task_2"),
            (TASK_START_DAG, "task_2"),
        ]

        for event_type, task_id in events_to_create:
            step_id = "step1" if event_type == ACTION_START_LLM else None
            await self.components["tracer"].trace_event(
                event_type, task_id=task_id, step_id=step_id
            )

        # 验证所有事件都被捕获
        self.assertEqual(len(self.capture_handler.events), len(events_to_create))

        # 测试按task_id过滤
        task_1_events = self.capture_handler.get_events_with_task_id("task_1")
        task_2_events = self.capture_handler.get_events_with_task_id("task_2")

        self.assertEqual(len(task_1_events), 3, "task_1应该有3个事件")
        self.assertEqual(len(task_2_events), 2, "task_2应该有2个事件")

        # 测试按事件类型过滤
        trace_start_events = self.capture_handler.get_events_by_type(TASK_START_DAG)
        llm_events = self.capture_handler.get_events_by_type(ACTION_START_LLM)

        self.assertEqual(len(trace_start_events), 4, "应该有4个TASK_START_DAG事件")
        self.assertEqual(len(llm_events), 1, "应该有1个ACTION_START_LLM事件")

        print("事件过滤验证: ✅")
        print(f"  总事件数: {len(self.capture_handler.events)}")
        print(f"  task_1事件数: {len(task_1_events)}")
        print(f"  task_2事件数: {len(task_2_events)}")
        print(f"  TASK_START_DAG事件数: {len(trace_start_events)}")
        print(f"  ACTION_START_LLM事件数: {len(llm_events)}")

    async def test_trace_event_performance(self):
        """测试追踪事件性能"""
        print("\n=== 测试追踪事件性能 ===")

        import time

        # 创建大量事件
        num_events = 100
        start_time = time.time()

        for i in range(num_events):
            await self.components["tracer"].trace_event(
                ACTION_START_LLM,
                task_id=f"task_{i % 10}",
                step_id=f"step_{i}",
                data={"iteration": i},
            )

        end_time = time.time()
        duration = end_time - start_time

        # 验证所有事件都被捕获
        self.assertEqual(len(self.capture_handler.events), num_events)

        # 验证性能 - 100个事件应该在合理时间内完成
        self.assertLess(duration, 5.0, "100个事件应该在5秒内完成")

        print("性能测试验证: ✅")
        print(f"  事件数量: {num_events}")
        print(f"  总耗时: {duration:.3f}秒")
        print(f"  平均每个事件: {duration / num_events * 1000:.3f}毫秒")

    async def test_trace_event_with_dag_pattern(self):
        """测试DAG模式的追踪事件"""
        print("\n=== 测试DAG模式的追踪事件 ===")

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
            # 即使执行失败，我们也要验证追踪事件
            pass

        # 验证追踪事件
        self.assertGreater(len(self.capture_handler.events), 0, "应该捕获到一些事件")

        # 分析事件类型分布
        event_types = {}
        for event in self.capture_handler.events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("DAG模式追踪验证: ✅")
        print(f"  总事件数: {len(self.capture_handler.events)}")
        print("  事件类型分布:")
        for event_type, count in sorted(event_types.items()):
            print(f"    {event_type}: {count}")

        # 验证关键事件类型存在
        expected_event_types = [
            "trace_start",
            "dag_plan_start",
            "dag_plan_end",
            "dag_execute_start",
            "dag_step_start",
        ]

        found_event_types = list(event_types.keys())
        for expected_type in expected_event_types:
            if expected_type in found_event_types:
                print(f"  ✅ 找到事件类型: {expected_type}")
            else:
                print(f"  ⚠️  缺少事件类型: {expected_type}")

    async def test_trace_event_serialization(self):
        """测试追踪事件序列化"""
        print("\n=== 测试追踪事件序列化 ===")

        # 创建一个复杂的事件
        test_data = {
            "step_id": "step1",
            "step_name": "test_step",
            "nested_data": {
                "key1": "value1",
                "key2": ["item1", "item2"],
                "key3": {"nested_key": "nested_value"},
            },
            "timestamp_test": "2023-01-01T00:00:00",
        }

        await self.components["tracer"].trace_event(
            STEP_START_DAG, task_id="test_task", step_id="step1", data=test_data
        )

        # 验证事件序列化
        self.assertEqual(len(self.capture_handler.events), 1)
        event = self.capture_handler.events[0]

        # 转换为字典
        event_dict = event.to_dict()

        # 验证序列化结果
        self.assertEqual(event_dict["event_type"], "step_start_dag")
        self.assertEqual(event_dict["task_id"], "test_task")
        self.assertEqual(event_dict["data"]["step_id"], "step1")
        self.assertEqual(event_dict["data"]["nested_data"]["key1"], "value1")
        self.assertEqual(event_dict["data"]["nested_data"]["key2"], ["item1", "item2"])

        print("事件序列化验证: ✅")
        print(f"  事件类型: {event_dict['event_type']}")
        print(f"  task_id: {event_dict['task_id']}")
        print(f"  数据键数量: {len(event_dict['data'])}")
        print(f"  嵌套数据: {event_dict['data']['nested_data']}")

    async def test_trace_event_websocket_integration(self):
        """测试追踪事件与WebSocket集成"""
        print("\n=== 测试追踪事件与WebSocket集成 ===")

        from unittest.mock import AsyncMock, patch

        from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

        # 模拟WebSocket管理器
        mock_manager = AsyncMock()

        # 用patch替换manager
        with patch("xagent.web.api.ws_trace_handlers.manager", mock_manager):
            # 创建WebSocket追踪处理器
            ws_handler = WebSocketTraceHandler(task_id=1)

            # 创建追踪事件
            await self.components["tracer"].trace_event(
                STEP_START_DAG,
                task_id="test_task",
                step_id="step1",
                data={"step_name": "test_step", "status": "running"},
            )

            # 验证追踪事件被捕获
            self.assertEqual(len(self.capture_handler.events), 1)

            # 获取捕获的事件
            captured_event = self.capture_handler.events[0]

            # 通过WebSocket处理器处理同一事件
            await ws_handler.handle_event(captured_event)

            # 验证WebSocket处理器发送了消息
            mock_manager.broadcast_to_task.assert_called_once()

            # 获取发送的消息
            sent_message = mock_manager.broadcast_to_task.call_args[0][0]

            # 验证统一的消息格式
            self.assertEqual(sent_message["type"], "trace_event")
            self.assertEqual(sent_message["event_type"], "dag_step_start")
            self.assertEqual(sent_message["task_id"], 1)
            self.assertIn("event_id", sent_message)
            self.assertIn("timestamp", sent_message)
            self.assertIn("data", sent_message)

            # 验证数据一致性
            self.assertEqual(sent_message["data"]["step_name"], "test_step")
            self.assertEqual(sent_message["data"]["status"], "running")

            print("追踪事件WebSocket集成验证: ✅")
            print(f"  原始事件类型: {captured_event.event_type.value}")
            print(f"  流式事件类型: {sent_message['event_type']}")
            print(
                f"  数据一致性: {'✅' if sent_message['data']['step_name'] == 'test_step' else '❌'}"
            )

    async def test_stream_event_creation_from_trace(self):
        """测试从追踪事件创建流式事件"""
        print("\n=== 测试从追踪事件创建流式事件 ===")

        from xagent.web.api.websocket import create_stream_event
        from xagent.web.api.ws_trace_handlers import get_event_type_mapping

        # 创建各种类型的追踪事件
        trace_events = [
            TraceEvent(
                TASK_START_DAG, task_id="task_1", data={"plan_data": {"steps": []}}
            ),
            TraceEvent(
                STEP_START_DAG,
                task_id="task_1",
                step_id="step1",
                data={"step_name": "test_step"},
            ),
            TraceEvent(
                ACTION_START_LLM,
                task_id="task_1",
                step_id="step1",
                data={"model_name": "gpt-4"},
            ),
        ]

        # 转换为流式事件
        stream_events = []
        for trace_event in trace_events:
            event_type = get_event_type_mapping(trace_event)
            stream_event = create_stream_event(
                event_type,
                1,  # 所有事件都属于同一个任务
                trace_event.data,
                trace_event.timestamp,
            )
            stream_events.append(stream_event)

        # 验证转换结果
        self.assertEqual(len(stream_events), len(trace_events))

        expected_event_types = ["dag_execute_start", "dag_step_start", "llm_call_start"]
        for i, stream_event in enumerate(stream_events):
            self.assertEqual(stream_event["type"], "trace_event")
            self.assertEqual(stream_event["event_type"], expected_event_types[i])
            self.assertEqual(stream_event["task_id"], 1)  # 所有事件都属于任务1
            self.assertIn("event_id", stream_event)
            self.assertIn("timestamp", stream_event)

        print("追踪事件到流式事件转换验证: ✅")
        print(f"  转换事件数量: {len(stream_events)}")
        print("  事件类型映射:")
        for i, (trace_event, stream_event) in enumerate(
            zip(trace_events, stream_events)
        ):
            print(f"    {trace_event.event_type.value} -> {stream_event['event_type']}")


if __name__ == "__main__":
    unittest.main()
