"""Tests for Langfuse tracer functionality."""

import asyncio
import tempfile
import time

import pytest

from tests.utils.langfuse_helpers import (
    verify_langfuse_messages_metadata,
    verify_langfuse_span_metadata,
)
from tests.utils.mock_helpers import (
    create_langfuse_mock,
    create_langfuse_span_mock,
    create_mock_message,
    create_temp_config_file,
)
from xagent.core.observability.langfuse_config import LangfuseConfig
from xagent.core.observability.langfuse_tracer import (
    LangfuseTracer,
    get_tracer,
    init_tracer,
    reset_tracer,
    trace_node,
    trace_tool_call,
)


def test_init_disabled_tracer(langfuse_tracer_reset):
    """Test creating tracer with disabled configuration."""
    config = LangfuseConfig(enabled=False)
    tracer = LangfuseTracer(config)

    assert tracer.config == config
    assert tracer.langfuse is None
    assert not tracer.is_enabled()


def test_init_enabled_tracer_no_keys(langfuse_tracer_reset):
    """Test creating tracer with enabled but no keys."""
    config = LangfuseConfig(enabled=True, public_key=None, secret_key=None)
    tracer = LangfuseTracer(config)

    assert tracer.config == config
    assert tracer.langfuse is None
    assert not tracer.is_enabled()


def test_init_enabled_tracer_with_keys(mocker, langfuse_tracer_reset):
    """Test creating tracer with enabled and valid keys."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)

    config = LangfuseConfig(
        enabled=True,
        public_key="test_public",
        secret_key="test_secret",
        host="https://custom.langfuse.com",
        debug=True,
        flush_at=20,
        flush_interval=1.0,
    )

    tracer = LangfuseTracer(config)

    assert tracer.config == config
    assert tracer.is_enabled()
    assert tracer.langfuse == mock_langfuse_instance

    # Verify Langfuse was initialized with correct parameters
    mock_langfuse_class.assert_called_once_with(
        public_key="test_public",
        secret_key="test_secret",
        host="https://custom.langfuse.com",
        debug=True,
        flush_at=20,
        flush_interval=1.0,
    )


def test_init_tracer_with_storage_root(temp_dir, langfuse_tracer_reset):
    """Test initializing tracer with storage root."""
    init_tracer(temp_dir)

    tracer = get_tracer()
    assert tracer is not None
    assert isinstance(tracer, LangfuseTracer)
    assert isinstance(tracer.config, LangfuseConfig)


def test_init_tracer_with_config_file(mocker, temp_dir, langfuse_tracer_reset):
    """Test initializing tracer with config file."""
    mock_langfuse_class, _ = create_langfuse_mock(mocker)

    # Create config file using helper
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)

    init_tracer(temp_dir)

    tracer = get_tracer()
    assert tracer is not None
    assert tracer.is_enabled()

    # Verify Langfuse was initialized with config file values
    mock_langfuse_class.assert_called_once_with(
        public_key="test_pub",
        secret_key="test_sec",
        host="https://cloud.langfuse.com",  # default
        debug=False,  # default
        flush_at=15,  # default
        flush_interval=0.5,  # default
    )


def test_get_tracer_before_init(langfuse_tracer_reset):
    """Test getting tracer before initialization."""
    tracer = get_tracer()
    assert tracer is None


def test_get_tracer_after_init(temp_dir, langfuse_tracer_reset):
    """Test getting tracer after initialization."""
    init_tracer(temp_dir)

    tracer = get_tracer()
    assert tracer is not None
    assert isinstance(tracer, LangfuseTracer)


@pytest.mark.asyncio
async def test_trace_node_no_tracer(langfuse_tracer_reset):
    """Test trace_node decorator when no tracer is initialized."""

    @trace_node("test_node", "agent")
    async def test_func(state):
        return {"result": "success"}

    result = await test_func({"input": "test"})
    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_trace_node_disabled_tracer(disabled_langfuse_config):
    """Test trace_node decorator with disabled tracer."""
    temp_dir, _ = disabled_langfuse_config

    @trace_node("test_node", "agent")
    async def test_func(state):
        return {"result": "success"}

    result = await test_func({"input": "test"})
    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_trace_node_with_enabled_tracer(mocker, temp_dir, langfuse_tracer_reset):
    """Test trace_node with enabled tracer captures execution details."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)
    mock_span = create_langfuse_span_mock(mocker, mock_langfuse_instance)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    @trace_node("test_node", "agent")
    async def test_func(state):
        await asyncio.sleep(0.1)  # Simulate some work
        # Return state with added message to test message tracking
        result = {"result": "success", "processed": len(state.get("messages", []))}
        if "messages" in state:
            result["messages"] = state["messages"] + [
                create_mock_message(mocker, "Response message", "ai")
            ]
        return result

    # Create mock messages to test message tracking
    mock_input_message = create_mock_message(mocker, "Input message", "human")

    start_time = time.time()
    result = await test_func({"messages": [mock_input_message]})
    execution_time = time.time() - start_time

    # Verify results
    assert result["result"] == "success"
    assert result["processed"] == 1
    assert len(result["messages"]) == 2  # Original + added message
    assert execution_time >= 0.1  # Should have taken at least 0.1 seconds

    # Verify span was created and updated
    mock_langfuse_instance.start_span.assert_called_once_with(name="agent_test_node")

    # Use helper to verify metadata
    metadata = verify_langfuse_span_metadata(
        mock_span,
        expected_node_id="test_node",
        expected_node_type="agent",
        check_execution_time=True,
    )

    # Verify messages tracking
    verify_langfuse_messages_metadata(
        metadata, expected_before_count=1, expected_after_count=2
    )


@pytest.mark.asyncio
async def test_trace_tool_call_captures_messages_before_and_after(
    mocker, temp_dir, langfuse_tracer_reset
):
    """Test trace_tool_call captures both input and output messages including tool results."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)
    mock_span = create_langfuse_span_mock(mocker, mock_langfuse_instance)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    @trace_tool_call("calculator")
    async def test_func(state):
        await asyncio.sleep(0.05)  # Simulate some work

        # Simulate tool execution that adds ToolMessage to state
        tool_result_message = create_mock_message(
            mocker,
            "Calculation result: 42",
            "tool",
            tool_call_id="call_123",
            name="calculator",
        )

        # Return updated state with tool result
        new_messages = state.get("messages", []) + [tool_result_message]
        return {"messages": new_messages, "result": "calculated"}

    # Create mock input messages including AI message with tool calls
    human_message = create_mock_message(mocker, "Calculate 2+2", "human")
    ai_message = create_mock_message(mocker, "I'll calculate that", "ai")
    ai_message.tool_calls = [
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_123"}
    ]

    # Execute the function
    result = await test_func({"messages": [human_message, ai_message]})

    # Verify results
    assert result == {
        "messages": [human_message, ai_message, mocker.ANY],
        "result": "calculated",
    }
    assert len(result["messages"]) == 3

    # Verify span was created and updated
    mock_langfuse_instance.start_span.assert_called_once_with(name="tool_calculator")
    mock_span.update.assert_called_once()
    mock_span.end.assert_called_once()

    # Get the metadata that was passed to span.update
    call_args = mock_span.update.call_args
    metadata = call_args[1]["metadata"]

    # Verify basic metadata
    assert "execution_time_seconds" in metadata
    assert metadata["execution_time_seconds"] >= 0.05
    assert metadata["id"] == "calculator"

    # Verify messages_before contains input messages
    assert "messages_before" in metadata
    assert metadata["messages_before_count"] == 2
    assert len(metadata["messages_before"]) == 2

    # Check human message
    assert metadata["messages_before"][0]["type"] == "HumanMessage"
    assert metadata["messages_before"][0]["content"] == "Calculate 2+2"

    # Check AI message with tool calls
    assert metadata["messages_before"][1]["type"] == "AiMessage"
    assert metadata["messages_before"][1]["content"] == "I'll calculate that"
    assert "tool_calls" in metadata["messages_before"][1]
    assert len(metadata["messages_before"][1]["tool_calls"]) == 1
    assert metadata["messages_before"][1]["tool_calls"][0]["name"] == "calculator"

    # Verify messages_after contains all messages including tool result
    assert "messages_after" in metadata
    assert metadata["messages_after_count"] == 3
    assert len(metadata["messages_after"]) == 3

    # Check that the tool result message is captured
    tool_result_msg = metadata["messages_after"][2]
    assert tool_result_msg["type"] == "ToolMessage"
    assert tool_result_msg["content"] == "Calculation result: 42"
    assert tool_result_msg["tool_call_id"] == "call_123"
    assert tool_result_msg["tool_name"] == "calculator"


@pytest.mark.asyncio
async def test_trace_tool_call_with_tool_call_metadata(mocker):
    """Test trace_tool_call captures tool call metadata."""
    # Mock Langfuse
    mock_langfuse_class = mocker.patch(
        "xagent.core.observability.langfuse_tracer.Langfuse"
    )
    mock_langfuse_instance = mocker.Mock()
    mock_langfuse_class.return_value = mock_langfuse_instance

    # Mock span
    mock_span = mocker.Mock()
    mock_langfuse_instance.start_span.return_value = mock_span

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config with enabled tracing
        config_path = temp_dir + "/langfuse_config.json"
        with open(config_path, "w") as f:
            f.write(
                '{"enabled": true, "public_key": "test_pub", "secret_key": "test_sec"}'
            )

        init_tracer(temp_dir)

        @trace_tool_call("calculator")
        async def test_func(state):
            await asyncio.sleep(0.05)  # Simulate some work
            return {"result": "calculated", "value": 42}

        # Create mock message with tool calls
        mock_message = mocker.Mock()
        mock_message.tool_calls = [
            {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1"},
            {"name": "other_tool", "args": {"param": "value"}, "id": "call_2"},
        ]

        # Execute the function
        import asyncio

        result = await test_func({"messages": [mock_message]})

        # Verify results
        assert result == {"result": "calculated", "value": 42}

        # Verify span was created and updated
        mock_langfuse_instance.start_span.assert_called_once_with(
            name="tool_calculator"
        )
        mock_span.update.assert_called_once()
        mock_span.end.assert_called_once()

        # Verify execution time metadata was reported
        call_args = mock_span.update.call_args
        metadata = call_args[1]["metadata"]
        assert "execution_time_seconds" in metadata
        assert metadata["execution_time_seconds"] >= 0.05
        assert metadata["id"] == "calculator"
        assert "tool_calls" in metadata
        assert metadata["tool_call_count"] == 2


@pytest.mark.asyncio
async def test_trace_node_with_exception(mocker, temp_dir, langfuse_tracer_reset):
    """Test trace_node decorator handles exceptions properly and reports execution time."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)
    mock_span = create_langfuse_span_mock(mocker, mock_langfuse_instance)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    @trace_node("test_node", "agent")
    async def test_func(state):
        await asyncio.sleep(0.02)  # Simulate some work before error
        raise ValueError("Test error")

    # Create mock message to test message tracking in exception case
    mock_input_message = create_mock_message(
        mocker, "Input message before error", "human"
    )

    with pytest.raises(ValueError, match="Test error"):
        await test_func({"messages": [mock_input_message], "input": "test"})

    # Verify span was created and updated with error
    mock_langfuse_instance.start_span.assert_called_once_with(name="agent_test_node")

    # Use helper to verify metadata including error
    metadata = verify_langfuse_span_metadata(
        mock_span,
        expected_node_id="test_node",
        expected_node_type="agent",
        check_execution_time=True,
        check_error=True,
    )

    # Verify messages tracking in exception case
    verify_langfuse_messages_metadata(metadata, expected_before_count=1)


@pytest.mark.asyncio
async def test_trace_tool_call_with_exception():
    """Test trace_tool_call decorator handles exceptions properly."""

    @trace_tool_call("calculator")
    async def test_func(state):
        raise RuntimeError("Calculation failed")

    with pytest.raises(RuntimeError, match="Calculation failed"):
        await test_func({"input": "test"})


@pytest.mark.asyncio
async def test_trace_node_preserves_function_metadata():
    """Test that trace_node preserves function metadata."""

    @trace_node("test_node", "agent")
    async def test_func(state):
        """Test function docstring."""
        return {"result": "success"}

    # Check that functools.wraps preserved the metadata
    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test function docstring."


@pytest.mark.asyncio
async def test_trace_tool_call_preserves_function_metadata():
    """Test that trace_tool_call preserves function metadata."""

    @trace_tool_call("calculator")
    async def test_func(state):
        """Test tool function docstring."""
        return {"result": "calculated"}

    # Check that functools.wraps preserved the metadata
    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test tool function docstring."


def test_tracer_initialization_with_different_configs(mocker):
    """Test tracer initialization with various configuration scenarios."""
    mock_langfuse_class = mocker.patch(
        "xagent.core.observability.langfuse_tracer.Langfuse"
    )

    # Test 1: Config with all custom values
    config1 = LangfuseConfig(
        enabled=True,
        public_key="custom_pub",
        secret_key="custom_sec",
        host="https://custom.host.com",
        debug=True,
        flush_at=25,
        flush_interval=2.0,
    )

    tracer1 = LangfuseTracer(config1)
    assert tracer1.is_enabled()

    mock_langfuse_class.assert_called_with(
        public_key="custom_pub",
        secret_key="custom_sec",
        host="https://custom.host.com",
        debug=True,
        flush_at=25,
        flush_interval=2.0,
    )

    # Test 2: Config with minimal values
    mock_langfuse_class.reset_mock()
    config2 = LangfuseConfig(enabled=True, public_key="min_pub", secret_key="min_sec")

    tracer2 = LangfuseTracer(config2)
    assert tracer2.is_enabled()

    mock_langfuse_class.assert_called_with(
        public_key="min_pub",
        secret_key="min_sec",
        host="https://cloud.langfuse.com",  # default
        debug=False,  # default
        flush_at=15,  # default
        flush_interval=0.5,  # default
    )


def test_thread_safe_tracer_initialization():
    """Test that tracer initialization is thread-safe."""
    import tempfile
    import threading

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config file
        config_path = temp_dir + "/langfuse_config.json"
        with open(config_path, "w") as f:
            f.write(
                '{"enabled": true, "public_key": "test_pub", "secret_key": "test_sec"}'
            )

        # Track results from different threads
        results = []

        def init_in_thread():
            init_tracer(temp_dir)
            tracer = get_tracer()
            results.append(tracer)

        # Create multiple threads that try to initialize the tracer
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=init_in_thread)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should get the same tracer instance
        assert len(results) == 5
        assert all(tracer is results[0] for tracer in results)
        assert all(tracer is not None for tracer in results)


def test_thread_safe_tracer_reset():
    """Test that tracer reset is thread-safe."""
    import tempfile
    import threading

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize tracer first
        init_tracer(temp_dir)
        initial_tracer = get_tracer()
        assert initial_tracer is not None

        # Track results from different threads
        results = []

        def reset_and_get():
            reset_tracer()
            tracer = get_tracer()
            results.append(tracer)

        # Create multiple threads that try to reset the tracer
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=reset_and_get)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should get None (tracer reset)
        assert len(results) == 5
        assert all(tracer is None for tracer in results)


@pytest.mark.asyncio
async def test_trace_node_captures_tool_calls_in_messages(
    mocker, temp_dir, langfuse_tracer_reset
):
    """Test that trace_node captures tool_calls information in AI messages."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)
    mock_span = create_langfuse_span_mock(mocker, mock_langfuse_instance)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    @trace_node("test_node", "agent")
    async def test_func(state):
        # Simulate adding an AI message with tool calls
        ai_message = create_mock_message(mocker, "I'll use the calculator", "ai")
        ai_message.tool_calls = [
            {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_123"},
            {"name": "weather", "args": {"location": "NYC"}, "id": "call_456"},
        ]

        new_messages = state.get("messages", []) + [ai_message]
        return {"messages": new_messages, "result": "done"}

    # Create mock input message
    input_message = create_mock_message(
        mocker, "What is 2+2 and weather in NYC?", "human"
    )

    result = await test_func({"messages": [input_message]})

    # Verify results
    assert result["result"] == "done"
    assert len(result["messages"]) == 2

    # Verify span was created and updated
    mock_langfuse_instance.start_span.assert_called_once_with(name="agent_test_node")
    mock_span.update.assert_called_once()
    mock_span.end.assert_called_once()

    # Get the metadata that was passed to span.update
    call_args = mock_span.update.call_args
    metadata = call_args[1]["metadata"]

    # Verify messages_before contains input message without tool_calls
    assert "messages_before" in metadata
    assert len(metadata["messages_before"]) == 1
    assert metadata["messages_before"][0]["type"] == "HumanMessage"
    assert (
        metadata["messages_before"][0]["content"] == "What is 2+2 and weather in NYC?"
    )
    # Human messages should not have tool_calls in the serialized data
    assert "tool_calls" not in metadata["messages_before"][0]

    # Verify messages_after contains both input and AI messages
    assert "messages_after" in metadata
    assert len(metadata["messages_after"]) == 2

    # Check the AI message in messages_after has tool_calls
    ai_msg_data = metadata["messages_after"][1]  # Second message should be AI
    assert (
        ai_msg_data["type"] == "AiMessage"
    )  # Note: mock creates "AiMessage" not "AIMessage"
    assert ai_msg_data["content"] == "I'll use the calculator"
    assert "tool_calls" in ai_msg_data, (
        "AI message should include tool_calls information"
    )
    assert len(ai_msg_data["tool_calls"]) == 2
    assert ai_msg_data["tool_calls"][0]["name"] == "calculator"
    assert ai_msg_data["tool_calls"][0]["args"]["expression"] == "2+2"
    assert ai_msg_data["tool_calls"][1]["name"] == "weather"
    assert ai_msg_data["tool_calls"][1]["args"]["location"] == "NYC"


@pytest.mark.asyncio
async def test_trace_node_captures_tool_result_messages(
    mocker, temp_dir, langfuse_tracer_reset
):
    """Test that trace_node captures ToolMessage results including success and failures."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)
    mock_span = create_langfuse_span_mock(mocker, mock_langfuse_instance)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    @trace_node("test_node", "agent")
    async def test_func(state):
        # Simulate adding tool result messages (both success and failure)
        success_tool_message = create_mock_message(
            mocker,
            "Tool executed successfully: result=42",
            "tool",
            tool_call_id="call_123",
            name="calculator",
        )
        failure_tool_message = create_mock_message(
            mocker,
            "Tool execution failed: invalid input",
            "tool",
            tool_call_id="call_456",
            name="search",
        )

        new_messages = state.get("messages", []) + [
            success_tool_message,
            failure_tool_message,
        ]
        return {"messages": new_messages, "result": "done"}

    # Create mock input messages
    human_message = create_mock_message(
        mocker, "Calculate 2+2 and search for info", "human"
    )
    ai_message = create_mock_message(mocker, "I'll use tools to help", "ai")
    ai_message.tool_calls = [
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_123"},
        {"name": "search", "args": {"query": "info"}, "id": "call_456"},
    ]

    result = await test_func({"messages": [human_message, ai_message]})

    # Verify results
    assert result["result"] == "done"
    assert len(result["messages"]) == 4  # human + ai + 2 tool results

    # Verify span was created and updated
    mock_langfuse_instance.start_span.assert_called_once_with(name="agent_test_node")
    mock_span.update.assert_called_once()
    mock_span.end.assert_called_once()

    # Get the metadata that was passed to span.update
    call_args = mock_span.update.call_args
    metadata = call_args[1]["metadata"]

    # Verify messages_before contains human and AI messages
    assert "messages_before" in metadata
    assert len(metadata["messages_before"]) == 2
    assert metadata["messages_before"][0]["type"] == "HumanMessage"
    assert metadata["messages_before"][1]["type"] == "AiMessage"
    # AI message should have tool_calls
    assert "tool_calls" in metadata["messages_before"][1]
    assert len(metadata["messages_before"][1]["tool_calls"]) == 2

    # Verify messages_after contains all messages including tool results
    assert "messages_after" in metadata
    assert len(metadata["messages_after"]) == 4

    # Check the successful tool result message
    success_msg_data = metadata["messages_after"][2]
    assert success_msg_data["type"] == "ToolMessage"
    assert success_msg_data["content"] == "Tool executed successfully: result=42"
    assert success_msg_data["tool_call_id"] == "call_123"
    assert success_msg_data["tool_name"] == "calculator"

    # Check the failed tool result message
    failure_msg_data = metadata["messages_after"][3]
    assert failure_msg_data["type"] == "ToolMessage"
    assert failure_msg_data["content"] == "Tool execution failed: invalid input"
    assert failure_msg_data["tool_call_id"] == "call_456"
    assert failure_msg_data["tool_name"] == "search"


def test_traced_graph_serialize_messages_includes_tool_results(
    mocker, temp_dir, langfuse_tracer_reset
):
    """Test that TracedGraph._serialize_messages includes tool results and tool_calls information."""
    mock_langfuse_class, mock_langfuse_instance = create_langfuse_mock(mocker)

    # Setup tracer with config after mocking
    config_data = {"enabled": True, "public_key": "test_pub", "secret_key": "test_sec"}
    create_temp_config_file(temp_dir, config_data)
    init_tracer(temp_dir)

    # Create a mock compiled graph
    mock_compiled_graph = mocker.Mock()

    # Create TracedGraph instance
    from xagent.core.observability.langfuse_tracer import TracedGraph

    traced_graph = TracedGraph(mock_compiled_graph, "test_graph")

    # Create test messages
    human_message = create_mock_message(mocker, "Search for test information", "human")
    ai_message = create_mock_message(mocker, "Let me search for that", "ai")
    ai_message.tool_calls = [
        {"name": "search", "args": {"query": "test"}, "id": "search_123"}
    ]

    # Create tool result messages (both success and failure)
    success_tool_message = create_mock_message(
        mocker,
        "Search completed: Found 5 results",
        "tool",
        tool_call_id="search_123",
        name="search",
    )
    failure_tool_message = create_mock_message(
        mocker,
        "Search failed: API timeout",
        "tool",
        tool_call_id="search_456",
        name="search",
    )

    test_data = {
        "messages": [
            human_message,
            ai_message,
            success_tool_message,
            failure_tool_message,
        ]
    }

    # Test the _serialize_messages method
    serialized = traced_graph._serialize_messages(test_data)

    # Verify the result
    assert len(serialized) == 4

    # Check human message (should not have tool_calls or tool info)
    human_msg_data = serialized[0]
    assert human_msg_data["type"] == "HumanMessage"
    assert human_msg_data["content"] == "Search for test information"
    assert "tool_calls" not in human_msg_data
    assert "tool_call_id" not in human_msg_data

    # Check AI message (should have tool_calls)
    ai_msg_data = serialized[1]
    assert (
        ai_msg_data["type"] == "AiMessage"
    )  # Note: mock creates "AiMessage" not "AIMessage"
    assert ai_msg_data["content"] == "Let me search for that"
    assert "tool_calls" in ai_msg_data, (
        "AI message should include tool_calls information"
    )
    assert len(ai_msg_data["tool_calls"]) == 1
    assert ai_msg_data["tool_calls"][0]["name"] == "search"
    assert ai_msg_data["tool_calls"][0]["args"]["query"] == "test"
    assert ai_msg_data["tool_calls"][0]["id"] == "search_123"

    # Check successful tool result message
    success_tool_msg_data = serialized[2]
    assert success_tool_msg_data["type"] == "ToolMessage"
    assert success_tool_msg_data["content"] == "Search completed: Found 5 results"
    assert success_tool_msg_data["tool_call_id"] == "search_123"
    assert success_tool_msg_data["tool_name"] == "search"

    # Check failed tool result message
    failure_tool_msg_data = serialized[3]
    assert failure_tool_msg_data["type"] == "ToolMessage"
    assert failure_tool_msg_data["content"] == "Search failed: API timeout"
    assert failure_tool_msg_data["tool_call_id"] == "search_456"
    assert failure_tool_msg_data["tool_name"] == "search"
