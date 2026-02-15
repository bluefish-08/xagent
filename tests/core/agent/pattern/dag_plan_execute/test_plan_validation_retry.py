"""
Test DAG plan generation retry mechanism for tool validation failures.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from xagent.core.agent.exceptions import DAGPlanGenerationError
from xagent.core.agent.pattern.dag_plan_execute.plan_generator import PlanGenerator


@pytest.fixture
def mock_llm():
    """Create a mock LLM with proper async generator support"""

    # Create the mock instance first
    mock_llm_instance = AsyncMock()
    mock_llm_instance.chat = AsyncMock()

    async def mock_stream_chat(**kwargs):
        """Mock stream_chat that yields a single chunk"""
        from xagent.core.model.chat.types import ChunkType, StreamChunk

        # Get the response from chat mock
        chat_result = mock_llm_instance.chat(**kwargs)
        # Handle both coroutines and direct values
        if hasattr(chat_result, "__await__"):
            response = await chat_result
        else:
            response = chat_result

        content = (
            response.get("content", "") if isinstance(response, dict) else response
        )

        yield StreamChunk(
            type=ChunkType.TOKEN,
            content=content,
            delta=content,
        )

    mock_llm_instance.stream_chat = mock_stream_chat
    return mock_llm_instance


@pytest.fixture
def mock_tracer():
    """Create a mock tracer"""
    tracer = MagicMock()
    tracer.trace_event = AsyncMock()
    return tracer


@pytest.fixture
def mock_tool():
    """Create a mock tool"""
    tool = MagicMock()
    tool.metadata.name = "write_file"
    tool.name = "write_file"
    tool.metadata.description = "Write content to a file"
    return tool


@pytest.mark.asyncio
async def test_plan_validation_retry_success(mock_llm, mock_tracer, mock_tool):
    """Test that plan generation retries successfully when tool validation fails initially"""

    # First response has invalid tool, second response has valid tool
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_names == ["write_file"]
    assert step.name == "Test Step"
    assert step.description == "A test step with valid tool"

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2

    # Verify the retry call included error context
    retry_call = mock_llm.chat.call_args_list[1]
    retry_messages = retry_call[1]["messages"]
    retry_user_message = retry_messages[-1]["content"]

    assert "PREVIOUS ERROR INFORMATION" in retry_user_message
    assert "Error Type: DAGPlanGenerationError" in retry_user_message
    assert "nonexistent_tool" in retry_user_message
    assert "write_file" in retry_user_message
    assert "Available Tools" in retry_user_message


@pytest.mark.asyncio
async def test_plan_validation_max_retries_exhausted(mock_llm, mock_tracer, mock_tool):
    """Test that plan generation fails after max retries when validation keeps failing"""

    # Always return invalid tool
    invalid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": invalid_response}

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should fail after max retries
    with pytest.raises(DAGPlanGenerationError) as exc_info:
        await plan_generator.generate_plan(
            goal="Test goal",
            tools=tools,
            iteration=1,
            history=[],
            tracer=mock_tracer,
            context=None,
        )

    # Verify error message
    assert "Generated plan references non-existent tools" in str(exc_info.value)

    # Verify LLM was called 3 times (initial + 2 retries)
    assert mock_llm.chat.call_count == 3

    # Verify all retry attempts included error context
    for i in range(1, 3):  # Retry calls at indices 1 and 2
        retry_call = mock_llm.chat.call_args_list[i]
        retry_messages = retry_call[1]["messages"]
        retry_user_message = retry_messages[-1]["content"]

        assert "PREVIOUS ERROR INFORMATION" in retry_user_message
        assert "Error Type: DAGPlanGenerationError" in retry_user_message


@pytest.mark.asyncio
async def test_plan_validation_no_retry_on_first_success(
    mock_llm, mock_tracer, mock_tool
):
    """Test that no retry happens when first plan validation succeeds"""

    valid_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "description": "A test step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.return_value = {"content": valid_response}

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed without retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_names == ["write_file"]

    # Verify LLM was called only once (no retry)
    assert mock_llm.chat.call_count == 1


@pytest.mark.asyncio
async def test_plan_extension_retry_success(mock_llm, mock_tracer, mock_tool):
    """Test retry mechanism for plan extension"""

    # Create current plan
    from xagent.core.agent.pattern.dag_plan_execute.models import (
        ExecutionPlan,
        PlanStep,
    )

    current_step = PlanStep(
        id="existing_step",
        name="Existing Step",
        description="An existing step",
        tool_names=[],
        dependencies=[],
        difficulty="easy",
    )
    current_plan = ExecutionPlan(
        id=str(uuid4()), goal="Existing goal", steps=[current_step], iteration=1
    )

    # First extension response has invalid tool, second has valid tool
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Step",
                    "description": "A new step with invalid tool",
                    "tool_names": ["nonexistent_tool"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "new_step",
                    "name": "New Step",
                    "description": "A new step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": ["existing_step"],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    tools = [mock_tool]
    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    additional_steps = await plan_generator.extend_plan(
        goal="Test goal",
        tools=tools,
        iteration=2,
        history=[],
        current_plan=current_plan,
        tracer=mock_tracer,
        context=None,
    )

    # Verify the additional steps
    assert len(additional_steps) == 1
    step = additional_steps[0]
    assert step.tool_names == ["write_file"]
    assert step.name == "New Step"
    assert step.description == "A new step with valid tool"
    assert step.dependencies == ["existing_step"]

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2


@pytest.mark.asyncio
async def test_plan_validation_multiple_missing_tools(mock_llm, mock_tracer):
    """Test retry mechanism with multiple missing tools"""

    # Create multiple valid tools
    write_tool = MagicMock()
    write_tool.metadata.name = "write_file"
    write_tool.name = "write_file"

    read_tool = MagicMock()
    read_tool.metadata.name = "read_file"
    read_tool.name = "read_file"

    tools = [write_tool, read_tool]

    # Response with multiple missing tools
    first_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "description": "First step with missing tool",
                    "tool_names": ["missing_tool_1"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step2",
                    "name": "Step 2",
                    "description": "Second step with missing tool",
                    "tool_names": ["missing_tool_2"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    # Valid response retry
    second_response = """{
        "plan": {
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "description": "First step with valid tool",
                    "tool_names": ["write_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                },
                {
                    "id": "step2",
                    "name": "Step 2",
                    "description": "Second step with valid tool",
                    "tool_names": ["read_file"],
                    "dependencies": [],
                    "difficulty": "easy"
                }
            ]
        }
    }"""

    mock_llm.chat.side_effect = [
        {"content": first_response},
        {"content": second_response},
    ]

    plan_generator = PlanGenerator(mock_llm)

    # Should succeed after retry
    plan = await plan_generator.generate_plan(
        goal="Test goal",
        tools=tools,
        iteration=1,
        history=[],
        tracer=mock_tracer,
        context=None,
    )

    # Verify the plan
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_names == ["write_file"]
    assert plan.steps[1].tool_names == ["read_file"]

    # Verify LLM was called twice (initial + retry)
    assert mock_llm.chat.call_count == 2

    # Verify retry message includes both missing tools
    retry_call = mock_llm.chat.call_args_list[1]
    retry_messages = retry_call[1]["messages"]
    retry_user_message = retry_messages[-1]["content"]

    assert "missing_tool_1" in retry_user_message
    assert "missing_tool_2" in retry_user_message
    assert "write_file, read_file" in retry_user_message
