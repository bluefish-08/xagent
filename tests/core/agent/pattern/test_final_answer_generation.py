"""
Unit tests for final answer generation functionality in DAG plan-execute pattern.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.agent.pattern.dag_plan_execute.result_analyzer import ResultAnalyzer
from xagent.core.agent.trace import Tracer
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk


class TestFinalAnswerGeneration:
    """Test cases for final answer generation functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        # Create the mock instance first
        mock_llm_instance = MagicMock(spec=BaseLLM)
        mock_llm_instance.chat = AsyncMock()

        async def mock_stream_chat(**kwargs):
            """Mock stream_chat that yields a single chunk"""
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
    def mock_tracer(self):
        """Create a mock tracer for testing."""
        return MagicMock(spec=Tracer)

    @pytest.fixture
    def result_analyzer(self, mock_llm, mock_tracer):
        """Create a ResultAnalyzer instance for testing."""
        return ResultAnalyzer(mock_llm, mock_tracer)

    @pytest.fixture
    def sample_execution_history(self):
        """Create sample execution history for testing."""
        return [
            {
                "iteration": 1,
                "plan": {
                    "goal": "Compare Google's latest text-to-image model with Qwen-Image",
                    "steps": [
                        {"id": "step1", "name": "Search Google model"},
                        {"id": "step2", "name": "Search Qwen-Image"},
                        {"id": "step3", "name": "Compare and analyze"},
                    ],
                },
                "results": [
                    {
                        "step_id": "step1",
                        "step_name": "Search Google model",
                        "status": "completed",
                        "result": {
                            "output": "Google's latest model is Gemini 2.5 Flash Image with advanced text rendering capabilities."
                        },
                    },
                    {
                        "step_id": "step2",
                        "step_name": "Search Qwen-Image",
                        "status": "completed",
                        "result": {
                            "output": "Qwen-Image is a multimodal model with strong image generation and editing capabilities."
                        },
                    },
                    {
                        "step_id": "step3",
                        "step_name": "Compare and analyze",
                        "status": "completed",
                        "result": {
                            "output": "Both models excel in different areas: Gemini in text rendering, Qwen in multimodal support."
                        },
                    },
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_final_answer_success(
        self, result_analyzer, sample_execution_history
    ):
        """Test successful final answer generation."""
        # Setup mock response
        mock_response = {
            "content": "Based on the analysis, Google's Gemini 2.5 Flash Image excels in text rendering and precision, while Qwen-Image offers superior multimodal capabilities and image editing features."
        }
        result_analyzer.llm.chat.return_value = mock_response

        # Test
        goal = "Compare Google's latest text-to-image model with Qwen-Image"
        result = await result_analyzer.generate_final_answer(
            goal, sample_execution_history
        )

        # Verify
        assert result == mock_response["content"]
        result_analyzer.llm.chat.assert_called_once()

        # Verify call arguments
        call_args = result_analyzer.llm.chat.call_args
        assert len(call_args[1]["messages"]) == 2  # system + user prompt
        assert goal in call_args[1]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_generate_final_answer_with_nested_output(self, result_analyzer):
        """Test final answer generation with nested output structure."""
        # Setup history with nested output structure
        history = [
            {
                "iteration": 1,
                "results": [
                    {
                        "step_id": "step1",
                        "step_name": "Analysis step",
                        "status": "completed",
                        "result": {
                            "output": {
                                "output": "This is the actual nested content that should be extracted."
                            }
                        },
                    }
                ],
            }
        ]

        mock_response = {"content": "Comprehensive analysis complete."}
        result_analyzer.llm.chat.return_value = mock_response

        # Test
        goal = "Analyze the data"
        result = await result_analyzer.generate_final_answer(goal, history)

        # Verify
        assert result == mock_response["content"]
        result_analyzer.llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_final_answer_llm_failure(
        self, result_analyzer, sample_execution_history
    ):
        """Test fallback to simple summary when LLM fails."""
        # Setup mock to raise exception
        result_analyzer.llm.chat.side_effect = Exception("LLM service unavailable")

        # Test
        goal = "Compare Google's latest text-to-image model with Qwen-Image"
        result = await result_analyzer.generate_final_answer(
            goal, sample_execution_history
        )

        # Verify fallback summary is returned
        assert "Task completed successfully with 3 steps" in result
        result_analyzer.llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_final_answer_empty_history(self, result_analyzer):
        """Test final answer generation with empty history."""
        mock_response = {"content": "No execution results available."}
        result_analyzer.llm.chat.return_value = mock_response

        # Test
        goal = "Test goal"
        result = await result_analyzer.generate_final_answer(goal, [])

        # Verify
        assert result == mock_response["content"]

    @pytest.mark.asyncio
    async def test_generate_final_answer_with_failed_steps(self, result_analyzer):
        """Test final answer generation with mixed success/failure results."""
        history = [
            {
                "iteration": 1,
                "results": [
                    {
                        "step_id": "step1",
                        "step_name": "Successful step",
                        "status": "completed",
                        "result": {"output": "Success content"},
                    },
                    {
                        "step_id": "step2",
                        "step_name": "Failed step",
                        "status": "failed",
                        "error": "Network timeout",
                    },
                ],
            }
        ]

        mock_response = {"content": "Partial analysis completed."}
        result_analyzer.llm.chat.return_value = mock_response

        # Test
        goal = "Test with failures"
        result = await result_analyzer.generate_final_answer(goal, history)

        # Verify
        assert result == mock_response["content"]

    def test_extract_content_from_result_simple_string(self, result_analyzer):
        """Test content extraction from simple string result."""
        result = "Simple string result"
        content = result_analyzer._extract_content_from_result(result)
        assert content == "Simple string result"

    def test_extract_content_from_result_dict_with_output(self, result_analyzer):
        """Test content extraction from dict with output key."""
        result = {"output": "This is the output content"}
        content = result_analyzer._extract_content_from_result(result)
        assert content == "This is the output content"

    def test_extract_content_from_result_dict_with_analysis(self, result_analyzer):
        """Test content extraction from dict with analysis_result key."""
        result = {"analysis_result": "This is the analysis content"}
        content = result_analyzer._extract_content_from_result(result)
        assert content == "This is the analysis content"

    def test_extract_content_from_result_nested_dict(self, result_analyzer):
        """Test content extraction from nested dict structure."""
        result = {"output": {"output": "Nested output content"}}
        content = result_analyzer._extract_content_from_result(result)
        assert content == "Nested output content"

    def test_extract_content_from_result_unknown_structure(self, result_analyzer):
        """Test content extraction from unknown structure."""
        result = {"unknown_key": "Unknown content"}
        content = result_analyzer._extract_content_from_result(result)
        assert "{'unknown_key': 'Unknown content'}" in content

    def test_summarize_execution_results(
        self, result_analyzer, sample_execution_history
    ):
        """Test execution results summarization."""
        summary = result_analyzer._summarize_execution_results(sample_execution_history)

        assert "Iteration 1:" in summary
        assert "Successful Steps:" in summary
        assert "Search Google model" in summary
        assert "Search Qwen-Image" in summary
        assert "Compare and analyze" in summary

    def test_summarize_execution_results_with_failures(self, result_analyzer):
        """Test summarization with failed steps."""
        history = [
            {
                "iteration": 1,
                "results": [
                    {
                        "step_id": "step1",
                        "step_name": "Successful step",
                        "status": "completed",
                        "result": {"output": "Success"},
                    },
                    {
                        "step_id": "step2",
                        "step_name": "Failed step",
                        "status": "failed",
                        "error": "Timeout error",
                    },
                ],
            }
        ]

        summary = result_analyzer._summarize_execution_results(history)

        assert "Successful Steps:" in summary
        assert "Failed Steps:" in summary
        assert "Timeout error" in summary

    def test_summarize_execution_results_empty_history(self, result_analyzer):
        """Test summarization with empty history."""
        summary = result_analyzer._summarize_execution_results([])
        assert summary == "No execution history available"

    def test_generate_fallback_summary_success(
        self, result_analyzer, sample_execution_history
    ):
        """Test fallback summary generation for successful execution."""
        summary = result_analyzer._generate_fallback_summary(sample_execution_history)
        assert "Task completed successfully with 3 steps" == summary

    def test_generate_fallback_summary_partial_success(self, result_analyzer):
        """Test fallback summary generation for partial success."""
        history = [
            {
                "results": [
                    {"status": "completed"},
                    {"status": "completed"},
                    {"status": "failed"},
                ]
            }
        ]

        summary = result_analyzer._generate_fallback_summary(history)
        assert "Partial success: 2 steps completed, 1 steps failed" == summary

    def test_generate_fallback_summary_all_failed(self, result_analyzer):
        """Test fallback summary generation for all failed steps."""
        history = [{"results": [{"status": "failed"}, {"status": "failed"}]}]

        summary = result_analyzer._generate_fallback_summary(history)
        assert "Task failed with 2 failed steps" == summary

    def test_generate_fallback_summary_empty_history(self, result_analyzer):
        """Test fallback summary generation with empty history."""
        summary = result_analyzer._generate_fallback_summary([])
        assert summary == "No execution results available"

    @pytest.mark.asyncio
    async def test_build_final_answer_prompt_structure(
        self, result_analyzer, sample_execution_history
    ):
        """Test that the final answer prompt has correct structure."""
        prompt = result_analyzer._build_final_answer_prompt(
            "Test goal", sample_execution_history
        )

        # Verify prompt structure
        assert len(prompt) == 2  # system and user messages
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"

        # Verify content
        system_content = prompt[0]["content"]
        user_content = prompt[1]["content"]

        assert "synthesizes execution results" in system_content
        assert "Test goal" in user_content
        assert "Execution Results:" in user_content
        assert "Search Google model" in user_content
