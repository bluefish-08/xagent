"""
Step agent factory for DAG plan-execute pattern.
"""

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from ....memory import MemoryStore
from ....memory.in_memory import InMemoryMemoryStore
from ....model.chat.basic.base import BaseLLM
from ....tools.adapters.vibe import Tool
from ....workspace import TaskWorkspace
from ...trace import Tracer

if TYPE_CHECKING:
    from ...agent import Agent

logger = logging.getLogger(__name__)


class StepAgentFactory:
    """Factory for creating specialized agents for plan steps"""

    def __init__(
        self,
        llm: BaseLLM,
        tracer: Tracer,
        workspace: TaskWorkspace,
        default_factory: Optional[Callable[[str, List[Tool], str], "Agent"]] = None,
        fast_llm: Optional[BaseLLM] = None,
        compact_llm: Optional[BaseLLM] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.llm = llm
        self.fast_llm = fast_llm
        self.compact_llm = (
            compact_llm or llm
        )  # Use main LLM if compact_llm not provided
        self.tracer = tracer
        self.workspace = workspace
        self.memory_store = memory_store
        self.default_factory = default_factory or self._default_step_agent_factory

    def create_step_agent(
        self,
        step_name: str,
        tools: List[Tool],
        difficulty: str = "hard",
    ) -> "Agent":
        """Create a specialized agent for a plan step"""
        return self.default_factory(step_name, tools, difficulty)

    def _default_step_agent_factory(
        self,
        step_name: str,
        tools: List[Tool],
        difficulty: str = "hard",
    ) -> "Agent":
        """Default factory for creating step agents."""

        # Import here to avoid circular import
        from ...agent import Agent

        # Choose LLM based on difficulty
        if difficulty == "easy" and self.fast_llm:
            step_llm = self.fast_llm
            logger.info(f"Using fast LLM for easy step: {step_name}")
        else:
            step_llm = self.llm
            logger.info(f"Using default LLM for {difficulty} step: {step_name}")

        # Import here to avoid circular import
        from ..react import ReActPattern

        # Create a ReAct pattern for this step, passing the tracer, workspace, and memory store
        react_pattern = ReActPattern(
            llm=step_llm,
            tracer=self.tracer,  # Pass the parent tracer
            compact_llm=self.compact_llm,  # Pass the compact LLM for context compression
            memory_store=self.memory_store,  # Pass the shared memory store
            is_sub_agent=True,  # Mark as sub-agent to prevent task completion events
        )

        # Create memory for this step (fallback to in-memory if no shared memory store)
        step_memory = self.memory_store or InMemoryMemoryStore()

        # Create specialized agent for this step
        agent = Agent(
            name=f"step_agent_{step_name}",
            patterns=[react_pattern],
            memory=step_memory,
            tools=tools,
            llm=step_llm,
        )

        return agent
