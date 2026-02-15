"""
Agent Tool - Wraps an Agent as a Tool for nested execution
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ...tools.adapters.vibe.base import AbstractBaseTool, ToolVisibility
from ..exceptions import AgentConfigurationError, AgentToolError

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


class CompactMode(Enum):
    """Context compacting modes"""

    COMPACT = "compact"
    FULL = "full"


class AgentTaskArgs(BaseModel):
    """Arguments for agent tool execution"""

    task: str = Field(description="Task for the agent to execute")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for the agent"
    )
    max_iterations: Optional[int] = Field(
        default=None, description="Maximum iterations for this execution"
    )


class AgentTool(AbstractBaseTool):
    """
    Tool that wraps an Agent, allowing agents to be used as tools by other agents.

    This enables nested agent execution where a parent agent can delegate
    specific tasks to specialized sub-agents.
    """

    def __init__(
        self,
        agent: "Agent",
        compact_mode: CompactMode = CompactMode.COMPACT,
        custom_description: Optional[str] = None,
    ):
        """
        Initialize AgentTool.

        Args:
            agent: The agent to wrap as a tool
            compact_mode: How to handle context (compact or full)
            custom_description: Custom description for this tool
        """
        self.agent = agent
        self.compact_mode = compact_mode

        # Set tool properties for AbstractBaseTool
        self._name = f"agent_{agent.name}"
        self._description = (
            custom_description or f"Execute specialized agent: {agent.name}"
        )
        self._visibility = ToolVisibility.PRIVATE

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def args_type(self) -> type[BaseModel]:
        return AgentTaskArgs

    def return_type(self) -> Type[BaseModel]:
        # Create a dynamic response model
        class AgentResponse(BaseModel):
            output: str
            success: bool

        return AgentResponse

    def is_async(self) -> bool:
        return True

    async def run_json_async(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute the agent tool with the given arguments."""
        # Convert dict args to AgentTaskArgs
        task_args = AgentTaskArgs(**args)
        return await self._execute_agent(task_args)

    def run_json_sync(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Synchronous version - not supported for agents."""
        raise NotImplementedError("AgentTool only supports async execution")

    async def save_state_json(self) -> Dict[str, Any]:
        """Save agent state - agents manage their own state."""
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """Load agent state - agents manage their own state."""
        pass

    async def _execute_agent(self, args: AgentTaskArgs) -> Dict[str, Any]:
        """
        Execute the wrapped agent.

        Raises:
            AgentToolError: When sub-agent execution fails
            AgentConfigurationError: When agent is misconfigured
        """
        logger.info(
            f"Executing sub-agent '{self.agent.name}' with task: {args.task[:100]}..."
        )

        if not self.agent.patterns:
            raise AgentConfigurationError(
                f"Sub-agent '{self.agent.name}' has no patterns configured",
                context={"agent_name": self.agent.name, "task": args.task[:100]},
            )

        try:
            # Get agent runner
            runner = self.agent.get_runner()

            # Set additional context if provided
            if args.context:
                for key, value in args.context.items():
                    runner.context.state[key] = value

            # Execute the agent
            result = await runner.run(args.task)

            # Store execution history in the agent
            if hasattr(runner, "messages") and runner.messages:
                self.agent.set_execution_history(runner.messages)

            self.agent.set_final_result(result)

            # Process result based on compact mode
            if self.compact_mode == CompactMode.COMPACT:
                return self._compact_result(result)
            else:
                return self._full_result(result)

        except Exception as e:
            # Wrap any exception in AgentToolError for consistent handling
            raise AgentToolError(
                agent_name=self.agent.name,
                message=f"Sub-agent execution failed: {str(e)}",
                sub_agent_error=e,
                context={
                    "task": args.task[:100],
                    "context_provided": args.context is not None,
                    "compact_mode": self.compact_mode.value,
                },
                cause=e,
            )

    def _compact_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact version of the agent execution result."""

        # Extract the most important information
        compact_result = {
            "success": result.get("success", False),
            "output": result.get("output", ""),
            "agent_name": self.agent.name,
            "compact_mode": "compact",
        }

        # Add iteration count if available
        if "iterations" in result:
            compact_result["iterations"] = result["iterations"]

        # Add error information if present
        if not result.get("success", False):
            compact_result["error"] = result.get("error", "Unknown error")

        # Add any key metrics or summary data
        if "metadata" in result:
            metadata = result["metadata"]
            compact_result["metadata"] = {
                "execution_type": metadata.get("execution_type", "unknown"),
                "tools_used": metadata.get("tools_used", 0),
            }

        return compact_result

    def _full_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return the full agent execution result."""

        full_result = result.copy()
        full_result["agent_name"] = self.agent.name
        full_result["compact_mode"] = "full"

        # Include execution history if available
        if self.agent.has_execution_history():
            full_result["execution_history"] = self.agent.get_execution_history()

        return full_result

    def get_agent(self) -> "Agent":
        """Get the wrapped agent instance."""
        return self.agent

    def set_compact_mode(self, mode: CompactMode) -> None:
        """Change the compact mode."""
        self.compact_mode = mode

    async def query_agent_details(self, query: str) -> str:
        """Query details about the agent's last execution."""
        return await self.agent.query_execution_details(query)


class QueryStepTool(AbstractBaseTool):
    """
    Tool for querying details from dependency steps in DAG execution.

    This allows a step to ask specific questions about the execution
    of its dependency steps.
    """

    def __init__(self, dag_pattern: Any) -> None:
        """Initialize with reference to the DAG pattern."""
        self.dag_pattern = dag_pattern
        self._name = "query_step_details"
        self._description = "Query details about a dependency step's execution"
        self._visibility = ToolVisibility.PRIVATE

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def args_type(self) -> type[BaseModel]:
        return QueryStepArgs

    def return_type(self) -> Type[BaseModel]:
        # Create a dynamic response model
        class QueryResponse(BaseModel):
            result: str

        return QueryResponse

    def is_async(self) -> bool:
        return True

    async def run_json_async(self, args: Mapping[str, Any]) -> str:
        """Query a specific step for details."""
        query_args = QueryStepArgs(**args)
        return await self._query_step(query_args)

    def run_json_sync(self, args: Mapping[str, Any]) -> str:
        """Synchronous version - not supported."""
        raise NotImplementedError("QueryStepTool only supports async execution")

    async def save_state_json(self) -> Dict[str, Any]:
        """No state to save."""
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """No state to load."""
        pass

    async def _query_step(self, args: "QueryStepArgs") -> str:
        """Query a specific step for details."""
        if hasattr(self.dag_pattern, "query_step_details"):
            result = await self.dag_pattern.query_step_details(args.step_id, args.query)
            return str(result)
        return f"Cannot query step {args.step_id}: DAG pattern doesn't support queries"


class QueryStepArgs(BaseModel):
    """Arguments for querying step details"""

    step_id: str = Field(description="ID of the step to query")
    query: str = Field(description="Specific question about the step's execution")
