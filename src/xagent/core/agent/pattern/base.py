from abc import ABC, abstractmethod
from typing import Any, Optional

from ...memory import MemoryStore
from ...tools.adapters.vibe import Tool
from ..context import AgentContext


class AgentPattern(ABC):
    """
    Abstract interface for agent execution patterns (e.g., React, Plan, Reflect).
    Each pattern must implement the 'run' method.
    """

    @abstractmethod
    async def run(
        self,
        task: str,
        memory: MemoryStore,
        tools: list[Tool],
        context: Optional[AgentContext] = None,
    ) -> dict[str, Any]:
        """
        Execute the pattern with given task, memory, tools, and context.

        Returns:
            dict with at least a 'success' boolean field.
        """
