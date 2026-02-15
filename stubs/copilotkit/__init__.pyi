from typing import Any, Sequence

class LangGraphAgent:
    name: str
    description: str
    graph: Any

    def __init__(self, *, name: str, description: str, graph: Any) -> None: ...

class CopilotKitRemoteEndpoint:
    def __init__(self, *, agents: Sequence[LangGraphAgent] | None = None) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
