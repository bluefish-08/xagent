from typing import Any

from ... import CopilotKitRemoteEndpoint

def add_fastapi_endpoint(
    app: Any, sdk: CopilotKitRemoteEndpoint, prefix: str = "/copilotkit"
) -> None: ...
