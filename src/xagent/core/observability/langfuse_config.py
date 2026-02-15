"""Configuration for Langfuse observability."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Set up logger
logger = logging.getLogger(__name__)


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse observability."""

    public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse server host"
    )
    enabled: bool = Field(default=True, description="Enable/disable Langfuse tracing")
    debug: bool = Field(default=False, description="Enable debug mode for Langfuse")
    flush_at: int = Field(default=15, description="Number of events to flush at once")
    flush_interval: float = Field(default=0.5, description="Flush interval in seconds")

    class Config:
        extra = "forbid"


def load_langfuse_config(storage_root: str) -> LangfuseConfig:
    """Load Langfuse configuration from file.

    Args:
        storage_root: Root directory where config files are stored (same as model storage)

    Returns:
        LangfuseConfig: Configuration object with defaults if file doesn't exist
    """
    config_path = Path(storage_root) / "langfuse_config.json"

    if not config_path.exists():
        # Return default configuration if file doesn't exist
        return LangfuseConfig()

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        return LangfuseConfig(**data)
    except json.JSONDecodeError as e:
        # Log error but return default configuration if file is corrupted
        logger.warning(f"Invalid JSON in Langfuse config file {config_path}: {e}")
        return LangfuseConfig()
    except Exception as e:
        # Log error but return default configuration if file reading fails
        logger.warning(f"Error loading Langfuse config from {config_path}: {e}")
        return LangfuseConfig()
