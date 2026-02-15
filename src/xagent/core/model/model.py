from typing import List, Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 180.0
    abilities: Optional[List[str]] = None
    description: Optional[str] = None
    max_retries: int = 10


class ChatModelConfig(ModelConfig):
    model_provider: str = "openai"  # openai, zhipu, dashscope, etc.
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None
    thinking_mode: bool = False


class ImageModelConfig(ModelConfig):
    model_provider: str = "openai"  # openai, zhipu, dashscope, etc.
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None


class EmbeddingModelConfig(ModelConfig):
    model_provider: str = "dashscope"  # openai, zhipu, dashscope, etc.
    dimension: Optional[int] = None
    instruct: Optional[str] = None


class RerankModelConfig(ModelConfig):
    top_n: Optional[int] = None
    instruct: Optional[str] = None
