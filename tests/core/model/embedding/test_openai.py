from typing import Any, Dict, List, Type

import pytest

from xagent.core.model.embedding import OpenAIEmbedding

from .test_embedding_base import BaseEmbeddingTest


class TestOpenAIEmbedding(BaseEmbeddingTest):
    """Test OpenAIEmbedding client."""

    def get_client_class(self) -> Type[OpenAIEmbedding]:
        return OpenAIEmbedding

    def get_default_model(self) -> str:
        return "text-embedding-3-small"

    def get_api_key_error_message(self) -> str:
        return "OPENAI_API_KEY is required"

    def get_embedding_error_message(self) -> str:
        return "OpenAI embedding failed"

    def get_mock_response(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        return {"data": [{"embedding": emb} for emb in embeddings]}

    def get_request_session_path(self) -> str:
        return "requests.Session.post"

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"base_url": "https://api.openai.com/v1/embeddings"}

    def verify_request_payload(
        self, payload: Dict[str, Any], texts: List[str], **kwargs
    ):
        """Verify OpenAI-specific request payload."""
        assert payload["model"] == self.get_default_model()
        assert payload["input"] == texts

        dimension = kwargs.get("dimension")
        if dimension:
            assert payload["dimensions"] == dimension
        else:
            assert "dimensions" not in payload

    def test_base_url_initialization(self):
        """Test base URL initialization."""
        client = OpenAIEmbedding(api_key="test_key", base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
