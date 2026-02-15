"""Tests for Azure OpenAI LangChain adapter."""

import pytest

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.langchain import create_base_chat_model


class TestAzureOpenAILangChainAdapter:
    """Test suite for Azure OpenAI LangChain adapter."""

    def test_create_azure_chat_model(self, mocker, monkeypatch):
        """Test that AzureChatOpenAI is instantiated correctly."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        # Mock AzureChatOpenAI to avoid langchain compatibility issues
        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.7,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, None)

        # Verify AzureChatOpenAI was called with correct parameters
        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["deployment_name"] == "gpt-4o"
        assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com"
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["timeout"] == 30.0

    def test_azure_chat_model_with_temperature_override(self, mocker, monkeypatch):
        """Test that temperature override works for Azure OpenAI."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.5,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, 0.9)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_azure_chat_model_uses_default_temperature(self, mocker, monkeypatch):
        """Test that default temperature from config is used when not overridden."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            default_temperature=0.7,
            default_max_tokens=1024,
            timeout=30.0,
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_azure_api_version_from_env(self, mocker, monkeypatch):
        """Test that api_version is correctly sourced from environment variable."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2025-04-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["api_version"] == "2025-04-01-preview"

    def test_azure_api_version_default(self, mocker, monkeypatch):
        """Test that default api_version is used when env var is not set."""
        # Ensure the env var is not set
        monkeypatch.delenv("OPENAI_API_VERSION", raising=False)

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs["api_version"] == "2024-08-01-preview"

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported model provider raises TypeError."""
        config = ChatModelConfig(
            id="test_model",
            model_provider="unsupported_provider",
            model_name="gpt-4o",
            api_key="test-api-key",
        )

        with pytest.raises(TypeError, match="Unsupported LLM model provider"):
            create_base_chat_model(config, None)

    def test_invalid_config_type_raises_error(self):
        """Test that non-ChatModelConfig raises TypeError."""
        from xagent.core.model import EmbeddingModelConfig

        config = EmbeddingModelConfig(
            id="test_model",
            model_provider="openai",
            model_name="text-embedding-ada-002",
            api_key="test-api-key",
        )

        with pytest.raises(TypeError, match="Unsupported Chat model type"):
            create_base_chat_model(config, None)

    def test_azure_openai_preserves_none_values(self, mocker, monkeypatch):
        """Test that None values for temperature and max_tokens are handled correctly."""
        monkeypatch.setenv("OPENAI_API_VERSION", "2024-08-01-preview")

        mock_azure = mocker.patch("xagent.core.model.chat.langchain.AzureChatOpenAI")

        config = ChatModelConfig(
            id="test_azure_model",
            model_provider="azure_openai",
            model_name="gpt-4o",
            base_url="https://test.openai.azure.com",
            api_key="test-api-key",
            # No default_temperature or default_max_tokens set
        )

        create_base_chat_model(config, None)

        call_kwargs = mock_azure.call_args[1]
        # When None is passed, AzureChatOpenAI will use its own defaults
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs
