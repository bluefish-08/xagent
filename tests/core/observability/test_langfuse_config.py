"""Tests for Langfuse configuration."""

import json
import tempfile
from pathlib import Path

import pytest

from xagent.core.observability.langfuse_config import (
    LangfuseConfig,
    load_langfuse_config,
)


class TestLangfuseConfig:
    """Test cases for LangfuseConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LangfuseConfig()

        assert config.public_key is None
        assert config.secret_key is None
        assert config.host == "https://cloud.langfuse.com"
        assert config.enabled is True
        assert config.debug is False
        assert config.flush_at == 15
        assert config.flush_interval == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LangfuseConfig(
            public_key="test_public",
            secret_key="test_secret",
            host="https://custom.langfuse.com",
            enabled=False,
            debug=True,
            flush_at=20,
            flush_interval=1.0,
        )

        assert config.public_key == "test_public"
        assert config.secret_key == "test_secret"
        assert config.host == "https://custom.langfuse.com"
        assert config.enabled is False
        assert config.debug is True
        assert config.flush_at == 20
        assert config.flush_interval == 1.0

    def test_invalid_config(self):
        """Test that invalid configuration raises validation error."""
        with pytest.raises(ValueError):
            LangfuseConfig(flush_at="invalid")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            LangfuseConfig(unknown_field="value")


class TestLoadLangfuseConfig:
    """Test cases for load_langfuse_config function."""

    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_langfuse_config(temp_dir)

            # Should return default config
            assert isinstance(config, LangfuseConfig)
            assert config.public_key is None
            assert config.secret_key is None
            assert config.host == "https://cloud.langfuse.com"
            assert config.enabled is True

    def test_load_config_file_exists(self):
        """Test loading config when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "langfuse_config.json"
            config_data = {
                "public_key": "test_public",
                "secret_key": "test_secret",
                "host": "https://custom.langfuse.com",
                "enabled": False,
                "debug": True,
                "flush_at": 25,
                "flush_interval": 1.5,
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = load_langfuse_config(temp_dir)

            assert config.public_key == "test_public"
            assert config.secret_key == "test_secret"
            assert config.host == "https://custom.langfuse.com"
            assert config.enabled is False
            assert config.debug is True
            assert config.flush_at == 25
            assert config.flush_interval == 1.5

    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "langfuse_config.json"

            with open(config_path, "w") as f:
                f.write("invalid json")

            config = load_langfuse_config(temp_dir)

            # Should return default config
            assert isinstance(config, LangfuseConfig)
            assert config.public_key is None
            assert config.host == "https://cloud.langfuse.com"

    def test_load_config_partial_data(self):
        """Test loading config with partial data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "langfuse_config.json"
            config_data = {
                "public_key": "test_public",
                "secret_key": "test_secret",
                # Missing other fields should use defaults
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = load_langfuse_config(temp_dir)

            assert config.public_key == "test_public"
            assert config.secret_key == "test_secret"
            assert config.host == "https://cloud.langfuse.com"  # default
            assert config.enabled is True  # default
            assert config.flush_at == 15  # default

    def test_load_config_invalid_data_types(self):
        """Test loading config with invalid data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "langfuse_config.json"
            config_data = {
                "public_key": "test_public",
                "secret_key": "test_secret",
                "flush_at": "invalid",  # Should be int
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = load_langfuse_config(temp_dir)

            # Should return default config due to validation error
            assert isinstance(config, LangfuseConfig)
            assert config.public_key is None
            assert config.host == "https://cloud.langfuse.com"

    def test_load_config_empty_file(self):
        """Test loading config with empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "langfuse_config.json"

            with open(config_path, "w") as f:
                f.write("")

            config = load_langfuse_config(temp_dir)

            # Should return default config
            assert isinstance(config, LangfuseConfig)
            assert config.public_key is None
            assert config.host == "https://cloud.langfuse.com"
