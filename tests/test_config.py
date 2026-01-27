"""
Tests for configuration and settings.

Tests the Settings class, environment variable handling, and utility functions.
"""

import os
from pathlib import Path

import pytest


class TestSettings:
    """Tests for Settings class."""

    def test_settings_default_api_key(self):
        """Settings should have a default API key (for development)."""
        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == "sk-dummy-key"

    def test_settings_from_env(self, monkeypatch):
        """Settings should load API key from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-12345")

        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == "sk-test-key-12345"

    def test_get_settings_returns_settings_instance(self):
        """get_settings should return a Settings instance."""
        from ai_analyst.utils.config import get_settings, Settings

        settings = get_settings()
        assert isinstance(settings, Settings)


class TestSanitizePath:
    """Tests for sanitize_path utility function."""

    def test_sanitize_string_path(self, tmp_path, monkeypatch):
        """Should convert string to Path object within the base directory."""
        from ai_analyst.utils import config

        monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)
        target = tmp_path / "data.csv"

        result = config.sanitize_path(str(target))
        assert isinstance(result, Path)
        assert result == target.resolve()

    def test_sanitize_relative_path(self, tmp_path, monkeypatch):
        """Should handle relative paths anchored to base directory."""
        from ai_analyst.utils import config

        monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)
        result = config.sanitize_path("data/file.csv")
        assert isinstance(result, Path)
        assert result == (tmp_path / "data" / "file.csv").resolve()

    def test_sanitize_path_with_spaces(self, tmp_path, monkeypatch):
        """Should handle paths with spaces."""
        from ai_analyst.utils import config

        monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)
        target = tmp_path / "my data" / "file.csv"

        result = config.sanitize_path(str(target))
        assert isinstance(result, Path)
        assert "my data" in str(result)

    def test_sanitize_empty_path(self, tmp_path, monkeypatch):
        """Should handle empty string path."""
        from ai_analyst.utils import config

        monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)
        result = config.sanitize_path("")
        assert isinstance(result, Path)
        assert result == tmp_path.resolve()

    def test_rejects_path_outside_base(self, tmp_path, monkeypatch):
        """Should reject paths that resolve outside of base directory."""
        from ai_analyst.utils import config

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        monkeypatch.setattr(config, "BASE_DATA_DIR", base_dir)

        with pytest.raises(ValueError):
            config.sanitize_path("../outside.csv")

    def test_sanitize_windows_style_path(self, tmp_path, monkeypatch):
        """Should reject Windows-style paths on non-Windows systems."""
        from ai_analyst.utils import config

        monkeypatch.setattr(config, "BASE_DATA_DIR", tmp_path)

        if os.name == "nt":
            result = config.sanitize_path("C:\\Users\\data.csv")
            assert isinstance(result, Path)
        else:
            with pytest.raises(ValueError):
                config.sanitize_path("C:\\Users\\data.csv")


class TestEnvironmentConfiguration:
    """Tests for environment-based configuration."""

    def test_missing_api_key_uses_default(self, monkeypatch):
        """Should use default key when env var is not set."""
        # Clear the env var if it exists
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Need to reload the module to pick up changed env
        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == "sk-dummy-key"

    def test_empty_api_key_is_empty_string(self, monkeypatch):
        """Empty env var should result in an empty string value."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        from ai_analyst.utils.config import Settings

        settings = Settings()
        # pydantic-settings uses the empty string from the env var
        # instead of falling back to the default.
        assert settings.anthropic_api_key == ""
