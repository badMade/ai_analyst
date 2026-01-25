"""
Tests for configuration and settings.

Tests the Settings class, environment variable handling, and utility functions.
"""

from pathlib import Path
import pytest

class TestSettings:
    """Tests for Settings class."""

    def test_settings_default_api_key(self):
        """Settings should have a default API key (for development)."""
        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == ""

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

    def test_sanitize_string_path(self, tmp_path):
        """Should convert string to Path object."""
        from ai_analyst.utils.config import sanitize_path

        file_path = tmp_path / "data.csv"
        file_path.touch()

        result = sanitize_path(str(file_path))
        assert isinstance(result, Path)
        assert result == file_path

    def test_sanitize_relative_path(self, tmp_path):
        """Should handle relative paths."""
        from ai_analyst.utils.config import sanitize_path

        (tmp_path / "data").mkdir()
        file_path = tmp_path / "data" / "file.csv"
        file_path.touch()

        result = sanitize_path("data/file.csv")
        assert isinstance(result, Path)
        assert result.resolve() == file_path.resolve()

    def test_sanitize_path_with_spaces(self, tmp_path):
        """Should handle paths with spaces."""
        from ai_analyst.utils.config import sanitize_path

        dir_with_space = tmp_path / "my data"
        dir_with_space.mkdir()
        file_path = dir_with_space / "file.csv"
        file_path.touch()

        result = sanitize_path(str(file_path))
        assert isinstance(result, Path)
        assert "my data" in str(result)
        assert result == file_path

    def test_sanitize_empty_path(self):
        """Should handle empty string path."""
        from ai_analyst.utils.config import sanitize_path
        from ai_analyst.utils import config

        result = sanitize_path("")
        assert isinstance(result, Path)
        assert result == config.BASE_DATA_DIR

    def test_sanitize_windows_style_path(self):
        """Should handle Windows-style paths on any platform."""
        from ai_analyst.utils.config import sanitize_path

        with pytest.raises(ValueError):
             sanitize_path("C:\\Users\\data.csv")


class TestEnvironmentConfiguration:
    """Tests for environment-based configuration."""

    def test_missing_api_key_uses_default(self, monkeypatch):
        """Should use default key when env var is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == ""

    def test_empty_api_key_is_empty_string(self, monkeypatch):
        """Empty env var should result in an empty string value."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == ""
