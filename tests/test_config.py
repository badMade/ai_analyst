"""
Tests for configuration and settings.

Tests the Settings class, environment variable handling, and utility functions.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_analyst.utils.config import sanitize_path

class TestSettings:
    """Tests for Settings class."""

    def test_settings_default_api_key(self):
        """Settings should have an empty default API key for security."""
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

    @pytest.fixture
    def base_dir(self, tmp_path, monkeypatch):
        """Fixture to set up a temporary BASE_DATA_DIR and patch it."""
        from ai_analyst.utils import config

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        resolved_base_dir = base_dir.resolve()
        monkeypatch.setattr(config, "BASE_DATA_DIR", resolved_base_dir)
        return resolved_base_dir

    def test_sanitize_string_path(self, base_dir):
        """Should convert string to Path object."""
        result = sanitize_path(str(base_dir / "data.csv"))
        assert isinstance(result, Path)
        assert result == base_dir / "data.csv"

    def test_sanitize_relative_path(self, base_dir):
        """Should handle relative paths."""
        result = sanitize_path("data/file.csv")
        assert isinstance(result, Path)
        assert result == base_dir / "data" / "file.csv"

    def test_sanitize_path_with_spaces(self, base_dir):
        """Should handle paths with spaces."""
        result = sanitize_path(str(base_dir / "my data" / "file.csv"))
        assert isinstance(result, Path)
        assert result == base_dir / "my data" / "file.csv"

    def test_sanitize_empty_path(self, base_dir):
        """Should handle empty string path."""
        result = sanitize_path("")
        assert isinstance(result, Path)
        assert result == base_dir

    def test_sanitize_windows_style_path(self, base_dir):
        """Should handle Windows-style paths on any platform."""
        with pytest.raises(ValueError):
            sanitize_path("C:\\Users\\data.csv")

    def test_sanitize_path_rejects_outside_base(self, base_dir):
        """Should reject paths outside of base directory."""
        outside_path = base_dir.parent / "outside.csv"
        with pytest.raises(ValueError):
            sanitize_path(outside_path)


class TestEnvironmentConfiguration:
    """Tests for environment-based configuration."""

    def test_missing_api_key_uses_default(self, monkeypatch):
        """Should use default key (empty) when env var is not set."""
        # Clear the env var if it exists
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Need to reload the module to pick up changed env
        from ai_analyst.utils.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key == ""

    def test_empty_api_key_is_empty_string(self, monkeypatch):
        """Empty env var should result in an empty string value."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        from ai_analyst.utils.config import Settings

        settings = Settings()
        # pydantic-settings uses the empty string from the env var
        # instead of falling back to the default.
        assert settings.anthropic_api_key == ""


class TestAuthMethod:
    """Tests for authentication method determination."""

    def test_get_auth_method_raises_error_when_no_key_and_no_pro(self, monkeypatch):
        """Should raise ValueError when no auth method is available."""
        from ai_analyst.utils.config import get_auth_method

        # Ensure no API key in settings
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Mock check_pro_subscription_available to return False
        with patch("ai_analyst.utils.config.check_pro_subscription_available", return_value=False):
            with pytest.raises(ValueError, match="No authentication method available"):
                get_auth_method()

    def test_get_auth_method_uses_api_key_if_present(self, monkeypatch):
        """Should use API key if present and Pro not preferred/available."""
        from ai_analyst.utils.config import get_auth_method, AuthMethod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        # Mock Pro check to false so it falls back to API key (or picks it if preference is API)
        # Default preference is PRO, but if PRO is not available, it uses API key.
        with patch("ai_analyst.utils.config.check_pro_subscription_available", return_value=False):
            method, key = get_auth_method()
            assert method == AuthMethod.API_KEY
            assert key == "sk-test"
