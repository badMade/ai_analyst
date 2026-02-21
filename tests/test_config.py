"""
Tests for configuration and settings.

Tests the Settings class, environment variable handling, and utility functions.
"""

import os
from pathlib import Path

import pytest

from ai_analyst.utils.config import sanitize_path

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

class TestAuthMethod:
    """Tests for get_auth_method logic."""

    def test_auth_preference_pro_prioritizes_subscription(self, monkeypatch):
        """Should prioritize Pro subscription when preference is 'pro'."""
        monkeypatch.setenv("AUTH_PREFERENCE", "pro")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        # Mock check_pro_subscription_available to return True
        monkeypatch.setattr(
            "ai_analyst.utils.config.check_pro_subscription_available",
            lambda: True
        )

        from ai_analyst.utils.config import get_auth_method, AuthMethod

        method, key = get_auth_method()
        assert method == AuthMethod.PRO_SUBSCRIPTION
        assert key is None

    def test_auth_preference_api_prioritizes_key(self, monkeypatch):
        """Should prioritize API key when preference is 'api'."""
        monkeypatch.setenv("AUTH_PREFERENCE", "api")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        # Mock check_pro_subscription_available to return True
        monkeypatch.setattr(
            "ai_analyst.utils.config.check_pro_subscription_available",
            lambda: True
        )

        from ai_analyst.utils.config import get_auth_method, AuthMethod

        method, key = get_auth_method()
        assert method == AuthMethod.API_KEY
        assert key == "sk-test-key"

    def test_fallback_to_api_key_when_pro_unavailable(self, monkeypatch):
        """Should fall back to API key when Pro is unavailable."""
        monkeypatch.setenv("AUTH_PREFERENCE", "pro")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        # Mock check_pro_subscription_available to return False
        monkeypatch.setattr(
            "ai_analyst.utils.config.check_pro_subscription_available",
            lambda: False
        )

        from ai_analyst.utils.config import get_auth_method, AuthMethod

        method, key = get_auth_method()
        assert method == AuthMethod.API_KEY
        assert key == "sk-test-key"

    def test_fallback_to_pro_when_api_key_missing(self, monkeypatch):
        """Should fall back to Pro subscription when API key is missing."""
        monkeypatch.setenv("AUTH_PREFERENCE", "api")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        # Mock check_pro_subscription_available to return True
        monkeypatch.setattr(
            "ai_analyst.utils.config.check_pro_subscription_available",
            lambda: True
        )

        from ai_analyst.utils.config import get_auth_method, AuthMethod

        method, key = get_auth_method()
        assert method == AuthMethod.PRO_SUBSCRIPTION
        assert key is None

    def test_no_auth_method_raises_error(self, monkeypatch):
        """Should raise ValueError when no auth method is available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        # Mock check_pro_subscription_available to return False
        monkeypatch.setattr(
            "ai_analyst.utils.config.check_pro_subscription_available",
            lambda: False
        )

        from ai_analyst.utils.config import get_auth_method

        with pytest.raises(ValueError, match="No authentication method available"):
            get_auth_method()
