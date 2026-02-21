"""
Tests for configuration and settings.

Tests the Settings class, environment variable handling, and utility functions.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

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


class TestCheckProSubscriptionAvailable:
    """Tests for check_pro_subscription_available function."""

    def test_claude_not_installed(self):
        """Should return False if claude CLI is not installed."""
        from ai_analyst.utils.config import check_pro_subscription_available

        with patch("shutil.which", return_value=None):
            assert check_pro_subscription_available() is False

    def test_claude_installed_but_not_authenticated(self):
        """Should return False if claude CLI is installed but auth-status fails."""
        from ai_analyst.utils.config import check_pro_subscription_available

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, ["claude", "auth-status"])
                assert check_pro_subscription_available() is False

                # Verify subprocess was called correctly
                mock_run.assert_called_once_with(
                    ["claude", "auth-status"],
                    check=True,
                    capture_output=True,
                    text=True
                )

    def test_claude_installed_and_authenticated(self):
        """Should return True if claude CLI is installed and auth-status succeeds."""
        from ai_analyst.utils.config import check_pro_subscription_available

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Authenticated")
                assert check_pro_subscription_available() is True

                mock_run.assert_called_once_with(
                    ["claude", "auth-status"],
                    check=True,
                    capture_output=True,
                    text=True
                )

    def test_claude_file_not_found_error(self):
        """Should return False if subprocess raises FileNotFoundError despite check."""
        from ai_analyst.utils.config import check_pro_subscription_available

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert check_pro_subscription_available() is False
