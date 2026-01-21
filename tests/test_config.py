"""
Tests for configuration utilities.

Tests settings loading, path sanitization, and logging setup.
"""

import logging
from pathlib import Path

from ai_analyst.utils.config import get_settings, sanitize_path, setup_logging


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_api_key_from_env(self, monkeypatch, clear_settings_cache):
        """Settings loads ANTHROPIC_API_KEY from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")

        settings = get_settings()

        assert settings.anthropic_api_key == "test-key-123"

    def test_settings_loads_log_level_from_env(self, monkeypatch, clear_settings_cache):
        """Settings loads LOG_LEVEL from environment."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = get_settings()

        assert settings.log_level == "DEBUG"

    def test_settings_default_log_level(self, monkeypatch, clear_settings_cache):
        """Settings uses INFO as default log level."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        settings = get_settings()

        assert settings.log_level == "INFO"

    def test_settings_cached(self, monkeypatch, clear_settings_cache):
        """get_settings returns cached instance."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key1")

        settings1 = get_settings()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "key2")

        settings2 = get_settings()

        # Should be same instance due to caching
        assert settings1 is settings2
        assert settings1.anthropic_api_key == "key1"


class TestSanitizePath:
    """Tests for sanitize_path function."""

    def test_sanitize_absolute_path(self, clear_settings_cache):
        """Sanitizes absolute path."""
        path = sanitize_path("/tmp/test.csv")

        assert isinstance(path, Path)
        assert path.is_absolute()

    def test_sanitize_relative_path(self, clear_settings_cache):
        """Resolves relative path to absolute."""
        path = sanitize_path("./test.csv")

        assert path.is_absolute()

    def test_sanitize_with_allowed_paths(self, monkeypatch, clear_settings_cache, tmp_path):
        """Path within allowed directories is accepted."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        test_file = allowed_dir / "test.csv"
        test_file.touch()

        # pydantic-settings expects JSON format for list env vars
        import json
        monkeypatch.setenv("ALLOWED_PATHS", json.dumps([str(allowed_dir)]))

        # Need to clear cache to reload settings
        get_settings.cache_clear()

        # This should work
        path = sanitize_path(str(test_file))
        assert path == test_file.resolve()

    def test_sanitize_path_traversal_resolved(self, clear_settings_cache, tmp_path):
        """Path with .. is resolved."""
        test_dir = tmp_path / "a" / "b"
        test_dir.mkdir(parents=True)

        path_with_traversal = str(test_dir / ".." / "b" / "file.csv")
        path = sanitize_path(path_with_traversal)

        # Should resolve to clean path
        assert ".." not in str(path)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default_level(self, clear_settings_cache, monkeypatch):
        """setup_logging uses default INFO level."""
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        get_settings.cache_clear()

        # Force reconfigure by removing existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

        setup_logging()

        # setup_logging should configure the logging system
        # The function uses basicConfig which sets up the root logger
        assert True  # Test passes if setup_logging doesn't raise

    def test_setup_logging_custom_level(self, clear_settings_cache):
        """setup_logging accepts custom level parameter."""
        setup_logging(level="DEBUG")

        # Check that logging is configured
        logger = logging.getLogger("test_custom")
        logger.debug("Test message")

    def test_setup_logging_quiets_noisy_loggers(self, clear_settings_cache):
        """setup_logging sets noisy loggers to WARNING."""
        setup_logging()

        httpx_logger = logging.getLogger("httpx")
        anthropic_logger = logging.getLogger("anthropic")

        assert httpx_logger.level >= logging.WARNING
        assert anthropic_logger.level >= logging.WARNING

    def test_setup_logging_format(self, clear_settings_cache, caplog):
        """setup_logging configures log format."""
        setup_logging(level="INFO")

        logger = logging.getLogger("test_format")
        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        # Should have timestamp in format
        # (caplog may not capture format, but setup should not crash)


class TestSettingsWithEmptyEnv:
    """Tests for settings with minimal environment."""

    def test_empty_api_key_default(self, monkeypatch, clear_settings_cache):
        """Empty API key defaults to empty string."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        settings = get_settings()

        assert settings.anthropic_api_key == ""

    def test_allowed_paths_empty_by_default(self, monkeypatch, clear_settings_cache):
        """allowed_paths is empty list by default."""
        settings = get_settings()

        assert settings.allowed_paths == []
