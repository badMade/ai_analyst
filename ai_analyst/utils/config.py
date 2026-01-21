"""
Configuration utilities.

Provides settings management, path sanitization, and logging setup.
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    allowed_paths: list[str] = Field(default_factory=list, validation_alias="ALLOWED_PATHS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def sanitize_path(file_path: str) -> Path:
    """
    Sanitize and validate a file path.

    Prevents path traversal attacks and ensures path is within allowed directories.

    Args:
        file_path: The file path to sanitize

    Returns:
        Sanitized Path object

    Raises:
        ValueError: If path contains dangerous patterns
    """
    path = Path(file_path).resolve()

    # Check against allowed paths if configured
    settings = get_settings()
    if settings.allowed_paths:
        allowed = False
        for allowed_path in settings.allowed_paths:
            allowed_base = Path(allowed_path).resolve()
            try:
                path.relative_to(allowed_base)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValueError(f"Path not in allowed directories: {path}")

    return path


def setup_logging(level: str | None = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Uses settings if not provided.
    """
    settings = get_settings()
    log_level = level or settings.log_level

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Quiet noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
