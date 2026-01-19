from pathlib import Path
import os
import logging
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    ai_analyst_log_level: str = "INFO"

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()

def setup_logging():
    logging.basicConfig(level=get_settings().ai_analyst_log_level)

def sanitize_path(path_str: str) -> Path:
    """
    Sanitize and resolve a file path.
    Prevents path traversal by ensuring the path is within the current working directory.
    """
    # Allow absolute paths if they are within CWD?
    # Or force everything to be relative to CWD?
    # For now, let's assume we treat the input as relative to CWD unless it's safe.

    base_dir = Path.cwd().resolve()

    # Check if absolute path
    path = Path(path_str)
    if path.is_absolute():
        # If absolute, it must be within base_dir
        target_path = path.resolve()
    else:
        target_path = (base_dir / path).resolve()

    # Enforce jail
    if not str(target_path).startswith(str(base_dir)):
        raise ValueError(f"Security Error: Path traversal detected. Access denied to {path_str}")

    return target_path
