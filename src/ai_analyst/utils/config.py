from pathlib import Path
from enum import Enum

from pydantic_settings import BaseSettings

BASE_DATA_DIR: Path = Path.cwd().resolve()


class AuthMethod(str, Enum):
    API_KEY = "api_key"
    PRO_SUBSCRIPTION = "pro_subscription"


class Settings(BaseSettings):
    anthropic_api_key: str

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()


def get_auth_method() -> tuple[AuthMethod, str]:
    """Resolve authentication mode from configured settings."""
    settings = get_settings()

    if settings.anthropic_api_key.startswith("sk-ant-api"):
        return AuthMethod.PRO_SUBSCRIPTION, settings.anthropic_api_key

    if not settings.anthropic_api_key or settings.anthropic_api_key == "sk-dummy-key":
        raise ValueError("Missing ANTHROPIC_API_KEY. Please set it in your environment or .env file.")

    return AuthMethod.API_KEY, settings.anthropic_api_key


def setup_logging() -> None:
    """Configure basic console logging for CLI and interactive usage."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def sanitize_path(path_str: str) -> Path:
    """
    Normalize and validate a user-provided path to prevent path traversal.

    All paths are resolved relative to the current working directory and must
    remain within that directory tree after resolution. This prevents paths
    like "../../secret.txt" or absolute paths outside the working directory
    from being used.

    :param path_str: User-provided path string.
    :return: A resolved, safe Path under the current working directory.
    :raises ValueError: If the path resolves outside the allowed root.
    """
    base_dir = BASE_DATA_DIR.resolve()

    # Expand user home (e.g., "~") but do not yet trust relativeness/absoluteness
    raw_path: Path = Path(path_str).expanduser()

    if "\\" in str(path_str):
        raise ValueError("Windows-style paths are not supported")

    if raw_path.is_absolute():
        candidate_path: Path = raw_path.resolve()
    else:
        # Anchor relative paths under the allowed base directory
        candidate_path = (base_dir / raw_path).resolve()

    try:
        # Ensure the final path is within the allowed base directory
        candidate_path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(
            f"Path {path_str!r} is outside the allowed directory {base_dir}"
        ) from exc

    return candidate_path
