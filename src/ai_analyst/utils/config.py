import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings


class AuthMethod(str, Enum):
    """Authentication method for Claude API."""
    PRO_SUBSCRIPTION = "pro_subscription"  # Claude Pro/Max subscription via OAuth
    API_KEY = "api_key"  # Direct API key


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    # User preference: "pro" for Pro subscription first, "api" for API key first
    auth_preference: str = "pro"


BASE_DATA_DIR: Path = Path.cwd().resolve()


def check_pro_subscription_available() -> bool:
    """
    Check if Claude Pro subscription authentication is available.

    This checks if the user has authenticated via `claude login` command
    which stores OAuth credentials locally.
    """
    # Check for Claude CLI config directory with stored credentials
    claude_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    claude_config_paths = [
        Path.home() / ".claude" / "credentials.json",
        Path.home() / ".config" / "claude" / "credentials.json",
    ]
    if claude_config_dir:
        claude_config_paths.append(Path(claude_config_dir) / "credentials.json")

    for config_path in claude_config_paths:
        if config_path.exists():
            return True

    return False


def get_auth_method() -> tuple[AuthMethod, str | None]:
    """
    Determine the best authentication method available.

    Returns:
        Tuple of (AuthMethod, api_key or None)

    Priority (based on auth_preference):
    1. If preference is "pro": Try Pro subscription first, then API key
    2. If preference is "api": Try API key first, then Pro subscription
    """
    settings = get_settings()
    api_key = settings.anthropic_api_key
    pro_available = check_pro_subscription_available()

    if settings.auth_preference.lower() == "pro":
        # Pro subscription first (user's preferred method)
        if pro_available:
            return AuthMethod.PRO_SUBSCRIPTION, None
        elif api_key:
            return AuthMethod.API_KEY, api_key
    else:
        # API key first
        if api_key:
            return AuthMethod.API_KEY, api_key
        elif pro_available:
            return AuthMethod.PRO_SUBSCRIPTION, None

    # Neither available
    raise ValueError(
        "No authentication method available.\n\n"
        "Option 1 (Recommended): Use your Claude Pro subscription\n"
        "  Run: claude login\n\n"
        "Option 2: Use API key\n"
        "  Set: export ANTHROPIC_API_KEY='your-api-key'\n"
    )


def get_settings() -> Settings:
    return Settings()


def sanitize_path(path: str | Path) -> Path:
    """
    Normalize and validate a filesystem path to prevent path traversal.

    All non-absolute paths are resolved relative to BASE_DATA_DIR, and the
    final resolved path must remain within BASE_DATA_DIR.
    """
    raw_path: Path = Path(path)

    if raw_path.is_absolute():
        candidate: Path = raw_path.resolve(strict=False)
    else:
        candidate = (BASE_DATA_DIR / raw_path).resolve(strict=False)

    if not candidate.is_relative_to(BASE_DATA_DIR):
        logging.error(
            "Refusing to access path outside of base directory: %s (base: %s)",
            candidate,
            BASE_DATA_DIR,
        )
        raise ValueError(
            f"Invalid path outside of allowed base directory: {candidate}"
        )

    return candidate
def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
