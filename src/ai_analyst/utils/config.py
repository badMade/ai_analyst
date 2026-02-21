import logging
import os
import re
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings


class AuthMethod(str, Enum):
    """Authentication method for Claude API."""
    PRO_SUBSCRIPTION = "pro_subscription"  # Claude Pro/Max subscription via OAuth
    API_KEY = "api_key"  # Direct API key


class Settings(BaseSettings):
    anthropic_api_key: str = "sk-dummy-key"
    # User preference: "pro" for Pro subscription first, "api" for API key first
    auth_preference: str = "pro"


BASE_DATA_DIR: Path = Path.cwd().resolve()


def check_pro_subscription_available() -> bool:
    """
    Check if Claude Pro subscription authentication is available.

    This checks if the user has authenticated via `claude login` command
    which stores OAuth credentials locally.
    """
    # First check if the CLI tool is available
    if not shutil.which("claude"):
        return False

    # Check authentication status
    try:
        result = subprocess.run(
            ["claude", "auth-status"],
            check=False,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
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
    api_key_clean = api_key.strip() if api_key else ""
    pro_available = check_pro_subscription_available()

    if settings.auth_preference.lower() == "pro":
        # Pro subscription first (user's preferred method)
        if pro_available:
            return AuthMethod.PRO_SUBSCRIPTION, None
        elif api_key_clean:
            return AuthMethod.API_KEY, api_key_clean
    else:
        # API key first
        if api_key_clean:
            return AuthMethod.API_KEY, api_key_clean
        elif pro_available:
            return AuthMethod.PRO_SUBSCRIPTION, None

    # Neither available
    raise ValueError(
        "No authentication method available. "
        "Option 1 (recommended): use your Claude Pro subscription "
        "(run: 'claude login'). "
        "Option 2: use an API key "
        "(set: export ANTHROPIC_API_KEY='your-api-key')."
    )


def get_settings() -> Settings:
    return Settings()


def sanitize_path(path: str | Path) -> Path:
    """
    Normalize and validate a filesystem path to prevent path traversal.

    All non-absolute paths are resolved relative to BASE_DATA_DIR, and the
    final resolved path must remain within BASE_DATA_DIR.
    """
    if os.name != "nt":
        path_str = str(path)
        if len(path_str) > 2 and path_str[1] == ":" and path_str[0].isalpha() and path_str[2] in ("\\", "/"):
            logging.error(
                "Refusing Windows-style path on non-Windows system: %s",
                path_str,
            )
            raise ValueError(
                "Invalid Windows-style path on non-Windows system: "
                f"{path_str}"
            )

    raw_path: Path = Path(path).expanduser()

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
