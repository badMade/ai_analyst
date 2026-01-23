import logging
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""


BASE_DATA_DIR: Path = Path.cwd().resolve()


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
