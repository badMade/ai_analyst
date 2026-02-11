from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DATA_DIR: Path = Path.cwd().resolve()

class Settings(BaseSettings):
    anthropic_api_key: str

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()

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
    base_dir: Path = Path.cwd().resolve()

    # Expand user home (e.g., "~") but do not yet trust relativeness/absoluteness
    raw_path: Path = Path(path_str).expanduser()

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
