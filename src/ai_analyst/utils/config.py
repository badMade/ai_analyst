from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""

def get_settings():
    return Settings()

def sanitize_path(path: str) -> Path:
    return Path(path)
