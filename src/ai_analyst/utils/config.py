from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = "dummy_key"

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

def sanitize_path(path_str: str) -> Path:
    return Path(path_str)
