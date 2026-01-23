import pytest
from ai_analyst.utils.config import Settings

def test_settings_default_api_key_is_none(monkeypatch):
    """Settings should have no default API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from ai_analyst.utils import config
    import importlib
    importlib.reload(config)
    settings = config.Settings()
    assert settings.anthropic_api_key == ""
