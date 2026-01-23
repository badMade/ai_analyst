from unittest.mock import patch
import pytest

from ai_analyst.utils.config import Settings, get_settings


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock get_settings to return a dummy key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy-key")
    # Force reload of settings
    # This is a bit of a hack, but it's the easiest way to ensure the new env var is picked up
    # on Pydantic's cached settings object.
    from ai_analyst.utils import config
    import importlib
    importlib.reload(config)
    return get_settings()


def test_analyst_initialization(mock_settings):
    """Test that StandaloneAnalyst initializes the Anthropic client correctly."""
    with patch("analyst.Anthropic") as mock_client:
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst()

        mock_client.assert_called_once_with(api_key="sk-dummy-key")
