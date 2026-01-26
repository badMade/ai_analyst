import pytest
from pathlib import Path
from ai_analyst.utils import config
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def override_base_data_dir(tmp_path: Path):
    """
    Fixture to override the BASE_DATA_DIR to a temporary directory for all tests.
    This prevents tests from failing due to path sanitization.
    """
    original_dir = config.BASE_DATA_DIR
    config.BASE_DATA_DIR = tmp_path
    yield
    config.BASE_DATA_DIR = original_dir

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock get_settings to return a dummy key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    yield

@pytest.fixture
def analyst(mock_settings):
    """Create analyst with mocked client."""
    with patch("anthropic.Anthropic") as mock_client_class:
        from analyst import StandaloneAnalyst

        analyst = StandaloneAnalyst()
        analyst.client = MagicMock()
        return analyst

@pytest.fixture
def analyst_with_data(analyst, sample_csv_file):
    """Create analyst with loaded dataset."""
    analyst.context.load_dataset(str(sample_csv_file), "test_data")
    return analyst

@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "sample_data.csv"
    file_path.write_text("id,name,value,category\n" + "\n".join([f"{i},{i*10},{i*100},{'A' if i % 2 == 0 else 'B'}" for i in range(100)]))
    return file_path

@pytest.fixture
def mock_api_response_end_turn():
    """Return a mock API response that ends the turn."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.text = "Analysis complete. The data shows positive trends."
    response.content = [text_block]
    return response

@pytest.fixture
def mock_api_response_tool_use():
    """Create a mock API response that requests tool use."""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "load_dataset"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"file_path": "/path/to/data.csv"}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_settings():
    """Create mock settings with a test API key."""
    from ai_analyst.utils.config import AuthMethod

    with patch("analyst.get_auth_method") as mock_get_auth_method:
        mock_get_auth_method.return_value = (AuthMethod.API_KEY, "test-api-key-12345")
        yield mock_get_auth_method


@pytest.fixture
def env_with_api_key(monkeypatch):
    """Set environment variable for API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key-from-env")
    return "test-api-key-from-env"


@pytest.fixture
def analysis_context():
    """Create a fresh AnalysisContext for testing."""
    from analyst import AnalysisContext
    return AnalysisContext()


@pytest.fixture
def loaded_context(analysis_context, sample_csv_file):
    """Create an AnalysisContext with a loaded dataset."""
    analysis_context.load_dataset(sample_csv_file, "test_data")
    return analysis_context
