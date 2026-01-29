"""
Shared test fixtures for AI Analyst tests.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1, 101),
        "category": np.random.choice(["A", "B", "C"], 100),
        "value": np.random.randn(100) * 100 + 500,
        "count": np.random.randint(1, 50, 100),
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def sample_dataframe_with_nulls():
    """Create a sample DataFrame with null values for testing data quality."""
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(1, 51),
        "name": ["Item " + str(i) for i in range(1, 51)],
        "value": np.random.randn(50) * 100 + 500,
        "category": np.random.choice(["X", "Y", "Z"], 50),
    })
    # Add some null values
    df.loc[5:10, "value"] = np.nan
    df.loc[15:18, "category"] = None
    return df


@pytest.fixture
def sample_dataframe_with_outliers():
    """Create a DataFrame with known outliers for testing."""
    np.random.seed(42)
    values = np.random.randn(100) * 10 + 50
    # Add clear outliers
    values[0] = 200  # Upper outlier
    values[1] = -100  # Lower outlier
    values[2] = 250  # Upper outlier
    return pd.DataFrame({
        "id": range(1, 101),
        "value": values,
    })


@pytest.fixture
def sample_csv_file(tmp_path, sample_dataframe):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_json_file(tmp_path, sample_dataframe):
    """Create a temporary JSON file for testing."""
    json_path = tmp_path / "test_data.json"
    sample_dataframe.to_json(json_path, orient="records", date_format="iso")
    return str(json_path)


@pytest.fixture
def sample_excel_file(tmp_path, sample_dataframe):
    """Create a temporary Excel file for testing."""
    excel_path = tmp_path / "test_data.xlsx"
    sample_dataframe.to_excel(excel_path, index=False)
    return str(excel_path)


@pytest.fixture
def sample_parquet_file(tmp_path, sample_dataframe):
    """Create a temporary Parquet file for testing."""
    parquet_path = tmp_path / "test_data.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    return str(parquet_path)


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing without API calls."""
    with patch("anthropic.Anthropic") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_api_response_end_turn():
    """Create a mock API response that ends the turn (no tool use)."""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"

    mock_text_block = MagicMock()
    mock_text_block.text = "Analysis complete. The data shows positive trends."
    mock_text_block.type = "text"

    mock_response.content = [mock_text_block]
    return mock_response


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

    with patch("ai_analyst.analyst.get_auth_method") as mock_get_auth_method:
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
    from ai_analyst.analyst import AnalysisContext
    return AnalysisContext()


@pytest.fixture
def loaded_context(analysis_context, sample_csv_file):
    """Create an AnalysisContext with a loaded dataset."""
    analysis_context.load_dataset(sample_csv_file, "test_data")
    return analysis_context
