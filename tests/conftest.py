"""
Pytest configuration and shared fixtures.

Provides mocks for external dependencies and sample data fixtures.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def mock_api_key(monkeypatch):
    """Ensure API key is set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key-12345")
    # Clear settings cache to pick up new env var
    from ai_analyst.utils.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def clear_settings_cache():
    """Clear the settings cache before test."""
    from ai_analyst.utils.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ============================================================================
# Sample DataFrames
# ============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Basic sample DataFrame with mixed types."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
    })


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """DataFrame with only numeric columns for statistical tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100) * 2 + 1,
        "c": np.random.randn(100) * 0.5 - 1,
    })


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """DataFrame containing null values."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.0, None, 30.0, None, 50.0],
        "category": ["A", "B", None, "D", "E"],
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame with clear outliers."""
    return pd.DataFrame({
        "normal": [10, 11, 12, 10, 11, 12, 10, 11, 100, 12],  # 100 is outlier
        "clean": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """DataFrame with single row."""
    return pd.DataFrame({"a": [1], "b": [2], "c": [3]})


@pytest.fixture
def constant_df() -> pd.DataFrame:
    """DataFrame with zero variance columns."""
    return pd.DataFrame({
        "constant": [5, 5, 5, 5, 5],
        "variable": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def trend_df() -> pd.DataFrame:
    """DataFrame with clear trend."""
    return pd.DataFrame({
        "increasing": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "decreasing": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "random": [5, 2, 8, 1, 9, 3, 7, 4, 6, 10],
    })


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_file(sample_df) -> Path:
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_df.to_csv(f, index=False)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_json_file(sample_df) -> Path:
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        sample_df.to_json(f, orient="records")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_excel_file(sample_df) -> Path:
    """Create a temporary Excel file."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = Path(f.name)
    sample_df.to_excel(path, index=False)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_parquet_file(sample_df) -> Path:
    """Create a temporary Parquet file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    sample_df.to_parquet(path, index=False)
    yield path
    path.unlink(missing_ok=True)


# ============================================================================
# Anthropic API Mocks
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_end_turn_response():
    """Mock response that ends the turn with text."""
    def _create_response(text: str):
        response = MagicMock()
        response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.text = text
        text_block.type = "text"
        response.content = [text_block]
        return response
    return _create_response


@pytest.fixture
def mock_tool_use_response():
    """Mock response that requests tool use."""
    def _create_response(tool_name: str, tool_input: dict, tool_id: str = "test_id"):
        response = MagicMock()
        response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_input
        tool_block.id = tool_id
        response.content = [tool_block]
        return response
    return _create_response


# ============================================================================
# AnalysisContext Fixtures
# ============================================================================

@pytest.fixture
def analysis_context():
    """Create a fresh AnalysisContext instance."""
    from analyst import AnalysisContext
    return AnalysisContext()


@pytest.fixture
def context_with_dataset(analysis_context, sample_csv_file):
    """AnalysisContext with a preloaded dataset."""
    analysis_context.load_dataset(str(sample_csv_file), "test_data")
    return analysis_context


# ============================================================================
# StandaloneAnalyst Fixtures
# ============================================================================

@pytest.fixture
def mock_analyst(mock_anthropic_client, mock_api_key):
    """Create a StandaloneAnalyst with mocked API."""
    from analyst import StandaloneAnalyst
    analyst = StandaloneAnalyst()
    analyst.client = mock_anthropic_client
    return analyst


@pytest.fixture
def analyst_with_data(mock_analyst, sample_csv_file):
    """StandaloneAnalyst with preloaded test data."""
    mock_analyst.context.load_dataset(str(sample_csv_file), "test_data")
    return mock_analyst
