"""
Tests for AnalysisContext class.

Tests dataset loading, retrieval, and error handling for different file formats.
"""

import pytest
import pandas as pd
from pathlib import Path


class TestAnalysisContextInit:
    """Tests for AnalysisContext initialization."""

    def test_init_creates_empty_datasets(self, analysis_context):
        """Context should initialize with empty datasets dict."""
        assert analysis_context.datasets == {}

    def test_init_creates_empty_results(self, analysis_context):
        """Context should initialize with empty results list."""
        assert analysis_context.results == []


class TestLoadDataset:
    """Tests for loading datasets from various file formats."""

    def test_load_csv_file(self, analysis_context, sample_csv_file):
        """Should load CSV file successfully."""
        result = analysis_context.load_dataset(sample_csv_file, "test_csv")

        assert result["name"] == "test_csv"
        assert result["rows"] == 100
        assert result["columns"] == 5
        assert "id" in result["column_names"]
        assert "value" in result["column_names"]

    def test_load_json_file(self, analysis_context, sample_json_file):
        """Should load JSON file successfully."""
        result = analysis_context.load_dataset(sample_json_file, "test_json")

        assert result["name"] == "test_json"
        assert result["rows"] == 100

    def test_load_excel_file(self, analysis_context, sample_excel_file):
        """Should load Excel file successfully."""
        result = analysis_context.load_dataset(sample_excel_file, "test_excel")

        assert result["name"] == "test_excel"
        assert result["rows"] == 100

    def test_load_parquet_file(self, analysis_context, sample_parquet_file):
        """Should load Parquet file successfully."""
        result = analysis_context.load_dataset(sample_parquet_file, "test_parquet")

        assert result["name"] == "test_parquet"
        assert result["rows"] == 100

    def test_load_with_default_name(self, analysis_context, sample_csv_file):
        """Should use filename as default dataset name."""
        result = analysis_context.load_dataset(sample_csv_file)

        assert result["name"] == "test_data"

    def test_load_file_not_found(self, analysis_context):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            analysis_context.load_dataset("/nonexistent/path/data.csv")

    def test_load_unsupported_format(self, analysis_context, tmp_path):
        """Should raise ValueError for unsupported file formats."""
        unsupported_file = tmp_path / "data.txt"
        unsupported_file.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported format"):
            analysis_context.load_dataset(str(unsupported_file))

    def test_load_returns_dtypes(self, analysis_context, sample_csv_file):
        """Should return column data types in result."""
        result = analysis_context.load_dataset(sample_csv_file)

        assert "dtypes" in result
        assert isinstance(result["dtypes"], dict)
        assert "id" in result["dtypes"]

    def test_load_returns_null_counts(self, analysis_context, tmp_path):
        """Should return null counts per column."""
        # Create file with nulls
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": ["x", None, "y", "z"]
        })
        csv_path = tmp_path / "nulls.csv"
        df.to_csv(csv_path, index=False)

        result = analysis_context.load_dataset(str(csv_path))

        assert result["null_counts"]["a"] == 1
        assert result["null_counts"]["b"] == 1

    def test_load_multiple_datasets(self, analysis_context, sample_csv_file, sample_json_file):
        """Should be able to load multiple datasets."""
        analysis_context.load_dataset(sample_csv_file, "first")
        analysis_context.load_dataset(sample_json_file, "second")

        assert len(analysis_context.datasets) == 2
        assert "first" in analysis_context.datasets
        assert "second" in analysis_context.datasets


class TestGetDataset:
    """Tests for retrieving loaded datasets."""

    def test_get_loaded_dataset(self, loaded_context):
        """Should return loaded dataset by name."""
        df = loaded_context.get_dataset("test_data")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_get_nonexistent_dataset(self, analysis_context):
        """Should raise ValueError for nonexistent dataset."""
        with pytest.raises(ValueError, match="not loaded"):
            analysis_context.get_dataset("nonexistent")

    def test_get_dataset_error_shows_available(self, loaded_context):
        """Error message should list available datasets."""
        with pytest.raises(ValueError, match="test_data"):
            loaded_context.get_dataset("wrong_name")

    def test_get_dataset_returns_mutable_reference(self, loaded_context):
        """Returned dataset should be a mutable reference to the stored DataFrame."""
        df = loaded_context.get_dataset("test_data")
        # Add a new column to modify the DataFrame in-place
        df["new_col"] = "test"

        # Retrieve the dataset again
        df_again = loaded_context.get_dataset("test_data")

        # Check if the modification is present
        assert "new_col" in df_again.columns
