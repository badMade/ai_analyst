"""
Tests for AnalysisContext class.

Tests data loading, dataset management, and error handling.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestAnalysisContextInit:
    """Tests for AnalysisContext initialization."""

    def test_init_creates_empty_datasets(self, analysis_context):
        """AnalysisContext initializes with empty datasets dict."""
        assert analysis_context.datasets == {}

    def test_init_creates_empty_results(self, analysis_context):
        """AnalysisContext initializes with empty results list."""
        assert analysis_context.results == []


class TestLoadDatasetCSV:
    """Tests for loading CSV files."""

    def test_load_csv_success(self, analysis_context, sample_csv_file):
        """Successfully load a CSV file."""
        result = analysis_context.load_dataset(str(sample_csv_file), "test")

        assert result["name"] == "test"
        assert result["rows"] == 5
        assert result["columns"] == 5
        assert "id" in result["column_names"]
        assert "name" in result["column_names"]

    def test_load_csv_uses_filename_as_default_name(self, analysis_context, sample_csv_file):
        """When name not provided, uses filename stem."""
        result = analysis_context.load_dataset(str(sample_csv_file))

        assert result["name"] == sample_csv_file.stem
        assert sample_csv_file.stem in analysis_context.datasets

    def test_load_csv_reports_dtypes(self, analysis_context, sample_csv_file):
        """Load result includes column dtypes."""
        result = analysis_context.load_dataset(str(sample_csv_file), "test")

        assert "dtypes" in result
        assert "id" in result["dtypes"]

    def test_load_csv_reports_null_counts(self, analysis_context, sample_csv_file):
        """Load result includes null counts per column."""
        result = analysis_context.load_dataset(str(sample_csv_file), "test")

        assert "null_counts" in result
        assert all(v == 0 for v in result["null_counts"].values())


class TestLoadDatasetJSON:
    """Tests for loading JSON files."""

    def test_load_json_success(self, analysis_context, sample_json_file):
        """Successfully load a JSON file."""
        result = analysis_context.load_dataset(str(sample_json_file), "json_data")

        assert result["name"] == "json_data"
        assert result["rows"] == 5


class TestLoadDatasetExcel:
    """Tests for loading Excel files."""

    def test_load_excel_success(self, analysis_context, sample_excel_file):
        """Successfully load an Excel file."""
        result = analysis_context.load_dataset(str(sample_excel_file), "excel_data")

        assert result["name"] == "excel_data"
        assert result["rows"] == 5


class TestLoadDatasetParquet:
    """Tests for loading Parquet files."""

    def test_load_parquet_success(self, analysis_context, sample_parquet_file):
        """Successfully load a Parquet file."""
        result = analysis_context.load_dataset(str(sample_parquet_file), "parquet_data")

        assert result["name"] == "parquet_data"
        assert result["rows"] == 5


class TestLoadDatasetErrors:
    """Tests for error handling during data loading."""

    def test_load_nonexistent_file_raises_error(self, analysis_context):
        """Loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            analysis_context.load_dataset("/nonexistent/path/file.csv", "test")

    def test_load_unsupported_format_raises_error(self, analysis_context):
        """Loading unsupported file format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                analysis_context.load_dataset(str(path), "test")
        finally:
            path.unlink(missing_ok=True)

    def test_load_empty_csv(self, analysis_context):
        """Loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")  # Headers only
            path = Path(f.name)

        try:
            result = analysis_context.load_dataset(str(path), "empty")
            assert result["rows"] == 0
            assert result["columns"] == 2
        finally:
            path.unlink(missing_ok=True)


class TestGetDataset:
    """Tests for retrieving loaded datasets."""

    def test_get_dataset_success(self, context_with_dataset):
        """Successfully retrieve a loaded dataset."""
        df = context_with_dataset.get_dataset("test_data")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_get_nonexistent_dataset_raises_error(self, analysis_context):
        """Retrieving non-existent dataset raises ValueError."""
        with pytest.raises(ValueError, match="not loaded"):
            analysis_context.get_dataset("nonexistent")

    def test_get_dataset_error_lists_available(self, context_with_dataset):
        """Error message includes list of available datasets."""
        with pytest.raises(ValueError) as exc_info:
            context_with_dataset.get_dataset("wrong_name")

        assert "test_data" in str(exc_info.value)


class TestMultipleDatasets:
    """Tests for managing multiple datasets."""

    def test_load_multiple_datasets(self, analysis_context, sample_csv_file, sample_json_file):
        """Can load and access multiple datasets."""
        analysis_context.load_dataset(str(sample_csv_file), "csv_data")
        analysis_context.load_dataset(str(sample_json_file), "json_data")

        assert len(analysis_context.datasets) == 2
        assert "csv_data" in analysis_context.datasets
        assert "json_data" in analysis_context.datasets

    def test_overwrite_dataset_with_same_name(self, analysis_context, sample_csv_file):
        """Loading with same name overwrites existing dataset."""
        analysis_context.load_dataset(str(sample_csv_file), "data")

        # Modify and reload
        analysis_context.load_dataset(str(sample_csv_file), "data")

        assert len(analysis_context.datasets) == 1


class TestDataWithNulls:
    """Tests for handling data with null values."""

    def test_load_data_with_nulls_reports_counts(self, analysis_context, df_with_nulls):
        """Null counts are correctly reported."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df_with_nulls.to_csv(f, index=False)
            path = Path(f.name)

        try:
            result = analysis_context.load_dataset(str(path), "nulls")

            assert result["null_counts"]["value"] == 2
            assert result["null_counts"]["category"] == 1
        finally:
            path.unlink(missing_ok=True)
