"""
Integration tests for AI Analyst.

End-to-end tests that verify the complete workflow.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd


class TestEndToEndAnalysis:
    """End-to-end tests for the analysis workflow."""

    @pytest.fixture
    def sample_sales_file(self, tmp_path: Path) -> str:
        """Create and return path to sample sales data CSV."""
        csv_path = tmp_path / "sample_sales.csv"

        num_rows = 30
        products = [f"Product {i % 5}" for i in range(num_rows)]
        quantities = [(i % 10) + 1 for i in range(num_rows)]
        prices = [10.0 + (i % 5) * 2.5 for i in range(num_rows)]
        categories = [f"Category {i % 3}" for i in range(num_rows)]

        df = pd.DataFrame(
            {
                "order_id": list(range(1, num_rows + 1)),
                "product": products,
                "quantity": quantities,
                "price": prices,
                "category": categories,
            }
        )
        df["revenue"] = df["quantity"] * df["price"]

        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def analyst_with_mock_api(self, mock_settings):
        """Create analyst with mocked API client."""
        with patch("anthropic.Anthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            analyst.client = MagicMock()
            return analyst

    def test_load_and_preview_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test loading data and previewing it."""
        analyst = analyst_with_mock_api

        # Load dataset
        load_result = analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })
        load_data = json.loads(load_result)

        assert load_data["name"] == "sales"
        assert load_data["rows"] == 30

        # Preview data
        preview_result = analyst._execute_tool("preview_data", {
            "dataset_name": "sales",
            "n_rows": 5
        })
        preview_data = json.loads(preview_result)

        assert len(preview_data["data"]) == 5
        assert "product" in preview_data["columns"]

    def test_statistical_analysis_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test complete statistical analysis workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Describe statistics
        stats_result = analyst._execute_tool("describe_statistics", {
            "dataset_name": "sales"
        })
        stats_data = json.loads(stats_result)

        assert "statistics" in stats_data
        assert len(stats_data["statistics"]) > 0

        # Check for expected columns in stats
        stat_columns = {s["column"] for s in stats_data["statistics"]}
        assert "quantity" in stat_columns or "price" in stat_columns

    def test_correlation_analysis_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test correlation analysis workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Compute correlations
        corr_result = analyst._execute_tool("compute_correlation", {
            "dataset_name": "sales",
            "method": "pearson"
        })
        corr_data = json.loads(corr_result)

        assert "correlations" in corr_data
        assert corr_data["method"] == "pearson"

    def test_data_quality_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test data quality check workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Check quality
        quality_result = analyst._execute_tool("check_data_quality", {
            "dataset_name": "sales"
        })
        quality_data = json.loads(quality_result)

        assert quality_data["total_rows"] == 30
        assert "quality_score" in quality_data
        assert quality_data["quality_score"] <= 100

    def test_group_analysis_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test grouped analysis workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Group analysis
        group_result = analyst._execute_tool("group_analysis", {
            "dataset_name": "sales",
            "group_by": "category",
            "agg_column": "revenue"
        })
        group_data = json.loads(group_result)

        assert group_data["group_by"] == "category"
        assert group_data["agg_column"] == "revenue"
        assert group_data["n_groups"] > 0

    def test_outlier_detection_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test outlier detection workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Detect outliers
        outlier_result = analyst._execute_tool("detect_outliers", {
            "dataset_name": "sales",
            "column": "revenue",
            "method": "iqr"
        })
        outlier_data = json.loads(outlier_result)

        assert "outlier_count" in outlier_data
        assert "lower_bound" in outlier_data
        assert "upper_bound" in outlier_data

    def test_trend_analysis_workflow(self, analyst_with_mock_api, sample_sales_file):
        """Test trend analysis workflow."""
        analyst = analyst_with_mock_api

        # Load
        analyst._execute_tool("load_dataset", {
            "file_path": sample_sales_file,
            "name": "sales"
        })

        # Analyze trend
        trend_result = analyst._execute_tool("analyze_trend", {
            "dataset_name": "sales",
            "column": "revenue"
        })
        trend_data = json.loads(trend_result)

        assert "trend" in trend_data


class TestMultipleDatasets:
    """Tests for working with multiple datasets."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst with mocked API."""
        with patch("anthropic.Anthropic"):
            from analyst import StandaloneAnalyst

            analyst = StandaloneAnalyst()
            return analyst

    def test_load_multiple_datasets(self, analyst, tmp_path):
        """Should be able to load and work with multiple datasets."""
        # Create two datasets
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"x": [10, 20], "y": [30, 40]})

        file1 = tmp_path / "data1.csv"
        file2 = tmp_path / "data2.csv"
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        # Load both
        analyst._execute_tool("load_dataset", {
            "file_path": str(file1),
            "name": "first"
        })
        analyst._execute_tool("load_dataset", {
            "file_path": str(file2),
            "name": "second"
        })

        # List datasets
        list_result = analyst._execute_tool("list_datasets", {})
        list_data = json.loads(list_result)

        assert list_data["count"] == 2
        assert "first" in list_data["datasets"]
        assert "second" in list_data["datasets"]

    def test_switch_between_datasets(self, analyst, tmp_path):
        """Should be able to switch between loaded datasets."""
        # Create datasets
        df1 = pd.DataFrame({"value": [100, 200, 300]})
        df2 = pd.DataFrame({"value": [1, 2, 3]})

        file1 = tmp_path / "big.csv"
        file2 = tmp_path / "small.csv"
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        analyst._execute_tool("load_dataset", {"file_path": str(file1), "name": "big"})
        analyst._execute_tool("load_dataset", {"file_path": str(file2), "name": "small"})

        # Analyze first
        stats1 = json.loads(analyst._execute_tool("describe_statistics", {
            "dataset_name": "big"
        }))

        # Analyze second
        stats2 = json.loads(analyst._execute_tool("describe_statistics", {
            "dataset_name": "small"
        }))

        # Values should be different
        mean1 = stats1["statistics"][0]["mean"]
        mean2 = stats2["statistics"][0]["mean"]
        assert mean1 != mean2


class TestErrorHandling:
    """Tests for error handling in workflows."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst with mocked API."""
        with patch("anthropic.Anthropic"):
            from analyst import StandaloneAnalyst

            return StandaloneAnalyst()

    def test_error_on_missing_dataset(self, analyst):
        """Should return error for operations on missing dataset."""
        result = analyst._execute_tool("preview_data", {
            "dataset_name": "nonexistent"
        })
        data = json.loads(result)

        assert "error" in data

    def test_error_on_invalid_column(self, analyst, tmp_path):
        """Should return error for invalid column name."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        file = tmp_path / "data.csv"
        df.to_csv(file, index=False)

        analyst._execute_tool("load_dataset", {"file_path": str(file), "name": "test"})

        result = analyst._execute_tool("detect_outliers", {
            "dataset_name": "test",
            "column": "nonexistent_column"
        })
        data = json.loads(result)

        assert "error" in data

    def test_error_on_invalid_file_format(self, analyst, tmp_path):
        """Should return error for unsupported file format."""
        file = tmp_path / "data.xyz"
        file.write_text("some data")

        result = analyst._execute_tool("load_dataset", {
            "file_path": str(file)
        })
        data = json.loads(result)

        assert "error" in data


class TestDataWithNulls:
    """Tests for handling data with null values."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst."""
        with patch("anthropic.Anthropic"):
            from analyst import StandaloneAnalyst

            return StandaloneAnalyst()

    def test_quality_check_detects_nulls(self, analyst, tmp_path):
        """Quality check should detect null values."""
        df = pd.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [1, 2, 3, 4, 5]
        })
        file = tmp_path / "nulls.csv"
        df.to_csv(file, index=False)

        analyst._execute_tool("load_dataset", {"file_path": str(file), "name": "nulls"})

        result = analyst._execute_tool("check_data_quality", {
            "dataset_name": "nulls"
        })
        data = json.loads(result)

        assert data["null_cells"] > 0
        assert data["null_percentage"] > 0

    def test_statistics_handle_nulls(self, analyst, tmp_path):
        """Statistics should handle null values gracefully."""
        df = pd.DataFrame({
            "value": [1, 2, None, 4, 5]
        })
        file = tmp_path / "data.csv"
        df.to_csv(file, index=False)

        analyst._execute_tool("load_dataset", {"file_path": str(file), "name": "data"})

        result = analyst._execute_tool("describe_statistics", {
            "dataset_name": "data"
        })
        data = json.loads(result)

        # Should compute stats ignoring nulls
        assert "statistics" in data
        stats = data["statistics"][0]
        assert stats["mean"] == 3.0  # (1+2+4+5)/4


class TestDuplicateDetection:
    """Tests for duplicate row detection."""

    @pytest.fixture
    def analyst(self, mock_settings):
        """Create analyst."""
        with patch("anthropic.Anthropic"):
            from analyst import StandaloneAnalyst

            return StandaloneAnalyst()

    def test_quality_check_detects_duplicates(self, analyst, tmp_path):
        """Quality check should detect duplicate rows."""
        df = pd.DataFrame({
            "a": [1, 2, 2, 3, 3],
            "b": ["x", "y", "y", "z", "z"]
        })
        file = tmp_path / "dups.csv"
        df.to_csv(file, index=False)

        analyst._execute_tool("load_dataset", {"file_path": str(file), "name": "dups"})

        result = analyst._execute_tool("check_data_quality", {
            "dataset_name": "dups"
        })
        data = json.loads(result)

        assert data["duplicate_rows"] == 2
        assert data["duplicate_percentage"] == 40.0
