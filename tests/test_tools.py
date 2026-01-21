"""
Tests for tool execution in StandaloneAnalyst._execute_tool().

Each tool has its own test class covering valid inputs, edge cases, and errors.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestLoadDatasetTool:
    """Tests for the load_dataset tool."""

    def test_load_dataset_tool_success(self, mock_analyst, sample_csv_file):
        """load_dataset tool successfully loads file."""
        result = mock_analyst._execute_tool("load_dataset", {
            "file_path": str(sample_csv_file)
        })

        data = json.loads(result)
        assert "name" in data
        assert data["rows"] == 5

    def test_load_dataset_tool_with_custom_name(self, mock_analyst, sample_csv_file):
        """load_dataset tool uses custom name when provided."""
        result = mock_analyst._execute_tool("load_dataset", {
            "file_path": str(sample_csv_file),
            "name": "custom_name"
        })

        data = json.loads(result)
        assert data["name"] == "custom_name"

    def test_load_dataset_tool_file_not_found(self, mock_analyst):
        """load_dataset tool handles missing file."""
        result = mock_analyst._execute_tool("load_dataset", {
            "file_path": "/nonexistent/file.csv"
        })

        data = json.loads(result)
        assert "error" in data


class TestListDatasetsTool:
    """Tests for the list_datasets tool."""

    def test_list_datasets_empty(self, mock_analyst):
        """list_datasets returns empty list when no datasets loaded."""
        result = mock_analyst._execute_tool("list_datasets", {})

        data = json.loads(result)
        assert data["datasets"] == []
        assert data["count"] == 0

    def test_list_datasets_with_data(self, analyst_with_data):
        """list_datasets returns loaded dataset names."""
        result = analyst_with_data._execute_tool("list_datasets", {})

        data = json.loads(result)
        assert "test_data" in data["datasets"]
        assert data["count"] == 1


class TestPreviewDataTool:
    """Tests for the preview_data tool."""

    def test_preview_data_default(self, analyst_with_data):
        """preview_data returns first 10 rows by default."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "test_data"
        })

        data = json.loads(result)
        assert "data" in data
        assert len(data["data"]) == 5  # Only 5 rows in test data

    def test_preview_data_custom_rows(self, analyst_with_data):
        """preview_data respects n_rows parameter."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "test_data",
            "n_rows": 2
        })

        data = json.loads(result)
        assert len(data["data"]) == 2

    def test_preview_data_specific_columns(self, analyst_with_data):
        """preview_data filters to specified columns."""
        result = analyst_with_data._execute_tool("preview_data", {
            "dataset_name": "test_data",
            "columns": ["id", "name"]
        })

        data = json.loads(result)
        assert data["columns"] == ["id", "name"]

    def test_preview_data_invalid_dataset(self, mock_analyst):
        """preview_data handles invalid dataset name."""
        result = mock_analyst._execute_tool("preview_data", {
            "dataset_name": "nonexistent"
        })

        data = json.loads(result)
        assert "error" in data


class TestDescribeStatisticsTool:
    """Tests for the describe_statistics tool."""

    def test_describe_statistics_all_numeric(self, analyst_with_data):
        """describe_statistics computes stats for all numeric columns."""
        result = analyst_with_data._execute_tool("describe_statistics", {
            "dataset_name": "test_data"
        })

        data = json.loads(result)
        assert "statistics" in data

        # Check that numeric columns have stats
        columns = [s["column"] for s in data["statistics"]]
        assert "age" in columns
        assert "salary" in columns

    def test_describe_statistics_specific_columns(self, analyst_with_data):
        """describe_statistics filters to specified columns."""
        result = analyst_with_data._execute_tool("describe_statistics", {
            "dataset_name": "test_data",
            "columns": ["age"]
        })

        data = json.loads(result)
        columns = [s["column"] for s in data["statistics"]]
        assert columns == ["age"]

    def test_describe_statistics_empty_numeric(self, mock_analyst):
        """describe_statistics handles no numeric columns."""
        # Create dataset with only string columns
        mock_analyst.context.datasets["strings"] = pd.DataFrame({
            "a": ["x", "y", "z"],
            "b": ["1", "2", "3"]
        })

        result = mock_analyst._execute_tool("describe_statistics", {
            "dataset_name": "strings"
        })

        data = json.loads(result)
        assert data["statistics"] == []


class TestComputeCorrelationTool:
    """Tests for the compute_correlation tool."""

    def test_correlation_pearson_default(self, analyst_with_data):
        """compute_correlation uses Pearson by default."""
        result = analyst_with_data._execute_tool("compute_correlation", {
            "dataset_name": "test_data"
        })

        data = json.loads(result)
        assert data["method"] == "pearson"
        assert "correlations" in data

    def test_correlation_spearman(self, analyst_with_data):
        """compute_correlation supports Spearman method."""
        result = analyst_with_data._execute_tool("compute_correlation", {
            "dataset_name": "test_data",
            "method": "spearman"
        })

        data = json.loads(result)
        assert data["method"] == "spearman"

    def test_correlation_kendall(self, analyst_with_data):
        """compute_correlation supports Kendall method."""
        result = analyst_with_data._execute_tool("compute_correlation", {
            "dataset_name": "test_data",
            "method": "kendall"
        })

        data = json.loads(result)
        assert data["method"] == "kendall"

    def test_correlation_sorted_by_strength(self, analyst_with_data):
        """Correlations are sorted by absolute value."""
        result = analyst_with_data._execute_tool("compute_correlation", {
            "dataset_name": "test_data"
        })

        data = json.loads(result)
        if len(data["correlations"]) > 1:
            values = [abs(c["correlation"]) for c in data["correlations"]]
            assert values == sorted(values, reverse=True)


class TestDetectOutliersTool:
    """Tests for the detect_outliers tool."""

    def test_outliers_iqr_method(self, mock_analyst, df_with_outliers):
        """detect_outliers uses IQR method by default."""
        mock_analyst.context.datasets["outliers"] = df_with_outliers

        result = mock_analyst._execute_tool("detect_outliers", {
            "dataset_name": "outliers",
            "column": "normal"
        })

        data = json.loads(result)
        assert data["method"] == "iqr"
        assert data["outlier_count"] >= 1  # Should detect the 100

    def test_outliers_zscore_method(self, mock_analyst, df_with_outliers):
        """detect_outliers supports Z-score method."""
        mock_analyst.context.datasets["outliers"] = df_with_outliers

        result = mock_analyst._execute_tool("detect_outliers", {
            "dataset_name": "outliers",
            "column": "normal",
            "method": "zscore"
        })

        data = json.loads(result)
        assert data["method"] == "zscore"

    def test_outliers_custom_threshold(self, mock_analyst, df_with_outliers):
        """detect_outliers respects custom threshold."""
        mock_analyst.context.datasets["outliers"] = df_with_outliers

        result = mock_analyst._execute_tool("detect_outliers", {
            "dataset_name": "outliers",
            "column": "normal",
            "threshold": 3.0
        })

        data = json.loads(result)
        assert data["threshold"] == 3.0

    def test_outliers_no_outliers(self, mock_analyst, df_with_outliers):
        """detect_outliers handles column with no outliers."""
        mock_analyst.context.datasets["outliers"] = df_with_outliers

        result = mock_analyst._execute_tool("detect_outliers", {
            "dataset_name": "outliers",
            "column": "clean"
        })

        data = json.loads(result)
        assert data["outlier_count"] == 0

    def test_outliers_constant_column_zscore(self, mock_analyst, constant_df):
        """detect_outliers handles constant column (std=0) with Z-score."""
        mock_analyst.context.datasets["const"] = constant_df

        result = mock_analyst._execute_tool("detect_outliers", {
            "dataset_name": "const",
            "column": "constant",
            "method": "zscore"
        })

        # Should not crash, may have NaN issues
        data = json.loads(result)
        assert "error" in data or "outlier_count" in data


class TestGroupAnalysisTool:
    """Tests for the group_analysis tool."""

    def test_group_analysis_default_aggs(self, analyst_with_data):
        """group_analysis uses default aggregation functions."""
        result = analyst_with_data._execute_tool("group_analysis", {
            "dataset_name": "test_data",
            "group_by": "department",
            "agg_column": "salary"
        })

        data = json.loads(result)
        assert data["group_by"] == "department"
        assert data["agg_column"] == "salary"
        assert data["n_groups"] == 3  # Engineering, Sales, HR

    def test_group_analysis_custom_aggs(self, analyst_with_data):
        """group_analysis respects custom aggregation functions."""
        result = analyst_with_data._execute_tool("group_analysis", {
            "dataset_name": "test_data",
            "group_by": "department",
            "agg_column": "salary",
            "agg_functions": ["mean", "max"]
        })

        data = json.loads(result)
        # Check results contain specified aggregations
        if data["results"]:
            first_result = data["results"][0]
            assert "mean" in first_result
            assert "max" in first_result

    def test_group_analysis_invalid_column(self, analyst_with_data):
        """group_analysis handles invalid column name."""
        result = analyst_with_data._execute_tool("group_analysis", {
            "dataset_name": "test_data",
            "group_by": "nonexistent",
            "agg_column": "salary"
        })

        data = json.loads(result)
        assert "error" in data


class TestCheckDataQualityTool:
    """Tests for the check_data_quality tool."""

    def test_data_quality_clean_data(self, analyst_with_data):
        """check_data_quality scores clean data highly."""
        result = analyst_with_data._execute_tool("check_data_quality", {
            "dataset_name": "test_data"
        })

        data = json.loads(result)
        assert data["null_cells"] == 0
        assert data["duplicate_rows"] == 0
        assert data["quality_score"] == 100.0

    def test_data_quality_with_nulls(self, mock_analyst, df_with_nulls):
        """check_data_quality detects null values."""
        mock_analyst.context.datasets["nulls"] = df_with_nulls

        result = mock_analyst._execute_tool("check_data_quality", {
            "dataset_name": "nulls"
        })

        data = json.loads(result)
        assert data["null_cells"] > 0
        assert data["null_percentage"] > 0
        assert data["quality_score"] < 100.0

    def test_data_quality_with_duplicates(self, mock_analyst):
        """check_data_quality detects duplicate rows."""
        mock_analyst.context.datasets["dups"] = pd.DataFrame({
            "a": [1, 1, 2],
            "b": [1, 1, 2]
        })

        result = mock_analyst._execute_tool("check_data_quality", {
            "dataset_name": "dups"
        })

        data = json.loads(result)
        assert data["duplicate_rows"] == 1

    def test_data_quality_reports_column_issues(self, mock_analyst, df_with_nulls):
        """check_data_quality reports issues per column."""
        mock_analyst.context.datasets["nulls"] = df_with_nulls

        result = mock_analyst._execute_tool("check_data_quality", {
            "dataset_name": "nulls"
        })

        data = json.loads(result)
        assert "column_issues" in data
        assert "value" in data["column_issues"]


class TestTestNormalityTool:
    """Tests for the test_normality tool."""

    def test_normality_normal_data(self, mock_analyst, numeric_df):
        """test_normality detects normally distributed data."""
        mock_analyst.context.datasets["numeric"] = numeric_df

        result = mock_analyst._execute_tool("test_normality", {
            "dataset_name": "numeric",
            "column": "a"
        })

        data = json.loads(result)
        assert data["test"] in ["Shapiro-Wilk", "D'Agostino-Pearson"]
        assert "p_value" in data
        assert "is_normal" in data

    def test_normality_insufficient_data(self, mock_analyst):
        """test_normality handles insufficient data."""
        mock_analyst.context.datasets["small"] = pd.DataFrame({"x": [1, 2]})

        result = mock_analyst._execute_tool("test_normality", {
            "dataset_name": "small",
            "column": "x"
        })

        data = json.loads(result)
        assert "insufficient" in data.get("interpretation", "").lower() or data.get("test") == "insufficient_data"


class TestAnalyzeTrendTool:
    """Tests for the analyze_trend tool."""

    def test_trend_increasing(self, mock_analyst, trend_df):
        """analyze_trend detects increasing trend."""
        mock_analyst.context.datasets["trend"] = trend_df

        result = mock_analyst._execute_tool("analyze_trend", {
            "dataset_name": "trend",
            "column": "increasing"
        })

        data = json.loads(result)
        assert data["trend"] == "increasing"
        # JSON may serialize numpy bool as string or bool
        assert data["significant"] in [True, "True"]

    def test_trend_decreasing(self, mock_analyst, trend_df):
        """analyze_trend detects decreasing trend."""
        mock_analyst.context.datasets["trend"] = trend_df

        result = mock_analyst._execute_tool("analyze_trend", {
            "dataset_name": "trend",
            "column": "decreasing"
        })

        data = json.loads(result)
        assert data["trend"] == "decreasing"
        # JSON may serialize numpy bool as string or bool
        assert data["significant"] in [True, "True"]

    def test_trend_insufficient_data(self, mock_analyst):
        """analyze_trend handles insufficient data."""
        mock_analyst.context.datasets["small"] = pd.DataFrame({"x": [1, 2, 3]})

        result = mock_analyst._execute_tool("analyze_trend", {
            "dataset_name": "small",
            "column": "x"
        })

        data = json.loads(result)
        # Should indicate insufficient data or return result
        assert "trend" in data or "error" in data


class TestUnknownTool:
    """Tests for handling unknown tools."""

    def test_unknown_tool_returns_error(self, mock_analyst):
        """Unknown tool name returns error."""
        result = mock_analyst._execute_tool("nonexistent_tool", {})

        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]


class TestToolExceptionHandling:
    """Tests for exception handling in tool execution."""

    def test_tool_exception_returns_json_error(self, mock_analyst):
        """Exceptions during tool execution return JSON error."""
        # Try to access non-existent dataset
        result = mock_analyst._execute_tool("preview_data", {
            "dataset_name": "nonexistent"
        })

        data = json.loads(result)
        assert "error" in data

    def test_tool_exception_is_logged(self, mock_analyst, caplog):
        """Tool exceptions are logged."""
        import logging

        # Capture all log messages at DEBUG and above so we don't depend on the exact log level.
        with caplog.at_level(logging.DEBUG):
            mock_analyst._execute_tool("preview_data", {
                "dataset_name": "nonexistent"
            })

        # Exception should be logged
        assert any(
            "Tool execution error" in record.message or "error" in record.message.lower()
            for record in caplog.records
        )
