"""
Tests for statistical analysis functions.

Tests the statistical helper functions used by the analyst.
"""

import numpy as np
import pandas as pd
import pytest


class TestComputeDescriptiveStats:
    """Tests for compute_descriptive_stats function."""

    def test_returns_mean(self):
        """Should compute mean correctly."""
        from ai_analyst.tools.statistical import compute_descriptive_stats

        series = pd.Series([1, 2, 3, 4, 5])
        result = compute_descriptive_stats(series)

        assert result["mean"] == 3.0

    def test_returns_std(self):
        """Should compute standard deviation."""
        from ai_analyst.tools.statistical import compute_descriptive_stats

        series = pd.Series([1, 2, 3, 4, 5])
        result = compute_descriptive_stats(series)

        assert "std" in result
        assert isinstance(result["std"], float)

    def test_returns_min_max(self):
        """Should compute min and max."""
        from ai_analyst.tools.statistical import compute_descriptive_stats

        series = pd.Series([10, 20, 30, 40, 50])
        result = compute_descriptive_stats(series)

        assert result["min"] == 10
        assert result["max"] == 50

    def test_handles_single_value(self):
        """Should handle series with single value."""
        from ai_analyst.tools.statistical import compute_descriptive_stats

        series = pd.Series([42])
        result = compute_descriptive_stats(series)

        assert result["mean"] == 42
        assert result["min"] == 42
        assert result["max"] == 42

    def test_handles_negative_values(self):
        """Should handle negative values."""
        from ai_analyst.tools.statistical import compute_descriptive_stats

        series = pd.Series([-10, -5, 0, 5, 10])
        result = compute_descriptive_stats(series)

        assert result["mean"] == 0.0
        assert result["min"] == -10
        assert result["max"] == 10


class TestTestNormality:
    """Tests for test_normality function."""

    def test_returns_test_result(self):
        """Should return a TestResult namedtuple."""
        from ai_analyst.tools.statistical import test_normality

        series = pd.Series(np.random.randn(100))
        result = test_normality(series)

        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "significant")
        assert hasattr(result, "interpretation")

    def test_returns_shapiro_wilk(self):
        """Should use Shapiro-Wilk test."""
        from ai_analyst.tools.statistical import test_normality

        series = pd.Series(np.random.randn(50))
        result = test_normality(series)

        assert result.test_name == "Shapiro-Wilk"

    def test_result_has_numeric_values(self):
        """Result should have numeric statistic and p_value."""
        from ai_analyst.tools.statistical import test_normality

        series = pd.Series([1, 2, 3, 4, 5])
        result = test_normality(series)

        assert isinstance(result.statistic, (int, float))
        assert isinstance(result.p_value, (int, float))


class TestTestCorrelationSignificance:
    """Tests for test_correlation_significance function."""

    def test_returns_tuple(self):
        """Should return correlation and p-value tuple."""
        from ai_analyst.tools.statistical import test_correlation_significance

        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])

        result = test_correlation_significance(x, y)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_numeric_values(self):
        """Should return numeric correlation and p-value."""
        from ai_analyst.tools.statistical import test_correlation_significance

        x = pd.Series([1, 2, 3])
        y = pd.Series([1, 2, 3])

        corr, p_value = test_correlation_significance(x, y)

        assert isinstance(corr, (int, float))
        assert isinstance(p_value, (int, float))


class TestDetectTrend:
    """Tests for detect_trend function."""

    def test_returns_dict(self):
        """Should return a dictionary with trend info."""
        from ai_analyst.tools.statistical import detect_trend

        values = np.array([1, 2, 3, 4, 5])
        result = detect_trend(values)

        assert isinstance(result, dict)

    def test_contains_trend_key(self):
        """Result should contain trend direction."""
        from ai_analyst.tools.statistical import detect_trend

        values = np.array([1, 2, 3, 4, 5])
        result = detect_trend(values)

        assert "trend" in result

    def test_contains_slope(self):
        """Result should contain slope."""
        from ai_analyst.tools.statistical import detect_trend

        values = np.array([1, 2, 3, 4, 5])
        result = detect_trend(values)

        assert "slope" in result

    def test_contains_p_value(self):
        """Result should contain p-value."""
        from ai_analyst.tools.statistical import detect_trend

        values = np.array([1, 2, 3, 4, 5])
        result = detect_trend(values)

        assert "p_value" in result


class TestTestResultNamedTuple:
    """Tests for TestResult namedtuple."""

    def test_create_test_result(self):
        """Should be able to create TestResult."""
        from ai_analyst.tools.statistical import TestResult

        result = TestResult(
            test_name="Test",
            statistic=0.95,
            p_value=0.05,
            significant=True,
            interpretation="Significant"
        )

        assert result.test_name == "Test"
        assert result.statistic == 0.95
        assert result.p_value == 0.05
        assert result.significant is True
        assert result.interpretation == "Significant"

    def test_test_result_is_immutable(self):
        """TestResult should be immutable (namedtuple behavior)."""
        from ai_analyst.tools.statistical import TestResult

        result = TestResult("Test", 0.5, 0.05, False, "Not significant")

        with pytest.raises(AttributeError):
            result.statistic = 0.99


class TestComputeDataframeStats:
    """Tests for compute_dataframe_stats function."""

    def test_computes_stats_for_dataframe(self):
        """Should compute stats for all numeric columns."""
        from ai_analyst.tools.statistical import compute_dataframe_stats

        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "w", "v"]
        })

        result = compute_dataframe_stats(df)

        assert len(result) == 2

        # Check column 'a'
        stats_a = next(r for r in result if r["column"] == "a")
        assert stats_a["mean"] == 3.0
        assert stats_a["min"] == 1.0
        assert stats_a["max"] == 5.0

        # Check column 'b'
        stats_b = next(r for r in result if r["column"] == "b")
        assert stats_b["mean"] == 30.0
        assert stats_b["min"] == 10.0
        assert stats_b["max"] == 50.0

    def test_handles_empty_dataframe(self):
        """Should return empty list for empty DataFrame."""
        from ai_analyst.tools.statistical import compute_dataframe_stats

        df = pd.DataFrame()
        result = compute_dataframe_stats(df)
        assert result == []

    def test_handles_no_numeric_columns(self):
        """Should return empty list if no numeric columns."""
        from ai_analyst.tools.statistical import compute_dataframe_stats

        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        result = compute_dataframe_stats(df)
        assert result == []

    def test_handles_missing_values(self):
        """Should handle missing values correctly."""
        from ai_analyst.tools.statistical import compute_dataframe_stats

        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5]
        })

        result = compute_dataframe_stats(df)
        stats = result[0]

        assert stats["count"] == 4.0
        assert stats["mean"] == 3.0
