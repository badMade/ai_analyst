"""
Tests for statistical analysis functions.

Tests the statistical helper functions used by the analyst.
"""

import pytest
import pandas as pd
import numpy as np


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
        from ai_analyst.tools.statistical import test_normality, TestResult

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

    def test_detects_strong_positive_correlation(self):
        """Should return a strong positive correlation for linear data."""
        from ai_analyst.tools.statistical import test_correlation_significance

        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])

        corr, p_value = test_correlation_significance(x, y)

        assert corr == pytest.approx(1.0)
        assert p_value < 0.05


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
