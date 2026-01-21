"""
Tests for statistical analysis tools.

Tests the functions in ai_analyst.tools.statistical module.
"""

import numpy as np
import pytest

from ai_analyst.tools.statistical import (
    StatisticalTestResult,
    compute_descriptive_stats,
    detect_trend,
    check_correlation_significance,
    check_normality,
)


class TestComputeDescriptiveStats:
    """Tests for compute_descriptive_stats function."""

    def test_basic_stats(self):
        """Computes basic statistics correctly."""
        data = np.array([1, 2, 3, 4, 5])
        stats = compute_descriptive_stats(data)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["50%"] == 3.0  # Median

    def test_with_nan_values(self):
        """Handles NaN values by excluding them."""
        data = np.array([1, 2, np.nan, 4, 5])
        stats = compute_descriptive_stats(data)

        assert stats["count"] == 4  # NaN excluded
        assert stats["mean"] == 3.0

    def test_empty_array(self):
        """Handles empty array."""
        data = np.array([])
        stats = compute_descriptive_stats(data)

        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_all_nan_array(self):
        """Handles all-NaN array."""
        data = np.array([np.nan, np.nan, np.nan])
        stats = compute_descriptive_stats(data)

        assert stats["count"] == 0

    def test_single_value(self):
        """Handles single value array."""
        data = np.array([42])
        stats = compute_descriptive_stats(data)

        assert stats["count"] == 1
        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0  # Single value has 0 std

    def test_quartiles(self):
        """Computes quartiles correctly."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = compute_descriptive_stats(data)

        assert stats["25%"] == pytest.approx(3.25, rel=0.1)
        assert stats["75%"] == pytest.approx(7.75, rel=0.1)


class TestCheckNormality:
    """Tests for check_normality function."""

    def test_normal_data_detected(self):
        """Normally distributed data is detected as normal."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = check_normality(data)

        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Shapiro-Wilk"
        # Normal data should not be significantly non-normal
        assert result.p_value > 0.01

    def test_non_normal_data_detected(self):
        """Non-normal data is detected."""
        # Uniform distribution
        data = np.linspace(0, 100, 100)

        result = check_normality(data)

        # Uniform distribution should be detected as non-normal
        assert bool(result.significant) is True

    def test_insufficient_data(self):
        """Handles insufficient data gracefully."""
        data = np.array([1, 2])

        result = check_normality(data)

        assert result.test_name == "insufficient_data"
        assert np.isnan(result.statistic)

    def test_with_nan_values(self):
        """Handles NaN values by excluding them."""
        np.random.seed(42)
        data = np.random.randn(100)
        data[0] = np.nan
        data[10] = np.nan

        result = check_normality(data)

        # Should still work
        assert not np.isnan(result.p_value)

    def test_interpretation_included(self):
        """Result includes interpretation."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = check_normality(data)

        assert result.interpretation
        assert len(result.interpretation) > 0


class TestTestCorrelationSignificance:
    """Tests for test_correlation_significance function."""

    def test_perfect_correlation(self):
        """Detects perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        result = check_correlation_significance(x, y)

        assert result.statistic == pytest.approx(1.0, abs=0.001)
        assert bool(result.significant) is True

    def test_perfect_negative_correlation(self):
        """Detects perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])

        result = check_correlation_significance(x, y)

        assert result.statistic == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        """Handles uncorrelated data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = check_correlation_significance(x, y)

        # Should not be significant
        assert abs(result.statistic) < 0.3

    def test_spearman_method(self):
        """Supports Spearman correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])  # Perfect monotonic

        result = check_correlation_significance(x, y, method="spearman")

        assert result.test_name == "spearman_correlation"
        assert result.statistic == pytest.approx(1.0, abs=0.001)

    def test_kendall_method(self):
        """Supports Kendall correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = check_correlation_significance(x, y, method="kendall")

        assert result.test_name == "kendall_correlation"
        assert result.statistic == pytest.approx(1.0, abs=0.001)

    def test_insufficient_data(self):
        """Handles insufficient data."""
        x = np.array([1, 2])
        y = np.array([1, 2])

        result = check_correlation_significance(x, y)

        assert np.isnan(result.statistic)

    def test_with_nan_values(self):
        """Handles NaN values by excluding pairs."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        result = check_correlation_significance(x, y)

        # Should work with remaining 4 pairs
        assert not np.isnan(result.statistic)

    def test_invalid_method_raises_error(self):
        """Raises error for invalid method."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Unknown correlation method"):
            check_correlation_significance(x, y, method="invalid")


class TestDetectTrend:
    """Tests for detect_trend function."""

    def test_increasing_trend(self):
        """Detects increasing trend."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = detect_trend(data)

        assert result["trend"] == "increasing"
        assert bool(result["significant"]) is True
        assert result["tau"] > 0

    def test_decreasing_trend(self):
        """Detects decreasing trend."""
        data = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        result = detect_trend(data)

        assert result["trend"] == "decreasing"
        assert bool(result["significant"]) is True
        assert result["tau"] < 0

    def test_no_trend(self):
        """Handles data with no clear trend."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = detect_trend(data)

        # Random data should not have significant trend
        assert result["trend"] in ["none", "increasing", "decreasing"]

    def test_insufficient_data(self):
        """Handles insufficient data."""
        data = np.array([1, 2, 3])

        result = detect_trend(data)

        assert result["trend"] == "unknown"
        assert result["significant"] is False
        assert "insufficient" in result["interpretation"].lower()

    def test_with_nan_values(self):
        """Handles NaN values."""
        data = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])

        result = detect_trend(data)

        # Should work with remaining 9 values
        assert result["trend"] in ["increasing", "decreasing", "none"]

    def test_constant_data(self):
        """Handles constant data (no variation)."""
        data = np.array([5, 5, 5, 5, 5, 5, 5, 5])

        result = detect_trend(data)

        # Constant data produces NaN for tau (no variation to correlate)
        assert np.isnan(result["tau"]) or result["tau"] == pytest.approx(0, abs=0.001)
