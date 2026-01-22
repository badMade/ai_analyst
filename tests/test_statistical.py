import pytest
import pandas as pd
from ai_analyst.tools.statistical import (
    compute_descriptive_stats,
    test_normality,
    test_correlation_significance,
    detect_trend,
)


def test_compute_descriptive_stats():
    """Test descriptive statistics computation."""
    series = pd.Series([1, 2, 3, 4, 5])
    stats = compute_descriptive_stats(series)

    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert stats["mean"] == 3.0
    assert stats["min"] == 1
    assert stats["max"] == 5


def test_test_normality():
    """Test normality testing function."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = test_normality(series)

    assert result.test_name == "Shapiro-Wilk"
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "significant")
    assert hasattr(result, "interpretation")


def test_correlation_significance():
    """Test correlation significance testing."""
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])

    correlation, p_value = test_correlation_significance(x, y)

    assert isinstance(correlation, (int, float))
    assert isinstance(p_value, (int, float))


def test_detect_trend():
    """Test trend detection."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = detect_trend(values)

    assert "trend" in result
    assert "slope" in result
    assert "p_value" in result
