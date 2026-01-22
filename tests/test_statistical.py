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
    assert stats["std"] == pytest.approx(1.58113883)
    assert stats["min"] == 1
    assert stats["max"] == 5


def test_test_normality():
    """Test normality testing function."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = test_normality(series)

    assert result.test_name == "Shapiro-Wilk"
    assert result.statistic == 0.99
    assert result.p_value == 0.5
    assert result.significant is False
    assert result.interpretation == "Normal"


def test_correlation_significance():
    """Test correlation significance testing."""
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])

    correlation, p_value = test_correlation_significance(x, y)

    assert correlation == 0.5
    assert p_value == 0.001


def test_detect_trend():
    """Test trend detection."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = detect_trend(values)

    assert result["trend"] == "increasing"
    assert result["slope"] == 0.1
    assert result["p_value"] == 0.05
