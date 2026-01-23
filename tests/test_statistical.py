"""
Tests for statistical analysis functions.

Tests the statistical helper functions used by the analyst.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ai_analyst.tools import statistical

def test_descriptive_stats():
    data = pd.Series([1, 2, 3, 4, 5])
    stats = statistical.compute_descriptive_stats(data)
    assert stats["mean"] == 3.0
    assert stats["std"] > 1.58 and stats["std"] < 1.59

def test_normality_normal_data():
    """Tests the normality test with normally distributed data."""
    np.random.seed(42)
    data = pd.Series(np.random.normal(loc=0, scale=1, size=100))
    result = statistical.test_normality(data)
    assert result.test_name == "Shapiro-Wilk"
    assert not result.significant
    assert result.interpretation == "Normal"

def test_normality_non_normal_data():
    """Tests the normality test with non-normally distributed data."""
    np.random.seed(42)
    data = pd.Series(np.random.uniform(low=0, high=1, size=100))
    result = statistical.test_normality(data)
    assert result.test_name == "Shapiro-Wilk"
    assert result.significant
    assert result.interpretation == "Not Normal"

def test_detect_trend_increasing():
    """Tests the trend detection with an increasing trend."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = statistical.detect_trend(data)
    assert result["trend"] == "increasing"
    assert result["slope"] > 0

def test_detect_trend_decreasing():
    """Tests the trend detection with a decreasing trend."""
    data = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    result = statistical.detect_trend(data)
    assert result["trend"] == "decreasing"
    assert result["slope"] < 0

def test_detect_trend_no_trend():
    """Tests the trend detection with no trend."""
    np.random.seed(42)
    data = pd.Series(np.random.normal(loc=0, scale=1, size=100))
    result = statistical.detect_trend(data)
    assert result["trend"] == "no trend"
