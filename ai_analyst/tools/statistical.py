"""
Statistical analysis tools.

Provides functions for descriptive statistics, normality testing,
correlation significance, and trend detection.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str


def compute_descriptive_stats(series: "np.ndarray | Any") -> dict:
    """
    Compute descriptive statistics for a numeric series.

    Args:
        series: Numeric data series (pandas Series or numpy array)

    Returns:
        Dictionary with count, mean, std, min, max, and quartiles
    """
    arr = np.asarray(series)
    arr = arr[~np.isnan(arr)]  # Remove NaN values

    if len(arr) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "25%": None,
            "50%": None,
            "75%": None,
        }

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "25%": float(np.percentile(arr, 25)),
        "50%": float(np.percentile(arr, 50)),
        "75%": float(np.percentile(arr, 75)),
    }


def check_normality(series: "np.ndarray | Any", alpha: float = 0.05) -> StatisticalTestResult:
    """
    Test if a series follows a normal distribution.

    Uses Shapiro-Wilk test for n < 5000, otherwise uses D'Agostino-Pearson.

    Args:
        series: Numeric data series
        alpha: Significance level (default 0.05)

    Returns:
        StatisticalTestResult with test details
    """
    arr = np.asarray(series)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 3:
        return StatisticalTestResult(
            test_name="insufficient_data",
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Insufficient data for normality test (need at least 3 values)"
        )

    if len(arr) < 5000:
        stat, p_value = stats.shapiro(arr)
        test_name = "Shapiro-Wilk"
    else:
        stat, p_value = stats.normaltest(arr)
        test_name = "D'Agostino-Pearson"

    significant = p_value < alpha

    if significant:
        interpretation = f"Data does not follow a normal distribution (p={p_value:.4f} < {alpha})"
    else:
        interpretation = f"Data appears normally distributed (p={p_value:.4f} >= {alpha})"

    return StatisticalTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        significant=significant,
        interpretation=interpretation
    )


def check_correlation_significance(
    x: "np.ndarray | Any",
    y: "np.ndarray | Any",
    method: str = "pearson",
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Test if correlation between two variables is statistically significant.

    Args:
        x: First variable
        y: Second variable
        method: Correlation method ('pearson', 'spearman', 'kendall')
        alpha: Significance level

    Returns:
        StatisticalTestResult with correlation and significance
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Remove pairs with NaN in either variable
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 3:
        return StatisticalTestResult(
            test_name=f"{method}_correlation",
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            interpretation="Insufficient data for correlation test"
        )

    if method == "pearson":
        stat, p_value = stats.pearsonr(x_arr, y_arr)
    elif method == "spearman":
        stat, p_value = stats.spearmanr(x_arr, y_arr)
    elif method == "kendall":
        stat, p_value = stats.kendalltau(x_arr, y_arr)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    significant = p_value < alpha

    if significant:
        interpretation = f"Significant {method} correlation of {stat:.4f} (p={p_value:.4f})"
    else:
        interpretation = f"No significant {method} correlation (r={stat:.4f}, p={p_value:.4f})"

    return StatisticalTestResult(
        test_name=f"{method}_correlation",
        statistic=float(stat),
        p_value=float(p_value),
        significant=significant,
        interpretation=interpretation
    )


def detect_trend(series: "np.ndarray | Any", alpha: float = 0.05) -> dict:
    """
    Detect monotonic trend using Mann-Kendall test.

    Args:
        series: Time-ordered numeric data
        alpha: Significance level

    Returns:
        Dictionary with trend direction, significance, and statistics
    """
    arr = np.asarray(series)
    arr = arr[~np.isnan(arr)]

    n = len(arr)

    if n < 4:
        return {
            "trend": "unknown",
            "significant": False,
            "p_value": None,
            "tau": None,
            "interpretation": "Insufficient data for trend detection (need at least 4 values)"
        }

    # Mann-Kendall test using scipy's kendalltau with indices
    indices = np.arange(n)
    tau, p_value = stats.kendalltau(indices, arr)

    significant = p_value < alpha

    if significant:
        if tau > 0:
            trend = "increasing"
            interpretation = f"Significant increasing trend detected (tau={tau:.4f}, p={p_value:.4f})"
        else:
            trend = "decreasing"
            interpretation = f"Significant decreasing trend detected (tau={tau:.4f}, p={p_value:.4f})"
    else:
        trend = "none"
        interpretation = f"No significant trend detected (tau={tau:.4f}, p={p_value:.4f})"

    return {
        "trend": trend,
        "significant": significant,
        "p_value": float(p_value),
        "tau": float(tau),
        "interpretation": interpretation
    }
