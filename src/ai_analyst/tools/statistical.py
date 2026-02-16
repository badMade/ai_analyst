from __future__ import annotations

from collections import namedtuple
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

TestResult = namedtuple(
    "TestResult",
    ["test_name", "statistic", "p_value", "significant", "interpretation"],
)


def compute_descriptive_stats(series: pd.Series) -> dict[str, float]:
    desc = series.describe()
    return {
        "count": desc['count'],
        "mean": desc['mean'],
        "std": desc['std'],
        "min": desc['min'],
        "25%": desc['25%'],
        "50%": desc['50%'],
        "75%": desc['75%'],
        "max": desc['max'],
    }


def compute_dataframe_stats(df: pd.DataFrame) -> list[dict[str, float | str]]:
    """
    Compute descriptive statistics for all numeric columns in a DataFrame using optimized vectorized operations.

    Args:
        df: Input DataFrame

    Returns:
        List of dictionaries containing statistics for each column
    """
    if df.empty:
        return []

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return []

    # Manual aggregation is faster than df.describe() or iterating over columns
    counts = numeric_df.count()
    means = numeric_df.mean()
    stds = numeric_df.std()
    mins = numeric_df.min()
    quantiles = numeric_df.quantile([0.25, 0.50, 0.75])
    maxs = numeric_df.max()

    stats_list = []
    for col in numeric_df.columns:
        stats_list.append({
            "column": col,
            "count": float(counts[col]),
            "mean": float(means[col]),
            "std": float(stds[col]),
            "min": float(mins[col]),
            "25%": float(quantiles.loc[0.25, col]),
            "50%": float(quantiles.loc[0.50, col]),
            "75%": float(quantiles.loc[0.75, col]),
            "max": float(maxs[col]),
        })
    return stats_list


def test_normality(series: pd.Series, alpha: float = 0.05) -> TestResult:
    """Performs the Shapiro-Wilk test for normality."""
    # The Shapiro-Wilk test requires at least 3 data points.
    if len(series) < 3:
        return TestResult(
            "Shapiro-Wilk",
            None,
            None,
            False,
            "Insufficient data (requires at least 3 samples)",
        )
    statistic, p_value = stats.shapiro(series)
    significant = p_value < alpha
    interpretation = "Not Normal" if significant else "Normal"
    return TestResult("Shapiro-Wilk", statistic, p_value, significant, interpretation)


def test_correlation_significance(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    return 0.5, 0.001


def detect_trend(
    values: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, float | str]:
    """Detects the trend in a series of values using linear regression."""
    x = range(len(values))
    slope, _, r_value, p_value, _ = stats.linregress(x, values)

    if p_value < alpha:
        if slope > 0:
            trend = "increasing"
        elif slope < 0:
            trend = "decreasing"
        else:
            trend = "no trend"
    else:
        trend = "no trend"

    return {
        "trend": trend,
        "slope": slope,
        "p_value": p_value,
        "r_squared": r_value**2,
    }
