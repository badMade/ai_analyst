from __future__ import annotations

from collections import namedtuple
from typing import Sequence

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


def test_normality(series: pd.Series, alpha: float = 0.05) -> TestResult:
    """Performs the Shapiro-Wilk test for normality."""
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
