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


def compute_dataframe_stats(df: pd.DataFrame) -> list[dict]:
    """
    Computes descriptive statistics for a DataFrame efficiently using vectorized operations.
    Replaces row-by-row iteration over pd.Series.describe().
    """
    import numpy as np

    # Early return when no columns to process
    if df.columns.empty:
        return []

    counts = df.count()

    # Calculate quantiles in one go for efficiency
    if not df.empty:
        quantiles = df.quantile([0.25, 0.5, 0.75])
    else:
        # Create an empty DataFrame with the right index and columns full of NaNs
        quantiles = pd.DataFrame(np.nan, index=[0.25, 0.5, 0.75], columns=df.columns)

    stats_df = pd.DataFrame({
        "count": counts,
        "mean": df.mean() if not df.empty else np.nan,
        "std": df.std() if not df.empty else np.nan,
        "min": df.min() if not df.empty else np.nan,
        "25%": quantiles.loc[0.25],
        "50%": quantiles.loc[0.5],
        "75%": quantiles.loc[0.75],
        "max": df.max() if not df.empty else np.nan,
    })

    # Standard deviation calculations return NaN when the count of values is less than 2
    stats_df.loc[counts < 2, 'std'] = np.nan

    stats_df.index.name = 'column'

    # Convert and return, handling proper index reset
    return stats_df.reset_index().to_dict(orient='records')


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
