from collections import namedtuple
from scipy import stats
from typing import Any

import numpy as np
import pandas as pd

TestResult = namedtuple("TestResult", ["test_name", "statistic", "p_value", "significant", "interpretation"])
TrendResult = namedtuple("TrendResult", ["slope", "intercept", "p_value", "trend"])

def compute_descriptive_stats(series: pd.Series | np.ndarray) -> dict[str, Any]:
    return series.describe().to_dict()

def test_normality(series: pd.Series | np.ndarray) -> TestResult:
    if isinstance(series, pd.Series):
        numeric_values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    else:
        numeric_array = np.asarray(series, dtype=float)
        numeric_values = numeric_array[~np.isnan(numeric_array)]

    if len(numeric_values) < 3:
        return TestResult("Shapiro-Wilk", 0.0, 1.0, False, "Insufficient data")

    stat, p = stats.shapiro(numeric_values)
    return TestResult(
        "Shapiro-Wilk",
        float(stat),
        float(p),
        bool(p < 0.05),
        "Normal" if p >= 0.05 else "Not Normal",
    )

def detect_trend(values: pd.Series | np.ndarray) -> dict[str, float | str]:
    """Detect trend using linear regression."""
    arr = np.asarray(values, dtype=float)
    x = np.arange(len(arr))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, arr)
    if p_value >= 0.05:
        trend = "flat"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    return {"trend": trend, "slope": float(slope), "p_value": float(p_value)}

def test_correlation_significance(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value for two series."""
    corr, p_value = stats.pearsonr(x, y)
    return float(corr), float(p_value)
