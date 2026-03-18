from collections import namedtuple
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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
    """Detect trend using linear regression.

    Handles pd.Series by coercing to numeric and dropping NaN values before
    fitting a linear regression model. Returns a dict with 'trend' direction
    ('increasing', 'decreasing', or 'flat'), 'slope', and 'p_value'.

    Raises:
        ValueError: If the cleaned data has fewer than 2 valid data points.
    """
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    else:
        numeric_array = np.asarray(values, dtype=float)
        arr = numeric_array[~np.isnan(numeric_array)]

    if arr.size < 2:
        raise ValueError("At least two valid data points are required to detect a trend.")

    x = np.arange(len(arr))
    slope, _intercept, _r_value, p_value, _std_err = stats.linregress(x, arr)
    if p_value >= 0.05:
        trend = "flat"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    return {"trend": trend, "slope": float(slope), "p_value": float(p_value)}

def test_correlation_significance(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value for two series or arrays.

    This function validates and cleans the input data by:
    - Converting to numeric types where possible
    - Dropping NaN values (aligned across both inputs)
    - Ensuring the inputs have matching lengths after cleaning
    - Requiring at least two valid data points

    Raises:
        ValueError: If the cleaned inputs have different lengths or fewer than two points.
    """
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        df = pd.concat([x, y], axis=1, keys=["x", "y"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        x_clean = df["x"].to_numpy()
        y_clean = df["y"].to_numpy()
    else:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must have the same length to compute Pearson correlation.")

        valid_mask = ~np.isnan(x_array) & ~np.isnan(y_array)
        x_clean = x_array[valid_mask]
        y_clean = y_array[valid_mask]

    if x_clean.size < 2 or y_clean.size < 2:
        raise ValueError("At least two valid data points are required to compute Pearson correlation.")

    corr, p_value = stats.pearsonr(x_clean, y_clean)
    return float(corr), float(p_value)
