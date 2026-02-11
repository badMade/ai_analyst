from collections import namedtuple
from scipy import stats
from typing import Any

NormalityResult = namedtuple("NormalityResult", ["test_name", "statistic", "p_value", "significant", "interpretation"])
TrendResult = namedtuple("TrendResult", ["slope", "intercept", "p_value", "trend"])

def compute_descriptive_stats(series: pd.Series | np.ndarray) -> dict[str, Any]:
    return series.describe().to_dict()

def test_normality(series: pd.Series | np.ndarray) -> NormalityResult:
    if isinstance(series, pd.Series):
        numeric_values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    else:
        numeric_array = np.asarray(series, dtype=float)
        numeric_values = numeric_array[~np.isnan(numeric_array)]

    if len(numeric_values) < 3:
        return NormalityResult("Shapiro-Wilk", 0.0, 1.0, False, "Insufficient data")

    stat, p = stats.shapiro(numeric_values)
    return NormalityResult(
        "Shapiro-Wilk",
        float(stat),
        float(p),
        bool(p < 0.05),
        "Normal" if p >= 0.05 else "Not Normal",
    )

def detect_trend(values: pd.Series | np.ndarray) -> dict[str, float | str]:
    # Mock trend detection
    return {"trend": "flat", "p_value": 1.0}

def test_correlation_significance(df: pd.DataFrame) -> None:
    pass
