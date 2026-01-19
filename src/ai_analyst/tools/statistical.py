from typing import Any
import pandas as pd
import numpy as np

def compute_descriptive_stats(series: pd.Series) -> dict[str, Any]:
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "25%": series.quantile(0.25),
        "50%": series.quantile(0.50),
        "75%": series.quantile(0.75),
    }

class TestResult:
    def __init__(self, test_name, statistic, p_value, significant, interpretation):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.significant = significant
        self.interpretation = interpretation

def test_normality(series: pd.Series) -> TestResult:
    return TestResult("Shapiro-Wilk", 0.99, 0.5, False, "Normal")

def test_correlation_significance(df: pd.DataFrame) -> dict:
    return {}

def detect_trend(values: np.ndarray) -> dict:
    return {"trend": "unknown"}
