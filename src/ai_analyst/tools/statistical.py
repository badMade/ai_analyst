from collections import namedtuple
import pandas as pd
import numpy as np
from scipy import stats

NormalityResult = namedtuple("NormalityResult", ["test_name", "statistic", "p_value", "significant", "interpretation"])
TrendResult = namedtuple("TrendResult", ["slope", "intercept", "p_value", "trend"])

def compute_descriptive_stats(series):
    return series.describe().to_dict()

def test_normality(series):
    if len(series) < 3:
         return NormalityResult("Shapiro-Wilk", 0.0, 1.0, False, "Insufficient data")
    stat, p = stats.shapiro(series)
    return NormalityResult("Shapiro-Wilk", stat, p, p < 0.05, "Normal" if p >= 0.05 else "Not Normal")

def detect_trend(values):
    # Mock trend detection
    return {"trend": "flat", "p_value": 1.0}

def test_correlation_significance(df):
    pass
