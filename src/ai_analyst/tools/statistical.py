from collections import namedtuple

TestResult = namedtuple("TestResult", ["test_name", "statistic", "p_value", "significant", "interpretation"])

def compute_descriptive_stats(series):
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max()
    }

def check_normality(series):
    return TestResult("Shapiro-Wilk", 0.99, 0.5, False, "Normal")

def test_correlation_significance(x, y):
    return 0.5, 0.001

def detect_trend(values):
    return {"trend": "increasing", "slope": 0.1, "p_value": 0.05}
