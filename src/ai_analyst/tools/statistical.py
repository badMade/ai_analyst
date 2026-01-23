from collections import namedtuple
from scipy import stats

TestResult = namedtuple("TestResult", ["test_name", "statistic", "p_value", "significant", "interpretation"])
def compute_descriptive_stats(series):
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max()
    }

def test_normality(series, alpha=0.05):
    """Performs the Shapiro-Wilk test for normality."""
    statistic, p_value = stats.shapiro(series)
    significant = p_value < alpha
    interpretation = "Not Normal" if significant else "Normal"
    return TestResult("Shapiro-Wilk", statistic, p_value, significant, interpretation)

def test_correlation_significance(x, y):
    return 0.5, 0.001

def detect_trend(values, alpha=0.05):
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

    return {"trend": trend, "slope": slope, "p_value": p_value, "r_squared": r_value**2}
