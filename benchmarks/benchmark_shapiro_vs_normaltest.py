import time
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def benchmark_normality():
    sample_sizes = [1000000, 5000000, 10000000]
    iterations = 5

    print(f"{'N':<10} {'Shapiro (ms)':<15} {'Normaltest (ms)':<15} {'Speedup (x)':<10}")
    print("-" * 55)

    for n in sample_sizes:
        data = pd.Series(np.random.randn(n))

        # Benchmark Shapiro
        start_shapiro = time.perf_counter()
        for _ in range(iterations):
            stats.shapiro(data)
        end_shapiro = time.perf_counter()
        avg_shapiro = (end_shapiro - start_shapiro) / iterations * 1000

        # Benchmark Normaltest
        start_normaltest = time.perf_counter()
        for _ in range(iterations):
            stats.normaltest(data)
        end_normaltest = time.perf_counter()
        avg_normaltest = (end_normaltest - start_normaltest) / iterations * 1000

        speedup = avg_shapiro / avg_normaltest if avg_normaltest > 0 else 0

        print(f"{n:<10} {avg_shapiro:<15.4f} {avg_normaltest:<15.4f} {speedup:<10.2f}")

if __name__ == "__main__":
    benchmark_normality()
