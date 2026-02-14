import time
import pandas as pd
import numpy as np
import warnings
from ai_analyst.tools.statistical import test_normality

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Large N
    N = 10000
    data = pd.Series(np.random.randn(N))

    start = time.perf_counter()
    result = test_normality(data)
    end = time.perf_counter()

    print(f"N={N}")
    print(f"Test Name: {result.test_name}")
    print(f"Time: {(end - start) * 1000:.4f} ms")

    if len(w) > 0:
        print("Warnings caught:")
        for warning in w:
            print(f"- {warning.category.__name__}: {warning.message}")
    else:
        print("No warnings caught.")

    print("-" * 30)

    # Small N
    N = 100
    data = pd.Series(np.random.randn(N))

    start = time.perf_counter()
    result = test_normality(data)
    end = time.perf_counter()

    print(f"N={N}")
    print(f"Test Name: {result.test_name}")
    print(f"Time: {(end - start) * 1000:.4f} ms")
