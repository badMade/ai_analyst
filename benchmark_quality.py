import os
import sys
import time
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock

# Mock Anthropic to avoid API key issues
mock_anthropic = MagicMock()
sys.modules["anthropic"] = mock_anthropic

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from analyst import StandaloneAnalyst
except ImportError:
    sys.path.append(os.getcwd())
    from analyst import StandaloneAnalyst

def benchmark():
    print("Generating synthetic data (1000 cols, 10000 rows)...")
    n_rows = 10000
    n_cols = 1000

    # Create data efficiently
    data = np.random.randn(n_rows, n_cols)
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])

    # Inject nulls in 10% of columns
    print("Injecting nulls...")
    cols_with_nulls = df.columns[:100]
    for col in cols_with_nulls:
        idx = np.random.choice(df.index, size=int(n_rows * 0.05), replace=False)
        df.loc[idx, col] = np.nan

    print("Data prepared.")

    analyst = StandaloneAnalyst()
    analyst.context.datasets["bench_data"] = df

    print("Running check_data_quality...")
    start_time = time.time()

    result_str = analyst._execute_tool("check_data_quality", {"dataset_name": "bench_data"})

    end_time = time.time()
    duration = end_time - start_time

    print(f"Benchmark Duration: {duration:.4f} seconds")

    result = json.loads(result_str)
    print(f"Quality Score: {result.get('quality_score')}")
    print(f"Columns with issues: {len(result.get('column_issues', {}))}")

if __name__ == "__main__":
    benchmark()
