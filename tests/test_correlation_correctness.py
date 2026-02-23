import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock, patch

def test_correlation_optimization_correctness():
    # Setup data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5], # Perfect correlation with A
        'C': [5, 4, 3, 2, 1], # Perfect negative correlation with A
        'D': [1, 1, 1, 1, 1], # Constant, should produce NaN correlation
        'E': [1, 3, 2, 5, 4]  # Random
    }
    df = pd.DataFrame(data)

    # Original logic (simulation)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method='pearson')

    expected_correlations = []
    cols = corr_matrix.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i+1:]:
            corr = corr_matrix.loc[col_a, col_b]
            if pd.notna(corr):
                expected_correlations.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(corr, 4)
                })
    expected_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # New logic (to be implemented)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask)

    # Explicit dropna() required for robust behavior
    stacked = upper.stack().dropna().reset_index()
    stacked.columns = ['column_a', 'column_b', 'correlation']
    stacked['correlation'] = stacked['correlation'].round(4)
    actual_correlations = stacked.to_dict(orient='records')
    actual_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    print(f"Expected count: {len(expected_correlations)}")
    print(f"Actual count: {len(actual_correlations)}")

    assert len(actual_correlations) == len(expected_correlations)

    # Verify content match
    def to_set(corrs):
        return set((c['column_a'], c['column_b'], c['correlation']) for c in corrs)

    expected_set = to_set(expected_correlations)
    actual_set = to_set(actual_correlations)

    assert expected_set == actual_set

    # Verify specific values
    # A-B should be 1.0
    ab_list = [c for c in actual_correlations if (c['column_a'] == 'A' and c['column_b'] == 'B')]
    assert len(ab_list) == 1
    assert ab_list[0]['correlation'] == 1.0

    # A-C should be -1.0
    ac_list = [c for c in actual_correlations if (c['column_a'] == 'A' and c['column_b'] == 'C')]
    assert len(ac_list) == 1
    assert ac_list[0]['correlation'] == -1.0

    # D should not appear
    d_present = any(c['column_a'] == 'D' or c['column_b'] == 'D' for c in actual_correlations)
    assert not d_present

if __name__ == "__main__":
    test_correlation_optimization_correctness()
    print("Test passed!")
