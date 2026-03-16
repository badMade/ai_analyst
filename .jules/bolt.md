## 2024-05-14 - Vectorized pandas Aggregation

**Learning:** When calculating descriptive statistics for multiple columns in a pandas DataFrame, looping over the columns and calling `Series.describe()` or performing manual aggregations is O(N) in the number of columns and incurs significant Python overhead.

**Action:** Replace `for col in df.columns: stats.append(compute_descriptive_stats(df[col]))` with vectorized aggregations on the entire DataFrame (e.g., `df.quantile([0.25, 0.50, 0.75])`, `df.mean()`, `df.std()`). Construct a new results DataFrame from these aggregate Series and convert it via `reset_index().to_dict(orient="records")`. For a 1000x1000 float DataFrame, this yields ~26x speedup. However, remember to properly handle edge cases where the DataFrame is empty or the quantiles calculation results in an empty DataFrame.
