## 2025-05-19 - Pandas Descriptive Statistics Optimization
**Learning:** Manual vectorized aggregation (e.g., `df.mean()`, `df.quantile()`) is significantly faster than `df.describe()` or column-wise iteration. Benchmarks showed ~3x speedup (78ms vs 230ms for 10000x100 DataFrame).
**Action:** Use manual aggregation when computing statistics for large DataFrames, especially when specific stats are needed.
