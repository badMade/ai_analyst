
## 2025-05-14 - [Vectorizing pandas describe_statistics]
**Learning:** Manual vectorized aggregation across a full `DataFrame` (using `df.count()`, `df.mean()`, `df.std()`, `df.quantile()`) is significantly faster than using `.describe()` in a per-column iteration. Iterating and calling `.describe()` row-by-row on a 1000-column dataframe takes ~1.7s, while using vectorized aggregates takes ~0.09s, an 18.8x speedup. Standard deviation edge cases where $N<2$ can be manually set to NaN.
**Action:** Always prefer vectorized operations (`df.mean()`, `df.std()`, `df.quantile()`) for summary statistics over applying `.describe()` repetitively on columns, especially for wide DataFrames.
