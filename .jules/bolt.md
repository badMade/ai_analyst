
## 2025-05-18 - Vectorizing Correlation Matrix Flattening
**Learning:** Extracting pairwise correlations from a large matrix using nested `for` loops and `loc` lookups is O(n²) and very slow for dataframes with hundreds of columns.
**Action:** Use vectorized pandas operations: `df.where(np.triu(np.ones(df.shape), k=1).astype(bool)).stack().dropna().reset_index()`. This extracts the upper triangle, flattens it, and drops NaNs, resulting in ~10x performance gains.
