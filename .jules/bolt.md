## 2025-05-15 - Pandas 3.0 Stack Behavior
**Learning:** `DataFrame.stack()` in pandas 3.0 has stricter `dropna` behavior. The argument `dropna=True` (which was default) now raises a ValueError if specified, claiming "the new implementation does not introduce rows of NA values". However, `stack()` DOES preserve existing NA values from the input DataFrame (e.g., from `where(mask)`). To reliably drop NAs after stacking (e.g., to remove lower triangle masking), one must call `.dropna()` explicitly on the resulting Series: `df.stack().dropna()`.
**Action:** When using `stack()` to reshape and filter, always chain `.dropna()` if you intend to remove NAs, rather than relying on `stack` arguments.

## 2025-05-15 - Correlation Matrix Optimization
**Learning:** Replacing nested Python loops with vectorized pandas operations (`stack()`) for flattening correlation matrices yields massive performance gains (~9.6x speedup for 200 columns).
**Action:** Always look for vectorized alternatives to iterating over DataFrame columns/rows, especially for N^2 operations like pairwise comparisons.
