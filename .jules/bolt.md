## 2024-05-23 - CSV Loading Performance
**Learning:** `pd.read_csv` default engine is significantly slower (approx 6.8x) than `engine="pyarrow"` for large datasets, and since `pyarrow` is already a dependency, not using it is a missed opportunity.
**Action:** Always check `pd.read_csv` calls and consider `engine="pyarrow"` if `pyarrow` is available in the environment.
