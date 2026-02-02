## 2025-05-14 - Pandas Stack Behavior Change
**Learning:** In Pandas 3.0.0+, `DataFrame.stack()` now preserves `NaN` values by default (unlike previous versions which dropped them). When using `stack()` to reshape data (e.g., correlation matrices), you must explicitly filter out `NaN`s if you want the old behavior.
**Action:** Always verify `stack()` output for unexpected `NaN`s and use `.dropna()` explicitly when reshaping sparse or masked data.
