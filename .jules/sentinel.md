## 2024-03-02 - Information Disclosure in AI Analyst Tool Loop

**Vulnerability:**
The `_execute_tool` method in `analyst.py` caught generic `Exception` objects and returned their raw string representation (`str(e)`) as JSON to the AI tool loop. For file operations (`FileNotFoundError`), this leaked absolute server paths. For path traversal blocking (`ValueError`), this confirmed the configured base directory path. For unexpected errors, this could leak database connections, passwords, or internal module structures.

**Learning:**
Even if an error is technically caught and gracefully handled, returning the raw error message to an LLM or user can constitute an Information Disclosure vulnerability (CWE-209).

**Prevention:**
1. Catch specific exceptions (e.g., `FileNotFoundError`, `ValueError`).
2. Sanitize error messages before returning them, stripping out absolute paths, internal configurations, or server details.
3. Use generic, safe messages (e.g., "Security Error", "An internal error occurred") for catch-all `Exception` blocks.
