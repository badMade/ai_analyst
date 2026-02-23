## 2025-05-15 - Information Leakage in Tool Execution
**Vulnerability:** The `_execute_tool` method was catching generic exceptions and returning `str(e)` to the caller (LLM). This leaked internal file paths (via `FileNotFoundError` or `ValueError` from `sanitize_path`) and potentially stack trace details.
**Learning:** Generic exception handling in LLM tools must sanitize error messages to prevent disclosing sensitive environment details to the model (and thus potentially the user).
**Prevention:** Catch specific exceptions (`FileNotFoundError`, `ValueError`) and return sanitized, user-friendly error messages. Use a generic "Internal Error" message for unhandled exceptions.
