## 2025-05-15 - Hardcoded Dummy Secrets in Pydantic Settings
**Vulnerability:** Found a hardcoded "dummy" API key (`sk-dummy-key`) used as a default value in `pydantic-settings` configuration.
**Learning:** Developers sometimes add dummy keys to satisfy type checks or local dev requirements, but this prevents "fail fast" behavior and can be confusing or insecure if the dummy key is inadvertently used.
**Prevention:** Always default sensitive configuration values to `""` (empty string) or `None` in `BaseSettings` classes, and enforce presence checks at runtime (e.g., via `get_auth_method` or Pydantic validation).
