## 2026-01-25 - [Hardcoded Dummy API Key]
**Vulnerability:** Hardcoded `sk-dummy-key` in `config.py` as a default value for `anthropic_api_key`.
**Learning:** Developers likely added this to avoid validation errors during local development or testing without an API key, but it creates a false sense of configuration and risks using invalid credentials silently.
**Prevention:** Always default sensitive configuration values to `""` (empty string) or `None`, and rely on explicit checks or environment variables.
