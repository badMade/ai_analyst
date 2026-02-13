## 2025-02-19 - Hardcoded Dummy API Key
**Vulnerability:** Default `anthropic_api_key` was set to `"sk-dummy-key"`.
**Learning:** This allowed tests to pass with improper mocking, as the real client accepted the dummy key format (though it would fail at runtime).
**Prevention:** Use empty strings for default secrets to force fast failure and expose testing gaps.
