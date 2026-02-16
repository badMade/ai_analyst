## 2025-05-14 - Hardcoded Dummy API Key
**Vulnerability:** A hardcoded dummy API key (`sk-dummy-key`) was present in the `Settings` class default value.
**Learning:** Hardcoded secrets, even dummy ones, can mislead developers, trigger false positives in security scanners, and prevent the application from failing fast when configuration is missing. It can also lead to accidental usage of invalid credentials.
**Prevention:** Always use empty strings or `None` as default values for sensitive configuration options to ensure they must be explicitly provided via environment variables or other secure means.
