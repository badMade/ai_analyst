import json
from unittest.mock import patch

import pytest

from analyst import StandaloneAnalyst


@pytest.fixture
def analyst():
    # Mock anthropic client creation
    with patch("analyst.Anthropic"):
        return StandaloneAnalyst()

def test_file_not_found_error_sanitization(analyst):
    # Simulate a FileNotFoundError from context.load_dataset
    with patch.object(analyst.context, 'load_dataset') as mock_load:
        # Simulate exception with an absolute path
        mock_load.side_effect = FileNotFoundError("File not found: /root/secret/data.csv")

        result_json = analyst._execute_tool("load_dataset", {"file_path": "/root/secret/data.csv"})
        result = json.loads(result_json)

        assert "error" in result
        # Check that only the filename is returned, not the absolute path
        assert result["error"] == "File not found: data.csv"
        assert "/root/secret" not in result["error"]

def test_value_error_security_sanitization(analyst):
    # Simulate a ValueError from path traversal attempt
    with patch.object(analyst.context, 'load_dataset') as mock_load:
        mock_load.side_effect = ValueError("Invalid path outside of allowed base directory: /etc/passwd")

        result_json = analyst._execute_tool("load_dataset", {"file_path": "../../../etc/passwd"})
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "Security Error"
        assert "/etc/passwd" not in result["error"]

def test_value_error_windows_path_sanitization(analyst):
    # Simulate a ValueError from Windows path attempt
    with patch.object(analyst.context, 'load_dataset') as mock_load:
        mock_load.side_effect = ValueError("Invalid Windows-style path on non-Windows system: C:\\Windows\\System32")

        result_json = analyst._execute_tool("load_dataset", {"file_path": "C:\\Windows\\System32"})
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "Security Error"
        assert "C:\\Windows" not in result["error"]

def test_generic_exception_sanitization(analyst):
    # Simulate a generic exception to ensure no stack traces or internal details leak
    with patch.object(analyst.context, 'load_dataset') as mock_load:
        mock_load.side_effect = Exception("Database connection failed: user=admin password=supersecret")

        result_json = analyst._execute_tool("load_dataset", {"file_path": "data.csv"})
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "An internal error occurred."
        assert "supersecret" not in result["error"]

def test_value_error_non_security(analyst):
    # Simulate a ValueError that is NOT a security issue (e.g., unsupported format)
    with patch.object(analyst.context, 'load_dataset') as mock_load:
        mock_load.side_effect = ValueError("Unsupported format: .txt")

        result_json = analyst._execute_tool("load_dataset", {"file_path": "data.txt"})
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "Unsupported format: .txt"
