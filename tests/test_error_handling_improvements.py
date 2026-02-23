
import json
import pytest
from unittest.mock import MagicMock, patch
from analyst import StandaloneAnalyst

class TestErrorHandlingImprovements:
    """Tests for improved error handling in StandaloneAnalyst._execute_tool."""

    @pytest.fixture
    def analyst(self):
        """Create analyst with mocked client."""
        with patch("anthropic.Anthropic"):
            return StandaloneAnalyst()

    def test_file_not_found_error_sanitized(self, analyst):
        """Test that FileNotFoundError does not leak absolute paths."""
        # Using load_dataset with a non-existent file
        # We need to ensure sanitize_path allows the path but exists returns False
        with patch("analyst.sanitize_path") as mock_sanitize:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            # Mock resolved path to look like an absolute server path
            mock_path.__str__.return_value = "/app/secrets/data.csv"
            mock_sanitize.return_value = mock_path

            result_json = analyst._execute_tool("load_dataset", {"file_path": "data.csv"})
            result = json.loads(result_json)

            assert "error" in result
            assert "File not found" in result["error"]
            # It should use the input filename 'data.csv' or 'specified file', NOT '/app/secrets/data.csv'
            assert "/app/secrets" not in result["error"]
            assert "data.csv" in result["error"]

    def test_path_traversal_error_sanitized(self, analyst):
        """Test that path traversal attempts result in a generic security error."""
        # Simulate sanitize_path raising ValueError with sensitive info
        with patch("analyst.sanitize_path") as mock_sanitize:
            mock_sanitize.side_effect = ValueError("Invalid path outside of allowed base directory: /etc/passwd")

            result_json = analyst._execute_tool("load_dataset", {"file_path": "../../../etc/passwd"})
            result = json.loads(result_json)

            assert "error" in result
            assert result["error"] == "Security Error: Invalid path or access denied."
            # Ensure leaked path is NOT in the error
            assert "/etc/passwd" not in result["error"]

    def test_key_error_handling(self, analyst):
        """Test that missing keys return a helpful error message."""
        # load_dataset requires file_path. If missing, it raises KeyError (or Type error depending on access)
        # But wait, load_dataset uses tool_input["file_path"]. So if key is missing, KeyError.

        result_json = analyst._execute_tool("load_dataset", {"wrong_key": "data.csv"})
        result = json.loads(result_json)

        assert "error" in result
        assert "Missing column or key" in result["error"]
        assert "file_path" in result["error"]  # The missing key name

    def test_generic_exception_handling(self, analyst):
        """Test that unexpected exceptions return a generic error message."""
        with patch("analyst.sanitize_path") as mock_sanitize:
            # Simulate a database connection error or other unexpected failure
            mock_sanitize.side_effect = RuntimeError("Database connection failed at 192.168.1.5")

            result_json = analyst._execute_tool("load_dataset", {"file_path": "data.csv"})
            result = json.loads(result_json)

            assert "error" in result
            assert result["error"] == "An internal error occurred during analysis."
            # Ensure internal details are not leaked
            assert "Database connection failed" not in result["error"]
            assert "192.168.1.5" not in result["error"]
