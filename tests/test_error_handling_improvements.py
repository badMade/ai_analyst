import json
from unittest.mock import MagicMock, patch

import pytest

from analyst import AnalysisContext, StandaloneAnalyst


class TestErrorHandlingImprovements:
    @pytest.fixture
    def analyst(self):
        analyst = StandaloneAnalyst()
        analyst.context = MagicMock(spec=AnalysisContext)
        # Mock sanitize_path for load_dataset
        self.mock_sanitize_patcher = patch("analyst.sanitize_path")
        self.mock_sanitize = self.mock_sanitize_patcher.start()
        return analyst

    @pytest.fixture(autouse=True)
    def stop_patching(self):
        yield
        self.mock_sanitize_patcher.stop()

    def test_key_error_handling(self, analyst):
        """Should return descriptive error for KeyError."""
        # Setup mock to raise KeyError
        analyst.context.get_dataset.side_effect = KeyError("missing_column")

        result = analyst._execute_tool("describe_statistics", {
            "dataset_name": "test_data",
            "column": "missing_column"
        })

        result_dict = json.loads(result)
        assert "error" in result_dict
        # We expect a specific message format for KeyError
        assert "Missing column or key" in result_dict["error"]
        assert "missing_column" in result_dict["error"]

    def test_value_error_handling(self, analyst):
        """Should return descriptive error for ValueError."""
        # Setup mock to raise ValueError
        analyst.context.get_dataset.side_effect = ValueError("Invalid dataset name")

        result = analyst._execute_tool("describe_statistics", {
            "dataset_name": "invalid_name"
        })

        result_dict = json.loads(result)
        assert "error" in result_dict
        # We expect a specific message format for ValueError
        assert "Invalid value" in result_dict["error"]
        assert "Invalid dataset name" in result_dict["error"]

    def test_file_not_found_error_handling(self, analyst):
        """Should return descriptive error for FileNotFoundError."""
        # Mock load_dataset on context to raise FileNotFoundError
        # Note: In _execute_tool, load_dataset calls self.context.load_dataset
        analyst.context.load_dataset.side_effect = FileNotFoundError("File not found: test.csv")

        result = analyst._execute_tool("load_dataset", {
            "file_path": "test.csv"
        })

        result_dict = json.loads(result)
        assert "error" in result_dict
        # We expect a specific message format for FileNotFoundError
        assert "File not found" in result_dict["error"]
        assert "test.csv" in result_dict["error"]

    def test_generic_exception_handling(self, analyst):
        """Should handle unexpected exceptions."""
        analyst.context.get_dataset.side_effect = RuntimeError("Something went wrong")

        result = analyst._execute_tool("describe_statistics", {
            "dataset_name": "test_data"
        })

        result_dict = json.loads(result)
        assert "error" in result_dict
        # Generic error should just contain the string representation
        assert "Something went wrong" in result_dict["error"]
