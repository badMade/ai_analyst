"""
Tests for the CLI commands.
"""

import pytest
from click.testing import CliRunner
from ai_analyst.cli import inspect

class TestInspectCommand:
    """Tests for the inspect command."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner instance."""
        return CliRunner()

    def test_inspect_valid_csv(self, runner, sample_csv_file):
        """Test inspecting a valid CSV file."""
        result = runner.invoke(inspect, [sample_csv_file])
        assert result.exit_code == 0
        assert "File:" in result.output
        assert "Shape:" in result.output
        assert "Column Information" in result.output
        assert "id" in result.output
        assert "category" in result.output

    def test_inspect_valid_json(self, runner, sample_json_file):
        """Test inspecting a valid JSON file."""
        result = runner.invoke(inspect, [sample_json_file])
        assert result.exit_code == 0
        assert "File:" in result.output
        assert "Shape:" in result.output
        assert "Column Information" in result.output

    def test_inspect_valid_excel(self, runner, sample_excel_file):
        """Test inspecting a valid Excel file."""
        result = runner.invoke(inspect, [sample_excel_file])
        assert result.exit_code == 0
        assert "File:" in result.output
        assert "Shape:" in result.output

    def test_inspect_valid_parquet(self, runner, sample_parquet_file):
        """Test inspecting a valid Parquet file."""
        result = runner.invoke(inspect, [sample_parquet_file])
        assert result.exit_code == 0
        assert "File:" in result.output
        assert "Shape:" in result.output

    def test_inspect_nonexistent_file(self, runner):
        """Test inspecting a non-existent file."""
        result = runner.invoke(inspect, ["nonexistent.csv"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid path" in result.output

    def test_inspect_unsupported_format(self, runner, tmp_path):
        """Test inspecting a file with an unsupported extension."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        result = runner.invoke(inspect, [str(f)])
        assert result.exit_code != 0
        assert "Unsupported format" in result.output

    def test_inspect_malformed_file(self, runner, tmp_path):
        """Test inspecting a malformed CSV file (empty)."""
        f = tmp_path / "empty.csv"
        f.touch()
        result = runner.invoke(inspect, [str(f)])
        assert result.exit_code != 0
        assert "Error reading file" in result.output
