"""
Tests for CLI commands.
"""

from click.testing import CliRunner
from ai_analyst.cli import main
import pytest

class TestInspectCommand:
    """Tests for the inspect command."""

    def test_inspect_invalid_path(self, tmp_path):
        """Should exit with code 2 if file does not exist."""
        runner = CliRunner()
        # Use tmp_path to ensure we are in a valid base directory context if needed
        # but the file itself doesn't exist
        non_existent_file = tmp_path / "non_existent.csv"

        result = runner.invoke(main, ["inspect", str(non_existent_file)])

        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_inspect_invalid_extension(self, tmp_path):
        """Should exit with code 1 if file has unsupported extension."""
        runner = CliRunner()

        # Create a file with unsupported extension
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("some content")

        result = runner.invoke(main, ["inspect", str(invalid_file)])

        assert result.exit_code == 1
        assert "Unsupported format" in result.output
        assert ".txt" in result.output

    def test_inspect_valid_csv(self, tmp_path):
        """Should successfully inspect a valid CSV file."""
        runner = CliRunner()

        # Create a valid CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2\nval3,val4")

        result = runner.invoke(main, ["inspect", str(csv_file)])

        assert result.exit_code == 0
        assert "File:" in result.output
        assert "2 rows x 2 columns" in result.output
        assert "col1" in result.output
        assert "col2" in result.output
