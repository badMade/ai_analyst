"""
Tests for CLI commands.
"""

from click.testing import CliRunner

from ai_analyst.cli import main


class TestInspectCommand:
    """Tests for the inspect command."""

    def test_inspect_non_existent_file(self):
        """Should fail with exit code 2 and error message for non-existent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "non_existent_file.csv"])

        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_inspect_unsupported_format(self, tmp_path):
        """Should fail with exit code 1 and error message for unsupported format."""
        # Create a dummy file with unsupported extension
        test_file = tmp_path / "test.txt"
        test_file.write_text("dummy content")

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(test_file)])

        assert result.exit_code == 1
        assert "Unsupported format" in result.output
        assert ".txt" in result.output

    def test_inspect_valid_csv(self, tmp_path):
        """Should succeed for valid CSV file."""
        # Create a dummy CSV file
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2")

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(test_file)])

        assert result.exit_code == 0
        assert "File:" in result.output
        assert "Shape: 1 rows x 2 columns" in result.output
        assert "col1" in result.output
        assert "col2" in result.output



    def test_inspect_empty_csv(self, tmp_path):
        """Should fail with exit code 1 and error message for empty CSV file."""
        test_file = tmp_path / "empty.csv"
        test_file.touch()

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(test_file)])

        assert result.exit_code == 1
        assert "Error reading file" in result.output
