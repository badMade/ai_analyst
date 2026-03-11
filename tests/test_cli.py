"""
Tests for CLI commands in ai_analyst.cli.
"""

from click.testing import CliRunner

from ai_analyst.cli import main


class TestCliInspect:
    """Tests for the inspect command."""

    def test_inspect_non_existent_file(self):
        """Should fail with exit code 2 and error message when file does not exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "non_existent.csv"])

        assert result.exit_code == 2
        assert "Path 'non_existent.csv' does not exist" in result.output

    def test_inspect_invalid_extension(self):
        """Should fail with exit code 1 and error message when file extension is not supported."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test.txt", "w") as f:
                f.write("hello")

            result = runner.invoke(main, ["inspect", "test.txt"])

            assert result.exit_code == 1
            assert "Unsupported format: .txt" in result.output
