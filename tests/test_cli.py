import pytest
from click.testing import CliRunner
from ai_analyst.cli import inspect

def test_inspect_valid_csv(sample_csv_file):
    """Test inspect command with a valid CSV file."""
    runner = CliRunner()
    result = runner.invoke(inspect, [sample_csv_file])
    assert result.exit_code == 0
    assert "File:" in result.output
    assert "Shape:" in result.output
    assert "Column Information" in result.output

def test_inspect_nonexistent_file():
    """Test inspect command with a non-existent file."""
    runner = CliRunner()
    result = runner.invoke(inspect, ["nonexistent.csv"])
    assert result.exit_code != 0
    # The exact message depends on click version but usually contains "does not exist"
    assert "does not exist" in result.output

def test_inspect_unsupported_format(tmp_path):
    """Test inspect command with an unsupported file format."""
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("content")
    runner = CliRunner()
    result = runner.invoke(inspect, [str(dummy_file)])
    assert result.exit_code == 1
    assert "Unsupported format" in result.output
