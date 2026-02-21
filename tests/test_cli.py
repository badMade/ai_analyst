"""
Tests for CLI commands in ai_analyst/cli.py.
"""
from click.testing import CliRunner
import pandas as pd
from ai_analyst.cli import inspect

def test_inspect_non_existent_file(tmp_path):
    """Test inspect command with a non-existent file."""
    runner = CliRunner()
    # Create a path that definitely does not exist within tmp_path
    non_existent_file = tmp_path / "non_existent_file.csv"

    result = runner.invoke(inspect, [str(non_existent_file)])

    assert result.exit_code == 2
    # Check for click's standard error message format for Path(exists=True)
    assert f"Path '{non_existent_file}' does not exist" in result.output

def test_inspect_invalid_extension(tmp_path):
    """Test inspect command with an unsupported file extension."""
    runner = CliRunner()

    # Create a dummy file with .txt extension
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("dummy content")

    result = runner.invoke(inspect, [str(test_file)])

    assert result.exit_code == 1
    assert "Unsupported format: .txt" in result.output

def test_inspect_valid_csv(tmp_path):
    """Test inspect command with a valid CSV file."""
    runner = CliRunner()

    # Create a dummy CSV file
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    test_file = tmp_path / "test_data.csv"
    df.to_csv(test_file, index=False)

    result = runner.invoke(inspect, [str(test_file)])

    assert result.exit_code == 0
    assert f"File: {test_file}" in result.output
    assert "2 rows x 2 columns" in result.output
    assert "Column Information" in result.output
    assert "col1" in result.output
    assert "col2" in result.output
