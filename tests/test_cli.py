import numpy as np
import pandas as pd
from click.testing import CliRunner

from ai_analyst.cli import main


def test_inspect_command(tmp_path):
    # Create a dummy parquet file
    df = pd.DataFrame({"A": [1, 2, np.nan], "B": ["x", "y", "z"]})
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)

    runner = CliRunner()
    result = runner.invoke(main, ["inspect", str(file_path)])

    assert result.exit_code == 0
    assert "Column Information" in result.output
    assert "A" in result.output
    assert "B" in result.output
    # Check for correct stats
    # A: 2 non-null, 33.3% null
    assert "2" in result.output  # Non-null count
    assert "33.3%" in result.output  # Null pct
    # B: 3 non-null, 0.0% null
    assert "3" in result.output
    assert "0.0%" in result.output


def test_inspect_command_large_file(tmp_path):
    # Test with a larger file to ensure no crashes
    rows = 100
    df = pd.DataFrame(np.random.randn(rows, 5), columns=[f"col_{i}" for i in range(5)])
    file_path = tmp_path / "large.parquet"
    df.to_parquet(file_path)

    runner = CliRunner()
    result = runner.invoke(main, ["inspect", str(file_path)])

    assert result.exit_code == 0
    assert "100 rows" in result.output
    assert "5 columns" in result.output
