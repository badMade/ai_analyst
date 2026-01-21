import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from analyst import AnalysisContext

@pytest.fixture
def analysis_context():
    return AnalysisContext()

def test_load_csv_with_pyarrow(analysis_context, tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3")

    with patch('pandas.read_csv') as mock_read_csv:
        analysis_context.load_dataset(str(csv_file))
        mock_read_csv.assert_called_once_with(csv_file, engine='pyarrow')
