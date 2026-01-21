import pytest
from unittest.mock import patch
from analyst import AnalysisContext
from ai_analyst.utils.config import sanitize_path

@pytest.fixture
def analysis_context():
    return AnalysisContext()

def test_load_csv_with_pyarrow(analysis_context, tmp_path) -> None:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3")

    with patch('analyst.pd.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        analysis_context.load_dataset(str(csv_file))
        expected_path = sanitize_path(str(csv_file))
        mock_read_csv.assert_called_once_with(expected_path, engine='pyarrow')
