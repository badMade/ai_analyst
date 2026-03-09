import pytest
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError

from ai_analyst.cli import get_cli_version

def test_get_cli_version_success():
    with patch("ai_analyst.cli.version") as mock_version:
        mock_version.return_value = "1.2.3"
        assert get_cli_version() == "1.2.3"
        mock_version.assert_called_once_with("ai-analyst")

def test_get_cli_version_not_found():
    with patch("ai_analyst.cli.version") as mock_version:
        mock_version.side_effect = PackageNotFoundError("ai-analyst")
        assert get_cli_version() == "0.0.0"
        mock_version.assert_called_once_with("ai-analyst")
