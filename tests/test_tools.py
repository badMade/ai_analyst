import pytest
import pandas as pd
from pathlib import Path
from analyst import AnalysisContext, TOOLS


def test_analysis_context_creation():
    """Test that AnalysisContext can be created."""
    context = AnalysisContext()
    assert context.datasets == {}


def test_tools_structure():
    """Test that TOOLS list has expected structure."""
    assert isinstance(TOOLS, list)
    assert len(TOOLS) > 0

    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool


def test_tool_names():
    """Test that expected tools are defined."""
    tool_names = [tool["name"] for tool in TOOLS]

    expected_tools = [
        "load_dataset",
        "list_datasets",
        "preview_data",
        "describe_statistics",
        "compute_correlation",
        "detect_outliers",
        "group_analysis",
        "check_data_quality",
        "test_normality",
        "analyze_trend",
    ]

    for expected in expected_tools:
        assert expected in tool_names, f"Tool '{expected}' not found in TOOLS"
