import pytest
from pathlib import Path
import os
from ai_analyst.utils.config import sanitize_path

def test_sanitize_path_valid():
    """Test that valid paths in CWD are allowed."""
    p = sanitize_path("README.md")
    assert p.name == "README.md"
    assert p.exists()

def test_sanitize_path_traversal():
    """Test that path traversal attempts are blocked."""
    # This assumes we are running from project root and parent is not writable/accessible or just testing the logic
    # We rely on the fact that CWD is the root of the repo
    with pytest.raises(ValueError, match="Security Error: Path traversal detected"):
        sanitize_path("../../../etc/passwd")

def test_sanitize_path_absolute_traversal():
    """Test that absolute paths outside CWD are blocked."""
    with pytest.raises(ValueError, match="Security Error: Path traversal detected"):
        sanitize_path("/etc/passwd")

def test_sanitize_path_subdirectory():
    """Test that paths in subdirectories are allowed."""
    # Create a dummy file in a subdir for testing
    Path("data").mkdir(exist_ok=True)
    Path("data/test.csv").touch()

    p = sanitize_path("data/test.csv")
    assert p.name == "test.csv"

    # Cleanup
    Path("data/test.csv").unlink()
    Path("data").rmdir()
