"""
Patch Utilities for AI Analyst Self-Healing System.

Utilities for creating, applying, and managing code patches.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from self_healing.models.fixes import CodePatch


@dataclass
class DiffHunk:
    """A single hunk in a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


def create_patch(
    original: str,
    modified: str,
    file_path: str = "file.py",
) -> str:
    """Create a unified diff patch."""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )

    return "".join(diff)


def apply_patch(
    content: str,
    patch: str,
) -> Optional[str]:
    """Apply a unified diff patch to content."""
    hunks = parse_unified_diff(patch)
    if not hunks:
        return None

    lines = content.split("\n")

    # Apply hunks in reverse order to preserve line numbers
    for hunk in reversed(hunks):
        start = hunk.old_start - 1
        end = start + hunk.old_count

        new_lines = []
        for line in hunk.lines:
            if line.startswith("+") and not line.startswith("+++"):
                new_lines.append(line[1:])
            elif not line.startswith("-"):
                new_lines.append(line[1:] if line.startswith(" ") else line)

        lines[start:end] = new_lines

    return "\n".join(lines)


def parse_unified_diff(patch: str) -> list[DiffHunk]:
    """Parse unified diff into hunks."""
    hunks = []
    current_hunk = None

    for line in patch.split("\n"):
        # Hunk header: @@ -start,count +start,count @@
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            if current_hunk:
                hunks.append(current_hunk)

            current_hunk = DiffHunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2) or 1),
                new_start=int(match.group(3)),
                new_count=int(match.group(4) or 1),
                lines=[],
            )
        elif current_hunk is not None:
            current_hunk.lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def replace_lines(
    content: str,
    start_line: int,
    end_line: int,
    new_content: str,
) -> str:
    """Replace lines in content."""
    lines = content.split("\n")
    new_lines = new_content.split("\n")

    lines[start_line - 1 : end_line] = new_lines
    return "\n".join(lines)


def insert_after_line(
    content: str,
    line_number: int,
    new_content: str,
) -> str:
    """Insert content after a specific line."""
    lines = content.split("\n")
    new_lines = new_content.split("\n")

    for i, new_line in enumerate(new_lines):
        lines.insert(line_number + i, new_line)

    return "\n".join(lines)


def find_and_replace(
    content: str,
    search: str,
    replace: str,
    count: int = -1,
) -> tuple[str, int]:
    """Find and replace text, returning new content and replacement count."""
    if count == -1:
        new_content = content.replace(search, replace)
        replacements = content.count(search)
    else:
        new_content = content.replace(search, replace, count)
        replacements = min(count, content.count(search))

    return new_content, replacements


def create_code_patch(
    file_path: Path,
    search: str,
    replace: str,
    description: str = "",
) -> Optional[CodePatch]:
    """Create a CodePatch for find/replace operation."""
    if not file_path.exists():
        return None

    content = file_path.read_text()

    if search not in content:
        return None

    new_content = content.replace(search, replace)

    # Find line numbers
    before_match = content.split(search)[0]
    line_start = before_match.count("\n") + 1
    line_end = line_start + search.count("\n")

    return CodePatch(
        file_path=file_path,
        original_content=content,
        new_content=new_content,
        line_start=line_start,
        line_end=line_end,
        description=description,
    )


def get_context_lines(
    content: str,
    line_number: int,
    context: int = 3,
) -> str:
    """Get lines around a specific line for context."""
    lines = content.split("\n")
    start = max(0, line_number - context - 1)
    end = min(len(lines), line_number + context)

    result = []
    for i in range(start, end):
        prefix = ">>> " if i == line_number - 1 else "    "
        result.append(f"{i + 1:4d} {prefix}{lines[i]}")

    return "\n".join(result)
