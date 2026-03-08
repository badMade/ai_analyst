"""Utils package for AI Analyst Self-Healing System."""

from self_healing.utils.patch_utils import (
    create_patch,
    apply_patch,
    parse_unified_diff,
    replace_lines,
    insert_after_line,
    find_and_replace,
    create_code_patch,
    get_context_lines,
    DiffHunk,
)

__all__ = [
    "create_patch",
    "apply_patch",
    "parse_unified_diff",
    "replace_lines",
    "insert_after_line",
    "find_and_replace",
    "create_code_patch",
    "get_context_lines",
    "DiffHunk",
]
