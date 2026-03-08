"""Models package for AI Analyst Self-Healing System."""

from self_healing.models.errors import (
    DetectedError,
    ErrorLocation,
    ErrorSeverity,
    ErrorType,
    EnvironmentType,
    ERROR_PATTERNS,
)
from self_healing.models.fixes import (
    CodePatch,
    Fix,
    FixPlan,
    FixResult,
    FixStatus,
    FixStrategy,
    ShellCommand,
    FIX_TEMPLATES,
)

__all__ = [
    # Errors
    "DetectedError",
    "ErrorLocation",
    "ErrorSeverity",
    "ErrorType",
    "EnvironmentType",
    "ERROR_PATTERNS",
    # Fixes
    "CodePatch",
    "Fix",
    "FixPlan",
    "FixResult",
    "FixStatus",
    "FixStrategy",
    "ShellCommand",
    "FIX_TEMPLATES",
]
