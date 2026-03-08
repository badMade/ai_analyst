"""
Fix Models for AI Analyst Self-Healing System.

Defines fix strategies, results, and code patches specific to
data analysis workflows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from self_healing.models.errors import DetectedError


class FixStrategy(Enum):
    """Strategies for fixing detected errors."""

    # Code modifications
    CODE_PATCH = "code_patch"
    ADD_IMPORT = "add_import"
    FIX_SYNTAX = "fix_syntax"

    # Data fixes
    DATA_RELOAD = "data_reload"
    ENCODING_FIX = "encoding_fix"
    TYPE_COERCE = "type_coerce"
    COLUMN_MAP = "column_map"

    # API fixes
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    REFRESH_AUTH = "refresh_auth"
    REDUCE_BATCH = "reduce_batch"

    # Resource fixes
    CACHE_CLEAR = "cache_clear"
    MEMORY_OPTIMIZE = "memory_optimize"
    CHUNK_PROCESSING = "chunk_processing"

    # Environment fixes
    INSTALL_DEPENDENCY = "install_dependency"
    CREATE_DIRECTORY = "create_directory"
    SET_PERMISSION = "set_permission"

    # CI fixes
    RETRY_CI = "retry_ci"
    FIX_WORKFLOW = "fix_workflow"
    UPDATE_CONFIG = "update_config"

    # Fallback
    MANUAL = "manual"
    ROLLBACK = "rollback"


class FixStatus(Enum):
    """Status of a fix attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class CodePatch:
    """Represents a code modification to apply."""

    file_path: Path
    original_content: str
    new_content: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "description": self.description,
            "original_length": len(self.original_content),
            "new_length": len(self.new_content),
        }


@dataclass
class ShellCommand:
    """Represents a shell command to execute."""

    command: str
    working_dir: Optional[Path] = None
    timeout: int = 60
    require_success: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "timeout": self.timeout,
            "description": self.description,
        }


@dataclass
class Fix:
    """
    Represents a fix to be applied for a detected error.

    Contains all information needed to apply and verify the fix.
    """

    error: Optional[DetectedError]
    strategy: FixStrategy
    description: str
    reasoning: str
    code_patches: list[CodePatch] = field(default_factory=list)
    shell_commands: list[ShellCommand] = field(default_factory=list)
    config_changes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    is_reversible: bool = True
    requires_restart: bool = False
    status: FixStatus = FixStatus.PENDING
    fix_id: str = field(default="")
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if not self.fix_id:
            self.fix_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique fix ID."""
        content = f"{self.strategy.value}:{self.description}:{self.created_at}"
        return f"FIX-{hashlib.sha256(content.encode()).hexdigest()[:8].upper()}"

    def mark_success(self) -> None:
        """Mark the fix as successful."""
        self.status = FixStatus.SUCCESS

    def mark_failed(self) -> None:
        """Mark the fix as failed."""
        self.status = FixStatus.FAILED

    def mark_partial(self) -> None:
        """Mark the fix as partially successful."""
        self.status = FixStatus.PARTIAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fix_id": self.fix_id,
            "error_id": self.error.error_id if self.error else None,
            "strategy": self.strategy.value,
            "description": self.description,
            "reasoning": self.reasoning,
            "code_patches": [p.to_dict() for p in self.code_patches],
            "shell_commands": [c.to_dict() for c in self.shell_commands],
            "config_changes": self.config_changes,
            "confidence": self.confidence,
            "is_reversible": self.is_reversible,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class FixResult:
    """Result of applying a fix."""

    fix: Fix
    success: bool
    output: str = ""
    error_message: str = ""
    verification_passed: bool = False
    new_errors: list[Any] = field(default_factory=list)
    attempt_number: int = 1
    duration_seconds: float = 0.0
    backup_path: Optional[Path] = None
    applied_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_id": self.fix.fix_id,
            "success": self.success,
            "output": self.output,
            "error_message": self.error_message,
            "verification_passed": self.verification_passed,
            "new_errors_count": len(self.new_errors),
            "attempt_number": self.attempt_number,
            "duration_seconds": self.duration_seconds,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "applied_at": self.applied_at,
        }


@dataclass
class FixPlan:
    """Plan for fixing multiple errors."""

    fixes: list[Fix] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    requires_approval: bool = False
    plan_id: str = field(default="")
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"PLAN-{hashlib.sha256(self.created_at.encode()).hexdigest()[:8].upper()}"
        if not self.execution_order:
            self.execution_order = [f.fix_id for f in self.fixes]

    def add_fix(self, fix: Fix) -> None:
        """Add a fix to the plan."""
        self.fixes.append(fix)
        self.execution_order.append(fix.fix_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "fixes": [f.to_dict() for f in self.fixes],
            "execution_order": self.execution_order,
            "estimated_duration": self.estimated_duration,
            "requires_approval": self.requires_approval,
            "created_at": self.created_at,
        }


# Common fix templates for analyst-specific errors
FIX_TEMPLATES = {
    "missing_import_pandas": {
        "strategy": FixStrategy.ADD_IMPORT,
        "description": "Add missing pandas import",
        "code": "import pandas as pd",
        "confidence": 0.95,
    },
    "missing_import_numpy": {
        "strategy": FixStrategy.ADD_IMPORT,
        "description": "Add missing numpy import",
        "code": "import numpy as np",
        "confidence": 0.95,
    },
    "encoding_utf8": {
        "strategy": FixStrategy.ENCODING_FIX,
        "description": "Use UTF-8 encoding with error handling",
        "code": "encoding='utf-8', errors='replace'",
        "confidence": 0.85,
    },
    "memory_chunked_read": {
        "strategy": FixStrategy.CHUNK_PROCESSING,
        "description": "Read data in chunks to reduce memory usage",
        "confidence": 0.80,
    },
    "api_retry_backoff": {
        "strategy": FixStrategy.RETRY_WITH_BACKOFF,
        "description": "Retry with exponential backoff",
        "confidence": 0.90,
    },
}
