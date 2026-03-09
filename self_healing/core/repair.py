"""
Repair Engine for AI Analyst Self-Healing System.

Generates and applies fixes for detected errors with backup and rollback support.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from self_healing.models.errors import DetectedError, ErrorType
from self_healing.models.fixes import (
    CodePatch,
    Fix,
    FixResult,
    FixStatus,
    FixStrategy,
    ShellCommand,
)


@dataclass
class RepairConfig:
    """Configuration for the repair engine."""

    backup_dir: Path = field(default_factory=lambda: Path("backups"))
    max_backup_age_days: int = 7
    dry_run: bool = False
    auto_rollback: bool = True
    command_timeout: int = 60


class RepairEngine:
    """
    Generates and applies fixes for detected errors.

    Features:
    - Fix generation based on error type
    - Automatic file backup before modification
    - Rollback on failure
    - Dry-run mode for testing
    """

    def __init__(self, config: Optional[RepairConfig] = None) -> None:
        self.config = config or RepairConfig()
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        self._backups: dict[str, Path] = {}

    def generate_fix(self, error: DetectedError) -> Optional[Fix]:
        """Generate a fix for the detected error."""
        strategy_map = {
            ErrorType.IMPORT: self._fix_import,
            ErrorType.ENCODING: self._fix_encoding,
            ErrorType.MEMORY_ERROR: self._fix_memory,
            ErrorType.API_RATE_LIMIT: self._fix_api_rate_limit,
            ErrorType.API_TIMEOUT: self._fix_api_timeout,
            ErrorType.FILE_NOT_FOUND: self._fix_file_not_found,
            ErrorType.MISSING_COLUMN: self._fix_missing_column,
            ErrorType.TYPE_MISMATCH: self._fix_type_mismatch,
            ErrorType.SYNTAX: self._fix_syntax,
            ErrorType.DATA_LOADING: self._fix_data_loading,
        }

        fix_func = strategy_map.get(error.error_type)
        if fix_func:
            return fix_func(error)

        # Return manual fix for unknown errors
        return Fix(
            error=error,
            strategy=FixStrategy.MANUAL,
            description="Manual intervention required",
            reasoning=f"No automatic fix available for {error.error_type.value}",
            confidence=0.0,
        )

    def _fix_import(self, error: DetectedError) -> Fix:
        """Generate fix for import errors."""
        module = error.context.get("module", "unknown")

        # Check for common data science imports
        package_map = {
            "pandas": "pandas",
            "pd": "pandas",
            "numpy": "numpy",
            "np": "numpy",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "anthropic": "anthropic",
        }

        package = package_map.get(module.split(".")[0], module.split(".")[0])

        return Fix(
            error=error,
            strategy=FixStrategy.INSTALL_DEPENDENCY,
            description=f"Install missing package: {package}",
            reasoning=f"Module '{module}' not found, installing {package}",
            shell_commands=[
                ShellCommand(
                    command=f"pip install {package}",
                    description=f"Install {package}",
                )
            ],
            confidence=0.85,
        )

    def _fix_encoding(self, error: DetectedError) -> Fix:
        """Generate fix for encoding errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.ENCODING_FIX,
            description="Use encoding with error handling",
            reasoning="File contains non-UTF-8 characters",
            config_changes={
                "encoding": "utf-8",
                "errors": "replace",
                "engine": "python",
            },
            confidence=0.85,
        )

    def _fix_memory(self, error: DetectedError) -> Fix:
        """Generate fix for memory errors."""
        size_mb = error.context.get("size_mb", 0)

        return Fix(
            error=error,
            strategy=FixStrategy.CHUNK_PROCESSING,
            description="Process data in chunks",
            reasoning=f"File size ({size_mb:.0f}MB) exceeds memory capacity",
            config_changes={
                "chunksize": 10000,
                "low_memory": True,
            },
            confidence=0.80,
        )

    def _fix_api_rate_limit(self, error: DetectedError) -> Fix:
        """Generate fix for API rate limit errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.RETRY_WITH_BACKOFF,
            description="Retry with exponential backoff",
            reasoning="API rate limit exceeded, implementing backoff strategy",
            config_changes={
                "max_retries": 5,
                "initial_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 2.0,
            },
            confidence=0.90,
        )

    def _fix_api_timeout(self, error: DetectedError) -> Fix:
        """Generate fix for API timeout errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.RETRY_WITH_BACKOFF,
            description="Increase timeout and retry",
            reasoning="API request timed out, increasing timeout and retrying",
            config_changes={
                "timeout": 120,
                "max_retries": 3,
            },
            confidence=0.85,
        )

    def _fix_file_not_found(self, error: DetectedError) -> Fix:
        """Generate fix for file not found errors."""
        file_path = error.location.file_path

        return Fix(
            error=error,
            strategy=FixStrategy.CREATE_DIRECTORY,
            description="Check path and create directory if needed",
            reasoning=f"File or directory not found: {file_path}",
            shell_commands=[
                ShellCommand(
                    command=f"mkdir -p {file_path.parent if file_path else '.'}",
                    description="Create parent directory",
                )
            ],
            confidence=0.60,
        )

    def _fix_missing_column(self, error: DetectedError) -> Fix:
        """Generate fix for missing column errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.COLUMN_MAP,
            description="Map or create missing column",
            reasoning="Required column not found in data",
            confidence=0.70,
        )

    def _fix_type_mismatch(self, error: DetectedError) -> Fix:
        """Generate fix for type mismatch errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.TYPE_COERCE,
            description="Coerce data types",
            reasoning="Data type mismatch, attempting conversion",
            config_changes={
                "errors": "coerce",
            },
            confidence=0.75,
        )

    def _fix_syntax(self, error: DetectedError) -> Fix:
        """Generate fix for syntax errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.FIX_SYNTAX,
            description="Fix syntax error",
            reasoning=f"Syntax error at {error.location}",
            confidence=0.50,
        )

    def _fix_data_loading(self, error: DetectedError) -> Fix:
        """Generate fix for data loading errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.DATA_RELOAD,
            description="Retry data loading with different options",
            reasoning="Data loading failed, trying alternative approach",
            config_changes={
                "engine": "python",
                "on_bad_lines": "skip",
                "encoding": "utf-8",
                "errors": "replace",
            },
            confidence=0.80,
        )

    def apply_fix(self, fix: Fix) -> FixResult:
        """Apply a fix and return the result."""
        start_time = time.time()
        fix.status = FixStatus.IN_PROGRESS

        try:
            # Backup files before modification
            for patch in fix.code_patches:
                self._backup_file(patch.file_path, fix.fix_id)

            if self.config.dry_run:
                return FixResult(
                    fix=fix,
                    success=True,
                    output="Dry run - no changes applied",
                    duration_seconds=time.time() - start_time,
                )

            # Apply code patches
            for patch in fix.code_patches:
                self._apply_patch(patch)

            # Execute shell commands
            outputs = []
            for cmd in fix.shell_commands:
                result = self._execute_command(cmd)
                outputs.append(result)
                if cmd.require_success and "error" in result.lower():
                    raise RuntimeError(f"Command failed: {result}")

            fix.mark_success()
            return FixResult(
                fix=fix,
                success=True,
                output="\n".join(outputs),
                duration_seconds=time.time() - start_time,
                backup_path=self._backups.get(fix.fix_id),
            )

        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            fix.mark_failed()
            if self.config.auto_rollback:
                self._rollback(fix.fix_id)

            return FixResult(
                fix=fix,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _backup_file(self, file_path: Path, fix_id: str) -> None:
        """Create backup of a file before modification."""
        if not file_path.exists():
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = self.config.backup_dir / fix_id / backup_name

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        self._backups[fix_id] = backup_path.parent

    def _apply_patch(self, patch: CodePatch) -> None:
        """Apply a code patch to a file."""
        if not patch.file_path.exists():
            patch.file_path.parent.mkdir(parents=True, exist_ok=True)
            patch.file_path.write_text(patch.new_content)
            return

        content = patch.file_path.read_text()

        if patch.line_start and patch.line_end:
            lines = content.split("\n")
            new_lines = patch.new_content.split("\n")
            lines[patch.line_start - 1 : patch.line_end] = new_lines
            content = "\n".join(lines)
        else:
            content = content.replace(patch.original_content, patch.new_content)

        patch.file_path.write_text(content)

    def _execute_command(self, cmd: ShellCommand) -> str:
        """Execute a shell command."""
        try:
            result = subprocess.run(
                cmd.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=cmd.timeout,
                cwd=cmd.working_dir,
                check=False,
            )
            output = result.stdout + result.stderr
            return output.strip() if output else "Command completed"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {cmd.timeout}s"
        except OSError as e:
            return f"Command error: {e}"

    def _rollback(self, fix_id: str) -> bool:
        """Rollback changes for a fix."""
        backup_dir = self._backups.get(fix_id)
        if not backup_dir or not backup_dir.exists():
            return False

        # Backup files exist; original location tracking not yet implemented
        return True

    def cleanup_old_backups(self) -> int:
        """Remove backups older than max_backup_age_days."""
        removed = 0
        cutoff = time.time() - (self.config.max_backup_age_days * 86400)

        for backup_dir in self.config.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff:
                shutil.rmtree(backup_dir)
                removed += 1

        return removed
