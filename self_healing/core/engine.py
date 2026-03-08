"""
Self-Healing Engine for AI Analyst.

Main orchestrator that coordinates detection, repair, and validation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from self_healing.core.detector import ErrorDetector
from self_healing.core.logger import HealingLogger, LogLevel
from self_healing.core.repair import RepairConfig, RepairEngine
from self_healing.core.validator import FixValidator, ValidationConfig
from self_healing.models.errors import DetectedError
from self_healing.models.fixes import Fix, FixPlan, FixResult, FixStatus


@dataclass
class HealingConfig:
    """Configuration for the self-healing engine."""

    auto_heal: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    log_dir: Path = field(default_factory=lambda: Path("self-heal-logs"))
    backup_dir: Path = field(default_factory=lambda: Path("backups"))
    log_level: LogLevel = LogLevel.INFO
    dry_run: bool = False
    validate_fixes: bool = True
    run_tests: bool = True


class SelfHealingEngine:
    """
    Main orchestrator for the self-healing system.

    Usage:
        engine = SelfHealingEngine(auto_heal=True)

        # Context manager for protected execution
        with engine.protect():
            run_analysis()

        # Manual healing
        errors = engine.scan_directory(Path("."))
        plan = engine.create_fix_plan(errors)
        results = engine.execute_plan(plan)
    """

    def __init__(
        self,
        config: Optional[HealingConfig] = None,
        auto_heal: bool = True,
    ) -> None:
        self.config = config or HealingConfig(auto_heal=auto_heal)

        # Initialize components
        self.logger = HealingLogger(
            log_dir=self.config.log_dir,
            log_level=self.config.log_level,
        )
        self.logger.initialize_changelog()

        self.detector = ErrorDetector(on_error_detected=self._on_error_detected)

        self.repair_engine = RepairEngine(
            RepairConfig(
                backup_dir=self.config.backup_dir,
                dry_run=self.config.dry_run,
            )
        )

        self.validator = FixValidator(
            ValidationConfig(
                run_tests=self.config.run_tests,
            )
        )

        # State
        self._active = False
        self._error_queue: list[DetectedError] = []
        self._fix_history: list[FixResult] = []

    @contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager for protected execution with auto-healing."""
        self._active = True
        self.detector.install_exception_hook()

        try:
            yield
        except Exception as e:
            error = DetectedError.from_exception(e)
            self._on_error_detected(error)

            if self.config.auto_heal:
                result = self.heal_error(error)
                if not result or not result.success:
                    raise
            else:
                raise
        finally:
            self.detector.uninstall_exception_hook()
            self._active = False

    def _on_error_detected(self, error: DetectedError) -> None:
        """Callback when an error is detected."""
        self.logger.log_error_detected(error)
        self._error_queue.append(error)

        if self.config.auto_heal and self._active:
            self.heal_error(error)

    def heal_error(
        self,
        error: DetectedError,
        max_retries: Optional[int] = None,
    ) -> Optional[FixResult]:
        """Attempt to heal a single error."""
        max_retries = max_retries or self.config.max_retries

        for attempt in range(1, max_retries + 1):
            # Generate fix
            fix = self.repair_engine.generate_fix(error)
            if not fix:
                self.logger.warning(f"No fix generated for {error.error_id}")
                return None

            self.logger.log_fix_proposed(fix)

            # Apply fix
            self.logger.log_fix_applied(fix)
            result = self.repair_engine.apply_fix(fix)
            result.attempt_number = attempt

            # Validate
            if self.config.validate_fixes and result.success:
                validation = self.validator.validate(fix, result)
                result.verification_passed = validation.passed
                self.logger.log_verification_result(
                    fix, validation.passed, validation.output
                )

            # Log completion
            self.logger.log_healing_complete(
                error, fix, result, result.duration_seconds
            )
            self._fix_history.append(result)

            if result.success and result.verification_passed:
                return result

            # Retry with delay
            if attempt < max_retries:
                time.sleep(self.config.retry_delay * attempt)

        self.logger.log_max_retries_exceeded(error, max_retries)
        return result if result else None

    def scan_directory(
        self,
        directory: Path,
        pattern: str = "*.py",
    ) -> list[DetectedError]:
        """Scan directory for errors."""
        errors = []
        directory = Path(directory)

        for file_path in directory.rglob(pattern):
            if "__pycache__" in str(file_path) or ".git" in str(file_path):
                continue

            # Check syntax
            syntax_errors = self.detector.check_syntax(file_path)
            errors.extend(syntax_errors)

            # Check dependencies
            dep_errors = self.detector.check_dependencies(file_path)
            errors.extend(dep_errors)

        return errors

    def scan_data_files(
        self,
        directory: Path,
        patterns: list[str] = None,
    ) -> list[DetectedError]:
        """Scan data files for potential issues."""
        patterns = patterns or ["*.csv", "*.json", "*.parquet"]
        errors = []
        directory = Path(directory)

        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                file_errors = self.detector.check_data_file(file_path)
                errors.extend(file_errors)

        return errors

    def create_fix_plan(self, errors: list[DetectedError]) -> FixPlan:
        """Create a plan for fixing multiple errors."""
        plan = FixPlan()

        for error in errors:
            fix = self.repair_engine.generate_fix(error)
            if fix:
                plan.add_fix(fix)

        # Estimate duration
        plan.estimated_duration = len(plan.fixes) * 5.0  # 5 seconds per fix estimate

        # Require approval for critical fixes
        plan.requires_approval = any(
            fix.confidence < 0.7 for fix in plan.fixes
        )

        return plan

    def execute_plan(
        self,
        plan: FixPlan,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[FixResult]:
        """Execute a fix plan."""
        results = []
        total = len(plan.fixes)

        for i, fix_id in enumerate(plan.execution_order):
            # Find fix by ID
            fix = next((f for f in plan.fixes if f.fix_id == fix_id), None)
            if not fix:
                continue

            self.logger.log_fix_applied(fix)
            result = self.repair_engine.apply_fix(fix)

            if self.config.validate_fixes and result.success:
                validation = self.validator.validate(fix, result)
                result.verification_passed = validation.passed

            results.append(result)
            self._fix_history.append(result)

            if on_progress:
                on_progress(i + 1, total)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get healing statistics."""
        stats = self.logger.get_statistics()

        # Add history summary
        successful = sum(1 for r in self._fix_history if r.success)
        verified = sum(1 for r in self._fix_history if r.verification_passed)

        stats.update({
            "total_fixes_in_session": len(self._fix_history),
            "successful_in_session": successful,
            "verified_in_session": verified,
        })

        return stats

    def clear_error_queue(self) -> None:
        """Clear the error queue."""
        self._error_queue.clear()

    def get_pending_errors(self) -> list[DetectedError]:
        """Get errors that haven't been fixed."""
        return list(self._error_queue)
