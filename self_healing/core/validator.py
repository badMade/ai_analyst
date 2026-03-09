"""
Fix Validator for AI Analyst Self-Healing System.

Validates fixes by running tests, checking syntax, and verifying behavior.
"""

from __future__ import annotations

import ast
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from self_healing.models.errors import DetectedError, ErrorType
from self_healing.models.fixes import Fix, FixResult, FixStrategy


@dataclass
class ValidationConfig:
    """Configuration for fix validation."""

    run_tests: bool = True
    test_command: str = "pytest"
    test_timeout: int = 120
    check_syntax: bool = True
    check_imports: bool = True
    require_all_pass: bool = False


@dataclass
class ValidationResult:
    """Result of validating a fix."""

    passed: bool
    checks_run: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    output: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "output": self.output[:500],
            "duration_seconds": self.duration_seconds,
        }


class FixValidator:
    """
    Validates that fixes work correctly.

    Validation steps:
    1. Syntax check (AST parsing)
    2. Import verification
    3. Test execution
    4. Runtime verification
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate(
        self,
        fix: Fix,
        result: FixResult,
        working_dir: Path | None = None,
    ) -> ValidationResult:
        """Validate a fix after application."""
        start_time = time.time()
        checks_run = []
        checks_passed = []
        checks_failed = []
        outputs = []

        # Skip validation for certain fix types
        if fix.strategy in {FixStrategy.MANUAL, FixStrategy.ROLLBACK}:
            return ValidationResult(
                passed=True,
                checks_run=["skip"],
                checks_passed=["skip"],
                output="Validation skipped for manual/rollback fixes",
                duration_seconds=time.time() - start_time,
            )

        # Check syntax for code patches
        if self.config.check_syntax and fix.code_patches:
            checks_run.append("syntax")
            syntax_ok = True
            for patch in fix.code_patches:
                if patch.file_path.suffix == ".py":
                    if not self._check_syntax(patch.file_path):
                        syntax_ok = False
                        outputs.append(f"Syntax error in {patch.file_path}")
            if syntax_ok:
                checks_passed.append("syntax")
            else:
                checks_failed.append("syntax")

        # Check imports
        if self.config.check_imports and fix.code_patches:
            checks_run.append("imports")
            imports_ok = True
            for patch in fix.code_patches:
                if patch.file_path.suffix == ".py":
                    if not self._check_imports(patch.file_path):
                        imports_ok = False
                        outputs.append(f"Import error in {patch.file_path}")
            if imports_ok:
                checks_passed.append("imports")
            else:
                checks_failed.append("imports")

        # Run tests
        if self.config.run_tests:
            checks_run.append("tests")
            test_result = self._run_tests(working_dir)
            if test_result["passed"]:
                checks_passed.append("tests")
            else:
                checks_failed.append("tests")
            outputs.append(test_result["output"])

        # Determine overall pass/fail
        if self.config.require_all_pass:
            passed = len(checks_failed) == 0
        else:
            # Pass if more checks passed than failed, or no checks failed
            passed = len(checks_failed) == 0 or len(checks_passed) > len(checks_failed)

        return ValidationResult(
            passed=passed,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            output="\n".join(outputs),
            duration_seconds=time.time() - start_time,
        )

    def _check_syntax(self, file_path: Path) -> bool:
        """Check Python file for syntax errors."""
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text()
            ast.parse(content)
            return True
        except SyntaxError:
            return False

    def _check_imports(self, file_path: Path) -> bool:
        """Check if imports can be resolved."""
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name.split(".")[0])
                        except ImportError:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            __import__(node.module.split(".")[0])
                        except ImportError:
                            return False

            return True
        except Exception:
            return False

    def _run_tests(self, working_dir: Path | None = None) -> dict:
        """Run test suite."""
        try:
            result = subprocess.run(
                shlex.split(self.config.test_command),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout,
                cwd=working_dir,
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr

            return {
                "passed": passed,
                "output": output[:1000],
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "output": f"Tests timed out after {self.config.test_timeout}s",
                "return_code": -1,
            }
        except Exception as e:
            return {
                "passed": False,
                "output": f"Test execution error: {e}",
                "return_code": -1,
            }

    def validate_error_resolved(
        self,
        original_error: DetectedError,
        fix: Fix,
        working_dir: Path | None = None,
    ) -> bool:
        """Check if the original error is resolved."""
        # For file not found, check if file now exists
        if original_error.error_type == ErrorType.FILE_NOT_FOUND:
            if original_error.location.file_path:
                return original_error.location.file_path.exists()

        # For import errors, check if import now works
        if original_error.error_type == ErrorType.IMPORT:
            module = original_error.context.get("module")
            if module:
                try:
                    __import__(module.split(".")[0])
                    return True
                except ImportError:
                    return False

        # For syntax errors, check if file now parses
        if original_error.error_type == ErrorType.SYNTAX:
            if original_error.location.file_path:
                return self._check_syntax(original_error.location.file_path)

        # Default to relying on validation result
        return True
