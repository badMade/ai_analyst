"""
Error Detector for AI Analyst Self-Healing System.

Detects errors through static analysis, runtime monitoring, and
pattern matching specific to data analysis workflows.
"""

from __future__ import annotations

import ast
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

from self_healing.models.errors import (
    DetectedError,
    ErrorLocation,
    ErrorSeverity,
    ErrorType,
    EnvironmentType,
    ERROR_PATTERNS,
)


class ErrorDetector:
    """
    Detects errors in code and runtime execution.

    Capabilities:
    - Static syntax checking via AST
    - Import/dependency validation
    - Pattern-based error detection
    - Runtime exception hooking
    - Data analysis specific checks
    """

    def __init__(
        self,
        on_error_detected: Optional[Callable[[DetectedError], None]] = None,
    ) -> None:
        self.on_error_detected = on_error_detected
        self._original_excepthook = None
        self._monitored_files: set[Path] = set()

    def install_exception_hook(self) -> None:
        """Install global exception hook for runtime monitoring."""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._exception_handler

    def uninstall_exception_hook(self) -> None:
        """Restore original exception hook."""
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None

    def _exception_handler(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_tb: Optional[object],
    ) -> None:
        """Handle uncaught exceptions."""
        if isinstance(exc_value, Exception):
            error = DetectedError.from_exception(exc_value)
            if self.on_error_detected:
                self.on_error_detected(error)

        # Call original handler
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_tb)

    def check_syntax(self, file_path: Path) -> list[DetectedError]:
        """Check Python file for syntax errors."""
        errors = []
        file_path = Path(file_path)

        if not file_path.exists():
            errors.append(
                DetectedError(
                    error_type=ErrorType.FILE_NOT_FOUND,
                    severity=ErrorSeverity.ERROR,
                    environment=EnvironmentType.PYTHON,
                    message=f"File not found: {file_path}",
                    location=ErrorLocation(file_path=file_path),
                )
            )
            return errors

        try:
            content = file_path.read_text()
            ast.parse(content)
        except SyntaxError as e:
            errors.append(
                DetectedError(
                    error_type=ErrorType.SYNTAX,
                    severity=ErrorSeverity.CRITICAL,
                    environment=EnvironmentType.PYTHON,
                    message=str(e.msg) if e.msg else "Syntax error",
                    location=ErrorLocation(
                        file_path=file_path,
                        line_number=e.lineno,
                        column=e.offset,
                    ),
                    context={"text": e.text},
                )
            )
        except Exception as e:
            errors.append(DetectedError.from_exception(e))

        return errors

    def check_dependencies(self, file_path: Path) -> list[DetectedError]:
        """Check for missing imports/dependencies."""
        errors = []
        file_path = Path(file_path)

        if not file_path.exists():
            return errors

        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._check_import(alias.name):
                            errors.append(
                                self._create_import_error(
                                    alias.name, file_path, node.lineno
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._check_import(node.module):
                        errors.append(
                            self._create_import_error(
                                node.module, file_path, node.lineno
                            )
                        )

        except SyntaxError:
            pass  # Handled by check_syntax
        except Exception as e:
            errors.append(DetectedError.from_exception(e))

        return errors

    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(module_name.split(".")[0])
            return True
        except ImportError:
            return False

    def _create_import_error(
        self,
        module_name: str,
        file_path: Path,
        line_number: int,
    ) -> DetectedError:
        """Create an import error."""
        return DetectedError(
            error_type=ErrorType.IMPORT,
            severity=ErrorSeverity.ERROR,
            environment=EnvironmentType.PYTHON,
            message=f"Cannot import module: {module_name}",
            location=ErrorLocation(
                file_path=file_path,
                line_number=line_number,
            ),
            context={"module": module_name},
        )

    def detect_from_output(
        self,
        output: str,
        file_path: Optional[Path] = None,
    ) -> list[DetectedError]:
        """Detect errors from command output or logs."""
        errors = []

        for error_type, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    # Extract line number if present
                    line_match = re.search(r"line (\d+)", output, re.IGNORECASE)
                    line_num = int(line_match.group(1)) if line_match else None

                    errors.append(
                        DetectedError(
                            error_type=error_type,
                            severity=DetectedError._determine_severity(error_type),
                            environment=EnvironmentType.PYTHON,
                            message=self._extract_error_message(output),
                            location=ErrorLocation(
                                file_path=file_path,
                                line_number=line_num,
                            ),
                            context={"raw_output": output[:500]},
                        )
                    )
                    break  # One error type per output block

        return errors

    def _extract_error_message(self, output: str) -> str:
        """Extract clean error message from output."""
        lines = output.strip().split("\n")

        # Look for common error patterns
        for line in reversed(lines):
            if "Error:" in line or "Exception:" in line:
                return line.strip()

        # Return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:200]

        return "Unknown error"

    def check_data_file(self, file_path: Path) -> list[DetectedError]:
        """Check data file for common issues."""
        errors = []
        file_path = Path(file_path)

        if not file_path.exists():
            errors.append(
                DetectedError(
                    error_type=ErrorType.FILE_NOT_FOUND,
                    severity=ErrorSeverity.ERROR,
                    environment=EnvironmentType.PANDAS,
                    message=f"Data file not found: {file_path}",
                    location=ErrorLocation(file_path=file_path),
                )
            )
            return errors

        # Check encoding
        try:
            file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(
                DetectedError(
                    error_type=ErrorType.ENCODING,
                    severity=ErrorSeverity.WARNING,
                    environment=EnvironmentType.PANDAS,
                    message=f"File has non-UTF-8 encoding: {file_path}",
                    location=ErrorLocation(file_path=file_path),
                )
            )

        # Check file size for memory concerns
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            errors.append(
                DetectedError(
                    error_type=ErrorType.MEMORY_ERROR,
                    severity=ErrorSeverity.WARNING,
                    environment=EnvironmentType.PANDAS,
                    message=f"Large file ({size_mb:.0f}MB) may cause memory issues",
                    location=ErrorLocation(file_path=file_path),
                    context={"size_mb": size_mb},
                )
            )

        return errors

    def check_ci_output(self, output: str) -> list[DetectedError]:
        """Detect errors from CI/CD output."""
        errors = []

        # Check for common CI failure patterns
        ci_patterns = {
            ErrorType.TEST_FAILURE: [
                r"FAILED",
                r"test.*failed",
                r"AssertionError",
            ],
            ErrorType.LINT_ERROR: [
                r"lint.*error",
                r"flake8",
                r"pylint",
                r"mypy.*error",
            ],
            ErrorType.CI_FAILURE: [
                r"workflow.*failed",
                r"job.*failed",
                r"build.*failed",
            ],
        }

        for error_type, patterns in ci_patterns.items():
            for pattern in patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    errors.append(
                        DetectedError(
                            error_type=error_type,
                            severity=ErrorSeverity.ERROR,
                            environment=EnvironmentType.CI,
                            message=self._extract_error_message(output),
                            context={"ci_output": output[:1000]},
                        )
                    )
                    break

        return errors
