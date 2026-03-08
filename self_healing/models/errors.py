"""
Error Models for AI Analyst Self-Healing System.

Defines error types, severities, and detection patterns specific to
data analysis workflows including pandas, numpy, API calls, and file I/O.
"""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ErrorType(Enum):
    """Categories of errors the system can detect and heal."""

    # Data errors
    DATA_LOADING = "data_loading"
    DATA_QUALITY = "data_quality"
    MISSING_COLUMN = "missing_column"
    TYPE_MISMATCH = "type_mismatch"
    ENCODING = "encoding"

    # Analysis errors
    ANALYSIS_FAILURE = "analysis_failure"
    STATISTICAL_ERROR = "statistical_error"
    MEMORY_ERROR = "memory_error"

    # API errors
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTH = "api_auth"
    API_TIMEOUT = "api_timeout"
    API_RESPONSE = "api_response"

    # File errors
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION = "permission"

    # Code errors
    SYNTAX = "syntax"
    IMPORT = "import"
    ATTRIBUTE = "attribute"
    KEY = "key"
    INDEX = "index"

    # CI/CD errors
    CI_FAILURE = "ci_failure"
    TEST_FAILURE = "test_failure"
    LINT_ERROR = "lint_error"

    # Generic
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for detected errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EnvironmentType(Enum):
    """Environment where the error occurred."""

    PYTHON = "python"
    PANDAS = "pandas"
    NUMPY = "numpy"
    API = "api"
    CI = "ci"
    SHELL = "shell"
    UNKNOWN = "unknown"


@dataclass
class ErrorLocation:
    """Location information for an error."""

    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    function_name: Optional[str] = None

    def __str__(self) -> str:
        parts = []
        if self.file_path:
            parts.append(str(self.file_path))
        if self.line_number:
            parts.append(f"line {self.line_number}")
        if self.column:
            parts.append(f"col {self.column}")
        if self.function_name:
            parts.append(f"in {self.function_name}")
        return ", ".join(parts) if parts else "unknown location"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "column": self.column,
            "function_name": self.function_name,
        }


@dataclass
class DetectedError:
    """
    Represents an error detected by the self-healing system.

    Contains all information needed to analyze and fix the error.
    """

    error_type: ErrorType
    severity: ErrorSeverity
    environment: EnvironmentType
    message: str
    location: ErrorLocation = field(default_factory=ErrorLocation)
    stack_trace: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error_id: str = field(default="")

    def __post_init__(self):
        if not self.error_id:
            self.error_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique error ID based on content."""
        content = f"{self.error_type.value}:{self.message}:{self.location}"
        return f"ERR-{hashlib.sha256(content.encode()).hexdigest()[:8].upper()}"

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        environment: EnvironmentType = EnvironmentType.PYTHON,
    ) -> DetectedError:
        """Create a DetectedError from a Python exception."""
        error_type = cls._classify_exception(exc)
        severity = cls._determine_severity(error_type)
        location = cls._extract_location(exc)

        return cls(
            error_type=error_type,
            severity=severity,
            environment=environment,
            message=str(exc),
            location=location,
            stack_trace=traceback.format_exc(),
            context={"exception_type": type(exc).__name__},
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> ErrorType:
        """Classify an exception into an ErrorType."""
        exc_type = type(exc).__name__
        exc_msg = str(exc).lower()

        # Data-related errors
        if "KeyError" in exc_type or "column" in exc_msg:
            return ErrorType.MISSING_COLUMN
        if "dtype" in exc_msg or "type" in exc_msg:
            return ErrorType.TYPE_MISMATCH
        if "encoding" in exc_msg or "codec" in exc_msg:
            return ErrorType.ENCODING
        if "memory" in exc_msg:
            return ErrorType.MEMORY_ERROR

        # File errors
        if "FileNotFoundError" in exc_type or "No such file" in exc_msg:
            return ErrorType.FILE_NOT_FOUND
        if "PermissionError" in exc_type:
            return ErrorType.PERMISSION

        # API errors
        if "rate" in exc_msg and "limit" in exc_msg:
            return ErrorType.API_RATE_LIMIT
        if "timeout" in exc_msg:
            return ErrorType.API_TIMEOUT
        if "401" in exc_msg or "403" in exc_msg or "auth" in exc_msg:
            return ErrorType.API_AUTH

        # Code errors
        if "SyntaxError" in exc_type:
            return ErrorType.SYNTAX
        if "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            return ErrorType.IMPORT
        if "AttributeError" in exc_type:
            return ErrorType.ATTRIBUTE
        if "KeyError" in exc_type:
            return ErrorType.KEY
        if "IndexError" in exc_type:
            return ErrorType.INDEX

        return ErrorType.UNKNOWN

    @staticmethod
    def _determine_severity(error_type: ErrorType) -> ErrorSeverity:
        """Determine severity based on error type."""
        critical_types = {
            ErrorType.MEMORY_ERROR,
            ErrorType.API_AUTH,
            ErrorType.SYNTAX,
        }
        error_types = {
            ErrorType.FILE_NOT_FOUND,
            ErrorType.IMPORT,
            ErrorType.DATA_LOADING,
            ErrorType.CI_FAILURE,
        }
        warning_types = {
            ErrorType.API_RATE_LIMIT,
            ErrorType.API_TIMEOUT,
            ErrorType.DATA_QUALITY,
            ErrorType.LINT_ERROR,
        }

        if error_type in critical_types:
            return ErrorSeverity.CRITICAL
        if error_type in error_types:
            return ErrorSeverity.ERROR
        if error_type in warning_types:
            return ErrorSeverity.WARNING
        return ErrorSeverity.INFO

    @staticmethod
    def _extract_location(exc: Exception) -> ErrorLocation:
        """Extract location from exception traceback."""
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            frame = tb[-1]
            return ErrorLocation(
                file_path=Path(frame.filename),
                line_number=frame.lineno,
                function_name=frame.name,
            )
        return ErrorLocation()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "environment": self.environment.value,
            "message": self.message,
            "location": self.location.to_dict(),
            "stack_trace": self.stack_trace,
            "context": self.context,
            "detected_at": self.detected_at,
        }


# Common error patterns for detection
ERROR_PATTERNS = {
    ErrorType.DATA_LOADING: [
        r"Could not read",
        r"Error tokenizing data",
        r"ParserError",
        r"EmptyDataError",
    ],
    ErrorType.MISSING_COLUMN: [
        r"KeyError.*not in index",
        r"Column.*not found",
        r"'(\w+)' not in list",
    ],
    ErrorType.TYPE_MISMATCH: [
        r"cannot convert",
        r"invalid literal",
        r"could not convert",
    ],
    ErrorType.API_RATE_LIMIT: [
        r"rate.?limit",
        r"429",
        r"too many requests",
    ],
    ErrorType.MEMORY_ERROR: [
        r"MemoryError",
        r"out of memory",
        r"killed",
    ],
}
