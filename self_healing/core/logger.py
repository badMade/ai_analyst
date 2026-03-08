"""
Healing Logger for AI Analyst Self-Healing System.

Provides structured logging, changelog generation, and audit trails
for all healing activities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from self_healing.models.errors import DetectedError
    from self_healing.models.fixes import Fix, FixResult


class LogLevel(Enum):
    """Log levels for the healing system."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealingLogEntry:
    """A single entry in the healing log."""

    entry_id: str
    timestamp: str
    level: LogLevel
    event_type: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "event_type": self.event_type,
            "message": self.message,
            "details": self.details,
        }


class HealingLogger:
    """
    Centralized logging for the self-healing system.

    Maintains:
    - Standard Python logging
    - JSON structured log
    - Human-readable changelog
    - Statistics tracking
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: LogLevel = LogLevel.INFO,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path("self-heal-logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_level = log_level
        self._entry_counter = 0
        self._stats = {
            "errors_detected": 0,
            "fixes_attempted": 0,
            "fixes_succeeded": 0,
            "fixes_failed": 0,
        }

        # Setup Python logger
        self._logger = logging.getLogger("self_healing")
        self._logger.setLevel(getattr(logging, log_level.value.upper()))

        # File handler
        log_file = self.log_dir / "healing.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        if not self._logger.handlers:
            self._logger.addHandler(handler)

            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self._logger.addHandler(console)

        # JSON log file
        self._json_log_path = self.log_dir / "healing_log.json"
        self._changelog_path = self.log_dir / "CHANGELOG.md"

    def _next_entry_id(self) -> str:
        """Generate next entry ID."""
        self._entry_counter += 1
        return f"HEAL-{self._entry_counter:04d}"

    def _write_json_entry(self, entry: HealingLogEntry) -> None:
        """Append entry to JSON log."""
        entries = []
        if self._json_log_path.exists():
            try:
                data = json.loads(self._json_log_path.read_text())
                entries = data.get("entries", [])
            except json.JSONDecodeError:
                entries = []

        entries.append(entry.to_dict())

        self._json_log_path.write_text(
            json.dumps({"version": "1.0", "entries": entries}, indent=2)
        )

    def initialize_changelog(self) -> None:
        """Initialize or update changelog header."""
        if not self._changelog_path.exists():
            header = f"""# AI Analyst Self-Healing Changelog

Automatically generated log of all healing activities.

Generated: {datetime.now(timezone.utc).isoformat()}

---

"""
            self._changelog_path.write_text(header)

    def _append_changelog(self, content: str) -> None:
        """Append content to changelog."""
        self.initialize_changelog()
        with open(self._changelog_path, "a") as f:
            f.write(content)

    # Logging methods
    def debug(self, message: str, **details: Any) -> None:
        self._logger.debug(message)
        self._log("debug", "DEBUG", message, details)

    def info(self, message: str, **details: Any) -> None:
        self._logger.info(message)
        self._log("info", "INFO", message, details)

    def warning(self, message: str, **details: Any) -> None:
        self._logger.warning(message)
        self._log("warning", "WARNING", message, details)

    def error(self, message: str, **details: Any) -> None:
        self._logger.error(message)
        self._log("error", "ERROR", message, details)

    def critical(self, message: str, **details: Any) -> None:
        self._logger.critical(message)
        self._log("critical", "CRITICAL", message, details)

    def _log(
        self,
        level: str,
        event_type: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        """Internal logging method."""
        entry = HealingLogEntry(
            entry_id=self._next_entry_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=LogLevel(level),
            event_type=event_type,
            message=message,
            details=details,
        )
        self._write_json_entry(entry)

    def log_error_detected(self, error: DetectedError) -> None:
        """Log a detected error."""
        self._stats["errors_detected"] += 1

        self.info(
            f"Error detected: {error.error_type.value}",
            error_id=error.error_id,
            severity=error.severity.value,
            message=error.message,
        )

        changelog_entry = f"""
## Error Detected: {error.error_id}

**Timestamp:** {error.detected_at}

- **Type:** {error.error_type.value.upper()}
- **Severity:** {error.severity.value.upper()}
- **Message:** {error.message}
- **Location:** {error.location}

"""
        self._append_changelog(changelog_entry)

    def log_fix_proposed(self, fix: Fix) -> None:
        """Log a proposed fix."""
        self.info(
            f"Fix proposed: {fix.strategy.value}",
            fix_id=fix.fix_id,
            confidence=fix.confidence,
            description=fix.description,
        )

    def log_fix_applied(self, fix: Fix) -> None:
        """Log an applied fix."""
        self._stats["fixes_attempted"] += 1

        self.info(
            f"Fix applied: {fix.fix_id}",
            strategy=fix.strategy.value,
            description=fix.description,
        )

    def log_verification_result(
        self,
        fix: Fix,
        passed: bool,
        output: str,
    ) -> None:
        """Log verification result."""
        if passed:
            self._stats["fixes_succeeded"] += 1
            self.info(f"Verification passed for {fix.fix_id}")
        else:
            self._stats["fixes_failed"] += 1
            self.warning(f"Verification failed for {fix.fix_id}", output=output)

    def log_healing_complete(
        self,
        error: DetectedError,
        fix: Fix,
        result: FixResult,
        duration: float,
    ) -> None:
        """Log complete healing cycle."""
        status = "SUCCESS" if result.success and result.verification_passed else "FAILED"

        changelog_entry = f"""
### Fix Applied: {fix.fix_id}

- **Strategy:** {fix.strategy.value}
- **Description:** {fix.description}
- **Reasoning:** {fix.reasoning}
- **Confidence:** {fix.confidence:.0%}

### Result

- **Status:** {status}
- **Verification:** {"PASSED" if result.verification_passed else "FAILED"}
- **Attempts:** {result.attempt_number}
- **Duration:** {duration:.2f}s

---

"""
        self._append_changelog(changelog_entry)

    def log_max_retries_exceeded(
        self,
        error: DetectedError,
        max_retries: int,
    ) -> None:
        """Log when max retries exceeded."""
        self.error(
            f"Max retries ({max_retries}) exceeded for {error.error_id}",
            error_type=error.error_type.value,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get healing statistics."""
        total = self._stats["fixes_attempted"]
        success_rate = (
            self._stats["fixes_succeeded"] / total * 100 if total > 0 else 0
        )

        return {
            **self._stats,
            "success_rate": f"{success_rate:.1f}%",
        }
