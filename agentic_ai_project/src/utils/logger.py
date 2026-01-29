"""Logger utility for Agentic AI.

Provides event tracking and structured logging for agents.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    level: LogLevel
    source: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "source": self.source,
            "message": self.message,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AgentLogger:
    """Logger for agent events and activities."""

    def __init__(
        self,
        name: str,
        log_file: str | Path | None = None,
        level: LogLevel = LogLevel.INFO,
    ):
        """Initialize the agent logger.

        Args:
            name: Logger name (usually agent name).
            log_file: Optional file path for log output.
            level: Minimum log level to record.
        """
        self.name = name
        self.level = level
        self.entries: list[LogEntry] = []
        self._log_file = Path(log_file) if log_file else None
        self._python_logger = logging.getLogger(f"agent.{name}")

    def log(
        self,
        level: LogLevel,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Log an event.

        Args:
            level: Log level.
            message: Log message.
            data: Optional structured data.

        Returns:
            The created log entry.
        """
    ) -> LogEntry | None:

        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            source=self.name,
            message=message,
            data=data or {},
        )

        self.entries.append(entry)

        # Also log to Python logger
        py_level = getattr(logging, level.value.upper())
        self._python_logger.log(py_level, f"{message} | {entry.data}")

        # Write to file if configured
        if self._log_file:
            self._write_to_file(entry)

        return entry

    def debug(self, message: str, **data) -> LogEntry:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, data)

    def info(self, message: str, **data) -> LogEntry:
        """Log info message."""
        return self.log(LogLevel.INFO, message, data)

    def warning(self, message: str, **data) -> LogEntry:
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, data)

    def error(self, message: str, **data) -> LogEntry:
        """Log error message."""
        return self.log(LogLevel.ERROR, message, data)

    def critical(self, message: str, **data) -> LogEntry:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, data)

    def log_action(
        self,
        action: str,
        parameters: dict[str, Any] | None = None,
        result: Any = None,
    ) -> LogEntry:
        """Log an agent action.

        Args:
            action: Action name.
            parameters: Action parameters.
            result: Action result.

        Returns:
            Log entry.
        """
        return self.info(
            f"Action: {action}",
            parameters=parameters,
            result=result,
        )

    def log_decision(
        self,
        decision: str,
        options: list[str] | None = None,
        reason: str = "",
    ) -> LogEntry:
        """Log a decision.

        Args:
            decision: The decision made.
            options: Available options.
            reason: Reason for decision.

        Returns:
            Log entry.
        """
        return self.info(
            f"Decision: {decision}",
            options=options,
            reason=reason,
        )

    def log_state_change(
        self,
        old_state: str,
        new_state: str,
        trigger: str = "",
    ) -> LogEntry:
        """Log a state change.

        Args:
            old_state: Previous state.
            new_state: New state.
            trigger: What triggered the change.

        Returns:
            Log entry.
        """
        return self.info(
            f"State: {old_state} -> {new_state}",
            trigger=trigger,
        )

    def get_entries(
        self,
        level: LogLevel | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[LogEntry]:
        """Get log entries with optional filtering.

        Args:
            level: Filter by level.
            since: Filter by timestamp.
            limit: Maximum entries to return.

        Returns:
            Filtered log entries.
        """
        entries = self.entries

        if level:
            entries = [e for e in entries if e.level == level]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if limit:
            entries = entries[-limit:]

        return entries

    def export_to_json(self, path: str | Path) -> None:
        """Export logs to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        data = [entry.to_dict() for entry in self.entries]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear all log entries."""
        self.entries.clear()

    def _write_to_file(self, entry: LogEntry) -> None:
        """Write entry to log file."""
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._log_file, "a") as f:
            f.write(entry.to_json() + "\n")
