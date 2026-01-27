"""Metrics utility for Agentic AI.

Provides performance measurement and tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import time
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric."""
    name: str
    type: MetricType
    value: float = 0.0
    values: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def record(self, value: float) -> None:
        """Record a value."""
        self.value = value
        self.values.append(value)
        self.timestamp = time.time()

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter."""
        self.value += amount
        self.timestamp = time.time()

    def get_stats(self) -> dict[str, float]:
        """Get statistics for histogram/timer metrics."""
        if not self.values:
            return {}

        return {
            "count": len(self.values),
            "sum": sum(self.values),
            "mean": statistics.mean(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "std": statistics.stdev(self.values) if len(self.values) > 1 else 0.0,
            "median": statistics.median(self.values),
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: "MetricsCollector", metric_name: str):
        self.collector = collector
        self.metric_name = metric_name
        self.start_time: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record(self.metric_name, duration)


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self, name: str = "default"):
        """Initialize the metrics collector.

        Args:
            name: Collector name.
        """
        self.name = name
        self.metrics: dict[str, Metric] = {}
        self._callbacks: list[Callable[[Metric], None]] = []

    def register(
        self,
        name: str,
        metric_type: MetricType,
        labels: dict[str, str] | None = None,
    ) -> Metric:
        """Register a new metric.

        Args:
            name: Metric name.
            metric_type: Type of metric.
            labels: Optional labels.

        Returns:
            The registered metric.
        """
        metric = Metric(
            name=name,
            type=metric_type,
            labels=labels or {},
        )
        self.metrics[name] = metric
        return metric

    def record(self, name: str, value: float) -> None:
        """Record a metric value.

        Args:
            name: Metric name.
            value: Value to record.
        """
        if name not in self.metrics:
            # Auto-register as gauge
            self.register(name, MetricType.GAUGE)

        metric = self.metrics[name]
        metric.record(value)

        # Trigger callbacks
        for callback in self._callbacks:
            callback(metric)

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            amount: Amount to increment.
        """
        if name not in self.metrics:
            self.register(name, MetricType.COUNTER)

        self.metrics[name].increment(amount)

    def timer(self, name: str) -> Timer:
        """Create a timer context manager.

        Args:
            name: Timer metric name.

        Returns:
            Timer context manager.
        """
        if name not in self.metrics:
            self.register(name, MetricType.TIMER)
        return Timer(self, name)

    def get(self, name: str) -> Metric | None:
        """Get a metric by name."""
        return self.metrics.get(name)

    def get_value(self, name: str) -> float:
        """Get current value of a metric."""
        metric = self.metrics.get(name)
        return metric.value if metric else 0.0

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all metrics.

        Returns:
            Dictionary of metric stats.
        """
        stats = {}
        for name, metric in self.metrics.items():
            if metric.type in (MetricType.HISTOGRAM, MetricType.TIMER):
                stats[name] = metric.get_stats()
            else:
                stats[name] = {"value": metric.value}
        return stats

    def add_callback(self, callback: Callable[[Metric], None]) -> None:
        """Add a callback for metric updates.

        Args:
            callback: Function to call on metric update.
        """
        self._callbacks.append(callback)

    def reset(self, name: str | None = None) -> None:
        """Reset metrics.

        Args:
            name: Specific metric to reset, or None for all.
        """
        if name:
            if name in self.metrics:
                metric = self.metrics[name]
                metric.value = 0.0
                metric.values.clear()
        else:
            for metric in self.metrics.values():
                metric.value = 0.0
                metric.values.clear()

    def export(self) -> dict[str, Any]:
        """Export all metrics.

        Returns:
            Dictionary of all metrics data.
        """
        return {
            "collector": self.name,
            "timestamp": time.time(),
            "metrics": {
                name: {
                    "type": metric.type.value,
                    "value": metric.value,
                    "labels": metric.labels,
                    "stats": metric.get_stats() if metric.values else None,
                }
                for name, metric in self.metrics.items()
            },
        }


# Global metrics collector
_global_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector("global")
    return _global_collector
