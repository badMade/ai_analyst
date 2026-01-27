"""Visualizer utility for Agentic AI.

Provides data visualization and insights.
"""

from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for a chart."""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 80
    height: int = 20
    show_grid: bool = True


class Visualizer:
    """Text-based data visualizer.

    Provides ASCII-art visualizations for terminal output.
    Can be extended for matplotlib/plotly integration.
    """

    def __init__(self, config: ChartConfig | None = None):
        """Initialize the visualizer.

        Args:
            config: Default chart configuration.
        """
        self.config = config or ChartConfig()

    def line_chart(
        self,
        data: list[float],
        config: ChartConfig | None = None,
    ) -> str:
        """Create a text-based line chart.

        Args:
            data: Data points to plot.
            config: Chart configuration.

        Returns:
            ASCII art chart.
        """
        config = config or self.config

        if not data:
            return "No data to display"

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 1

        lines = []

        # Title
        if config.title:
            lines.append(config.title.center(config.width))
            lines.append("")

        # Y-axis label
        if config.y_label:
            lines.append(f"{config.y_label}")

        # Chart body
        for row in range(config.height, -1, -1):
            threshold = min_val + (row / config.height) * range_val
            line = f"{threshold:8.2f} |"

            for i, val in enumerate(data[:config.width - 10]):
                if val >= threshold:
                    line += "*"
                elif config.show_grid:
                    line += "."
                else:
                    line += " "

            lines.append(line)

        # X-axis
        lines.append(" " * 9 + "+" + "-" * min(len(data), config.width - 10))

        # X-axis label
        if config.x_label:
            lines.append(" " * 9 + config.x_label.center(config.width - 10))

        return "\n".join(lines)

    def bar_chart(
        self,
        data: dict[str, float],
        config: ChartConfig | None = None,
    ) -> str:
        """Create a text-based horizontal bar chart.

        Args:
            data: Dictionary of labels to values.
            config: Chart configuration.

        Returns:
            ASCII art bar chart.
        """
        config = config or self.config

        if not data:
            return "No data to display"

        max_val = max(data.values())
        max_label_len = max(len(str(k)) for k in data.keys())
        bar_width = config.width - max_label_len - 15

        lines = []

        # Title
        if config.title:
            lines.append(config.title.center(config.width))
            lines.append("")

        # Bars
        for label, value in data.items():
            bar_len = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * bar_len
            line = f"{label:>{max_label_len}} | {bar} {value:.2f}"
            lines.append(line)

        return "\n".join(lines)

    def histogram(
        self,
        data: list[float],
        bins: int = 10,
        config: ChartConfig | None = None,
    ) -> str:
        """Create a text-based histogram.

        Args:
            data: Data points.
            bins: Number of bins.
            config: Chart configuration.

        Returns:
            ASCII art histogram.
        """
        config = config or self.config

        if not data:
            return "No data to display"

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 1
        bin_width = range_val / bins

        # Count values in each bin
        counts = [0] * bins
        for val in data:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1

        max_count = max(counts)
        bar_height = config.height

        lines = []

        # Title
        if config.title:
            lines.append(config.title.center(config.width))
            lines.append("")

        # Chart body (vertical bars)
        for row in range(bar_height, 0, -1):
            threshold = (row / bar_height) * max_count
            line = f"{threshold:5.0f} |"

            for count in counts:
                if count >= threshold:
                    line += " ██"
                else:
                    line += "   "

            lines.append(line)

        # X-axis
        lines.append(" " * 6 + "+" + "-" * (bins * 3))

        # Bin labels
        labels = "      "
        for i in range(bins):
            bin_start = min_val + i * bin_width
            labels += f"{bin_start:3.0f}"
        lines.append(labels)

        return "\n".join(lines)

    def table(
        self,
        data: list[dict[str, Any]],
        columns: list[str] | None = None,
    ) -> str:
        """Create a text-based table.

        Args:
            data: List of dictionaries.
            columns: Column names (auto-detected if None).

        Returns:
            ASCII art table.
        """
        if not data:
            return "No data to display"

        columns = columns or list(data[0].keys())

        # Calculate column widths
        widths = {}
        for col in columns:
            max_width = len(col)
            for row in data:
                val_width = len(str(row.get(col, "")))
                max_width = max(max_width, val_width)
            widths[col] = max_width + 2

        # Build table
        lines = []

        # Header separator
        separator = "+" + "+".join("-" * widths[col] for col in columns) + "+"
        lines.append(separator)

        # Header row
        header = "|" + "|".join(
            col.center(widths[col]) for col in columns
        ) + "|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for row in data:
            row_str = "|" + "|".join(
                str(row.get(col, "")).center(widths[col])
                for col in columns
            ) + "|"
            lines.append(row_str)

        lines.append(separator)

        return "\n".join(lines)

    def progress_bar(
        self,
        current: float,
        total: float,
        width: int = 50,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Create a progress bar.

        Args:
            current: Current progress value.
            total: Total value.
            width: Bar width.
            prefix: Prefix text.
            suffix: Suffix text.

        Returns:
            Progress bar string.
        """
        percent = (current / total) * 100 if total > 0 else 0
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)

        return f"{prefix}|{bar}| {percent:5.1f}% {suffix}"

    def sparkline(self, data: list[float], width: int = 20) -> str:
        """Create a sparkline visualization.

        Args:
            data: Data points.
            width: Maximum width.

        Returns:
            Sparkline string.
        """
        if not data:
            return ""

        # Sparkline characters from low to high
        chars = "▁▂▃▄▅▆▇█"

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 1

        # Downsample if needed
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]

        sparkline = ""
        for val in data:
            idx = int((val - min_val) / range_val * (len(chars) - 1))
            sparkline += chars[idx]

        return sparkline
