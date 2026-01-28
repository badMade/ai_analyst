"""
MCP Data Analyst Server

Provides data analysis capabilities via Model Context Protocol (MCP).
Wraps the StandaloneAnalyst functionality for MCP clients.
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Check if MCP dependencies are available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    stdio_server = None
    Tool = None
    TextContent = None


class DataAnalystServer:
    """MCP Server for data analysis operations."""

    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP dependencies not installed. "
                "Install with: pip install ai-analyst[mcp]"
            )

        self.server = Server("data-analyst")
        self._setup_handlers()

        # Lazy import to avoid circular dependencies
        from ai_analyst.analyst import AnalysisContext
        self.context = AnalysisContext()

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available analysis tools."""
            return [
                Tool(
                    name="load_dataset",
                    description="Load a dataset from a file (CSV, JSON, Excel, Parquet) into memory for analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the data file"
                            },
                            "name": {
                                "type": "string",
                                "description": "Optional name for the dataset (defaults to filename)"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="list_datasets",
                    description="List all currently loaded datasets.",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="preview_data",
                    description="Preview the first N rows of a loaded dataset.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "Name of the loaded dataset"
                            },
                            "n_rows": {
                                "type": "integer",
                                "description": "Number of rows to preview (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["dataset_name"]
                    }
                ),
                Tool(
                    name="describe_statistics",
                    description="Compute descriptive statistics for numeric columns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "Name of the loaded dataset"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific columns to analyze (optional)"
                            }
                        },
                        "required": ["dataset_name"]
                    }
                ),
                Tool(
                    name="check_data_quality",
                    description="Assess data quality including missing values and duplicates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "Name of the loaded dataset"
                            }
                        },
                        "required": ["dataset_name"]
                    }
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute an analysis tool."""
            try:
                result = await self._execute_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                logger.exception(f"Tool execution error: {name}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict[str, Any]:
        """Execute a tool and return result."""
        import numpy as np
        import pandas as pd

        from ai_analyst.tools.statistical import compute_descriptive_stats

        if tool_name == "load_dataset":
            return self.context.load_dataset(
                tool_input["file_path"],
                tool_input.get("name")
            )

        elif tool_name == "list_datasets":
            return {
                "datasets": list(self.context.datasets.keys()),
                "count": len(self.context.datasets)
            }

        elif tool_name == "preview_data":
            df = self.context.get_dataset(tool_input["dataset_name"])
            n_rows = tool_input.get("n_rows", 10)

            return {
                "data": df.head(n_rows).to_dict(orient="records"),
                "total_rows": len(df),
                "columns": df.columns.tolist()
            }

        elif tool_name == "describe_statistics":
            df = self.context.get_dataset(tool_input["dataset_name"])
            columns = tool_input.get("columns")

            if columns:
                df = df[columns]

            numeric_df = df.select_dtypes(include=[np.number])
            stats = []

            for col in numeric_df.columns:
                stats.append({
                    "column": col,
                    **compute_descriptive_stats(numeric_df[col])
                })

            return {"statistics": stats}

        elif tool_name == "check_data_quality":
            df = self.context.get_dataset(tool_input["dataset_name"])

            total_rows = len(df)
            total_cells = df.size

            null_counts = df.isna().sum()
            null_cells = null_counts.sum()
            null_percentage = (null_cells / total_cells) * 100 if total_cells > 0 else 0

            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0

            quality_score = 100 - (null_percentage * 0.5 + duplicate_percentage * 0.5)

            return {
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "null_cells": int(null_cells),
                "null_percentage": round(null_percentage, 2),
                "duplicate_rows": int(duplicate_rows),
                "duplicate_percentage": round(duplicate_percentage, 2),
                "quality_score": round(quality_score, 2)
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def run_server() -> None:
    """Entry point for running the MCP data analyst server."""
    if not MCP_AVAILABLE:
        print("Error: MCP dependencies not installed.")
        print("Install with: pip install ai-analyst[mcp]")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO)
    server = DataAnalystServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    run_server()
