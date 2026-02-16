"""
Standalone AI Analyst

Direct analysis interface that works without MCP client adapters.
Uses Claude API directly with tool definitions for a simpler standalone setup.
"""

import asyncio
import json
import logging

import numpy as np
import pandas as pd
from anthropic import Anthropic, AsyncAnthropic

from ai_analyst.tools.statistical import (
    compute_descriptive_stats,
    detect_trend,
    test_normality,
)
from ai_analyst.utils.config import (
    AuthMethod,
    get_auth_method,
    sanitize_path,
)

logger = logging.getLogger(__name__)


class AnalysisContext:
    """Holds loaded datasets and analysis state."""

    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}
        self.results: list[dict] = []

    def load_dataset(self, file_path: str, name: str | None = None) -> dict:
        """Load dataset from file."""
        path = sanitize_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        dataset_name = name or path.stem
        self.datasets[dataset_name] = df

        return {
            "name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isna().sum().to_dict()
        }

    def get_dataset(self, name: str) -> pd.DataFrame:
        """Get loaded dataset by name."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not loaded. Available: {list(self.datasets.keys())}")
        return self.datasets[name]


# Tool definitions for Claude API
TOOLS = [
    {
        "name": "load_dataset",
        "description": "Load a dataset from a file (CSV, JSON, Excel, Parquet) into memory for analysis.",
        "input_schema": {
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
    },
    {
        "name": "list_datasets",
        "description": "List all currently loaded datasets.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "preview_data",
        "description": "Preview the first N rows of a loaded dataset.",
        "input_schema": {
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
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to include (optional)"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "describe_statistics",
        "description": "Compute descriptive statistics (mean, std, min, max, quartiles) for numeric columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to analyze (optional, all numeric if omitted)"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "compute_correlation",
        "description": "Compute correlation matrix between numeric columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to include (optional)"
                },
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman", "kendall"],
                    "description": "Correlation method (default: pearson)"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "detect_outliers",
        "description": "Detect outliers in a numeric column using IQR or Z-score method.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "column": {
                    "type": "string",
                    "description": "Column to analyze for outliers"
                },
                "method": {
                    "type": "string",
                    "enum": ["iqr", "zscore"],
                    "description": "Detection method (default: iqr)"
                },
                "threshold": {
                    "type": "number",
                    "description": "IQR multiplier (default 1.5) or Z-score threshold (default 3)"
                }
            },
            "required": ["dataset_name", "column"]
        }
    },
    {
        "name": "group_analysis",
        "description": "Perform grouped aggregation analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "group_by": {
                    "type": "string",
                    "description": "Column to group by"
                },
                "agg_column": {
                    "type": "string",
                    "description": "Column to aggregate"
                },
                "agg_functions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Aggregation functions (default: count, mean, sum, min, max)"
                }
            },
            "required": ["dataset_name", "group_by", "agg_column"]
        }
    },
    {
        "name": "check_data_quality",
        "description": "Perform comprehensive data quality assessment including missing values, duplicates, and issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "test_normality",
        "description": "Test if a column follows a normal distribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "column": {
                    "type": "string",
                    "description": "Column to test"
                }
            },
            "required": ["dataset_name", "column"]
        }
    },
    {
        "name": "analyze_trend",
        "description": "Detect monotonic trends in a numeric column using Mann-Kendall test.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Name of the loaded dataset"
                },
                "column": {
                    "type": "string",
                    "description": "Column to analyze for trends"
                }
            },
            "required": ["dataset_name", "column"]
        }
    }
]


class StandaloneAnalyst:
    """Standalone AI Analyst using direct Claude API calls."""

    SYSTEM_PROMPT = """You are an expert data analyst assistant. You have tools to load, inspect, and analyze datasets.

When analyzing data:
1. First load the dataset and understand its structure
2. Check data quality before detailed analysis
3. Use appropriate statistical methods
4. Explain findings clearly with actionable insights
5. Note any limitations or caveats

Be thorough but efficient. Present results in a structured, easy-to-understand format."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        # Get authentication method (Pro subscription first, API key as fallback)
        auth_method, api_key = get_auth_method()

        if auth_method == AuthMethod.PRO_SUBSCRIPTION:
            logger.info("Using Claude Pro subscription authentication")
        else:
            logger.info("Using API key authentication")

        self.model = model
        self._api_key = api_key
        self._auth_method = auth_method
        self._client: Anthropic | None = None
        self._async_client: AsyncAnthropic | None = None
        # Backwards compatibility: some callers (e.g. direct `_execute_tool()` usage)
        # may rely on a persistent `context` attribute. New analysis calls should use
        # per-request AnalysisContext instances created inside `analyze`/`analyze_async`.
        self.context: AnalysisContext = AnalysisContext()
        self.max_iterations = 15

    @property
    def client(self) -> Anthropic:
        """Lazily initialize the synchronous Anthropic client."""
        if self._client is None:
            if self._auth_method == AuthMethod.PRO_SUBSCRIPTION:
                self._client = Anthropic()
            else:
                self._client = Anthropic(api_key=self._api_key)
        return self._client

    @client.setter
    def client(self, value: Anthropic) -> None:
        """Allow tests to inject a mocked client."""
        self._client = value

    @property
    def async_client(self) -> AsyncAnthropic:
        """Lazily initialize the asynchronous Anthropic client."""
        if self._async_client is None:
            if self._auth_method == AuthMethod.PRO_SUBSCRIPTION:
                self._async_client = AsyncAnthropic()
            else:
                self._async_client = AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    @async_client.setter
    def async_client(self, value: AsyncAnthropic) -> None:
        """Allow tests to inject a mocked async client."""
        self._async_client = value

    def _build_messages(self, query: str, file_path: str | None = None) -> list[dict]:
        """Build initial message payload for the Anthropic API."""
        user_content = query
        if file_path:
            user_content = f"Analyze the file at: {file_path}\n\n{query}"
        return [{"role": "user", "content": user_content}]

    def _extract_text_response(self, response) -> str:
        """Extract the final text content from a model response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _execute_tool(self, tool_name: str, tool_input: dict, context: AnalysisContext | None = None) -> str:
        """Execute a tool and return result as string."""
        context = context or self.context
        try:
            if tool_name == "load_dataset":
                result = context.load_dataset(
                    tool_input["file_path"],
                    tool_input.get("name")
                )

            elif tool_name == "list_datasets":
                result = {
                    "datasets": list(context.datasets.keys()),
                    "count": len(context.datasets)
                }

            elif tool_name == "preview_data":
                df = context.get_dataset(tool_input["dataset_name"])
                n_rows = tool_input.get("n_rows", 10)
                columns = tool_input.get("columns")

                if columns:
                    df = df[columns]

                result = {
                    "data": df.head(n_rows).to_dict(orient="records"),
                    "total_rows": len(df),
                    "columns": df.columns.tolist()
                }

            elif tool_name == "describe_statistics":
                df = context.get_dataset(tool_input["dataset_name"])
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

                result = {"statistics": stats}

            elif tool_name == "compute_correlation":
                df = context.get_dataset(tool_input["dataset_name"])
                columns = tool_input.get("columns")
                method = tool_input.get("method", "pearson")

                if columns:
                    df = df[columns]

                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr(method=method)

                correlations = []
                cols = corr_matrix.columns.tolist()
                for i, col_a in enumerate(cols):
                    for col_b in cols[i+1:]:
                        corr = corr_matrix.loc[col_a, col_b]
                        if pd.notna(corr):
                            correlations.append({
                                "column_a": col_a,
                                "column_b": col_b,
                                "correlation": round(corr, 4)
                            })

                correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                result = {"correlations": correlations, "method": method}

            elif tool_name == "detect_outliers":
                df = context.get_dataset(tool_input["dataset_name"])
                column = tool_input["column"]
                method = tool_input.get("method", "iqr")
                threshold = tool_input.get("threshold", 1.5 if method == "iqr" else 3)

                series = df[column].dropna()

                if method == "iqr":
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    outliers = series[(series < lower) | (series > upper)]
                else:
                    mean, std = series.mean(), series.std()
                    z_scores = np.abs((series - mean) / std)
                    outliers = series[z_scores > threshold]
                    lower = mean - threshold * std
                    upper = mean + threshold * std

                result = {
                    "method": method,
                    "threshold": threshold,
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "outlier_count": len(outliers),
                    "outlier_percentage": round(len(outliers) / len(series) * 100, 2),
                    "total_analyzed": len(series)
                }

            elif tool_name == "group_analysis":
                df = context.get_dataset(tool_input["dataset_name"])
                group_by = tool_input["group_by"]
                agg_column = tool_input["agg_column"]
                agg_functions = tool_input.get("agg_functions", ["count", "mean", "sum", "min", "max"])

                grouped = df.groupby(group_by)[agg_column].agg(agg_functions)

                result = {
                    "group_by": group_by,
                    "agg_column": agg_column,
                    "n_groups": len(grouped),
                    "results": grouped.reset_index().to_dict(orient="records")
                }

            elif tool_name == "check_data_quality":
                df = context.get_dataset(tool_input["dataset_name"])

                total_rows = len(df)
                total_cells = df.size
                # Calculate null counts for all columns at once to avoid re-scanning in the loop
                null_counts = df.isna().sum()
                null_cells = null_counts.sum()
                duplicate_rows = df.duplicated().sum()

                column_issues = {}
                # Vectorized calculation of null percentages using precomputed null_counts
                null_pcts = (null_counts / total_rows) * 100

                # Filter only columns with nulls
                cols_with_nulls = null_pcts[null_pcts > 0]

                for col, pct in cols_with_nulls.items():
                    column_issues[col] = [f"Missing: {pct:.1f}%"]

                null_percentage = (null_cells / total_cells) * 100 if total_cells else 0.0
                duplicate_percentage = (
                    (duplicate_rows / total_rows) * 100 if total_rows else 0.0
                )
                quality_score = 100 - (null_percentage * 0.5 + duplicate_percentage * 0.5)

                result = {
                    "total_rows": total_rows,
                    "total_columns": len(df.columns),
                    "null_cells": int(null_cells),
                    "null_percentage": round(null_percentage, 2),
                    "duplicate_rows": int(duplicate_rows),
                    "duplicate_percentage": round(duplicate_percentage, 2),
                    "column_issues": column_issues,
                    "quality_score": round(quality_score, 2)
                }

            elif tool_name == "test_normality":
                df = context.get_dataset(tool_input["dataset_name"])
                column = tool_input["column"]

                test_result = test_normality(df[column].dropna())
                result = {
                    "column": column,
                    "test": test_result.test_name,
                    "statistic": round(test_result.statistic, 4) if not np.isnan(test_result.statistic) else None,
                    "p_value": round(test_result.p_value, 4) if not np.isnan(test_result.p_value) else None,
                    "is_normal": not test_result.significant,
                    "interpretation": test_result.interpretation
                }

            elif tool_name == "analyze_trend":
                df = context.get_dataset(tool_input["dataset_name"])
                column = tool_input["column"]

                trend_result = detect_trend(df[column].dropna().values)
                result = trend_result

            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return json.dumps({"error": str(e)})

    def _process_tool_use_blocks(self, response, messages: list[dict], context: AnalysisContext) -> None:
        """Process tool use blocks for sync analysis loop."""
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logger.info(f"Executing tool: {block.name}")
                result = self._execute_tool(block.name, block.input, context=context)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    async def _process_tool_use_blocks_async(self, response, messages: list[dict], context: AnalysisContext) -> None:
        """Process tool use blocks for async analysis loop."""
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logger.info(f"Executing tool: {block.name}")
                result = await asyncio.to_thread(self._execute_tool, block.name, block.input, context)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    def analyze(self, query: str, file_path: str | None = None) -> str:
        """
        Run analysis based on user query.
        
        Args:
            query: Analysis request
            file_path: Optional file to analyze (will be mentioned in query context)
        
        Returns:
            Final analysis response
        """
        messages = self._build_messages(query=query, file_path=file_path)
        context = self.context

        # Agentic loop
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}")

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )
            except Exception:
                logger.exception("Anthropic API call failed during analyze")
                raise

            # Check if done
            if response.stop_reason == "end_turn":
                return self._extract_text_response(response)

            # Process tool use
            if response.stop_reason == "tool_use":
                self._process_tool_use_blocks(response, messages, context)
            else:
                # Unexpected stop reason
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

        return "Analysis reached maximum iterations. Please try a more specific query."

    async def analyze_async(self, query: str, file_path: str | None = None) -> str:
        """
        Async implementation of analyze using AsyncAnthropic.

        Args:
            query: Analysis request
            file_path: Optional file to analyze (will be mentioned in query context)

        Returns:
            Final analysis response
        """
        messages = self._build_messages(query=query, file_path=file_path)
        context = AnalysisContext()

        # Agentic loop
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}")

            try:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )
            except Exception:
                logger.exception("AsyncAnthropic API call failed during analyze_async")
                raise

            # Check if done
            if response.stop_reason == "end_turn":
                return self._extract_text_response(response)

            # Process tool use
            if response.stop_reason == "tool_use":
                await self._process_tool_use_blocks_async(response, messages, context)
            else:
                # Unexpected stop reason
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

        return "Analysis reached maximum iterations. Please try a more specific query."



def create_analyst(model: str = "claude-sonnet-4-20250514") -> StandaloneAnalyst:
    """Create a standalone analyst instance."""
    return StandaloneAnalyst(model=model)
