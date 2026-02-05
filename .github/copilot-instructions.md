# GitHub Copilot Custom Instructions

This repository contains **AI Analyst**, a Python-based data analysis tool powered by the Anthropic Claude API.

## Project Context

AI Analyst provides:
- Interactive REPL mode for data exploration
- Automated data analysis and insights generation
- Multi-format file processing (CSV, JSON, Excel, Parquet)
- Statistical analysis with tool-use agentic loop
- Claude Pro OAuth and API key authentication

## Copilot Agent Instructions

When working as an agent on this codebase:

### Task Approach
1. **Read before modifying** - Always examine existing code before making changes
2. **Minimal changes** - Make only the changes necessary to accomplish the task
3. **Preserve patterns** - Follow existing code patterns and conventions
4. **Test coverage** - Include tests for new functionality
5. **Security first** - Use `sanitize_path()` for all file operations

### Common Tasks

**Adding a new analysis tool:**
1. Add the tool function in `analyst.py` or `src/ai_analyst/tools/statistical.py`
2. Register the tool in `StandaloneAnalyst.TOOLS` list
3. Add handler in `_execute_tool()` method
4. Write tests in `tests/test_analyst.py` or `tests/test_statistical.py`

**Adding a CLI command:**
1. Add command in `src/ai_analyst/cli.py` using Click decorators
2. Follow existing patterns with `@click.command()` and `@click.option()`
3. Register command in the CLI group

**Modifying configuration:**
1. Update Pydantic model in `src/ai_analyst/utils/config.py`
2. Add environment variable documentation in both `CLAUDE.md` and this file

## Code Style Requirements

### Python Standards
- **Python 3.10+** features required (use `match` statements, union types with `|`)
- Follow **PEP 8** style guidelines strictly
- Always include **type hints** for function parameters and return values
- **Line length**: 100 characters max
- Keep functions focused with single responsibilities

### Import Organization
```python
# Standard library
import os
from pathlib import Path
from typing import Any

# Third-party
import pandas as pd
import numpy as np
from anthropic import Anthropic
from pydantic import BaseModel

# Local
from ai_analyst.utils.config import get_settings, sanitize_path
from ai_analyst.tools.statistical import normality_test
```

### Error Handling
```python
# Preferred pattern - specific exceptions with context
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found: {path}")
except pd.errors.EmptyDataError:
    raise ValueError(f"Empty dataset: {path}")
except Exception as e:
    logger.error(f"Failed to load {path}: {e}")
    raise
```

## Architecture

### Repository Structure
```
ai_analyst/
├── run.py                      # CLI entry point wrapper
├── analyst.py                  # Core StandaloneAnalyst class (main engine)
├── interactive.py              # REPL mode implementation
├── src/ai_analyst/
│   ├── cli.py                  # Click CLI commands
│   ├── tools/
│   │   └── statistical.py      # Statistical analysis functions
│   └── utils/
│       └── config.py           # Pydantic settings, auth, path sanitization
├── tests/                      # pytest test suite
└── .github/
    ├── copilot/mcp.json        # MCP server configuration
    └── copilot-instructions.md # This file
```

### Agentic Loop Pattern

The core analysis engine (`StandaloneAnalyst.analyze()`) implements a tool-use agentic loop:

```python
# Pattern used in analyst.py
async def analyze(self, query: str) -> str:
    messages = [{"role": "user", "content": query}]

    for _ in range(MAX_ITERATIONS):  # 15 max iterations
        response = await client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            tools=self.TOOLS,
            messages=messages,
        )

        # Check for tool use
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" and not tool_use_blocks:
            # Final text response - return to user
            return extract_text(response)

        # Execute tools and continue loop
        for tool_use in tool_use_blocks:
            result = self._execute_tool(tool_use.name, tool_use.input)
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            ]})
```

### Available Analysis Tools

| Tool | Description | Module |
|------|-------------|--------|
| `load_dataset` | Load CSV, JSON, Excel, Parquet files | analyst.py |
| `list_datasets` | List all loaded datasets | analyst.py |
| `preview_data` | Preview first N rows | analyst.py |
| `describe_statistics` | Compute descriptive stats | analyst.py |
| `compute_correlation` | Correlation matrix (Pearson/Spearman/Kendall) | analyst.py |
| `detect_outliers` | IQR or Z-score outlier detection | analyst.py |
| `group_analysis` | Grouped aggregations | analyst.py |
| `check_data_quality` | Missing values, duplicates, quality scores | analyst.py |
| `test_normality` | Shapiro-Wilk normality test | statistical.py |
| `analyze_trend` | Mann-Kendall trend detection | statistical.py |

## Key Patterns

### Claude API Usage
```python
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model=os.getenv("AI_ANALYST_MODEL", "claude-sonnet-4-20250514"),
    system="You are AI Analyst, a data analysis assistant.",
    max_tokens=4096,
    tools=TOOLS,  # Tool definitions for agentic loop
    messages=[{"role": "user", "content": prompt}],
)
```

### DataFrame Operations
```python
# Prefer vectorized operations
df["normalized"] = (df["value"] - df["value"].mean()) / df["value"].std()

# Use .copy() to avoid SettingWithCopyWarning
subset = df[df["status"] == "active"].copy()
subset["processed"] = True

# Method chaining for readability
result = (
    df.groupby("category")
    .agg({"value": ["mean", "std"], "count": "sum"})
    .reset_index()
)
```

### Safe Data Loading
```python
from pathlib import Path
import pandas as pd
from ai_analyst.utils.config import sanitize_path

def load_data(path: str | Path) -> pd.DataFrame:
    """Load data from various formats with validation."""
    safe_path = sanitize_path(path)  # Prevents path traversal

    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {safe_path}")

    suffix = safe_path.suffix.lower()

    loaders = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
    }

    if suffix not in loaders:
        raise ValueError(f"Unsupported format: {suffix}")

    df = loaders[suffix](safe_path)
    if df.empty:
        raise ValueError(f"Empty dataset: {safe_path}")
    return df
```

### CLI Commands (Click)
```python
import click
from rich.console import Console

console = Console()

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--query", "-q", help="Analysis query")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def analyze(file: str, query: str | None, verbose: bool) -> None:
    """Analyze a dataset with natural language queries."""
    if verbose:
        console.print(f"[dim]Loading {file}...[/dim]")
    # ... implementation
```

### Rich Console Output
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Status messages
console.print("[green]Analysis complete![/green]")
console.print("[yellow]Warning:[/yellow] Missing values detected")

# Tables
table = Table(title="Summary Statistics")
table.add_column("Metric", style="cyan")
table.add_column("Value", justify="right")
table.add_row("Mean", "42.5")
console.print(table)

# Panels for emphasis
console.print(Panel("Key Insight: Strong positive correlation", title="Finding"))
```

## Security Guidelines

**Critical security rules:**

1. **Path Sanitization** - Always use `sanitize_path()` for file operations
   ```python
   from ai_analyst.utils.config import sanitize_path
   safe_path = sanitize_path(user_input_path)  # Validates within BASE_DATA_DIR
   ```

2. **No Hardcoded Credentials** - Use environment variables
   ```python
   # NEVER do this
   api_key = "sk-ant-..."

   # Always do this
   api_key = os.environ.get("ANTHROPIC_API_KEY")
   ```

3. **Avoid Dangerous Operations** - No `eval()`, `exec()`, or shell injection
   ```python
   # NEVER do this
   result = eval(user_query)

   # Instead, use safe alternatives
   result = pd.eval(expression, local_dict={"df": df})  # Limited safe eval
   ```

4. **Sanitize Log Output** - Mask sensitive data
   ```python
   logger.info(f"API call to {endpoint}")  # OK
   logger.info(f"Using key {api_key}")     # NEVER - exposes secrets
   ```

## Testing Requirements

### Test Structure
```python
# tests/test_analyst.py
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for API tests."""
    with patch("analyst.Anthropic") as mock:
        yield mock

@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10.0, 20.0, 30.0],
        "category": ["A", "B", "A"]
    })

def test_load_dataset(sample_csv_file):
    """Test dataset loading with various formats."""
    # Test implementation

@pytest.mark.asyncio
async def test_analyze_query(mock_anthropic, sample_df):
    """Test the agentic analysis loop."""
    # Test implementation
```

### Running Tests
```bash
# All tests with coverage
pytest

# Specific test file
pytest tests/test_analyst.py -v

# Pattern matching
pytest -k "test_load" -v
```

## Dependencies

### Core (Required)
| Package | Version | Purpose |
|---------|---------|---------|
| anthropic | >=0.40.0 | Claude API SDK |
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| scipy | >=1.11.0 | Scientific functions |
| pydantic | >=2.0.0 | Data validation |
| click | >=8.1.0 | CLI framework |
| rich | >=13.0.0 | Terminal formatting |

### Optional Groups
- `mcp` - MCP framework, LangChain ecosystem
- `viz` - matplotlib, seaborn, plotly
- `ml` - scikit-learn
- `dev` - pytest, ruff, mypy, pre-commit
- `notebook` - jupyter, ipykernel

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Conditional | - | API key (required if not using Pro) |
| `AUTH_PREFERENCE` | No | `pro` | Auth priority: `pro` or `api` |
| `AI_ANALYST_MODEL` | No | `claude-sonnet-4-20250514` | Model ID |
| `AI_ANALYST_LOG_LEVEL` | No | `INFO` | Logging verbosity |

## PR Review Checklist

When reviewing pull requests, verify:

- [ ] Type hints present on all functions
- [ ] No hardcoded credentials or API keys
- [ ] Path sanitization used for file operations
- [ ] Proper error handling with specific exceptions
- [ ] Memory-efficient pandas operations (no unnecessary copies)
- [ ] Tests included for new functionality
- [ ] Documentation updated if API changes
- [ ] No sensitive data in log statements
- [ ] CLI commands follow Click conventions
- [ ] Ruff and mypy checks pass
