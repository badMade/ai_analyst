# GitHub Copilot Custom Instructions

This repository contains **AI Analyst**, a Python-based data analysis tool powered by the Anthropic Claude API.

## Project Context

AI Analyst provides:
- Interactive REPL mode for data exploration
- Automated data analysis and insights generation
- CSV/DataFrame processing with pandas
- Statistical analysis tools

## Code Style Requirements

When suggesting or reviewing code:

### Python Standards
- **Python 3.10+** features are required (use `match` statements, union types with `|`, etc.)
- Follow **PEP 8** style guidelines strictly
- Always include **type hints** for function parameters and return values
- Keep functions focused with single responsibilities

### Import Organization
```python
# Standard library
import os
from pathlib import Path

# Third-party
import pandas as pd
import numpy as np
from anthropic import Anthropic

# Local
from ai_analyst.utils.config import Config
```

### Error Handling
- Use specific exception types, not bare `except:`
- Always handle API errors gracefully with retries where appropriate
- Log errors with context using the standard logging module

## Key Patterns

### Claude API Usage
```python
# Preferred pattern for Claude API calls
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model=os.getenv("AI_ANALYST_MODEL", "claude-sonnet-4-20250514"),
    system="You are AI Analyst, a Python-based data analysis assistant.",
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
)
```

### DataFrame Operations
- Prefer vectorized operations over iterrows()
- Use `.copy()` to create explicit copies of DataFrame slices, which avoids `SettingWithCopyWarning`.
- Chain operations using method chaining where readable
- Always validate data types before operations

### CLI Commands (Click)
```python
@click.command()
@click.option("--input", "-i", type=click.Path(exists=True), help="Input file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def command(input: str, verbose: bool) -> None:
    """Command description."""
    pass
```

## Architecture

```
src/ai_analyst/
├── analyst.py          # Core AIAnalyst class - main entry point
├── cli.py              # Click-based CLI commands
├── interactive.py      # REPL implementation
├── tools/
│   └── statistical.py  # Statistical analysis functions
└── utils/
    └── config.py       # Configuration management
```

## Security Guidelines

**Critical security rules for all code:**

1. **Never hardcode API keys** - Always use environment variables
2. **Validate all file paths** - Prevent path traversal attacks
3. **Sanitize user input** - Especially in REPL mode
4. **No secrets in logs** - Mask sensitive data in log output
5. **Safe DataFrame operations** - Avoid `eval()` or `exec()` on user data

## Dependencies

### Core (Required)
- `anthropic` - Claude API client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical functions

### CLI/UI
- `click` - Command-line interface
- `rich` - Terminal formatting
- `pydantic` - Data validation

### Optional
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning features

## PR Review Checklist

When reviewing pull requests, verify:

- [ ] Type hints present on all functions
- [ ] No hardcoded credentials or API keys
- [ ] Proper error handling with specific exceptions
- [ ] Memory-efficient pandas operations (no unnecessary copies)
- [ ] Tests included for new functionality
- [ ] Documentation updated if API changes
- [ ] No sensitive data in log statements
- [ ] CLI commands follow Click conventions

## Testing Requirements

- Use `pytest` for all tests
- Test files should mirror source structure in `tests/`
- Mock external API calls (Claude, file I/O)
- Aim for meaningful test coverage on new code

## Common Patterns to Suggest

### Configuration Loading
```python
from ai_analyst.utils.config import get_settings

settings = get_settings()
```

### Data Loading
```python
import pandas as pd
from pathlib import Path

def load_data(path: str | Path) -> pd.DataFrame:
    """Load data from various file formats (e.g., CSV, JSON) with validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
from pathlib import Path
import pandas as pd
from ai_analyst.utils.config import sanitize_path

def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV data with validation."""
    safe_path = sanitize_path(path)
    df = pd.read_csv(safe_path)
    if df.empty:
        raise ValueError(f"Empty dataset: {safe_path}")
    return df
```

### Rich Console Output
```python
from rich.console import Console
from rich.table import Table

console = Console()
console.print("[green]Success![/green]")
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API authentication |
| `AI_ANALYST_MODEL` | No | Model ID (default: claude-sonnet-4-20250514) |
| `AI_ANALYST_LOG_LEVEL` | No | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
