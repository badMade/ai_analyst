# CLAUDE.md - AI Analyst Project Guide

## Project Overview

AI Analyst is a standalone Python AI-powered data analyst that uses the Anthropic Claude API with agentic tool use. It enables automated data analysis and insights generation through an agentic loop pattern.

## Key Files

- `analyst.py` - Core engine with `StandaloneAnalyst` class and tool definitions (agentic loop, max 15 iterations)
- `interactive.py` - Interactive REPL mode for exploratory data analysis
- `run.py` - CLI entry point
- `src/ai_analyst/tools/statistical.py` - Statistical analysis functions
- `src/ai_analyst/utils/config.py` - Configuration and path sanitization

## Architecture

```
User Query → Claude API + Tools → Tool Execution → Results → Claude Interpretation → Repeat → Final Response
```

The analyst uses 10 built-in tools: `load_dataset`, `list_datasets`, `preview_data`, `describe_statistics`, `compute_correlation`, `detect_outliers`, `group_analysis`, `check_data_quality`, `test_normality`, `analyze_trend`

## Development Commands

```bash
# Run analysis
python run.py analyze data.csv -q "What are the trends?"

# Interactive REPL
python run.py interactive data.csv

# Inspect data
python run.py inspect data.csv

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check .
ruff format .

# Type checking
mypy .
```

## Code Style

- Python 3.10+ (uses `|` union types, `match` statements)
- Line length: 100 characters
- Use `ruff` for linting and formatting
- Strict type hints with `mypy`
- Pydantic for data validation

## Adding New Analysis Tools

1. Define tool schema in `TOOLS` array in `analyst.py`
2. Implement handler in `_execute_tool()` method
3. Add statistical functions to `src/ai_analyst/tools/statistical.py` if needed

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for Claude API access
- `AI_ANALYST_MODEL` - Model to use (default: claude-sonnet-4-20250514)
- `AI_ANALYST_LOG_LEVEL` - Logging level (default: INFO)

## Security Guidelines

- Never hardcode API keys; use environment variables
- Use `sanitize_path()` from `utils/config.py` for file paths
- Validate all user inputs before processing
- No `eval()` or `exec()` on user data
- Use specific exception handling (no bare `except:`)

## Testing

- Use pytest with `pytest-asyncio` for async tests
- Mock Claude API calls in tests
- Test DataFrame operations with sample data
- Coverage reports via `pytest-cov`

## Dependencies

Core: `anthropic`, `pandas`, `numpy`, `scipy`, `pydantic`, `click`, `rich`

Optional groups:
- `[mcp]` - MCP framework integration
- `[viz]` - matplotlib, seaborn, plotly
- `[ml]` - scikit-learn
- `[dev]` - testing and linting tools
