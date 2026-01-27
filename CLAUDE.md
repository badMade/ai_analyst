# Claude Code Instructions

This repository contains an AI-powered data analyst tool built with the Anthropic Claude API.

## Project Overview

AI Analyst is a Python-based tool that leverages Claude for intelligent data analysis. It provides:
- Interactive REPL mode for data exploration
- Automated data analysis and insights generation
- CSV, JSON, Excel, and Parquet file processing
- Statistical analysis with tool-use agentic loop

## Repository Structure

```
ai_analyst/
├── run.py                      # CLI entry point wrapper
├── analyst.py                  # Core StandaloneAnalyst class (main analysis engine)
├── interactive.py              # REPL mode implementation
├── benchmark_quality.py        # Performance benchmarking utility
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # User documentation
├── CLAUDE.md                   # This file (AI assistant instructions)
│
├── src/ai_analyst/             # Main package source
│   ├── __init__.py
│   ├── cli.py                  # Click CLI commands (interactive, analyze, inspect, auth)
│   ├── tools/
│   │   ├── __init__.py
│   │   └── statistical.py      # Statistical analysis functions
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Settings, authentication, path sanitization
│
├── tests/                      # Test suite (pytest)
│   ├── conftest.py             # Pytest fixtures
│   ├── test_analyst.py         # StandaloneAnalyst tests
│   ├── test_config.py          # Configuration tests
│   ├── test_context.py         # AnalysisContext tests
│   ├── test_interactive.py     # REPL interface tests
│   ├── test_statistical.py     # Statistical tools tests
│   └── test_integration.py     # Integration tests
│
├── data/                       # Sample data files
│   └── sample_sales.csv
│
├── .github/
│   ├── workflows/              # CI/CD workflows
│   │   ├── claude.yml          # Claude Code assistant & agent
│   │   ├── claude-review.yml   # Custom Claude PR review
│   │   ├── google-ai.yml       # Gemini & Jules integration
│   │   ├── google-ai-review.yml
│   │   ├── openai-codex.yml    # ChatGPT & Codex integration
│   │   ├── openai-review.yml
│   │   ├── git-actions.yml     # Git automation
│   │   └── pr-validation.yml   # Basic PR validation
│   ├── actions/                # Custom GitHub actions
│   ├── scripts/                # Python review/automation scripts
│   └── PULL_REQUEST_TEMPLATE.md
│
└── .devcontainer/              # Dev container configuration
    └── devcontainer.json
```

## Key Components

### Core Files

| File | Description |
|------|-------------|
| `analyst.py` | Main `StandaloneAnalyst` class with Claude API integration, `AnalysisContext` for dataset management, and 10 analysis tools |
| `interactive.py` | REPL interface using Rich console for interactive data exploration |
| `src/ai_analyst/cli.py` | Click-based CLI with `interactive`, `analyze`, `inspect`, and `auth` commands |
| `src/ai_analyst/utils/config.py` | Pydantic settings, authentication logic, path sanitization |
| `src/ai_analyst/tools/statistical.py` | Statistical functions (descriptive stats, normality tests, trend detection) |

### Available Analysis Tools

The `StandaloneAnalyst` provides these tools via Claude's tool-use API:

| Tool | Description |
|------|-------------|
| `load_dataset` | Load CSV, JSON, Excel (.xlsx/.xls), Parquet files |
| `list_datasets` | List all currently loaded datasets |
| `preview_data` | Preview first N rows of a dataset |
| `describe_statistics` | Compute mean, std, min, max, quartiles for numeric columns |
| `compute_correlation` | Compute correlation matrix (Pearson, Spearman, Kendall) |
| `detect_outliers` | Detect outliers using IQR or Z-score methods |
| `group_analysis` | Perform grouped aggregations |
| `check_data_quality` | Assess missing values, duplicates, quality scores |
| `test_normality` | Shapiro-Wilk normality test |
| `analyze_trend` | Mann-Kendall trend detection |

## Development Guidelines

### Code Style

- **Python version**: 3.10+ required
- **Style**: Follow PEP 8 guidelines
- **Type hints**: Required for all function signatures (mypy strict mode)
- **Line length**: 100 characters max
- **Formatter/Linter**: Ruff with pycodestyle, Pyflakes, isort, flake8-bugbear rules
- Keep functions focused and well-documented
- Avoid over-engineering; make minimal changes needed

### Dependencies

**Core** (required):
- `anthropic>=0.40.0` - Claude API SDK
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.11.0` - Scientific functions
- `pydantic>=2.0.0` - Data validation
- `click>=8.1.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting
- `openpyxl>=3.1.0` - Excel support
- `pyarrow>=14.0.0` - Parquet support

**Optional groups** (install with `pip install -e ".[group]"`):
- `mcp` - MCP framework, LangChain ecosystem
- `viz` - matplotlib, seaborn, plotly
- `ml` - scikit-learn
- `dev` - pytest, ruff, mypy, pre-commit
- `notebook` - jupyter, ipykernel

### Authentication

AI Analyst supports two authentication methods:

#### Option 1: Claude Pro Subscription (Recommended)
```bash
claude login
```
Authenticates via OAuth and stores credentials locally. No API credits needed.

#### Option 2: API Key (Fallback)
```bash
export ANTHROPIC_API_KEY='your-api-key'
```

#### Authentication Priority
```bash
export AUTH_PREFERENCE=pro  # Use Pro subscription first (default)
export AUTH_PREFERENCE=api  # Use API key first
```

Check authentication status:
```bash
python run.py auth
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key (fallback if Pro not available) | - |
| `AUTH_PREFERENCE` | Authentication priority: `pro` or `api` | `pro` |
| `AI_ANALYST_MODEL` | Claude model to use | `claude-sonnet-4-20250514` |
| `AI_ANALYST_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |

### CLI Commands

```bash
# Analyze a file with a query
python run.py analyze data.csv -q "What are the sales trends?"

# Start interactive REPL session
python run.py interactive data.csv

# Quick data structure inspection
python run.py inspect data.csv

# Check authentication status
python run.py auth

# Show help
python run.py --help
```

## Testing

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_analyst.py

# Run with verbose output
pytest -v

# Run without coverage
pytest --no-cov
```

### Test Structure

- `conftest.py` - Shared fixtures (mock settings, sample data, API response mocks)
- Tests use pytest-asyncio for async tests
- Coverage tracking via pytest-cov

### Writing Tests

- Mock the Anthropic client for API tests
- Use `sample_csv_file` fixture for test data
- Test both success and error paths
- Ensure new features have corresponding tests

## Code Quality

### Linting and Formatting

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type checking
mypy src/ai_analyst
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Security Considerations

- **Path Sanitization**: All file paths are validated via `sanitize_path()` to prevent path traversal attacks. Paths must remain within `BASE_DATA_DIR`.
- **API Key Protection**: Never commit API keys. Use environment variables.
- **Input Validation**: User queries are passed directly to Claude; file operations are sandboxed.

## PR Review Guidelines

When reviewing PRs:

1. **Error Handling**: Verify proper exception handling, especially for file I/O and API calls
2. **Security**: Ensure no API keys or sensitive data are exposed; check path sanitization
3. **Memory Efficiency**: Validate pandas operations don't load excessive data into memory
4. **API Best Practices**: Check Claude API usage follows tool-use patterns correctly
5. **Type Safety**: Ensure type hints are present and mypy passes
6. **Test Coverage**: New features should include tests

## AI Integration

This repository integrates multiple AI assistants for development and PR reviews:

### Claude Code (Anthropic)
- `@claude` - General assistance and code review
- `@claude agent` - Automated code changes (members/owners only)
- Automatic PR reviews via custom Python script

### Google AI
- `@gemini` - Gemini AI Studio assistance
- `@jules` - Google Labs Jules coding agent
- **Gemini Code Assist** - Automatic PR reviews

### OpenAI ChatGPT/Codex
- `@chatgpt` - ChatGPT GPT-4o assistance
- `@codex` - Code-focused help
- `@codex-agent` - Automated code changes
- **ChatGPT Codex Review** - Automatic PR reviews

### Trigger Keywords

| Keyword | Action |
|---------|--------|
| `@claude` | Claude assistant responds |
| `@claude agent` | Claude makes code changes |
| `@gemini` | Gemini assistant responds |
| `@jules` | Jules coding agent |
| `@chatgpt` | ChatGPT responds |
| `@codex-agent` | Codex makes code changes |
| `@codex` | Code-focused help |

All PRs receive automated reviews from Claude, Gemini, and ChatGPT.

## GitHub Actions Secrets

Required secrets for CI/CD:

| Secret | Purpose |
|--------|---------|
| `ANTHROPIC_API_KEY` | Claude API access |
| `GEMINI_API_KEY` | Gemini API access |
| `OPENAI_API_KEY` | OpenAI API access |

## Architecture Notes

### Agentic Loop

The `StandaloneAnalyst.analyze()` method implements an agentic loop:

1. User query is sent to Claude with available tools
2. Claude responds with either text or tool calls
3. If tool calls: execute tools, add results, continue loop
4. If text response (`end_turn`): return final analysis
5. Maximum 15 iterations to prevent infinite loops

### Dataset Context

`AnalysisContext` maintains state across tool calls:
- Loaded datasets stored by name
- Results history preserved
- Supports multiple datasets simultaneously

### File Format Support

| Format | Extension | Reader |
|--------|-----------|--------|
| CSV | `.csv` | `pd.read_csv` |
| JSON | `.json` | `pd.read_json` |
| Excel | `.xlsx`, `.xls` | `pd.read_excel` |
| Parquet | `.parquet` | `pd.read_parquet` |
