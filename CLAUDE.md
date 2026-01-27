# Claude Code Instructions

This repository contains **AI Analyst**, an AI-powered data analyst tool built with the Anthropic Claude API.

## Project Overview

AI Analyst is a Python-based tool that leverages Claude for intelligent data analysis. It provides:
- Interactive REPL mode for data exploration
- Automated data analysis and insights via agentic tool-use loop
- CSV, JSON, Excel, and Parquet file processing
- Statistical analysis, correlation detection, outlier detection, and trend analysis

**Project Metadata:**
- Version: 0.1.0
- License: MIT
- Python: 3.10+
- Status: Alpha

## Repository Structure

```
ai_analyst/
├── run.py                          # CLI entry point
├── analyst.py                      # Core standalone analyst (571 LOC)
├── interactive.py                  # REPL mode (107 LOC)
├── benchmark_quality.py            # Performance benchmarking
│
├── src/ai_analyst/                 # Package source code
│   ├── __init__.py
│   ├── cli.py                      # Click CLI commands (181 LOC)
│   ├── tools/
│   │   ├── __init__.py
│   │   └── statistical.py          # Statistical analysis functions
│   └── utils/
│       ├── __init__.py
│       └── config.py               # Settings and authentication
│
├── tests/                          # Pytest test suite
│   ├── conftest.py                 # Shared fixtures
│   ├── test_analyst.py             # StandaloneAnalyst tests
│   ├── test_config.py              # Config & path sanitization tests
│   ├── test_context.py             # AnalysisContext tests
│   ├── test_integration.py         # End-to-end tests
│   ├── test_interactive.py         # Interactive mode tests
│   └── test_statistical.py         # Statistical function tests
│
├── .github/                        # GitHub Actions & automation
│   ├── workflows/                  # CI/CD workflows
│   │   ├── pr-validation.yml
│   │   ├── claude.yml              # Claude Code assistant
│   │   ├── claude-review.yml       # Claude PR review
│   │   ├── google-ai.yml           # Gemini & Jules integration
│   │   ├── google-ai-review.yml    # Gemini PR review
│   │   ├── openai-codex.yml        # ChatGPT/Codex integration
│   │   ├── openai-review.yml       # OpenAI PR review
│   │   └── git-actions.yml
│   ├── actions/                    # Reusable GitHub Actions
│   │   ├── claude-agent/
│   │   ├── gemini-assistant/
│   │   ├── gemini-review/
│   │   ├── chatgpt-assistant/
│   │   ├── codex-agent/
│   │   └── jules-agent/
│   ├── scripts/                    # CI/CD helper scripts (Python)
│   │   ├── claude_pr_review.py
│   │   ├── claude_agent.py
│   │   ├── gemini_assistant.py
│   │   ├── gemini_pr_review.py
│   │   ├── chatgpt_assistant.py
│   │   ├── chatgpt_pr_review.py
│   │   └── codex_agent.py
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── copilot-instructions.md
│
├── .devcontainer/                  # Dev container configuration
│   └── devcontainer.json
│
├── data/                           # Sample data files
│   └── sample_sales.csv
│
├── pyproject.toml                  # Project configuration
├── README.md
└── CLAUDE.md                       # This file
```

## Core Components

### `analyst.py` - StandaloneAnalyst

The main data analysis engine using Claude's tool-use pattern.

**Key Classes:**
- `AnalysisContext`: Manages loaded datasets and state
- `StandaloneAnalyst`: Orchestrates analysis via agentic loop (max 15 iterations)

**Available Analysis Tools (10):**
| Tool | Description |
|------|-------------|
| `load_dataset` | Load CSV, JSON, Excel, Parquet files |
| `list_datasets` | List all loaded datasets |
| `preview_data` | Preview rows with column filtering |
| `describe_statistics` | Mean, std, min, max, quartiles |
| `compute_correlation` | Pearson/Spearman/Kendall correlation |
| `detect_outliers` | IQR or Z-score outlier detection |
| `group_analysis` | Grouped aggregation analysis |
| `check_data_quality` | Missing values, duplicates, quality score |
| `test_normality` | Shapiro-Wilk normality test |
| `analyze_trend` | Mann-Kendall trend detection |

### `src/ai_analyst/cli.py` - CLI Interface

Click-based CLI with Rich terminal formatting.

**Commands:**
```bash
ai-analyst analyze <file> -q "query"   # Single analysis
ai-analyst interactive [file]          # REPL session
ai-analyst inspect <file>              # Quick data inspection
ai-analyst auth                        # Check authentication status
```

### `interactive.py` - REPL Mode

Interactive analysis session with conversation history.

**REPL Commands:**
- `load <file>` - Load additional data file
- `clear` - Clear conversation history
- `help` - Show help
- `quit` - Exit session

### `src/ai_analyst/utils/config.py` - Configuration

- Authentication management (Pro subscription vs API key)
- Path sanitization for security
- Settings via Pydantic

### `src/ai_analyst/tools/statistical.py` - Statistical Tools

Statistical analysis functions:
- `compute_descriptive_stats(series)` - Basic statistics
- `test_normality(series, alpha)` - Shapiro-Wilk test
- `test_correlation_significance(x, y)` - Correlation analysis
- `detect_trend(values, alpha)` - Linear regression trend

## Development Guidelines

### Code Style
- Python 3.10+ features required
- PEP 8 compliant (enforced by Ruff)
- **Mandatory type hints** on all function signatures
- Line length: 100 characters
- Use specific exception handling (no bare `except:`)
- Prefer vectorized pandas operations over loops

### Linting & Type Checking
```bash
ruff check .                    # Linting
ruff format .                   # Formatting
mypy src/ai_analyst             # Type checking
```

### Dependencies

**Core (Required):**
- `anthropic >= 0.40.0` - Claude API client
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical operations
- `scipy >= 1.11.0` - Statistical functions
- `pydantic >= 2.0.0` - Data validation
- `rich >= 13.0.0` - Terminal UI
- `click >= 8.1.0` - CLI framework

**Optional Groups:**
- `[dev]` - pytest, mypy, ruff, pre-commit
- `[viz]` - matplotlib, seaborn, plotly
- `[ml]` - scikit-learn
- `[mcp]` - Model Context Protocol server mode
- `[notebook]` - Jupyter support

### Installation
```bash
pip install -e .              # Core only
pip install -e '.[dev]'       # With dev tools
pip install -e '.[dev,viz]'   # Multiple groups
```

## Authentication

AI Analyst supports two authentication methods:

### Option 1: Claude Pro Subscription (Recommended)
```bash
claude login
```
- Uses OAuth, stores credentials in `~/.claude/credentials.json` or `~/.config/claude/credentials.json`
- No API credits consumed

### Option 2: API Key (Fallback)
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Priority Control
```bash
export AUTH_PREFERENCE=pro    # Pro subscription first (default)
export AUTH_PREFERENCE=api    # API key first
```

### Check Status
```bash
python run.py auth
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key (fallback) | None |
| `AUTH_PREFERENCE` | Auth priority: "pro" or "api" | "pro" |
| `AI_ANALYST_MODEL` | Model to use | claude-sonnet-4-20250514 |
| `AI_ANALYST_LOG_LEVEL` | Logging verbosity | INFO |

## Testing

```bash
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest --cov=src/ai_analyst     # With coverage
pytest tests/test_analyst.py    # Specific file
```

**Test Files:**
- `test_analyst.py` - Core analyst initialization and tool execution
- `test_config.py` - Settings, path sanitization, environment config
- `test_context.py` - Dataset loading and retrieval
- `test_integration.py` - End-to-end analysis workflows
- `test_interactive.py` - REPL mode functionality
- `test_statistical.py` - Statistical function correctness

**Fixtures in `conftest.py`:**
- `analyst` - Analyst with mocked API client
- `analyst_with_data` - Analyst with loaded sample data
- `sample_csv_file` - 100-row CSV fixture
- `analysis_context`, `loaded_context` - Context fixtures

## PR Review Guidelines

When reviewing PRs:
1. **Error Handling**: Check for proper exception handling with specific types
2. **Security**: Verify API keys and sensitive data are not exposed
3. **Memory Efficiency**: Ensure pandas operations avoid unnecessary copies
4. **Type Safety**: Verify type hints are present and accurate
5. **Test Coverage**: Require tests for new functionality
6. **Path Safety**: Ensure file paths use `sanitize_path()` function

## Security Considerations

- **Path Traversal Prevention**: All file paths validated via `sanitize_path()` in `config.py`
- **API Key Protection**: Never hardcode; use environment variables only
- **Input Validation**: Pydantic models validate all settings
- **No Dynamic Code Execution**: No `eval()` or `exec()` on user data
- **Credential Masking**: API keys masked in logs and terminal output

## AI Assistant Integration

This repository integrates multiple AI assistants for development and review:

### Claude Code (Anthropic)
- `@claude` - General assistance and code review
- `@claude agent` - Automated code changes (owner/member only)
- **Claude Code Review** - Automatic PR reviews

### Google AI
- `@gemini` - Gemini AI Studio assistance
- `@jules` - Google Labs Jules coding agent
- **Gemini Code Assist** - Automatic PR reviews

### OpenAI ChatGPT/Codex
- `@chatgpt` - ChatGPT GPT-4o assistance
- `@codex` - Code-focused help
- `@codex-agent` - Automated code changes
- **ChatGPT Codex Review** - Automatic PR reviews

All PRs receive automated reviews from Claude, Gemini, and ChatGPT.

## Usage Examples

### Command Line
```bash
# Single analysis query
python run.py analyze data/sample_sales.csv -q "What are the sales trends?"

# Interactive REPL session
python run.py interactive data/sample_sales.csv

# Quick data inspection
python run.py inspect data/sample_sales.csv
```

### Python API
```python
from analyst import StandaloneAnalyst

analyst = StandaloneAnalyst()
response = analyst.analyze(
    "What correlations exist between price and sales?",
    file_path="data/sample_sales.csv"
)
print(response)
```

### Async API
```python
import asyncio
from analyst import StandaloneAnalyst

async def main():
    analyst = StandaloneAnalyst()
    response = await analyst.analyze_async(
        "Detect outliers in the revenue column",
        file_path="data/sample_sales.csv"
    )
    print(response)

asyncio.run(main())
```

## Data Format Support

| Format | Extension | Reader |
|--------|-----------|--------|
| CSV | `.csv` | `pd.read_csv()` |
| JSON | `.json` | `pd.read_json()` |
| Excel | `.xlsx`, `.xls` | `pd.read_excel()` |
| Parquet | `.parquet` | `pd.read_parquet()` |

## Architecture Flow

```
User Input (CLI/REPL/Python API)
         │
         ▼
┌─────────────────────────────────┐
│     StandaloneAnalyst           │
│  (Agentic Loop, max 15 turns)   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Claude API (Tool Use)      │
│  - Receives query + tools       │
│  - Returns tool calls or text   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Tool Execution            │
│  - AnalysisContext (datasets)   │
│  - Statistical functions        │
│  - Pandas operations            │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Response to User           │
│  (Rich terminal / Return value) │
└─────────────────────────────────┘
```

## Dev Container

The repository includes a dev container configuration (`.devcontainer/devcontainer.json`):
- Python 3.10 base image
- Pre-installed extensions: Python, Ruff, Jupyter, GitHub Copilot
- Jupyter port 8888 forwarded
- Auto-installs all optional dependencies
