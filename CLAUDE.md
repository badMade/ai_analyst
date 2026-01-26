# Claude Code Instructions

This repository contains an AI-powered data analyst tool built with the Anthropic Claude API.

## Project Overview

AI Analyst is a Python-based tool that leverages Claude for intelligent data analysis. It provides:
- Interactive REPL mode for data exploration
- Automated data analysis and insights
- CSV/DataFrame processing capabilities

## Repository Structure

ai-analyst/
├── run.py                  # CLI entry point
├── analyst.py              # Core standalone analyst
├── interactive.py          # REPL mode
├── src/ai_analyst/
│   ├── cli.py              # Click CLI commands
│   ├── tools/statistical.py
│   └── utils/config.py     # Settings and authentication
├── data/sample_sales.csv
└── pyproject.toml

## Development Guidelines

### Code Style
- Use Python 3.10+ features
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Keep functions focused and well-documented

### Dependencies
- Core: `anthropic`, `pandas`, `numpy`, `scipy`
- CLI: `click`, `rich`, `pydantic`
- Optional: `matplotlib`, `scikit-learn` for visualization and ML

### Authentication

AI Analyst supports two authentication methods, with Claude Pro subscription as the primary option:

#### Option 1: Claude Pro Subscription (Recommended)
Use your existing Claude Pro/Max subscription:
```bash
claude login
```
This authenticates via OAuth and stores credentials locally. No API credits needed!

#### Option 2: API Key (Fallback)
Set the API key environment variable:
```bash
export ANTHROPIC_API_KEY='your-api-key'
```

#### Authentication Priority
By default, Pro subscription is checked first. To change this:
```bash
export AUTH_PREFERENCE=api  # Use API key first
export AUTH_PREFERENCE=pro  # Use Pro subscription first (default)
```

Check your authentication status:
```bash
python run.py auth
```

### Environment Variables
- `ANTHROPIC_API_KEY` - API key (fallback if Pro subscription not available)
- `AUTH_PREFERENCE` - Authentication priority: "pro" (default) or "api"
- `AI_ANALYST_MODEL` - Model to use (default: claude-sonnet-4-20250514)
- `AI_ANALYST_LOG_LEVEL` - Logging verbosity

### Testing
- Run tests with `pytest`
- Ensure code coverage for new features
- Test both CLI and programmatic interfaces

## Key Files

- **analyst.py**: Contains the main `AIAnalyst` class that handles data loading, Claude API interactions, and analysis generation
- **interactive.py**: Implements the REPL interface for interactive data exploration
- **run.py**: CLI entry point using Click framework

## PR Review Guidelines

When reviewing PRs:
1. Check for proper error handling
2. Verify API key and sensitive data are not exposed
3. Ensure pandas operations are memory-efficient
4. Validate Claude API usage follows best practices

## AI Integration

This repository integrates multiple AI assistants:

### Claude Code (Anthropic)
- `@claude` - General assistance and code review
- `@claude agent` - Automated code changes

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
