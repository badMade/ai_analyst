# AI Analyst

Standalone Python AI-powered data analyst using Claude API with agentic tool use.

## Quick Start

```bash
# 1. Clone/copy the project
cd ai-analyst

# 2. Install minimal dependencies
pip install -e .

# 3. Set API key
export ANTHROPIC_API_KEY="your-key"

# 4. Run analysis
python run.py analyze data/sample_sales.csv -q "What are the sales trends by region?"
```

**Or without installation:**
```bash
pip install anthropic pandas numpy scipy pydantic pydantic-settings rich click openpyxl pyarrow
python run.py analyze your_data.csv
```

## Commands

```bash
# Single analysis
python run.py analyze data.csv -q "Summarize key statistics"

# Interactive REPL session
python run.py interactive data.csv

# Quick data inspection
python run.py inspect data.csv

# Show help
python run.py --help
```

## Python API

```python
from ai_analyst.analyst import StandaloneAnalyst

analyst = StandaloneAnalyst()

# Load and analyze
response = analyst.analyze(
    "What are the correlations between price and sales?",
    file_path="sales_data.csv"
)
print(response)
```

## Available Analysis Tools

| Tool | Description |
|------|-------------|
| `load_dataset` | Load CSV, JSON, Excel, Parquet |
| `preview_data` | Preview rows |
| `describe_statistics` | Mean, std, quartiles |
| `compute_correlation` | Correlation matrix |
| `detect_outliers` | IQR/Z-score outliers |
| `group_analysis` | Grouped aggregations |
| `check_data_quality` | Missing, duplicates, issues |
| `test_normality` | Shapiro-Wilk test |
| `analyze_trend` | Mann-Kendall trend detection |

## Configuration

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `AI_ANALYST_MODEL` | Model (default: claude-sonnet-4-20250514) |
| `AI_ANALYST_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR |

## Project Structure

```
ai-analyst/
├── run.py                  # Standalone runner
├── src/ai_analyst/
│   ├── analyst.py          # Core standalone analyst
│   ├── cli.py              # Click CLI
│   ├── interactive.py      # REPL mode
│   ├── tools/statistical.py
│   └── utils/config.py
├── data/sample_sales.csv
└── pyproject.toml
```

## Integration with Mothership

Later, to integrate into mothership:

```bash
# Copy to mothership
cp -r ai-analyst ~/mothership/apps/

# Install as editable
pip install -e ~/mothership/apps/ai-analyst

# Import in your code
from ai_analyst.analyst import StandaloneAnalyst
```

**Optional MCP mode** (for server integration):
```bash
pip install -e ".[mcp]"
python -m ai_analyst.mcp_servers.data_analyst
```

## License

MIT
