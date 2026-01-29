#!/usr/bin/env python3
"""
Standalone runner for AI Analyst.

After installing the package (e.g., `pip install -e .`), you can run:
    ai-analyst analyze data.csv
    ai-analyst interactive
    ai-analyst inspect data.csv

From a source checkout, you can also run:
    python run.py analyze data.csv
    python run.py interactive
    python run.py inspect data.csv
"""

from ai_analyst.cli import main

if __name__ == "__main__":
    main()
