#!/usr/bin/env python3
"""
Standalone runner for AI Analyst.

Run after installing the package (e.g., pip install -e .):
    python run.py analyze data.csv
    python run.py interactive
    python run.py inspect data.csv
"""

from ai_analyst.cli import main

if __name__ == "__main__":
    main()
