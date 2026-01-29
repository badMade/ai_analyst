#!/usr/bin/env python3
"""
Standalone runner for AI Analyst.

From the project root (source checkout), with your virtual environment
activated, you can run:
activated and dependencies installed (e.g., via 'pip install -e .'), you can run:
    python run.py analyze data.csv
    python run.py interactive
    python run.py inspect data.csv
"""

from ai_analyst.cli import main

if __name__ == "__main__":
    main()
