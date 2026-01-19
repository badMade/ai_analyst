#!/usr/bin/env python3
"""
Standalone runner for AI Analyst.

Run without installation:
    python run.py analyze data.csv
    python run.py interactive
    python run.py inspect data.csv
"""

import os
import sys

# Add src to path for running without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ai_analyst.cli import main

if __name__ == "__main__":
    main()
