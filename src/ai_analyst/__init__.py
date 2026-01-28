"""
AI Analyst - AI-powered data analysis tool using Claude API.

This package provides data analysis capabilities with agentic tool use.
"""

from ai_analyst.analyst import (
    AnalysisContext,
    StandaloneAnalyst,
    create_analyst,
    TOOLS,
)
from ai_analyst.interactive import run_interactive

__version__ = "0.1.0"

__all__ = [
    "AnalysisContext",
    "StandaloneAnalyst",
    "create_analyst",
    "run_interactive",
    "TOOLS",
]
