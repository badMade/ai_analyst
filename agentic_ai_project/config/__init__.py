"""Configuration module for Agentic AI Project.

This module provides configuration management for agents, models,
environments, and logging.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent

__all__ = ["CONFIG_DIR"]
