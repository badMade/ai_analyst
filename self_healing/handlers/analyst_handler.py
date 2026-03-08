"""
Analyst Handler for AI Analyst Self-Healing System.

Handles data analysis-specific errors including pandas, numpy, API calls,
and statistical operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from self_healing.models.errors import DetectedError, ErrorType, EnvironmentType, ErrorSeverity
from self_healing.models.fixes import Fix, FixStrategy

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class AnalystHandler:
    """
    Handles data analysis-specific errors.

    Specialized fixes for:
    - Data loading (CSV, JSON, Parquet)
    - API rate limits (Claude, other services)
    - Memory optimization
    - Statistical computation errors
    """

    def __init__(self) -> None:
        self._encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        self._chunk_sizes = [1000, 5000, 10000, 50000]

    # Data Loading Fixes

    def fix_data_loading(
        self,
        error: DetectedError,
        file_path: Path,
    ) -> Fix:
        """Generate fix for data loading errors."""
        strategies = []

        # Suggest encoding fixes
        if "encoding" in error.message.lower() or "codec" in error.message.lower():
            strategies.append({
                "encoding": "latin1",
                "description": "Try Latin-1 encoding",
            })

        # Suggest parser options
        if "tokenizing" in error.message.lower():
            strategies.append({
                "on_bad_lines": "skip",
                "description": "Skip malformed lines",
            })

        # Suggest engine change
        strategies.append({
            "engine": "python",
            "description": "Use Python parser for better error handling",
        })

        return Fix(
            error=error,
            strategy=FixStrategy.DATA_RELOAD,
            description="Retry data loading with alternative options",
            reasoning=f"Data loading failed: {error.message}",
            config_changes={
                "strategies": strategies,
                "file_path": str(file_path),
            },
            confidence=0.80,
        )

    def get_encoding_fix_code(self, file_path: str) -> str:
        """Generate code to read file with encoding detection."""
        return f'''
# Auto-generated encoding fix
import pandas as pd

def read_with_encoding_detection(path: str) -> pd.DataFrame:
    """Try multiple encodings until one works."""
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # Last resort: read with error replacement
    return pd.read_csv(path, encoding="utf-8", errors="replace")

df = read_with_encoding_detection("{file_path}")
'''

    def get_chunked_read_code(self, file_path: str, chunksize: int = 10000) -> str:
        """Generate code for chunked file reading."""
        return f'''
# Auto-generated chunked read
import pandas as pd
from typing import Iterator

def read_in_chunks(path: str, chunksize: int = {chunksize}) -> pd.DataFrame:
    """Read large file in chunks to avoid memory errors."""
    chunks = []
    
    for chunk in pd.read_csv(path, chunksize=chunksize):
        # Process chunk here if needed
        chunks.append(chunk)
    
    return pd.concat(chunks, ignore_index=True)

df = read_in_chunks("{file_path}")
'''

    # API Error Fixes

    def fix_api_rate_limit(self, error: DetectedError) -> Fix:
        """Generate fix for API rate limit errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.RETRY_WITH_BACKOFF,
            description="Implement exponential backoff for API calls",
            reasoning="API rate limit exceeded",
            config_changes={
                "max_retries": 5,
                "initial_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 2.0,
            },
            confidence=0.90,
        )

    def get_backoff_wrapper_code(self) -> str:
        """Generate code for API retry with backoff."""
        return '''
# Auto-generated backoff wrapper
import time
import random
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar("T")

def with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for exponential backoff retry."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            jitter = random.uniform(0, 0.1 * delay)
                            time.sleep(delay + jitter)
                            delay = min(delay * exponential_base, max_delay)
                            continue
                    raise
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# Usage:
# @with_backoff(max_retries=5)
# async def call_claude_api(...):
#     ...
'''

    # Memory Optimization Fixes

    def fix_memory_error(self, error: DetectedError) -> Fix:
        """Generate fix for memory errors."""
        size_mb = error.context.get("size_mb", 0)

        return Fix(
            error=error,
            strategy=FixStrategy.MEMORY_OPTIMIZE,
            description="Optimize memory usage for large data",
            reasoning=f"Memory error processing {size_mb:.0f}MB of data",
            config_changes={
                "use_chunks": True,
                "chunksize": 10000,
                "dtype_optimization": True,
                "gc_collect": True,
            },
            confidence=0.75,
        )

    def get_memory_optimization_code(self) -> str:
        """Generate code for memory optimization."""
        return '''
# Auto-generated memory optimization
import gc
import pandas as pd
import numpy as np

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory by optimizing data types."""
    
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    
    gc.collect()
    return df

def process_large_file(path: str, processor: callable) -> list:
    """Process large file in chunks with memory management."""
    results = []
    
    for chunk in pd.read_csv(path, chunksize=10000):
        chunk = optimize_dtypes(chunk)
        result = processor(chunk)
        results.append(result)
        gc.collect()
    
    return results
'''

    # Statistical Error Fixes

    def fix_statistical_error(self, error: DetectedError) -> Fix:
        """Generate fix for statistical computation errors."""
        return Fix(
            error=error,
            strategy=FixStrategy.CODE_PATCH,
            description="Add error handling for statistical operations",
            reasoning=f"Statistical computation failed: {error.message}",
            config_changes={
                "handle_nan": True,
                "handle_inf": True,
                "min_samples": 2,
            },
            confidence=0.70,
        )

    def get_safe_stats_code(self) -> str:
        """Generate code for safe statistical operations."""
        return '''
# Auto-generated safe statistics
import numpy as np
import pandas as pd
from typing import Optional

def safe_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str = "pearson",
) -> Optional[float]:
    """Calculate correlation with error handling."""
    
    # Remove NaN and Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return None
    
    try:
        return x_clean.corr(y_clean, method=method)
    except Exception:
        return None

def safe_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics with error handling."""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {}
    for col in numeric_cols:
        clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        stats[col] = {
            "count": len(clean),
            "mean": clean.mean() if len(clean) > 0 else np.nan,
            "std": clean.std() if len(clean) > 1 else np.nan,
            "min": clean.min() if len(clean) > 0 else np.nan,
            "max": clean.max() if len(clean) > 0 else np.nan,
        }
    
    return pd.DataFrame(stats).T
'''

    # Decorator for protected execution

    def retry_with_backoff(
        self,
        config: Optional[RetryConfig] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for retry with exponential backoff."""
        config = config or RetryConfig()

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                delay = config.initial_delay
                last_exception = None

                for attempt in range(config.max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e).lower()

                        # Check if retryable
                        if not any(
                            x in error_msg
                            for x in ["rate", "429", "timeout", "connection"]
                        ):
                            raise

                        if attempt < config.max_retries - 1:
                            if config.jitter:
                                import random
                                jitter = random.uniform(0, 0.1 * delay)
                                delay += jitter

                            time.sleep(delay)
                            delay = min(delay * config.exponential_base, config.max_delay)

                if last_exception:
                    raise last_exception
                return func(*args, **kwargs)

            return wrapper

        return decorator
