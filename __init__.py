"""
Sigma Dollar Bars Package

This package provides functionality for creating dollar bars from OHLCV data.
Dollar bars are created by aggregating price-volume data until a specified dollar value is reached.
"""

from .dollar_bars import (
    generate_dollar_bars,
    data_profiler,
    describe_dollar_bars,
)

__version__ = "0.1.0"

__all__ = [
    "generate_dollar_bars",
    "data_profiler",
    "describe_dollar_bars",
]
