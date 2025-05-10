"""
Dollar Bars - A Python package for creating dollar bars from OHLCV data.
"""

from .dollar_bars import (
    generate_dollar_bars,
    data_profiler,
    describe_dollar_bars,
    _simplify_number,
)

__version__ = "0.1.0"
__all__ = [
    "generate_dollar_bars",
    "data_profiler",
    "describe_dollar_bars",
    "_simplify_number",
]
