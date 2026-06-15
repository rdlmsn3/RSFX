"""
detectors/signal.py
-------------------
Shared signal dataclass used by all strategies and the renderer.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PatternSignal:
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    confidence: float
    metadata: dict = field(default_factory=dict)
