"""
detectors/strategies/__init__.py
--------------------------------
Pluggable strategy interface for pattern detection.

Exports:
  - BaseStrategy (abstract base class)
  - EMAStochasticStrategy (M1 only)
  - EMAStochasticMTFStrategy (H1+M5+M1)
  - STRATEGY_REGISTRY (central catalog)
  - CATEGORY_ORDER, CATEGORY_DESCRIPTIONS
"""

from .base import BaseStrategy
from .ema_stochastic import EMAStochasticStrategy
from .ema_stochastic_mtf import EMAStochasticMTFStrategy
from .registry import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    get_strategy_params,
    get_strategies_by_category,
    get_all_categories,
)
from .categories import CATEGORY_ORDER, CATEGORY_DESCRIPTIONS

__all__ = [
    "BaseStrategy",
    "EMAStochasticStrategy",
    "EMAStochasticMTFStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "get_strategy_params",
    "get_strategies_by_category",
    "get_all_categories",
    "CATEGORY_ORDER",
    "CATEGORY_DESCRIPTIONS",
]
