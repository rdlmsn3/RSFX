"""
detectors/strategies/categories.py
-----------------------------------
Strategy category definitions for UI grouping.

Categories organize strategies by their primary approach:
- Single TF: Strategies using only one timeframe
- Two TF: Strategies using H1 + M5
- Price Action: Candlestick pattern-based strategies
- Volume Based: Volume indicator strategies
- Channel Based: Channel breakout/bounce strategies
- etc.
"""

from __future__ import annotations

# Category display order (UI rendering order)
CATEGORY_ORDER: list[str] = [
    "Single TF - Existing",
    "Two TF - Existing",
    "Single TF - EMA/RSI",
    "Single TF - Trend Following",
    "Two TF - H1+M5",
    "Price Action",
    "Volume Based",
    "Channel Based",
    "Momentum + Mean Reversion",
    "Multi-Indicator Confluence",
    "Session/Pivot",
    "Trend Following Scalps",
    "Divergence",
    "Hybrid",
    "Advanced",
]

# Category descriptions (for UI tooltips/help text)
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Single TF - Existing": "Original strategies (M1 only)",
    "Two TF - Existing": "Original multi-timeframe strategies",
    "Single TF - EMA/RSI": "EMA/RSI-based strategies on M5",
    "Single TF - Trend Following": "Trend-following strategies on M5",
    "Two TF - H1+M5": "H1 trend + M5 entry strategies",
    "Price Action": "Candlestick pattern strategies",
    "Volume Based": "Volume indicator strategies",
    "Channel Based": "Channel breakout/bounce strategies",
    "Momentum + Mean Reversion": "Momentum and mean reversion combos",
    "Multi-Indicator Confluence": "3+ indicator confirmation strategies",
    "Session/Pivot": "Time-based and pivot point strategies",
    "Trend Following Scalps": "Scalping strategies for trending markets",
    "Divergence": "Divergence-based reversal strategies",
    "Hybrid": "Combined approach strategies",
    "Advanced": "Complex multi-indicator strategies",
}
