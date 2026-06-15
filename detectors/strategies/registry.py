"""
detectors/strategies/registry.py
---------------------------------
Central registry mapping strategy names → classes + metadata.

Each entry contains:
  - class: The strategy class (subclass of BaseStrategy)
  - category: UI grouping category
  - timeframes: Required timeframe labels
  - description: Human-readable description
  - params: Dict of parameter_name → {type, default, min, max}
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Registry: populated lazily on first access
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {}


def _register(
    name: str,
    cls: type,
    category: str,
    timeframes: list[str],
    description: str,
    params: dict[str, dict[str, Any]],
) -> None:
    """Register a strategy in the global registry."""
    STRATEGY_REGISTRY[name] = {
        "class": cls,
        "category": category,
        "timeframes": timeframes,
        "description": description,
        "params": params,
    }


def get_strategy_class(name: str) -> type:
    """Get strategy class by name. Raises KeyError if not found."""
    return STRATEGY_REGISTRY[name]["class"]


def get_strategy_params(name: str) -> dict[str, dict[str, Any]]:
    """Get strategy parameter definitions by name."""
    return STRATEGY_REGISTRY[name]["params"]


def get_strategies_by_category(category: str) -> list[str]:
    """Get all strategy names in a category."""
    return [
        name
        for name, info in STRATEGY_REGISTRY.items()
        if info["category"] == category
    ]


def get_all_categories() -> list[str]:
    """Get all unique categories."""
    return list(dict.fromkeys(
        info["category"] for info in STRATEGY_REGISTRY.values()
    ))


# ---------------------------------------------------------------------------
# Import all strategy classes to trigger registration
# ---------------------------------------------------------------------------
# This is done at module level so that importing registry.py
# automatically registers all strategies.

def _populate_registry() -> None:
    """Import all strategy modules and register their classes."""
    from .ema_stochastic import EMAStochasticStrategy
    from .ema_stochastic_mtf import EMAStochasticMTFStrategy

    # Group 1: Single TF - EMA/RSI Based
    from .ema_rsi_cross import EmaRsiCrossStrategy
    from .ema_stoch_cross import EmaStochCrossStrategy
    from .bb_rsi_bounce import BbRsiBounceStrategy
    from .bb_squeeze_breakout import BbSqueezeBreakoutStrategy
    from .macd_ema_trend import MacdEmaTrendStrategy
    from .macd_histogram_div import MACDHistogramDivStrategy
    from .rsi_ema_trend import RSIEMATrendStrategy
    from .stoch_ema_trend import StochEMATrendStrategy
    from .cci_ema import CCIERMATrendStrategy
    from .williams_ema import WilliamsEMAStrategy

    # Group 2: Single TF - Trend Following
    from .supertrend_ema import SupertrendEmaStrategy
    from .parabolic_sar_ema import ParabolicSarEmaStrategy
    from .adx_di_ema import AdxDiEmaStrategy
    from .keltner_breakout import KeltnerBreakoutStrategy
    from .donchian_breakout import DonchianBreakoutStrategy
    from .heikin_ashi_ema import HeikinAshiEMAStrategy
    from .ma_ribbon_pullback import MARibbonPullbackStrategy
    from .ma_envelope_bounce import MAEnvelopeBounceStrategy
    from .rsi_divergence_ema import RSIDivergenceEMAStrategy
    from .stoch_divergence_ema import StochDivergenceEMAStrategy

    # Group 3: Two TF (H1+M5)
    from .h1_trend_m5_ema_cross import H1TrendM5EmaCrossStrategy
    from .h1_trend_m5_rsi import H1TrendM5RsiStrategy
    from .h1_trend_m5_stoch import H1TrendM5StochStrategy
    from .h1_trend_m5_macd import H1TrendM5MacdStrategy
    from .h1_trend_m5_bb import H1TrendM5BbStrategy
    from .h1_adx_m5_ema import H1AdxM5EmaStrategy

    # Group 4: Price Action
    from .pin_bar_ema import PinBarEmaStrategy
    from .engulfing_ema import EngulfingEmaStrategy
    from .inside_bar_breakout import InsideBarBreakoutStrategy
    from .three_bar_reversal import ThreeBarReversalStrategy
    from .morning_evening_star import MorningEveningStarStrategy
    from .harami_trend import HaramiTrendStrategy
    from .tweezer_reversal import TweezerReversalStrategy
    from .marubozu_trend import MarubozuTrendStrategy

    # Group 5: Volume Based
    from .volume_spike_ema import VolumeSpikeEMAStrategy
    from .volume_profile_ema import VolumeProfileEMAStrategy
    from .obv_ema import OBVEMAStrategy
    from .ad_ema import ADEMStrategy

    # Group 6: Channel Based
    from .donchian_rsi import DonchianRsiStrategy
    from .keltner_rsi import KeltnerRsiStrategy
    from .atr_channel_breakout import AtrChannelBreakoutStrategy

    # Group 7: Momentum + Mean Reversion
    from .rsi_bb_squeeze import RsiBbSqueezeStrategy
    from .stoch_bb_bounce import StochBbBounceStrategy
    from .macd_bb_breakout import MacdBbBreakoutStrategy

    # Group 8: Multi-Indicator Confluence
    from .triple_confirm import TripleConfirmStrategy
    from .trend_momentum_vol import TrendMomentumVolStrategy
    from .adx_rsi_ema import AdxRsiEmaStrategy

    # Group 9: Session/Pivot
    from .london_ny_breakout import LondonNyBreakoutStrategy
    from .asian_range_breakout import AsianRangeBreakoutStrategy
    from .pivot_ema_bounce import PivotEmaBounceStrategy
    from .pivot_rsi_bounce import PivotRsiBounceStrategy
    from .fib_ema_bounce import FibEmaBounceStrategy
    from .fib_rsi_bounce import FibRsiBounceStrategy

    # Group 10: Trend Following Scalps
    from .ema_ribbon_pullback import EmaRibbonPullbackStrategy
    from .vwap_ema_cross import VwapEmaCrossStrategy
    from .vwap_bounce import VwapBounceStrategy
    from .ichimoku_cloud_bounce import IchimokuCloudBounceStrategy
    from .ichimoku_cloud_break import IchimokuCloudBreakStrategy

    # Group 11: Divergence
    from .rsi_divergence import RSIDivergenceStrategy
    from .macd_divergence import MACDDivergenceStrategy
    from .stoch_divergence import StochDivergenceStrategy
    from .volume_divergence import VolumeDivergenceStrategy

    # Group 12: Hybrid
    from .trend_mean_reversion import TrendMeanReversionStrategy
    from .breakout_retest import BreakoutRetestStrategy
    from .momentum_exhaustion import MomentumExhaustionStrategy
    from .scalp_pullback import ScalpPullbackStrategy
    from .gap_fill import GapFillStrategy

    # Group 13: Advanced
    from .ema_macd_rsi_confluence import EmaMacdRsiConfluenceStrategy
    from .bb_stoch_volume import BbStochVolumeStrategy
    from .supertrend_rsi_ema import SupertrendRsiEmaStrategy

    # =========================================================================
    # Register all strategies
    # =========================================================================

    # --- Existing strategies ---
    _register(
        name="ema_stochastic",
        cls=EMAStochasticStrategy,
        category="Single TF - Existing",
        timeframes=["M1"],
        description="EMA crossover + Stochastic + Candlestick (original)",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "stoch_k": {"type": "int", "default": 5, "min": 3, "max": 20},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "oversold": {"type": "float", "default": 20.0, "min": 5, "max": 40},
            "overbought": {"type": "float", "default": 80.0, "min": 60, "max": 95},
        },
    )
    _register(
        name="ema_stochastic_mtf",
        cls=EMAStochasticMTFStrategy,
        category="Two TF - Existing",
        timeframes=["M1", "M5", "H1"],
        description="3-layer MTF: H1 trend + M5 momentum + M1 entry (original)",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_stoch_k": {"type": "int", "default": 5, "min": 3, "max": 20},
            "m5_stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "m5_oversold": {"type": "float", "default": 20.0, "min": 5, "max": 40},
            "m5_overbought": {"type": "float", "default": 80.0, "min": 60, "max": 95},
        },
    )

    # =========================================================================
    # Group 1: Single TF - EMA/RSI Based
    # =========================================================================
    _register(
        name="ema_rsi_cross",
        cls=EmaRsiCrossStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="EMA 9/21 crossover + RSI 14 confirmation",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long_threshold": {"type": "float", "default": 40.0, "min": 20, "max": 50},
            "rsi_short_threshold": {"type": "float", "default": 60.0, "min": 50, "max": 80},
            "atr_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "sl_atr_mult": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0},
            "tp_atr_mult": {"type": "float", "default": 2.0, "min": 1.0, "max": 10.0},
        },
    )
    _register(
        name="ema_stoch_cross",
        cls=EmaStochCrossStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="EMA 9/21 + Stochastic 5,3,3 crossover",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "stoch_k": {"type": "int", "default": 5, "min": 3, "max": 20},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "oversold": {"type": "float", "default": 20.0, "min": 5, "max": 40},
            "overbought": {"type": "float", "default": 80.0, "min": 60, "max": 95},
        },
    )
    _register(
        name="bb_rsi_bounce",
        cls=BbRsiBounceStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="Bollinger Band bounce + RSI confirmation",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long_threshold": {"type": "float", "default": 35.0, "min": 20, "max": 50},
            "rsi_short_threshold": {"type": "float", "default": 65.0, "min": 50, "max": 80},
        },
    )
    _register(
        name="bb_squeeze_breakout",
        cls=BbSqueezeBreakoutStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="Bollinger squeeze breakout with volume",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "squeeze_lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "volume_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        },
    )
    _register(
        name="macd_ema_trend",
        cls=MacdEmaTrendStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="MACD cross + EMA 50 trend filter",
        params={
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="macd_histogram_div",
        cls=MACDHistogramDivStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="MACD histogram divergence from price",
        params={
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )
    _register(
        name="rsi_ema_trend",
        cls=RSIEMATrendStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="RSI bounce + EMA 50 trend filter",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long_bounce": {"type": "float", "default": 40.0, "min": 25, "max": 50},
            "rsi_short_bounce": {"type": "float", "default": 60.0, "min": 50, "max": 75},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="stoch_ema_trend",
        cls=StochEMATrendStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="Stochastic cross + EMA 9/21 trend",
        params={
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="cci_ema",
        cls=CCIERMATrendStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="CCI crossover + EMA 50 trend",
        params={
            "cci_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="williams_ema",
        cls=WilliamsEMAStrategy,
        category="Single TF - EMA/RSI",
        timeframes=["M5"],
        description="Williams %R crossover + EMA 9/21 trend",
        params={
            "williams_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )

    # =========================================================================
    # Group 2: Single TF - Trend Following
    # =========================================================================
    _register(
        name="supertrend_ema",
        cls=SupertrendEmaStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Supertrend flip + EMA 50 trend",
        params={
            "atr_period": {"type": "int", "default": 10, "min": 5, "max": 30},
            "atr_mult": {"type": "float", "default": 3.0, "min": 1.0, "max": 5.0},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="parabolic_sar_ema",
        cls=ParabolicSarEmaStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Parabolic SAR flip + EMA 9/21",
        params={
            "sar_af": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.1},
            "sar_max": {"type": "float", "default": 0.2, "min": 0.1, "max": 0.5},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="adx_di_ema",
        cls=AdxDiEmaStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="ADX trend strength + DI cross + EMA 50",
        params={
            "adx_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "adx_threshold": {"type": "float", "default": 25.0, "min": 15, "max": 40},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="keltner_breakout",
        cls=KeltnerBreakoutStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Keltner channel breakout with volume",
        params={
            "kc_ema": {"type": "int", "default": 20, "min": 10, "max": 50},
            "atr_period": {"type": "int", "default": 10, "min": 5, "max": 30},
            "atr_mult": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
            "volume_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        },
    )
    _register(
        name="donchian_breakout",
        cls=DonchianBreakoutStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Donchian channel breakout with volume",
        params={
            "donchian_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "consolidation_bars": {"type": "int", "default": 5, "min": 3, "max": 15},
            "volume_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        },
    )
    _register(
        name="heikin_ashi_ema",
        cls=HeikinAshiEMAStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Heikin Ashi color change + EMA 9/21",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="ma_ribbon_pullback",
        cls=MARibbonPullbackStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="EMA ribbon alignment + pullback to EMA 13",
        params={
            "ema_1": {"type": "int", "default": 5, "min": 3, "max": 10},
            "ema_2": {"type": "int", "default": 8, "min": 5, "max": 15},
            "ema_3": {"type": "int", "default": 13, "min": 10, "max": 20},
            "ema_4": {"type": "int", "default": 21, "min": 15, "max": 30},
        },
    )
    _register(
        name="ma_envelope_bounce",
        cls=MAEnvelopeBounceStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="MA envelope bounce + RSI",
        params={
            "ema_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "envelope_pct": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="rsi_divergence_ema",
        cls=RSIDivergenceEMAStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="RSI divergence + EMA 9/21 trend",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="stoch_divergence_ema",
        cls=StochDivergenceEMAStrategy,
        category="Single TF - Trend Following",
        timeframes=["M5"],
        description="Stochastic divergence + EMA 9/21 trend",
        params={
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )

    # =========================================================================
    # Group 3: Two TF (H1+M5)
    # =========================================================================
    _register(
        name="h1_trend_m5_ema_cross",
        cls=H1TrendM5EmaCrossStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 EMA trend + M5 EMA crossover entry",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "m5_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="h1_trend_m5_rsi",
        cls=H1TrendM5RsiStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 EMA trend + M5 RSI bounce",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "m5_rsi_long": {"type": "float", "default": 40.0, "min": 25, "max": 50},
            "m5_rsi_short": {"type": "float", "default": 60.0, "min": 50, "max": 75},
        },
    )
    _register(
        name="h1_trend_m5_stoch",
        cls=H1TrendM5StochStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 EMA trend + M5 Stochastic cross",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "m5_stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
        },
    )
    _register(
        name="h1_trend_m5_macd",
        cls=H1TrendM5MacdStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 EMA trend + M5 MACD cross",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "m5_macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "m5_macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
        },
    )
    _register(
        name="h1_trend_m5_bb",
        cls=H1TrendM5BbStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 EMA trend + M5 Bollinger bounce",
        params={
            "h1_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "h1_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "m5_bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "m5_bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "m5_rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "m5_rsi_threshold": {"type": "float", "default": 35.0, "min": 20, "max": 50},
        },
    )
    _register(
        name="h1_adx_m5_ema",
        cls=H1AdxM5EmaStrategy,
        category="Two TF - H1+M5",
        timeframes=["M5", "H1"],
        description="H1 ADX trend strength + M5 EMA cross",
        params={
            "h1_adx_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "h1_adx_threshold": {"type": "float", "default": 25.0, "min": 15, "max": 40},
            "m5_ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "m5_ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )

    # =========================================================================
    # Group 4: Price Action
    # =========================================================================
    _register(
        name="pin_bar_ema",
        cls=PinBarEmaStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Pin bar pattern + EMA 50 trend",
        params={
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
            "wick_ratio": {"type": "float", "default": 2.0, "min": 1.5, "max": 3.0},
        },
    )
    _register(
        name="engulfing_ema",
        cls=EngulfingEmaStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Engulfing pattern + EMA 9/21 trend",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="inside_bar_breakout",
        cls=InsideBarBreakoutStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Inside bar breakout + EMA trend",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="three_bar_reversal",
        cls=ThreeBarReversalStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Three consecutive bars pattern + RSI",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="morning_evening_star",
        cls=MorningEveningStarStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Morning/Evening star pattern + EMA 50",
        params={
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="harami_trend",
        cls=HaramiTrendStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Harami pattern + EMA 9/21 + RSI",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="tweezer_reversal",
        cls=TweezerReversalStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Tweezer top/bottom + EMA 9/21",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "tolerance_pct": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.05},
        },
    )
    _register(
        name="marubozu_trend",
        cls=MarubozuTrendStrategy,
        category="Price Action",
        timeframes=["M5"],
        description="Marubozu candle + EMA 9/21",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "body_ratio": {"type": "float", "default": 0.8, "min": 0.6, "max": 0.95},
        },
    )

    # =========================================================================
    # Group 5: Volume Based
    # =========================================================================
    _register(
        name="volume_spike_ema",
        cls=VolumeSpikeEMAStrategy,
        category="Volume Based",
        timeframes=["M5"],
        description="Volume spike + bullish/bearish close + EMA trend",
        params={
            "volume_mult": {"type": "float", "default": 2.0, "min": 1.2, "max": 4.0},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="volume_profile_ema",
        cls=VolumeProfileEMAStrategy,
        category="Volume Based",
        timeframes=["M5"],
        description="Price bounce from high-volume node + EMA trend",
        params={
            "volume_lookback": {"type": "int", "default": 50, "min": 20, "max": 100},
            "node_threshold": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="obv_ema",
        cls=OBVEMAStrategy,
        category="Volume Based",
        timeframes=["M5"],
        description="OBV crosses its EMA + price trend",
        params={
            "obv_ema_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )
    _register(
        name="ad_ema",
        cls=ADEMStrategy,
        category="Volume Based",
        timeframes=["M5"],
        description="A/D line crosses its EMA + price trend",
        params={
            "ad_ema_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "ema_trend": {"type": "int", "default": 50, "min": 20, "max": 200},
        },
    )

    # =========================================================================
    # Group 6: Channel Based
    # =========================================================================
    _register(
        name="donchian_rsi",
        cls=DonchianRsiStrategy,
        category="Channel Based",
        timeframes=["M5"],
        description="Donchian breakout + RSI confirmation",
        params={
            "donchian_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="keltner_rsi",
        cls=KeltnerRsiStrategy,
        category="Channel Based",
        timeframes=["M5"],
        description="Keltner breakout + RSI confirmation",
        params={
            "kc_ema": {"type": "int", "default": 20, "min": 10, "max": 50},
            "atr_period": {"type": "int", "default": 10, "min": 5, "max": 30},
            "atr_mult": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="atr_channel_breakout",
        cls=AtrChannelBreakoutStrategy,
        category="Channel Based",
        timeframes=["M5"],
        description="ATR channel breakout with volume spike",
        params={
            "ema_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "atr_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "atr_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
            "volume_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        },
    )

    # =========================================================================
    # Group 7: Momentum + Mean Reversion
    # =========================================================================
    _register(
        name="rsi_bb_squeeze",
        cls=RsiBbSqueezeStrategy,
        category="Momentum + Mean Reversion",
        timeframes=["M5"],
        description="RSI bounce + Bollinger squeeze expansion",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "squeeze_lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long": {"type": "float", "default": 35.0, "min": 20, "max": 50},
            "rsi_short": {"type": "float", "default": 65.0, "min": 50, "max": 80},
        },
    )
    _register(
        name="stoch_bb_bounce",
        cls=StochBbBounceStrategy,
        category="Momentum + Mean Reversion",
        timeframes=["M5"],
        description="Bollinger bounce + Stochastic cross",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
        },
    )
    _register(
        name="macd_bb_breakout",
        cls=MacdBbBreakoutStrategy,
        category="Momentum + Mean Reversion",
        timeframes=["M5"],
        description="Bollinger squeeze + MACD cross + breakout",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "squeeze_lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
        },
    )

    # =========================================================================
    # Group 8: Multi-Indicator Confluence
    # =========================================================================
    _register(
        name="triple_confirm",
        cls=TripleConfirmStrategy,
        category="Multi-Indicator Confluence",
        timeframes=["M5"],
        description="EMA + RSI + MACD all confirming",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
        },
    )
    _register(
        name="trend_momentum_vol",
        cls=TrendMomentumVolStrategy,
        category="Multi-Indicator Confluence",
        timeframes=["M5"],
        description="EMA trend + Stochastic + Bollinger expansion",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )
    _register(
        name="adx_rsi_ema",
        cls=AdxRsiEmaStrategy,
        category="Multi-Indicator Confluence",
        timeframes=["M5"],
        description="ADX trend + RSI + EMA alignment",
        params={
            "adx_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "adx_threshold": {"type": "float", "default": 25.0, "min": 15, "max": 40},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )

    # =========================================================================
    # Group 9: Session/Pivot
    # =========================================================================
    _register(
        name="london_ny_breakout",
        cls=LondonNyBreakoutStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="London/NY session range breakout + EMA",
        params={
            "session_lookback_minutes": {"type": "int", "default": 15, "min": 5, "max": 60},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="asian_range_breakout",
        cls=AsianRangeBreakoutStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="Asian session range breakout during London/NY",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="pivot_ema_bounce",
        cls=PivotEmaBounceStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="Pivot point bounce + EMA trend",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="pivot_rsi_bounce",
        cls=PivotRsiBounceStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="Pivot point bounce + RSI extreme",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long": {"type": "float", "default": 35.0, "min": 20, "max": 50},
            "rsi_short": {"type": "float", "default": 65.0, "min": 50, "max": 80},
        },
    )
    _register(
        name="fib_ema_bounce",
        cls=FibEmaBounceStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="Fibonacci 61.8% bounce + EMA trend",
        params={
            "lookback": {"type": "int", "default": 50, "min": 20, "max": 100},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="fib_rsi_bounce",
        cls=FibRsiBounceStrategy,
        category="Session/Pivot",
        timeframes=["M5"],
        description="Fibonacci 61.8% bounce + RSI extreme",
        params={
            "lookback": {"type": "int", "default": 50, "min": 20, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long": {"type": "float", "default": 35.0, "min": 20, "max": 50},
            "rsi_short": {"type": "float", "default": 65.0, "min": 50, "max": 80},
        },
    )

    # =========================================================================
    # Group 10: Trend Following Scalps
    # =========================================================================
    _register(
        name="ema_ribbon_pullback",
        cls=EmaRibbonPullbackStrategy,
        category="Trend Following Scalps",
        timeframes=["M5"],
        description="EMA ribbon aligned + pullback entry",
        params={
            "ema_1": {"type": "int", "default": 5, "min": 3, "max": 10},
            "ema_2": {"type": "int", "default": 8, "min": 5, "max": 15},
            "ema_3": {"type": "int", "default": 13, "min": 10, "max": 20},
            "ema_4": {"type": "int", "default": 21, "min": 15, "max": 30},
        },
    )
    _register(
        name="vwap_ema_cross",
        cls=VwapEmaCrossStrategy,
        category="Trend Following Scalps",
        timeframes=["M5"],
        description="Price crosses VWAP + EMA trend",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="vwap_bounce",
        cls=VwapBounceStrategy,
        category="Trend Following Scalps",
        timeframes=["M5"],
        description="Price bounces from VWAP + RSI",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long": {"type": "float", "default": 40.0, "min": 25, "max": 50},
            "rsi_short": {"type": "float", "default": 60.0, "min": 50, "max": 75},
            "atr_mult": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
        },
    )
    _register(
        name="ichimoku_cloud_bounce",
        cls=IchimokuCloudBounceStrategy,
        category="Trend Following Scalps",
        timeframes=["M5"],
        description="Ichimoku cloud bounce + Tenkan/Kijun cross",
        params={
            "tenkan": {"type": "int", "default": 9, "min": 5, "max": 20},
            "kijun": {"type": "int", "default": 26, "min": 15, "max": 50},
            "senkou_b": {"type": "int", "default": 52, "min": 30, "max": 100},
        },
    )
    _register(
        name="ichimoku_cloud_break",
        cls=IchimokuCloudBreakStrategy,
        category="Trend Following Scalps",
        timeframes=["M5"],
        description="Ichimoku cloud breakout + full alignment",
        params={
            "tenkan": {"type": "int", "default": 9, "min": 5, "max": 20},
            "kijun": {"type": "int", "default": 26, "min": 15, "max": 50},
            "senkou_b": {"type": "int", "default": 52, "min": 30, "max": 100},
        },
    )

    # =========================================================================
    # Group 11: Divergence
    # =========================================================================
    _register(
        name="rsi_divergence",
        cls=RSIDivergenceStrategy,
        category="Divergence",
        timeframes=["M5"],
        description="RSI divergence from price",
        params={
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )
    _register(
        name="macd_divergence",
        cls=MACDDivergenceStrategy,
        category="Divergence",
        timeframes=["M5"],
        description="MACD divergence from price",
        params={
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )
    _register(
        name="stoch_divergence",
        cls=StochDivergenceStrategy,
        category="Divergence",
        timeframes=["M5"],
        description="Stochastic divergence from price",
        params={
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )
    _register(
        name="volume_divergence",
        cls=VolumeDivergenceStrategy,
        category="Divergence",
        timeframes=["M5"],
        description="OBV divergence from price",
        params={
            "obv_ema_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        },
    )

    # =========================================================================
    # Group 12: Hybrid
    # =========================================================================
    _register(
        name="trend_mean_reversion",
        cls=TrendMeanReversionStrategy,
        category="Hybrid",
        timeframes=["M5"],
        description="EMA trend + RSI pullback in trend direction",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long_pullback": {"type": "float", "default": 40.0, "min": 25, "max": 50},
            "rsi_short_pullback": {"type": "float", "default": 60.0, "min": 50, "max": 75},
            "rsi_exit": {"type": "float", "default": 70.0, "min": 60, "max": 85},
        },
    )
    _register(
        name="breakout_retest",
        cls=BreakoutRetestStrategy,
        category="Hybrid",
        timeframes=["M5"],
        description="Key level breakout + retest entry",
        params={
            "breakout_lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
            "retest_tolerance_pct": {"type": "float", "default": 0.1, "min": 0.02, "max": 0.5},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )
    _register(
        name="momentum_exhaustion",
        cls=MomentumExhaustionStrategy,
        category="Hybrid",
        timeframes=["M5"],
        description="Strong momentum candle + Stoch extreme",
        params={
            "body_ratio_threshold": {"type": "float", "default": 0.7, "min": 0.5, "max": 0.9},
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
        },
    )
    _register(
        name="scalp_pullback",
        cls=ScalpPullbackStrategy,
        category="Hybrid",
        timeframes=["M5"],
        description="ADX strong trend + RSI pullback entry",
        params={
            "adx_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "adx_threshold": {"type": "float", "default": 30.0, "min": 20, "max": 45},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long_pullback": {"type": "float", "default": 40.0, "min": 25, "max": 50},
            "rsi_short_pullback": {"type": "float", "default": 60.0, "min": 50, "max": 75},
        },
    )
    _register(
        name="gap_fill",
        cls=GapFillStrategy,
        category="Hybrid",
        timeframes=["M5"],
        description="Gap at session open + fade with RSI",
        params={
            "gap_threshold_pips": {"type": "float", "default": 5.0, "min": 2.0, "max": 20.0},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "rsi_long": {"type": "float", "default": 35.0, "min": 20, "max": 50},
            "rsi_short": {"type": "float", "default": 65.0, "min": 50, "max": 80},
        },
    )

    # =========================================================================
    # Group 13: Advanced
    # =========================================================================
    _register(
        name="ema_macd_rsi_confluence",
        cls=EmaMacdRsiConfluenceStrategy,
        category="Advanced",
        timeframes=["M5"],
        description="3-indicator confluence (EMA + MACD + RSI)",
        params={
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 26},
            "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
        },
    )
    _register(
        name="bb_stoch_volume",
        cls=BbStochVolumeStrategy,
        category="Advanced",
        timeframes=["M5"],
        description="Bollinger + Stochastic + Volume spike",
        params={
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            "stoch_k": {"type": "int", "default": 14, "min": 5, "max": 21},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "volume_mult": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        },
    )
    _register(
        name="supertrend_rsi_ema",
        cls=SupertrendRsiEmaStrategy,
        category="Advanced",
        timeframes=["M5"],
        description="Supertrend + RSI + EMA alignment",
        params={
            "atr_period": {"type": "int", "default": 10, "min": 5, "max": 30},
            "atr_mult": {"type": "float", "default": 3.0, "min": 1.0, "max": 5.0},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
        },
    )


# Auto-populate on import
_populate_registry()
