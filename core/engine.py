"""
core/engine.py
--------------
Shared data primitives and trading math for the RSFX platform.

CandleArrays is the canonical container for OHLCV NumPy arrays
used by strategies, the backtester, and the live engine.

Functions
---------
    compute_tp_sl()      — TP/SL from signal + ATR fallback + S/R override
    build_result()       — Stats dict from trade list (win_rate, PF, etc.)
    compute_pnl()        — Direction-aware PnL in pips
    apply_spread()       — Deduct spread cost from PnL
    check_min_rr()       — Filter: is R:R ratio above threshold?
    check_dedup()        — Filter: entry price too close to last trade?
    update_equity()      — Balance curve, peak, max drawdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from detectors.strategies.base import BaseStrategy
from detectors.signal import PatternSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CandleArrays — canonical OHLCV container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandleArrays:
    """
    Full-series NumPy arrays extracted once from the DataFrame.
    Strategies receive these arrays + the current index i.
    Indexing a NumPy array is ~100x faster than pandas .iloc[i]['col'].
    """
    timestamps: np.ndarray   # dtype=datetime64[ns]
    opens:      np.ndarray
    highs:      np.ndarray
    lows:       np.ndarray
    closes:     np.ndarray
    volumes:    np.ndarray
    n:          int          # total length

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CandleArrays":
        return cls(
            timestamps = df.index.values,
            opens      = df["open"].values,
            highs      = df["high"].values,
            lows       = df["low"].values,
            closes     = df["close"].values,
            volumes    = df["volume"].values if "volume" in df.columns else np.zeros(len(df)),
            n          = len(df),
        )


# ---------------------------------------------------------------------------
# TP/SL computation
# ---------------------------------------------------------------------------

def compute_tp_sl(
    signal: PatternSignal,
    arrays,                    # CandleArrays
    i: int,
    lookback: int = 100,
    use_sr: bool = False,
) -> tuple[float, float]:
    """
    Get TP/SL from signal metadata, with ATR fallback and optional S/R override.

    Parameters
    ----------
    signal   : PatternSignal with metadata containing take_profit, stop_loss, direction
    arrays   : CandleArrays (needs .opens, .highs, .lows, .closes)
    i        : Current candle index
    lookback : Window size for ATR/S/R computation
    use_sr   : If True, try to improve TP/SL using Support/Resistance levels

    Returns
    -------
    (tp, sl) — take-profit and stop-loss prices (0.0, 0.0 on failure)
    """
    tp = signal.metadata.get("take_profit", 0.0)
    sl = signal.metadata.get("stop_loss", 0.0)
    direction = signal.metadata.get("direction", "")

    if tp and sl:
        entry = float(arrays.closes[i])
        # Sanity: ensure correct direction
        if direction == "LONG":
            if tp <= entry:
                tp = entry + abs(tp - entry)
            if sl >= entry:
                sl = entry - abs(sl - entry)
        else:
            if tp >= entry:
                tp = entry - abs(tp - entry)
            if sl <= entry:
                sl = entry + abs(sl - entry)

        # S/R override
        if use_sr:
            tp, sl = _try_sr_override(arrays, i, lookback, entry, direction, tp, sl)

        return tp, sl

    # ATR fallback
    win_start = max(0, i - lookback)
    _df = pd.DataFrame(
        {"high":  arrays.highs[win_start:i+1],
         "low":   arrays.lows[win_start:i+1],
         "close": arrays.closes[win_start:i+1]},
    )
    try:
        BaseStrategy.compute_tp_sl(signal, _df)
        tp = signal.metadata.get("take_profit", 0.0)
        sl = signal.metadata.get("stop_loss", 0.0)

        # S/R override on ATR fallback too
        if use_sr and tp and sl:
            entry = float(arrays.closes[i])
            tp, sl = _try_sr_override(arrays, i, lookback, entry, direction, tp, sl)

        return tp, sl
    except Exception:
        return 0.0, 0.0


def _try_sr_override(
    arrays, i: int, lookback: int,
    entry: float, direction: str,
    tp: float, sl: float,
) -> tuple[float, float]:
    """Attempt S/R override — returns (tp, sl) unchanged on failure."""
    try:
        from detectors.support_resistance import SupportResistance
        win_start = max(0, i - lookback)
        _sr_df = pd.DataFrame(
            {"open":  arrays.opens[win_start:i+1],
             "high":  arrays.highs[win_start:i+1],
             "low":   arrays.lows[win_start:i+1],
             "close": arrays.closes[win_start:i+1]},
        )
        sr = SupportResistance(_sr_df, pip_tolerance=0.10, min_touches=2)
        atr_sl = abs(entry - sl)
        sr_tp, sr_sl = sr.get_tp_sl(entry, direction, atr_sl)
        if sr_tp and sr_sl:
            return sr_tp, sr_sl
    except Exception:
        pass
    return tp, sl


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

def compute_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    pip_value: float,
) -> float:
    """Direction-aware PnL in pips."""
    if direction == "LONG":
        return (exit_price - entry_price) / pip_value
    else:
        return (entry_price - exit_price) / pip_value


def apply_spread(pnl_pips: float, spread_pips: float) -> float:
    """Deduct round-trip spread cost from PnL."""
    return pnl_pips - spread_pips


# ---------------------------------------------------------------------------
# Trade filters
# ---------------------------------------------------------------------------

def check_min_rr(entry_price: float, tp: float, sl: float, pip_value: float,
                 min_rr: float) -> bool:
    """Return True if risk:reward ratio >= min_rr."""
    risk = abs(entry_price - sl) / pip_value
    reward = abs(tp - entry_price) / pip_value
    if risk <= 0:
        return False
    return (reward / risk) >= min_rr


def check_dedup(last_entry_price: Optional[float], entry_price: float,
                tol: float = 1e-6) -> bool:
    """Return True if entry is a duplicate (too close to last entry)."""
    if last_entry_price is not None and abs(last_entry_price - entry_price) < tol:
        return True
    return False


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def update_equity(
    open_price: Optional[float],
    direction: str,
    current_close: float,
    realized_pnl_pips: float,
    pip_value: float,
    initial_balance: float,
    lot_size: float,
    balance_curve: list[float],
    peak_balance: float,
    max_dd: float,
) -> tuple[list[float], float, float]:
    """
    Append to balance curve and update peak / max drawdown.

    Returns (balance_curve, peak_balance, max_dd).
    """
    unreal = 0.0
    if open_price is not None:
        unreal = compute_pnl(direction, open_price, current_close, pip_value)
    bal = initial_balance + (realized_pnl_pips + unreal) * lot_size * 100
    balance_curve.append(bal)
    peak_balance = max(peak_balance, bal)
    if peak_balance > 0:
        max_dd = max(max_dd, (peak_balance - bal) / peak_balance * 100)
    return balance_curve, peak_balance, max_dd


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def build_result(
    name: str,
    trades: list,              # list[Trade]
    max_dd: float = 0.0,
    balance_curve: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute standard stats dict from a list of Trade objects.

    Returns dict with keys: strategy, total_trades, win_rate, total_pnl_pips,
    profit_factor, expectancy_pips, avg_mae_pips, avg_mfe_pips, etc.
    """
    if balance_curve is None:
        balance_curve = []

    total   = len(trades)
    winning = [t for t in trades if t.pnl_pips > 0]
    losing  = [t for t in trades if t.pnl_pips <= 0]

    total_pnl    = sum(t.pnl_pips for t in trades)
    gross_profit = sum(t.pnl_pips for t in winning)
    gross_loss   = abs(sum(t.pnl_pips for t in losing))
    win_rate     = len(winning) / total if total > 0 else 0.0
    avg_win      = gross_profit / len(winning) if winning else 0.0
    avg_loss     = gross_loss   / len(losing)  if losing  else 0.0
    expectancy   = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "strategy":         name,
        "total_trades":     total,
        "winning_trades":   len(winning),
        "losing_trades":    len(losing),
        "win_rate":         round(win_rate * 100, 2),
        "total_pnl_pips":   round(total_pnl, 2),
        "avg_pnl_pips":     round(total_pnl / total, 2) if total > 0 else 0,
        "expectancy_pips":  round(expectancy, 2),
        "profit_factor":    round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "max_drawdown_pct": round(max_dd, 2),
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "avg_mae_pips":     round(sum(t.mae_pips for t in trades) / total, 2) if total > 0 else 0,
        "avg_mfe_pips":     round(sum(t.mfe_pips for t in trades) / total, 2) if total > 0 else 0,
        "trades":           trades,
        "balance_curve":    balance_curve,
    }
