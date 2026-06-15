
"""
backtest.py
-----------
Fast walk-forward backtester — all registered strategies on USDJPY M1.

Speed architecture
------------------
The original design called strategy.evaluate(window_slice, ts) on every
single candle, which means every strategy was re-running pandas rolling
calculations on a fresh 100-row DataFrame slice 29,000 times.

Benchmarks on this machine:
  rolling inside loop (old):    ~8s / strategy  →  ~9 min for 70 strategies
  pre-computed arrays (new):   ~0.01s / strategy →  <1 min for 70 strategies

Two changes drive all the speed:

1.  Pre-compute all indicators once on the full series.
    Strategies implement a new `precompute(arrays)` method that receives
    the full NumPy arrays and returns a dict of pre-built indicator arrays.
    During the walk-forward loop, the backtester passes only the current
    index i plus the pre-computed arrays — no slicing, no rolling, no pandas.

2.  Parallel execution across CPU cores (ProcessPoolExecutor).
    Each strategy runs in its own process on a copy of the NumPy arrays.
    On a quad-core machine this cuts wall time by ~4×.
    On a single-core machine it adds ~0.1s overhead but never regresses.

Backward compatibility
----------------------
Strategies that do NOT implement `precompute()` fall back to the original
window-slice path automatically, so nothing breaks.

Usage:
    python3 backtest.py

Output:
    backtest_results.csv   — per-strategy summary
    backtest_trades.csv    — every individual trade
"""

from __future__ import annotations
import sys
import os
import logging
import time
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import HistDataAdapter, TickDataAdapter
from core.market_data_store import MarketDataStore
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
from detectors.strategies.base import BaseStrategy
from detectors.signal import PatternSignal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backtest")


# =========================================================================
# Trade model
# =========================================================================

@dataclass
class Trade:
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: str   = ""
    pnl_pips:    float = 0.0
    mae_pips:    float = 0.0
    mfe_pips:    float = 0.0
    bars_held:   int   = 0
    risk_pips:   float = 0.0
    reward_pips: float = 0.0
    rr_ratio:    float = 0.0
    signal_meta: dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy":    self.strategy,
            "direction":   self.direction,
            "entry_time":  self.entry_time,
            "entry_price": self.entry_price,
            "stop_loss":   self.stop_loss,
            "take_profit": self.take_profit,
            "exit_time":   self.exit_time,
            "exit_price":  self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pips":    round(self.pnl_pips,    2),
            "mae_pips":    round(self.mae_pips,     2),
            "mfe_pips":    round(self.mfe_pips,     2),
            "bars_held":   self.bars_held,
            "risk_pips":   round(self.risk_pips,    2),
            "reward_pips": round(self.reward_pips,  2),
            "rr_ratio":    round(self.rr_ratio,     2),
            **{f"sig_{k}": v for k, v in self.signal_meta.items()
               if isinstance(v, (int, float, str, bool))},
        }


# =========================================================================
# Candle arrays — passed to strategies instead of DataFrame slices
# =========================================================================

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


# =========================================================================
# TP/SL vectorised scanner
# =========================================================================

def find_exit(
    arrays: CandleArrays,
    entry_idx: int,
    direction: str,
    tp: float,
    sl: float,
    pip_value: float,
) -> tuple[int, str, float, float, float]:
    """
    Scan forward from entry_idx to find first TP or SL hit.

    Returns
    -------
    (exit_idx, reason, exit_price, mae_pips, mfe_pips)

    This is O(k) where k = bars until exit, but uses numpy vectorised
    comparison so the constant factor is ~50× smaller than a Python loop.
    For trades that hit TP/SL quickly (typical) k is small.
    For EOD trades we scan to the end — acceptable since EOD is rare.
    """
    h = arrays.highs[entry_idx + 1:]
    l = arrays.lows[entry_idx  + 1:]
    ep = arrays.closes[entry_idx]

    if direction == "LONG":
        tp_hits = np.nonzero(h >= tp)[0]
        sl_hits = np.nonzero(l <= sl)[0]
        mfe_raw = h - ep
        mae_raw = l - ep
    else:
        tp_hits = np.nonzero(l <= tp)[0]
        sl_hits = np.nonzero(h >= sl)[0]
        mfe_raw = ep - l
        mae_raw = ep - h

    tp_bar = int(tp_hits[0]) if len(tp_hits) else arrays.n
    sl_bar = int(sl_hits[0]) if len(sl_hits) else arrays.n

    if tp_bar == arrays.n and sl_bar == arrays.n:
        # Neither hit — EOD
        exit_idx = arrays.n - 1
        exit_price = float(arrays.closes[-1])
        reason = "EOD"
    elif tp_bar <= sl_bar:
        exit_idx = entry_idx + 1 + tp_bar
        exit_price = tp
        reason = "TP"
    else:
        exit_idx = entry_idx + 1 + sl_bar
        exit_price = sl
        reason = "SL"

    # MAE/MFE over the bars the trade was actually open
    bars_open = exit_idx - entry_idx
    if bars_open > 0:
        mfe = float(mfe_raw[:bars_open].max()) / pip_value
        mae = float(mae_raw[:bars_open].min()) / pip_value   # negative
    else:
        mfe = mae = 0.0

    return exit_idx, reason, exit_price, mae, mfe


# =========================================================================
# Tick-level TP/SL scanner
# =========================================================================

def find_exit_ticks(
    ticks: pd.DataFrame,
    entry_timestamp: pd.Timestamp,
    direction: str,
    tp: float,
    sl: float,
    pip_value: float,
    m1_timestamps: np.ndarray,
) -> tuple[int, str, float, float, float]:
    """
    Tick-level TP/SL scanner — vectorised over raw ticks.

    Parameters
    ----------
    ticks : DataFrame
        Raw ticks with DatetimeIndex, columns [bid, ask, volume].
    entry_timestamp : Timestamp
        When the trade was entered.
    direction : str
        "LONG" or "SHORT".
    tp, sl : float
        Take-profit and stop-loss prices.
    pip_value : float
        Size of one pip (0.01 for JPY).
    m1_timestamps : np.ndarray
        M1 candle timestamps (sorted ASC) — used to convert exit tick
        time back to a candle index for the main loop.

    Returns
    -------
    (exit_idx, reason, exit_price, mae_pips, mfe_pips)
        Same signature as find_exit() for drop-in compatibility.
    """
    if ticks is None or ticks.empty:
        return len(m1_timestamps) - 1, "EOD", float(m1_timestamps[-1]), 0.0, 0.0

    # Slice ticks from entry forward
    future = ticks.loc[entry_timestamp:]
    if future.empty:
        return len(m1_timestamps) - 1, "EOD", 0.0, 0.0, 0.0

    bid = future["bid"].values
    ask = future["ask"].values
    n = len(bid)

    if direction == "LONG":
        # TP: bid reaches tp → fill at bid
        tp_hits = np.nonzero(bid >= tp)[0]
        # SL: bid drops to sl → fill at bid
        sl_hits = np.nonzero(bid <= sl)[0]
        # MAE/MFE relative to entry midprice
        entry_mid = (bid[0] + ask[0]) / 2.0
        mfe_raw = bid - entry_mid   # favorable = bid rises
        mae_raw = bid - entry_mid   # adverse = bid drops (negative)
    else:
        # TP: ask drops to tp → fill at ask
        tp_hits = np.nonzero(ask <= tp)[0]
        # SL: ask rises to sl → fill at ask
        sl_hits = np.nonzero(ask >= sl)[0]
        entry_mid = (bid[0] + ask[0]) / 2.0
        mfe_raw = entry_mid - ask   # favorable = ask drops
        mae_raw = entry_mid - ask   # adverse = ask rises (negative)

    tp_tick = int(tp_hits[0]) if len(tp_hits) else n
    sl_tick = int(sl_hits[0]) if len(sl_hits) else n

    if tp_tick == n and sl_tick == n:
        # Neither hit — EOD at last tick
        exit_price = float(bid[-1]) if direction == "LONG" else float(ask[-1])
        reason = "EOD"
        exit_tick_idx = n - 1
    elif tp_tick <= sl_tick:
        exit_price = float(bid[tp_tick]) if direction == "LONG" else float(ask[tp_tick])
        reason = "TP"
        exit_tick_idx = tp_tick
    else:
        exit_price = float(bid[sl_tick]) if direction == "LONG" else float(ask[sl_tick])
        reason = "SL"
        exit_tick_idx = sl_tick

    # MAE/MFE over the ticks the trade was actually open
    ticks_open = exit_tick_idx + 1
    if ticks_open > 0:
        mfe = float(mfe_raw[:ticks_open].max()) / pip_value
        mae = float(mae_raw[:ticks_open].min()) / pip_value  # negative
    else:
        mfe = mae = 0.0

    # Convert exit tick timestamp → M1 candle index
    exit_ts = future.index[exit_tick_idx]
    exit_idx = int(np.searchsorted(m1_timestamps, np.datetime64(exit_ts), side="right")) - 1
    exit_idx = max(0, min(exit_idx, len(m1_timestamps) - 1))

    return exit_idx, reason, exit_price, mae, mfe


# =========================================================================
# Single-strategy runner (top-level so it is picklable for multiprocessing)
# =========================================================================

def _run_strategy_job(job: dict) -> dict:
    """
    Runs one strategy over the full candle series.

    Accepts a plain dict (picklable) so ProcessPoolExecutor can ship it
    across process boundaries without pickling the strategy object itself.

    job keys:
        name, strategy_cls, required_tfs,
        arrays_dict   (CandleArrays fields as plain dicts/arrays),
        tf_arrays     (dict of tf -> CandleArrays for higher timeframes),
        pip_value, initial_balance, lot_size, lookback
    """
    import sys, importlib
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent))

    name          = job["name"]
    strategy_cls  = job["strategy_cls"]
    required_tfs  = job["required_tfs"]
    pip_value     = job["pip_value"]
    initial_balance = job["initial_balance"]
    lot_size      = job["lot_size"]
    lookback      = job["lookback"]
    use_sr        = job.get("use_sr", False)

    # Reconstruct CandleArrays
    ad = job["arrays_dict"]
    arrays = CandleArrays(
        timestamps = ad["timestamps"],
        opens      = ad["opens"],
        highs      = ad["highs"],
        lows       = ad["lows"],
        closes     = ad["closes"],
        volumes    = ad["volumes"],
        n          = ad["n"],
    )

    # Reconstruct higher-TF arrays
    tf_arrays: dict[str, CandleArrays] = {}
    for tf, tad in job["tf_arrays_dict"].items():
        tf_arrays[tf] = CandleArrays(
            timestamps = tad["timestamps"],
            opens      = tad["opens"],
            highs      = tad["highs"],
            lows       = tad["lows"],
            closes     = tad["closes"],
            volumes    = tad["volumes"],
            n          = tad["n"],
        )

    # Instantiate strategy
    try:
        strategy: BaseStrategy = strategy_cls()
    except Exception as exc:
        return _error_result(name, str(exc))

    # ------------------------------------------------------------------
    # Fast path: strategy supports precompute()
    # ------------------------------------------------------------------
    supports_precompute = hasattr(strategy, "precompute") and callable(strategy.precompute)

    precomputed: dict[str, Any] = {}
    if supports_precompute:
        try:
            precomputed = strategy.precompute(arrays, tf_arrays) or {}
        except Exception as exc:
            logger.warning("%s.precompute() failed (%s) — falling back to window path", name, exc)
            supports_precompute = False

    # ------------------------------------------------------------------
    # Walk-forward loop
    # ------------------------------------------------------------------
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None

    realized_pnl_pips = 0.0
    balance_curve     = [initial_balance]
    peak_balance      = initial_balance
    max_dd            = 0.0

    _dedup_tol = 1e-6
    last_entry_price: Optional[float] = None

    for i in range(lookback, arrays.n):
        # ── 1. Resolve open trade if it hasn't been closed yet ────────────
        # (open_trade was already forward-scanned at entry; we just check
        #  if we've reached its exit bar)
        if open_trade is not None and i >= open_trade._exit_idx:
            realized_pnl_pips += open_trade.pnl_pips
            trades.append(open_trade)
            open_trade = None

        # ── 2. Skip signal evaluation if trade is open ────────────────────
        if open_trade is not None:
            # Update equity curve
            ep    = open_trade.entry_price
            close = float(arrays.closes[i])
            unreal = (close - ep if open_trade.direction == "LONG"
                      else ep - close) / pip_value
            bal = initial_balance + (realized_pnl_pips + unreal) * lot_size * 100
            balance_curve.append(bal)
            peak_balance = max(peak_balance, bal)
            max_dd = max(max_dd, (peak_balance - bal) / peak_balance * 100)
            continue

        # ── 3. Get signals ─────────────────────────────────────────────────
        signals: list[PatternSignal] = []
        try:
            if supports_precompute:
                # Fast path: strategy uses pre-computed indicator arrays
                signals = strategy.evaluate_fast(i, arrays, precomputed) or []
            else:
                # Fallback: rebuild DataFrame window (original behaviour)
                # Rebuild from job's stored store reference is not available
                # in worker — instead reconstruct a small window DataFrame
                win_start = max(0, i - lookback)
                ts_window = arrays.timestamps[win_start:i+1]
                window_df = pd.DataFrame(
                    {
                        "open":  arrays.opens[win_start:i+1],
                        "high":  arrays.highs[win_start:i+1],
                        "low":   arrays.lows[win_start:i+1],
                        "close": arrays.closes[win_start:i+1],
                        "volume": arrays.volumes[win_start:i+1],
                    },
                    index=pd.DatetimeIndex(ts_window),
                )
                windows = {"M1": window_df}
                # Add higher-TF windows if strategy needs them
                for tf in required_tfs:
                    if tf == "M1":
                        continue
                    if tf in tf_arrays:
                        tfa = tf_arrays[tf]
                        ts_cur = arrays.timestamps[i]
                        pos = int(np.searchsorted(tfa.timestamps, ts_cur, side="right"))
                        ws = max(0, pos - lookback)
                        if pos > 0:
                            windows[tf] = pd.DataFrame(
                                {
                                    "open":  tfa.opens[ws:pos],
                                    "high":  tfa.highs[ws:pos],
                                    "low":   tfa.lows[ws:pos],
                                    "close": tfa.closes[ws:pos],
                                    "volume": tfa.volumes[ws:pos],
                                },
                                index=pd.DatetimeIndex(tfa.timestamps[ws:pos]),
                            )
                cur_ts = pd.Timestamp(arrays.timestamps[i])
                signals = strategy.evaluate(windows, cur_ts) or []
        except Exception as exc:
            logger.debug("%s evaluate error at i=%d: %s", name, i, exc)
            signals = []

        # ── 4. Open trade on first valid signal ────────────────────────────
        if signals:
            sig = signals[0]
            direction   = sig.metadata.get("direction", "")
            close_price = float(arrays.closes[i])
            entry_price = sig.metadata.get("entry_price", close_price)
            tp = sig.metadata.get("take_profit", 0.0)
            sl = sig.metadata.get("stop_loss",   0.0)

            # ATR fallback for missing TP/SL
            if not tp or not sl:
                try:
                    primary_tf = required_tfs[0] if required_tfs else "M1"
                    win_start = max(0, i - lookback)
                    _df = pd.DataFrame(
                        {"high": arrays.highs[win_start:i+1],
                         "low":  arrays.lows[win_start:i+1],
                         "close": arrays.closes[win_start:i+1]},
                    )
                    BaseStrategy.compute_tp_sl(sig, _df)
                    tp = sig.metadata.get("take_profit", 0.0)
                    sl = sig.metadata.get("stop_loss",   0.0)
                except Exception:
                    pass

            # S/R override: if use_sr is enabled, try to improve TP/SL
            if use_sr and tp and sl:
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
                    atr_sl = abs(entry_price - sl)  # current SL distance as ATR proxy
                    sr_tp, sr_sl = sr.get_tp_sl(entry_price, direction, atr_sl)
                    if sr_tp and sr_sl:
                        tp = sr_tp
                        sl = sr_sl
                except Exception:
                    pass  # fall back to ATR TP/SL

            # Deduplicate
            if (last_entry_price is not None and
                    abs(last_entry_price - entry_price) < _dedup_tol):
                pass
            elif direction in ("LONG", "SHORT") and tp and sl:
                risk   = abs(entry_price - sl)  / pip_value
                reward = abs(tp - entry_price)  / pip_value

                # Forward-scan for exit NOW (vectorised, O(k) numpy)
                exit_idx, reason, exit_price, mae, mfe = find_exit(
                    arrays, i, direction, tp, sl, pip_value
                )
                cur_ts = pd.Timestamp(arrays.timestamps[i])
                exit_ts = pd.Timestamp(arrays.timestamps[exit_idx])

                if direction == "LONG":
                    pnl = (exit_price - entry_price) / pip_value
                else:
                    pnl = (entry_price - exit_price) / pip_value

                trade = Trade(
                    strategy    = name,
                    direction   = direction,
                    entry_time  = cur_ts,
                    entry_price = entry_price,
                    stop_loss   = sl,
                    take_profit = tp,
                    exit_time   = exit_ts,
                    exit_price  = exit_price,
                    exit_reason = reason,
                    pnl_pips    = round(pnl, 2),
                    mae_pips    = round(mae, 2),
                    mfe_pips    = round(mfe, 2),
                    bars_held   = exit_idx - i,
                    risk_pips   = round(risk,   2),
                    reward_pips = round(reward, 2),
                    rr_ratio    = round(reward / risk, 2) if risk > 0 else 0.0,
                    signal_meta = dict(sig.metadata),
                )
                # Store exit index so we know when to close it in the loop
                trade._exit_idx = exit_idx   # type: ignore[attr-defined]
                open_trade  = trade
                last_entry_price = entry_price
                realized_pnl_pips += 0   # don't add yet; add at close bar

        # ── 5. Equity curve ────────────────────────────────────────────────
        close = float(arrays.closes[i])
        unreal = 0.0
        if open_trade is not None:
            ep = open_trade.entry_price
            unreal = (close - ep if open_trade.direction == "LONG"
                      else ep - close) / pip_value
        bal = initial_balance + (realized_pnl_pips + unreal) * lot_size * 100
        balance_curve.append(bal)
        peak_balance = max(peak_balance, bal)
        max_dd = max(max_dd, (peak_balance - bal) / peak_balance * 100)

    # Close any still-open trade at EOD
    if open_trade is not None:
        realized_pnl_pips += open_trade.pnl_pips
        trades.append(open_trade)

    return _build_result(name, trades, max_dd, balance_curve)


def _error_result(name: str, err: str) -> dict:
    return {
        "strategy": name, "error": err,
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0, "total_pnl_pips": 0, "avg_pnl_pips": 0,
        "expectancy_pips": 0, "profit_factor": 0, "max_drawdown_pct": 0,
        "gross_profit": 0, "gross_loss": 0,
        "avg_mae_pips": 0, "avg_mfe_pips": 0,
        "trades": [], "balance_curve": [],
    }


def _build_result(
    name: str,
    trades: list[Trade],
    max_dd: float,
    balance_curve: list[float],
) -> dict:
    total        = len(trades)
    winning      = [t for t in trades if t.pnl_pips > 0]
    losing       = [t for t in trades if t.pnl_pips <= 0]
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
        "profit_factor":    round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "avg_mae_pips":     round(sum(t.mae_pips for t in trades) / total, 2) if total > 0 else 0,
        "avg_mfe_pips":     round(sum(t.mfe_pips for t in trades) / total, 2) if total > 0 else 0,
        "trades":           trades,
        "balance_curve":    balance_curve,
    }


# =========================================================================
# Backtester — orchestrates parallel jobs
# =========================================================================

class Backtester:
    """
    Orchestrates parallel strategy backtests.

    Each strategy runs in its own worker (process or thread depending on OS).
    All heavy computation is in _run_strategy_job() which is top-level and
    therefore picklable for multiprocessing.
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        symbol: str          = "USDJPY",
        initial_balance: float = 10_000.0,
        lot_size: float      = 0.01,
        pip_value: float     = 0.01,
        max_workers: Optional[int] = None,
        lookback: int        = 100,
        use_sr: bool         = False,
    ) -> None:
        self._store    = data_store
        self._symbol   = symbol
        self._initial_balance = initial_balance
        self._lot_size = lot_size
        self._pip_value = pip_value
        self._lookback  = lookback
        self._use_sr    = use_sr

        cpus = os.cpu_count() or 1
        self._max_workers = max_workers if max_workers is not None else max(1, cpus - 1)

        # Pre-extract arrays once — shared across all strategy jobs
        m1_df = data_store.get_data(symbol, "M1")
        self._arrays = CandleArrays.from_dataframe(m1_df)

        # Higher-TF arrays
        self._tf_arrays: dict[str, CandleArrays] = {}
        for tf in data_store.available_timeframes(symbol):
            if tf == "M1":
                continue
            try:
                self._tf_arrays[tf] = CandleArrays.from_dataframe(
                    data_store.get_data(symbol, tf)
                )
            except Exception:
                pass

    def _make_job(self, name: str, strategy_cls, required_tfs: list[str]) -> dict:
        """Build a picklable job dict for the worker process."""
        def arr_dict(a: CandleArrays) -> dict:
            return {
                "timestamps": a.timestamps,
                "opens":  a.opens,
                "highs":  a.highs,
                "lows":   a.lows,
                "closes": a.closes,
                "volumes": a.volumes,
                "n": a.n,
            }
        return {
            "name":           name,
            "strategy_cls":   strategy_cls,
            "required_tfs":   required_tfs,
            "arrays_dict":    arr_dict(self._arrays),
            "tf_arrays_dict": {tf: arr_dict(a) for tf, a in self._tf_arrays.items()},
            "pip_value":      self._pip_value,
            "initial_balance": self._initial_balance,
            "lot_size":       self._lot_size,
            "lookback":       self._lookback,
            "use_sr":         self._use_sr,
        }

    def run_all(
        self,
        registry: dict,
        progress: bool = True,
    ) -> list[dict]:
        """
        Run all strategies and return list of result dicts.

        Uses ProcessPoolExecutor for parallel execution.
        Falls back to sequential if workers=1 or pickling fails.
        """
        jobs = []
        for name, info in registry.items():
            jobs.append(self._make_job(name, info["class"], info["timeframes"]))

        results: list[dict] = []
        completed = 0
        total = len(jobs)
        t_start = time.perf_counter()

        if self._max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._max_workers
            ) as ex:
                futures = {ex.submit(_run_strategy_job, j): j["name"] for j in jobs}
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = _error_result(name, str(exc))
                    results.append(result)
                    completed += 1
                    if progress:
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed / completed * (total - completed) if completed else 0
                        status = "✓" if result["total_trades"] > 0 else "○"
                        err    = f"  ERROR: {result.get('error','')}" if result.get("error") else ""
                        print(
                            f"  {status} [{completed:2d}/{total}] {name:40s}"
                            f"  trades={result['total_trades']:3d}"
                            f"  pnl={result['total_pnl_pips']:+8.1f} pips"
                            f"  WR={result['win_rate']:.0f}%"
                            f"  ETA {eta:.0f}s{err}"
                        )
        else:
            # Sequential fallback
            for j in jobs:
                name = j["name"]
                try:
                    result = _run_strategy_job(j)
                except Exception as exc:
                    result = _error_result(name, str(exc))
                results.append(result)
                completed += 1
                if progress:
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / completed * (total - completed) if completed else 0
                    status = "✓" if result["total_trades"] > 0 else "○"
                    print(
                        f"  {status} [{completed:2d}/{total}] {name:40s}"
                        f"  trades={result['total_trades']:3d}"
                        f"  pnl={result['total_pnl_pips']:+8.1f} pips"
                        f"  WR={result['win_rate']:.0f}%"
                        f"  ETA {eta:.0f}s"
                    )

        return results


# =========================================================================
# Reporting
# =========================================================================

def _save_summary_csv(
    results: list[dict],
    out_dir: Path,
    meta: dict,
) -> Path:
    timestamp = meta["timestamp"]
    data_stem = Path(meta["data_file"]).stem
    filter_tag = meta.get("filter_tag", "all")

    fname = f"bt_{data_stem}_{filter_tag}_{timestamp}.csv"
    out_path = out_dir / fname

    rows = [
        {
            "backtest_timestamp": timestamp,
            "data_file": meta["data_file"],
            "symbol": meta["symbol"],
            "strategies_filter": meta.get("strategies_filter", "all"),
            "n_strategies_run": meta["n_strategies"],
            **{k: r[k] for k in (
                "strategy", "total_trades", "win_rate", "total_pnl_pips",
                "avg_pnl_pips", "expectancy_pips", "profit_factor",
                "max_drawdown_pct", "gross_profit", "gross_loss",
                "avg_mae_pips", "avg_mfe_pips",
            )},
        }
        for r in results if r["total_trades"] > 0
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values("total_pnl_pips", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1
    df.index.name = "rank"
    df.to_csv(str(out_path))
    print(f"Summary saved  → {out_path}")

    # Latest symlink
    latest = out_dir / "backtest_results_latest.csv"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_path.name)

    return out_path


def _save_trades_csv(
    results: list[dict],
    out_dir: Path,
    meta: dict,
) -> Path:
    timestamp = meta["timestamp"]
    data_stem = Path(meta["data_file"]).stem
    filter_tag = meta.get("filter_tag", "all")

    fname = f"bt_trades_{data_stem}_{filter_tag}_{timestamp}.csv"
    out_path = out_dir / fname

    rows = []
    for r in results:
        for t in r["trades"]:
            d = t.to_dict()
            d["backtest_timestamp"] = timestamp
            d["data_file"] = meta["data_file"]
            d["symbol"] = meta["symbol"]
            d["strategies_filter"] = meta.get("strategies_filter", "all")
            rows.append(d)

    if not rows:
        print("No trades to export.")
        return out_path

    pd.DataFrame(rows).sort_values(["entry_time", "strategy"]).to_csv(
        str(out_path), index=False
    )
    print(f"Trades saved   → {out_path}  ({len(rows):,} rows)")

    # Latest symlink
    latest = out_dir / "backtest_trades_latest.csv"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_path.name)

    return out_path


def _print_ranked_table(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("RANKED RESULTS")
    print("=" * 110)
    print(
        f"{'Rank':>4}  {'Strategy':40s}  {'Trades':>6}  {'Win%':>5}  "
        f"{'PnL':>10}  {'Exp':>7}  {'PF':>6}  {'MaxDD%':>7}  "
        f"{'AvgMAE':>7}  {'AvgMFE':>7}"
    )
    print("-" * 110)
    for i, r in enumerate(results, 1):
        if r["total_trades"] == 0:
            continue
        print(
            f"{i:4d}  {r['strategy']:40s}  {r['total_trades']:6d}  "
            f"{r['win_rate']:5.1f}  {r['total_pnl_pips']:+10.1f}  "
            f"{r['expectancy_pips']:+7.1f}  {r['profit_factor']:6.2f}  "
            f"{r['max_drawdown_pct']:7.2f}  "
            f"{r['avg_mae_pips']:+7.1f}  {r['avg_mfe_pips']:+7.1f}"
        )


def _print_top_n(results: list[dict], n: int = 5) -> None:
    print(f"\n{'='*80}\nTOP {n} STRATEGIES\n{'='*80}")
    active = [r for r in results if r["total_trades"] > 0]
    for i, r in enumerate(active[:n], 1):
        print(f"\n  #{i} {r['strategy']}")
        print(f"     Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.1f}%")
        print(f"     Total PnL: {r['total_pnl_pips']:+.1f} pips | "
              f"Avg: {r['avg_pnl_pips']:+.1f} | Expectancy: {r['expectancy_pips']:+.1f}")
        print(f"     Profit Factor: {r['profit_factor']:.2f} | Max DD: {r['max_drawdown_pct']:.2f}%")
        print(f"     Avg MAE: {r['avg_mae_pips']:+.1f} pips | Avg MFE: {r['avg_mfe_pips']:+.1f} pips")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RSFX Backtester — run strategies on USDJPY M1"
    )
    parser.add_argument(
        "--strategies", "-s",
        type=str,
        default=None,
        help="Comma-separated strategy names to run (default: all). "
             "Partial matches allowed, e.g. 'tweezer,h1_trend' runs all "
             "strategies containing either substring.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to M1 CSV file (default: data/DAT_ASCII_USDJPY_M1_202605.csv)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="USDJPY",
        help="Symbol name (default: USDJPY)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPUs - 1)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV output files",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top strategies to display (default: 5)",
    )
    parser.add_argument(
        "--use-sr",
        action="store_true",
        help="Use S/R levels for TP/SL instead of ATR (support/resistance aware exits)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("RSFX BACKTESTER — USDJPY M1")
    print("=" * 80)

    # Load CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(__file__).parent.parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv"
    print(f"\nLoading {csv_path}...")

    # Auto-detect: tick data (bid/ask columns) vs candle data (OHLC columns)
    with open(csv_path, "r") as f:
        header = f.readline().strip().lower()
    if "bid" in header and "ask" in header:
        print("  Detected tick data format (bid/ask) — building M1 candles from ticks")
        m1_df = TickDataAdapter().load(str(csv_path))
    else:
        m1_df = HistDataAdapter().load(str(csv_path))
    print(f"Loaded {len(m1_df):,} M1 candles  {m1_df.index[0]} → {m1_df.index[-1]}")

    store = MarketDataStore()
    store.load_symbol(args.symbol, m1_df)

    # Load strategies
    print("\nLoading strategies...")
    _populate_registry()
    print(f"Found {len(STRATEGY_REGISTRY)} strategies total")

    # Filter strategies if --strategies provided
    registry = STRATEGY_REGISTRY
    if args.strategies:
        filter_names = [s.strip().lower() for s in args.strategies.split(",")]
        registry = {
            name: info
            for name, info in STRATEGY_REGISTRY.items()
            if any(f in name.lower() for f in filter_names)
        }
        if not registry:
            print(f"\nERROR: No strategies matched: {args.strategies}")
            print("Available strategies:")
            for name in sorted(STRATEGY_REGISTRY):
                print(f"  {name}")
            sys.exit(1)
        print(f"  Filtered to {len(registry)} strategies:")
        for name in sorted(registry):
            print(f"    • {name}")

    print()

    # Workers
    cpus = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else max(1, cpus - 1)
    print(f"Running on {workers} worker(s) (CPUs: {cpus})\n")

    bt = Backtester(store, symbol=args.symbol, max_workers=workers, use_sr=args.use_sr)

    t0 = time.perf_counter()
    results = bt.run_all(registry, progress=True)
    elapsed = time.perf_counter() - t0

    sr_tag = " [S/R-aware TP/SL]" if args.use_sr else ""
    print(f"\nCompleted {len(registry)} strategies in {elapsed:.1f}s "
          f"({elapsed/len(registry):.2f}s/strategy){sr_tag}")

    results.sort(key=lambda x: x["total_pnl_pips"], reverse=True)

    _print_ranked_table(results)
    _print_top_n(results, n=args.top)

    if not args.no_save:
        base = Path(__file__).parent.parent / 'results'
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        data_file = str(csv_path) if not csv_path.is_relative_to(Path(__file__).parent.parent) else str(csv_path.relative_to(Path(__file__).parent.parent))
        filter_tag = args.strategies.replace(",", "_").replace(" ", "") if args.strategies else "all"
        meta = {
            "timestamp": run_ts,
            "data_file": data_file,
            "symbol": args.symbol,
            "strategies_filter": args.strategies or "all",
            "filter_tag": filter_tag,
            "n_strategies": len(registry),
        }
        _save_summary_csv(results, base, meta)
        _save_trades_csv(results,  base, meta)


if __name__ == "__main__":
    main()
