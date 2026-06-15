"""
detectors/strategies/ema_macd_rsi_confluence.py
------------------------------------------------
3-indicator confluence: EMA crossover + MACD histogram + RSI momentum (M5 only).

Rules:
  LONG:  EMA9 > EMA21 + MACD > signal + RSI > 50
  SHORT: EMA9 < EMA21 + MACD < signal + RSI < 50
  Exit when any 2 of 3 indicators flip direction.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .base import BaseStrategy
from detectors.signal import PatternSignal

logger = logging.getLogger(__name__)

try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA-Lib not installed. EmaMacdRsiConfluence strategy signals disabled.")


class EmaMacdRsiConfluenceStrategy(BaseStrategy):
    """EMA + MACD + RSI 3-indicator confluence with exit logic."""

    name = "ema_macd_rsi_confluence"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.macd_slow + self.macd_signal, self.ema_slow, self.rsi_period) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        macd_line, signal_line, _ = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # --- Current conditions ---
        ema_bull = ema_f[-1] > ema_s[-1]
        macd_bull = macd_line[-1] > signal_line[-1]
        rsi_bull = rsi[-1] > 50

        # --- Previous conditions (for exit detection) ---
        ema_bull_prev = ema_f[-2] > ema_s[-2]
        macd_bull_prev = macd_line[-2] > signal_line[-2]
        rsi_bull_prev = rsi[-2] > 50

        # --- Full confluence signals ---
        if ema_bull and macd_bull and rsi_bull:
            confidence = self._calc_confidence(ema_f[-1], ema_s[-1], macd_line[-1], signal_line[-1], rsi[-1])
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "macd": float(macd_line[-1]),
                    "macd_signal": float(signal_line[-1]),
                    "rsi": float(rsi[-1]),
                    "flip": False,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, RSI=%.1f)", current_timestamp, self.name, rsi[-1])

        elif not ema_bull and not macd_bull and not rsi_bull:
            confidence = self._calc_confidence(ema_f[-1], ema_s[-1], macd_line[-1], signal_line[-1], rsi[-1])
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "macd": float(macd_line[-1]),
                    "macd_signal": float(signal_line[-1]),
                    "rsi": float(rsi[-1]),
                    "flip": False,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, RSI=%.1f)", current_timestamp, self.name, rsi[-1])

        # --- Exit detection: any 2 of 3 indicators flipped from previous bar ---
        flips = 0
        if ema_bull != ema_bull_prev:
            flips += 1
        if macd_bull != macd_bull_prev:
            flips += 1
        if rsi_bull != rsi_bull_prev:
            flips += 1

        if flips >= 2:
            # Determine exit direction based on current state
            bull_count = sum([ema_bull, macd_bull, rsi_bull])
            if bull_count >= 2:
                exit_dir = "LONG"
            else:
                exit_dir = "SHORT"

            detected.append(PatternSignal(
                name=f"{self.name}_EXIT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.8,
                metadata={
                    "strategy": self.name,
                    "direction": exit_dir,
                    "flip": True,
                    "flips": flips,
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "macd": float(macd_line[-1]),
                    "macd_signal": float(signal_line[-1]),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("EXIT signal at %s (strategy=%s, flips=%d)", current_timestamp, self.name, flips)

        return detected

    @staticmethod
    def _calc_confidence(ema_f, ema_s, macd, signal, rsi) -> float:
        """Scale confidence based on indicator separation from neutral."""
        ema_sep = abs(ema_f - ema_s) / ema_s if ema_s != 0 else 0
        macd_sep = abs(macd - signal) / abs(signal) if signal != 0 else 0
        rsi_dist = abs(rsi - 50) / 50.0
        raw = 0.5 + min(ema_sep, 0.15) + min(macd_sep, 0.15) + min(rsi_dist * 0.2, 0.15)
        return round(min(raw, 1.0), 2)
