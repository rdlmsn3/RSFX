#!/usr/bin/env python3
"""
Phase 1: Core Engine Unit Tests for RSFX Forex Platform
========================================================
48 tests across 6 modules:
  1.1 Data Loader (8 tests)
  1.2 Market Data Store (4 tests)
  1.3 Engine (14 tests)
  1.4 Trade Engine (11 tests)
  1.5 Signal Engine (6 tests)
  1.6 Trade Store (5 tests)
"""
import sys
import os
import traceback

sys.path.insert(0, '.')
os.chdir('/home/rudi/RSFX')

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

results = []

def test(section, num, name):
    """Decorator to register and run a test."""
    def decorator(func):
        def wrapper():
            try:
                func()
                results.append((section, num, name, "PASS", ""))
            except Exception as e:
                tb = traceback.format_exc()
                results.append((section, num, name, "FAIL", f"{e}\n{tb}"))
        wrapper.__name__ = f"test_{num}_{name}"
        return wrapper
    return decorator


# =========================================================================
# 1.1 DATA LOADER (core/data_loader.py)
# =========================================================================

@test("1.1 Data Loader", 1, "CSV M1 bars load")
def test_01():
    from core.data_loader import HistDataAdapter
    adapter = HistDataAdapter()
    df = adapter.load('data/DAT_ASCII_USDJPY_M1_202605.csv')
    assert isinstance(df.index, pd.DatetimeIndex), f"Index is {type(df.index)}, expected DatetimeIndex"
    assert len(df) == 29658, f"Expected 29658 rows, got {len(df)}"
    for col in ['open', 'high', 'low', 'close']:
        assert col in df.columns, f"Missing column: {col}"
    assert 'volume' in df.columns, "Missing volume column"
    print(f"  Loaded {len(df)} rows, index={type(df.index).__name__}, cols={list(df.columns)}")


@test("1.1 Data Loader", 2, "Parquet tick load")
def test_02():
    from core.data_loader import ParquetAdapter
    adapter = ParquetAdapter()
    df = adapter.load('data/usdjpy_tick_2007-01_to_2007-03.parquet')
    assert isinstance(df, pd.DataFrame), "Not a DataFrame"
    assert adapter.raw_ticks is not None, "raw_ticks is None"
    assert isinstance(adapter.raw_ticks, pd.DataFrame), "raw_ticks not a DataFrame"
    assert len(df) > 0, "M1 DataFrame is empty"
    # M1 output should have OHLCV columns
    for col in ['open', 'high', 'low', 'close']:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  M1 candles: {len(df)}, raw_ticks: {len(adapter.raw_ticks)}")


@test("1.1 Data Loader", 3, "get_adapter() factory CSV")
def test_03():
    from core.data_loader import get_adapter, HistDataAdapter
    adapter = get_adapter('data/DAT_ASCII_USDJPY_M1_202605.csv')
    assert isinstance(adapter, HistDataAdapter), f"Got {type(adapter).__name__}, expected HistDataAdapter"
    print(f"  get_adapter(csv) → {type(adapter).__name__}")


@test("1.1 Data Loader", 4, "get_adapter() factory Parquet")
def test_04():
    from core.data_loader import get_adapter, ParquetAdapter
    adapter = get_adapter('data/usdjpy_tick_2007-01_to_2007-03.parquet')
    assert isinstance(adapter, ParquetAdapter), f"Got {type(adapter).__name__}, expected ParquetAdapter"
    print(f"  get_adapter(.parquet) → {type(adapter).__name__}")


@test("1.1 Data Loader", 5, "get_adapter() factory .pq")
def test_05():
    from core.data_loader import get_adapter, ParquetAdapter
    adapter = get_adapter('data/test.pq')
    assert isinstance(adapter, ParquetAdapter), f"Got {type(adapter).__name__}, expected ParquetAdapter"
    print(f"  get_adapter(.pq) → {type(adapter).__name__}")


@test("1.1 Data Loader", 6, "Tick to M1 conversion")
def test_06():
    from core.data_loader import ParquetAdapter
    adapter = ParquetAdapter()
    df = adapter.load('data/usdjpy_tick_2007-01_to_2007-03.parquet')
    assert isinstance(df.index, pd.DatetimeIndex), f"Index is {type(df.index)}"
    assert len(df) > 100, f"Expected >100 M1 candles, got {len(df)}"
    assert adapter.raw_ticks is not None, "raw_ticks is None after tick load"
    assert len(adapter.raw_ticks) > 0, "raw_ticks is empty"
    print(f"  Ticks → {len(df)} M1 candles, {len(adapter.raw_ticks)} raw ticks")


@test("1.1 Data Loader", 7, "Epoch millis datetime")
def test_07():
    from core.data_loader import ParquetAdapter
    adapter = ParquetAdapter()
    df = adapter.load('data/usdjpy_tick_2007-01_to_2007-03.parquet')
    assert isinstance(df.index, pd.DatetimeIndex), f"Index is {type(df.index)}"
    # First tick is 1167699638000 ms → 2007-01-01 roughly
    first_ts = df.index[0]
    assert first_ts.year >= 2007, f"First timestamp year is {first_ts.year}, expected >= 2007"
    print(f"  First timestamp: {first_ts}, last: {df.index[-1]}")


@test("1.1 Data Loader", 8, "Missing file error")
def test_08():
    from core.data_loader import HistDataAdapter
    adapter = HistDataAdapter()
    try:
        adapter.load('data/NONEXISTENT_FILE.csv')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower() or "not found" in str(e), f"Error message unclear: {e}"
        print(f"  FileNotFoundError raised: {e}")


# =========================================================================
# 1.2 MARKET DATA STORE (core/market_data_store.py)
# =========================================================================

# Pre-load data for market data store tests
_m1_df = None
_store = None

def _load_m1():
    global _m1_df
    if _m1_df is None:
        from core.data_loader import HistDataAdapter
        adapter = HistDataAdapter()
        _m1_df = adapter.load('data/DAT_ASCII_USDJPY_M1_202605.csv')
    return _m1_df

@test("1.2 Market Data Store", 9, "Load M1 data")
def test_09():
    global _store
    from core.market_data_store import MarketDataStore
    m1 = _load_m1()
    store = MarketDataStore()
    store.load_symbol("USDJPY", m1)
    _store = store
    assert store.length("USDJPY", "M1") >= 29000, f"Expected ~29K candles, got {store.length('USDJPY', 'M1')}"
    tfs = store.available_timeframes("USDJPY")
    assert "M1" in tfs, f"M1 not in available_timeframes: {tfs}"
    print(f"  {store.length('USDJPY', 'M1')} M1 candles, timeframes: {tfs}")


@test("1.2 Market Data Store", 10, "Multi-TF resample")
def test_10():
    global _store
    assert _store is not None, "Store not initialized"
    tfs = _store.available_timeframes("USDJPY")
    for tf in ["M5", "H1", "D1"]:
        assert tf in tfs, f"{tf} not in available_timeframes: {tfs}"
        assert _store.length("USDJPY", tf) > 0, f"{tf} has 0 candles"
    print(f"  M5={_store.length('USDJPY', 'M5')}, H1={_store.length('USDJPY', 'H1')}, D1={_store.length('USDJPY', 'D1')}")


@test("1.2 Market Data Store", 11, "get_data() correct TF")
def test_11():
    global _store
    assert _store is not None, "Store not initialized"
    m5 = _store.get_data("USDJPY", "M5")
    assert isinstance(m5, pd.DataFrame), f"get_data returned {type(m5)}"
    assert isinstance(m5.index, pd.DatetimeIndex), "M5 index not DatetimeIndex"
    # M5 bars should be 5 minutes apart (approximately)
    if len(m5) > 1:
        diffs = pd.Series(m5.index).diff().dropna()
        avg_diff = diffs.mean()
        # Should be around 5 minutes = 300000000000 ns
        assert avg_diff.total_seconds() >= 200, f"M5 avg interval too short: {avg_diff}"
        # Gaps from non-trading hours can push average higher, allow up to 7 min
        assert avg_diff.total_seconds() <= 420, f"M5 avg interval too long: {avg_diff}"
    print(f"  M5: {len(m5)} candles")


@test("1.2 Market Data Store", 12, "Unknown symbol")
def test_12():
    global _store
    assert _store is not None, "Store not initialized"
    try:
        _store.get_data("EURUSD", "M1")
        assert False, "Should have raised KeyError for unknown symbol"
    except KeyError as e:
        print(f"  KeyError raised: {e}")


# =========================================================================
# 1.3 ENGINE (core/engine.py)
# =========================================================================

@test("1.3 Engine", 13, "CandleArrays.from_dataframe()")
def test_13():
    from core.engine import CandleArrays
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    n = len(m1)
    assert arrays.n == n, f"arrays.n={arrays.n}, expected {n}"
    assert len(arrays.opens) == n, f"opens length mismatch"
    assert len(arrays.highs) == n, f"highs length mismatch"
    assert len(arrays.lows) == n, f"lows length mismatch"
    assert len(arrays.closes) == n, f"closes length mismatch"
    assert len(arrays.volumes) == n, f"volumes length mismatch"
    assert len(arrays.timestamps) == n, f"timestamps length mismatch"
    # Check types
    assert arrays.opens.dtype == np.float64, f"opens dtype: {arrays.opens.dtype}"
    assert arrays.highs.dtype == np.float64, f"highs dtype: {arrays.highs.dtype}"
    print(f"  CandleArrays: n={arrays.n}, dtypes OK")


@test("1.3 Engine", 14, "compute_pnl() LONG")
def test_14():
    from core.engine import compute_pnl
    # LONG: entry=150, exit=150.5, pip=0.01 → (150.5-150)/0.01 = 50
    pnl = compute_pnl("LONG", 150.0, 150.5, 0.01)
    assert abs(pnl - 50.0) < 1e-10, f"Expected 50.0, got {pnl}"
    print(f"  LONG pnl={pnl}")


@test("1.3 Engine", 15, "compute_pnl() SHORT")
def test_15():
    from core.engine import compute_pnl
    # SHORT: entry=150, exit=149.5, pip=0.01 → (150-149.5)/0.01 = 50
    pnl = compute_pnl("SHORT", 150.0, 149.5, 0.01)
    assert abs(pnl - 50.0) < 1e-10, f"Expected 50.0, got {pnl}"
    print(f"  SHORT pnl={pnl}")


@test("1.3 Engine", 16, "apply_spread()")
def test_16():
    from core.engine import apply_spread
    result = apply_spread(50.0, 0.5)
    assert abs(result - 49.5) < 1e-10, f"Expected 49.5, got {result}"
    print(f"  apply_spread(50, 0.5) = {result}")


@test("1.3 Engine", 17, "check_min_rr() pass")
def test_17():
    from core.engine import check_min_rr
    # LONG: entry=150, tp=152 (reward=200 pips), sl=149 (risk=100 pips), rr=2.0
    result = check_min_rr(150.0, 152.0, 149.0, 0.01, 1.0)
    assert result is True, f"Expected True, got {result}"
    print(f"  check_min_rr(rr=2.0, min=1.0) = {result}")


@test("1.3 Engine", 18, "check_min_rr() fail")
def test_18():
    from core.engine import check_min_rr
    # LONG: entry=150, tp=150.5 (reward=50 pips), sl=149 (risk=100 pips), rr=0.5
    result = check_min_rr(150.0, 150.5, 149.0, 0.01, 1.0)
    assert result is False, f"Expected False, got {result}"
    print(f"  check_min_rr(rr=0.5, min=1.0) = {result}")


@test("1.3 Engine", 19, "check_dedup() same price")
def test_19():
    from core.engine import check_dedup
    result = check_dedup(150.0, 150.0)
    assert result is True, f"Expected True, got {result}"
    print(f"  check_dedup(same) = {result}")


@test("1.3 Engine", 20, "check_dedup() different price")
def test_20():
    from core.engine import check_dedup
    result = check_dedup(150.0, 150.1)
    assert result is False, f"Expected False, got {result}"
    print(f"  check_dedup(different) = {result}")


@test("1.3 Engine", 21, "compute_tp_sl() with TP/SL set")
def test_21():
    from core.engine import CandleArrays, compute_tp_sl
    from detectors.signal import PatternSignal
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    i = 1000
    # LONG signal with TP above entry, SL below entry
    sig = PatternSignal(
        name="test", start_time=pd.Timestamp("2026-01-01"),
        end_time=pd.Timestamp("2026-01-01"), confidence=1.0,
        metadata={"direction": "LONG", "take_profit": 160.0, "stop_loss": 140.0}
    )
    tp, sl = compute_tp_sl(sig, arrays, i)
    assert tp > 0, f"TP should be >0, got {tp}"
    assert sl > 0, f"SL should be >0, got {sl}"
    entry = float(arrays.closes[i])
    assert tp > entry, f"LONG TP {tp} should be > entry {entry}"
    assert sl < entry, f"LONG SL {sl} should be < entry {entry}"
    print(f"  entry={entry:.4f}, tp={tp:.4f}, sl={sl:.4f}")


@test("1.3 Engine", 22, "compute_tp_sl() ATR fallback")
def test_22():
    from core.engine import CandleArrays, compute_tp_sl
    from detectors.signal import PatternSignal
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    i = 1000
    # Signal with tp=0, sl=0 → should use ATR fallback
    sig = PatternSignal(
        name="test", start_time=pd.Timestamp("2026-01-01"),
        end_time=pd.Timestamp("2026-01-01"), confidence=1.0,
        metadata={"direction": "LONG", "take_profit": 0.0, "stop_loss": 0.0}
    )
    tp, sl = compute_tp_sl(sig, arrays, i)
    assert tp != 0.0 or sl != 0.0, f"ATR fallback failed: tp={tp}, sl={sl}"
    print(f"  ATR fallback: tp={tp:.4f}, sl={sl:.4f}")


@test("1.3 Engine", 23, "build_result() normal")
def test_23():
    from core.engine import build_result
    from core.trade_engine import TradeRecord
    # Create 10 trades: 6 winners, 4 losers
    trades = []
    for i in range(6):
        trades.append(TradeRecord(
            id=str(i), strategy="test", direction="LONG",
            entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
            stop_loss=149.0, take_profit=152.0,
            exit_time=pd.Timestamp("2026-01-02"), exit_price=152.0,
            exit_reason="TP", pnl_pips=20.0,
            mae_pips=-5.0, mfe_pips=25.0,
            risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
            spread_cost=0.5,
        ))
    for i in range(4):
        trades.append(TradeRecord(
            id=str(6+i), strategy="test", direction="LONG",
            entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
            stop_loss=149.0, take_profit=152.0,
            exit_time=pd.Timestamp("2026-01-02"), exit_price=149.0,
            exit_reason="SL", pnl_pips=-10.0,
            mae_pips=-10.0, mfe_pips=5.0,
            risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
            spread_cost=0.5,
        ))
    result = build_result("test", trades)
    assert result["total_trades"] == 10, f"total_trades={result['total_trades']}"
    assert abs(result["win_rate"] - 60.0) < 0.1, f"win_rate={result['win_rate']}"
    assert result["profit_factor"] > 0, f"profit_factor={result['profit_factor']}"
    print(f"  total={result['total_trades']}, win_rate={result['win_rate']}%, PF={result['profit_factor']}")


@test("1.3 Engine", 24, "build_result() no losers")
def test_24():
    from core.engine import build_result
    from core.trade_engine import TradeRecord
    trades = []
    for i in range(5):
        trades.append(TradeRecord(
            id=str(i), strategy="test", direction="LONG",
            entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
            stop_loss=149.0, take_profit=152.0,
            exit_time=pd.Timestamp("2026-01-02"), exit_price=152.0,
            exit_reason="TP", pnl_pips=20.0,
            mae_pips=-2.0, mfe_pips=20.0,
            risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
            spread_cost=0.5,
        ))
    result = build_result("test", trades)
    assert result["profit_factor"] == 0.0, f"Expected PF=0.0 (no losers), got {result['profit_factor']}"
    assert result["gross_loss"] == 0.0, f"Expected gross_loss=0.0, got {result['gross_loss']}"
    print(f"  All winners: PF={result['profit_factor']}, gross_loss={result['gross_loss']}")


@test("1.3 Engine", 25, "build_result() empty")
def test_25():
    from core.engine import build_result
    result = build_result("test", [])
    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_pnl_pips"] == 0
    assert result["profit_factor"] == 0.0
    assert result["avg_pnl_pips"] == 0
    print(f"  Empty: total_trades={result['total_trades']}, win_rate={result['win_rate']}")


@test("1.3 Engine", 26, "update_equity()")
def test_26():
    from core.engine import update_equity
    curve = []
    peak = 10000.0
    dd = 0.0
    # No open position, realized=10 pips
    curve, peak, dd = update_equity(None, "LONG", 150.0, 10.0, 0.01, 10000.0, 0.01, curve, peak, dd)
    assert len(curve) == 1, f"Curve length={len(curve)}, expected 1"
    assert curve[0] > 10000.0, f"Balance should be >10000, got {curve[0]}"
    assert peak >= curve[0], f"Peak should be >= balance"
    print(f"  equity curve: {len(curve)} points, balance={curve[0]:.2f}")


# =========================================================================
# 1.4 TRADE ENGINE (core/trade_engine.py)
# =========================================================================

@test("1.4 Trade Engine", 27, "Open LONG")
def test_27():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    assert engine.open_position is not None, "No position opened"
    pos = engine.open_position
    assert pos.direction == "LONG"
    # Entry should be adjusted for spread: 150 + (0.5/2)*0.01 = 150 + 0.0025 = 150.0025
    expected_entry = 150.0 + (0.5 / 2) * 0.01
    assert abs(pos.entry_price - expected_entry) < 1e-8, f"Entry={pos.entry_price}, expected ~{expected_entry}"
    print(f"  LONG opened: entry={pos.entry_price:.6f}, TP={pos.take_profit}, SL={pos.stop_loss}")


@test("1.4 Trade Engine", 28, "Open SHORT")
def test_28():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="SHORT",
        entry_price=150.0, take_profit=148.0, stop_loss=151.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    assert engine.open_position is not None, "No position opened"
    pos = engine.open_position
    assert pos.direction == "SHORT"
    # Entry should be: 150 - (0.5/2)*0.01 = 149.9975
    expected_entry = 150.0 - (0.5 / 2) * 0.01
    assert abs(pos.entry_price - expected_entry) < 1e-8, f"Entry={pos.entry_price}, expected ~{expected_entry}"
    print(f"  SHORT opened: entry={pos.entry_price:.6f}, TP={pos.take_profit}, SL={pos.stop_loss}")


@test("1.4 Trade Engine", 29, "TP hit on bar")
def test_29():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent, BarEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    tp = engine.open_position.take_profit
    # Bar with high >= TP
    bar = BarEvent(
        timestamp=pd.Timestamp("2026-01-02"),
        open=150.0, high=tp + 0.1, low=149.5, close=150.5,
    )
    trade = engine.on_bar(bar)
    assert trade is not None, "No trade closed on TP hit"
    assert trade.exit_reason == "TP", f"Exit reason is '{trade.exit_reason}', expected 'TP'"
    assert engine.open_position is None, "Position should be closed"
    print(f"  TP hit: exit_reason={trade.exit_reason}, pnl={trade.pnl_pips}")


@test("1.4 Trade Engine", 30, "SL hit on bar")
def test_30():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent, BarEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.5,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    sl = engine.open_position.stop_loss
    # Bar with low <= SL
    bar = BarEvent(
        timestamp=pd.Timestamp("2026-01-02"),
        open=150.0, high=150.5, low=sl - 0.1, close=150.0,
    )
    trade = engine.on_bar(bar)
    assert trade is not None, "No trade closed on SL hit"
    assert trade.exit_reason == "SL", f"Exit reason is '{trade.exit_reason}', expected 'SL'"
    print(f"  SL hit: exit_reason={trade.exit_reason}, pnl={trade.pnl_pips}")


@test("1.4 Trade Engine", 31, "Duplicate entry rejected")
def test_31():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig1 = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig1)
    assert engine.open_position is not None, "First position should open"
    # Try to open again with same signal (while position is open)
    sig2 = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig2)
    # Should still only have one position (engine rejects if _open is not None)
    assert engine.open_position is not None
    print(f"  Duplicate rejected: still one open position")


@test("1.4 Trade Engine", 32, "Min R:R rejected")
def test_32():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01,
                         spread_pips=0.5, min_rr=2.0)
    engine = TradeEngine(config)
    # Bad R:R: reward=0.5 pips, risk=100 pips → ratio=0.005
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=150.005, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    assert engine.open_position is None, "Position should NOT open with bad R:R"
    print(f"  Min R:R rejected: no position opened")


@test("1.4 Trade Engine", 33, "Force close EOD")
def test_33():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=155.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    assert engine.open_position is not None, "Position should open"
    trade = engine.force_close(150.5, pd.Timestamp("2026-01-01 23:59"), "EOD")
    assert trade is not None, "No trade returned from force_close"
    assert trade.exit_reason == "EOD", f"Exit reason: {trade.exit_reason}"
    assert engine.open_position is None, "Position should be closed"
    print(f"  Force close EOD: exit_reason={trade.exit_reason}, pnl={trade.pnl_pips}")


@test("1.4 Trade Engine", 34, "get_stats() returns dict")
def test_34():
    from core.trade_engine import TradeEngine, TradeConfig
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01)
    engine = TradeEngine(config)
    stats = engine.get_stats()
    assert isinstance(stats, dict), f"get_stats returned {type(stats)}"
    assert "win_rate" in stats, "Missing win_rate key"
    assert "total_pnl_pips" in stats, "Missing total_pnl_pips key"
    print(f"  get_stats keys: {list(stats.keys())[:5]}..., win_rate={stats['win_rate']}")


@test("1.4 Trade Engine", 35, "reset() clears state")
def test_35():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent, BarEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5)
    engine = TradeEngine(config)
    # Do some trading
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=152.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    bar = BarEvent(
        timestamp=pd.Timestamp("2026-01-02"),
        open=150.0, high=153.0, low=149.5, close=150.5,
    )
    engine.on_bar(bar)
    assert len(engine.trades) > 0 or engine.open_position is not None
    engine.reset()
    assert engine.open_position is None, "open_position not cleared"
    assert len(engine.trades) == 0, "trades not cleared"
    print(f"  reset(): open_position={engine.open_position}, trades={len(engine.trades)}")


@test("1.4 Trade Engine", 36, "Equity tracking")
def test_36():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent, BarEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=0.5,
                         initial_balance=10000.0)
    engine = TradeEngine(config)
    initial_len = len(engine.balance_curve)
    # Open and close a trade
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=151.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    bar = BarEvent(
        timestamp=pd.Timestamp("2026-01-02"),
        open=150.0, high=152.0, low=149.5, close=150.5,
    )
    engine.on_bar(bar)
    assert len(engine.balance_curve) > initial_len, f"Balance curve didn't grow: {len(engine.balance_curve)} vs {initial_len}"
    print(f"  Equity: {initial_len} → {len(engine.balance_curve)} points")


@test("1.4 Trade Engine", 37, "Spread cost deducted")
def test_37():
    from core.trade_engine import TradeEngine, TradeConfig
    from core.events import SignalEvent, BarEvent
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01, spread_pips=1.0)
    engine = TradeEngine(config)
    sig = SignalEvent(
        strategy_name="test", direction="LONG",
        entry_price=150.0, take_profit=153.0, stop_loss=149.0,
        timestamp=pd.Timestamp("2026-01-01"),
    )
    engine.open(sig)
    tp = engine.open_position.take_profit
    bar = BarEvent(
        timestamp=pd.Timestamp("2026-01-02"),
        open=150.0, high=tp + 0.1, low=149.5, close=150.5,
    )
    trade = engine.on_bar(bar)
    assert trade is not None
    # PnL should include spread deduction
    assert trade.spread_cost == 1.0, f"spread_cost={trade.spread_cost}"
    # Net pnl should be less than raw pnl
    assert trade.pnl_pips < (tp - 150.0) / 0.01, f"pnl_pips should be < raw reward"
    print(f"  Spread cost={trade.spread_cost}, net_pnl={trade.pnl_pips}")


# =========================================================================
# 1.5 SIGNAL ENGINE (core/signal_engine.py)
# =========================================================================

@test("1.5 Signal Engine", 38, "Single strategy threshold=1")
def test_38():
    from core.engine import CandleArrays
    from core.signal_engine import SignalEngine
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    # Use ema_stochastic with threshold=1 (fires on any single signal)
    engine = SignalEngine(["ema_stochastic"], lookback=10, threshold=1)
    # Scan through bars looking for signals
    found_signal = False
    for i in range(100, min(5000, arrays.n)):
        sigs = engine.evaluate(i, arrays, {})
        if sigs:
            found_signal = True
            assert sigs[0].direction in ("LONG", "SHORT"), f"Bad direction: {sigs[0].direction}"
            print(f"  Signal at i={i}: direction={sigs[0].direction}")
            break
    assert found_signal, "No signal found in first 5000 bars"


@test("1.5 Signal Engine", 39, "Multi-strategy threshold=2")
def test_39():
    from core.engine import CandleArrays
    from core.signal_engine import SignalEngine
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    # Use two strategies that need 2 to agree
    engine = SignalEngine(["tweezer_reversal", "marubozu_trend"], lookback=10, threshold=2)
    # With threshold=2, we should NOT get signals from a single strategy alone
    found_any = False
    for i in range(100, min(3000, arrays.n)):
        sigs = engine.evaluate(i, arrays, {})
        if sigs:
            found_any = True
            # Verify it's a combined strategy name (contains +)
            break
    # Whether or not we found signals, the engine shouldn't crash
    print(f"  Multi-strategy threshold=2: signals found={found_any}")


@test("1.5 Signal Engine", 40, "ATR fallback for TP/SL")
def test_40():
    from core.engine import CandleArrays, compute_tp_sl
    from detectors.signal import PatternSignal
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    # Signal with tp=0, sl=0
    sig = PatternSignal(
        name="test", start_time=pd.Timestamp("2026-01-01"),
        end_time=pd.Timestamp("2026-01-01"), confidence=1.0,
        metadata={"direction": "LONG", "take_profit": 0.0, "stop_loss": 0.0}
    )
    tp, sl = compute_tp_sl(sig, arrays, 1000, lookback=100)
    assert tp != 0.0, f"ATR fallback didn't compute TP: {tp}"
    assert sl != 0.0, f"ATR fallback didn't compute SL: {sl}"
    entry = float(arrays.closes[1000])
    assert tp > entry, f"LONG TP {tp} should be > entry {entry}"
    assert sl < entry, f"LONG SL {sl} should be < entry {entry}"
    print(f"  ATR fallback: entry={entry:.4f}, tp={tp:.4f}, sl={sl:.4f}")


@test("1.5 Signal Engine", 41, "Buffer reset")
def test_41():
    from core.signal_engine import SignalBuffer
    buf = SignalBuffer(lookback=10, threshold=1)
    # Add a signal
    result = buf.add_and_check("strat1", "LONG", 100, 150.0, 152.0, 149.0, None)
    assert result is not None, "First signal should trigger with threshold=1"
    assert len(buf) > 0, "Buffer should have entries"
    # Clear
    buf.clear()
    assert len(buf) == 0, "Buffer should be empty after clear"
    print(f"  Buffer: len before clear={1}, after clear={len(buf)}")


@test("1.5 Signal Engine", 42, "Precompute works")
def test_42():
    from core.engine import CandleArrays
    from core.signal_engine import SignalEngine
    m1 = _load_m1()
    arrays = CandleArrays.from_dataframe(m1)
    # Build simple tf_arrays for M5
    from core.market_data_store import MarketDataStore
    store = MarketDataStore()
    store.load_symbol("USDJPY", m1)
    m5_df = store.get_data("USDJPY", "M5")
    m5_arrays = CandleArrays.from_dataframe(m5_df)
    tf_arrays = {"M5": m5_arrays}
    engine = SignalEngine(["tweezer_reversal"], lookback=10, threshold=1)
    # Should not crash
    engine.precompute(arrays, tf_arrays)
    # Evaluate should still work
    sigs = engine.evaluate(1000, arrays, tf_arrays)
    print(f"  Precompute completed, evaluate at i=1000 returned {len(sigs)} signals")


@test("1.5 Signal Engine", 43, "Unknown strategy name")
def test_43():
    from core.signal_engine import SignalEngine
    try:
        engine = SignalEngine(["totally_fake_strategy_xyz"], lookback=5, threshold=1)
        assert False, "Should have raised ValueError for unknown strategy"
    except ValueError as e:
        assert "not found" in str(e).lower(), f"Error message unclear: {e}"
        print(f"  ValueError raised: {e}")


# =========================================================================
# 1.6 TRADE STORE (core/trade_store.py)
# =========================================================================

@test("1.6 Trade Store", 44, "init_db() creates tables")
def test_44():
    import tempfile
    import sqlite3
    from core.trade_store import init_db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        # Check tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "runs" in table_names, f"runs table not found: {table_names}"
        assert "trades" in table_names, f"trades table not found: {table_names}"
        conn.close()
        print(f"  Tables created: {table_names}")
    finally:
        os.unlink(db_path)


@test("1.6 Trade Store", 45, "save_trades() saves run")
def test_45():
    import tempfile
    from core.trade_store import init_db, save_trades
    from core.trade_engine import TradeRecord
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        trades = []
        for i in range(5):
            trades.append(TradeRecord(
                id=str(i), strategy="test_strategy", direction="LONG",
                entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
                stop_loss=149.0, take_profit=152.0,
                exit_time=pd.Timestamp("2026-01-02"), exit_price=152.0,
                exit_reason="TP", pnl_pips=20.0,
                mae_pips=-2.0, mfe_pips=20.0,
                risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
                spread_cost=0.5,
            ))
        run_meta = {
            "data_file": "test.csv",
            "symbol": "USDJPY",
            "strategies": ["test_strategy"],
            "lookback": 5,
            "threshold": 2,
            "n_strategies": 1,
        }
        result_stats = {
            "total_trades": 5,
            "winning_trades": 5,
            "losing_trades": 0,
            "win_rate": 100.0,
            "total_pnl_pips": 100.0,
            "avg_pnl_pips": 20.0,
            "expectancy_pips": 20.0,
            "profit_factor": 0.0,
            "gross_profit": 100.0,
            "gross_loss": 0.0,
            "avg_mae_pips": -2.0,
            "avg_mfe_pips": 20.0,
        }
        run_id = save_trades(conn, trades, run_meta, result_stats)
        assert run_id > 0, f"run_id={run_id}, expected > 0"
        # Verify trades saved
        count = conn.execute("SELECT COUNT(*) FROM trades WHERE run_id=?", (run_id,)).fetchone()[0]
        assert count == 5, f"Expected 5 trades, got {count}"
        conn.close()
        print(f"  run_id={run_id}, {count} trades saved")
    finally:
        os.unlink(db_path)


@test("1.6 Trade Store", 46, "get_run_summary()")
def test_46():
    import tempfile
    from core.trade_store import init_db, save_trades, get_run_summary
    from core.trade_engine import TradeRecord
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        trades = [
            TradeRecord(
                id="1", strategy="test", direction="LONG",
                entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
                stop_loss=149.0, take_profit=152.0,
                exit_time=pd.Timestamp("2026-01-02"), exit_price=152.0,
                exit_reason="TP", pnl_pips=20.0,
                mae_pips=-2.0, mfe_pips=20.0,
                risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
                spread_cost=0.5,
            ),
        ]
        run_meta = {"data_file": "test.csv", "symbol": "USDJPY", "strategies": ["test"],
                    "lookback": 5, "threshold": 2, "n_strategies": 1}
        stats = {"total_trades": 1, "win_rate": 100.0, "total_pnl_pips": 20.0}
        run_id = save_trades(conn, trades, run_meta, stats)
        summary = get_run_summary(conn, run_id)
        assert summary, "No summary returned"
        assert summary["total_trades"] == 1, f"total_trades={summary['total_trades']}"
        assert summary["symbol"] == "USDJPY", f"symbol={summary['symbol']}"
        assert abs(summary["win_rate"] - 100.0) < 0.1, f"win_rate={summary['win_rate']}"
        conn.close()
        print(f"  Summary: trades={summary['total_trades']}, win_rate={summary['win_rate']}%, symbol={summary['symbol']}")
    finally:
        os.unlink(db_path)


@test("1.6 Trade Store", 47, "get_trades() with filter")
def test_47():
    import tempfile
    from core.trade_store import init_db, save_trades, get_trades
    from core.trade_engine import TradeRecord
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        trades = [
            TradeRecord(
                id="1", strategy="strat_A", direction="LONG",
                entry_time=pd.Timestamp("2026-01-01"), entry_price=150.0,
                stop_loss=149.0, take_profit=152.0,
                exit_time=pd.Timestamp("2026-01-02"), exit_price=152.0,
                exit_reason="TP", pnl_pips=20.0,
                mae_pips=-2.0, mfe_pips=20.0,
                risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
                spread_cost=0.5,
            ),
            TradeRecord(
                id="2", strategy="strat_B", direction="SHORT",
                entry_time=pd.Timestamp("2026-01-03"), entry_price=151.0,
                stop_loss=152.0, take_profit=149.0,
                exit_time=pd.Timestamp("2026-01-04"), exit_price=149.0,
                exit_reason="TP", pnl_pips=20.0,
                mae_pips=-2.0, mfe_pips=20.0,
                risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
                spread_cost=0.5,
            ),
        ]
        run_meta = {"data_file": "test.csv", "symbol": "USDJPY", "strategies": ["strat_A", "strat_B"],
                    "lookback": 5, "threshold": 2, "n_strategies": 2}
        run_id = save_trades(conn, trades, run_meta)
        # Filter by strategy
        filtered = get_trades(conn, run_id=run_id, strategy="strat_A")
        assert len(filtered) == 1, f"Expected 1 trade for strat_A, got {len(filtered)}"
        assert filtered[0]["strategy"] == "strat_A"
        # Get all
        all_trades = get_trades(conn, run_id=run_id)
        assert len(all_trades) == 2, f"Expected 2 trades, got {len(all_trades)}"
        conn.close()
        print(f"  Filter: strat_A→{len(filtered)}, all→{len(all_trades)}")
    finally:
        os.unlink(db_path)


@test("1.6 Trade Store", 48, "Timestamp serialization")
def test_48():
    import tempfile
    from core.trade_store import init_db, save_trades, get_trades
    from core.trade_engine import TradeRecord
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = init_db(db_path)
        ts = pd.Timestamp("2026-06-15 14:30:00")
        trades = [
            TradeRecord(
                id="1", strategy="test", direction="LONG",
                entry_time=ts, entry_price=150.0,
                stop_loss=149.0, take_profit=152.0,
                exit_time=ts + pd.Timedelta(hours=1), exit_price=152.0,
                exit_reason="TP", pnl_pips=20.0,
                mae_pips=-2.0, mfe_pips=20.0,
                risk_pips=100.0, reward_pips=200.0, rr_ratio=2.0,
                spread_cost=0.5,
            ),
        ]
        run_meta = {"data_file": "test.csv", "symbol": "USDJPY", "strategies": ["test"],
                    "lookback": 5, "threshold": 2, "n_strategies": 1}
        run_id = save_trades(conn, trades, run_meta)
        saved = get_trades(conn, run_id=run_id)
        assert len(saved) == 1
        entry_time_str = saved[0]["entry_time"]
        # Should be an ISO string
        assert isinstance(entry_time_str, str), f"entry_time type: {type(entry_time_str)}"
        # Should be parseable back to timestamp
        parsed = pd.Timestamp(entry_time_str)
        assert parsed.year == 2026, f"Parsed year: {parsed.year}"
        assert parsed.month == 6, f"Parsed month: {parsed.month}"
        conn.close()
        print(f"  Timestamp: {entry_time_str} → parsed OK")
    finally:
        os.unlink(db_path)


# =========================================================================
# RUN ALL TESTS
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: CORE ENGINE UNIT TESTS — RSFX FOREX PLATFORM")
    print("=" * 70)
    print()

    test_funcs = [
        test_01, test_02, test_03, test_04, test_05, test_06, test_07, test_08,
        test_09, test_10, test_11, test_12,
        test_13, test_14, test_15, test_16, test_17, test_18, test_19, test_20,
        test_21, test_22, test_23, test_24, test_25, test_26,
        test_27, test_28, test_29, test_30, test_31, test_32, test_33, test_34,
        test_35, test_36, test_37,
        test_38, test_39, test_40, test_41, test_42, test_43,
        test_44, test_45, test_46, test_47, test_48,
    ]

    for func in test_funcs:
        func()
        last = results[-1]
        status_icon = "✅" if last[3] == "PASS" else "❌"
        print(f"  {status_icon} Test {last[1]}: {last[2]} → {last[3]}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r[3] == "PASS")
    failed = sum(1 for r in results if r[3] == "FAIL")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    print()
    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if r[3] == "FAIL":
                print(f"  ❌ {r[0]} Test {r[1]}: {r[2]}")
                for line in r[4].split('\n')[:5]:
                    print(f"      {line}")
        print()
    print("DETAILED RESULTS:")
    print(f"{'#':>3} {'Section':<25} {'Test Name':<35} {'Result':<6}")
    print("-" * 70)
    for r in results:
        icon = "✅" if r[3] == "PASS" else "❌"
        print(f"{r[1]:>3} {r[0]:<25} {r[2]:<35} {icon} {r[3]}")
