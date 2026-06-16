# RSFX Post-Refactor Testing Plan

Major refactor complete. 3 UIs unified around core engine.  
Test everything systematically — no component skipped.

---

## Phase 1: Core Engine (unit-level)

### 1.1 Data Loader — `core/data_loader.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | CSV M1 bars load | `DAT_ASCII_USDJPY_M1_202605.csv` | 29,658 rows, DatetimeIndex, OHLCV columns |
| 2 | Parquet tick load | `usdjpy_tick_2024-01_to_2024-03.parquet` | DataFrame with bid/ask, raw_ticks not None |
| 3 | `get_adapter()` factory — CSV | `.csv` path | Returns `HistDataAdapter` |
| 4 | `get_adapter()` factory — Parquet | `.parquet` path | Returns `ParquetAdapter` |
| 5 | `get_adapter()` factory — `.pq` | `.pq` path | Returns `ParquetAdapter` |
| 6 | Tick → M1 conversion | Tick parquet | M1 DataFrame + raw_ticks stored |
| 7 | Epoch millis datetime | Parquet with `Timestamp_ms` | DatetimeIndex parsed correctly |
| 8 | Missing file error | Non-existent path | Clear error, no crash |

### 1.2 Market Data Store — `core/market_data_store.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 9 | Load M1 data | 29K M1 candles | `available_timeframes()` → `["M1"]` |
| 10 | Multi-TF resample | M1 data | M5, H1, D1 all present |
| 11 | `get_data()` returns correct TF | `"M5"` | DataFrame with M5 frequency |
| 12 | Unknown symbol | `"EURUSD"` not loaded | Empty/exception |

### 1.3 Engine — `core/engine.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 13 | `CandleArrays.from_dataframe()` | M1 DataFrame | All arrays correct length, types |
| 14 | `compute_pnl()` LONG | entry=150, exit=150.5, pip=0.01 | +50 pips |
| 15 | `compute_pnl()` SHORT | entry=150, exit=149.5, pip=0.01 | +50 pips |
| 16 | `apply_spread()` | pnl=50, spread=0.5 | 49.5 |
| 17 | `check_min_rr()` pass | rr=2.0, min=1.0 | True |
| 18 | `check_min_rr()` fail | rr=0.5, min=1.0 | False |
| 19 | `check_dedup()` same price | same entry | True |
| 20 | `check_dedup()` different | different entry | False |
| 21 | `compute_tp_sl()` with TP/SL | signal has tp/sl | Returns them (sanity-fixed) |
| 22 | `compute_tp_sl()` ATR fallback | signal has tp=0 | ATR-based TP/SL |
| 23 | `build_result()` normal | 10 trades, 6 winners | win_rate=60, PF correct |
| 24 | `build_result()` no losers | all winners | profit_factor=0.0 (not inf) |
| 25 | `build_result()` empty | 0 trades | All stats 0, no crash |
| 26 | `update_equity()` | Simulated balance | Curve appended, peak/DD updated |

### 1.4 Trade Engine — `core/trade_engine.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 27 | Open LONG position | Valid signal | Position opened, entry adjusted for spread |
| 28 | Open SHORT position | Valid signal | Position opened |
| 29 | Reject TP hit on same bar | TP=high | Trade closed with reason="TP" |
| 30 | Reject SL hit on same bar | SL=low | Trade closed with reason="SL" |
| 31 | Reject duplicate entry | Same price twice | Second ignored |
| 32 | Reject min R:R | risk > reward | Position not opened |
| 33 | Force close (EOD) | Open position | Closed with reason="EOD" |
| 34 | `get_stats()` returns dict | After trades | Keys: win_rate, total_pnl_pips, etc. |
| 35 | `reset()` clears state | After trades | No open position, empty trades |
| 36 | Equity tracking | Multiple trades | Balance curve grows, DD updates |
| 37 | Spread cost deducted | Trade closed | pnl_pips includes spread deduction |

### 1.5 Signal Engine — `core/signal_engine.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 38 | Single strategy, threshold=1 | tweezer_reversal | Signals returned on pattern bars |
| 39 | Multi-strategy, threshold=2 | 2 strategies | Only fires when 2 agree within lookback |
| 40 | ATR fallback for TP/SL | Strategy returns tp=0 | compute_tp_sl fills in ATR-based values |
| 41 | Buffer reset | After run | Old signals don't carry over |
| 42 | Precompute works | Call precompute() | No crash, strategies have cached data |
| 43 | Unknown strategy name | `"nonexistent"` | Graceful skip, no crash |

### 1.6 Trade Store — `core/trade_store.py`
| # | Test | Input | Expected |
|---|------|-------|----------|
| 44 | `init_db()` creates tables | Fresh DB | runs + trades tables exist |
| 45 | `save_trades()` saves run | 5 trades | run_id returned, 5 rows in trades table |
| 46 | `get_run_summary()` | Saved run | All stats match what was saved |
| 47 | `get_trades()` with filter | filter by strategy | Only matching trades returned |
| 48 | Timestamp serialization | pd.Timestamp in trades | Stored as ISO string |

---

## Phase 2: Backtest Module (CLI)

### 2.1 CLI Backtester — `python3 -m backtest run`
| # | Test | Command | Expected |
|---|------|---------|----------|
| 49 | Single strategy | `run -s tweezer_reversal` | CSV output in results/ |
| 50 | Multiple strategies | `run -s tweezer_reversal,cci_ema` | Both strategies in output |
| 51 | Partial name match | `run -s h1_trend` | All H1 trend strategies matched |
| 52 | All strategies | `run` (no -s) | All 72 strategies |
| 53 | Custom CSV | `run --csv data/sample.csv` | Loads sample, runs |
| 54 | S/R toggle | `run -s h1_trend_m5_rsi --use-sr` | S/R override applied |
| 55 | Custom params | `run -s cci_ema --spread 1.0 --min-rr 2.0` | Params applied |
| 56 | Output CSV exists | After run | `results/bt_*.csv` created |
| 57 | SQLite saved | After run | `results/trades.db` has new run |
| 58 | `--no-save` flag | `run -s cci_ema --no-save` | No SQLite entry |

### 2.2 Correlation — `python3 -m backtest correlation`
| # | Test | Command | Expected |
|---|------|---------|----------|
| 59 | Run with existing results | `correlation` | Uses last bt_*.csv |
| 60 | Output file | After run | `results/corr_*.csv` created |

### 2.3 Portfolio — `python3 -m backtest portfolio`
| # | Test | Command | Expected |
|---|------|---------|----------|
| 61 | 2+3 combos | `portfolio` | Results sorted by Sharpe |
| 62 | Output file | After run | `results/portfolio_results.csv` created |

### 2.4 Buckets — `python3 -m backtest` bucket operations
| # | Test | Command | Expected |
|---|------|---------|----------|
| 63 | Load bucket | Via UI/API | Config restored correctly |
| 64 | Save bucket | Via UI/API | JSON file in buckets/ |

---

## Phase 3: FastAPI UI — `ui/backtest/server.py`

### 3.1 Endpoints
| # | Test | Endpoint | Expected |
|---|------|----------|----------|
| 65 | GET / serves HTML | `GET /` | 200, HTML content |
| 66 | GET /strategies | `GET /strategies` | JSON, 72 strategies |
| 67 | GET /files | `GET /files` | JSON, lists CSV + Parquet files |
| 68 | GET /buckets | `GET /buckets` | JSON, list of saved buckets |
| 69 | POST /run — single strategy | Valid request | JSON with result + trades |
| 70 | POST /run — multi strategy | 2+ strategies | Confluence results |
| 71 | POST /run — invalid strategy | `"nonexistent"` | Error message |
| 72 | POST /run — threshold > count | threshold=5, 2 strategies | Error message |
| 73 | POST /run — JSON serializable | Any valid run | No `Infinity`, no `NaN`, no numpy types |
| 74 | POST /run — SQLite saved | After run | run_id in response, row in DB |
| 75 | POST /save-bucket | Valid config | Bucket JSON created |
| 76 | GET /load-bucket/{name} | Existing bucket | Full config returned |
| 77 | GET /progress | Any time | `{"running": false, "pct": 100}` |

### 3.2 Edge Cases
| # | Test | Input | Expected |
|---|------|-------|----------|
| 78 | Empty strategies list | `[]` | Error or graceful handling |
| 79 | Very large dataset | Full 2008 parquet | Completes without timeout |
| 80 | Concurrent requests | 2x POST /run | Both complete (or queued) |

---

## Phase 4: Streamlit UI — `ui/streamlit_app/app.py`

### 4.1 Smoke Tests
| # | Test | Action | Expected |
|---|------|--------|----------|
| 81 | App loads | `streamlit run ui/streamlit_app/app.py` | No import errors |
| 82 | Default CSV loads | Auto-loads USDJPY M1 | Charts render |
| 83 | File selector | Pick different CSV | Data reloads |
| 84 | Strategy selector | Pick strategies | Checkboxes work |
| 85 | Timeframe tabs | M1/M5/H1/D1 | Each shows correct chart |
| 86 | Playback controls | Play/Pause/Speed | Candle advances |
| 87 | Equity curve | After backtest | Balance line visible |
| 88 | Trade table | After backtest | Rows with entry/exit/PnL |

---

## Phase 5: Integration (end-to-end)

### 5.1 Full Pipeline
| # | Test | Flow | Expected |
|---|------|------|----------|
| 89 | CSV → backtest → SQLite | CLI run with 1 strategy | Trades in trades.db |
| 90 | Parquet → backtest → SQLite | CLI run with parquet | Same result quality |
| 91 | FastAPI run → SQLite → verify | POST /run, then query DB | run_id matches |
| 92 | CLI result → correlation | Run bt, then correlation | Correlation uses bt output |
| 93 | Cross-UI consistency | Same params on CLI + FastAPI | Same trade count (±spread timing) |

### 5.2 Regression Guards
| # | Test | Check | Expected |
|---|------|-------|----------|
| 94 | No stale imports | `grep -rn "from views\.\|from backtest\.engine\|from backtest\.ui"` | Zero matches |
| 95 | All files compile | Import every .py in core/, ui/, backtest/ | No ModuleNotFoundError |
| 96 | No inf/nan in JSON | Run any backtest, inspect response | All values finite |
| 97 | Timestamps as strings | Check TradeRecord.to_dict() | entry_time, exit_time are str |

---

## Execution Order

```
Phase 1 (Core)        → 48 tests   — run first, catches logic bugs
Phase 2 (CLI)         → 10 tests   — validates CLI + backtest module
Phase 3 (FastAPI)     → 16 tests   — validates web UI + JSON
Phase 4 (Streamlit)   → 8 tests    — manual smoke tests
Phase 5 (Integration) → 9 tests    — end-to-end validation
─────────────────────────────────
Total:                 91 tests
```

## How to Run

```bash
cd /home/rudi/RSFX

# Phase 1: Core (automated)
/usr/bin/python3 -c "
import sys; sys.path.insert(0, '.')
# Run each test inline or via pytest
"

# Phase 2: CLI
python3 -m backtest run -s tweezer_reversal
python3 -m backtest run -s tweezer_reversal,cci_ema --csv data/sample.csv
python3 -m backtest correlation
python3 -m backtest portfolio --max-combo 3

# Phase 3: FastAPI (server must be running)
curl http://localhost:8502/strategies
curl http://localhost:8502/files
curl -X POST http://localhost:8502/run -H "Content-Type: application/json" \
  -d '{"strategies":["tweezer_reversal"],"lookback":5,"threshold":1,...}'

# Phase 4: Streamlit (manual)
streamlit run ui/streamlit_app/app.py

# Phase 5: Integration
# Verify SQLite after each run
sqlite3 results/trades.db "SELECT COUNT(*) FROM runs; SELECT COUNT(*) FROM trades;"
```
