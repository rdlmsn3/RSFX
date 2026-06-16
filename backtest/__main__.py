"""
RSFX Backtest Module — unified entry point.

Usage:
    python3 -m backtest run -s tweezer_reversal,h1_trend_m5_rsi
    python3 -m backtest correlation
    python3 -m backtest portfolio
    python3 -m backtest ui
"""
import sys

COMMANDS = {
    "run":          "ui.cli:main",
    "correlation":  "backtest.correlation:main",
    "portfolio":    "backtest.portfolio:main",
    "ui":           "backtest.ui.server:main",
}

HELP = """
RSFX Backtest Module
====================

Commands:
  run           Unified backtest (SignalEngine + TradeEngine)
  correlation   Strategy correlation analysis
  portfolio     Portfolio optimizer (2+3 strategy combos)
  ui            Web UI for backtester (port 8502)

Examples:
  python3 -m backtest run -s tweezer_reversal,h1_trend_m5_rsi --lookback 5
  python3 -m backtest correlation
  python3 -m backtest portfolio --max-combo 3
  python3 -m backtest ui

Each command accepts --help for detailed options.
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    # Shift argv: remove 'backtest' prefix, keep rest for the subcommand
    sys.argv = [f"python3 -m backtest {cmd}"] + sys.argv[2:]

    module_path, func_name = COMMANDS[cmd].split(":")
    mod = __import__(module_path, fromlist=[func_name])
    func = getattr(mod, func_name)
    func()


if __name__ == "__main__":
    main()
