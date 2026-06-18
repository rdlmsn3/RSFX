"""
strategy_insights.py
-------------------
Practical strategy analysis with actionable insights.
Analyzes ALL runs in the database to find the best strategies and combinations.
"""

import sqlite3
import argparse
import os
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from datetime import datetime
import sys


class StrategyInsights:
    def __init__(self, db_path: str, output_dir: str = None):
        """Initialize with database connection."""
        self.db_path = self._resolve_db_path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.trades = None
        self.runs = None
        self.strategies = []
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "analysis_results"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this analysis run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _resolve_db_path(self, db_path: str) -> str:
        """Resolve database path with multiple fallback options."""
        path = Path(db_path)
        
        if path.is_absolute():
            if path.exists():
                return str(path)
            else:
                raise FileNotFoundError(f"Database not found: {path}")
        
        search_paths = [
            path,
            Path.cwd() / path,
            Path(__file__).parent / path,
            Path(__file__).parent / "data" / path,
            Path(__file__).parent.parent / path,
            Path.home() / "Documents" / path,
            Path.home() / "data" / path,
            Path("/data") / path,
        ]
        
        if not path.suffix:
            for ext in ['.db', '.sqlite', '.sqlite3']:
                search_paths.append(path.with_suffix(ext))
        
        for search_path in search_paths:
            if search_path.exists():
                return str(search_path)
        
        print(f"\n❌ ERROR: Database '{db_path}' not found")
        print("\nSearched in:")
        for search_path in search_paths[:5]:
            print(f"  • {search_path}")
        
        raise FileNotFoundError(f"Could not find database: {db_path}")
    
    def _parse_datetime(self, series):
        """Parse datetime from TEXT format 'YYYY-MM-DD HH:MM:SS'"""
        try:
            return pd.to_datetime(series, format='%Y-%m-%d %H:%M:%S', errors='raise')
        except:
            pass
        
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(series, format=fmt, errors='raise')
            except:
                continue
        
        return pd.to_datetime(series, errors='coerce')
    
    def _save_csv(self, df, filename):
        """Save DataFrame to CSV with timestamp."""
        if df is not None and not df.empty:
            filepath = self.output_dir / f"{self.timestamp}_{filename}"
            df.to_csv(filepath)
            print(f"   💾 Saved: {filepath}")
            return str(filepath)
        return None
    
    def list_runs(self):
        """List all available runs in the database."""
        query = """
        SELECT 
            id, 
            timestamp, 
            symbol, 
            strategies,
            n_strategies,
            total_trades,
            win_rate,
            total_pnl_pips,
            profit_factor
        FROM runs 
        ORDER BY timestamp DESC
        LIMIT 30
        """
        runs = pd.read_sql_query(query, self.conn)
        
        if runs.empty:
            print("\n❌ No runs found in database")
            return None
        
        print("\n📋 AVAILABLE RUNS IN DATABASE")
        print("=" * 90)
        print(runs.to_string(index=False))
        print("-" * 90)
        print(f"Total runs: {len(runs)}")
        
        # Save to CSV
        self._save_csv(runs, "available_runs.csv")
        
        return runs
    
    def load_all_data(self, min_trades: int = 10):
        """Load ALL trades and runs from the database."""
        print("\n📊 Loading ALL runs from database...")
        
        # Load all runs
        self.runs = pd.read_sql_query(
            "SELECT * FROM runs ORDER BY timestamp DESC", 
            self.conn
        )
        
        if self.runs.empty:
            raise ValueError("No runs found in database")
        
        # Load all trades
        query = """
        SELECT 
            t.strategy,
            t.entry_time,
            t.exit_time,
            t.pnl_pips,
            t.direction,
            r.id as run_id,
            r.symbol,
            r.timestamp as run_timestamp
        FROM trades t
        JOIN runs r ON t.run_id = r.id
        """
        self.trades = pd.read_sql_query(query, self.conn)
        
        # Parse datetime fields
        self.trades['entry_time'] = self._parse_datetime(self.trades['entry_time'])
        self.trades['exit_time'] = self._parse_datetime(self.trades['exit_time'])
        
        # Remove any rows with invalid dates
        before = len(self.trades)
        self.trades = self.trades.dropna(subset=['entry_time', 'exit_time'])
        after = len(self.trades)
        
        if before != after:
            print(f"⚠️  Removed {before - after} rows with invalid dates")
        
        self.strategies = sorted(self.trades['strategy'].unique())
        
        print(f"📊 Loaded {len(self.trades):,} trades from {len(self.strategies)} strategies")
        print(f"📈 Loaded {len(self.runs)} runs")
        print(f"💾 Database: {self.db_path}")
        print(f"📁 Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Show run summary
        print("\n📋 RUN SUMMARY:")
        run_summary = self.runs.groupby('strategies').agg({
            'id': 'count',
            'total_pnl_pips': 'sum',
            'win_rate': 'mean'
        }).round(2)
        run_summary.columns = ['Runs', 'Total PnL', 'Avg Win Rate']
        print(run_summary.to_string())
        print("-" * 60)
        
        # Save run summary
        self._save_csv(run_summary.reset_index(), "run_summary.csv")
        
    def get_strategy_summary(self):
        """Show summary of all strategies across all runs."""
        summary = self.trades.groupby('strategy').agg({
            'pnl_pips': ['count', 'sum', 'mean'],
            'direction': lambda x: (x == 'LONG').sum() if 'direction' in self.trades.columns else 0
        }).round(2)
        summary.columns = ['Trades', 'Total PnL', 'Avg PnL', 'Long Trades']
        summary['Win Rate'] = (
            self.trades.groupby('strategy')['pnl_pips']
            .apply(lambda x: (x > 0).mean() * 100)
            .round(1)
        )
        summary = summary.sort_values('Total PnL', ascending=False)
        
        print("\n📊 STRATEGY SUMMARY (All Runs)")
        print("=" * 60)
        print(summary.to_string())
        
        # Save to CSV
        self._save_csv(summary.reset_index(), "strategy_summary.csv")
        
        return summary
    
    def get_top_performers(self, top_n: int = 10):
        """Show best performing strategies."""
        summary = self.trades.groupby('strategy').agg({
            'pnl_pips': ['count', 'sum', 'mean'],
        }).round(2)
        summary.columns = ['Trades', 'Total PnL', 'Avg PnL']
        summary['Win Rate'] = (
            self.trades.groupby('strategy')['pnl_pips']
            .apply(lambda x: (x > 0).mean() * 100)
            .round(1)
        )
        
        top = summary.sort_values('Total PnL', ascending=False).head(top_n)
        
        print("\n🏆 TOP PERFORMING STRATEGIES")
        print("=" * 60)
        print(top.to_string())
        
        # Save to CSV
        self._save_csv(top.reset_index(), f"top_{top_n}_performers.csv")
        
        return top
    
    def analyze_correlation(self, min_trades: int = 10):
        """Analyze strategy correlations across all runs based on hourly return distributions."""
        df = self.trades.copy()
        
        # Use exit_time resampled hourly to track performance periods accurately
        print("📊 Building hourly return matrix for core correlation metrics...")
        hourly_pnl = df.groupby(['strategy', pd.Grouper(key='exit_time', freq='1h')])['pnl_pips'].sum().unstack(level=0)
        
        # Filter columns to only include strategies with enough data
        trade_counts = self.trades.groupby('strategy').size()
        valid_strategies = trade_counts[trade_counts >= min_trades].index
        
        # Fill missing periods with 0 (no trades taken in that hour) to prevent alignment errors
        hourly_pnl = hourly_pnl[valid_strategies].fillna(0)
        
        if len(valid_strategies) < 2:
            print("\n⚠️  Need at least 2 strategies with enough trades for correlation analysis")
            return None
        
        # Calculate cross-correlation across strategy performance rows
        corr_matrix = hourly_pnl.corr()
        
        # Find redundant pairs (high correlation)
        redundant_pairs = []
        diverse_pairs = []
        
        for i, s1 in enumerate(corr_matrix.columns):
            for s2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[s1, s2]
                if not pd.isna(corr):
                    if corr > 0.7:
                        redundant_pairs.append({
                            'Strategy 1': s1,
                            'Strategy 2': s2,
                            'Correlation': round(corr, 3)
                        })
                    elif corr < 0.3:
                        diverse_pairs.append({
                            'Strategy 1': s1,
                            'Strategy 2': s2,
                            'Correlation': round(corr, 3)
                        })
        
        print("\n🔄 STRATEGY CORRELATION INSIGHTS")
        print("=" * 60)
        
        if redundant_pairs:
            print("\n⚠️  REDUNDANT STRATEGIES (Correlation > 0.7)")
            print("   Consider using only one from each pair:")
            df_red = pd.DataFrame(redundant_pairs).sort_values('Correlation', ascending=False)
            for _, row in df_red.iterrows():
                print(f"   • {row['Strategy 1']} ↔ {row['Strategy 2']}: {row['Correlation']:.2f}")
            
            # Save redundant pairs
            self._save_csv(df_red, "redundant_pairs.csv")
        else:
            print("\n✅ No highly correlated pairs found - good diversification!")
        
        if diverse_pairs:
            print("\n🌟 DIVERSIFICATION OPPORTUNITIES (Correlation < 0.3)")
            print("   These strategies work well together:")
            df_div = pd.DataFrame(diverse_pairs).sort_values('Correlation')
            for _, row in df_div.head(10).iterrows():
                print(f"   • {row['Strategy 1']} ↔ {row['Strategy 2']}: {row['Correlation']:.2f}")
            
            # Save diverse pairs
            self._save_csv(df_div, "diverse_pairs.csv")
        
        # Save full correlation matrix
        self._save_csv(corr_matrix.reset_index().rename(columns={'index': 'strategy'}), "correlation_matrix.csv")
        
        return corr_matrix
    
    def analyze_confluence(self):
        """Analyze what happens when strategies agree."""
        # Create a copy with all needed columns
        df = self.trades.copy()
        df['entry_minute'] = df['entry_time'].dt.floor('1min')
        df['is_win'] = df['pnl_pips'] > 0
        
        # Group by minute to count strategies
        minute_group = df.groupby('entry_minute')
        
        # Get number of strategies per minute
        n_strategies = minute_group['strategy'].nunique()
        
        # Get total PnL per minute
        total_pnl = minute_group['pnl_pips'].sum()
        
        # Calculate win rate per minute
        win_rate = minute_group['is_win'].mean() * 100
        
        # Get number of trades per minute
        n_trades = minute_group.size()
        
        # Combine into a DataFrame
        confluence_data = pd.DataFrame({
            'n_strategies': n_strategies,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'n_trades': n_trades
        })
        
        # Calculate average PnL per trade for each minute
        confluence_data['avg_pnl'] = confluence_data['total_pnl'] / confluence_data['n_trades']
        
        # Aggregate by confluence level
        confluence = confluence_data.groupby('n_strategies').agg({
            'n_trades': 'sum',
            'win_rate': 'mean',
            'avg_pnl': 'mean'
        }).round(2)
        
        confluence.columns = ['Total Trades', 'Avg Win Rate %', 'Avg PnL']
        confluence.index.name = 'Strategies Agreeing'
        
        print("\n🤝 SIGNAL CONFLUENCE ANALYSIS")
        print("=" * 60)
        print("What happens when multiple strategies fire at the same time:")
        print(confluence.to_string())
        
        # Find the optimal confluence level
        if len(confluence) > 0:
            best_level = confluence['Avg Win Rate %'].idxmax()
            best_wr = confluence.loc[best_level, 'Avg Win Rate %']
            print(f"\n💡 Best signal quality: {best_level} strategies agreeing = {best_wr:.1f}% win rate")
            
            if len(confluence) > 1:
                base_wr = confluence.iloc[0]['Avg Win Rate %']
                improvement = best_wr - base_wr
                if improvement > 5:
                    print(f"   ✅ Signal improves significantly by {improvement:.1f}% (from {base_wr:.1f}% → {best_wr:.1f}%)")
                elif improvement > 2:
                    print(f"   ⚠️  Signal improves moderately by {improvement:.1f}% (from {base_wr:.1f}% → {best_wr:.1f}%)")
                else:
                    print(f"   ⚠️  More strategies agreeing doesn't improve much ({base_wr:.1f}% → {best_wr:.1f}%)")
        
        # Save confluence analysis
        self._save_csv(confluence.reset_index(), "confluence_analysis.csv")
        
        return confluence
    
    def _print_progress(self, current, total, prefix=''):
        """Print progress bar."""
        bar_length = 40
        percent = current / total
        arrow = '=' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(arrow))
        
        sys.stdout.write(f'\r{prefix} [{arrow}{spaces}] {percent:.1%} ({current}/{total})')
        sys.stdout.flush()
    
    def recommend_portfolio(self, max_strategies: int = 4, top_n: int = 15, min_trades: int = 50):
        """
        Recommend the best portfolio based on performance and diversification.
        
        Args:
            max_strategies: Maximum strategies in portfolio (2-4)
            top_n: Only consider top N performing strategies
            min_trades: Minimum trades for a strategy to be considered
        """
        # Rank strategies by performance and filter
        strategy_stats = self.trades.groupby('strategy').agg({
            'pnl_pips': ['sum', 'count', 'mean']
        }).round(2)
        strategy_stats.columns = ['Total PnL', 'Trades', 'Avg PnL']
        
        # Only consider strategies with enough trades
        strategy_stats = strategy_stats[strategy_stats['Trades'] >= min_trades]
        strategy_stats = strategy_stats.sort_values('Total PnL', ascending=False)
        
        # Take top N
        top_strategies = strategy_stats.head(top_n).index.tolist()
        
        print(f"\n🎯 PORTFOLIO OPTIMIZATION")
        print("=" * 60)
        print(f"Total strategies in database: {len(self.strategies)}")
        print(f"Strategies with >= {min_trades} trades: {len(strategy_stats)}")
        print(f"Analyzing top {len(top_strategies)} performers (--top-n={top_n})")
        
        if len(top_strategies) < 2:
            print("⚠️  Not enough strategies with sufficient trades")
            return None
        
        # Show top strategies being analyzed
        print("\n📊 Top strategies being considered:")
        for i, (strat, row) in enumerate(strategy_stats.head(top_n).iterrows(), 1):
            win_rate = self.trades[self.trades['strategy'] == strat]['pnl_pips'].apply(
                lambda x: x > 0
            ).mean() * 100
            print(f"   {i:2d}. {strat:30s} PnL: {row['Total PnL']:8.1f}  Trades: {int(row['Trades']):4d}  WR: {win_rate:5.1f}%")
        
        # Save top strategies list
        top_strategies_df = pd.DataFrame({
            'Rank': range(1, len(top_strategies) + 1),
            'Strategy': top_strategies
        })
        self._save_csv(top_strategies_df, "top_strategies_list.csv")
        
        # Use only top strategies
        strategies = top_strategies
        max_n = min(max_strategies, len(strategies))
        
        # Calculate total combinations
        total_combos = 0
        for n in range(2, max_n + 1):
            total_combos += len(list(combinations(strategies, n)))
        
        print(f"\n⏳ Testing {total_combos:,} combinations of 2-{max_n} strategies...")
        print(f"   (This may take a moment)\n")
        
        # Fixed Curve Metrics to capture actual active hour returns instead of cumulative slopes
        print("📊 Building periodic distribution vectors...")
        periodic_returns = {}
        for i, strat in enumerate(strategies):
            self._print_progress(i + 1, len(strategies), '   Aggregating performance')
            strat_trades = self.trades[self.trades['strategy'] == strat].copy()
            # Calculate total localized hourly yield performance instead of step forward-fills
            periodic_returns[strat] = strat_trades.set_index('exit_time')['pnl_pips'].resample('1h').sum()
        
        print("\n📊 Aligning and masking data periods...")
        # Fill non-active trading hours with 0 to track portfolio divergence cleanly
        combined_returns = pd.DataFrame(periodic_returns).fillna(0)
        
        print("📊 Calculating synchronized return matrix...")
        corr_matrix = combined_returns.corr()
        
        # Test combinations
        print("\n🔍 Testing strategy combinations...")
        results = []
        combo_count = 0
        
        for n in range(2, max_n + 1):
            combos = list(combinations(strategies, n))
            print(f"\n   Testing {len(combos):,} combinations of {n} strategies...")
            
            for i, combo in enumerate(combos):
                # Update progress every 100 combinations
                if i % 100 == 0:
                    self._print_progress(i, len(combos), f'   Testing {n}-strategy combos')
                
                sub_corr = corr_matrix.loc[list(combo), list(combo)]
                mask = np.ones((n, n), dtype=bool)
                np.fill_diagonal(mask, False)
                avg_corr = sub_corr.values[mask].mean() if n > 1 else 0
                
                total_pnl = self.trades[self.trades['strategy'].isin(combo)]['pnl_pips'].sum()
                avg_win_rate = self.trades[self.trades['strategy'].isin(combo)].groupby('strategy')['pnl_pips'].apply(
                    lambda x: (x > 0).mean() * 100
                ).mean()
                
                score = total_pnl * (1 - avg_corr * 0.5)
                
                results.append({
                    'Strategies': " + ".join(combo),
                    'Count': n,
                    'Avg Correlation': round(avg_corr, 3),
                    'Total PnL': round(total_pnl, 1),
                    'Avg Win Rate': round(avg_win_rate, 1),
                    'Score': round(score, 1)
                })
                combo_count += 1
            
            print()  # New line after progress
        
        print(f"\n✅ Tested {combo_count:,} combinations")
        
        if results:
            df_results = pd.DataFrame(results).sort_values('Score', ascending=False).head(20)
            
            print("\n🎯 TOP 20 PORTFOLIO RECOMMENDATIONS")
            print("=" * 60)
            print(df_results[['Strategies', 'Count', 'Avg Correlation', 'Total PnL', 'Avg Win Rate']].to_string(index=False))
            
            # Save portfolio recommendations
            self._save_csv(df_results, "portfolio_recommendations.csv")
            
            if len(df_results) > 0:
                best = df_results.iloc[0]
                print(f"\n💎 RECOMMENDED PORTFOLIO: {best['Strategies']}")
                print(f"   • Expected Total PnL: {best['Total PnL']:.1f}")
                print(f"   • Average Win Rate: {best['Avg Win Rate']:.1f}%")
                print(f"   • Diversification (low correlation): {best['Avg Correlation']:.3f}")
                
                if best['Avg Correlation'] < 0.3:
                    print("   ✅ Well diversified!")
                elif best['Avg Correlation'] < 0.6:
                    print("   ⚠️  Moderate correlation - some diversification benefit")
                else:
                    print("   ⚠️  High correlation - limited diversification benefit")
                
                # Save just the best portfolio
                best_portfolio = pd.DataFrame([{
                    'Strategy': s.strip(),
                    'Portfolio': best['Strategies'],
                    'Total PnL': best['Total PnL'],
                    'Avg Win Rate': best['Avg Win Rate'],
                    'Avg Correlation': best['Avg Correlation']
                } for s in best['Strategies'].split('+')])
                self._save_csv(best_portfolio, "best_portfolio.csv")
            
            return df_results
        else:
            print("\n⚠️  No portfolio combinations found")
            return None
    
    def analyze_by_symbol(self):
        """Analyze strategy performance by symbol."""
        if 'symbol' in self.trades.columns:
            by_symbol = self.trades.groupby('symbol').agg({
                'pnl_pips': ['count', 'sum', 'mean'],
                'strategy': lambda x: x.nunique()
            }).round(2)
            by_symbol.columns = ['Trades', 'Total PnL', 'Avg PnL', 'Strategies']
            by_symbol['Win Rate'] = (
                self.trades.groupby('symbol')['pnl_pips']
                .apply(lambda x: (x > 0).mean() * 100)
                .round(1)
            )
            
            print("\n📊 PERFORMANCE BY SYMBOL")
            print("=" * 60)
            print(by_symbol.to_string())
            
            # Save by symbol analysis
            self._save_csv(by_symbol.reset_index(), "performance_by_symbol.csv")
            
            return by_symbol
        return None
    
    def generate_insights(self, top_n: int = 15, max_portfolio: int = 4, min_trades: int = 50):
        """Generate all insights and create a summary."""
        self.load_all_data()
        
        print("\n" + "=" * 60)
        print("📈 STRATEGY INSIGHTS SUMMARY")
        print("=" * 60)
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🏷️  Analysis ID: {self.timestamp}")
        print("=" * 60)
        
        # Show strategy summary
        summary = self.get_strategy_summary()
        
        # Show performance by symbol
        self.analyze_by_symbol()
        
        # Analyze correlations
        corr_matrix = self.analyze_correlation(min_trades=min_trades)
        
        # Analyze confluence
        confluence = self.analyze_confluence()
        
        # Get top performers
        top = self.get_top_performers(top_n=top_n)
        
        # Recommend portfolio
        portfolio = self.recommend_portfolio(
            max_strategies=max_portfolio,
            top_n=top_n,
            min_trades=min_trades
        )
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        print("=" * 60)
        print(f"\n📁 All CSV files saved to: {self.output_dir}")
        print(f"🏷️  Analysis ID: {self.timestamp}")
        
        # Actionable recommendations
        print("\n📋 ACTIONABLE RECOMMENDATIONS:")
        print("-" * 60)
        
        if len(top) > 0:
            top1 = top.index[0]
            top2 = top.index[1] if len(top) > 1 else None
            print(f"• Top strategy: {top1} with {top.iloc[0]['Total PnL']:.1f} PnL")
            if top2:
                print(f"• Second best: {top2} with {top.iloc[1]['Total PnL']:.1f} PnL")
        
        if len(self.strategies) > 1:
            print(f"• {len(self.strategies)} strategies available for combination")
        else:
            print("• Only 1 strategy found - run more backtests to build a portfolio")
        
        if corr_matrix is not None and len(corr_matrix) > 1:
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            max_corr = upper.max().max()
            if not pd.isna(max_corr):
                if max_corr > 0.8:
                    print("• ⚠️  Some strategies are highly correlated (>.8) - reduce redundancy")
                elif max_corr > 0.6:
                    print("• ⚠️  Moderate correlation (>.6) - consider diversifying")
                else:
                    print("• ✅ Strategies are well diversified")
        
        # Summary of saved files
        print("\n📁 SAVED FILES:")
        csv_files = sorted(self.output_dir.glob(f"{self.timestamp}_*.csv"))
        for f in csv_files:
            print(f"   • {f.name}")
        
        return {
            'summary': summary,
            'top_performers': top,
            'correlation': corr_matrix,
            'confluence': confluence,
            'portfolio': portfolio
        }

def main():
    parser = argparse.ArgumentParser(
        description="Get practical strategy insights from backtest database"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="backtest.db",
        help="Path to SQLite database (absolute or relative)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for CSV files (default: ./analysis_results)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top strategies to analyze for portfolio (default: 15)"
    )
    parser.add_argument(
        "--max-portfolio",
        type=int,
        default=4,
        help="Maximum strategies in portfolio (2-5, default: 4)"
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        help="Minimum trades for a strategy to be considered (default: 50)"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available runs and exit"
    )
    args = parser.parse_args()
    
    try:
        insights = StrategyInsights(args.db, args.output)
        
        if args.list_runs:
            insights.list_runs()
            return
        
        insights.generate_insights(
            top_n=args.top_n,
            max_portfolio=args.max_portfolio,
            min_trades=args.min_trades
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()