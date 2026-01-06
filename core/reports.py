"""Report management for Pairs Trading app.

Handles:
- Saving backtest results to disk with full parameters
- Loading and comparing historical runs
- Benchmark index comparison (Nifty, Sensex, etc.)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------
# Report Data Model
# ---------------------------------------------

@dataclass
class ReportMetadata:
    """Metadata for a saved backtest run."""
    run_id: str
    timestamp: str
    universe: List[str]
    data_config: Dict[str, Any]
    stage1_weights: Dict[str, float]
    stage2_weights: Dict[str, float]
    backtest_config: Dict[str, Any]
    selected_pairs: List[str]
    num_trades: int


class ReportManager:
    """Manages saving and loading backtest reports."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def save_report(
        self,
        result,
        universe: List[str],
        data_config: Dict,
        stage1_weights: Dict[str, float],
        stage2_weights: Dict[str, float],
        backtest_config: Dict,
    ) -> str:
        """Save a backtest result with all parameters.

        Returns:
            run_id: Unique identifier for this run
        """
        # Generate unique run ID
        timestamp = datetime.utcnow()
        run_id = timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir = self.reports_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = ReportMetadata(
            run_id=run_id,
            timestamp=timestamp.isoformat(),
            universe=universe,
            data_config=data_config,
            stage1_weights=stage1_weights,
            stage2_weights=stage2_weights,
            backtest_config=backtest_config,
            selected_pairs=result.params.get("selected_pairs", []),
            num_trades=result.metrics.get("Turnover.Trades", 0),
        )

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)

        # Save parameters
        with open(run_dir / "params.json", "w") as f:
            json.dump(result.params, f, indent=2, default=str)

        # Save time series data
        result.equity_gross.to_csv(run_dir / "equity_gross.csv")
        result.equity_net.to_csv(run_dir / "equity_net.csv")
        result.pnl_gross.to_csv(run_dir / "pnl_gross.csv")
        result.pnl_net.to_csv(run_dir / "pnl_net.csv")
        result.turnover.to_csv(run_dir / "turnover.csv")

        # Save trades
        if not result.trades.empty:
            result.trades.to_csv(run_dir / "trades.csv")

        return run_id

    def list_reports(self) -> List[ReportMetadata]:
        """List all saved reports, sorted by timestamp (newest first)."""
        reports = []

        for run_dir in self.reports_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metadata_file = run_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    reports.append(ReportMetadata(**data))
            except Exception:
                continue

        # Sort by timestamp, newest first
        reports.sort(key=lambda r: r.timestamp, reverse=True)
        return reports

    def load_report(self, run_id: str) -> Dict:
        """Load a complete report by run_id."""
        run_dir = self.reports_dir / run_id

        if not run_dir.exists():
            raise ValueError(f"Report {run_id} not found")

        # Load all components
        with open(run_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        with open(run_dir / "metrics.json", "r") as f:
            metrics = json.load(f)

        with open(run_dir / "params.json", "r") as f:
            params = json.load(f)

        equity_gross = pd.read_csv(
            run_dir / "equity_gross.csv", index_col=0, parse_dates=True).squeeze()
        equity_net = pd.read_csv(
            run_dir / "equity_net.csv", index_col=0, parse_dates=True).squeeze()
        pnl_gross = pd.read_csv(run_dir / "pnl_gross.csv",
                                index_col=0, parse_dates=True).squeeze()
        pnl_net = pd.read_csv(run_dir / "pnl_net.csv",
                              index_col=0, parse_dates=True).squeeze()
        turnover = pd.read_csv(run_dir / "turnover.csv",
                               index_col=0, parse_dates=True).squeeze()

        trades_file = run_dir / "trades.csv"
        if trades_file.exists():
            trades = pd.read_csv(trades_file, index_col=0, parse_dates=True)
        else:
            trades = pd.DataFrame()

        return {
            "metadata": metadata,
            "metrics": metrics,
            "params": params,
            "equity_gross": equity_gross,
            "equity_net": equity_net,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "turnover": turnover,
            "trades": trades,
        }

    def delete_report(self, run_id: str) -> bool:
        """Delete a report by run_id."""
        run_dir = self.reports_dir / run_id

        if not run_dir.exists():
            return False

        import shutil
        shutil.rmtree(run_dir)
        return True


# ---------------------------------------------
# Benchmark Comparison
# ---------------------------------------------

class BenchmarkComparison:
    """Compare strategy returns with market indices."""

    INDIAN_INDICES = {
        "NIFTY 50": "^NSEI",
        "NIFTY 100": "^CNX100",
        "NIFTY 200": "^CNX200",
        "NIFTY 500": "^CNX500",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
    }

    @staticmethod
    def fetch_index_returns(
        index_name: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Fetch index returns for comparison.

        Args:
            index_name: Name from INDIAN_INDICES
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series of cumulative returns (normalized to start at 0)
        """
        if index_name not in BenchmarkComparison.INDIAN_INDICES:
            raise ValueError(f"Unknown index: {index_name}")

        ticker = BenchmarkComparison.INDIAN_INDICES[index_name]

        try:
            # Import yfinance here to avoid issues
            import yfinance as yf

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
            )

            if data.empty:
                raise ValueError(f"No data for {index_name}")

            # Use Adj Close if available, else Close
            if "Adj Close" in data.columns:
                prices = data["Adj Close"]
            else:
                prices = data["Close"]

            # Ensure we have a Series, not DataFrame
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]  # Take first column if DataFrame

            # Calculate cumulative returns (normalized to 0 at start)
            returns = prices.pct_change().fillna(0)
            cum_returns = (1 + returns).cumprod() - 1

            # Set the series name
            cum_returns.name = index_name
            return cum_returns

        except Exception as e:
            raise RuntimeError(f"Failed to fetch {index_name}: {str(e)}")

    @staticmethod
    def fetch_multiple_indices(
        index_names: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.Series]:
        """Fetch multiple index returns for comparison.

        Args:
            index_names: List of index names from INDIAN_INDICES
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping index names to cumulative return series
        """
        results = {}
        for index_name in index_names:
            try:
                results[index_name] = BenchmarkComparison.fetch_index_returns(
                    index_name, start_date, end_date
                )
            except Exception as e:
                # Log error but continue with other indices
                print(f"Failed to fetch {index_name}: {e}")
                continue
        return results

    @staticmethod
    def compare_with_benchmark(
        strategy_equity: pd.Series,
        index_name: str,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Compare strategy performance with a benchmark index.

        Args:
            strategy_equity: Strategy equity curve
            index_name: Name from INDIAN_INDICES
            initial_capital: Starting capital for normalization

        Returns:
            Dict with comparison metrics and aligned series
        """
        start_date = strategy_equity.index.min().strftime("%Y-%m-%d")
        end_date = strategy_equity.index.max().strftime("%Y-%m-%d")

        # Fetch benchmark
        benchmark_returns = BenchmarkComparison.fetch_index_returns(
            index_name, start_date, end_date
        )

        # Align dates
        common_dates = strategy_equity.index.intersection(
            benchmark_returns.index)

        if len(common_dates) < 2:
            raise ValueError("Insufficient overlapping dates for comparison")

        strategy_aligned = strategy_equity.reindex(common_dates)
        benchmark_aligned = benchmark_returns.reindex(common_dates)

        # Normalize both to same starting capital
        strategy_returns = (strategy_aligned -
                            initial_capital) / initial_capital

        # Calculate metrics
        try:
            strategy_total_return = float(strategy_returns.iloc[-1])
        except (ValueError, TypeError):
            strategy_total_return = 0.0

        try:
            benchmark_total_return = float(benchmark_aligned.iloc[-1])
        except (ValueError, TypeError):
            benchmark_total_return = 0.0

        excess_return = strategy_total_return - benchmark_total_return

        # Calculate tracking error (std of return differences)
        strategy_daily_returns = strategy_aligned.pct_change().fillna(0)
        benchmark_daily_returns = benchmark_aligned.pct_change().fillna(0)

        try:
            excess_daily = strategy_daily_returns - benchmark_daily_returns
            excess_std = excess_daily.std()
            if excess_std > 0 and not np.isnan(excess_std) and len(excess_daily) > 1:
                tracking_error = float(excess_std * np.sqrt(252))
            else:
                tracking_error = 0.0
        except (ValueError, TypeError):
            tracking_error = 0.0

        # Information ratio
        excess_daily = strategy_daily_returns - benchmark_daily_returns
        try:
            excess_mean = excess_daily.mean()
            excess_std = excess_daily.std()
            if (excess_std > 0 and not np.isnan(excess_std) and
                    not np.isnan(excess_mean) and len(excess_daily) > 1):
                info_ratio = float(excess_mean / excess_std * np.sqrt(252))
            else:
                info_ratio = 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            info_ratio = 0.0

        return {
            "strategy_return": strategy_total_return,
            "benchmark_return": benchmark_total_return,
            "excess_return": excess_return,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "strategy_series": strategy_returns,
            "benchmark_series": benchmark_aligned,
            "common_dates": common_dates,
        }


__all__ = ["ReportManager", "ReportMetadata", "BenchmarkComparison"]
