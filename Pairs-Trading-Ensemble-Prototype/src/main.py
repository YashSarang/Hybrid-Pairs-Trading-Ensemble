"""
main.py – One-command driver for the pairs-trading ensemble workflow
===================================================================
Run from project root:
    python -m src.main --tickers AAPL,MSFT,GOOGL,META,NVDA --start 2015-01-01 --backtest

Steps executed
--------------
1. **Download** adjusted-close data via `fetch.download_prices`.
2. **Feature-engineer** rolling correlation & spread Z-score (`features.build_pair_features`).
3. **Generate labels & train** RF classifier (unless `--skip-train`).
4. **Rank** current pairs & print top-K ensemble scores.
5. Optionally **back-test** the mean-reversion strategy and print summary stats,
   saving a trade log CSV and charts to the reports directory.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from .fetch import download_prices
from .features import build_pair_features
from .ml_module import generate_labels_backtest, train_rf, load_model
from .ensemble import rank_pairs
from .backtest import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Pairs-trading ensemble driver")
    p.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated ticker list (e.g., AAPL,MSFT,GOOGL)",
    )
    p.add_argument("--start", default="2015-01-01",
                   help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (default today)")
    p.add_argument("--skip-train", action="store_true",
                   help="Use existing RF model")
    p.add_argument("--top-k", type=int, default=5,
                   help="How many pairs to display")
    p.add_argument("--backtest", action="store_true", help="Run toy back-test")
    p.add_argument("--plot", action="store_true",
                   help="Show equity curve (interactive, requires matplotlib)")
    p.add_argument("--report-dir", default="reports",
                   help="Directory for CSVs and charts (default: reports/)")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU (RAPIDS/cuDF/cuML) when available for features/model")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Lightweight plotting helpers (no external deps beyond matplotlib)
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_equity_chart(equity: pd.Series, out_path: Path):
    if equity is None or equity.empty:
        return
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    equity.sort_index().plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pnl_by_pair(trades: pd.DataFrame, out_path: Path):
    if trades is None or trades.empty:
        return
    import matplotlib.pyplot as plt
    agg = trades.groupby("pair")["pnl"].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    agg.plot(kind="bar", ax=ax)
    ax.set_title("Total PnL by Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("PnL")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_trade_timeline(trades: pd.DataFrame, out_path: Path):
    if trades is None or trades.empty:
        return
    import matplotlib.pyplot as plt
    df = trades.copy()
    if "close_date" not in df.columns or "pnl" not in df.columns:
        return
    df["close_date"] = pd.to_datetime(df["close_date"])
    df = df.dropna(subset=["close_date", "pnl"])
    if df.empty:
        return
    # one point per trade at close time, size by |pnl|
    sizes = (df["pnl"].abs() + 1e-9) * 5.0
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.scatter(df["close_date"], df["pnl"], s=sizes, alpha=0.6)
    ax.axhline(0, lw=1)
    ax.set_title("Trade Timeline (PnL at Close)")
    ax.set_xlabel("Close Time")
    ax.set_ylabel("PnL per Trade")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        raise ValueError(
            "Need at least two tickers to form pairs (got: %r)" % tickers)

    report_dir = _ensure_dir(Path(args.report_dir))
    charts_dir = _ensure_dir(report_dir / "charts")

    logger.info("Downloading price data …")
    prices = download_prices(tickers, start=args.start,
                             end=args.end, show_progress=False)

    logger.info("Building rolling features …")
    # GPU acceleration for z_spread when --gpu is set and RAPIDS is available
    features = build_pair_features(prices, use_gpu=args.gpu)

    if not args.skip_train:
        logger.info("Generating labels and training RF model …")
        labels = generate_labels_backtest(prices)
        # GPU training via cuML RF when --gpu is set and available
        train_rf(features, labels, use_gpu=args.gpu)
    else:
        logger.info("Skipping training – loading existing model …")
        load_model()  # triggers lazy load so ensemble has it

    logger.info("Ranking pairs on latest date …")
    top_pairs = rank_pairs(prices, features, top_k=args.top_k)
    print("\nTop pairs by ensemble score:\n",
          top_pairs.to_string(index=False), "\n")

    if args.backtest:
        logger.info("Running toy mean-reversion back-test …")
        equity, trades = run_backtest(prices, features, top_k=args.top_k)

        # ----- Persist trade log -----
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        trades_csv = report_dir / f"trades_{ts}.csv"
        trades.to_csv(trades_csv, index=False)
        logger.info(f"Trades saved → {trades_csv.resolve()}")

        # ----- Summary stats -----
        stats = {
            "final PnL": float(equity.iloc[-1]) if len(equity) else 0.0,
            "max drawdown": float((equity.cummax() - equity).max()) if len(equity) else 0.0,
            "trade count": int(len(trades)),
            "win rate": float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
            "avg pnl / trade": float(trades["pnl"].mean()) if len(trades) else 0.0,
            "median hold (days)": float(trades["hold_days"].median()) if "hold_days" in trades and len(trades) else 0.0,
        }
        print("Back-test summary:\n", pd.Series(stats).to_string())

        # ----- Charts to files -----
        _save_equity_chart(equity, charts_dir / "equity_curve.png")
        _save_pnl_by_pair(trades, charts_dir / "pnl_by_pair.png")
        _save_trade_timeline(trades, charts_dir / "trade_timeline.png")
        logger.info(f"Charts saved → {charts_dir.resolve()}")

        # ----- Optional interactive plot -----
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                equity.plot(title="Equity Curve")
                plt.show()
            except ImportError:
                logger.warning(
                    "matplotlib not installed; skipping interactive plot")


if __name__ == "__main__":
    main()
