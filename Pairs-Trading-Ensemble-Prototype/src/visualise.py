# src/visualize.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity_curve(equity: pd.Series, out_path: str):
    fig = plt.figure(figsize=(10, 5))
    equity.sort_index().plot(ax=plt.gca())
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    _save(fig, Path(out_path))


def plot_pnl_by_pair(trades: pd.DataFrame, out_path: str):
    if trades.empty:
        return
    agg = trades.groupby("pair")["pnl"].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 5))
    agg.plot(kind="bar", ax=plt.gca())
    plt.title("Total PnL by Pair")
    plt.xlabel("Pair")
    plt.ylabel("PnL")
    _save(fig, Path(out_path))


def plot_trade_timeline(trades: pd.DataFrame, out_path: str):
    if trades.empty:
        return
    df = trades.copy()
    df["dur"] = (df["dt_close"] - df["dt_open"]).dt.total_seconds() / 3600.0
    df["sign"] = df["pnl"].fillna(0).apply(lambda x: 1 if x >= 0 else -1)
    # one point per trade at close time, size by |pnl|
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    s = (df["pnl"].abs().fillna(0) + 1e-9) * 5  # scale markers
    ax.scatter(df["dt_close"], df["pnl"], s=s, alpha=0.6)
    ax.axhline(0, lw=1)
    plt.title("Trade Timeline (PnL at Close)")
    plt.xlabel("Close Time")
    plt.ylabel("PnL per Trade")
    _save(fig, Path(out_path))
