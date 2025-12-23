"""
backtest.py – Toy mean-reversion back-tester for ensemble-selected pairs
========================================================================
The logic is intentionally **simple & readable** so you can extend it later with
transaction-costs, stop-losses, or more sophisticated sizing.

Trading rules
-------------
* Universe: Top-K pairs as ranked by `ensemble.rank_pairs` each day.
* Entry:  Go **long spread** (‐1 × stock2, +1 × stock1) if `z_spread < -entry_z`.
          Go **short spread** (+1 × stock2, ‐1 × stock1) if `z_spread >  entry_z`.
* Exit:   Close position when `|z_spread| < exit_z` **or** max_holding days reached.
* Sizing: Fixed notional ``capital_per_trade`` **per leg**; dollar-neutral.

Outputs
-------
`run_backtest` returns a tuple `(equity_curve, trades)` where
* **equity_curve** – pd.Series of cumulative returns (index = date)
* **trades**        – pd.DataFrame with columns:
    [open_date, close_date, pair, side, entry_z, exit_z, hold_days,
     t1, t2, entry_price1, entry_price2, exit_price1, exit_price2,
     qty1, qty2, pnl]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .ensemble import rank_pairs

# ---------------------------------------------------------------------------
# Data class to track open positions
# ---------------------------------------------------------------------------


@dataclass
class Position:
    pair: str
    side: int                 # +1 long spread, -1 short spread
    entry_date: pd.Timestamp
    entry_prices: Tuple[float, float]  # (p1, p2) at entry
    entry_z: float
    qty1: float               # notional / p1
    qty2: float               # notional / p2

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _calc_z(spread: pd.Series, window: int = 60) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=0)
    return (spread - mean) / std

# ---------------------------------------------------------------------------
# Main back-test function
# ---------------------------------------------------------------------------


def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    lookback: int = 60,
    top_k: int = 5,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    max_holding: int = 20,
    capital_per_trade: float = 10000.0,
):
    """Run a vectorised day-by-day mean-reversion strategy.

    Returns
    -------
    equity_curve : pd.Series  cumulative PnL
    trades       : pd.DataFrame  trade log (one row per **closed** trade)
    """
    prices = prices.sort_index()
    dates = prices.index

    # Pre-compute z-scores for all pairs to avoid inside-loop rolling
    zscores = {}
    pairs = features["pair"].unique()
    for pair in pairs:
        t1, t2 = pair.split("-")
        spread = prices[t1] - prices[t2]
        zscores[pair] = _calc_z(spread, lookback)

    open_positions: List[Position] = []
    closed_trades: List[dict] = []
    equity = []
    cum_pnl = 0.0

    # For richer logs, keep lightweight "entry events" too (not returned separately)
    entry_events: List[dict] = []

    for date in dates:
        # --------------------- Exit logic -------------------------
        still_open = []
        for pos in open_positions:
            age = (date - pos.entry_date).days
            t1, t2 = pos.pair.split("-")
            price1, price2 = prices.loc[date, [t1, t2]]
            z_today = zscores[pos.pair].loc[date]
            exit_cond = (abs(z_today) < exit_z) or (age >= max_holding)

            if exit_cond and not np.isnan(z_today):
                # Realized PnL with explicit dollar-neutral sizing:
                # side=+1 (long spread) => +qty1 on t1, -qty2 on t2
                # side=-1 (short spread) => -qty1 on t1, +qty2 on t2
                pnl_leg1 = (
                    pos.qty1 * (price1 - pos.entry_prices[0])) * pos.side
                pnl_leg2 = (
                    pos.qty2 * (pos.entry_prices[1] - price2)) * pos.side
                pnl = pnl_leg1 + pnl_leg2
                cum_pnl += pnl

                closed_trades.append({
                    "open_date": pos.entry_date,
                    "close_date": date,
                    "pair": pos.pair,
                    "side": pos.side,
                    "entry_z": pos.entry_z,
                    "exit_z": float(z_today),
                    "hold_days": int(age),
                    "t1": t1,
                    "t2": t2,
                    "entry_price1": float(pos.entry_prices[0]),
                    "entry_price2": float(pos.entry_prices[1]),
                    "exit_price1": float(price1),
                    "exit_price2": float(price2),
                    # signed exposure on t1
                    "qty1": float(pos.qty1) * (1 if pos.side == +1 else -1),
                    # signed exposure on t2
                    "qty2": float(pos.qty2) * (-1 if pos.side == +1 else +1),
                    "pnl": float(pnl),
                    "equity_after": float(cum_pnl),
                    "reason_close": "mean_revert" if abs(z_today) < exit_z else "max_holding",
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # --------------------- Entry logic ------------------------
        todays_top = rank_pairs(
            prices.loc[:date], features.loc[features["date"]
                                            <= date], top_k=top_k
        )
        for _, row in todays_top.iterrows():
            pair = row["pair"]
            if any(p.pair == pair for p in open_positions):
                continue  # already in position
            z_today = zscores[pair].loc[date]
            if np.isnan(z_today):
                continue

            if z_today > entry_z:
                side = -1  # short spread: short t1, long t2 (sell high spread)
            elif z_today < -entry_z:
                side = +1  # long spread: long t1, short t2 (buy low spread)
            else:
                continue  # no signal

            t1, t2 = pair.split("-")
            price1, price2 = prices.loc[date, [t1, t2]]

            # Dollar-neutral per leg: allocate fixed notional each side
            qty1 = capital_per_trade / price1
            qty2 = capital_per_trade / price2

            open_positions.append(
                Position(pair, side, date, (price1, price2),
                         float(z_today), qty1, qty2)
            )

            # Lightweight entry log (kept for completeness)
            entry_events.append({
                "event": "open",
                "date": date,
                "pair": pair,
                "side": side,
                "entry_z": float(z_today),
                "t1": t1, "t2": t2,
                "entry_price1": float(price1),
                "entry_price2": float(price2),
                "qty1": float(qty1) * (1 if side == +1 else -1),
                "qty2": float(qty2) * (-1 if side == +1 else +1),
                "capital_per_leg": float(capital_per_trade),
            })

        equity.append(cum_pnl)

    equity_curve = pd.Series(equity, index=dates, name="equity")
    trades_df = pd.DataFrame(closed_trades)

    # Ensure dtypes for downstream plotting
    if not trades_df.empty:
        trades_df["open_date"] = pd.to_datetime(trades_df["open_date"])
        trades_df["close_date"] = pd.to_datetime(trades_df["close_date"])
        numeric_cols = [
            "entry_z", "exit_z", "hold_days",
            "entry_price1", "entry_price2", "exit_price1", "exit_price2",
            "qty1", "qty2", "pnl", "equity_after"
        ]
        for c in numeric_cols:
            if c in trades_df.columns:
                trades_df[c] = pd.to_numeric(trades_df[c], errors="coerce")

    return equity_curve, trades_df
