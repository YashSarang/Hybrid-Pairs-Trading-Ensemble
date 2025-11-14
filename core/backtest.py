# core/backtest.py
"""Backtest engine and Indian cost model for the Pairs Trading app.

This module is UI-agnostic. It handles:
- IndianCosts: approximate NSE charge pack (editable per broker)
- BacktestConfig: capital & execution constraints
- BacktestResult: equity, trades, metrics, params
- backtest_pairs: vectorized market-neutral backtest across selected pairs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import math

import numpy as np
import pandas as pd

# Import only what's needed at runtime; keep type imports behind TYPE_CHECKING
if TYPE_CHECKING:
    from .selectors import Pair
    from .entry import EntryExitModel

# Be resilient to how the package gets run (module vs local script)
try:
    from .ensemble import ensemble_signals
except Exception:  # pragma: no cover
    # Fallback if relative import path resolution differs during local runs
    from core.ensemble import ensemble_signals  # type: ignore


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def _annualize_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=0)
    if std == 0:
        return 0.0
    return float((r.mean() / std) * math.sqrt(periods_per_year))


def _zscore(x: pd.Series, lookback: int) -> pd.Series:
    m = x.rolling(lookback, min_periods=lookback // 3).mean()
    s = x.rolling(lookback, min_periods=lookback // 3).std(ddof=0)
    return (x - m) / (s.replace(0, np.nan) + 1e-9)


# ---------------------------------------------
# Costs & Config
# ---------------------------------------------
@dataclass(frozen=True)
class IndianCosts:
    """Approximate Indian cash-equity cost pack (per leg unless noted).

    brokerage_bps: broker commission (per leg)
    exchange_txn_bps: NSE transaction charge (per leg)
    sebi_bps: SEBI charges (per leg)
    stt_bps_sell: STT on sell leg (delivery ≈ 10 bps; intraday lower)
    gst_rate: GST applied on (brokerage + exchange) only (not on STT/stamp)
    stamp_bps_buy: stamp duty on buy leg
    intraday: whether the model represents intraday vs delivery (for defaults)
    slippage_bps_per_leg: modeled slippage per leg
    """
    brokerage_bps: float = 3.0
    exchange_txn_bps: float = 0.345
    sebi_bps: float = 0.01
    stt_bps_sell: float = 10.0
    gst_rate: float = 0.18
    stamp_bps_buy: float = 1.0
    intraday: bool = True
    slippage_bps_per_leg: float = 2.0

    def round_trip_cost_fraction(self) -> float:
        """Estimated *round-trip* cost fraction for changing a pair position.

        Includes two legs (long A, short B) and both sides (buy & sell) of a position flip.
        """
        base_bps = self.brokerage_bps + self.exchange_txn_bps + self.sebi_bps
        gst_bps = (self.brokerage_bps + self.exchange_txn_bps) * self.gst_rate
        # Two legs per change; add statutory items once per buy/sell, plus modeled slippage
        statutory_bps = self.stt_bps_sell + self.stamp_bps_buy
        slippage_bps = 2 * self.slippage_bps_per_leg  # two legs
        total_bps = 2 * (base_bps + gst_bps) + statutory_bps + slippage_bps
        return float(total_bps / 1e4)


@dataclass
class BacktestConfig:
    capital: float = 100_000.0                 # starting equity in ₹
    max_concurrent_pairs: int = 5
    per_trade_cap: float = 20_000.0            # notional cap per pair (both legs combined)
    costs: IndianCosts = field(default_factory=IndianCosts)
    periods_per_year: int = 252
    # Soft ("unstrict") stop-loss on spread z
    soft_stop_z: float = 3.0
    soft_stop_decay: float = 0.5
    soft_stop_persist_bars: int = 5
    # Z-score lookback for soft stop
    soft_stop_lookback: int = 60


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # capital + cumulative PnL
    trades: pd.DataFrame     # index=time, columns: pair, signal
    metrics: Dict[str, float]
    params: Dict


# ---------------------------------------------
# Engine
# ---------------------------------------------
# --- Add/replace these in core/backtest.py ---

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd

# ... keep your existing helpers and IndianCosts/BacktestConfig ...

@dataclass
class BacktestResult:
    # New: keep both gross & net
    equity_gross: pd.Series           # capital + cumsum(gross_pnl)
    equity_net: pd.Series             # capital + cumsum(net_pnl)  (depends on costs)
    pnl_gross: pd.Series              # per-bar PnL without frictions
    pnl_net: pd.Series                # per-bar PnL after costs
    turnover: pd.Series               # per-bar #position changes across all pairs
    trades: pd.DataFrame              # index=time, columns: pair, signal
    metrics: Dict[str, float]         # includes Gross.* and Net.* keys
    params: Dict                      # includes notional_each & costs

def _metrics_from_pnl(pnl: pd.Series, capital: float, periods_per_year: int) -> Dict[str, float]:
    if pnl.empty:
        return {"Return": 0.0, "Sharpe": 0.0, "Volatility": 0.0, "MaxDrawdown": 0.0}
    eq = (capital + pnl.cumsum())
    ret = pnl / max(capital, 1.0)
    max_eq = eq.cummax().replace(0, np.nan)
    dd = (max_eq - eq) / max_eq
    vol = float(ret.std(ddof=0) * math.sqrt(periods_per_year)) if len(ret) else 0.0
    # annualized Sharpe
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0 if len(r) < 2 or r.std(ddof=0) == 0 else float((r.mean() / r.std(ddof=0)) * math.sqrt(periods_per_year))
    return {
        "Return": float((eq.iloc[-1] - capital) / capital),
        "Sharpe": sharpe,
        "Volatility": vol,
        "MaxDrawdown": float(dd.max()) if len(dd) else 0.0,
    }

def backtest_pairs(
    prices: pd.DataFrame,
    selected_pairs: List["Pair"],
    entry_models: Dict[str, "EntryExitModel"],
    entry_weights: Dict[str, float],
    cfg: BacktestConfig,
) -> BacktestResult:
    if prices is None or prices.empty:
        raise ValueError("prices is empty")

    index = prices.index
    pnl_gross = pd.Series(0.0, index=index)
    cost_series = pd.Series(0.0, index=index)
    turnover_series = pd.Series(0.0, index=index)
    all_trades: List[pd.DataFrame] = []

    notional_each = float(min(cfg.capital / max(1, cfg.max_concurrent_pairs), cfg.per_trade_cap))
    cost_frac = float(cfg.costs.round_trip_cost_fraction())

    for p in selected_pairs:
        if p.a not in prices.columns or p.b not in prices.columns:
            continue
        a = prices[p.a].dropna(); b = prices[p.b].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) < 200:
            continue
        a, b = a.reindex(idx).ffill(), b.reindex(idx).ffill()

        # signals
        signals_by_model: Dict[str, pd.Series] = {}
        for name, model in entry_models.items():
            sig = model.trade_signals(a, b).reindex(idx).fillna(0)
            signals_by_model[name] = sig
        sig = ensemble_signals(signals_by_model, entry_weights).astype(float)

        # soft stop (unchanged)
        spread = a - b
        z = _zscore(spread, cfg.soft_stop_lookback)
        breach = z.abs() > cfg.soft_stop_z
        breach_persist = breach.rolling(cfg.soft_stop_persist_bars).sum().fillna(0) >= cfg.soft_stop_persist_bars
        scale = pd.Series(1.0, index=idx); scale.loc[breach] = cfg.soft_stop_decay
        sig_scaled = (sig * scale).round().astype(int)
        sig_scaled.loc[breach_persist] = 0

        # returns & gross pnl
        r_a = a.pct_change().fillna(0.0)
        r_b = b.pct_change().fillna(0.0)
        r_spread = r_a - r_b

        sig_prev = sig_scaled.shift(1).fillna(0).astype(int)
        pair_turnover = (sig_scaled != sig_prev).astype(int)

        pair_gross = (sig_prev * r_spread) * notional_each
        pair_costs = (pair_turnover * cost_frac) * notional_each  # purely a % of notional

        # accumulate
        pnl_gross = pnl_gross.add(pair_gross.reindex(index).fillna(0.0), fill_value=0.0)
        cost_series = cost_series.add(pair_costs.reindex(index).fillna(0.0), fill_value=0.0)
        turnover_series = turnover_series.add(pair_turnover.reindex(index).fillna(0.0), fill_value=0.0)

        # trades table
        trans_idx = sig_scaled.index[pair_turnover.astype(bool)]
        if len(trans_idx):
            df = pd.DataFrame({"time": trans_idx,
                               "pair": [f"{p.a}-{p.b}"] * len(trans_idx),
                               "signal": sig_scaled.loc[trans_idx].values}).set_index("time")
            all_trades.append(df)

    # Build equity series
    equity_gross = (cfg.capital + pnl_gross.cumsum()).rename("equity_gross")
    pnl_net = (pnl_gross - cost_series).rename("pnl_net")
    equity_net = (cfg.capital + pnl_net.cumsum()).rename("equity_net")

    # Metrics (both)
    m_g = _metrics_from_pnl(pnl_gross, cfg.capital, cfg.periods_per_year)
    m_n = _metrics_from_pnl(pnl_net,   cfg.capital, cfg.periods_per_year)
    metrics = {
        "Gross.Return": m_g["Return"],
        "Gross.Sharpe": m_g["Sharpe"],
        "Gross.Volatility": m_g["Volatility"],
        "Gross.MaxDrawdown": m_g["MaxDrawdown"],
        "Net.Return": m_n["Return"],
        "Net.Sharpe": m_n["Sharpe"],
        "Net.Volatility": m_n["Volatility"],
        "Net.MaxDrawdown": m_n["MaxDrawdown"],
        "Turnover.Trades": int(turnover_series.sum()),
    }

    trades = pd.concat(all_trades) if all_trades else pd.DataFrame(columns=["pair", "signal"])
    params = {
        "selected_pairs": [f"{p.a}-{p.b}" for p in selected_pairs],
        "entry_weights": entry_weights,
        "notional_each": notional_each,
        "costs": {
            "brokerage_bps": cfg.costs.brokerage_bps,
            "exchange_txn_bps": cfg.costs.exchange_txn_bps,
            "sebi_bps": cfg.costs.sebi_bps,
            "stt_bps_sell": cfg.costs.stt_bps_sell,
            "gst_rate": cfg.costs.gst_rate,
            "stamp_bps_buy": cfg.costs.stamp_bps_buy,
            "slippage_bps_per_leg": cfg.costs.slippage_bps_per_leg,
            "intraday": cfg.costs.intraday,
        },
    }

    return BacktestResult(
        equity_gross=equity_gross,
        equity_net=equity_net,
        pnl_gross=pnl_gross.rename("pnl_gross"),
        pnl_net=pnl_net,
        turnover=turnover_series.rename("turnover"),
        trades=trades,
        metrics=metrics,
        params=params,
    )


__all__ = ["IndianCosts", "BacktestConfig", "BacktestResult", "backtest_pairs"]
