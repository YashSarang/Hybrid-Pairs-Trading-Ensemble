"""Streamlit entrypoint for the NSE Pairs Trading app.

This app uses the refactored core modules:
- core.data: DataConfig, YFinanceNSESource
- core.selectors: Pair + selection models
- core.entry: Entry models (ZScore, OU, KalmanHedge)
- core.ensemble: ensembling utilities
- core.backtest: costs, config, engine (now returns GROSS & NET)

Pages:
- Simulator (main)
- Reports (saved evaluations from current session)

Note: Default results record **gross** performance (no costs). NET is an overlay
computed from user-set cost params for comparison.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Core imports
from core.data import DataConfig, YFinanceNSESource
from core.selectors import (
    Pair,
    PairSelector,
    CorrelationSelector,
    DistanceSelector,
    CointegrationSelector,
    CombinedCriteriaSelector,
    MLSelector,
)
from core.entry import ZScoreThreshold, OUThreshold, KalmanHedge
from core.ensemble import normalize_weights, ensemble_pair_scores, scores_to_frame
from core.backtest import IndianCosts, BacktestConfig, backtest_pairs

APP_TITLE = "Comparative Analysis for Pairs Trading (NSE)"
DEFAULT_START = (datetime.utcnow().date() - timedelta(days=365 * 10 + 30))
DEFAULT_END = datetime.utcnow().date()
FREQ_LABELS = {"1D": "Daily", "1H": "Hourly", "1min": "Minute"}


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def _pct(v: float) -> float:
    return float(v) * 100.0

def _overlay_costs(
    pnl_gross: pd.Series,
    turnover: pd.Series,
    notional_each: float,
    capital: float,
    periods_per_year: int,
    costs: IndianCosts,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """Apply cost overlay to a gross PnL using stored turnover & notional.
    Returns (pnl_net, equity_net, metrics_dict).
    """
    if pnl_gross.empty:
        equity_net = pd.Series([], dtype=float)
        return pnl_gross, equity_net, {
            "Net.Return": 0.0,
            "Net.Sharpe": 0.0,
            "Net.Volatility": 0.0,
            "Net.MaxDrawdown": 0.0,
        }

    cost_frac = float(costs.round_trip_cost_fraction())
    cost_series = turnover.fillna(0.0) * (cost_frac * notional_each)
    pnl_net = pnl_gross - cost_series

    equity_net = (capital + pnl_net.cumsum()).rename("equity_net")
    ret = pnl_net / max(capital, 1.0)

    # metrics
    eq = equity_net
    max_eq = eq.cummax().replace(0, np.nan)
    dd = (max_eq - eq) / max_eq
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0 if len(r) < 2 or r.std(ddof=0) == 0 else float((r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year))
    vol = float(ret.std(ddof=0) * np.sqrt(periods_per_year)) if len(ret) else 0.0
    metrics = {
        "Net.Return": float((eq.iloc[-1] - capital) / capital) if len(eq) else 0.0,
        "Net.Sharpe": sharpe,
        "Net.Volatility": vol,
        "Net.MaxDrawdown": float(dd.max()) if len(dd) else 0.0,
    }
    return pnl_net.rename("pnl_net"), equity_net, metrics


# ---------------------------------------------
# Universe builder
# ---------------------------------------------

def nse_pool_selector() -> List[str]:
    st.subheader("Universe (NSE)")
    mode = st.radio(
        "Choose pool source",
        ["Manual", "Upload CSV", "Index (paste/upload)"],
        horizontal=True,
        help="Use 'Index' to paste a list or upload a CSV with constituents.",
    )

    universe: List[str] = []

    if mode == "Manual":
        txt = st.text_area(
            "Enter NSE symbols (comma-separated, without .NS suffix)",
            value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK",
        )
        universe = [t.strip() for t in txt.split(",") if t.strip()]

    elif mode == "Upload CSV":
        f = st.file_uploader("Upload a CSV with a 'Ticker' column (without .NS)", type=["csv"], key="csv_universe")
        if f is not None:
            df = pd.read_csv(f)
            if "Ticker" in df.columns:
                universe = [str(x).strip() for x in df["Ticker"].tolist() if str(x).strip()]
                st.success(f"Loaded {len(universe)} tickers from file.")

    else:  # Index (paste/upload)
        st.caption("Paste or upload your index constituents")
        txt = st.text_area("Paste tickers (comma/space/newline separated)", value="")
        f = st.file_uploader("Or upload a CSV with 'Ticker' column", type=["csv"], key="csv_index")
        pasted: List[str] = []
        if txt:
            raw = [
                x.strip()
                for chunk in txt.split("\n")
                for x in chunk.replace("\t", " ").replace(";", ",").replace(" ", ",").split(",")
            ]
            pasted = [x for x in raw if x]
        uploaded: List[str] = []
        if f is not None:
            df = pd.read_csv(f)
            if "Ticker" in df.columns:
                uploaded = [str(x).strip() for x in df["Ticker"].tolist() if str(x).strip()]
        universe = list(dict.fromkeys(pasted + uploaded))
        if universe:
            st.success(f"Prepared universe of {len(universe)} tickers from index inputs.")
        else:
            st.info("Provide paste or CSV to define constituents.")

    # Optional metadata/sector filters
    st.markdown("**Optional Filters** (requires metadata CSV with: `Ticker`, `Sector`, `Industry`, `MarketCap`, `ADV`) ")
    meta_file = st.file_uploader("Upload metadata CSV for filters (optional)", type=["csv"], key="csv_meta")
    if meta_file is not None:
        meta = pd.read_csv(meta_file)
        if "Ticker" in meta.columns:
            sectors = sorted([s for s in meta.get("Sector", pd.Series(dtype=str)).dropna().unique()])
            if sectors:
                picked = st.multiselect("Include sectors", options=sectors, default=sectors)
                if picked:
                    keep = meta["Sector"].isin(picked)
                    allowed = set(meta.loc[keep, "Ticker"].astype(str))
                    universe = [t for t in universe if t in allowed] if universe else list(allowed)
            if "MarketCap" in meta.columns:
                try:
                    cap_min = st.number_input("Min MarketCap (₹)", value=0.0, step=1e7, format="%0.0f")
                    if cap_min > 0:
                        allowed = set(meta.loc[meta["MarketCap"] >= cap_min, "Ticker"].astype(str))
                        universe = [t for t in universe if t in allowed]
                except Exception:
                    pass
            if "ADV" in meta.columns:
                try:
                    adv_min = st.number_input("Min ADV (₹ per day)", value=0.0, step=1e6, format="%0.0f")
                    if adv_min > 0:
                        allowed = set(meta.loc[meta["ADV"] >= adv_min, "Ticker"].astype(str))
                        universe = [t for t in universe if t in allowed]
                except Exception:
                    pass
            st.info(f"Universe after filters: {len(universe)} tickers")
        else:
            st.warning("No 'Ticker' column found in metadata; filters skipped.")

    return universe


# ---------------------------------------------
# Sidebar controls
# ---------------------------------------------

def sidebar_controls():
    st.sidebar.header("Configuration")
    freq = st.sidebar.selectbox("Data frequency", options=list(FREQ_LABELS.keys()), format_func=lambda x: FREQ_LABELS[x])
    start = st.sidebar.date_input("Start date", value=DEFAULT_START)
    end = st.sidebar.date_input("End date", value=DEFAULT_END)
    price_field = st.sidebar.selectbox("Price field", options=["Adj Close", "Close"])  

    # Stage 1 weights
    st.sidebar.subheader("Stage 1: Pair Selection Weights")
    s1_models = {
        CorrelationSelector.name: st.sidebar.slider("Correlation", 0.0, 1.0, 0.3, 0.05),
        DistanceSelector.name: st.sidebar.slider("Distance (Gatev)", 0.0, 1.0, 0.2, 0.05),
        CointegrationSelector.name: st.sidebar.slider("Cointegration", 0.0, 1.0, 0.3, 0.05),
        CombinedCriteriaSelector.name: st.sidebar.slider("Combined Criteria", 0.0, 1.0, 0.1, 0.05),
        MLSelector.name: st.sidebar.slider("Supervised ML", 0.0, 1.0, 0.1, 0.05),
    }

    # Stage 2 weights
    st.sidebar.subheader("Stage 2: Entry/Exit Weights")
    s2_models = {
        ZScoreThreshold.name: st.sidebar.slider("Mean Reversion (±2σ)", 0.0, 1.0, 0.5, 0.05),
        OUThreshold.name: st.sidebar.slider("OU Model", 0.0, 1.0, 0.5, 0.05),
        KalmanHedge.name: st.sidebar.slider("Kalman Hedge (placeholder)", 0.0, 1.0, 0.0, 0.05),
    }

    s1_weights = normalize_weights(s1_models)
    s2_weights = normalize_weights(s2_models)

    st.sidebar.subheader("Backtest & Costs (NSE)")
    capital = st.sidebar.number_input("Initial capital (₹)", value=100_000, step=10_000)
    per_trade_cap = st.sidebar.number_input("Max per pair (₹)", value=20_000, step=5_000)
    max_pairs = st.sidebar.slider("Max concurrent pairs", 1, 25, 5)

    st.sidebar.caption("Approximate cost preset — edit as needed.")
    brokerage_bps = st.sidebar.number_input("Brokerage (bps per leg)", value=3.0, step=0.5)
    exchange_txn_bps = st.sidebar.number_input("Exchange txn (bps)", value=0.345, step=0.01)
    sebi_bps = st.sidebar.number_input("SEBI charges (bps)", value=0.01, step=0.01)
    stt_bps_sell = st.sidebar.number_input("STT (sell) bps", value=10.0, step=0.5)
    gst_rate = st.sidebar.number_input("GST rate", value=0.18, step=0.01)
    stamp_bps_buy = st.sidebar.number_input("Stamp (buy) bps", value=1.0, step=0.1)
    intraday = st.sidebar.checkbox("Intraday (affects STT/charges)", value=True)
    slippage_bps = st.sidebar.number_input("Slippage (bps per leg) [overlay only]", value=2.0, step=0.5)

    soft_stop = st.sidebar.checkbox("Enable unstrict soft stop-loss", value=True)
    soft_stop_z = st.sidebar.number_input("Soft stop z-threshold", value=3.0, step=0.5)
    soft_stop_decay = st.sidebar.slider("Scale factor on breach", 0.1, 1.0, 0.5, 0.1)
    soft_stop_persist = st.sidebar.number_input("Exit if breach persists (bars)", value=5, min_value=1, step=1)

    run_btn = st.sidebar.button("Run Simulation")

    periods = 252 if freq == "1D" else (24 * 252 if freq == "1H" else int(252 * 6.5 * 60))

    costs = IndianCosts(
        brokerage_bps=float(brokerage_bps),
        exchange_txn_bps=float(exchange_txn_bps),
        sebi_bps=float(sebi_bps),
        stt_bps_sell=float(stt_bps_sell),
        gst_rate=float(gst_rate),
        stamp_bps_buy=float(stamp_bps_buy),
        intraday=bool(intraday),
        slippage_bps_per_leg=float(slippage_bps),
    )

    bt_cfg = BacktestConfig(
        capital=float(capital),
        max_concurrent_pairs=int(max_pairs),
        per_trade_cap=float(per_trade_cap),
        costs=costs,  # used for NET overlay; engine records GROSS by default
        periods_per_year=int(periods),
        soft_stop_z=float(soft_stop_z if soft_stop else 9e9),
        soft_stop_decay=float(soft_stop_decay if soft_stop else 1.0),
        soft_stop_persist_bars=int(soft_stop_persist if soft_stop else 1_000_000),
    )

    data_cfg = DataConfig(start=start, end=end, freq=freq, price_field=price_field)
    return data_cfg, s1_weights, s2_weights, bt_cfg, run_btn


# ---------------------------------------------
# Reports store
# ---------------------------------------------

def save_run_to_session(result, meta: Dict):
    if "runs" not in st.session_state:
        st.session_state["runs"] = []
    st.session_state["runs"].append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": result.metrics,
            "params": {**result.params, **meta},
            "equity_gross": result.equity_gross.to_frame("equity_gross"),
            "equity_net": result.equity_net.to_frame("equity_net"),
            "pnl_gross": result.pnl_gross,
            "pnl_net": result.pnl_net,
            "turnover": result.turnover,
            "trades": result.trades,
        }
    )


def render_reports_page():
    st.header("Saved Evaluations")
    runs = st.session_state.get("runs", [])
    if not runs:
        st.info("No saved runs yet. Execute a simulation on the Simulator page.")
        return

    options = [
        f"Run {i+1} • {r['timestamp']} • Net Sharpe {r['metrics'].get('Net.Sharpe', 0):.2f}"
        for i, r in enumerate(runs)
    ]
    idx = st.selectbox("Select a run", options=list(range(len(runs))), format_func=lambda i: options[i])
    run = runs[idx]

    st.subheader("Key Metrics (Gross vs Net)")
    kpi = pd.DataFrame(
        {
            "Gross": {
                "Return": run["metrics"].get("Gross.Return", 0.0),
                "Sharpe": run["metrics"].get("Gross.Sharpe", 0.0),
                "Volatility": run["metrics"].get("Gross.Volatility", 0.0),
                "Max Drawdown": run["metrics"].get("Gross.MaxDrawdown", 0.0),
            },
            "Net": {
                "Return": run["metrics"].get("Net.Return", 0.0),
                "Sharpe": run["metrics"].get("Net.Sharpe", 0.0),
                "Volatility": run["metrics"].get("Net.Volatility", 0.0),
                "Max Drawdown": run["metrics"].get("Net.MaxDrawdown", 0.0),
            },
        }
    ).T
    st.dataframe(
        (kpi * 100).round(2).rename(columns={"Return": "Return (%)", "Volatility": "Vol (%)", "Max Drawdown": "Max DD (%)"})
    )

    st.subheader("Equity Curves")
    eq = run["equity_gross"].join(run["equity_net"], how="outer")
    st.line_chart(eq)

    st.subheader("Trades (head)")
    st.dataframe(run["trades"].head(300))


# ---------------------------------------------
# Simulator page
# ---------------------------------------------

def simulator_page():
    st.header("Simulator")
    universe = nse_pool_selector()
    data_cfg, s1_weights, s2_weights, bt_cfg, run_btn = sidebar_controls()

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Stage 1 Weights")
        st.json(s1_weights)
        st.subheader("Stage 2 Weights")
        st.json(s2_weights)
        st.subheader("Backtest Config")
        st.json(
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else (v.__dict__ if hasattr(v, "__dict__") else v)
                )
                for k, v in bt_cfg.__dict__.items()
            }
        )

    if run_btn:
        if not universe:
            st.warning("Please provide NSE tickers (e.g., RELIANCE, TCS, INFY,…)")
            st.stop()
        try:
            prices = YFinanceNSESource().get_prices(universe, data_cfg)
        except Exception as e:
            st.error(f"Data load failed: {e}")
            st.stop()

        st.success(
            f"Loaded prices for {len(prices.columns)} tickers, {len(prices)} rows from "
            f"{prices.index.min().date()} to {prices.index.max().date()}."
        )
        st.dataframe(prices.tail())

        # Stage 1: Pair Selection
        st.header("Stage 1: Pairs Selection")
        candidates = [Pair(universe[i], universe[j]) for i in range(len(universe)) for j in range(i + 1, len(universe))]
        st.caption(f"Candidates: {len(candidates)} pairs")

        selectors: Dict[str, PairSelector] = {
            CorrelationSelector.name: CorrelationSelector(lookback=252),
            DistanceSelector.name: DistanceSelector(lookback=252, mode="zscore"),
            CointegrationSelector.name: CointegrationSelector(lookback=504, pvalue_threshold=0.05),
            CombinedCriteriaSelector.name: CombinedCriteriaSelector(),
            MLSelector.name: MLSelector(),
        }
        scores_by_model: Dict[str, List] = {}
        for name, selector in selectors.items():
            st.write(f"Scoring pairs with **{name}**…")
            sel = selector.fit(prices)
            scores_by_model[name] = sel.score_pairs(prices, candidates)

        top_pairs = ensemble_pair_scores(scores_by_model, s1_weights, top_k=25)
        st.dataframe(scores_to_frame(top_pairs))

        # Stage 2: Entry/Exit & Backtest
        st.header("Stage 2: Entry/Exit & Backtest")
        entry_models = {
            ZScoreThreshold.name: ZScoreThreshold(),
            OUThreshold.name: OUThreshold(),
            KalmanHedge.name: KalmanHedge(),
        }
        res = backtest_pairs(prices, [ps.pair for ps in top_pairs], entry_models, s2_weights, bt_cfg)

        # Backward-compat shim: if engine returns old shape, synthesize gross/net
        if not hasattr(res, "equity_gross"):
            # old engine had: res.equity_curve, res.trades, res.metrics, res.params
            equity_curve = getattr(res, "equity_curve", None)
            if equity_curve is None or equity_curve.empty:
                st.error("Backtest returned empty equity curve.")
                st.stop()
            # Rebuild minimal fields
            res.equity_gross = equity_curve.rename("equity_gross")
            res.equity_net = res.equity_gross.copy()
            # Try to reconstruct per-bar PnL (diff of equity)
            pnl = equity_curve.diff().fillna(0.0)
            res.pnl_gross = pnl.rename("pnl_gross")
            res.pnl_net = pnl.rename("pnl_net")
            # If turnover not present, create zeros
            if not hasattr(res, "turnover"):
                res.turnover = pd.Series(0.0, index=equity_curve.index, name="turnover")
            # Patch metrics into Gross.* / Net.* keys for UI
            m = getattr(res, "metrics", {}) or {}
            res.metrics = {
                "Gross.Return": m.get("Total Return", 0.0),
                "Gross.Sharpe": m.get("Sharpe", 0.0),
                "Gross.Volatility": m.get("Volatility", 0.0),
                "Gross.MaxDrawdown": m.get("Max Drawdown", 0.0),
                "Net.Return": m.get("Total Return", 0.0),
                "Net.Sharpe": m.get("Sharpe", 0.0),
                "Net.Volatility": m.get("Volatility", 0.0),
                "Net.MaxDrawdown": m.get("Max Drawdown", 0.0),
                "Turnover.Trades": m.get("Turnover (trades)", 0),
            }
            # Ensure params has notional_each for overlays; default if missing
            if "notional_each" not in res.params:
                res.params["notional_each"] = float(bt_cfg.per_trade_cap)

        # KPIs: Gross vs Net
        st.subheader("Performance (Gross vs Net)")
        kpi = pd.DataFrame(
            {
                "Gross": {
                    "Return": res.metrics.get("Gross.Return", 0.0),
                    "Sharpe": res.metrics.get("Gross.Sharpe", 0.0),
                    "Volatility": res.metrics.get("Gross.Volatility", 0.0),
                    "Max Drawdown": res.metrics.get("Gross.MaxDrawdown", 0.0),
                },
                "Net": {
                    "Return": res.metrics.get("Net.Return", 0.0),
                    "Sharpe": res.metrics.get("Net.Sharpe", 0.0),
                    "Volatility": res.metrics.get("Net.Volatility", 0.0),
                    "Max Drawdown": res.metrics.get("Net.MaxDrawdown", 0.0),
                },
            }
        ).T
        st.dataframe(
            (kpi * 100)
            .round(2)
            .rename(columns={"Return": "Return (%)", "Volatility": "Vol (%)", "Max Drawdown": "Max DD (%)"})
        )

        st.line_chart(pd.concat([res.equity_gross.rename("Gross"), res.equity_net.rename("Net")], axis=1))

        # What-if Scenarios (cost overlay without re-running signals)
        with st.expander("Compare scenarios (what-if fees)", expanded=False):
            presets = {
                "Zero-cost (baseline)": dict(
                    brokerage_bps=0, exchange_txn_bps=0, sebi_bps=0, stt_bps_sell=0, stamp_bps_buy=0, gst_rate=0, slippage_bps_per_leg=0
                ),
                "Discount broker (low fees)": dict(
                    brokerage_bps=1.0,
                    exchange_txn_bps=0.32,
                    sebi_bps=0.01,
                    stt_bps_sell=10.0,
                    stamp_bps_buy=0.5,
                    gst_rate=0.18,
                    slippage_bps_per_leg=1.0,
                ),
                "Conservative (higher slippage)": dict(
                    brokerage_bps=3.0,
                    exchange_txn_bps=0.345,
                    sebi_bps=0.01,
                    stt_bps_sell=10.0,
                    stamp_bps_buy=1.0,
                    gst_rate=0.18,
                    slippage_bps_per_leg=4.0,
                ),
            }
            rows = []
            for name, spec in presets.items():
                costs_i = IndianCosts(**spec)
                pnl_net_i, equity_net_i, m = _overlay_costs(
                    res.pnl_gross, res.turnover, res.params["notional_each"], bt_cfg.capital, bt_cfg.periods_per_year, costs_i
                )
                rows.append(
                    {
                        "Scenario": name,
                        "Net Return (%)": _pct(m["Net.Return"]),
                        "Net Sharpe": m["Net.Sharpe"],
                        "Net Vol (%)": _pct(m["Net.Volatility"]),
                        "Net Max DD (%)": _pct(m["Net.MaxDrawdown"]),
                    }
                )
            st.dataframe(pd.DataFrame(rows).round(3))

        st.subheader("Trades (sample)")
        st.dataframe(res.trades.head(200))

        st.download_button(
            label="Download Trades CSV",
            data=res.trades.to_csv().encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Equity (Gross & Net) CSV",
            data=pd.concat([res.equity_gross, res.equity_net], axis=1).to_csv().encode("utf-8"),
            file_name="equity_curves.csv",
            mime="text/csv",
        )

        # Save run to session for Reports page
        save_run_to_session(
            res,
            meta={
                "universe": universe,
                "freq": data_cfg.freq,
                "start": str(data_cfg.start),
                "end": str(data_cfg.end),
            },
        )

    with col2:
        st.markdown(
            """
            ### Notes on Stage 1 Specifications
            - **Correlation-based**: window defaults to 252 trading days; alternative windows 60–504 days.
            - **Distance (Gatev 2006)**: z-score normalization and L2 distance; alternative is cumulative-return distance.
            - **Cointegration (Engle–Granger)**: p-value threshold 0.05 by default. Alternative: Johansen test.
            - **Combined criteria (Sarmento & Horta, 2021)**: cointegration p-threshold, Hurst<0.5, half-life<limit, min 2σ hits.

            ### Costs (NSE)
            The engine always records **gross** PnL. NET overlays are computed from sidebar cost inputs or scenario presets.

            ### Unstrict Stop-Loss
            Soft z-threshold scales positions; exit if breach persists N bars.

            ### TODOs
            - Add one-click NIFTY50/100/200 pools from a maintained CSV.
            - Add time-stop and hard SL/TP options.
            - Add rolling retraining for ML and richer features.
            - Capacity checks vs. ADV for NSE.
            """
        )


# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    page = st.sidebar.radio("Page", ["Simulator", "Reports"], index=0)
    if page == "Reports":
        render_reports_page()
    else:
        simulator_page()


if __name__ == "__main__":
    main()
