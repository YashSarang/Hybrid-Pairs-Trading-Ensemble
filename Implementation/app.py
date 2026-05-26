"""Streamlit entrypoint for the NSE Pairs Trading app.

This app uses the refactored core modules:
- core.data: DataConfig, YFinanceNSESource
- core.selectors: Pair + selection models
- core.entry: Entry models (ZScore, OU, KalmanHedge, MLSignal)
- core.ensemble: ensembling utilities
- core.backtest: costs, config, engine (now returns GROSS & NET)

Pages:
- Simulator (main)
- Reports (saved evaluations from current session)

Note: Default results record **gross** performance (no costs). NET is an overlay
computed from user-set cost params for comparison.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    LSTMSelector,
    TransformerSelector,
    GNNSelector,
)
from core.entry import ZScoreThreshold, OUThreshold, KalmanHedge, MLSignal
from core.ensemble import normalize_weights, ensemble_pair_scores, scores_to_frame
from core.backtest import IndianCosts, BacktestConfig, backtest_pairs
from core.reports import ReportManager, BenchmarkComparison
from core.predictions import PredictionEngine
from nse_symbols_reference import NSE_STOCK_SYMBOLS, format_symbols_for_input

APP_TITLE = "Comparative Analysis for Pairs Trading (NSE)"
DEFAULT_START = (datetime.now(timezone.utc).date() -
                 timedelta(days=365 * 10 + 30))
DEFAULT_END = datetime.now(timezone.utc).date()
FREQ_LABELS = {"1D": "Daily", "1H": "Hourly"}


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
    sharpe = 0.0 if len(r) < 2 or r.std(ddof=0) == 0 else float(
        (r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year))
    vol = float(ret.std(ddof=0) * np.sqrt(periods_per_year)
                ) if len(ret) else 0.0
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
    st.subheader("Stock Universe Selection")

    mode = st.radio(
        "Choose how to select stocks",
        ["Manual Entry", "Upload CSV", "Index Constituents"],
        horizontal=True,
        help="Select stocks manually, upload a file, or use index constituents.",
    )

    universe: List[str] = []
    
    # Add helpful stock symbol reference dropdown
    with st.expander("NSE Stock Symbol Reference (Click to expand)", expanded=False):
        st.markdown("### Quick Reference: Copy symbols from sectors below")
        st.info("**Tip:** Select a sector, click 'Copy Symbols', then paste into the input box above")
        
        # Sector selection
        sector = st.selectbox(
            "Select Sector:",
            options=list(NSE_STOCK_SYMBOLS.keys()),
            index=len(NSE_STOCK_SYMBOLS) - 1  # Default to "Top 30 Liquid Stocks"
        )
        
        if sector:
            sector_data = NSE_STOCK_SYMBOLS[sector]
            st.markdown(f"**{sector}**")
            st.caption(sector_data["description"])
            
            # Format symbols for easy copying
            symbols_text = format_symbols_for_input(sector_data["symbols"])
            
            # Display in a text area for easy selection and copying
            st.text_area(
                f"Symbols ({len(sector_data['symbols'])} stocks)",
                value=symbols_text,
                height=120,
                help="Click inside, press Ctrl+A to select all, then Ctrl+C to copy"
            )
            
            # Also show as a list for reference
            cols = st.columns(3)
            for i, symbol in enumerate(sector_data["symbols"]):
                with cols[i % 3]:
                    st.code(symbol, language=None)
        
        st.markdown("---")
        st.caption(f"**Total unique symbols across all sectors:** {len(set(s for d in NSE_STOCK_SYMBOLS.values() for s in d['symbols']))}")

    if "Manual Entry" in mode:
        st.markdown("**Enter NSE stock symbols** (without .NS suffix)")
        txt = st.text_area(
            "Stock symbols (comma-separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, BHARTIARTL, ASIANPAINT",
            height=100,
            help="Enter NSE symbols separated by commas. Example: RELIANCE, TCS, INFY"
        )
        universe = [t.strip().upper() for t in txt.split(",") if t.strip()]

        if universe:
            st.success(
                f"[OK] Selected {len(universe)} stocks: {', '.join(universe[:5])}{' ...' if len(universe) > 5 else ''}")

    elif "Upload CSV" in mode:
        st.markdown("**Upload a CSV file** with stock symbols")
        f = st.file_uploader("Choose CSV file with 'Ticker' column", type=[
                             "csv"], key="csv_universe")
        if f is not None:
            df = pd.read_csv(f)
            if "Ticker" in df.columns:
                universe = [str(x).strip().upper()
                            for x in df["Ticker"].tolist() if str(x).strip()]
                st.success(f"[OK] Loaded {len(universe)} tickers from file.")
            else:
                st.error("[Error] CSV must have a 'Ticker' column")

    else:  # Index constituents
        st.markdown("**Use index constituents** or paste your own list")

        col1, col2 = st.columns([1, 1])

        with col1:
            txt = st.text_area(
                "Paste tickers (comma/space/newline separated)",
                value="",
                height=150,
                help="Paste stock symbols in any format - commas, spaces, or new lines"
            )

        with col2:
            f = st.file_uploader("Or upload CSV with 'Ticker' column", type=[
                                 "csv"], key="csv_index")

        pasted: List[str] = []
        if txt:
            raw = [
                x.strip().upper()
                for chunk in txt.split("\n")
                for x in chunk.replace("\t", " ").replace(";", ",").replace(" ", ",").split(",")
            ]
            pasted = [x for x in raw if x and x.isalpha()]

        uploaded: List[str] = []
        if f is not None:
            df = pd.read_csv(f)
            if "Ticker" in df.columns:
                uploaded = [str(x).strip().upper()
                            for x in df["Ticker"].tolist() if str(x).strip()]

        # Remove duplicates while preserving order
        universe = list(dict.fromkeys(pasted + uploaded))

        if universe:
            st.success(f"[OK] Prepared universe of {len(universe)} tickers")
        else:
            st.info("Tip: Paste tickers or upload CSV to define your universe")

    # Optional metadata/sector filters
    st.markdown(
        "**Optional Filters** (requires metadata CSV with: `Ticker`, `Sector`, `Industry`, `MarketCap`, `ADV`) ")
    meta_file = st.file_uploader(
        "Upload metadata CSV for filters (optional)", type=["csv"], key="csv_meta")
    if meta_file is not None:
        meta = pd.read_csv(meta_file)
        if "Ticker" in meta.columns:
            sectors = sorted([s for s in meta.get(
                "Sector", pd.Series(dtype=str)).dropna().unique()])
            if sectors:
                picked = st.multiselect(
                    "Include sectors", options=sectors, default=sectors)
                if picked:
                    keep = meta["Sector"].isin(picked)
                    allowed = set(meta.loc[keep, "Ticker"].astype(str))
                    universe = [
                        t for t in universe if t in allowed] if universe else list(allowed)
            if "MarketCap" in meta.columns:
                try:
                    cap_min = st.number_input(
                        "Min MarketCap (₹)", value=0.0, step=1e7, format="%0.0f")
                    if cap_min > 0:
                        allowed = set(
                            meta.loc[meta["MarketCap"] >= cap_min, "Ticker"].astype(str))
                        universe = [t for t in universe if t in allowed]
                except Exception:
                    pass
            if "ADV" in meta.columns:
                try:
                    adv_min = st.number_input(
                        "Min ADV (₹ per day)", value=0.0, step=1e6, format="%0.0f")
                    if adv_min > 0:
                        allowed = set(
                            meta.loc[meta["ADV"] >= adv_min, "Ticker"].astype(str))
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

    # Data configuration
    st.sidebar.subheader("Data Settings")
    freq = st.sidebar.selectbox("Data frequency", options=list(
        FREQ_LABELS.keys()), format_func=lambda x: FREQ_LABELS[x])
    start = st.sidebar.date_input("Start date", value=DEFAULT_START)
    end = st.sidebar.date_input("End date", value=DEFAULT_END)
    price_field = st.sidebar.selectbox(
        "Price field", options=["Adj Close", "Close"])

    # Stage 1 weights
    st.sidebar.subheader("Stage 1: Pair Selection")
    st.sidebar.caption("Adjust weights for different pair selection methods")
    s1_models = {
        CorrelationSelector.name: st.sidebar.slider("Correlation", 0.0, 1.0, 0.25, 0.05),
        DistanceSelector.name: st.sidebar.slider("Distance (Gatev)", 0.0, 1.0, 0.15, 0.05),
        CointegrationSelector.name: st.sidebar.slider("Cointegration", 0.0, 1.0, 0.25, 0.05),
        CombinedCriteriaSelector.name: st.sidebar.slider("Combined Criteria", 0.0, 1.0, 0.1, 0.05),
        MLSelector.name: st.sidebar.slider("Supervised ML", 0.0, 1.0, 0.15, 0.05),
        LSTMSelector.name: st.sidebar.slider("LSTM/BiLSTM", 0.0, 1.0, 0.1, 0.05),
        TransformerSelector.name: st.sidebar.slider("Transformer", 0.0, 1.0, 0.1, 0.05),
        GNNSelector.name: st.sidebar.slider("GNN", 0.0, 1.0, 0.1, 0.05),
    }

    # Stage 2 weights
    st.sidebar.subheader("Stage 2: Entry/Exit")
    st.sidebar.caption("Adjust weights for different trading signals")
    s2_models = {
        ZScoreThreshold.name: st.sidebar.slider("Mean Reversion (±2σ)", 0.0, 1.0, 0.5, 0.05),
        OUThreshold.name: st.sidebar.slider("OU Model", 0.0, 1.0, 0.5, 0.05),
        KalmanHedge.name: st.sidebar.slider("Kalman Hedge", 0.0, 1.0, 0.0, 0.05),
        MLSignal.name: st.sidebar.slider("ML Signal", 0.0, 1.0, 0.0, 0.05),
    }

    s1_weights = normalize_weights(s1_models)
    s2_weights = normalize_weights(s2_models)

    st.sidebar.subheader("Backtest & Costs (NSE)")
    capital = st.sidebar.number_input(
        "Initial capital (₹)", value=100_000, step=10_000)
    per_trade_cap = st.sidebar.number_input(
        "Max per pair (₹)", value=20_000, step=5_000)
    max_pairs = st.sidebar.slider("Max concurrent pairs", 1, 25, 5)

    # Cost configuration toggle
    use_advanced_costs = st.sidebar.checkbox(
        "Advanced Cost Configuration", value=False)

    if use_advanced_costs:
        st.sidebar.caption("Custom cost parameters — edit as needed.")
        brokerage_bps = st.sidebar.number_input(
            "Brokerage (bps per leg)", value=3.0, step=0.5)
        exchange_txn_bps = st.sidebar.number_input(
            "Exchange txn (bps)", value=0.345, step=0.01)
        sebi_bps = st.sidebar.number_input(
            "SEBI charges (bps)", value=0.01, step=0.01)
        stt_bps_sell = st.sidebar.number_input(
            "STT (sell) bps", value=10.0, step=0.5)
        gst_rate = st.sidebar.number_input("GST rate", value=0.18, step=0.01)
        stamp_bps_buy = st.sidebar.number_input(
            "Stamp (buy) bps", value=1.0, step=0.1)
        intraday = st.sidebar.checkbox(
            "Intraday (affects STT/charges)", value=True)
        slippage_bps = st.sidebar.number_input(
            "Slippage (bps per leg)", value=2.0, step=0.5)
    else:
        # Cost presets
        cost_preset = st.sidebar.selectbox(
            "Cost Preset",
            options=["Standard Broker", "Discount Broker",
                     "Premium Broker", "Zero Cost (Testing)"],
            index=0
        )

        if cost_preset == "Discount Broker":
            # Zerodha/Upstox type costs (2024-2026)
            # Flat ₹20 per order = 0 bps for large trades
            brokerage_bps, exchange_txn_bps, sebi_bps = 0.0, 0.322, 0.01
            stt_bps_sell, gst_rate, stamp_bps_buy = 10.0, 0.18, 1.5
            slippage_bps, intraday = 2.0, False  # Delivery trading
        elif cost_preset == "Premium Broker":
            # Traditional full-service broker
            brokerage_bps, exchange_txn_bps, sebi_bps = 25.0, 0.345, 0.01  # 0.25% brokerage
            stt_bps_sell, gst_rate, stamp_bps_buy = 10.0, 0.18, 1.5
            slippage_bps, intraday = 3.0, True
        elif cost_preset == "Zero Cost (Testing)":
            brokerage_bps, exchange_txn_bps, sebi_bps = 0.0, 0.0, 0.0
            stt_bps_sell, gst_rate, stamp_bps_buy = 0.0, 0.0, 0.0
            slippage_bps, intraday = 0.0, True
        else:  # Standard Broker
            # Mid-tier broker costs
            brokerage_bps, exchange_txn_bps, sebi_bps = 5.0, 0.345, 0.01  # 0.05% brokerage
            stt_bps_sell, gst_rate, stamp_bps_buy = 10.0, 0.18, 1.5
            slippage_bps, intraday = 2.0, True

        # Show selected preset values with accurate round-trip cost
        # Create temporary IndianCosts object to calculate exact round-trip
        temp_costs = IndianCosts(
            brokerage_bps=float(brokerage_bps),
            exchange_txn_bps=float(exchange_txn_bps),
            sebi_bps=float(sebi_bps),
            stt_bps_sell=float(stt_bps_sell),
            gst_rate=float(gst_rate),
            stamp_bps_buy=float(stamp_bps_buy),
            slippage_bps=float(slippage_bps),
            intraday=intraday
        )
        rt_cost = temp_costs.round_trip_cost_fraction() * 10000  # Convert to bps
        st.sidebar.caption(f"Using {cost_preset} preset:")
        st.sidebar.caption(f"• Brokerage: {brokerage_bps} bps per leg")
        st.sidebar.caption(f"• Round-trip cost: {rt_cost:.2f} bps")

    soft_stop = st.sidebar.checkbox(
        "Enable unstrict soft stop-loss", value=True)
    soft_stop_z = st.sidebar.number_input(
        "Soft stop z-threshold", value=3.0, step=0.5)
    soft_stop_decay = st.sidebar.slider(
        "Scale factor on breach", 0.1, 1.0, 0.5, 0.1)
    soft_stop_persist = st.sidebar.number_input(
        "Exit if breach persists (bars)", value=5, min_value=1, step=1)

    run_btn = st.sidebar.button(
        "Run Simulation", type="primary", use_container_width=True)

    periods = 252 if freq == "1D" else 24 * 252  # Daily or Hourly

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
        soft_stop_persist_bars=int(
            soft_stop_persist if soft_stop else 1_000_000),
    )

    data_cfg = DataConfig(start=start, end=end, freq=freq,
                          price_field=price_field)
    return data_cfg, s1_weights, s2_weights, bt_cfg, run_btn


# ---------------------------------------------
# Reports store
# ---------------------------------------------

def get_report_manager():
    """Get or create ReportManager instance with caching for performance."""
    if "report_manager" not in st.session_state:
        st.session_state["report_manager"] = ReportManager()
    return st.session_state["report_manager"]


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_cached_report(run_id: str):
    """Load report with caching to improve performance."""
    report_mgr = get_report_manager()
    return report_mgr.load_report(run_id)


def get_cached_reports_list():
    """Get reports list - simplified without caching to avoid serialization issues."""
    report_mgr = get_report_manager()
    return report_mgr.list_reports()


def render_reports_page():
    """Render the reports page with saved backtest analysis and prediction generation.

    This page provides comprehensive analysis of saved backtest runs including:
    - Performance metrics comparison (gross vs net returns)
    - Benchmark comparison against Indian market indices
    - Parameter inspection and trade analysis
    - Real-time prediction generation using historical settings

    Key Features:
    - Cached report loading for improved performance
    - Interactive benchmark comparison with multiple indices
    - One-click prediction generation from any historical run
    - Comprehensive trade analysis and export functionality
    """
    # Add description and stats with performance optimization
    report_mgr = get_report_manager()
    reports = get_cached_reports_list()  # Use cached version

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        **Analyze your trading strategies** by reviewing saved reports, comparing performance 
        metrics, and benchmarking against market indices.
        """)

    with col2:
        st.metric("Total Reports", len(reports))

    with col3:
        if reports:
            latest_report = reports[0]
            st.metric("Latest Run", latest_report.run_id[:8] + "...")

    st.divider()

    if not reports:
        st.info(" No saved reports yet. Run your first simulation to get started!")
        st.markdown("**Quick Start:**")
        st.markdown("1. Go to Simulator")
        st.markdown("2. Select NSE tickers (e.g., RELIANCE, TCS, INFY)")
        st.markdown("3. Click 'Run Simulation'")
        st.markdown("4. Return here to view results")
        return

    # Report selector
    options = [
        f"{r.run_id} • {r.timestamp[:19]} • {len(r.universe)} tickers • {r.num_trades} trades"
        for r in reports
    ]
    selected_idx = st.selectbox("Select a report", options=list(
        range(len(reports))), format_func=lambda i: options[i])
    selected_report = reports[selected_idx]

    # Load full report data with caching
    report = load_cached_report(selected_report.run_id)

    # Display parameters
    with st.expander(" Run Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Configuration**")
            st.json(report["metadata"]["data_config"])

            st.markdown("**Stage 1 Weights (Pair Selection)**")
            st.json(report["metadata"]["stage1_weights"])

        with col2:
            st.markdown("**Stage 2 Weights (Entry/Exit)**")
            st.json(report["metadata"]["stage2_weights"])

            st.markdown("**Backtest Configuration**")
            st.json(report["metadata"]["backtest_config"])

        st.markdown("**Universe**")
        st.write(", ".join(report["metadata"]["universe"]))

        st.markdown("**Selected Pairs**")
        st.write(", ".join(report["metadata"]["selected_pairs"]))

        # Add prediction generation button
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            generate_predictions_btn = st.button(
                " Generate Current Predictions",
                type="secondary",
                help="Generate real-time predictions using this report's exact settings"
            )
        with col2:
            if generate_predictions_btn:
                st.info("Generating predictions with this report's settings...")

        if generate_predictions_btn:
            with st.spinner("Generating predictions using report settings..."):
                try:
                    from core.predictions import PredictionEngine

                    # Initialize prediction engine
                    engine = PredictionEngine(lookback_days=252)

                    # Generate predictions using report settings
                    prediction_result = engine.get_predictions_from_report(
                        report_data=report,
                        top_k=10,
                        min_data_points=100
                    )

                    if prediction_result.recommendations:
                        st.success(
                            f"[OK] Generated {len(prediction_result.recommendations)} predictions!")

                        # Store in session state for display
                        st.session_state[f"predictions_for_{selected_report.run_id}"] = prediction_result

                        # Display predictions summary
                        st.markdown("**Current Predictions Summary:**")
                        pred_col1, pred_col2, pred_col3 = st.columns(3)

                        with pred_col1:
                            st.metric("Top Pairs", len(
                                prediction_result.recommendations))
                        with pred_col2:
                            st.metric("Data Freshness",
                                      prediction_result.data_freshness)
                        with pred_col3:
                            top_score = prediction_result.recommendations[
                                0].score if prediction_result.recommendations else 0
                            st.metric("Top Score", f"{top_score:.3f}")

                        # Show top 3 predictions
                        st.markdown("**Top 3 Current Recommendations:**")
                        for i, rec in enumerate(prediction_result.recommendations[:3], 1):
                            signal_strength = abs(rec.ensemble_signal)
                            signal_icon = "[On]" if signal_strength > 0.5 else "[Warn]" if signal_strength > 0.2 else "[Neutral]"
                            st.write(
                                f"{i}. {rec.pair.a}/{rec.pair.b} - Score: {rec.score:.3f} {signal_icon}")

                    else:
                        st.warning(
                            "[Warning] No predictions generated. Check market data availability.")

                except Exception as e:
                    st.error(f"[Error] Prediction generation failed: {str(e)}")
                    st.info(
                        "Tip: This might be due to market data availability or network issues.")

    # Key Metrics
    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gross Return",
                  f"{report['metrics']['Gross.Return']*100:.2f}%")
        st.metric("Net Return", f"{report['metrics']['Net.Return']*100:.2f}%")
    with col2:
        st.metric("Gross Sharpe", f"{report['metrics']['Gross.Sharpe']:.2f}")
        st.metric("Net Sharpe", f"{report['metrics']['Net.Sharpe']:.2f}")
    with col3:
        st.metric("Gross Volatility",
                  f"{report['metrics']['Gross.Volatility']*100:.2f}%")
        st.metric("Net Volatility",
                  f"{report['metrics']['Net.Volatility']*100:.2f}%")
    with col4:
        st.metric("Gross Max DD",
                  f"{report['metrics']['Gross.MaxDrawdown']*100:.2f}%")
        st.metric("Net Max DD",
                  f"{report['metrics']['Net.MaxDrawdown']*100:.2f}%")

    # Benchmark Comparison
    st.subheader("Benchmark Comparison")

    compare_benchmark = st.checkbox("Compare with Indices", value=False)

    if compare_benchmark:
        col1, col2 = st.columns([1, 2])

        with col1:
            use_net = st.radio(
                "Compare", ["Net Returns", "Gross Returns"], index=0)

            st.markdown("**Select Indices to Compare:**")
            selected_indices = []
            for index_name in BenchmarkComparison.INDIAN_INDICES.keys():
                if st.checkbox(index_name, key=f"idx_{index_name}"):
                    selected_indices.append(index_name)

        with col2:
            if selected_indices:
                if st.button("Fetch Benchmark Data", type="primary"):
                    with st.spinner(f"Fetching data for {len(selected_indices)} indices..."):
                        try:
                            equity_series = report["equity_net"] if use_net == "Net Returns" else report["equity_gross"]
                            initial_capital = report["metadata"]["backtest_config"]["capital"]

                            start_date = equity_series.index.min().strftime("%Y-%m-%d")
                            end_date = equity_series.index.max().strftime("%Y-%m-%d")

                            # Fetch multiple indices
                            benchmark_data = BenchmarkComparison.fetch_multiple_indices(
                                selected_indices, start_date, end_date
                            )

                            if not benchmark_data:
                                st.error(
                                    "[Error] Failed to fetch any benchmark data")
                                st.stop()

                            st.success(
                                f"[OK] Fetched data for {len(benchmark_data)} indices!")

                            # Prepare strategy data
                            strategy_returns = (
                                equity_series - initial_capital) / initial_capital

                            # Create comparison DataFrame
                            comparison_df = pd.DataFrame({
                                "Strategy": strategy_returns * 100
                            })

                            # Add each benchmark
                            comparisons = {}
                            for idx_name, idx_data in benchmark_data.items():
                                # Align dates
                                common_dates = strategy_returns.index.intersection(
                                    idx_data.index)
                                if len(common_dates) >= 2:
                                    aligned_benchmark = idx_data.reindex(
                                        common_dates)
                                    aligned_strategy = strategy_returns.reindex(
                                        common_dates)

                                    comparison_df[idx_name] = aligned_benchmark * 100

                                    # Calculate comparison metrics
                                    try:
                                        strategy_total = float(
                                            aligned_strategy.iloc[-1])
                                        benchmark_total = float(
                                            aligned_benchmark.iloc[-1])
                                        excess_return = strategy_total - benchmark_total

                                        comparisons[idx_name] = {
                                            "strategy_return": strategy_total,
                                            "benchmark_return": benchmark_total,
                                            "excess_return": excess_return,
                                        }
                                    except (ValueError, TypeError):
                                        comparisons[idx_name] = {
                                            "strategy_return": 0.0,
                                            "benchmark_return": 0.0,
                                            "excess_return": 0.0,
                                        }

                            # Display comparison chart
                            st.markdown("**Returns Comparison Chart**")
                            st.line_chart(comparison_df)

                            # Display comparison table
                            st.markdown("**Performance Comparison**")
                            comparison_table = []
                            for idx_name, metrics in comparisons.items():
                                comparison_table.append({
                                    "Index": idx_name,
                                    "Strategy Return (%)": f"{metrics['strategy_return']*100:.2f}",
                                    "Index Return (%)": f"{metrics['benchmark_return']*100:.2f}",
                                    "Excess Return (%)": f"{metrics['excess_return']*100:.2f}",
                                    "Outperformed": "[OK]" if metrics['excess_return'] > 0 else "[Error]"
                                })

                            if comparison_table:
                                st.dataframe(pd.DataFrame(
                                    comparison_table), use_container_width=True)

                                # Summary insights
                                outperformed = sum(
                                    1 for c in comparisons.values() if c['excess_return'] > 0)
                                total = len(comparisons)

                                if outperformed > total / 2:
                                    st.success(
                                        f" Strategy outperformed {outperformed}/{total} indices!")
                                else:
                                    st.warning(
                                        f" Strategy outperformed {outperformed}/{total} indices")

                        except Exception as e:
                            st.error(
                                f"[Error] Failed to fetch benchmark data: {str(e)}")
                            st.info("Tip: **Troubleshooting tips:**")
                            st.info("• Check your internet connection")
                            st.info("• Try selecting fewer indices")
                            st.info("• Ensure the date range has available data")
            else:
                st.info(" Select one or more indices above to compare")

    # Equity Curves
    st.subheader(" Equity Curves")
    equity_df = pd.DataFrame({
        "Gross": report["equity_gross"],
        "Net": report["equity_net"]
    })
    st.line_chart(equity_df)

    # Ensemble Weights Display
    st.subheader(" Ensemble Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Stage 1: Pair Selection Weights**")
        s1_weights_df = pd.DataFrame([
            {"Model": k, "Weight": f"{v:.3f}", "Percentage": f"{v*100:.1f}%"}
            for k, v in report["metadata"]["stage1_weights"].items()
        ])
        st.dataframe(s1_weights_df, use_container_width=True, hide_index=True)

        # Visual representation
        import plotly.express as px
        try:
            fig1 = px.pie(
                values=list(report["metadata"]["stage1_weights"].values()),
                names=list(report["metadata"]["stage1_weights"].keys()),
                title="Stage 1 Weight Distribution"
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            # Fallback if plotly chart fails
            st.warning(f"Could not render Stage 1 pie chart: {e}")

    with col2:
        st.markdown("**Stage 2: Entry/Exit Weights**")
        s2_weights_df = pd.DataFrame([
            {"Model": k, "Weight": f"{v:.3f}", "Percentage": f"{v*100:.1f}%"}
            for k, v in report["metadata"]["stage2_weights"].items()
        ])
        st.dataframe(s2_weights_df, use_container_width=True, hide_index=True)

        # Visual representation
        try:
            fig2 = px.pie(
                values=list(report["metadata"]["stage2_weights"].values()),
                names=list(report["metadata"]["stage2_weights"].keys()),
                title="Stage 2 Weight Distribution"
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            # Fallback if plotly chart fails
            st.warning(f"Could not render Stage 2 pie chart: {e}")

    # Trades
    st.subheader(" Trade Log")
    if not report["trades"].empty:
        st.dataframe(report["trades"], use_container_width=True)

        # Download trades
        st.download_button(
            label="Download Trades CSV",
            data=report["trades"].to_csv().encode("utf-8"),
            file_name=f"trades_{selected_report.run_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades recorded for this run.")

    # Delete report option
    st.divider()
    if st.button(" Delete This Report", type="secondary"):
        if report_mgr.delete_report(selected_report.run_id):
            st.success(
                f"Report {selected_report.run_id} deleted successfully!")
            st.rerun()
        else:
            st.error("Failed to delete report.")


def render_predictions_page():
    """Render the predictions page with real-time trading recommendations.

    This page provides real-time pairs trading recommendations using the same ensemble
    methodology as backtesting but applied to current market data. It offers:

    Key Features:
    - Real-time pair scoring using 5-model ensemble (Stage 1)
    - Current entry/exit signals using 3-model ensemble (Stage 2)  
    - Market regime analysis for trading context
    - Confidence scoring based on data quality and signal consistency
    - Multiple universe selection methods (last simulation, manual, CSV)
    - Customizable strategy weights or inheritance from previous runs
    - Export functionality for further analysis

    Technical Implementation:
    - Fetches latest market data via yfinance API
    - Applies same selectors as backtesting for consistency
    - Generates actionable recommendations with risk metrics
    - Provides market context through volatility and correlation analysis

    Performance Optimizations:
    - Reduced lookback periods for faster computation
    - Efficient data caching and reuse
    - Graceful error handling with informative feedback
    - Progressive loading with status indicators
    """
    # Add description and quick stats
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        **Get real-time trading recommendations** based on your current strategy configuration. 
        Analyze market conditions and identify potential pairs without running full backtests.
        """)

    with col2:
        st.metric("Update Frequency", "Real-time")

    with col3:
        st.metric("Lookback Period", "252 days")

    st.divider()

    # Configuration section
    st.subheader("Prediction Configuration")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Universe selection (simplified)
        st.markdown("**Stock Universe**")
        universe_mode = st.radio(
            "Select universe",
            ["Use Last Simulation", "Select from Report",
                "Quick Entry", "Upload CSV"],
            horizontal=True,
            help="Choose how to define the stock universe for predictions"
        )

        universe = []
        if universe_mode == "Use Last Simulation":
            # Try to get universe from last report
            report_mgr = get_report_manager()
            reports = get_cached_reports_list()
            if reports:
                latest_report = load_cached_report(reports[0].run_id)
                universe = latest_report["metadata"]["universe"]
                st.success(
                    f"[OK] Using {len(universe)} stocks from latest simulation")
                st.caption(
                    f"Stocks: {', '.join(universe[:5])}{' ...' if len(universe) > 5 else ''}")
            else:
                st.warning(
                    "No previous simulations found. Please use Quick Entry.")
                universe_mode = "Quick Entry"

        elif universe_mode == "Select from Report":
            # Allow selection from any saved report
            report_mgr = get_report_manager()
            reports = get_cached_reports_list()
            if reports:
                report_options = [
                    f"{r.run_id[:8]}... • {r.timestamp[:19]} • {len(r.universe)} stocks"
                    for r in reports
                ]
                selected_report_idx = st.selectbox(
                    "Choose a report to use its settings",
                    options=list(range(len(reports))),
                    format_func=lambda i: report_options[i],
                    help="Select a report to inherit its universe, weights, and parameters"
                )

                if selected_report_idx is not None:
                    selected_report = reports[selected_report_idx]
                    report_data = load_cached_report(selected_report.run_id)

                    # Extract settings from report
                    universe = report_data["metadata"]["universe"]

                    # Store report settings in session state for weight inheritance
                    st.session_state["selected_report_settings"] = {
                        "universe": universe,
                        "stage1_weights": report_data["metadata"]["stage1_weights"],
                        "stage2_weights": report_data["metadata"]["stage2_weights"],
                        "run_id": selected_report.run_id
                    }

                    st.success(
                        f"[OK] Using settings from report {selected_report.run_id[:8]}...")
                    st.caption(f"Universe: {len(universe)} stocks")
                    st.caption(
                        f"Stocks: {', '.join(universe[:5])}{' ...' if len(universe) > 5 else ''}")
            else:
                st.warning(
                    "No saved reports found. Please run a simulation first.")
                universe_mode = "Quick Entry"
        
        # Add helpful stock symbol reference dropdown
        with st.expander("NSE Stock Symbol Reference", expanded=False):
            st.markdown("### Quick Reference: Copy symbols from sectors below")
            st.info("**Tip:** Select a sector below and copy the symbols to paste above")
            
            # Sector selection
            sector_pred = st.selectbox(
                "Select Sector:",
                options=list(NSE_STOCK_SYMBOLS.keys()),
                index=len(NSE_STOCK_SYMBOLS) - 1,  # Default to "Top 30 Liquid Stocks"
                key="pred_sector_select"
            )
            
            if sector_pred:
                sector_data = NSE_STOCK_SYMBOLS[sector_pred]
                st.caption(sector_data["description"])
                
                # Format symbols for easy copying
                symbols_text = format_symbols_for_input(sector_data["symbols"])
                
                # Display in a text area for easy selection and copying
                st.text_area(
                    f"Symbols ({len(sector_data['symbols'])} stocks)",
                    value=symbols_text,
                    height=100,
                    key="pred_symbols_display",
                    help="Select all (Ctrl+A) and copy (Ctrl+C)"
                )

        if universe_mode == "Quick Entry":
            txt = st.text_area(
                "Enter NSE symbols (comma-separated)",
                value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, BHARTIARTL, ASIANPAINT",
                height=100,
                help="Enter NSE symbols without .NS suffix"
            )
            universe = [t.strip().upper() for t in txt.split(",") if t.strip()]

        elif universe_mode == "Upload CSV":
            f = st.file_uploader("Upload CSV with 'Ticker' column", type=[
                                 "csv"], key="pred_csv")
            if f is not None:
                df = pd.read_csv(f)
                if "Ticker" in df.columns:
                    universe = [str(x).strip().upper()
                                for x in df["Ticker"].tolist() if str(x).strip()]
                    st.success(f"[OK] Loaded {len(universe)} tickers from file")
                else:
                    st.error("[Error] CSV must have a 'Ticker' column")

    with col2:
        # Strategy weights (simplified)
        st.markdown("**Strategy Weights**")

        # Get weights from selected report or last simulation or use defaults
        s1_weights = {
            "Correlation": 0.25,
            "Distance (Gatev)": 0.15,
            "Cointegration": 0.25,
            "Combined Criteria": 0.1,
            "Supervised ML": 0.15,
            LSTMSelector.name: 0.1,
            TransformerSelector.name: 0.1,
            GNNSelector.name: 0.1,
        }
        s2_weights = {
            "Mean Reversion (±2σ)": 0.5,
            "OU Model": 0.5,
            "Kalman Hedge": 0.0,
            "ML Signal": 0.0,
        }

        # Check if we have selected report settings
        if "selected_report_settings" in st.session_state:
            report_settings = st.session_state["selected_report_settings"]
            s1_weights = report_settings["stage1_weights"]
            s2_weights = report_settings["stage2_weights"]
            st.info(
                f"Using weights from report {report_settings['run_id'][:8]}...")
        else:
            # Try to get weights from last report
            report_mgr = get_report_manager()
            reports = get_cached_reports_list()
            if reports:
                latest_report = load_cached_report(reports[0].run_id)
                s1_weights = latest_report["metadata"]["stage1_weights"]
                s2_weights = latest_report["metadata"]["stage2_weights"]
                st.info("Using weights from latest simulation")

        use_custom_weights = st.checkbox("Customize weights", value=False)

        if use_custom_weights:
            st.markdown("**Stage 1 Weights (Pair Selection)**")
            s1_weights = {
                "Correlation": st.slider("Correlation", 0.0, 1.0, s1_weights.get("Correlation", 0.25), 0.05),
                "Distance (Gatev)": st.slider("Distance", 0.0, 1.0, s1_weights.get("Distance (Gatev)", 0.15), 0.05),
                "Cointegration": st.slider("Cointegration", 0.0, 1.0, s1_weights.get("Cointegration", 0.25), 0.05),
                "Combined Criteria": st.slider("Combined", 0.0, 1.0, s1_weights.get("Combined Criteria", 0.1), 0.05),
                "Supervised ML": st.slider("ML", 0.0, 1.0, s1_weights.get("Supervised ML", 0.15), 0.05),
                LSTMSelector.name: st.slider("LSTM/BiLSTM", 0.0, 1.0, s1_weights.get(LSTMSelector.name, 0.1), 0.05),
                TransformerSelector.name: st.slider("Transformer", 0.0, 1.0, s1_weights.get(TransformerSelector.name, 0.1), 0.05),
                GNNSelector.name: st.slider("GNN", 0.0, 1.0, s1_weights.get(GNNSelector.name, 0.1), 0.05),
            }

            st.markdown("**Stage 2 Weights (Entry/Exit)**")
            s2_weights = {
                "Mean Reversion (±2σ)": st.slider("Z-Score", 0.0, 1.0, s2_weights.get("Mean Reversion (±2σ)", 0.5), 0.05),
                "OU Model": st.slider("OU Model", 0.0, 1.0, s2_weights.get("OU Model", 0.5), 0.05),
                "Kalman Hedge": st.slider("Kalman", 0.0, 1.0, s2_weights.get("Kalman Hedge", 0.0), 0.05),
                "ML Signal": st.slider("ML Signal", 0.0, 1.0, s2_weights.get("ML Signal", 0.0), 0.05),
            }
        else:
            # Display current weights
            st.json({"Stage 1": s1_weights, "Stage 2": s2_weights})

    # Normalize weights
    s1_weights = normalize_weights(s1_weights)
    s2_weights = normalize_weights(s2_weights)

    # Prediction controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        top_k = st.number_input("Top recommendations",
                                min_value=5, max_value=20, value=10, step=1)
    with col2:
        min_data_points = st.number_input(
            "Min data points", min_value=50, max_value=500, value=100, step=10)
    with col3:
        refresh_btn = st.button(" Get Predictions",
                                type="primary", use_container_width=True)

    # Generate predictions
    if refresh_btn and universe:
        with st.spinner("Fetching market data and generating predictions..."):
            try:
                # Initialize prediction engine
                engine = PredictionEngine(lookback_days=252)

                # Generate predictions (simplified without caching to avoid serialization issues)
                result = engine.get_predictions(
                    universe=universe,
                    stage1_weights=s1_weights,
                    stage2_weights=s2_weights,
                    top_k=top_k,
                    min_data_points=min_data_points,
                )

                if not result.recommendations:
                    st.error(
                        "[Error] No recommendations generated. Check your universe and try again.")
                    st.stop()

                # Store result in session state
                st.session_state["prediction_result"] = result
                st.success(
                    f"[OK] Generated {len(result.recommendations)} recommendations!")

            except Exception as e:
                st.error(f"[Error] Prediction failed: {str(e)}")
                st.info("Tip: **Troubleshooting tips:**")
                st.info("• Check your internet connection")
                st.info("• Verify stock symbols are correct")
                st.info("• Try with fewer stocks or longer lookback period")
                st.stop()

    # Display results if available
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]

        # Market overview
        st.subheader(" Market Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall Volatility",
                      f"{result.market_regime.overall_volatility*100:.1f}%")
        with col2:
            st.metric("Correlation Regime",
                      result.market_regime.correlation_regime)
        with col3:
            st.metric("Data Freshness", result.data_freshness)
        with col4:
            st.metric("Pairs Analyzed", result.total_pairs_analyzed)

        # Recommendations table
        st.subheader("Top Pair Recommendations")

        if result.recommendations:
            # Create recommendations dataframe
            rec_data = []
            for rec in result.recommendations:
                # Signal interpretation
                signal_strength = abs(rec.ensemble_signal)
                if signal_strength > 0.7:
                    signal_desc = "[Off] Strong" if rec.ensemble_signal < 0 else "[On] Strong"
                elif signal_strength > 0.3:
                    signal_desc = "[Warn] Moderate" if rec.ensemble_signal < 0 else "[Warn] Moderate"
                else:
                    signal_desc = "[Neutral] Weak"

                # Direction
                if rec.ensemble_signal > 0.1:
                    direction = f"Long {rec.pair.a} / Short {rec.pair.b}"
                elif rec.ensemble_signal < -0.1:
                    direction = f"Short {rec.pair.a} / Long {rec.pair.b}"
                else:
                    direction = "Neutral"

                rec_data.append({
                    "Rank": rec.rank,
                    "Pair": f"{rec.pair.a} / {rec.pair.b}",
                    "Score": f"{rec.score:.3f}",
                    "Signal": signal_desc,
                    "Direction": direction,
                    "Z-Score": f"{rec.z_score:.2f}",
                    "Correlation": f"{rec.correlation:.2f}",
                    "Confidence": f"{rec.confidence:.2f}",
                    "Price A": f"₹{rec.last_price_a:.2f}",
                    "Price B": f"₹{rec.last_price_b:.2f}",
                })

            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, use_container_width=True)

            # Download recommendations
            st.download_button(
                label=" Download Recommendations CSV",
                data=rec_df.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            # Detailed analysis for top recommendation
            if result.recommendations:
                st.subheader(" Top Recommendation Analysis")
                top_rec = result.recommendations[0]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(f"**{top_rec.pair.a} / {top_rec.pair.b}**")
                    st.metric("Pair Score", f"{top_rec.score:.3f}")
                    st.metric("Ensemble Signal",
                              f"{top_rec.ensemble_signal:.2f}")
                    st.metric("Current Z-Score", f"{top_rec.z_score:.2f}")
                    st.metric("Confidence", f"{top_rec.confidence:.2f}")

                with col2:
                    st.markdown("**Individual Model Signals**")
                    for model_name, signal in top_rec.signals.items():
                        signal_color = "[On]" if signal > 0.1 else "[Off]" if signal < -0.1 else "[Neutral]"
                        st.write(f"{signal_color} {model_name}: {signal:.2f}")

                    st.markdown("**Risk Metrics**")
                    st.write(f" Volatility: {top_rec.volatility*100:.1f}%")
                    st.write(f" Correlation: {top_rec.correlation:.2f}")
                    st.write(
                        f" Current Spread: ₹{top_rec.current_spread:.2f}")

        else:
            st.info(
                "No recommendations available. Click 'Get Predictions' to generate recommendations.")

    elif not universe:
        st.info(
            " Please select a stock universe and click 'Get Predictions' to start.")
    else:
        st.info(" Click 'Get Predictions' to generate real-time recommendations.")


# ---------------------------------------------
# Simulator page
# ---------------------------------------------

def simulator_page():
    # Add description and quick stats
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        **Configure your pairs trading strategy** by selecting stocks, adjusting model weights, 
        and setting backtest parameters. All runs are automatically saved for analysis.
        """)

    with col2:
        if "report_manager" in st.session_state:
            report_count = len(
                st.session_state["report_manager"].list_reports())
            st.metric("Saved Reports", report_count)

    with col3:
        st.metric("Default Capital", "₹1,00,000")

    st.divider()

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
            st.warning(
                "Please provide NSE tickers (e.g., RELIANCE, TCS, INFY,…)")
            st.stop()
        try:
            prices = YFinanceNSESource().get_prices(universe, data_cfg)
        except Exception as e:
            st.error(f"Data load failed: {e}")
            st.stop()

        # Update universe to only include tickers that were successfully downloaded
        successful_tickers = list(prices.columns)
        failed_tickers = [t for t in universe if t not in successful_tickers]
        
        if failed_tickers:
            st.warning(
                f"Failed to download data for {len(failed_tickers)} ticker(s): "
                f"{', '.join(failed_tickers)}"
            )
        
        # Use only successful tickers for pair selection
        universe = successful_tickers
        
        if len(universe) < 2:
            st.error(
                f"Not enough valid tickers to form pairs. Need at least 2, got {len(universe)}. "
                "Please check your ticker symbols or try adding .NS suffix manually."
            )
            st.stop()

        st.success(
            f"Loaded prices for {len(prices.columns)} tickers, {len(prices)} rows from "
            f"{prices.index.min().date()} to {prices.index.max().date()}."
        )
        st.dataframe(prices.tail())

        # Stage 1: Pair Selection
        st.header("Stage 1: Pairs Selection")
        candidates = [Pair(universe[i], universe[j]) for i in range(
            len(universe)) for j in range(i + 1, len(universe))]
        st.caption(f"Candidates: {len(candidates)} pairs")

        selectors: Dict[str, PairSelector] = {
            CorrelationSelector.name: CorrelationSelector(lookback=252),
            DistanceSelector.name: DistanceSelector(lookback=252, mode="zscore"),
            CointegrationSelector.name: CointegrationSelector(lookback=504, pvalue_threshold=0.05),
            CombinedCriteriaSelector.name: CombinedCriteriaSelector(),
            MLSelector.name: MLSelector(),
            LSTMSelector.name: LSTMSelector(),
            TransformerSelector.name: TransformerSelector(),
            GNNSelector.name: GNNSelector(),
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
            MLSignal.name: MLSignal(),
        }
        res = backtest_pairs(
            prices, [ps.pair for ps in top_pairs], entry_models, s2_weights, bt_cfg)

        # Backward-compat shim: if engine returns old shape, synthesize gross/net
        if not hasattr(res, "equity_gross"):
            st.warning("Backtest engine returned an incompatible result format. Ensure core.backtest is updated.")
            st.stop()

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

        st.line_chart(pd.concat([res.equity_gross.rename(
            "Gross"), res.equity_net.rename("Net")], axis=1))

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
                    res.pnl_gross, res.turnover, res.params[
                        "notional_each"], bt_cfg.capital, bt_cfg.periods_per_year, costs_i
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
            data=pd.concat([res.equity_gross, res.equity_net],
                           axis=1).to_csv().encode("utf-8"),
            file_name="equity_curves.csv",
            mime="text/csv",
        )

        # Save report to disk
        report_mgr = get_report_manager()
        run_id = report_mgr.save_report(
            result=res,
            universe=universe,
            data_config={
                "start": str(data_cfg.start),
                "end": str(data_cfg.end),
                "freq": data_cfg.freq,
                "price_field": data_cfg.price_field,
            },
            stage1_weights=s1_weights,
            stage2_weights=s2_weights,
            backtest_config={
                "capital": bt_cfg.capital,
                "max_concurrent_pairs": bt_cfg.max_concurrent_pairs,
                "per_trade_cap": bt_cfg.per_trade_cap,
                "periods_per_year": bt_cfg.periods_per_year,
                "soft_stop_z": bt_cfg.soft_stop_z,
                "soft_stop_decay": bt_cfg.soft_stop_decay,
                "soft_stop_persist_bars": bt_cfg.soft_stop_persist_bars,
            },
        )
        st.success(f" Report saved successfully!")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"**Run ID:** `{run_id}`")
        with col2:
            st.info(" View in Reports page →")

        # Quick metrics summary
        st.markdown("**Quick Summary:**")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        with quick_col1:
            st.metric("Net Return",
                      f"{res.metrics.get('Net.Return', 0)*100:.1f}%")
        with quick_col2:
            st.metric("Net Sharpe", f"{res.metrics.get('Net.Sharpe', 0):.2f}")
        with quick_col3:
            st.metric("Max Drawdown",
                      f"{res.metrics.get('Net.MaxDrawdown', 0)*100:.1f}%")
        with quick_col4:
            st.metric("Trades", f"{res.metrics.get('Turnover.Trades', 0)}")

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
    st.set_page_config(
        page_title="NSE Pairs Trading Simulator",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=""
    )

    # Professional CSS Theme
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: #f8fafc; }
        .stSidebar { background-color: #1e293b !important; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; color: #f1f5f9; }
        .stMetric { background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; border: 1px solid #334155; }
        .stButton button { background-color: #2563eb; color: white; border: none; border-radius: 0.375rem; }
        .stButton button:hover { background-color: #1d4ed8; }
        div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)

    # Main navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Simulator", "Predictions", "Reports"],
        index=0,
        help="Navigate between simulation, predictions, and report analysis"
    )

    if "Reports" in page:
        st.title("Trading Reports & Analysis")
        render_reports_page()
    elif "Predictions" in page:
        st.title("Real-time Predictions & Recommendations")
        render_predictions_page()
    else:
        st.title("NSE Pairs Trading Simulator")
        simulator_page()


if __name__ == "__main__":
    main()
