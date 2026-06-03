# Multi-Market Experimental Ablation

**Purpose:** Test whether the hybrid ensemble pairs trading findings from the Indian NSE market generalize to other equity markets.

**Motivation:** The core thesis claims (LSTM+Correlation 2-selector parsimony principle, ensemble construction benefits, deep learning for pair selection) were validated on Indian NSE data (2020-2025). This experimental ablation extends the same methodology to three additional markets to test cross-market generalizability.

---

## Markets

| Market | Index | Stocks | Transaction Cost (Round-Trip) |
|--------|-------|--------|-------------------------------|
| 🇮🇳 India | NSE Nifty 50 | 35 | ~60 bps |
| 🇺🇸 United States | S&P 100 | 35 | ~5 bps |
| 🇧🇷 Brazil | Bovespa (IBOV) | 35 | ~30 bps |
| 🇬🇧 United Kingdom | FTSE 100 | 35 | ~10 bps |

**Date Range:** 2020-01-01 to 2025-05-01 (all markets, for comparability)

**Frequency:** Daily (1D) — consistent with thesis Experiment E1 finding that daily outperforms hourly

---

## Research Questions

### RQ1: Ensemble Ranking Stability
Does the relative performance ranking of the 8 pair selectors remain stable across markets?

**Hypothesis:** Correlation and LSTM selectors will rank highly across all markets; GNN and CombinedCriteria will rank poorly universally.

### RQ2: Parsimony Principle Generalization
Does the 2-selector (LSTM + Correlation) ensemble outperform the 8-selector equal-weight ensemble in all markets?

**Hypothesis:** Yes — the parsimony principle should hold whenever negative-alpha selectors contaminate the ensemble.

### RQ3: Transaction Cost Sensitivity
How does the gross-to-net alpha gap scale with transaction costs across the 4 markets?

**Hypothesis:** Sharpe ratio degradation will be proportional to cost (US minimal, India maximal).

### RQ4: Regime Conditional Performance
Do pairs trading strategies exhibit similar regime-conditional patterns across markets (e.g., underperformance during persistent trends)?

**Hypothesis:** Yes — mean-reversion strategies should underperform during strong directional trends globally.

---

## Directory Structure

```
experimental-ablation/
├── README.md                  # This file
├── configs/
│   ├── india.yaml             # NSE universe + cost model (baseline from thesis)
│   ├── us.yaml                # S&P 100 universe + cost model
│   ├── brazil.yaml            # Bovespa universe + cost model
│   └── uk.yaml                # FTSE 100 universe + cost model
├── data/
│   ├── india/                 # Cached price data (Parquet)
│   ├── us/
│   ├── brazil/
│   └── uk/
├── results/
│   ├── india/                 # WFV results JSON per market
│   ├── us/
│   ├── brazil/
│   ├── uk/
│   └── cross_market_summary/  # Comparative tables + charts
└── scripts/
    ├── run_multi_market_wfv.py      # Master runner (all 4 markets)
    ├── fetch_market_data.py         # Data downloader (yfinance)
    ├── compare_markets.py           # Cross-market result aggregation
    └── visualize_cross_market.py    # Comparative plots
```

---

## Methodology

### Data Collection
- **Source:** yfinance (Yahoo Finance)
- **Universe:** Top 35 stocks by market cap from each market's primary index
- **Resampling:** Daily close prices, forward-fill holidays/missing
- **Validation:** Min 80% data availability over 2020-2025

### Walk-Forward Validation Design
Same 6-fold WFV design as thesis:

| Fold | Train Start | Train End | Test Start | Test End | Period |
|------|-------------|-----------|------------|----------|--------|
| 1 | 2020-01-01 | 2020-12-31 | 2021-01-01 | 2021-06-30 | Post-Covid |
| 2 | 2020-07-01 | 2021-06-30 | 2021-07-01 | 2021-12-31 | Bull Run |
| 3 | 2021-01-01 | 2021-12-31 | 2022-01-01 | 2022-06-30 | Rate Hike |
| 4 | 2021-07-01 | 2022-06-30 | 2022-07-01 | 2022-12-31 | Inflation Peak |
| 5 | 2022-01-01 | 2022-12-31 | 2023-01-01 | 2023-06-30 | Normalization |
| 6 | 2022-07-01 | 2023-06-30 | 2023-07-01 | 2024-12-31 | Expansion |

**Note:** Regime labels (Post-Covid, Bull Run) are approximate; actual macro conditions differ by market.

### Cost Models

Transaction cost breakdowns by market (source: industry estimates + regulatory docs):

**India (NSE):**
```python
IndianCosts(
    brokerage_bps=3.0,
    exchange_bps=0.345,
    sebi_bps=0.01,
    stt_bps=10.0,
    gst_rate=0.18,
    stamp_bps=1.0,
    slippage_bps=2.0
)  # Total: ~60 bps round-trip
```

**United States:**
```python
USCosts(
    brokerage_bps=0.5,    # Institutional rates
    sec_fee_bps=0.23,     # SEC Section 31 fee
    finra_taf_bps=0.013,  # FINRA TAF
    slippage_bps=2.0
)  # Total: ~5 bps round-trip
```

**Brazil:**
```python
BrazilCosts(
    brokerage_bps=2.5,
    bovespa_fee_bps=0.3,
    settlement_bps=0.25,
    iof_bps=0.38,         # IOF tax (financials)
    slippage_bps=5.0      # Higher due to lower liquidity
)  # Total: ~30 bps round-trip
```

**United Kingdom:**
```python
UKCosts(
    brokerage_bps=1.0,
    stamp_duty_bps=5.0,   # 0.5% on purchases only
    ptm_levy_bps=0.01,    # Panel on Takeovers and Mergers
    slippage_bps=2.0
)  # Total: ~10 bps round-trip
```

### Ensemble Configuration
All markets use identical selector ensemble weights (from thesis `experiments/config.py`):

```python
SELECTOR_WEIGHTS = {
    "correlation": 0.15,
    "distance": 0.10,
    "cointegration": 0.10,
    "combined": 0.05,
    "ml": 0.15,
    "lstm": 0.25,
    "transformer": 0.10,
    "gnn": 0.10,
}
```

Signal models also identical:
```python
SIGNAL_WEIGHTS = {
    "zscore": 0.30,
    "ou": 0.30,
    "kalman": 0.25,
    "ml": 0.15,
}
```

### Evaluation Metrics
Same as thesis Chapter 4:

- **Net Sharpe Ratio** (primary metric)
- Gross Sharpe Ratio
- Max Drawdown
- Win Rate
- Average Trade Duration
- Total Trades
- Turnover (annualized)

---

## Expected Runtime

**Per-market estimate:**
- Data download: ~5 min
- WFV (6 folds × 8 selectors × 4 signals): ~8-12 hours
- Results aggregation: ~5 min

**Total for 4 markets:** ~40-50 hours (parallelizable across markets)

---

## Deliverables

### 1. Results JSON per market
`results/{market}/wfv_all_results.json` — same schema as thesis `experiments/results/`

### 2. Cross-Market Summary Tables
`results/cross_market_summary/`
- `selector_ranking.csv` — Net SR rank per selector per market
- `parsimony_test.csv` — 2-selector vs 8-selector comparison
- `cost_sensitivity.csv` — Gross vs Net Sharpe by market
- `regime_performance.csv` — Per-fold Sharpe by market

### 3. Visualizations
- Heatmap: Selector Net SR across markets
- Bar chart: Parsimony principle (2-sel vs 8-sel) per market
- Line chart: Gross-to-Net degradation vs transaction cost
- Fold-level performance scatter: India vs US vs Brazil vs UK

---

## Limitations & Caveats

1. **Survivorship bias:** Using 2025 index constituents backdated to 2020 (acceptable for ablation study, not for production)
2. **Market microstructure differences:** Circuit breakers, lot sizes, settlement cycles differ — not modeled
3. **Short-selling constraints:** Assume frictionless shorting (may overstate Brazil/India performance)
4. **Data quality:** yfinance may have gaps/errors for emerging markets (Brazil, India)
5. **Currency effects:** Ignoring FX risk (e.g., BRL/USD volatility for Brazilian pairs)
6. **Regulatory differences:** Tax treatment, margin requirements vary — not modeled

---

## Usage

### Step 1: Fetch Data
```bash
cd experimental-ablation/scripts
python fetch_market_data.py --markets india us brazil uk
```

### Step 2: Run WFV (per market)
```bash
python run_multi_market_wfv.py --market us --n_folds 6
```

### Step 3: Compare Results
```bash
python compare_markets.py --markets india us brazil uk
python visualize_cross_market.py --output ../results/cross_market_summary/
```

---

## Integration with Thesis

**Status:** Experimental extension — NOT part of core thesis.

**Thesis scope:** Indian NSE market only (Chapter 4 results)

**This ablation:** Appendix material demonstrating cross-market generalizability (or lack thereof)

**If findings are strong:** Upgrade to Chapter 6 "Cross-Market Validation" in final thesis

---

**Author:** Yash Sarang  
**Date:** May 29, 2026  
**Agent:** TARS (Hermes)
