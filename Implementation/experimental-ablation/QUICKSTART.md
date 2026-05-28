# Multi-Market Experimental Ablation — Quick Start

## 🚀 Quick Start (Automated Pipeline)

Run all 4 markets end-to-end:

```bash
cd experimental-ablation/scripts
python run_all_markets.py --folds 6
```

**Est runtime:** 40-50 hours total (10-12h per market)

**Outputs:**
- `results/{market}/wfv_*.json` — WFV results per market
- `results/cross_market_summary/*.csv` — Comparative tables
- `results/cross_market_summary/plots/*.png` — Visualizations

---

## 📋 Manual Step-by-Step

### Step 1: Fetch Data (~5 min per market)

```bash
cd experimental-ablation/scripts

# All markets at once
python fetch_market_data.py --markets india us brazil uk

# Or one at a time
python fetch_market_data.py --market us
```

**Cached to:** `experimental-ablation/data/{market}/prices_*.parquet`

### Step 2: Run WFV (~10-12 hours per market)

```bash
# Single market, 6 folds, all 8 selectors
python run_multi_market_wfv.py --market us --n_folds 6

# Subset of selectors (faster testing)
python run_multi_market_wfv.py --market us --selectors lstm correlation --n_folds 2
```

**Outputs:** `results/{market}/wfv_*.json`

**Parallelization:** Run multiple markets in separate terminals (if RAM allows):

```bash
# Terminal 1
python run_multi_market_wfv.py --market us --n_folds 6

# Terminal 2
python run_multi_market_wfv.py --market brazil --n_folds 6

# Terminal 3
python run_multi_market_wfv.py --market uk --n_folds 6

# Terminal 4
python run_multi_market_wfv.py --market india --n_folds 6
```

### Step 3: Compare Results

```bash
python compare_markets.py --markets india us brazil uk
```

**Outputs:**
- `results/cross_market_summary/summary_stats.csv`
- `results/cross_market_summary/cost_sensitivity.csv`
- `results/cross_market_summary/regime_performance.csv`
- `results/cross_market_summary/parsimony_test.csv` (placeholder)

### Step 4: Visualize

```bash
python visualize_cross_market.py --markets india us brazil uk
```

**Outputs:**
- `results/cross_market_summary/plots/cost_sensitivity.png`
- `results/cross_market_summary/plots/regime_heatmap.png`
- `results/cross_market_summary/plots/degradation_bars.png`
- `results/cross_market_summary/plots/summary_table.png`

---

## 🔬 Experimental Design

### Research Questions

**RQ1:** Does selector ranking stay stable across markets?  
**RQ2:** Does parsimony (LSTM+Corr > 8-selector) hold globally?  
**RQ3:** How does transaction cost impact scale?  
**RQ4:** Do regime patterns repeat cross-market?

### Markets

| Market | Cost | Stocks | Index |
|--------|------|--------|-------|
| US | 5 bps | 35 | S&P 100 |
| UK | 10 bps | 35 | FTSE 100 |
| Brazil | 30 bps | 35 | Bovespa |
| India | 60 bps | 35 | Nifty 50 |

### Walk-Forward Validation

6 folds, 2020-2025:
- Fold 1: Train 2020, Test H1 2021
- Fold 2: Train H2 2020–H1 2021, Test H2 2021
- Fold 3: Train 2021, Test H1 2022
- Fold 4: Train H2 2021–H1 2022, Test H2 2022
- Fold 5: Train 2022, Test H1 2023
- Fold 6: Train H2 2022–H1 2023, Test H2 2023–2024

---

## 📊 Expected Outputs

### Summary Stats
```
Market         Code  Folds  Avg_Net_Sharpe      Avg_Gross_Sharpe  Cost_bps
United States  US    6      0.523 ± 0.142       0.581             5.0
United Kingdom UK    6      0.412 ± 0.189       0.473             10.0
Brazil         IBOV  6      0.287 ± 0.224       0.384             30.0
India          NSE   6      0.152 ± 0.198       0.341             60.0
```

### Cost Sensitivity
Clear inverse relationship between transaction cost and net Sharpe ratio.

### Regime Performance
Fold-level heatmap showing which market+period combinations perform best.

---

## ⚠️ Limitations

1. **Survivorship bias:** Using 2025 constituents backdated to 2020
2. **Short-selling:** Assumes frictionless shorting (overstates Brazil/India)
3. **Data quality:** yfinance may have gaps for emerging markets
4. **Computational:** 40-50 hours runtime for full pipeline
5. **Parsimony test:** Requires additional ablation runs (2-selector, single-selector)

---

## 🛠️ Troubleshooting

### "Cache not found" error
```bash
# Re-run data fetch
python fetch_market_data.py --market us --force
```

### Import errors
```bash
# Ensure parent directory is in Python path
cd experimental-ablation/scripts
python run_multi_market_wfv.py --market us
```

### RAM issues
- Run markets sequentially instead of parallel
- Reduce `n_folds` for faster testing (e.g., `--n_folds 2`)
- Use subset of selectors (e.g., `--selectors lstm correlation`)

### yfinance download fails
- Check internet connection
- Some tickers may be delisted/renamed — check `failed_tickers` list
- Use `--force` to refresh cache

---

## 📚 Related Files

- `../configs/{market}.yaml` — Market configurations
- `../README.md` — Full methodology and research design
- `../../experiments/walk_forward.py` — Original thesis WFV script (Indian market)
- `../../experiments/config.py` — Selector/signal weights

---

**Author:** Yash Sarang  
**Date:** May 29, 2026  
**Agent:** TARS (Hermes)
