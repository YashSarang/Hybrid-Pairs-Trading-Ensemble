# Project Narrative — Hybrid Pairs Trading Ensemble
**Complete research history: what we did, why, what we found**
Last updated: 2026-06-04

---

## 1. PROJECT GENESIS

**Research question:** Can combining classical statistical arbitrage methods with modern ML in an ensemble framework produce more robust out-of-sample pair selection for NSE equities?

**Why NSE?** Indian large-cap equities (Nifty 100) are liquid, yfinance-accessible, and underrepresented in pairs trading literature. The high-friction cost environment (STT, stamp duty) also tests whether gross alpha survives realistic transaction costs.

**Initial stack:** Python, Streamlit UI, yfinance, XGBoost, TensorFlow (LSTM/Transformer/GNN), Kalman Filter.

**Two-stage pipeline concept:**
- Stage 1: 8 algorithms score every candidate pair — 4 statistical + 4 ML
- Stage 2: 4 signal models generate entry/exit signals per selected pair
- Backtest: vectorized PnL with realistic IndianCosts model

---

## 2. PHASE 1 — SYSTEM BUILD + BASELINE EXPERIMENTS

### Cost Model Discovery (E0 — implicit)
Initial cost model had ~22.9 bps round-trip. Research into 2024-2026 NSE discount broker rates (Zerodha/Groww model) revealed the code overestimated by ~40%. Brokerage = 0 bps (flat-fee brokers), stamp duty = 1.5 bps (not 1.0). Corrected to 16.28 bps round-trip. All final thesis results use corrected model. Old results backed up to `experiments/results/backup_old_costs/`.

### E1 — Frequency Comparison (Why daily?)
**Why:** Needed to justify daily vs hourly data empirically.
**Setup:** 700-day window, 34 tickers, stat-only selectors, same cost model.
**Result:**
- Daily: Gross SR 1.144, Net SR −2.294, 672 trades/yr, Hurst 0.190
- Hourly: Gross SR 0.488, Net SR −6.554, 904 trades/yr, Hurst 0.251
- Hourly strategy bankrupt (Max DD 214%)

**Key insight:** Pairs are genuinely mean-reverting (Hurst 0.19) but signal over-trades. Cost drag = 16.29pp/yr at 672 trades. Daily chosen for all subsequent work.

### E2 — Hold Period Sweep (Why min_hold=30?)
**Why:** E1 showed cost drag was the killer. Hold period is the lever.
**Setup:** Swept min_hold ∈ {0,5,10,15,20,25,30,40} on 10-year daily data.
**Result:**
| Hold | Net SR | Trades/yr | Cost Drag |
|------|--------|-----------|-----------|
| 0 | −1.889 | 746 | bankrupt |
| 20 | +0.087 | 200 | 3.33pp |
| **30** | **+0.481** | **156** | **2.09pp** |
| 40 | −0.239 | 134 | 3.25pp |

**Why 30 is the peak:** Matches OU mean-reversion half-life of ~20-30 days. Hold=40 overshoots reversion; position often held through counter-move. **min_hold=30 locked as DEFAULT in config.py.**

### E3 — Ablation Study (What drives performance?)
**Why:** Prove ensemble > individual components. Thesis central claim.
**Setup:** 6 WFV folds (2020–2025), each selector/signal in isolation.

**Stage 1 ablation (stat-only mode, Full-OOS Net SR):**
| Config | Net SR |
|--------|--------|
| Cointegration_only | +0.119 |
| Combined_only | +0.119 (identical to Cointegration) |
| Correlation_only | −0.091 |
| Distance_only | −0.070 |
| S1_Ensemble | −0.189 |

**Finding:** Cointegration=Combined (same top-10 every fold). Ensemble underperforms best individual. Effective diversity = 3 selectors, not 4.

**Stage 2 ablation (Full-OOS Net SR):**
| Config | Net SR | Trades/yr |
|--------|--------|-----------|
| **OU_only** | **+0.359** | **87** |
| Kalman_only | −0.053 | 117 |
| ZScore_only | −0.156 | 116 |
| ML_only | −0.401 | 112 |
| S2_Ensemble | −0.189 | 142 |

**Finding:** OU dominates. MLSignal (XGBoost) is worst — overfit in-sample features don't generalise. Equal-weight ensemble dragged down by ML. **OU-only selected as headline Stage 2.**

**Full-mode ablation (8 selectors, Full-OOS Net SR):**
| Config | Net SR |
|--------|--------|
| LSTM_only | +0.305 |
| Correlation_only | +0.151 |
| Distance_only | −0.165 |
| ML_only | −0.192 |
| Cointegration_only | −0.289 |
| GNN_only | −0.448 |
| Combined_only | −0.824 |
| S1_Ensemble (8) | −0.719 |

**Finding:** Equal-weight 8-selector ensemble is catastrophically bad. Bad selectors dilute good ones. This is a publishable negative result.

**Bug fixed here:** TransformerSelector used Lambda layer with captured tf.constant — failed on CUDA cluster. Fixed by replacing with `_PositionalEncodingLayer` class (2026-04-06).

### E4 — Walk-Forward Validation
**Setup:** 6 expanding-window folds (train starts 2016-01-01; test = each year 2020-2025).

Stat-only + OU-only results:
| Fold | Year | Net SR |
|------|------|--------|
| 1 | 2020 | -0.158 |
| 2 | 2021 | -0.523 |
| 3 | 2022 | +0.548 |
| 4 | 2023 | +0.130 |
| 5 | 2024 | -0.718 |
| 6 | 2025 | +0.757 |
Full-OOS Net SR: +0.359 | Mean: +0.405 +/- 0.578 | 67% folds positive

Full-mode (8 selectors) + OU-only: Full-OOS Net SR +0.067 — DL selectors dilute stat pairs.
stat_ml (XGBoost) + OU-only: Full-OOS Net SR -0.163 — XGBoost MLSelector hurts.

### E5 — Benchmark Comparison
- Beta vs Nifty 50: **0.071** (near-zero = market neutral confirmed)
- Jensen alpha: +2.58%/yr net
- Max DD: -13.4% vs -38.4% (Nifty 50)
- Strategy CAGR 2.43% vs Nifty 13.69%

### E6 — Statistical Significance
- Gross SR p=0.038 — significant at 5%
- Net SR p=0.148 — not significant
- After Bonferroni (x5): OU_only p_adj=0.750 — no config survives

---

## 3. PHASE 2 — FATAL CONFOUND IDENTIFIED

### The Problem
Early thesis claimed "geographic alpha dominates methodology." But:
- NSE baseline used Nifty 100 (35 tickers, diluted quality)
- "India multi-market" used Nifty 50 (35 tickers, blue-chips only)
- Both geography AND universe changed simultaneously -> conclusion invalid

### Salvage Experiments (June 2026)
2x2 control matrix:
| Universe | Method | Net SR |
|----------|--------|--------|
| Nifty 100 | Expanding | -0.409 |
| Nifty 100 | Rolling | +0.052 |
| Nifty 50 | Rolling | +0.752 |
| Nifty 50 | Expanding | +1.064 |

Result: SCENARIO A confirmed — Universe quality (Nifty 50 vs 100) explains +0.700 Sharpe lift.

ML non-determinism discovered:
- Same config (seed=42) on GPU: Sharpes +0.398, -0.386, +0.840 (variance 1.226)
- Fix: CPU-only determinism (CUDA_VISIBLE_DEVICES="", TF_DETERMINISTIC_OPS=1)
- CPU-deterministic range: +0.353 to +0.484 (variance reduced 9x)

---

## 4. PHASE 3 — MULTI-MARKET EXPANSION

All results summary:
| Market | Signal | Runs | Mean Net SR | Notes |
|--------|--------|------|-------------|-------|
| NSE Nifty 50 rolling | ZScore | 2 | +0.908 | Primary; p=0.036 (4-fold) |
| NSE Nifty 50 expanding | ZScore | 2 | +1.064 | |
| NSE Nifty 100 rolling | ZScore | 1 | +0.052 | Baseline |
| NSE Nifty 100 expanding | ZScore | 1 | -0.409 | Failed baseline |
| India multi-mkt GPU | ZScore | 3 | +0.284 mean | GPU non-determinism; best-of-3 = +0.840 |
| US S&P 500 | ZScore | 1 | +0.774 | Exploratory; n=1, regime-contingent |
| Brazil B3 | OU | 3 | +0.107 | Best = +0.321 |
| UK FTSE 100 | ZScore | 2 | +0.010 | First +0.265, second -0.245 |

---

## 5. PHASE 4 — JFM SUBMISSION + CRITIQUE ROUNDS

### Round 1 (11 items fixed)
- 8-fold extension: Mean +0.242, p=0.473 — not significant
- Bootstrap DiD: P(Universe Quality > Methodology) = 0.811
- HAC Newey-West correction (lag=1)
- Fama-French alpha REMOVED (circular: manufactured monthly returns)
- CVaR recomputed from correct 2021-2024 period
- Survivorship x0.92 REMOVED (wrong paper/domain)

### Round 2 (errors introduced in R1 fixed)
3 fatal flaws from Round 1 revisions corrected. Sections 4.4.9/4.4.10 relocated. Pre-registration framework added.

### Round 3
New VAE + copula reference added to abstract (FATAL — not implemented). 8-selector claim uncorrected (CNNSelector disabled -> only 7 active).

### Round 4 — CURRENT STATUS (29 open issues)
Root causes of REJECT:
1. Abstract describes VAE/copula — not in study
2. US ZScore sign wrong (-0.297 vs actual +0.774)
3. 8-selector claim propagated (should be 7 active)
4. 16x multiplier and +0.840 headline still in Ch3/Ch4/Ch5
5. ML overfitting undisclosed
6. Fund launch recommendation in Ch5 (inappropriate)

---

## 6. PHASE 5 — 16-FOLD LONG-RUN VALIDATION

16-fold NSE Nifty 50 (2005-2024): Mean +0.101, p=0.651, 11/16 positive
16-fold NSE Nifty 100 (2005-2024): Mean +0.162, p=0.449, 9/16 positive
Paired comparison: p=0.703 — universe quality not structural over 20 years.

Key insight: 2021-2024 result (+0.752, p=0.036) was regime artefact — post-COVID recovery anomalously favourable. Regime-conditionality is the honest publishable contribution.

---

## 7. CURRENT CRITICAL NUMBERS

| Metric | Value |
|--------|-------|
| Headline Net SR (Nifty50, stat-only, ZScore, rolling, 4-fold) | +0.752 |
| Headline CI | [+0.422, +1.082] |
| Headline p-value | 0.036 |
| Fold-by-fold | +1.127, +0.218, +0.627, +1.036 |
| Total trades (headline) | 126 |
| Active selectors | 7 (CNNSelector disabled) |
| India cost | 16.28 bps round-trip |
| US ZScore (exploratory, n=1) | +0.774 |
| India multi-mkt mean | +0.284 |
| CPU deterministic range | +0.353-+0.484 |
| Honest multiplier | 5.5x (Nifty50 rolling vs Nifty100 rolling) |
| 16-fold Nifty50 mean | +0.101, p=0.651 |
| Model count correct | 7x4x4=112 (not 192) |

---

## 8. VENUE ASSESSMENT

| Venue | Viability | Rationale |
|-------|-----------|-----------|
| JFM | 20% -> 50% after FATAL fixes | No significant 20-year result; 4-fold significant but regime-conditional |
| QF | 35% -> 65% after fixes | Tolerates shorter samples, methodology-focused |
| Emerging Markets Review | Best fit | NSE/India scope, accepts exploratory framing |
| Finance Research Letters | Viable | 4000-word note on regime-conditionality |

---

## 9. KEY BUGS FIXED

| Bug | Fix |
|-----|-----|
| TATAMOTORS.NS YFTzMissingError | Replaced with M&M.NS |
| pandas 'H' deprecated | "1h" |
| xgboost missing | pip install xgboost scikit-learn |
| MLSignal LabelEncoder non-contiguous labels | LabelEncoder remaps 0-based |
| MLSelector._label() always NaN | Removed shift; rolling sum on in-sample data |
| TransformerSelector Lambda+GPU failure | Replaced with _PositionalEncodingLayer class |
| lookback=252 exhausted test windows | Fixed to lookback=126 |
| WFV folds hardcoded | Refactored to dynamic from config n_folds |
| Cost model 22.9 -> 16.28 bps | NSE 2024 rate research |
| Fama-French alpha circular | Section removed |
| CVaR from wrong period | WFV script patched; recomputed |
| Survivorship x0.92 invalid | Numerical correction removed |
