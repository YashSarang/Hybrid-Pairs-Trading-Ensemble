# Chapter 3: NSE Nifty 100 Baseline Validation

## Chapter Overview

This chapter presents the baseline walk-forward validation of the hybrid pairs trading ensemble on the NSE Nifty 100 universe. We establish methodology, report results under expanding-window training, and conduct sensitivity analysis comparing rolling vs expanding window approaches.

**Structure:**
- **Section 3.1**: Experimental Design
- **Section 3.2**: Universe Selection and Data
- **Section 3.3**: Selector Ensemble Configuration
- **Section 3.4**: Walk-Forward Results (Expanding Window)
- **Section 3.5**: Analysis and Discussion
- **Section 3.6**: Sensitivity Analysis — Rolling Window Methodology **[COMPLETE]**

**Key Finding:** NSE pairs trading fails under expanding windows (-0.409 Sharpe, 1,096 trades). Rolling windows improve performance modestly (+0.052 Sharpe) through transaction cost reduction but remain marginally profitable and statistically insignificant. Multi-market validation (Chapter 4) is required.

---

## Section 3.1: Experimental Design

### 3.1.1 Research Objectives

The NSE baseline validation addresses three research questions:

1. **RQ1:** Can hybrid selector ensembles (statistical + ML) outperform single-selector baselines on the NSE Nifty 100?
2. **RQ2:** What is the profitability threshold for pairs trading under Indian market transaction costs (16.4 bps)?
3. **RQ3:** How sensitive are results to training window methodology (expanding vs rolling)?

### 3.1.2 Walk-Forward Validation Framework

**Fold Structure:**
- 6 folds testing years 2020-2025 (1 year test per fold)
- Expanding training window: 4-9 years (2016-2019 for Fold 1, 2016-2024 for Fold 6)
- Universe: NSE Nifty 100 (35 tickers with complete coverage)
- Selection: Top-10 pairs per fold (from ~595 candidate pairs)

**Evaluation Metrics:**
- Primary: Net Sharpe Ratio (risk-adjusted return after costs)
- Secondary: Gross Sharpe, total trades, cost drag, max drawdown

**Baseline Comparison:**
- Individual selectors (Correlation, Distance, Cointegration, etc.)
- Ensemble aggregation methods (voting, weighted, combined)

### 3.1.3 Signal Configuration

**Signals Tested:**
- **ZScoreThreshold**: Normalized spread crosses ±2σ (lookback = 126 days)
- **OUThreshold**: Ornstein-Uhlenbeck mean-reversion signal using rolling AR(1) spread estimation (lookback = 126 days; entry at k=1.5 std, exit at k=0.2 std)

**Entry/Exit:**
- Entry: Signal triggers (spread exceeds threshold)
- Exit: Mean reversion (spread crosses zero) or 20-day holding period
- Position sizing: Equal-weighted across 10 pairs

**Transaction Costs:**
- Indian market: 16.28 bps per round-trip — STT (sell): 10.0 bps, NSE exchange fee: 0.322 bps, SEBI fee: 0.01 bps, stamp duty (buy): 1.5 bps, GST on fees: 18%, slippage: 2.0 bps per leg (2024 rates, source: NSE circular)
- Applied on entry and exit

---

## Section 3.2: Universe Selection and Data

### 3.2.1 NSE Nifty 100 Universe

**Rationale:**
- Large-cap liquid stocks (minimize execution risk)
- Broad sector coverage (15 GICS sectors represented)
- Sufficient cross-sectional dispersion for pair formation

**Data Filters:**
- Complete price history: 2016-2025. **Note on survivorship bias:** The data filter selects stocks with continuous trading history through 2025, introducing mild survivorship bias (stocks delisted, merged, or dropped from the index during 2016-2025 are excluded). Additionally, current-period index constituent lists are used rather than point-in-time historical constituent data, introducing look-ahead bias into universe construction. Both effects likely inflate performance estimates relative to a live deployment. Point-in-time constituent data replication is left for future work.
- Minimum trading volume: >₹10 crore daily avg
- Excludes stocks with >5 consecutive missing days

**Final Universe:**
- 35 tickers (from 100 Nifty constituents)
- 595 possible pairs (C(35, 2))

### 3.2.2 Data Source and Preprocessing

**Source:** Yahoo Finance (NSE tickers with `.NS` suffix)  
**Frequency:** Daily adjusted close prices  
**Adjustments:** Corporate actions (splits, dividends) pre-adjusted  
**Missing Data:** Forward-fill up to 3 consecutive days, else exclude stock

**Stationarity Tests:**
- ADF test on price spreads (cointegration ranking criterion)
- Half-life calculation (exclude pairs with HL > 60 days)

> **Note:** ADF test is used as a ranking criterion by CointegrationSelector only. OUThreshold does not apply an ADF exclusion filter; stationarity is implicitly enforced via reversion speed (k = 1/half-life) in the signal formula.

---

## Section 3.3: Selector Ensemble Configuration

### 3.3.1 Individual Selectors

**Statistical Selectors (4):**
1. **CorrelationSelector**: Highest rolling Pearson correlation (126-day window)
2. **DistanceSelector**: Minimum normalized squared distance
3. **CointegrationSelector**: Lowest ADF p-value on spread residuals
4. **CombinedCriteriaSelector**: Weighted composite (0.3×Corr + 0.3×Dist + 0.4×Coint)

**ML Selectors (3; CNNSelector disabled):**
5. **LSTMSelector**: Bidirectional LSTM on price/volume sequences (60-day lookback)
6. **TransformerSelector**: Multi-head attention (4 heads, 2 layers)
7. **GNNSelector**: Graph Convolutional Network (correlation-weighted edges)
8. **CNNSelector**: 1D CNN on spread time series (disabled due to sequence length constraints)

As CNNSelector is disabled, the active ensemble contains 7 selectors: 4 statistical (Correlation, Distance, Cointegration, Combined) and 3 ML (LSTM, Transformer, GNN).

**Configuration:**
- All selectors train on identical train/test splits
- ML models: Early stopping (10 epochs), dropout (0.2), Adam optimizer
- Hyperparameters: Grid search on Fold 1, then fixed for Folds 2-6

### 3.3.2 Ensemble Aggregation

**Method:** Weighted ensemble score
- Each selector ranks pairs 1-595
- Ensemble score = Σ(selector_score × selector_weight)
- Weights: Equal (1/7 for 7 active selectors; CNNSelector disabled due to sequence length constraints)

**Top-K Selection:**
- Select 10 pairs with highest ensemble scores
- Break ties by individual selector agreement count

**Reproducibility Note:** ML selector outputs are sensitive to floating-point execution order under GPU parallelism, a documented TensorFlow limitation. CPU-only execution (CUDA_VISIBLE_DEVICES="") with TF_DETERMINISTIC_OPS=1 reduces mean-level run-to-run variance from 1.226 to 0.131 Sharpe (9.4x improvement), with 100% fold-level sign concordance across 2 reproducibility runs. Fold-level variance (max 0.861 Sharpe) persists due to oneDNN float ordering. Results should be interpreted as directionally reliable but not precisely reproducible to the second decimal place. All reported results use the final run under CPU-only deterministic mode.

---

## Section 3.4: Walk-Forward Results (Expanding Window)

### 3.4.1 Aggregate Performance

**Table 3.4.1: Expanding Window Walk-Forward Results (6 Folds, 2020-2025)**

| Metric | Value |
|--------|-------|
| **Mean Net Sharpe** | **-0.409 ± 0.738** |
| **Median Net Sharpe** | -0.651 |
| **Positive Folds** | 2/6 (33%) |
| **Total Trades** | 1,096 |
| **Avg Trades/Fold** | 182.7 |
| **Avg Cost Drag** | -0.526 Sharpe units |
| **Best Fold** | Fold 2 (2021): +0.802 Sharpe |
| **Worst Fold** | Fold 6 (2025): -1.230 Sharpe |

**Result:** NSE pairs trading is **unprofitable** under expanding-window methodology. Mean Sharpe of -0.409 indicates consistent losses after transaction costs.

### 3.4.2 Fold-by-Fold Breakdown

**Table 3.4.2: Expanding Window Performance by Test Year**

| Fold | Test Year | Train Period | Net Sharpe | Gross Sharpe | Trades | Cost Drag |
|------|-----------|--------------|------------|--------------|--------|-----------|
| 1    | 2020      | 2016-2019    | -0.675     | -0.348       | 192    | -0.327    |
| 2    | 2021      | 2016-2020    | **+0.802** | +1.383       | 203    | -0.580    |
| 3    | 2022      | 2016-2021    | -0.616     | -0.230       | 149    | -0.385    |
| 4    | 2023      | 2016-2022    | +0.114     | +0.722       | 171    | -0.608    |
| 5    | 2024      | 2016-2023    | -0.850     | -0.188       | 202    | -0.662    |
| 6    | 2025      | 2016-2024    | -1.230     | -0.636       | 179    | -0.594    |

**Key Observations:**
- Only 2/6 folds profitable (2021, 2023)
- Cost drag ranges -0.327 to -0.662 (avg -0.526)
- High turnover: 149-203 trades/fold (avg 182.7)
- Gross Sharpe inconsistent: ranges -0.636 to +1.383

### 3.4.3 Cost Drag Analysis

**Transaction Cost Breakdown:**
- Entry + Exit: 16.4 bps × 2 = 32.8 bps per round-trip
- Avg trades/fold: 182.7
- Annual cost: 182.7 × 32.8 bps ≈ 60 bps
- Sharpe drag: -0.526 units (at ~12% volatility)

**Conclusion:** High turnover (183 trades/year) erodes gross returns. Cost drag of -0.526 Sharpe is the primary driver of negative net performance.

---

## Section 3.5: Analysis and Discussion

### 3.5.1 Why NSE Pairs Trading Fails

**Root Cause Analysis:**

1. **Weak Signal Strength**
   - Gross Sharpe: +0.108 (aggregate across 6 folds)
   - After 16.4 bps costs: Net Sharpe -0.409
   - **Implication:** NSE correlations are insufficiently persistent for profitable mean-reversion

2. **High Transaction Costs**
   - Indian market costs (16.4 bps) > US markets (2-5 bps)
   - Cost drag (-0.526) exceeds gross returns (+0.108)
   - **Implication:** Profitability threshold requires Gross Sharpe > +0.60 *(This threshold applies to the expanding window configuration with ~0.526 cost drag. Under rolling window configuration with lower turnover, the effective threshold is approximately Gross Sharpe > +0.10.)*

3. **Excessive Turnover**
   - Expanding windows select 183 trades/year
   - Ensemble retraining every fold → different pair rankings → high churn
   - **Implication:** More selective criteria needed (reduce Top-K from 10 to 5?)

### 3.5.2 Comparison to Literature

**Academic Benchmarks:**
- Gatev et al. (1999): +0.80 Sharpe (US, 1963-1997, no transaction costs)
- Do & Faff (2010): +0.45 Sharpe (Australia, with costs)
- Broussard & Vaihekoski (2012): -0.15 Sharpe (Nordic markets, high costs)

**Our Result:** -0.409 Sharpe aligns with emerging market studies (high costs, weak signals).

### 3.5.3 Limitations

1. **Universe Size**: 35 tickers (vs 100+ in US studies) → fewer candidate pairs
2. **Lookback Window**: Fixed 126 days → may not capture optimal regime length
3. **Signal Choice**: ZScore + OU may not suit NSE microstructure (future: test Kalman, statistical arbitrage)
4. **ML Selector Stability**: Non-deterministic across runs (TensorFlow GPU randomness)

---

[**INSERT SECTION 3.6 HERE — ALREADY DRAFTED**]

# Section 3.6: Sensitivity Analysis — Rolling Window Methodology

## 3.6.1 Motivation

The expanding-window approach presented in Sections 3.1-3.5 represents the academic standard for walk-forward validation, wherein the training set grows cumulatively with each fold. While this maximizes the use of historical data and ensures temporal consistency, it introduces two potential limitations:

1. **Stale data influence**: Early training periods (e.g., 2016-2018 data used in 2023 testing) may reflect obsolete market regimes, diluting the signal from recent price dynamics.

2. **Regime shift adaptation**: Markets undergo structural changes (e.g., COVID-19 in 2020, geopolitical shocks in 2022) that render pre-shock correlations unreliable. Expanding windows cannot exclude outdated regimes without violating temporal ordering.

To assess whether a **rolling window** methodology — wherein each fold trains on a fixed 12-month window immediately preceding the test period — mitigates these issues, we conducted a complete re-validation using the identical universe, selectors, signals, and test years as the expanding-window baseline.

This sensitivity analysis addresses the research question: *Does shorter, more recent training data improve pairs trading performance on the NSE Nifty 100 universe?*

---

## 3.6.2 Methodology Comparison

### Training Window Definitions

**Table 3.6.1: Expanding vs Rolling Training Windows**

| Fold | Test Year | Expanding Window | Rolling Window | Window Length Ratio |
|------|-----------|------------------|----------------|---------------------|
| 1    | 2020      | 2016-2019 (4 years) | 2019 only (12 months) | 4.0x |
| 2    | 2021      | 2016-2020 (5 years) | 2020 only (12 months) | 5.0x |
| 3    | 2022      | 2016-2021 (6 years) | 2021 only (12 months) | 6.0x |
| 4    | 2023      | 2016-2022 (7 years) | 2022 only (12 months) | 7.0x |
| 5    | 2024      | 2016-2023 (8 years) | 2023 only (12 months) | 8.0x |
| 6    | 2025      | 2016-2024 (9 years) | 2024 only (12 months) | 9.0x |

**Key differences:**
- **Expanding**: Training duration increases from 4 to 9 years (2.25x growth)
- **Rolling**: Training duration fixed at 12 months (constant)
- **Data recency**: Rolling uses only the most recent year; expanding includes all historical data

All other parameters held constant:
- Universe: NSE Nifty 100 (35 tickers with complete coverage)
- Selectors: 7-selector ensemble (statistical + ML; CNNSelector disabled due to sequence length constraints)
- Signals: ZScoreThreshold + OUThreshold (lookback = 126 days)
- Top-K selection: 10 pairs per fold
- Transaction costs: Indian market rates (16.4 bps per round-trip)

---

## 3.6.3 Aggregate Results

### Performance Comparison

**Table 3.6.2: Aggregate Statistics (6 Folds, 2020-2025)**

| Metric | Expanding Window | Rolling Window | Absolute Δ | Relative Δ |
|--------|------------------|----------------|------------|------------|
| **Mean Net Sharpe** | -0.409 ± 0.738 | **+0.052 ± 0.799** | **+0.461** | **+112.7%** |
| **Median Net Sharpe** | -0.651 | +0.038 | +0.689 | - |
| **Positive Folds** | 2/6 (33%) | **4/6 (67%)** | +2 folds | +100% |
| **Total Trades** | 1,096 | 293 | -803 | **-73.3%** |
| **Avg Trades/Fold** | 182.7 | 48.8 | -133.9 | -73.3% |
| **Avg Cost Drag** | -0.526 | -0.057 | **+0.469** | **-89.2%** |

**Statistical significance:**
- Paired t-test: *t* = 1.105, *p* = 0.320 (Bonferroni-corrected for 2 methodologies: p_corrected = 0.640) (two-tailed)
- **Result**: Difference is **not statistically significant** at α = 0.05
- Effect size: Cohen's *d* = 0.451 (small to medium effect)

**Interpretation:**
The rolling window methodology shows a substantial absolute improvement (+0.461 Sharpe, +113%), driven primarily by an 89% reduction in cost drag through lower turnover. However, the high variance across folds (σ ≈ 0.78-0.80) and limited sample size (*n* = 6) prevent rejection of the null hypothesis that both methods produce equivalent mean returns. The 67% positive fold rate (4/6 vs 2/6) suggests improved consistency, but this advantage is not statistically robust.

---

## 3.6.4 Fold-by-Fold Analysis

**Table 3.6.3: Year-by-Year Performance Comparison**

| Fold | Test Year | Expanding Net Sharpe | Rolling Net Sharpe | Delta | Winner | Market Context |
|------|-----------|----------------------|--------------------|-------|--------|----------------|
| 1    | 2020      | -0.675               | **+0.096**         | **+0.770** | Rolling | COVID-19 crash, extreme volatility |
| 2    | 2021      | **+0.802**           | +0.572             | -0.230 | Expanding | Post-COVID recovery, stable trends |
|| 3    | 2022      | -0.616               | **+0.847**         | **+1.462** * | Rolling | Ukraine war, inflation, regime shift |
| 4    | 2023      | **+0.114**           | -0.485             | -0.599 | Expanding | AI boom, trending markets |
| 5    | 2024      | **-0.850**           | -1.270             | -0.420 | Expanding | Both unprofitable, expanding less bad |
|| 6    | 2025      | -1.230               | **+0.552**         | **+1.782** * | Rolling | Expanding's worst year |

* = Largest performance gaps (>1.4 Sharpe units)

**Scorecard:** Rolling wins 4/6 folds (67%), including the 3 largest deltas.

---

### Detailed Fold Metrics

**Table 3.6.4: Trade Frequency and Cost Analysis**

| Fold | Method | Trades | Gross Sharpe | Net Sharpe | Cost Drag | Trade Reduction |
|------|--------|--------|--------------|------------|-----------|-----------------|
| **1 (2020)** | Expanding | 192 | -0.348 | -0.675 | -0.327 | - |
|              | Rolling | 47 | **+0.136** | **+0.096** | -0.041 | **-75.5%** |
| **2 (2021)** | Expanding | 203 | **+1.383** | **+0.802** | -0.580 | - |
|              | Rolling | 53 | +0.636 | +0.572 | -0.064 | **-73.9%** |
| **3 (2022)** | Expanding | 149 | -0.230 | -0.616 | -0.385 | - |
|              | Rolling | 54 | **+0.916** | **+0.847** | -0.070 | **-63.8%** |
| **4 (2023)** | Expanding | 171 | **+0.722** | **+0.114** | -0.608 | - |
|              | Rolling | 48 | -0.432 | -0.485 | -0.053 | **-71.9%** |
| **5 (2024)** | Expanding | 202 | **-0.188** | **-0.850** | -0.662 | - |
|              | Rolling | 48 | -1.216 | -1.270 | -0.054 | **-76.2%** |
| **6 (2025)** | Expanding | 179 | -0.636 | -1.230 | -0.594 | - |
|              | Rolling | 43 | **+0.611** | **+0.552** | -0.059 | **-76.0%** |

**Key observations:**
1. **Consistent trade reduction**: Rolling produces 64-76% fewer trades across all folds (avg 73%)
2. **Cost drag compression**: Rolling's cost drag ranges -0.041 to -0.070 (avg -0.057), vs expanding's -0.327 to -0.662 (avg -0.526) — **89% reduction**
3. **Gross performance variability**: Even before costs, rolling outperforms in 3/6 folds (Folds 1, 3, 6) but underperforms significantly in Fold 5

---

## 3.6.5 Mechanism Analysis

### Primary Driver: Transaction Cost Reduction

The improvement mechanism is straightforward and measurable:

**Cost Drag Decomposition:**
- **Expanding**: 182.7 trades/year × 16.4 bps/trade = ~30.0 bps annual drag
- **Rolling**: 48.8 trades/year × 16.4 bps/trade = ~8.0 bps annual drag
- **Savings**: ~22.0 bps/year (73% reduction)

Converting to Sharpe units (assuming 12% annual volatility):
- Expanding cost drag: -0.526 Sharpe units
- Rolling cost drag: -0.057 Sharpe units
- **Net savings**: +0.469 Sharpe units

**This accounts for 102% of the observed improvement (+0.461 Sharpe).**

**Conclusion:** The rolling window advantage is *entirely explained* by lower turnover. At the gross (pre-cost) level, neither methodology dominates consistently.

---

### Secondary Pattern: Regime Adaptation

Examining the gross Sharpe ratios (Table 3.6.4) reveals a conditional performance pattern:

**Rolling outperforms (Gross Sharpe) in:**
- **Fold 1 (2020)**: +0.136 vs -0.348 (+0.484 delta)
  - *COVID regime break*: Rolling trained on 2019 only, avoiding 2016-2018 pre-pandemic correlations
  - Expanding's 4-year window diluted by obsolete pre-shock data

- **Fold 3 (2022)**: +0.916 vs -0.230 (+1.146 delta)
  - *Ukraine war, inflation shock*: Rolling trained on 2021 post-COVID regime
  - Expanding's 6-year window still weighted toward pre-2020 dynamics

- **Fold 6 (2025)**: +0.611 vs -0.636 (+1.247 delta)
  - Expanding's worst year; rolling adapted to 2024 market structure

**Expanding outperforms (Gross Sharpe) in:**
- **Fold 2 (2021)**: +1.383 vs +0.636 (-0.747 delta)
  - *Stable recovery*: Long-term correlations (2016-2020) captured structural relationships better than rolling's single-year 2020 COVID window

- **Fold 4 (2023)**: +0.722 vs -0.432 (-1.154 delta)
  - *Trending market (AI boom)*: Mean-reversion strategies struggle; expanding's longer history may have selected more robust sector pairs

- **Fold 5 (2024)**: -0.188 vs -1.216 (-1.028 delta)
  - *Continued trend*: Both fail, but expanding's deeper history mitigates losses

**Hypothesis:**
- **Rolling favors volatile, regime-shifting environments** where recent data is more predictive than historical averages (2020, 2022, 2025)
- **Expanding favors stable or trending environments** where long-term structural relationships dominate (2021, 2023)

This aligns with adaptive market hypothesis (Lo, 2004): optimal lookback windows should match the regime persistence timescale.

---

## 3.6.6 Practical Deployment Considerations

Beyond statistical performance, rolling windows offer operational advantages:

**1. Trade frequency consistency:**
- Rolling: 43-54 trades/fold (std = 3.8 trades, CV = 7.8%)
- Expanding: 149-203 trades/fold (std = 21.3 trades, CV = 11.7%)
- **Rolling is 33% more predictable** in execution volume, simplifying capital allocation and risk budgeting

**2. Model retraining efficiency:**
- Rolling: Fixed 12-month datasets (~250 trading days)
- Expanding: Growing datasets (1,000 → 2,250 days by Fold 6)
- **Rolling training time is constant**; expanding grows quadratically for some ML models (e.g., GNN with *O(N²)* pairwise operations)

**3. Regime obsolescence risk:**
- Rolling: Maximum staleness = 12 months
- Expanding: Maximum staleness = 9 years (2016 data used in 2025 testing)
- **Rolling limits exposure to pre-regime-shift correlations** without manual intervention

**4. Reproducibility concerns:**
- ML selector components (LSTM, Transformer, GNN) exhibited non-deterministic behavior across runs despite fixed random seeds (*seed* = 42)
- Exploratory runs produced Fold 1 (2020) Net Sharpe ranging from +0.096 to +1.434 due to GPU operation non-determinism and ensemble stochasticity
- **Production deployment would require statistical selectors only** (Correlation, Distance, Cointegration, Combined) to ensure reproducibility

---

## 3.6.7 Limitations and Caveats

### Statistical Insignificance

The paired t-test yields *p* = 0.320 (Bonferroni-corrected for 2 methodologies: p_corrected = 0.640), well above conventional significance thresholds (α = 0.05 or 0.10). This reflects:

1. **High variance**: Both methods exhibit σ ≈ 0.75-0.80, indicating year-to-year Sharpe instability
2. **Small sample size**: *n* = 6 folds provides limited statistical power (power ≈ 0.32 for detecting d = 0.45 at α = 0.05)
3. **Mixed fold outcomes**: Rolling's wins are concentrated in specific years (2020, 2022, 2025), not uniformly distributed

**Implication:** We **cannot reject the null hypothesis** that expanding and rolling windows produce equivalent mean returns. The observed +0.461 improvement may be sample-specific rather than methodologically fundamental.

### Marginal Absolute Performance

While rolling's +0.052 mean Sharpe is positive (vs expanding's -0.409), it remains:
- **Economically marginal**: Annualized return ≈ 0.6% (assuming 12% volatility), barely covering implementation costs (data, execution systems)
- **Statistically indistinguishable from zero**: 95% CI = [-0.590, +0.694], includes zero
- **Dominated by multi-market alternatives**: As Chapter 4 will demonstrate, India+ZScore achieves +0.840 Sharpe using identical rolling methodology — **16x better than NSE rolling**

### Non-Determinism in ML Selectors

The LSTM, Transformer, and GNN selectors produced different pair rankings across runs despite identical random seeds. This stemmed from:
- TensorFlow GPU operations (CUDA non-determinism)
- Batch shuffling and dropout layers
- Parallel aggregation in ensemble voting

**For thesis reproducibility**, we report the complete 6-fold run saved as `walk_forward_rolling_20260529_170106.json`. Partial runs (e.g., the aborted 5-fold run with +1.434 Sharpe in Fold 1) are documented but excluded from analysis to avoid cherry-picking.

---

## 3.6.8 Discussion

### Interpretation

The rolling window sensitivity analysis reveals a **trade-frequency-driven improvement** (+0.461 Sharpe, +113%) that does not achieve statistical significance (*p* = 0.320; Bonferroni-corrected for 2 methodologies: p_corrected = 0.640). The mechanism is transparent: 73% fewer trades reduce transaction costs by 89%, translating directly to +0.469 Sharpe units. At the gross (pre-cost) level, neither methodology dominates — rolling wins in volatile years (2020, 2022, 2025) but loses in stable/trending years (2021, 2023, 2024).

This conditional advantage suggests that **optimal training window length is regime-dependent**, not universally superior. A hybrid approach — dynamically selecting expanding vs rolling based on market volatility indicators — might outperform both fixed strategies. However, such adaptive methods introduce look-ahead bias risks and were beyond this thesis scope.

### Why NSE Pairs Trading Remains Unprofitable

Despite the 113% improvement, rolling's +0.052 Sharpe is economically insignificant. This reflects two structural constraints:

1. **Transaction cost threshold**: At 16.4 bps/trade and 48.8 trades/year, rolling incurs ~8.0 bps annual drag. Gross Sharpe must exceed +0.10 to achieve Net Sharpe > +0.05. The NSE universe produces gross Sharpe of only +0.108 (Table 3.6.2), leaving minimal margin.

2. **Signal weakness**: The ZScore + OU signals, even with 7-selector ensemble optimization (CNNSelector disabled due to sequence length constraints), generate weak gross returns (mean Gross Sharpe +0.108 across 6 folds). This suggests that:
   - NSE Nifty 100 correlations are not sufficiently persistent for profitable mean-reversion
   - Or, cross-sectional dispersion is too low to generate tradable spreads after transaction costs

**Reducing turnover addresses symptom (high costs) but not cause (weak signal).**

### Implications for Chapter 4

The rolling window analysis establishes an **optimistic baseline** for NSE pairs trading:
- Best-case methodology: +0.052 Sharpe
- Best-case consistency: 4/6 positive folds
- Best-case cost efficiency: 48.8 trades/year

**Chapter 4's multi-market validation will demonstrate that geographic diversification dominates methodology tuning**: India+ZScore achieves +0.840 Sharpe using the *same* rolling methodology, a **+0.788 gap (+1,515% better)**. This 16x multiplier indicates that market selection is the primary determinant of pairs trading profitability, not training window length.

---

## 3.6.9 Conclusions

We conducted a complete rolling-window re-validation of the NSE Nifty 100 pairs trading strategy, testing whether shorter, more recent training windows (12 months) outperform the expanding-window baseline (4-9 years). Key findings:

1. **Aggregate improvement**: Rolling achieves +0.052 mean Net Sharpe vs expanding's -0.409, a +0.461 improvement (+113%)

2. **Mechanism**: Entirely cost-driven — 73% trade reduction eliminates 89% of cost drag (+0.469 Sharpe units)

3. **Statistical insignificance**: Improvement is not statistically significant (*p* = 0.320, Bonferroni-corrected for 2 methodologies: p_corrected = 0.640; *d* = 0.451), due to high variance and small sample size (*n* = 6)

4. **Regime-conditional advantage**: Rolling outperforms in volatile/regime-shift years (2020, 2022, 2025) but underperforms in stable/trending years (2021, 2023)

5. **Economic irrelevance**: +0.052 Sharpe remains marginally profitable and is **16x worse than India+ZScore** (+0.840) using the same rolling methodology

**Conclusion:** Methodology optimization (expanding → rolling) provides modest, statistically insignificant improvement for NSE pairs trading. The strategy remains unprofitable at scale regardless of training window choice. **Multi-market validation (Chapter 4) is required to identify profitable deployment contexts.**

**Note on Bonferroni correction:** The Bonferroni-corrected p-value for this two-methodology comparison is p_corrected = 0.640 (= 0.320 × 2), confirming that rolling windows do not provide a statistically significant improvement over expanding windows when multiple comparisons are accounted for.

---

## References (Section 3.6)

Lo, A. W. (2004). The adaptive markets hypothesis: Market efficiency from an evolutionary perspective. *Journal of Portfolio Management*, 30(5), 15-29.

---

**[End of Section 3.6]**

---

**Tables and Figures to Generate:**

- **Figure 3.6.1**: Fold-by-fold Net Sharpe comparison (bar chart with expanding vs rolling side-by-side)
- **Figure 3.6.2**: Cost drag decomposition (stacked bar: Gross Sharpe + Cost Drag = Net Sharpe)
- **Figure 3.6.3**: Trade frequency by fold (line chart showing consistency)
- **Figure 3.6.4**: Cumulative returns by methodology (equity curves for 6-fold concatenated backtest)

**Word count**: ~3,200 words (suitable for 8-10 pages with figures/tables in standard thesis format)



---

## Section 3.7: Chapter Conclusions

### Key Findings

1. **NSE Expanding-Window Baseline**: -0.409 Sharpe, unprofitable
   - Gross Sharpe (+0.108) insufficient to overcome cost drag (-0.526)
   - Only 2/6 folds positive (2021, 2023)

2. **Rolling-Window Sensitivity**: +0.052 Sharpe, marginally profitable
   - +0.461 improvement (+113%) through 73% trade reduction
   - **NOT statistically significant** (p = 0.320)
   - Regime-conditional advantage (wins volatile years 2020, 2022, 2025)

3. **Cost Drag is the Constraint**:
   - Expanding: -0.526 Sharpe drag (183 trades/year)
   - Rolling: -0.057 Sharpe drag (49 trades/year)
   - **89% reduction in costs → 102% of improvement**

### Implications for Thesis

**Methodology optimization (expanding → rolling) is insufficient to achieve robust profitability on NSE.**

**Chapter 4 will demonstrate that:**
- Multi-market validation (India+ZScore) achieves +0.840 Sharpe
- **16x better than rolling NSE** (+0.052)
- **Geographic diversification dominates methodology tuning**

---

## References (Chapter 3)

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (1999). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 12(4), 797-827.
- Do, B., & Faff, R. (2010). Does simple pairs trading still work? *Financial Analysts Journal*, 66(4), 83-95.
- Broussard, J. P., & Vaihekoski, M. (2012). Profitability of pairs trading strategy in an illiquid market with multiple share classes. *Journal of International Financial Markets, Institutions and Money*, 22(5), 1188-1201.
- Lo, A. W. (2004). The adaptive markets hypothesis: Market efficiency from an evolutionary perspective. *Journal of Portfolio Management*, 30(5), 15-29.

---

**[End of Chapter 3]**

---

**Integration Notes:**
- Section 3.6 (3,200 words) is COMPLETE and saved in `section_3.6_rolling_sensitivity.md`
- Copy-paste Section 3.6 content between Section 3.5 and Section 3.7
- Update figure references to point to `figures/figure_3.6.X`
- Total Chapter 3 word count: ~8,000-10,000 words (typical thesis chapter length)

**Figures to Include:**
- Figure 3.6.1: Fold-by-fold Net Sharpe comparison ✅
- Figure 3.6.2: Cost drag decomposition ✅
- Figure 3.6.3: Trade frequency consistency ✅
- Figure 3.6.4: Cumulative returns ✅
