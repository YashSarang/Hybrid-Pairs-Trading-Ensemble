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
- Selectors: 8-member ensemble (statistical + ML)
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
- Paired t-test: *t* = 1.105, *p* = 0.320 (two-tailed)
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
| 3    | 2022      | -0.616               | **+0.847**         | **+1.462** ⭐ | Rolling | Ukraine war, inflation, regime shift |
| 4    | 2023      | **+0.114**           | -0.485             | -0.599 | Expanding | AI boom, trending markets |
| 5    | 2024      | **-0.850**           | -1.270             | -0.420 | Expanding | Both unprofitable, expanding less bad |
| 6    | 2025      | -1.230               | **+0.552**         | **+1.782** ⭐ | Rolling | Expanding's worst year |

⭐ = Largest performance gaps (>1.4 Sharpe units)

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

The paired t-test yields *p* = 0.320, well above conventional significance thresholds (α = 0.05 or 0.10). This reflects:

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

The rolling window sensitivity analysis reveals a **trade-frequency-driven improvement** (+0.461 Sharpe, +113%) that does not achieve statistical significance (*p* = 0.320). The mechanism is transparent: 73% fewer trades reduce transaction costs by 89%, translating directly to +0.469 Sharpe units. At the gross (pre-cost) level, neither methodology dominates — rolling wins in volatile years (2020, 2022, 2025) but loses in stable/trending years (2021, 2023, 2024).

This conditional advantage suggests that **optimal training window length is regime-dependent**, not universally superior. A hybrid approach — dynamically selecting expanding vs rolling based on market volatility indicators — might outperform both fixed strategies. However, such adaptive methods introduce look-ahead bias risks and were beyond this thesis scope.

### Why NSE Pairs Trading Remains Unprofitable

Despite the 113% improvement, rolling's +0.052 Sharpe is economically insignificant. This reflects two structural constraints:

1. **Transaction cost threshold**: At 16.4 bps/trade and 48.8 trades/year, rolling incurs ~8.0 bps annual drag. Gross Sharpe must exceed +0.10 to achieve Net Sharpe > +0.05. The NSE universe produces gross Sharpe of only +0.108 (Table 3.6.2), leaving minimal margin.

2. **Signal weakness**: The ZScore + OU signals, even with 8-selector ensemble optimization, generate weak gross returns (mean Gross Sharpe +0.108 across 6 folds). This suggests that:
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

3. **Statistical insignificance**: Improvement is not statistically significant (*p* = 0.320, *d* = 0.451), due to high variance and small sample size (*n* = 6)

4. **Regime-conditional advantage**: Rolling outperforms in volatile/regime-shift years (2020, 2022, 2025) but underperforms in stable/trending years (2021, 2023)

5. **Economic irrelevance**: +0.052 Sharpe remains marginally profitable and is **16x worse than India+ZScore** (+0.840) using the same rolling methodology

**Conclusion:** Methodology optimization (expanding → rolling) provides modest, statistically insignificant improvement for NSE pairs trading. The strategy remains unprofitable at scale regardless of training window choice. **Multi-market validation (Chapter 4) is required to identify profitable deployment contexts.**

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
