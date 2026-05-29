# Chapter 3: NSE Nifty 100 Baseline Validation

## Chapter Overview

This chapter presents the baseline walk-forward validation of the hybrid pairs trading ensemble on the NSE Nifty 100 universe. We establish methodology, report results under expanding-window training, and conduct sensitivity analysis comparing rolling vs expanding window approaches.

**Structure:**
- **Section 3.1**: Experimental Design
- **Section 3.2**: Universe Selection and Data
- **Section 3.3**: Selector Ensemble Configuration
- **Section 3.4**: Walk-Forward Results (Expanding Window)
- **Section 3.5**: Analysis and Discussion
- **Section 3.6**: Sensitivity Analysis — Rolling Window Methodology ✅ **[COMPLETE]**

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
- **OUThreshold**: Ornstein-Uhlenbeck mean-reversion with MLE parameters (lookback = 126 days)

**Entry/Exit:**
- Entry: Signal triggers (spread exceeds threshold)
- Exit: Mean reversion (spread crosses zero) or 20-day holding period
- Position sizing: Equal-weighted across 10 pairs

**Transaction Costs:**
- Indian market: 16.4 bps per round-trip (STT + brokerage + slippage)
- Applied on entry and exit

---

## Section 3.2: Universe Selection and Data

### 3.2.1 NSE Nifty 100 Universe

**Rationale:**
- Large-cap liquid stocks (minimize execution risk)
- Broad sector coverage (15 GICS sectors represented)
- Sufficient cross-sectional dispersion for pair formation

**Data Filters:**
- Complete price history: 2016-2025 (no survivorship bias)
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
- ADF test on price spreads (cointegration filter)
- Half-life calculation (exclude pairs with HL > 60 days)

---

## Section 3.3: Selector Ensemble Configuration

### 3.3.1 Individual Selectors

**Statistical Selectors (4):**
1. **CorrelationSelector**: Highest rolling Pearson correlation (126-day window)
2. **DistanceSelector**: Minimum normalized squared distance
3. **CointegrationSelector**: Lowest ADF p-value on spread residuals
4. **CombinedCriteriaSelector**: Weighted composite (0.3×Corr + 0.3×Dist + 0.4×Coint)

**ML Selectors (4):**
5. **LSTMSelector**: Bidirectional LSTM on price/volume sequences (60-day lookback)
6. **TransformerSelector**: Multi-head attention (4 heads, 2 layers)
7. **GNNSelector**: Graph Convolutional Network (correlation-weighted edges)
8. **CNNSelector**: 1D CNN on spread time series (disabled due to data requirements)

**Configuration:**
- All selectors train on identical train/test splits
- ML models: Early stopping (10 epochs), dropout (0.2), Adam optimizer
- Hyperparameters: Grid search on Fold 1, then fixed for Folds 2-6

### 3.3.2 Ensemble Aggregation

**Method:** Weighted ensemble score
- Each selector ranks pairs 1-595
- Ensemble score = Σ(selector_score × selector_weight)
- Weights: Equal (1/8 for 8 selectors)

**Top-K Selection:**
- Select 10 pairs with highest ensemble scores
- Break ties by individual selector agreement count

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
   - **Implication:** Profitability threshold requires Gross Sharpe > +0.60

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
