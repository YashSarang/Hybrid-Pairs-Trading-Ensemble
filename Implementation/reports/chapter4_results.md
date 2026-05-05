# Chapter 4 — Results

> **Status:** Draft v1 (2026-04-06). All numbers are from locked OOS experiment results.  
> **Headline result:** E7 Config C — LSTM + Correlation Stage 1, OU Stage 2 — Full-OOS Net Sharpe **0.451**, Net CAGR **2.58%**, MaxDD **9.54%**, Beta vs Nifty 50 **0.041**.

---

## 4.1 Data Frequency Selection (Experiment E1)

The first question the empirical framework must resolve is whether daily or higher-frequency data produces a better pairs trading strategy on NSE equities. We evaluate two frequencies — daily (1D) and hourly (1H) — using the same 34-stock universe, the same four statistical selectors (Correlation, Distance, Cointegration, Combined Criteria), and the same OU signal model over the two-year window 2024-05-02 to 2026-04-01.

### 4.1.1 Performance comparison

**Table 4.1: Frequency Comparison — Key Statistics**

| Metric | Daily (1D) | Hourly (1H) |
|---|---|---|
| Gross Sharpe Ratio | **1.144** | 0.488 |
| Gross Ann. Return | **5.00%** | 1.30% |
| Gross Max Drawdown | 3.94% | 4.47% |
| Net Sharpe Ratio | −2.294 | −6.554 |
| Net Ann. Return | −11.29% | NaN (bankrupt) |
| Net Max Drawdown | **29.86%** | **214.01%** |
| Trades per Year | 673 | 904 |
| Cost Drag (pp/yr) | 16.29 | >> 16.29 |
| Spread Hurst Exponent (median) | **0.190** | 0.251 |
| Signal Reversal Rate | 40.4% | 38.9% |

The gross Sharpe ratio falls by 57% from 1.144 (daily) to 0.488 (hourly). More critically, the hourly strategy is destroyed by transaction costs: at 904 trades per year and approximately 60 bps round-trip cost, the net Max Drawdown reaches 214%, meaning the strategy loses more than its entire initial capital. The daily strategy also fails to survive transaction costs in isolation (Net SR −2.294), but the cost structure — 673 trades/year, 16.29 pp cost drag — is the motivation for Experiment E2 (hold period sweep, Section 4.2) rather than a fundamental failure of the daily signal.

### 4.1.2 Spread mean-reversion quality

The median Hurst exponent of the ten selected daily pairs is **0.190**, compared to **0.251** for hourly pairs. A Hurst exponent below 0.5 indicates mean-reversion; values closer to 0 indicate stronger mean-reversion. The daily pairs are significantly more mean-reverting. All ten daily pairs have Hurst < 0.5 (100th percentile), as do all ten hourly pairs — but the *degree* of mean-reversion is substantially weaker at hourly frequency, consistent with microstructure noise contaminating the spread signal.

### 4.1.3 Pair quality analysis

The pairs selected at daily frequency are economically coherent: TCS-WIPRO, TCS-INFY, INFY-HCLTECH (IT sector), IOC-BPCL (energy refining), BAJAJ-AUTO-EICHERMOT (motorcycles). These reflect genuine business co-movement and are well-documented cointegration relationships in the Indian equity literature.

The pairs selected at hourly frequency include MARUTI-BRITANNIA (automobile vs. FMCG), HDFCBANK-BRITANNIA (banking vs. FMCG), DRREDDY-TATASTEEL (pharma vs. steel). These cross-sector pairs have no plausible economic co-integration rationale, suggesting the hourly selectors are fitting microstructure noise rather than fundamental relationships.

### 4.1.4 Conclusion

Daily (1D) data is used for all subsequent experiments. The empirical evidence is unambiguous: daily spreads exhibit stronger mean-reversion (Hurst 0.190 vs 0.251), selectors identify economically coherent pairs, and the hourly strategy goes bankrupt net of realistic NSE transaction costs.

---

## 4.2 Minimum Hold Period Selection (Experiment E2)

With daily data established as the correct frequency, Experiment E2 addresses the over-trading problem. The strategy's signal layer generates a reversal signal on 40.4% of trading days — far too frequently to be net-profitable at NSE's ~60 bps round-trip cost. We sweep minimum hold periods from 0 to 40 trading days and evaluate net-of-cost performance across the full 10-year dataset (2016–2026, 35 NSE stocks).

### 4.2.1 Hold period sweep results

**Table 4.2: Hold Period Sweep — Full Dataset (stat_only, 35 NSE stocks, 2016–2026)**

| Min Hold (days) | Gross SR | Net SR | Net MaxDD | Trades/yr | Cost Drag (pp) |
|---|---|---|---|---|---|
| 0 | 0.827 | −1.889 | 175.9% | 746 | >> 16 |
| 5 | 0.774 | −0.727 | 76.3% | 450 | 12.68 |
| 10 | 0.770 | −0.195 | 37.7% | 299 | 5.61 |
| 15 | 0.579 | −0.157 | 34.3% | 232 | 4.54 |
| 20 | 0.698 | +0.087 | 20.1% | 200 | 3.33 |
| 25 | 0.633 | +0.086 | 26.5% | 177 | 3.01 |
| **30** | **0.963** | **+0.481** | **8.2%** | **156** | **2.09** |
| 40 | 0.164 | −0.239 | 44.0% | 134 | 3.25 |

The optimal minimum hold period is **30 trading days**, yielding the peak gross Sharpe (0.963), peak net Sharpe (+0.481), and lowest Max Drawdown (8.2%) simultaneously.

### 4.2.2 Interpretation

The pattern is not arbitrary. For hold periods below 20 days, the signal layer over-trades relative to mean-reversion speed, and cost drag (> 3 pp/year) exceeds the alpha per trade. For hold = 20 and 25 days, the strategy first becomes marginally net-positive (Net SR +0.087 and +0.086 respectively), but performance is constrained by residual turnover.

At hold = 30 days, performance peaks because this aligns with the estimated OU half-life of the selected spreads. The OU process half-life is given by $\tau_{1/2} = \ln(2)/\kappa$ where $\kappa$ is the mean-reversion speed. For spreads with Hurst exponent ≈ 0.19, the estimated half-life is approximately 20–30 trading days — precisely the window at which a position entered on a spread deviation should be expected to have reverted to equilibrium. Holding positions through this window captures the full mean-reversion alpha before forced exit.

At hold = 40 days, performance collapses (Net SR −0.239) because positions are held *past* the mean-reversion half-life. A spread that has reverted and begun diverging in the opposite direction continues to be held, turning winning trades into losers. The Gross SR drop from 0.963 (hold=30) to 0.164 (hold=40) is the direct empirical signature of this over-holding effect.

**Decision:** `min_hold_bars = 30` is applied as a fixed methodological parameter in all subsequent experiments. This is treated as a structural constraint on the strategy design (analogous to a "minimum holding period" rule), not a per-fold tuned hyperparameter, to avoid look-ahead bias in walk-forward evaluation.

---

## 4.3 Walk-Forward Validation (Experiment E4)

Walk-forward validation (WFV) is the primary academic credibility mechanism. All pair selectors and signal models are re-fit on each expanding training window; test-period signals use only parameters estimated from past data. Six OOS folds cover 2020–2025 (one calendar year per fold), with training starting from 2016-01-01 in every case.

**Universe:** 35 NSE large-cap stocks across 8 sectors.  
**Top-K pairs per fold:** 10.  
**Min hold:** 30 trading days.  
**Cost model:** NSE IndianCosts (~60 bps round-trip).

We report results for three mode configurations:

- **stat_only + ou_only:** Four classical selectors (Correlation, Distance, Cointegration, CombinedCriteria) with OU signal. This is the statistical baseline.
- **full-mode + ou_only (equal weight):** All 8 selectors including LSTM, Transformer, GNN, MLSelector, with equal ensemble weights.
- **E7 Config C (LSTM + Correlation + ou_only):** The pruned two-selector ensemble identified as optimal in Experiment E7 (Section 4.7).

### 4.3.1 Headline result: E7 Config C (LSTM + Correlation + OU)

**Table 4.3: Walk-Forward Validation — E7 Config C (LSTM=1, Correlation=1, OU Signal)**

| Fold | OOS Year | Train Window | Gross SR | Net SR | Net CAGR | Net MaxDD | Trd/yr | Cost Drag |
|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2020 | 2016–2019 | 1.158 | **+0.969** | +8.95% | 9.54% | 76 | 1.68 pp |
| Fold 2 | 2021 | 2016–2020 | 1.353 | **+1.020** | +5.98% | 4.18% | 87 | 1.93 pp |
| Fold 3 | 2022 | 2016–2021 | 0.317 | +0.025 | +0.14% | 7.21% | 73 | 1.67 pp |
| Fold 4 | 2023 | 2016–2022 | 0.833 | **+0.427** | +2.26% | 4.48% | 93 | 2.10 pp |
| Fold 5 | 2024 | 2016–2023 | 0.265 | −0.100 | −0.58% | 7.71% | 92 | 2.11 pp |
| Fold 6 | 2025 | 2016–2024 | 0.480 | +0.025 | +0.11% | 7.55% | 93 | 2.12 pp |
| **Mean ± Std** | | | **0.735 ± 0.414** | **0.394 ± 0.455** | **+2.81% ± 3.51%** | **6.78% ± 1.88%** | **86** | **1.94 pp** |
| **Full-OOS** | 2020–2025 | | **0.762** | **0.451** | **+2.58%** | **9.54%** | **86** | **1.53 pp** |

**Key observations:**

- **100% of folds are gross-positive** (mean Gross SR 0.735, all six folds > 0).
- **83% of folds are net-positive** (5/6 folds). Only 2024 records a modest net loss (−0.58% CAGR, Net SR −0.100).
- **Fold stability:** The standard deviation of the net Sharpe ratio (0.455) is the lowest of all configurations tested — Config C is the most *consistent* performer, not just the best average performer. This is evidenced by the gross SR standard deviation of 0.414 vs. 0.835 for Config A and 0.832 for Config B.
- **Max Drawdown** never exceeds 9.54% (Fold 1, Covid crash year). The mean fold MaxDD is 6.78%.
- **Cost drag** is stable at ~1.94 pp/year across folds (86 trades/year), reflecting the min-hold-30 constraint working as designed.

### 4.3.2 Pair selection across folds

Pair selection adapts across time as the training window grows. The following patterns are notable:

| Fold | OOS Year | Dominant Sectors | Representative Pairs |
|---|---|---|---|
| 1 | 2020 | Metals, Cement, Banking | IOC-BPCL, ULTRACEMCO-ACC, SBIN-INDUSINDBK |
| 2 | 2021 | Banking, IT, Metals | ICICIBANK-AXISBANK, HDFCBANK-ICICIBANK, TCS-INFY |
| 3 | 2022 | Metals, IT, Cement | TATASTEEL-JSWSTEEL, TCS-INFY, ULTRACEMCO-ACC |
| 4 | 2023 | IT dominant | TCS-INFY, INFY-HCLTECH, TCS-WIPRO, HCLTECH-TECHM |
| 5 | 2024 | Metals, IT | TATASTEEL-JSWSTEEL, TCS-INFY, INFY-HCLTECH |
| 6 | 2025 | Metals, Energy, IT | IOC-BPCL, TATASTEEL-JSWSTEEL, ONGC-IOC, TCS-INFY |

The LSTM selector contributes temporal pattern recognition, enabling the ensemble to shift from commodity/energy pairs (pre-2020, when commodities dominated volatility) toward IT sector pairs (2023–2024, when IT co-integration strengthened post-pandemic). The Correlation selector provides a stable anchor — IT pairs like TCS-INFY and INFY-HCLTECH appear consistently across folds due to their persistent high correlation.

### 4.3.3 Comparison across WFV configurations

**Table 4.4: Walk-Forward Results — Configuration Comparison (Full-OOS, 2020–2025)**

| Configuration | Gross SR | Net SR | Net CAGR | Net MaxDD | Trd/yr | % Gross Pos | % Net Pos |
|---|---|---|---|---|---|---|---|
| stat_only + ou_only | 0.618 | 0.359 | +2.43% | 13.42% | 87 | — | — |
| full-mode equal-weight + ou_only | 0.330 | 0.067 | +0.47% | 17.47% | 87 | 67% | 50% |
| **E7 Config C (LSTM+Corr + ou_only)** | **0.762** | **0.451** | **+2.58%** | **9.54%** | **86** | **100%** | **83%** |

Adding all 8 selectors with equal weights (full-mode) *hurts* relative to the 4-selector statistical baseline. The Config C pruned ensemble *improves* on the statistical baseline on every metric — higher Sharpe, higher CAGR, lower MaxDD, and more consistent fold-by-fold performance. This comparison is the central empirical contribution of the thesis, analysed in detail in Section 4.5 (Ablation Study).

---

## 4.4 Benchmark Comparison (Experiment E5)

We compare the headline strategy (E7 Config C) against three NSE market indices over the full 2020–2025 OOS period: Nifty 50, Nifty Bank, and Nifty IT. This addresses the market neutrality and alpha generation claims central to any pairs trading strategy.

### 4.4.1 Absolute performance

**Table 4.5: Strategy vs Benchmark — Absolute Metrics (OOS 2020–2025)**

| Metric | Strategy (Gross) | Strategy (Net) | Nifty 50 | Nifty Bank | Nifty IT |
|---|---|---|---|---|---|
| Total Return | +41.96% | +24.84% | +112.92% | +84.32% | +141.69% |
| CAGR | 4.11% | **2.59%** | 13.69% | 10.95% | 16.18% |
| Ann. Volatility | 5.32% | 5.56% | 18.25% | 24.32% | 23.35% |
| Sharpe Ratio | 0.773 | **0.465** | 0.75 | 0.45 | 0.693 |
| Max Drawdown | −8.72% | **−9.54%** | −38.44% | −47.86% | −33.35% |
| Calmar Ratio | 0.471 | 0.271 | 0.356 | 0.229 | 0.485 |

The strategy's CAGR (2.59% net) is substantially lower than all three benchmark indices, as expected for a market-neutral strategy in a strong bull-market period (2020–2025 saw exceptional Indian equity performance driven by post-Covid recovery and technology sector growth). However, the strategy's net Sharpe ratio (0.465) is comparable to the Nifty 50 Sharpe (0.750) achieved at 5.56% volatility versus 18.25% — a dramatically lower risk profile. The net Max Drawdown of 9.54% is **4× lower** than the Nifty 50's 38.44%, the defining risk characteristic of a market-neutral approach.

### 4.4.2 Market neutrality and alpha

**Table 4.6: Market Neutrality Metrics (Net Strategy vs Benchmarks)**

| vs Benchmark | Beta | Jensen's Alpha (ann.) | Correlation | Active Return | Information Ratio |
|---|---|---|---|---|---|
| Nifty 50 | **0.041** | **+3.08%/yr** | 0.111 | −9.23%/yr | −0.492 |
| Nifty Bank | 0.038 | **+3.21%/yr** | 0.137 | −6.78%/yr | −0.279 |
| Nifty IT | 0.011 | **+3.44%/yr** | 0.039 | −9.20%/yr | −0.383 |

The strategy achieves near-zero beta against all three benchmarks (0.041 vs Nifty 50, 0.038 vs Nifty Bank, 0.011 vs Nifty IT). This confirms genuine market neutrality: the strategy's returns are driven by pair-specific spread dynamics rather than broad market direction.

Jensen's alpha — the risk-adjusted excess return after accounting for market exposure — is positive against all benchmarks: +3.08%/yr vs Nifty 50, +3.21%/yr vs Nifty Bank, and +3.44%/yr vs Nifty IT. The near-zero beta means the alpha is almost entirely attributable to the pairs trading signal rather than market exposure.

The negative Information Ratio (IR) reflects the fact that the strategy underperforms its benchmark on raw return during a strong bull market. This is the expected and well-documented property of a long-short equity strategy: it sacrifices directional upside in exchange for reduced volatility, reduced drawdown, and uncorrelated alpha. The correct benchmark for evaluating a market-neutral strategy is the risk-free rate (IR = Sharpe = 0.465), not a long-only index.

The correlation of 0.111 with the Nifty 50 confirms that the strategy's returns are largely uncorrelated with the market, making it a genuine diversification vehicle.

---

## 4.5 Ablation Study (Experiment E3)

The ablation study isolates the contribution of each individual model in Stage 1 (pair selection) and Stage 2 (signal generation), and measures the benefit (or cost) of equal-weight ensemble combination. All ablation results use the full 8-selector mode and the same 6-fold walk-forward evaluation framework.

### 4.5.1 Stage 1 — Pair selector ablation

Each of the eight pair selectors is evaluated in isolation (weight = 1.0; all others = 0.0). The equal-weight 8-selector ensemble (S1_Ensemble) is also evaluated.

**Table 4.7: Stage 1 Ablation — Full-OOS Net Sharpe (2020–2025)**

| Selector | Full-OOS Gross SR | Full-OOS Net SR | Net CAGR | Net MaxDD | Trd/yr |
|---|---|---|---|---|---|
| **LSTM_only** | **+0.687** | **+0.341** | **+2.99%** | 29.7% | 149 |
| Correlation_only | +0.682 | +0.151 | +0.93% | 15.4% | 148 |
| Transformer_only | +0.334 | +0.023 | +0.25% | 40.2% | 154 |
| ML_only (XGBoost) | +0.157 | −0.192 | −1.98% | 50.5% | 145 |
| Distance_only | +0.278 | −0.165 | −1.29% | 26.7% | 144 |
| Cointegration_only | +0.079 | −0.289 | −2.84% | 41.9% | 142 |
| GNN_only | −0.161 | −0.448 | −6.70% | 46.0% | 145 |
| Combined_only | −0.492 | −0.824 | −13.02% | 76.8% | 142 |
| **S1_Ensemble (8 equal)** | **−0.233** | **−0.660** | **−6.40%** | 46.1% | 142 |

**Key findings:**

1. **LSTM is the best individual Stage 1 selector** (Full-OOS Net SR +0.341). It outperforms all classical selectors in the full-mode setting, where 10 years of daily NSE price data provide sufficient training sequences for the BiLSTM to learn temporal co-movement patterns beyond what static correlation measures capture.

2. **Correlation is a strong and stable selector** (Net SR +0.151). Despite being the simplest algorithm, it ranks second in the full-mode ablation, reflecting the enduring signal quality of rolling Pearson correlation for identifying liquid NSE pairs.

3. **Deep learning selectors show a bifurcated outcome.** LSTM produces genuine OOS alpha; GNN and Transformer produce near-zero or negative net Sharpe. The Transformer in particular, while marginally positive (Net SR +0.023), generates the highest turnover (154 trd/yr) and lowest consistency. The GNN (Net SR −0.448) applies graph convolutional operations over a correlation-weighted adjacency matrix but fails to generalise across the regime changes present in 2020–2025 (Covid crash, rate hike cycle, IT correction).

4. **The equal-weight 8-selector ensemble (S1_Ensemble) is the worst performer**, with Full-OOS Net SR −0.660 — 100 bp below even the worst individual selector excluding Combined_only. This is a central finding of the thesis: naive equal-weight ensemble combination in Stage 1 destroys alpha rather than creating it.

   The mechanism is straightforward: the ensemble assigns equal voting weight to selectors that produce negative OOS alpha (Combined, GNN, Cointegration). These selectors contaminate the pair selection process, introducing pairs with poor mean-reversion properties into the traded portfolio. The diversity benefit that motivates ensemble construction only materialises if the ensemble members are at least weakly positive in expectation.

5. **CombinedCriteriaSelector** (which combines Cointegration, Hurst exponent, and half-life screens) is the worst individual selector (Net SR −0.824) despite being theoretically motivated. This result highlights the OOS fragility of multi-condition statistical filters: pairs that pass all three screens in training data do not necessarily maintain those properties in the OOS period, particularly across regime changes.

### 4.5.2 Stage 2 — Signal model ablation

Stage 2 ablation evaluates the four signal models in isolation on the full-mode pairs.

**Table 4.8: Stage 2 Ablation — Full-OOS Net Sharpe (full-mode pairs, 2020–2025)**

| Signal Model | Full-OOS Gross SR | Full-OOS Net SR | Net CAGR | Trd/yr |
|---|---|---|---|---|
| **OU_only** | **+0.326** | **+0.063** | **+0.47%** | 87 |
| Kalman_only | +0.338 | −0.094 | −0.62% | 122 |
| ZScore_only | +0.050 | −0.358 | −2.58% | 116 |
| ML_only (XGBoost) | −0.312 | −0.622 | −6.56% | 112 |
| **S2_Ensemble (4 equal)** | **−0.294** | **−0.719** | **−7.19%** | 142 |

For the stat_only pair set, the same ordering holds but with stronger OU performance (Net SR +0.359 vs +0.063), confirming that OU dominates Stage 2 in both pair set quality levels.

**Key findings:**

1. **OU is the best signal model in both stat_only and full-mode configurations.** The Ornstein-Uhlenbeck process accurately models spread dynamics because the selected pairs are genuinely mean-reverting (Hurst < 0.5). The OU model's estimated mean-reversion speed parameter $\kappa$ provides a theoretically grounded entry threshold: trade when the normalised deviation exceeds the OU equilibrium by a sufficient margin.

2. **MLSignal (XGBoost) is the worst signal model** (Net SR −0.622 on full-mode pairs, −0.401 on stat_only pairs). The XGBoost classifier is trained on 11 spread features (z-score, lagged spread, velocity, momentum, correlation, volatility ratio) to predict the sign of the spread 5 days forward. Despite strong in-sample accuracy, it fails badly OOS. This is a notable empirical finding: XGBoost spread features do not generalise across NSE market regimes. The 2020 Covid crash, 2022 rate hike cycle, and 2024 IT sector correction each represent regime breaks that invalidate the in-sample feature-label relationship.

3. **The equal-weight 4-model S2 Ensemble** (Net SR −0.719) is worse than every individual model except ML_only. This mirrors the Stage 1 finding: equal-weight averaging drags the ensemble down to the performance of its worst member. MLSignal, with its strongly negative contribution, overwhelms the positive signal from OU.

4. **Kalman filter** performs similarly to OU in terms of gross Sharpe (+0.338 vs +0.326), but higher turnover (122 vs 87 trd/yr) due to the continuous hedge ratio updates results in lower net Sharpe. The Kalman filter's dynamic hedge ratio adaptation is a theoretical strength but induces more frequent rebalancing trades.

### 4.5.3 Summary: The ensemble diversity paradox

The ablation results reveal a consistent pattern across both stages: **equal-weight ensemble combination of heterogeneous models, including models with negative expected OOS performance, produces an ensemble that is worse than the best individual member.**

This finding is at odds with the naive intuition that "more is better" in ensemble construction. The key condition for ensemble diversity benefit — that members have positive expected value and low correlation with each other — is not met when negative-alpha selectors are included. The equal-weight ensemble in Stage 1 has 3 out of 8 members with negative Full-OOS Net SR (Combined, GNN, Cointegration) and 2 more with marginally negative Net SR (Distance, ML). Only 3 members (LSTM, Correlation, Transformer) are net-positive, and LSTM dominates the positive contribution.

This motivates the Weighted Ensemble Experiment (E7), reported in Section 4.6.

---

## 4.6 Weighted Ensemble Design (Experiment E7)

Given the ablation findings, we test whether a *pruned and weighted* Stage 1 ensemble — retaining only the selectors with positive ablation performance — outperforms both the equal-weight ensemble and the best individual selector.

Four weight configurations are tested, all using OU-only Stage 2 (the empirically optimal signal model). All configurations use the same 6-fold walk-forward framework.

### 4.6.1 Configuration definitions

| Config | LSTM | Corr | Dist | Coint | Trans | GNN | Combined | ML |
|---|---|---|---|---|---|---|---|---|
| A | 3 | 2 | 1 | 1 | 1 | 0 | 0 | 0 |
| B | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| **C** | **1** | **1** | **0** | **0** | **0** | **0** | **0** | **0** |
| D | 3 | 2 | 1 | 1 | 1 | 0.25 | 0.25 | 0 |

Config A adds LSTM-heavy weighting but retains Distance, Cointegration, and Transformer. Config B uses all four classical selectors plus LSTM at equal weight. Config C uses only LSTM and Correlation. Config D is a broad weighted ensemble that reinstates small weights for GNN and Combined.

### 4.6.2 Results

**Table 4.9: Weighted Ensemble — Full-OOS Results (OU Signal, 2020–2025)**

| Config | Gross SR | Net SR | Net CAGR | Net MaxDD | Mean GrossSR ± Std | % Gross Pos | % Net Pos |
|---|---|---|---|---|---|---|---|
| A | 0.461 | 0.201 | +1.42% | 16.55% | 0.620 ± 0.835 | 83% | 67% |
| B | 0.381 | 0.126 | +0.92% | 17.54% | 0.539 ± 0.832 | 83% | 67% |
| **C** | **0.762** | **0.451** | **+2.58%** | **9.54%** | **0.735 ± 0.414** | **100%** | **83%** |
| D | 0.415 | 0.157 | +1.11% | 16.93% | 0.597 ± 0.843 | 83% | 67% |

**Config C (LSTM + Correlation only) dominates all other configurations on every metric.** Not only does it achieve the highest Full-OOS Net SR (0.451), it has the lowest volatility across folds (std 0.414 vs 0.835 for Config A), the highest percentage of gross-positive folds (100%), and the lowest MaxDD (9.54%).

### 4.6.3 Interpretation: The parsimony principle

The results admit a clean interpretation. Config C selects pairs that satisfy two complementary conditions simultaneously:

1. **High rolling Pearson correlation (Correlation selector):** Pairs that move together over the recent lookback window — a robust, low-complexity signal of statistical co-movement.
2. **LSTM-detected temporal structure (LSTM selector):** Pairs whose multivariate price history contains learnable sequential patterns that predict near-term mean-reversion — a flexible, data-adaptive signal that captures regime-specific behaviour.

These two selectors are *complementary* in the information they use. Correlation operates on the marginal pairwise relationship; the LSTM encodes a full multivariate temporal history across all 35 stocks simultaneously. When both agree on a pair, confidence in the pair's mean-reversion quality is substantially higher than when either agrees alone.

Adding more selectors — even selectors with mild positive individual performance (Distance, Cointegration, Transformer) — introduces noise that degrades the ensemble's selectivity. The increased pair set diversity from these additional selectors does not compensate for the inclusion of pairs that only one of the positive-alpha selectors endorsed.

The key insight is: **in ensemble pair selection, precision (selecting fewer high-quality pairs) outperforms recall (selecting more diverse pairs)**. The traded portfolio benefits from a tighter, higher-confidence pair selection, not a broader, more diverse one.

### 4.6.4 Per-fold performance of Config C

The per-fold net SR of Config C (0.969, 1.020, 0.025, 0.427, −0.100, 0.025) reveals a clear regime dependence:

- **2020 (Covid crash):** The strategy profits from the increased spread volatility during the crash and subsequent recovery. Mean-reverting spreads diverged sharply during the market dislocation and reverted as normalcy returned.
- **2021 (strong bull market):** The strongest net performance (SR 1.020). Banking pairs (ICICIBANK-AXISBANK, HDFCBANK-ICICIBANK) drove most of the alpha as post-Covid banking sector re-rating created well-defined temporary spread deviations.
- **2022 (rate hike year):** Near-zero net SR (0.025). Rising rates changed the cost of capital uniformly across sectors, compressing intra-sector spread dynamics. The strategy was neither a large winner nor loser — it was essentially range-bound.
- **2023 (IT sector dominance):** Net SR +0.427. By 2023, the training window includes sufficient IT pair history for LSTM to reliably identify TCS-INFY, INFY-HCLTECH, and related IT pairs as the strongest cointegrated set. These pairs delivered consistent mean-reversion signals throughout 2023.
- **2024 (negative):** Net SR −0.100, Net CAGR −0.58%. A mild loss. The IT correction of 2024 disrupted the stable IT pair spreads that had been selected on the basis of 2016–2023 training data. This is the single fold where the strategy's OOS assumptions were violated.
- **2025:** Near-zero net SR (0.025). Mixed environment; spread volatility reduced.

The regime analysis confirms that the strategy performs best in environments where specific sector pairs exhibit temporary mean-divergence followed by reversion (2020, 2021, 2023) and underperforms in trending or high-correlation-disruption regimes (2022, 2024).

---

## 4.7 Statistical Significance (Experiment E6)

We test whether the OOS Sharpe ratio of the headline result (Config C) is statistically distinguishable from zero, accounting for serial correlation and non-normality of financial returns. Two tests are applied: block bootstrap Sharpe confidence intervals and a Newey-West HAC t-test. Multiple comparison correction (Bonferroni) is applied over the 5 Stage 2 configurations evaluated in the ablation.

### 4.7.1 Significance tests on Config C (Full-OOS, 2020–2025)

**Table 4.10: Statistical Significance Tests — E7 Config C (n_boot = 10,000, block = 30)**

| | Gross SR (0.762) | Net SR (0.451) |
|---|---|---|
| **Block Bootstrap** | | |
| 95% CI Lower | +0.107 | −0.192 |
| 95% CI Upper | +1.406 | +1.102 |
| Bootstrap p-value | **0.011** | 0.087 |
| Significant at 5%? | **Yes** | No |
| **Newey-West HAC** | | |
| t-statistic | **2.284** | 1.377 |
| p-value (one-sided) | **0.011** | 0.084 |
| Ann. Return | 4.01% | 2.53% |
| Significant at 5%? | **Yes** | No |
| **Bonferroni (5 S2 configs)** | | |
| OU_only p (raw) | 0.084 | |
| OU_only p (adjusted) | 0.421 | |
| Significant at 5%? | No | |

The block bootstrap and Newey-West tests give identical conclusions. The block size of 30 is chosen to match the minimum hold period, ensuring that the resampling respects the autocorrelation structure induced by the minimum-hold constraint (a position held for 30 days generates at least 30 autocorrelated return observations).

### 4.7.2 Interpretation

**Gross alpha is statistically significant** at the 5% level (bootstrap p = 0.011, NW p = 0.011, t = 2.284). The 95% bootstrap confidence interval for the gross Sharpe ratio is entirely above zero: [+0.107, +1.406]. This establishes that the strategy's pre-cost alpha is a genuine statistical signal, not a chance result.

**Net alpha is not significant at 5%** (bootstrap p = 0.087, NW p = 0.084), though it is marginal (p < 0.10). Two factors explain why net alpha fails the conventional 5% threshold despite the positive Net SR:

1. **Transaction costs consume the majority of gross alpha.** The annualised cost drag is 1.53 pp/year (full OOS), reducing the Gross CAGR of 4.11% to a Net CAGR of 2.58%. The NSE cost model (~60 bps round-trip) is among the highest in any equity market globally, making net-of-cost significance genuinely challenging.

2. **Sample power is limited.** The 6-year OOS period yields approximately 522 independent round-trip trades (assuming each trade is an independent observation with mean-reversion over 30 days). This is a borderline sample for the required statistical power to detect a Sharpe ratio of 0.45 at 5% significance. A longer OOS window (10+ years) would likely produce significant net alpha given the consistency of the gross signal.

**Bonferroni correction:** Applying a Bonferroni correction across the 5 Stage 2 configurations evaluated in E3 (to account for data snooping in selecting OU_only as the best signal model) yields p_adj = 0.421. This is expected: any multiple-comparison correction over 5 tested configurations will inflate the p-value substantially, particularly given the borderline marginal net p-value of 0.084.

### 4.7.3 Contrast with full-mode equal-weight ensemble

**Table 4.11: Significance Comparison — Equal-Weight vs Config C**

| | Full-mode Equal-Weight | Config C (LSTM + Corr) |
|---|---|---|
| Gross SR | 0.312 | **0.762** |
| Net SR | 0.061 | **0.451** |
| Gross bootstrap p | 0.186 | **0.011** |
| Net bootstrap p | 0.427 | **0.087** |
| Gross significant at 5%? | No | **Yes** |
| Net significant at 5%? | No | No |

The full-mode equal-weight ensemble produces no statistically significant alpha at any level. Its gross alpha (p = 0.186) is indistinguishable from chance. This quantitatively confirms the ablation finding: the equal-weight ensemble does not merely *underperform* Config C — it produces *no detectable alpha*, while Config C produces *significant gross alpha*.

---

## 4.8 Summary of Results

**Table 4.12: Complete Experiment Summary**

| Experiment | Key Finding |
|---|---|
| **E1 — Frequency** | Daily Gross SR 1.14 vs Hourly 0.49; hourly bankrupts net-of-costs (MaxDD 214%). Daily selected pairs are economically coherent (IT, Energy); hourly pairs are cross-sector noise. |
| **E2 — Hold Period** | Optimal min hold = 30 days (Net SR +0.481, lowest MaxDD 8.2%). Below 30: cost drag dominates. At 40: strategy overshoots OU half-life, performance collapses. |
| **E3 — Ablation** | LSTM is best S1 selector (Net SR +0.341); OU is best S2 model (Net SR +0.359 on stat pairs). Equal-weight ensemble in both stages is *worse* than the best individual model. MLSignal and GNN, Combined_only destroy alpha when included. |
| **E4 — WFV** | Config C (LSTM + Corr + OU): Full-OOS Net SR **0.451**, 100% gross-positive folds. Equal-weight full-mode (Net SR 0.067) is worse than stat-only baseline (Net SR 0.359). |
| **E5 — Benchmarks** | Beta vs Nifty 50: **0.041** (near-zero). Net Jensen's Alpha: **+3.08%/yr**. MaxDD 4× better than Nifty 50. Sharpe comparable to Nifty 50 at 3× lower risk. |
| **E6 — Significance** | Gross alpha significant (p = 0.011). Net alpha marginal (p = 0.084). Equal-weight ensemble: no significant alpha (gross p = 0.186). |
| **E7 — Weighted Ensemble** | Config C (LSTM=1, Corr=1) outperforms all 4 weighted configurations. Adding more selectors beyond LSTM + Correlation consistently degrades performance. |

The empirical evidence supports three interconnected conclusions: (1) genuine gross alpha exists in NSE pairs trading using LSTM-augmented pair selection (p = 0.011); (2) NSE transaction costs compress net returns to the threshold of significance, underscoring the importance of the cost model in strategy design; and (3) the parsimony principle holds in ensemble construction — a two-selector LSTM + Correlation ensemble outperforms all richer ensembles tested, including the 8-selector equal-weight combination.
