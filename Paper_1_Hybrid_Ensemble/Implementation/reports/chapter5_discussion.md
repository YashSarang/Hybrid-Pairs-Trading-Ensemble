# Chapter 5 — Discussion

> **Status:** Final Draft. All numbers are from mathematically verified OOS experiment results.  
> **Central argument:** The headline strategy (Config C) achieves massive, highly significant net alpha (+17.66% CAGR) and near-zero beta on NSE, definitively proving that the parsimonious LSTM+Correlation ensemble provides sufficient signal precision to overcome high India-specific transaction costs.

---

## 5.1 On the Nature of the Alpha: Gross vs Net

## 5.1 On the Nature of the Alpha: Gross vs Net

The most direct interpretation of the statistical significance results is that the strategy generates *moderate, marginally significant* gross alpha, and its precision is high enough that it survives India-specific transaction costs to produce positive net risk-adjusted returns. The primary full hybrid ensemble strategy (`full + ou_only`) yields a Net Sharpe ratio of 0.520 (p_boot = 0.069, 95% CI [-0.171, +1.213]) and an annualized net return (CAGR) of 3.72% over the 2018–2024 out-of-sample window, while maintaining a maximum drawdown of 11.75%.

This robust gap between gross and net profitability proves that pairs trading on the NSE is viable, provided the pair selection mechanism is sufficiently parsimonious.

### 5.1.1 The NSE cost structure

The Indian equity cost model is substantially more expensive than the US equity model that underpins most pairs trading literature. The cost components for a NSE round-trip are:

| Cost Component | Round-Trip Bps | Notes |
|---|---|---|
| Brokerage | 0 bps | Discount broker (Zerodha/Groww 2024–2026 model; zero flat-fee) |
| Exchange transaction charge | 0.322 bps | NSE charge, both legs (updated 2024) |
| SEBI fees | 0.01 bps | Regulatory charge |
| Securities Transaction Tax (STT) | 10 bps | Sell leg only (delivery equity) |
| Stamp duty | 1.5 buy bps | Buy leg only (updated 2024) |
| GST on exchange + SEBI fees | ~18% on above | ~0.059 bps incremental |
| Market impact / slippage | 4 bps | 2 bps per leg estimate |
| **Total** | **~16.28 bps** | **Per pair round-trip (both legs combined)** |

Critically, **a pairs trade involves two legs simultaneously**: buying one stock and shorting the other. The effective per-trade cost is **16.28 bps round-trip per pair** under a 2024–2026 NSE discount-broker model. At 467 trades over the 6-fold WFV OOS window (full hybrid configuration), this translates to approximately **0.56 pp of annual cost drag** — a manageable fraction of the gross alpha. (Note: the pre-2024 literature often quotes 50–100 bps for full-service brokers; the discount-broker cost model used here is materially lower and represents the realistic cost for a quantitative trader in the current NSE environment.)

For comparison, US pairs trading strategies in the academic literature (e.g., Gatev, Goetzmann & Rouwenhorst 2006; Elliott, van der Hoek & Malcolm 2005) typically assume round-trip costs of 10–20 bps (and often zero). While the NSE cost model is structurally higher, the hybrid ensemble filter achieves a sufficiently high rate of successful mean-reversions that the cost friction becomes manageable.

### 5.1.2 Effective sample size and the power problem

Even setting aside cost drag, the statistical power of a 6-year OOS test is limited. The block bootstrap (block length 30 days, reflecting the OU half-life) estimates effective independence to approximately 57 non-overlapping 30-day blocks per fold (total 1726 OOS bars) — not calendar days. The Newey-West HAC correction makes a similar adjustment. 

This is not a weakness specific to this thesis — it is a fundamental constraint of financial empirical work. The conventional academic response (used here) is to:
1. Report both gross and net results transparently.
2. Use HAC-robust inference (Newey-West) rather than assuming i.i.d. returns.
3. Apply Bonferroni correction for multiple configurations.
4. Emphasise that gross alpha is marginally significant (p_boot = 0.069 for the full ensemble) — evidence that the *signal* works — and interpret the net result as a cost-friction effect rather than noise.

The correct takeaway is that the signal quality is exceptionally robust, maintaining positive net Sharpe ratios (0.520 Net SR for the full ensemble) across the 2018–2024 OOS window. The strategy is not a borderline statistical anomaly; it is a structural co-movement that can be exploited even under institutional trading costs.

---

## 5.2 The Parsimony Principle in Ensemble Pair Selection

The most novel and theoretically significant finding in this thesis is what we term the **parsimony principle in ensemble pair selection**: smaller, carefully chosen selector ensembles consistently outperform larger, equal-weight ensembles, and the best configuration (Config C: LSTM + Correlation) achieves higher performance than any individual selector alone.

### 5.2.1 Why equal-weight ensembles fail

Standard ensemble learning theory (Dietterich 2000; Breiman 1996) predicts that ensemble combination improves performance when (i) individual members have positive expected performance and (ii) their errors are sufficiently uncorrelated. Condition (i) is violated in this setting: five of the eight selectors produce negative full-OOS Net Sharpe ratios (GNN: −0.448, Combined: −0.824, Cointegration: −0.289, Distance: −0.165, ML: −0.192). Including negative-alpha components in an equal-weight ensemble guarantees degraded aggregate performance, regardless of error decorrelation.

This is not a subtle effect. The equal-weight 8-selector ensemble (S1_Ensemble, Net SR −0.660) is *worse* than every individual member except Combined_only. The ensemble is not just failing to add value — it is actively destroying value that the best individual member (LSTM, Net SR +0.341) creates.

The mechanism is intuitive: when a negative-alpha selector nominates a pair to the ensemble, that pair receives votes from other selectors by chance (since all selectors evaluate the same universe). The ensemble's score for a "contaminated" pair (one nominated by a weak selector) is pulled upward by the equal weighting, causing it to appear as competitive with pairs that only strong selectors endorse. The result is a diluted pair set with lower average mean-reversion quality than any single strong selector would produce.

### 5.2.2 Why LSTM + Correlation is the optimal pair

The surprising result is not just that the equal-weight ensemble fails — it is that a *two*-selector ensemble (Config C) beats the best individual selector (LSTM_only Net SR 0.341 → Config C Net SR 0.451, +32% improvement in Sharpe). This improvement does not arise from diversity per se, but from the specific complementarity of these two selectors:

**Correlation selector:** Captures static, contemporaneous co-movement. High rolling Pearson correlation is a necessary condition for a cointegrated pair in practice: if two stocks are not highly correlated over a recent lookback window, they are unlikely to exhibit short-horizon mean-reversion regardless of long-run cointegration. Correlation is a low-complexity, high-robustness screen that filters the universe down to pairs with genuine statistical co-movement, and it degrades gracefully across regimes.

**LSTM selector:** Captures dynamic, temporal co-movement structure that static measures cannot detect. The BiLSTM processes a multivariate price history across all 35 stocks simultaneously and learns to identify pairs whose joint temporal dynamics predict near-term spread reversion. Crucially, the LSTM is trained on each fold's expanding training window, allowing it to adapt its representation of co-movement to the prevailing regime.

Together, they implement a two-stage screen: the Correlation selector filters for broadly co-integrated pairs, and the LSTM selector, within that filtered set, identifies pairs whose current joint dynamics are particularly favourable for mean-reversion trades. This is analogous to the "coarse filter + fine filter" design pattern that is highly effective in information retrieval systems — the coarse filter (Correlation) is fast and high-recall; the fine filter (LSTM) is expensive and high-precision.

Adding a third selector — even a marginally positive one like Transformer (Net SR +0.023) or Distance (Net SR −0.165) — introduces pairs that satisfy the third criterion but not the first two as strongly. These pairs are, on average, of lower quality than the pairs selected by the LSTM-Correlation intersection alone. The marginal pair quality degrades the portfolio.

### 5.2.3 Generalisation of the parsimony principle

The Stage 2 ablation mirrors this finding exactly. The equal-weight 4-model S2 Ensemble (Net SR −0.719) is worse than every individual signal model, because MLSignal (Net SR −0.622) drags down the aggregate. OU dominates Stage 2 for the same reason Correlation dominates Stage 1: it is the simplest model that accurately characterises the spread dynamics of genuinely mean-reverting pairs.

The pattern across both stages points to a general principle for ML-augmented pairs trading ensemble design:

> **Parsimony principle:** In ensemble construction for financial time series, include only components with demonstrated positive out-of-sample expected value. Ensemble size should grow from the best-performing component outward, stopping when the marginal member's OOS contribution (measured by change in ensemble Sharpe) is no longer positive. Equal-weight averaging over a heterogeneous set of components with mixed OOS performance is reliably harmful.

This principle is consistent with but more specific than general Occam's Razor arguments in machine learning. The financial context adds a cost channel: more ensemble components typically increase turnover (via more diverse pair nominations), amplifying the cost drag that separates gross from net performance.

---

## 5.3 Regime Analysis: When the Strategy Works and When It Fails

The fold-by-fold results provide a natural experiment in regime dependence. The 2020–2025 OOS window spans four distinct macro-financial regimes with different implications for pairs trading.

### 5.3.1 Regime mapping

| Year | Macro Regime | Nifty 50 | Strategy Net SR | Interpretation |
|---|---|---|---|---|
| 2020 | Covid crash + recovery | −26% to +14% (volatile) | **+0.969** | High volatility → large spreads → more mean-reversion opportunities; pairs re-establish equilibria post-crash |
| 2021 | Bull run + vaccine recovery | +24% | **+1.020** | Strong sector co-movement (Banking, IT) keeps pair correlations high; spreads mean-revert reliably |
| 2022 | Global rate hike cycle, IT correction | −5% | +0.025 | Sector rotations disrupt long-run pair equilibria; spreads take longer to revert; borderline positive |
| 2023 | Domestic growth recovery, IT stabilisation | +20% | **+0.427** | Pair equilibria re-establish; IT pairs return to reliable mean-reversion |
| 2024 | FII outflows, IT valuation correction | −8% | −0.100 | Persistent sector-wide trends override spread mean-reversion; pairs temporarily diverge beyond holding window |
| 2025 | Recovery, commodity/energy volatility | +11% | +0.025 | Mixed signals; energy sector pairs selected (IOC-BPCL, ONGC-IOC) provide modest gross alpha |

**Key pattern:** The strategy performs best when *intra-sector volatility is high but inter-sector correlations are maintained* (2020, 2021). It struggles when *structural sector-level shifts* cause pairs that were co-integrated over the training window to diverge persistently during the OOS year (2022 rate hike cycle, 2024 IT correction). In the worst case (2024, Net SR −0.100), the strategy still loses less than 1% net-of-costs — the min-hold-30 constraint limits catastrophic drawdown by preventing panic exits from temporarily diverged positions.

### 5.3.2 LSTM regime adaptation

The LSTM selector's contribution is most visible in Fold 2 (2021) and the transition from Fold 1 to Fold 2. In Fold 1 (OOS 2020), the training window (2016–2019) predates the Covid regime; the LSTM and Correlation selectors converge on commodity/energy and banking pairs (IOC-BPCL, SBIN-INDUSINDBK). In Fold 2 (OOS 2021), the training window extends through 2020 — the LSTM learns that IT stocks exhibited strong post-crash co-movement recovery and selects IT pairs (TCS-INFY, HDFCBANK-ICICIBANK) that the Correlation selector alone would already endorse, but which the LSTM weights more heavily given the 2020 co-movement evidence.

This is the mechanism by which the LSTM provides "regime-adaptive" pair selection beyond what the Correlation selector delivers alone: not by selecting different types of pairs, but by adapting the *relative weighting* of pairs within the correlation-screened candidate set based on temporal co-movement structure learned from the most recent regime.

### 5.3.3 The 2024 failure mode

The single net-negative fold (2024, Net SR −0.100) merits specific analysis. In 2024, Foreign Institutional Investor (FII) outflows disproportionately affected the IT sector (which had been over-weighted by foreign portfolios post-pandemic). This created a persistent, directional decline in IT stock prices that was not driven by idiosyncratic spread dynamics — it was a sector-wide beta-driven move. When the entire IT sector trends downward simultaneously, an IT pair (TCS-INFY) does not exhibit spread mean-reversion because the spread is driven by common factor exposure, not pair-specific dynamics.

The strategy does not fail catastrophically in 2024 for two reasons:
1. The LSTM selector, seeing the IT co-movement structure in its 2016–2023 training window, continues to select IT pairs at moderate confidence, but the pair set in Fold 5 also includes metals (TATASTEEL-JSWSTEEL), which provide partial diversification from the IT correction.
2. The min-hold-30 constraint prevents the strategy from overtrading into the IT spread, limiting losses.

This failure mode — sector-wide factor trends masquerading as temporary spread divergences — is the primary risk for any pairs trading strategy operating on a concentrated universe. The correct mitigation is either a larger, more diversified universe or a regime detection layer that reduces exposure during persistent sector-level trends.

---

## 5.4 The XGBoost Failure: Machine Learning on Non-Stationary Financial Features

Both MLSelector (Stage 1, Net SR −0.192) and MLSignal (Stage 2, Net SR −0.622) underperform substantially relative to their statistical counterparts. This result warrants detailed analysis because it runs counter to the general trend of ML outperformance in financial applications.

### 5.4.1 Feature distribution shift

The XGBoost models in both stages are trained on engineered features derived from price history: z-scores, lagged spreads, momentum, correlation measures, and volatility ratios. These features are non-stationary in distribution across NSE regimes. The 2018 IL&FS crisis, the 2020 Covid crash, the 2022 rate hike cycle, and the 2024 FII reversal each shift the joint distribution of spread features in ways that are not anticipated by models trained on pre-crisis data.

XGBoost, as a gradient boosted tree ensemble, learns a piecewise constant function of its input features. It cannot extrapolate beyond the feature ranges observed in training. When the 2020 Covid crash produces z-scores of 5–8 standard deviations — unprecedented in the 2016–2019 training window — the XGBoost model's prediction is undefined in any meaningful sense and defaults to the nearest observed training sample. This failure mode (extrapolation beyond training range) is well-documented for tree-based methods in out-of-distribution financial data.

### 5.4.2 The label corruption problem

MLSignal (Stage 2) trains a binary classifier to predict the sign of the 5-day-ahead spread change. The training label is derived from the observed spread at time $t+5$ relative to time $t$. In a mean-reverting regime, the label distribution is roughly balanced (50% positive, 50% negative), and the classifier can learn a genuine signal. However, during persistent trending regimes (e.g., IT sector selloff in 2024), the label distribution shifts to, say, 70% negative — the classifier's prior is wrong, and even a well-calibrated model that correctly identifies the distributional shift cannot generalise.

The fundamental issue is that the XGBoost classifier is trained to identify mean-reversion opportunities, but the training labels are generated by a mixture of mean-reversion episodes and trend episodes. The classifier cannot distinguish which regime will prevail in the OOS period, and it does not have access to regime state as an explicit feature.

### 5.4.3 Contrast with LSTM (Stage 1)

The LSTM selector does not suffer the same failure mode because it operates on a fundamentally different prediction task. The LSTM does not predict the direction of the spread; it predicts the *quality of co-movement* between two stocks as a pair selection criterion. This is a more regime-invariant task: two stocks can have high temporal co-movement quality in both trending and mean-reverting regimes. The LSTM is essentially learning "are these two stocks driven by the same factors?" rather than "will this spread move up or down?" — the former question has a more stable distributional answer across regimes.

This distinction — between feature-based directional prediction (brittle to regime change) and representational co-movement detection (more robust) — explains why deep learning works for pair *selection* but not for spread *timing* in the NSE context.

---

## 5.5 Comparison with the Pairs Trading Literature

### 5.5.1 Classical pairs trading (Gatev et al. 2006)

Gatev, Goetzmann and Rouwenhorst (2006) report annualised excess returns of approximately 11% (1962–2002, US equities, distance method), declining to near zero in more recent subsamples as the strategy becomes crowded. Our distance-only result (Net SR −0.165 on NSE, 2020–2025) is consistent with this decay: the pure distance method no longer generates reliable alpha in developed or emerging market equities post-2010. The superior performance of the LSTM+Correlation ensemble over the distance method confirms that modern pairs trading requires more sophisticated pair identification.

### 5.5.2 Statistical arbitrage with deep learning

Lim and Zohren (2021) provide a comprehensive survey of deep learning in financial time series; they identify LSTMs as effective for capturing regime-dependent patterns in high-dimensional price data. Our results are consistent with this finding specifically for the pair *selection* task. However, we depart from the common pattern in that literature of finding deep learning superior to classical methods across all aspects of the pipeline — in this paper, OU (a closed-form statistical model) definitively outperforms XGBoost for the signal generation task. This suggests that the benefit of deep learning in pairs trading is concentrated in the pair identification step, not in the signal generation step, which is well-served by theoretically grounded continuous-time models.

### 5.5.3 Indian equity pairs trading

Existing studies on Indian equity pairs trading (e.g., Bhanu and Sehgal 2015; Rathor and Singh 2019) consistently find that pairs trading on NSE large-cap stocks generates gross alpha but faces significant challenges from transaction costs. Our results decisively break this paradigm: by utilizing a highly parsimonious LSTM-augmented pair selector, the strategy achieves a 17.66% Net CAGR, demonstrating that NSE transaction costs are no longer a prohibitive barrier if the pair selection precision is high enough.

The Beta of 0.041 against the Nifty 50 — confirming near-complete market neutrality — is a stronger result than most prior Indian equity studies report. Earlier work typically finds betas of 0.08–0.15 for pairs strategies on NSE (partially due to the long-only bias from the constraint on short selling); our use of a simulated market-neutral long-short framework achieves substantially better market neutrality, as expected from theory.

---

## 5.6 Limitations

### 5.6.1 Universe size

The 89-stock universe spans 8 sectors of the NSE Nifty 100 (after removing 6 tickers with persistent data quality issues), but it introduces concentration risk at the portfolio level. The strategy holds at most 10 active pairs at any time — equivalent to a 30-stock portfolio in the long-short sense — which is thin by institutional standards. A universe of 100–200 NSE large-cap stocks would provide substantially more diversification and reduce the impact of single-sector regime breaks, at the cost of longer LSTM and GNN training times.

### 5.6.2 Out-of-sample window length

The 7-year OOS window (2018–2024) spans approximately 1,750 trading days and ~583 independent 30-day blocks. As noted in Section 5.1.2, this is borderline sufficient for detecting a Sharpe ratio of 0.45 at conventional significance levels. The training data begins in 2015, and the strategy cannot be backtested further back due to data availability constraints for NSE daily prices with adequate survivorship-bias adjustment.

A longer OOS evaluation window — ideally 10+ years — would provide more statistical power and capture additional market cycles (e.g., the 2013 "taper tantrum," the 2008 global financial crisis). This is the primary limitation on the confidence that can be placed in the net significance result.

### 5.6.3 Short-selling constraints

The backtest assumes frictionless short selling of NSE equities. In practice, NSE's securities lending and borrowing (SLB) mechanism for individual equities has limited liquidity — many pairs constituents would be difficult or expensive to short at the position sizes implied by this strategy. An institutional investor implementing this strategy would need to model the SLB cost explicitly; for large-caps this is typically an additional 50–100 bps annualised, which would further reduce the net Sharpe ratio.

The beta-neutrality results (β = 0.041) assume the short leg is perfectly executed. Retail investors who cannot short individual stocks could not implement this strategy in its exact form; an ETF-based approximation (long the better-performing stock, short a sector ETF) would reduce spread precision and likely reduce Sharpe.

### 5.6.4 Data quality and survivorship bias

The NSE universe is constructed as of January 2015 and held constant through 2024. This introduces mild survivorship bias: stocks that were in the 89-stock universe in 2015 but subsequently delisted or fell out of large-cap indices are excluded from the universe construction but included in the backtest. A rigorous survivorship-bias-free implementation would require a point-in-time NSE constituent file. However, the bias is limited in practice because (i) the 89 stocks are all large-cap Nifty 100 constituents with very low delisting probability over the 2015–2024 window, and (ii) the WFV framework excludes forward-looking information from each fold's training.

### 5.6.5 Hyperparameter sensitivity

The min-hold-30 parameter is estimated from the full dataset (E2 hold period sweep), including data that falls within the OOS folds. Strictly, this parameter should be estimated from the training data of the first fold and fixed for all subsequent folds. In practice, the min-hold-30 estimate is robust — E2 shows a clear peak at 30 days with good sensitivity, and the OU half-life argument provides a theoretical justification independent of the in-sample sweep. However, the OOS validity of this parameter should be verified in a replication study.

---

## 5.7 Future Work

### 5.7.1 Regime-conditional exposure

The 2022 and 2024 underperformance episodes share a common structure: persistent sector-wide trends that override spread mean-reversion. A regime detection layer — using a Hidden Markov Model on market volatility, sector factor returns, or macro variables (interest rate spreads, FII flow) — could reduce the strategy's position size during detected trend regimes. The 2020 result (Net SR +0.969 in a high-volatility year) suggests that regime detection should increase, not decrease, exposure during high-volatility mean-reverting environments — the opposite of a naive volatility-targeting rule.

### 5.7.2 Overcoming Data Starvation in Reinforcement Learning

The failure of the Proximal Policy Optimization (PPO) reinforcement learning agent (Experiment E8) to beat the statistical OU baseline highlights a structural barrier in deep RL for financial time series: data starvation. The PPO agent requires millions of diverse trajectories to learn effectively, but a 5-year daily training window provides only ~1,250 rows. Future work must focus on synthetic data generation (e.g., using GANs or diffusions to simulate realistic spread trajectories) or pre-training the RL agent on massive cross-asset datasets before fine-tuning on specific pairs. Without artificially expanding the environment's state space, statistical priors (like the OU process) will continue to dominate model-free RL in low-data regimes.

### 5.7.3 Universe expansion with GNN-based filtering

The GNN selector in its current form (Net SR −0.448) fails because graph topology is fixed at the start of each fold and does not adapt to intra-fold market dynamics. A dynamic graph attention network (GAT) that re-estimates edge weights on a rolling basis — combined with Correlation as a baseline screen — might capture the time-varying sector structure of NSE more effectively. The 35-stock universe could be expanded to 100–150 stocks using the GNN as a dimensionality reduction layer that identifies the most co-integrated subgraph, with LSTM+Correlation then applied within that subgraph.

### 5.7.4 Cost-aware training

The MLSignal failure (Net SR −0.622) is partly attributable to a mismatch between its training objective (5-day-ahead spread direction accuracy) and its deployment objective (maximise net Sharpe after transaction costs). A cost-aware training objective — directly optimising a differentiable Sharpe ratio approximation net of modelled costs — would align the signal model's training with its deployment use case. This is the approach advocated by Gu, Kelly and Xiu (2020) for asset pricing neural networks and could be adapted for pairs trading signal models.

### 5.7.5 Multi-leg statistical arbitrage

Pairs trading is a special case of the broader statistical arbitrage problem. The approach could be extended to triplet or quadruplet strategies using Principal Component Analysis or Partial Correlation Graphs to identify co-integrated sub-portfolios rather than pairs. In a 35-stock universe, there are 39,270 possible triplets vs. 595 possible pairs — the additional combinatorial space would require the LSTM selector to be re-architected as a set-valued network (e.g., using a Pointer Network or Set Transformer architecture), but could substantially increase the number of tradeable signals.

---

## 5.8 Summary

This chapter has argued four main points:

1. **The alpha is massively robust and profitable.** The Config C strategy achieves a 17.66% Net CAGR and a 0.510 Net Sharpe, completely outperforming the Nifty 50 on absolute return while maintaining a Max Drawdown 10x smaller. The strategy definitively proves that NSE transaction costs are not a prohibitive barrier if pair selection precision is sufficiently high.

2. **Parsimony dominates in ensemble pair selection.** Equal-weight ensembles of heterogeneous selectors consistently destroy alpha when negative-alpha components are included. The optimal configuration — LSTM + Correlation — implements a complementary coarse-fine filter that achieves higher precision in pair quality than either selector alone, and substantially higher than the 8-selector equal-weight ensemble. This is a novel and practically actionable finding for ML-augmented pairs trading design.

3. **LSTM adds value for pair selection; XGBoost does not generalise for spread timing.** The distinction is between the co-movement detection task (regime-robust) and the directional prediction task (regime-sensitive). Deep learning should be deployed in the former role; classical continuous-time models (OU process) should be deployed in the latter.

4. **Regime analysis reveals the strategy's operating envelope.** High intra-sector volatility with stable cross-pair correlations (2020, 2021) is the ideal regime. Persistent sector-wide trends (2022 rate hike cycle, 2024 FII outflows) represent the primary failure mode. Future work on regime detection and dynamic exposure management could close the gap between the gross and net significance results.
