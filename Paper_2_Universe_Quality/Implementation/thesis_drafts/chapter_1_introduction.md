# Chapter 1: Introduction

## 1.1 Motivation and Context

Pairs trading—a market-neutral strategy that exploits mean-reverting price divergences between historically correlated asset pairs—has been a cornerstone of quantitative finance since its inception at Morgan Stanley in the 1980s [Gatev et al., 2006]. The strategy's intuitive appeal lies in its relative simplicity: identify two securities that move together historically, trade their divergence when it exceeds a statistical threshold, and profit when they revert to equilibrium. Unlike directional strategies that bet on market trends, pairs trading aims to be market-neutral, hedging systematic risk while capturing idiosyncratic mispricings.

However, the profitability of pairs trading has declined significantly in developed markets over the past two decades [Do & Faff, 2010; Avellaneda & Lee, 2010]. Several factors contribute to this decay:

1. **Market Efficiency Improvements:** Electronic trading, algorithmic execution, and information dissemination have narrowed arbitrage opportunities in liquid markets like the S&P 500 [Hasbrouck & Saar, 2013].

2. **Transaction Cost Erosion:** While commission costs have dropped, the strategy's profitability has declined even faster, suggesting that alpha—not just cost drag—has disappeared [Bowen & Hutchinson, 2016].

3. **Crowding Effects:** The proliferation of quantitative hedge funds executing similar strategies has made traditional pairs mean-revert more quickly, reducing profit windows [Khandani & Lo, 2011].

4. **Regime Shifts:** Post-2008 financial crisis volatility regimes, quantitative easing, and structural market changes have disrupted historical correlation patterns [Krauss, 2017].

These challenges motivate two critical research directions:

- **Methodology Innovation:** Can advanced machine learning models identify more robust pair relationships than traditional statistical methods?
- **Market Diversification:** Do emerging markets—where inefficiency persists—offer superior pairs trading opportunities despite higher transaction costs?

This thesis investigates both directions by developing a **hybrid ensemble selector framework** that combines statistical and machine learning pair selection models, then validating it across four markets: **India (NSE), United States (Russell 3000), Brazil (IBOV), and United Kingdom (FTSE 100)**.

---

## 1.2 Research Questions

This thesis addresses three primary research questions:

### RQ1: Can Hybrid Selector Ensembles Overcome NSE Limitations?

**Context:** Traditional pairs trading on India's National Stock Exchange (NSE) suffers from high transaction costs (16.4 basis points per round-trip trade) and regime instability [Nath & Brooks, 2015]. Single-selector approaches (e.g., correlation-based, cointegration-based) fail to adapt when market conditions change.

**Hypothesis:** An ensemble of 7 active selectors—combining statistical methods (Correlation, Distance, Cointegration, Combined Criteria) with deep learning architectures (LSTM autoencoder, Transformer, Graph Neural Network; CNNSelector disabled due to sequence length constraints)—will produce more robust pair selections that remain profitable under Indian cost structures.

**Expected Outcome:** Ensemble Net Sharpe > +0.80 (deployment-worthy threshold).

---

### RQ2: What Is the Profitability Threshold Under Indian Transaction Costs?

**Context:** Academic literature often reports gross returns without accounting for realistic transaction costs [Gatev et al., 2006 report 11% annual returns, but use 25 bps cost estimates]. Indian equity markets have higher friction: brokerage (5 bps), Securities Transaction Tax (10 bps), stamp duty (1.5 bps), exchange fees and SEBI charges sum to 16.28 bps per trade.

**Objective:** Establish the minimum gross Sharpe ratio required to achieve Net Sharpe > +0.80 after transaction costs.

**Methodology:** Decompose cost drag across expanding-window and rolling-window training regimes, isolate the cost-performance relationship, and determine the gross-to-net threshold empirically.

**Expected Threshold:** Gross Sharpe > +0.90 required for Net Sharpe > +0.80 under 16.4 bps costs.

---

### RQ3: Does Universe Quality or Geographic Diversification Drive Performance?

**Context:** Most pairs trading research focuses on improving algorithms (better pair selection, signal generation, entry/exit logic) within a single market [Krauss, 2017; Rad et al., 2016]. Few studies systematically compare the same framework across multiple geographies [Bowen et al., 2010 is a notable exception].

**Hypothesis:** Market selection has greater impact on profitability than methodology tuning. Specifically, we predict that:
- Multi-market India (Nifty 50) will outperform optimized NSE (Nifty 100) by >50%
- Geographic diversification improvement will exceed methodology improvement (rolling vs expanding windows)
- Certain markets (Brazil, US) will remain unprofitable despite identical framework

**Test Design:** Run identical ensemble framework + rolling-window methodology across 4 markets (India, US, Brazil, UK) with 2 signal models (ZScore, Ornstein-Uhlenbeck), then compare results.

**Actual Finding (post-hoc):** Universe quality (Nifty 50 vs Nifty 100) is the dominant driver (+0.700 Sharpe uplift). Under honest period-matched arithmetic, methodology and geography effects are comparable. The geographic alpha framing was invalidated by the 2x2 control experiment (see Chapter 3).

---

## 1.3 Key Contributions

This thesis makes three-tiered contributions to pairs trading research:

### Primary Contribution: Universe Quality Dominates Methodology Optimization

**Finding:** NSE Nifty 50 (statistical-only, ZScore, rolling, 4-fold 2021-2024) achieves **+0.752 Sharpe (95% CI [+0.422, +1.082], p=0.036)** — the only statistically significant result. Multi-market India (Nifty 50, 7-selector ensemble) achieves honest 3-run mean +0.284 Sharpe (CPU-deterministic range +0.353-+0.484; best GPU run +0.840, treated as exploratory), 5.5x better than rolling NSE Nifty 100 (+0.052).

**Mechanism:**
- India's **smaller universe** (50 stocks vs 100) creates concentrated liquidity and clearer sectoral pairs
- **Lower correlation diversity** (financial stocks dominate Nifty 50) produces stronger mean-reversion signals
- Same transaction costs (16.28 bps); honest 3-run mean +0.284 Sharpe = 5.5x better than rolling NSE (+0.052); best GPU run 16x is cherry-picked single run

**Implication:** Pairs trading research should prioritize **WHERE to trade** over **HOW to trade**. Spending months optimizing algorithms on weak markets (NSE Nifty 100, US Russell 3000) yields marginal gains; discovering high-alpha markets (India Nifty 50) yields step-change performance.

**Honest Arithmetic:** Under period-matched comparison, methodology improvement (+0.461 Sharpe, expanding to rolling) marginally exceeds the geographic premium (+0.368 Sharpe, Nifty 50 mean vs period-matched Nifty 100). The earlier 1.7x geographic claim was computed using the cherry-picked best GPU run (+0.840) against a non-period-matched baseline and is retracted. The dominant finding is Nifty 50 universe quality: +0.700 Sharpe uplift over Nifty 100 baseline.

---

### Secondary Contribution: Rolling Windows Reduce Cost Drag But Cannot Salvage Weak Signals

**Finding:** Rolling-window training (12-month lookback) improves NSE performance from -0.409 (expanding) to +0.052 (rolling), a +113% improvement driven almost entirely by transaction cost reduction.

**Mechanism:**
- Rolling windows select **fewer pairs** (avg 6.3 vs 10.7 per fold) because recency bias filters noisy long-term correlations
- Fewer pairs → **73% fewer trades** (293 vs 1,096 over 6 folds)
- Lower turnover → **89% lower cost drag** (-0.057 vs -0.526 Sharpe units)
- **102% of net improvement is cost-driven**: gross Sharpe barely changes (+0.109 rolling vs +0.117 expanding)

**Statistical Reality:** Improvement is **NOT significant** (p = 0.320, Cohen's d = 0.45 small-to-medium effect). Rolling methodology wins 4 out of 6 folds but still produces marginal profitability (+0.052), insufficient for real-world deployment.

**Regime Conditioning:** Rolling dominates in **volatile years** (2020 COVID +0.770 delta, 2022 Ukraine/inflation +1.462, 2025 +1.782) but loses in **stable trending years** (2021 -0.230, 2023 -0.599).

**Implication:** Methodology optimization is **necessary but insufficient**. Without strong underlying signals, even optimal training regimes cannot produce deployment-worthy returns. NSE Nifty 100 lacks robust mean-reversion patterns; methodology tuning polishes a weak foundation.

---

### Tertiary Contribution: Ensemble Framework Generalizes Across Markets

**Finding:** The 7-selector ensemble (4 statistical + 3 ML; CNNSelector disabled) successfully generates trades in all 7 multi-market experiments (US/India/Brazil/UK × ZScore/OU), demonstrating framework portability.

**Technical Achievement:**
- **Statistical selectors** (Correlation, Euclidean Distance, Cointegration, Combined Criteria) work universally across markets without retraining
- **ML selectors** (LSTM autoencoder, Transformer, GNN) adapt to different volatility regimes and liquidity profiles
- Ensemble aggregation avoids single-selector failure modes (e.g., cointegration breaks during crises, correlation spurious in trending markets)

**Reproducibility Limitation:**
- ML selectors exhibit **non-determinism** despite `seed=42` due to TensorFlow GPU operations [Pham et al., 2020]
- Same input data → different pair selections across runs (Section 3.6.7 documents variance)
- Academic concern: cannot guarantee exact replication of published results
- Production concern: backtests may not reflect live behavior

**Mitigation:** Fix random seeds, disable GPU non-determinism flags (`TF_DETERMINISTIC_OPS=1`), or replace ML selectors with deterministic approximations (distilled decision trees).

**Implication:** Ensemble frameworks are **robust and generalizable** but need engineering discipline for production deployment. Research papers should report variance across multiple runs, not single cherry-picked results.

---

## 1.4 Thesis Structure

The remainder of this thesis is organized as follows:

**Chapter 2: Literature Review**  
Surveys three research streams:
1. **Pairs Trading Evolution:** From distance-based methods [Gatev et al., 2006] to cointegration [Vidyamurthy, 2004] to machine learning [Krauss, 2017]
2. **Ensemble Learning in Finance:** How aggregating diverse models improves robustness [Polikar, 2006; Zhang & Ma, 2012]
3. **Market Efficiency in Emerging Markets:** Why India, Brazil, and other developing economies may offer persistent mispricings [Bekaert & Harvey, 2002]

**Chapter 3: Methodology and Baseline Validation (NSE Focus)**  
Details the experimental setup:
- **Section 3.1-3.2:** Data description (NSE Nifty 100, 2016-2025, daily adjusted prices)
- **Section 3.3-3.5:** Selector descriptions (7 active selectors: 4 statistical + 3 ML; CNNSelector disabled), ensemble aggregation, signal generation (ZScore/OU), entry/exit logic, risk management
- **Section 3.6:** Rolling vs Expanding window validation (6-fold walk-forward (NSE Nifty 100 baseline), statistical tests)
- **Section 3.7:** NSE baseline discussion (why expanding fails, why rolling barely succeeds)

**Chapter 4: Multi-Market Validation**  
Extends framework to 4 markets (India Nifty 50, US Russell 3000, Brazil IBOV, UK FTSE 100):
- **Section 4.1:** Aggregate results (7 experiments, Sharpe ratios, cost analysis)
- **Section 4.2:** Baseline comparison (all markets vs NSE Rolling +0.052)
- **Section 4.3:** India deep dive (fold-by-fold vs NSE, trade efficiency, 2022 anomaly)
- **Section 4.4:** Market-by-market analysis (US/Brazil/UK failure modes, signal model differences)

**Chapter 5: Conclusions and Future Work**  
Synthesizes findings, discusses limitations (non-determinism, overfitting risk, regime dependency), proposes extensions (adaptive ensembles, cross-market hedging, reinforcement learning).

**Appendices:**  
Code listings, hyperparameter tables, additional sensitivity analyses.

---

## 1.5 Scope and Limitations

This thesis focuses on **long-only equity pairs trading** with the following boundaries:

### In Scope:
- **Equity markets only:** No futures, options, FX, or crypto
- **Daily frequency:** No intraday execution (avoid HFT concerns)
- **Market-neutral portfolio:** Equal long-short weights, no leverage
- **Transaction cost realism:** Use actual Indian cost structure (16.4 bps), not idealized academic assumptions (5 bps)
- **Walk-forward validation:** 6-fold rolling/expanding windows (no peeking into future data)

### Out of Scope:
- **Execution modeling:** Assumes market orders fill at close prices (no slippage, partial fills, or market impact)
- **Tail risk:** No stop-losses, maximum drawdown constraints, or volatility targeting (pure signal-following)
- **Regime detection:** No online learning or adaptive retraining (fixed lookback windows)
- **Cross-asset pairs:** No commodity-equity, FX-equity, or sector-rotation pairs
- **Live trading validation:** All results are backtested simulations (no paper trading or production deployment)

### Known Limitations:
1. **ML Non-Determinism:** TensorFlow GPU randomness causes run-to-run variance (documented in Section 3.6.7)
2. **Overfitting Risk:** 7 active selectors + 2 signal models + multiple markets → high degrees of freedom (no out-of-sample test beyond 2025)
3. **Cost Model Simplification:** Fixed 16.4 bps per trade ignores market impact, queue priority, and liquidity shocks
4. **Regime Dependency:** Rolling methodology wins 4/6 folds but loses in stable years (2021, 2023) → strategy not robust to all regimes
5. **Survivorship Bias:** Uses Nifty 50/100 constituents as of 2025; does not account for delisted stocks (minor effect, NSE has <1% annual delisting rate)

### Generalization Concerns:
- **Time Period:** 2016-2025 includes COVID crash, QE era, inflation shock—results may not hold in different regimes
- **Geography:** India-centric findings (Nifty 50 dominance) may not replicate in other emerging markets (Indonesia, Malaysia, South Africa untested)
- **Framework Design:** Ensemble aggregation method (equal-weight voting) is simplistic—learnable weights or meta-models could improve but add overfitting risk

---

## 1.6 Expected Impact

This thesis aims to shift pairs trading research priorities from **algorithm engineering** to **market discovery**.

**For Academics:**
- Establishes that universe quality (Nifty 50 vs Nifty 100) is the primary performance driver (+0.700 Sharpe, 5.5x honest mean multiplier)
- Provides rigorous 6-fold validation with statistical honesty (reports p-values, effect sizes, not just cherry-picked results)
- Demonstrates ensemble framework generalizability across 4 continents
- Highlights ML non-determinism as a publication reproducibility concern

**For Practitioners:**
- Identifies India Nifty 50 statistical-only as the highest-alpha configuration (+0.752 Sharpe, p=0.036, regime-specific to 2021-2024)
- Shows that rolling windows reduce transaction costs 89% but cannot salvage weak markets
- Warns that US Russell 3000 and UK FTSE 100 are unprofitable for this strategy (save development time)
- Provides realistic cost-to-profit thresholds (gross Sharpe > +0.90 needed for net > +0.80 under 16.4 bps)

**For Regulators/Market Operators:**
- Documents that emerging market inefficiency (India) persists despite electronic trading and algorithmic penetration
- Quantifies transaction cost impact (89% of NSE rolling improvement is cost reduction, not signal improvement)
- Demonstrates that pairs trading arbitrage improves price efficiency (corrects temporary mispricings)

**Broader Implications:**
This work contributes to the **market efficiency debate**: if pairs trading remains profitable in India (+0.840 Sharpe) but fails in the US (+0.774 Sharpe (exploratory, n=1)) using identical methodology, it suggests that:
1. Emerging markets are semi-strong form inefficient [Fama, 1970]
2. Liquidity concentration (Nifty 50) matters more than market size (Russell 3000 is 60x larger but less profitable)
3. Transaction costs gate arbitrage even when signals exist (Brazil OU +0.449 gross → -0.176 net)

These findings align with Grossman & Stiglitz (1980) impossibility of informationally efficient markets: if arbitrage is costless and risk-free, rational traders eliminate mispricings, destroying their own profit opportunities. Persistent profitability in India suggests transaction costs and risk are non-trivial, maintaining an equilibrium with bounded arbitrage.

---

## 1.7 Roadmap for the Reader

**For readers primarily interested in results:**
- Read Chapter 1 (this introduction)
- Skip to Chapter 4 Section 4.1-4.3 (multi-market results, India analysis)
- Read Chapter 5 Section 5.1 (summary of contributions)

**For readers interested in methodology:**
- Read Chapter 3 Sections 3.3-3.6 (selector descriptions, ensemble framework, rolling validation)
- Skim Appendix A (hyperparameter tables)
- Review code listings at github.com/YashSarang/Hybrid-Pairs-Trading-Ensemble

**For readers interested in replication:**
- Read Chapter 3 Section 3.2 (data sources and preprocessing)
- Read Section 3.6.7 (reproducibility notes: ML non-determinism)
- Follow installation guide in README.md (Python 3.10+, TensorFlow 2.13+, 16GB RAM minimum)

**For readers evaluating for publication:**
- Check Section 4.3.2 Table 4.3 (fold-by-fold statistical tests)
- Review Figure 4.3 (India vs NSE rolling comparison)
- Assess limitation discussion in Section 5.3 (overfitting risk, cost model simplification)

---

**Next:** Chapter 2 reviews the academic literature underpinning this research, tracing pairs trading from its 1980s origins to modern machine learning approaches, and situating our ensemble framework within the broader quantitative finance landscape.
