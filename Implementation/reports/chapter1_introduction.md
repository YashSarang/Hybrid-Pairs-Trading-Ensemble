# Chapter 1 — Introduction

> **Status:** Draft v1 (2026-04-06). Written after Chapters 2–5 to ensure accurate framing of results and contributions.

---

## 1.1 Motivation

The search for returns that are uncorrelated with broad market movements is one of the oldest problems in quantitative finance. A long-only investor in Indian equities from 2020 to 2025 earned strong returns — the Nifty 50 compounded at approximately 13.7% per year — but endured a maximum drawdown of 38.4% (the 2020 Covid crash) and sustained high volatility of 18.3% per year. For an institutional allocator managing a large portfolio, an investor building a non-directional overlay, or a proprietary trading desk seeking to diversify factor exposure, the question is: can a systematic strategy generate positive risk-adjusted returns on Indian equities while remaining genuinely uncorrelated with the market index?

Pairs trading — taking simultaneous long and short positions in two stocks whose prices have historically co-moved, betting that any temporary divergence will revert — is one of the oldest and best-studied approaches to this problem. The strategy's market-neutral structure means that, in principle, it should generate returns driven by the idiosyncratic spread between two stocks rather than by the direction of the broad market. If the spread between TCS and Infosys widens beyond its historical range, a pairs trader buys the cheaper and sells the more expensive, expecting convergence regardless of whether the overall market rises or falls.

The challenge is that pairs trading, in its classical formulation, relies on two conditions that are increasingly hard to satisfy in modern equity markets: (i) a reliable method for identifying pairs with durable mean-reversion properties, and (ii) sufficiently low transaction costs to profit from the spreads that do revert. The first condition has been eroded by the growth of systematic quantitative trading, which has crowded out many classical pairs signals (Gatev et al. 2006; Do & Faff 2010). The second condition is particularly severe in the Indian context: at approximately 60 basis points round-trip, NSE transaction costs are three to five times higher than in US equity markets, making the difference between gross and net performance a first-order concern rather than a footnote.

The emergence of deep learning has created a potential path forward on the first condition. LSTM networks can identify temporal patterns of co-movement that static correlation measures miss; Transformer encoders capture long-range dependencies in spread dynamics; Graph Neural Networks model the entire correlation structure of the equity universe simultaneously. These architectures have shown promise in preliminary studies on developed equity markets, but three important questions remain unanswered in the literature:

1. Do deep learning pair selectors generate reliable out-of-sample alpha on an emerging market like NSE, with its specific cost structure and regime dynamics?
2. How should classical statistical selectors and deep learning selectors be combined — does an equal-weight ensemble of all available selectors improve on the best individual, or does combining selectors with heterogeneous quality destroy alpha?
3. Which component of the pairs trading pipeline — pair *selection* or signal *generation* — benefits more from deep learning, and why?

This thesis provides answers to all three questions through a rigorous empirical study on 35 NSE large-cap equities over 2016–2026, with strict out-of-sample validation using a 6-fold expanding-window walk-forward framework.

---

## 1.2 Research Problem

The central research problem is the design and evaluation of a **hybrid ensemble pairs trading strategy** that integrates classical statistical pair selection methods with deep learning approaches, evaluated under production-realistic conditions on NSE equities.

The problem has three sub-components:

**P1 — Pair Selection:** Given a universe of 35 NSE large-cap equities and 595 candidate pair combinations, which combination of selection criteria most reliably identifies the 10 pairs that will exhibit profitable mean-reversion over a future one-year OOS window? Does deep learning (LSTM, Transformer, GNN) improve selection quality relative to classical methods (Correlation, Distance, Cointegration, Combined Criteria), and how should the two classes of selector be combined?

**P2 — Signal Generation:** Given a selected set of 10 pairs, which signal model — Ornstein-Uhlenbeck threshold, rolling z-score, Kalman filter dynamic hedge, or XGBoost classifier — generates the most profitable entry and exit signals OOS? Does ensemble combination of multiple signal models improve on the best individual?

**P3 — Net Profitability:** Is the strategy's alpha — as measured by OOS Sharpe ratio, CAGR, and Jensen's alpha relative to the Nifty 50 — statistically significant and economically meaningful after accounting for the full NSE transaction cost structure? Is the strategy genuinely market-neutral?

---

## 1.3 Research Questions

This thesis addresses five specific research questions:

**RQ1:** Does daily data produce superior pairs trading performance to hourly data on NSE equities, and why?

**RQ2:** What is the optimal minimum holding period for NSE pairs trades, and what is the economic mechanism underlying this optimum?

**RQ3:** Which individual pair selector generates the highest OOS net Sharpe ratio on NSE, and does an equal-weight ensemble of all selectors outperform the best individual?

**RQ4:** Which Stage 2 signal model generates the highest OOS net Sharpe ratio, and does ensemble combination of signal models outperform the best individual?

**RQ5:** Does the headline strategy generate statistically significant OOS alpha after accounting for NSE transaction costs, multiple testing, and the serial correlation in returns induced by the minimum hold period constraint?

---

## 1.4 Methodology Overview

The empirical framework is a two-stage pipeline evaluated using **6-fold expanding-window walk-forward validation** covering 2020–2025 (one OOS year per fold, training always starting from 2016-01-01).

**Stage 1 (Pair Selection)** deploys eight algorithms: four classical statistical selectors (Correlation, Distance, Cointegration, Combined Criteria) and four machine learning/deep learning selectors (XGBoost MLSelector, Bidirectional LSTM, Transformer encoder, Graph Convolutional Network). Selectors score all 595 candidate pairs; a weighted ensemble selects the top 10.

**Stage 2 (Signal Generation)** applies four models to each selected pair: the OUThreshold (Ornstein-Uhlenbeck AR(1) estimation), ZScoreThreshold (rolling ±2σ bands), KalmanHedge (dynamic hedge ratio via linear Kalman filter), and MLSignal (XGBoost triclass classifier). A minimum holding period of 30 trading days is enforced to reduce transaction cost drag.

The cost model applies the full NSE charge structure (~60 bps round-trip), and statistical significance is assessed using the block bootstrap (B=10,000, block=30) and Newey-West HAC t-test, with Bonferroni correction for multiple testing.

The experimental programme spans seven experiments (E1–E7):
- **E1:** Daily vs hourly frequency comparison
- **E2:** Minimum hold period sweep
- **E3:** Full ablation study — each selector and signal model evaluated in isolation
- **E4:** Walk-forward validation — headline OOS result
- **E5:** Benchmark comparison against Nifty 50, Nifty Bank, and Nifty IT
- **E6:** Statistical significance testing
- **E7:** Weighted ensemble — testing pruned configurations against the full equal-weight baseline

---

## 1.5 Key Findings

The thesis yields five main empirical findings:

**F1 — Daily frequency dominates.** Daily spreads exhibit substantially stronger mean-reversion (Hurst exponent 0.190 vs 0.251 at hourly frequency) and produce economically coherent, sector-consistent pairs. The hourly strategy becomes insolvent net of NSE transaction costs (net MaxDD 214%), confirming daily data as the correct frequency.

**F2 — Hold period optimum is theory-consistent.** The optimal minimum hold period is 30 trading days, yielding the peak net Sharpe ratio of +0.481 in the full-dataset sweep. This aligns with the estimated OU half-life of the selected pairs (≈20–30 days for Hurst ≈ 0.19), providing an independent theoretical validation of the empirically derived parameter.

**F3 — Parsimony principle in ensemble pair selection.** The 2-selector ensemble (LSTM + Correlation only) achieves a Full-OOS Net Sharpe of **0.451** — higher than the best individual selector (LSTM alone: 0.341) and more than double the 8-selector equal-weight ensemble (Net SR −0.660). Equal-weight combination of heterogeneous selectors, when several have negative expected OOS performance, systematically destroys alpha. This is the central novel finding of the thesis.

**F4 — LSTM improves pair selection; XGBoost fails for signal generation.** LSTM is the best individual Stage 1 selector (Net SR +0.341), demonstrating that temporal co-movement patterns beyond static correlation add genuine predictive value for pair identification. However, XGBoost-based signal generation (MLSignal, Net SR −0.622) is the worst signal model, because the directional spread prediction task is brittle to the regime shifts present in the 2020–2025 NSE data. The implication is that deep learning benefits are concentrated in the *regime-robust co-movement detection* task (pair selection), not the *regime-sensitive directional prediction* task (signal timing).

**F5 — Near-zero market beta and statistically significant gross alpha.** The headline strategy (LSTM + Correlation + OU, Config C) achieves Beta = 0.041 against the Nifty 50 — near-complete market neutrality — and Jensen's alpha of +3.08%/year. Gross alpha is statistically significant (block bootstrap p = 0.011; Newey-West p = 0.011). Net alpha is borderline significant (p = 0.084), with the gap attributable to the ~1.94 pp/year cost drag rather than signal weakness.

---

## 1.6 Contributions

This thesis makes five contributions to the academic literature on pairs trading and quantitative finance:

**C1 — Hybrid two-stage ensemble for NSE pairs trading.** The first study to combine eight pair selectors (four classical, four deep learning) with four signal models in a unified two-stage framework on Indian NSE equities, evaluated with a rigorous 6-fold WFV protocol.

**C2 — Parsimony principle for ensemble pair selection.** A concrete, mechanism-level demonstration that equal-weight inclusion of negative-alpha selectors in a pair selection ensemble destroys performance, and that a minimal 2-selector ensemble (LSTM + Correlation) achieves higher OOS Sharpe than any larger configuration. This finding extends ensemble learning theory to the pair selection problem and provides an actionable design rule: include only selectors with demonstrated positive OOS contribution.

**C3 — Differential deep learning contribution across pipeline stages.** An empirical separation of where deep learning adds value (pair *selection* via temporal co-movement detection) versus where it fails (spread *timing* via directional prediction) on an emerging market. This distinction is not present in any prior pairs trading study and has direct practical implications for hybrid system design.

**C4 — NSE transaction cost analysis.** A quantification of how the ~60 bps NSE round-trip cost translates into cost drag (≈1.94 pp/year at 86 trades/year), the implication for gross-to-net alpha gap, and the number of OOS years required to achieve statistical significance at the net level. This provides a template for evaluating pairs strategies in other high-cost emerging markets.

**C5 — Regime-conditional performance attribution.** A fold-by-fold analysis of how macro regime (Covid crash, bull run, rate hike cycle, FII reversal) determines strategy performance on NSE, identifying the failure mode (persistent sector-wide trends overriding spread mean-reversion) and its frequency in the 2020–2025 period.

---

## 1.7 Scope and Limitations

The study has three primary limitations that bound the scope of its claims:

**Universe size:** The 35-stock universe is sufficient for the experimental programme but introduces concentration risk. The 2024 underperformance is directly linked to IT sector-wide trends affecting a large fraction of the active pairs. Conclusions about strategy scalability to a larger universe require separate investigation.

**Short-selling assumption:** The backtest assumes frictionless short selling at the closing price. In practice, NSE's SLB mechanism charges additional borrowing fees (estimated at 50–100 bps annualised) that are not modelled. The net alpha estimates in this thesis are therefore an upper bound on achievable net performance for an implementer who must borrow shares.

**OOS window length:** Six years (2020–2025) is sufficient for statistical power at the gross level (p = 0.011) but borderline at the net level (p = 0.084). The claim of net statistical significance requires a longer OOS evaluation to become unambiguous, which is a data availability constraint rather than a methodology limitation.

---

## 1.8 Thesis Organisation

The remainder of this thesis is organised as follows:

**Chapter 2 — Literature Review** situates the thesis within five bodies of prior work: classical pairs trading, statistical arbitrage foundations (cointegration, OU process, Hurst exponent), machine learning for pair selection, deep learning for financial time series, and ensemble methods in quantitative finance. The chapter concludes with a comparison table positioning the thesis against the eleven most closely related papers.

**Chapter 3 — Data and Methodology** describes the 35-stock NSE universe, data sources and preprocessing, the full NSE transaction cost model, all eight Stage 1 pair selectors (with mathematical formulations verified against the implementation), all four Stage 2 signal models, the walk-forward validation design with its no-look-ahead guarantee, the backtesting engine, and the statistical significance testing procedures.

**Chapter 4 — Results** reports the empirical findings from all seven experiments (E1–E7): frequency comparison, hold period sweep, walk-forward validation across configurations, benchmark comparison, ablation study, weighted ensemble results, and statistical significance tests. All numbers are sourced from locked OOS result files; no in-sample results are reported as primary findings.

**Chapter 5 — Discussion** interprets the findings: the NSE cost structure as the driver of the gross-to-net significance gap; the parsimony principle and its consistency with ensemble learning theory; regime analysis explaining fold-by-fold performance; the XGBoost failure mode and its contrast with LSTM's success; comparison with prior literature; and limitations and future work directions.
