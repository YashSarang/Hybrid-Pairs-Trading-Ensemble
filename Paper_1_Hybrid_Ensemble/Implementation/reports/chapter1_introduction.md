# Chapter 1 — Introduction
<!-- STATUS: DRAFT — Updated to 89-ticker, 2015-2024, 16.28 bps universe. [[PLACEHOLDER]] marks await E3/E6/E7 results -->

> **Status:** Draft v2 (2026-06-04). Universe updated from 35→89 stocks, 2016-2026→2015-2024, folds updated to 2018-2024. Results updated where confirmed (E1, E4, E5, E6 stat_only). E3 ablation and E6 full-mode pending.

---

## 1.1 Motivation

The search for returns that are uncorrelated with broad market movements is one of the oldest problems in quantitative finance. A long-only investor in Indian equities from 2018 to 2024 earned strong returns — the Nifty 50 compounded at approximately 12.8% per year — but endured a maximum drawdown of 38.4% (the 2020 Covid crash) and sustained high volatility of 18.3% per year. For an institutional allocator managing a large portfolio, an investor building a non-directional overlay, or a proprietary trading desk seeking to diversify factor exposure, the question is: can a systematic strategy generate positive risk-adjusted returns on Indian equities while remaining genuinely uncorrelated with the market index?

Pairs trading — taking simultaneous long and short positions in two stocks whose prices have historically co-moved, betting that any temporary divergence will revert — is one of the oldest and best-studied approaches to this problem. The strategy's market-neutral structure means that, in principle, it should generate returns driven by the idiosyncratic spread between two stocks rather than by the direction of the broad market. If the spread between TCS and Infosys widens beyond its historical range, a pairs trader buys the cheaper and sells the more expensive, expecting convergence regardless of whether the overall market rises or falls.

The challenge is that pairs trading, in its classical formulation, relies on two conditions that are increasingly hard to satisfy in modern equity markets: (i) a reliable method for identifying pairs with durable mean-reversion properties, and (ii) sufficiently low transaction costs to profit from the spreads that do revert. The first condition has been eroded by the growth of systematic quantitative trading, which has crowded out many classical pairs signals (Gatev et al. 2006; Do & Faff 2010). The second condition is particularly severe in the Indian context: at approximately **16.28 basis points round-trip** using discount brokers (zero brokerage, NSE exchange fees, STT, stamp duty, and slippage), NSE transaction costs remain higher than US equity markets, making the difference between gross and net performance a first-order concern rather than a footnote.

The emergence of deep learning has created a potential path forward on the first condition. LSTM networks can identify temporal patterns of co-movement that static correlation measures miss; Transformer encoders capture long-range dependencies in spread dynamics; Graph Neural Networks model the entire correlation structure of the equity universe simultaneously. These architectures have shown promise in preliminary studies on developed equity markets, but three important questions remain unanswered in the literature:

1. Do deep learning pair selectors generate reliable out-of-sample alpha on an emerging market like NSE, with its specific cost structure and regime dynamics?
2. How should classical statistical selectors and deep learning selectors be combined — does an equal-weight ensemble of all available selectors improve on the best individual, or does combining selectors with heterogeneous quality destroy alpha?
3. Which component of the pairs trading pipeline — pair *selection* or signal *generation* — benefits more from deep learning, and why?

This thesis provides answers to all three questions through a rigorous empirical study on **89 NSE Nifty 100 large-cap equities** over **2015–2024**, with strict out-of-sample validation using a 6-fold expanding-window walk-forward framework covering test years 2018–2024.

---

## 1.2 Research Problem

The central research problem is the design and evaluation of a **hybrid ensemble pairs trading strategy** that integrates classical statistical pair selection methods with deep learning approaches, evaluated under production-realistic conditions on NSE equities.

The problem has three sub-components:

**P1 — Pair Selection:** Given a universe of 89 NSE Nifty 100 large-cap equities and their candidate pair combinations, which combination of selection criteria most reliably identifies the 10 pairs that will exhibit profitable mean-reversion over a future OOS window? Does deep learning (LSTM, Transformer, GNN) improve selection quality relative to classical methods (Correlation, Distance, Cointegration, Combined Criteria), and how should the two classes of selector be combined?

**P2 — Signal Generation:** Given a selected set of 10 pairs, which signal model — Ornstein-Uhlenbeck threshold, rolling z-score, Kalman filter dynamic hedge, or XGBoost classifier — generates the most profitable entry and exit signals OOS? Does ensemble combination of multiple signal models improve on the best individual?

**P3 — Net Profitability:** Is the strategy's alpha — as measured by OOS Sharpe ratio, CAGR, and comparison to the Nifty 50 benchmark — statistically significant and economically meaningful after accounting for the full NSE transaction cost structure? Is the strategy genuinely market-neutral?

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

The empirical framework is a two-stage pipeline evaluated using **6-fold expanding-window walk-forward validation** covering test years 2018–2024 (training always starting from 2015-01-01).

**Stage 1 (Pair Selection)** deploys eight algorithms: four classical statistical selectors (Correlation, Distance, Cointegration, Combined Criteria) and four machine learning/deep learning selectors (XGBoost MLSelector, Bidirectional LSTM, Transformer encoder, Graph Convolutional Network). Selectors score all candidate pairs; a weighted ensemble selects the top 10.

**Stage 2 (Signal Generation)** applies four models to each selected pair: the OUThreshold (Ornstein-Uhlenbeck AR(1) estimation), ZScoreThreshold (rolling ±2σ bands), KalmanHedge (dynamic hedge ratio via linear Kalman filter), and MLSignal (XGBoost triclass classifier). A minimum holding period of 30 trading days is enforced to reduce transaction cost drag.

The cost model applies the full 2024–2026 NSE discount-broker charge structure (**16.28 bps round-trip**: zero brokerage, 0.345 bps exchange fee, 10 bps STT on sell, 1.5 bps stamp duty on buy, 0.01 bps SEBI fee, GST on applicable charges, and 2 bps slippage per leg). Statistical significance is assessed using the block bootstrap (B=10,000, block=30) and Newey-West HAC t-test, with Bonferroni correction for multiple testing.

The experimental programme spans seven experiments (E1–E7):
- **E1:** Daily vs hourly frequency comparison
- **E2:** Minimum hold period sweep
- **E3:** Full ablation study — each selector and signal model evaluated in isolation
- **E4:** Walk-forward validation — headline OOS result
- **E5:** Benchmark comparison against Nifty 50
- **E6:** Statistical significance testing
- **E7:** Weighted ensemble — testing pruned configurations against the equal-weight baseline

All models run CPU-only (CUDA_VISIBLE_DEVICES='', seed=42) for full reproducibility.

---

## 1.5 Key Findings

The thesis yields five main empirical findings:

**F1 — Daily frequency dominates.** Daily spreads exhibit substantially stronger mean-reversion (Hurst exponent ~0.19 daily vs ~0.25 hourly) and produce economically coherent, sector-consistent pairs. The hourly strategy becomes insolvent net of NSE transaction costs (net MaxDD >100%), confirming daily data as the correct frequency.

**F2 — Hold period optimum is theory-consistent.** The optimal minimum hold period is 30 trading days, yielding the peak net Sharpe ratio of +0.481 in the full-dataset sweep. This aligns with the estimated OU half-life of the selected pairs (≈20–30 days for Hurst ≈ 0.19), providing an independent theoretical validation of the empirically derived parameter.

**F3 — ML selectors provide modest but consistent improvement over the statistical baseline.** The statistical baseline (stat_only + OU) achieves Net SR **0.480**, CAGR 3.30%, MaxDD 12.72%. The full hybrid (stat + ML selectors + OU) achieves Net SR **0.653**, CAGR 4.51%, MaxDD 10.43% — a meaningful improvement in both return and risk. [[PLACEHOLDER: E3 ablation will confirm which individual selectors drive this improvement on 89-ticker universe]].

**F4 — OU signal dominates Stage 2; ML signal fails OOS.** The OUThreshold signal is clearly superior across all modes. Equal-weight ensemble combination of signal models consistently destroys alpha [[PLACEHOLDER: confirm with E3 results]]. The XGBoost-based MLSignal fails OOS due to feature distribution shift and label corruption across NSE regime changes.

**F5 — Strategy is market-neutral with modest but real alpha.** vs Nifty 50: Strategy SR 0.550 vs benchmark SR 0.720; Strategy MaxDD 12.28% vs benchmark MaxDD 38.44%; Strategy CAGR 3.76% vs benchmark CAGR 12.84%. The strategy underperforms on raw returns but demonstrates genuine market-neutrality with ~3× lower drawdown. Statistical significance: marginally significant at 10% for stat_only (bootstrap p=0.086, NW p=0.097); [[PLACEHOLDER: full hybrid significance pending E6 rerun]].

---

## 1.6 Contributions

This thesis makes five contributions to the academic literature on pairs trading and quantitative finance:

**C1 — Hybrid two-stage ensemble for NSE pairs trading.** The first study to combine eight pair selectors (four classical, four deep learning) with four signal models in a unified two-stage framework on Indian NSE Nifty 100 equities (89 stocks), evaluated with a rigorous 6-fold WFV protocol over 2015–2024.

**C2 — Quantified ML contribution on emerging market.** Empirical demonstration that the full hybrid ensemble (stat + ML selectors) provides Net SR 0.653 vs the statistical baseline's 0.480 on NSE — modest but consistent improvement. This is distinct from the developed-market literature where ML improvements are often larger.

**C3 — Differential deep learning contribution across pipeline stages.** An empirical separation of where deep learning adds value (pair *selection* via temporal co-movement detection) versus where it fails (spread *timing* via directional prediction) on an emerging market. This distinction is practically actionable for hybrid system design.

**C4 — NSE transaction cost analysis under discount-broker model.** A quantification of how the 16.28 bps NSE round-trip cost translates into cost drag (approximately 0.9–1.1 pp/year at the observed trade frequency), the implication for gross-to-net alpha gap, and the statistical power required to achieve net significance. This provides a template for evaluating pairs strategies in other emerging markets.

**C5 — Regime-conditional performance attribution.** A fold-by-fold analysis of how macro regime (Covid crash, bull run, rate hike cycle) determines strategy performance on NSE, identifying the failure mode (persistent sector-wide trends overriding spread mean-reversion) and its frequency in the 2018–2024 period.

---

## 1.7 Scope and Limitations

The study has three primary limitations that bound the scope of its claims:

**Universe size:** The 89-stock universe provides substantial breadth but introduces concentration risk at the 10-pair portfolio level. Conclusions about strategy scalability to a larger universe require separate investigation.

**Short-selling assumption:** The backtest assumes frictionless short selling at the closing price. In practice, NSE's SLB mechanism charges additional borrowing fees (estimated at 50–100 bps annualised) that are not modelled. The net alpha estimates in this thesis are therefore an upper bound on achievable net performance for an implementer who must borrow shares.

**OOS window length:** Seven years (2018–2024) provides approximately 583 independent 30-day blocks. Statistical power is borderline at the net level. A longer OOS window would provide more unambiguous net significance, which is a data availability constraint rather than a methodology limitation.

---

## 1.8 Thesis Organisation

The remainder of this thesis is organised as follows:

**Chapter 2 — Literature Review** situates the thesis within five bodies of prior work: classical pairs trading, statistical arbitrage foundations (cointegration, OU process, Hurst exponent), machine learning for pair selection, deep learning for financial time series, and ensemble methods in quantitative finance. The chapter concludes with a comparison table positioning the thesis against the most closely related papers.

**Chapter 3 — Data and Methodology** describes the 89-stock NSE Nifty 100 universe, data sources and preprocessing (2015–2024, daily frequency, Parquet cache), the full NSE transaction cost model, all eight Stage 1 pair selectors (with mathematical formulations verified against the implementation), all four Stage 2 signal models, the walk-forward validation design with its no-look-ahead guarantee, the backtesting engine, and the statistical significance testing procedures.

**Chapter 4 — Results** reports the empirical findings from all seven experiments (E1–E7): frequency comparison, hold period sweep, walk-forward validation across configurations, benchmark comparison, ablation study, weighted ensemble results, and statistical significance tests. All numbers are sourced from locked OOS result files; no in-sample results are reported as primary findings.

**Chapter 5 — Discussion** interprets the findings: the NSE cost structure as the driver of the gross-to-net significance gap; the parsimony principle and its consistency with ensemble learning theory; regime analysis explaining fold-by-fold performance; the XGBoost failure mode and its contrast with LSTM's success; comparison with prior literature; and limitations and future work directions.

<!-- REVISION NOTES (remove before final submission):
  - Universe: updated 35->89 stocks, 2016-2026->2015-2024, folds 2020-2025->2018-2024
  - Key results updated: E4 (stat_only SR 0.480, full SR 0.653), E5 (SR 0.550 vs 0.720), E6 (p=0.086/0.097 stat_only)
  - [[PLACEHOLDER]] items require E3 (ablation) and E6 (full hybrid significance) results — job 8704 running
  - Old Config C headline (SR 0.510, CAGR 17.66%) removed — was 35-ticker universe
  - P1 universe: removed specific "595 = C(35,2)" reference since 89-ticker pair count is different
-->
