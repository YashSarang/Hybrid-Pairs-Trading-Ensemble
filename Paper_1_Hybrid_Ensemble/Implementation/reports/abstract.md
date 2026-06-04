# Abstract
<!-- STATUS: DRAFT — Numbers updated to 89-ticker universe where confirmed; [[PLACEHOLDER]] marks await E3/E6 completion -->

Pairs trading—taking simultaneous long and short positions in two historically co-moving stocks—is a well-studied market-neutral strategy, but its profitability on emerging markets with high transaction costs and limited liquidity remains an open question. This thesis presents a **two-stage hybrid ensemble** framework for pairs trading on Indian NSE Nifty 100 large-cap equities, combining classical statistical methods with deep learning models, and evaluates it under production-realistic conditions.

**Stage 1** (Pair Selection) ensembles eight algorithms spanning correlation, distance-based methods, cointegration, machine learning (XGBoost), and deep learning (LSTM, Transformer, Graph Neural Network). **Stage 2** (Signal Generation) ensembles four models: z-score threshold, Ornstein-Uhlenbeck mean-reversion, Kalman filter dynamic hedge, and gradient-boosted classifier. The strategy is evaluated on **89 NSE Nifty 100 stocks** across 8 sectors using expanding-window walk-forward validation with **six out-of-sample folds covering 2018–2024**, with all transaction costs modeled using the 2024–2026 NSE discount-broker cost structure (**16.28 basis points round-trip** including zero brokerage, exchange fees, STT, stamp duty, and slippage).

Key findings: (1) **Daily data outperforms hourly** (gross Sharpe 1.14 vs. 0.49) due to superior signal quality and lower turnover. (2) The optimal minimum holding period is **30 trading days**, aligning with the Ornstein-Uhlenbeck mean-reversion half-life. (3) The **statistical baseline** (stat_only + OU signal) achieves **net Sharpe +0.480**, CAGR 3.30%, MaxDD 12.72% over 6 OOS folds. (4) The **full hybrid ensemble** (ML + stat selectors + OU signal) achieves **net Sharpe +0.653**, CAGR 4.51%, MaxDD 10.43%, demonstrating measurable but modest ML contribution. (5) [[PLACEHOLDER — E3 ablation pending job 8704]]: Individual selector ablation results for the 89-ticker universe. (6) Equal-weight ensemble combination consistently destroys alpha — [[PLACEHOLDER: confirm with E3 results]]. The strategy substantially underperforms the Nifty 50 on absolute returns (SR 0.550 vs. 0.720) but maintains a MaxDD **3× smaller** (12.28% vs. 38.44%), confirming market-neutral characteristics.

Statistical significance: the stat_only configuration is marginally significant at the 10% level (bootstrap p=0.086, Newey-West p=0.097) but does not reach the conventional 5% threshold. [[PLACEHOLDER — E6 significance for full hybrid pending]].

This is among the first studies to combine Graph Neural Networks, Transformers, and LSTM selectors on NSE equity pairs under strict walk-forward validation and realistic Indian transaction cost assumptions. The results confirm that the full hybrid framework provides modest but consistent improvement over the statistical baseline, with the market-neutral drawdown profile as the primary investment case.

**Keywords:** Pairs trading, ensemble methods, deep learning, LSTM, Graph Neural Networks, Transformers, NSE equities, walk-forward validation, mean reversion, Ornstein-Uhlenbeck process

---

**Thesis Structure:**

- **Chapter 1:** Introduction — Motivation, research problem, and research questions
- **Chapter 2:** Literature Review — Classical pairs trading, cointegration, deep learning in finance, RL for trading
- **Chapter 3:** Methodology — Dataset, cost model, selector/signal algorithms, ensemble framework, backtesting
- **Chapter 4:** Results — Frequency analysis, hold period optimization, ablation study, walk-forward validation, benchmarks
- **Chapter 5:** Discussion — Why ML adds modest value; ensemble pitfalls; regime analysis
- **Chapter 6:** Conclusion — Summary of contributions, practical implications, limitations, future work

---

**Supervised by:** [Advisor Name]  
**Department:** [Computer Science / Financial Engineering / Statistics]  
**Institution:** IIT Bombay  
**Submission Date:** [Month Year]

---

**Word Count:** ~420 words (excluding metadata)

<!-- REVISION NOTES (remove before final submission):
  - All E1 (daily vs hourly), E4, E5 numbers reflect 89-ticker, 16.28 bps, 2015-2024 runs
  - E3 ablation (individual selector breakdown) PENDING job 8704 — fill in [[PLACEHOLDER]] fields
  - E6 significance for full hybrid PENDING — fill in [[PLACEHOLDER]] field
  - Config C (LSTM+Corr) specific results from old 35-ticker universe (SR 0.510, CAGR 17.66%) have been removed
  - Headline now is: stat_only SR 0.480 | full hybrid SR 0.653
-->
