# Abstract

Pairs trading—taking simultaneous long and short positions in two historically co-moving stocks—is a well-studied market-neutral strategy, but its profitability on emerging markets with high transaction costs and limited liquidity remains an open question. This thesis presents a **two-stage hybrid ensemble** framework for pairs trading on Indian NSE large-cap equities, combining classical statistical methods with deep learning models, and evaluates it under production-realistic conditions.

**Stage 1** (Pair Selection) ensembles eight algorithms spanning correlation, distance-based methods, cointegration, machine learning (XGBoost), and deep learning (LSTM, Transformer, Graph Neural Network). **Stage 2** (Signal Generation) ensembles four models: z-score threshold, Ornstein-Uhlenbeck mean-reversion, Kalman filter dynamic hedge, and gradient-boosted classifier. The strategy is evaluated on 35 NSE stocks across 8 sectors using expanding-window walk-forward validation with six out-of-sample folds covering 2020–2025, with all transaction costs modeled using the 2024–2026 NSE cost structure (16.3 basis points round-trip including zero brokerage from discount brokers, exchange fees, STT, stamp duty, and slippage).

Key findings: (1) **Daily data outperforms hourly** (gross Sharpe 1.14 vs. 0.49) due to superior signal quality and lower turnover. (2) The optimal minimum holding period is **30 trading days**, aligning with the Ornstein-Uhlenbeck mean-reversion half-life. (3) The **Ornstein-Uhlenbeck signal model** alone (net Sharpe +0.359) outperforms an equal-weight ensemble (net Sharpe −0.189), demonstrating that naive ensemble combination can destroy alpha. (4) A **pruned two-model selector ensemble** (LSTM + Correlation) paired with OU signal achieves the headline result: **net Sharpe ratio +0.510** and **net return +17.66%** over six years of OOS testing, substantially outperforming both the statistical baseline (net Sharpe +0.359) and the Nifty 50 benchmark.

This is the first study to combine Graph Neural Networks, Transformers, and LSTM selectors on NSE equity pairs, and the first to demonstrate net-of-cost profitability with strict walk-forward validation under realistic Indian transaction cost assumptions. The results confirm that deep learning models (specifically LSTM) add measurable value to pair selection when combined with classical methods in a weighted ensemble, but that model pruning is essential to avoid overfitting and noise aggregation.

**Keywords:** Pairs trading, ensemble methods, deep learning, LSTM, Graph Neural Networks, Transformers, NSE equities, walk-forward validation, mean reversion, Ornstein-Uhlenbeck process

---

**Thesis Structure:**

- **Chapter 1:** Introduction — Motivation, research problem, and research questions
- **Chapter 2:** Literature Review — Classical pairs trading, cointegration, deep learning in finance, RL for trading
- **Chapter 3:** Methodology — Dataset, cost model, selector/signal algorithms, ensemble framework, backtesting
- **Chapter 4:** Results — Frequency analysis, hold period optimization, ablation study, walk-forward validation, benchmarks
- **Chapter 5:** Discussion — Why gross alpha exists but net alpha is hard to capture; ensemble benefits and pitfalls
- **Chapter 6:** Conclusion — Summary of contributions, practical implications, limitations, future work

---

**Supervised by:** [Advisor Name]  
**Department:** [Computer Science / Financial Engineering / Statistics]  
**Institution:** [University Name]  
**Submission Date:** [Month Year]

---

**Word Count:** 379 words (excluding metadata)
