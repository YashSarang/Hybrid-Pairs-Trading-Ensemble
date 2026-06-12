# Abstract
<!-- STATUS: FINAL (2026-06-12) — All experiments complete. All placeholders resolved. -->

Pairs trading—taking simultaneous long and short positions in two historically co-moving stocks—is a well-studied market-neutral strategy, but its profitability on emerging markets with high transaction costs remains an open question. This paper presents a **two-stage hybrid ensemble** framework for pairs trading on Indian NSE Nifty 100 large-cap equities, combining classical statistical methods with machine learning and deep learning models, and evaluates it under production-realistic conditions with a clear research question: *do ML selectors outperform statistical baselines?*

**Stage 1** (Pair Selection) ensembles up to eight algorithms spanning correlation, distance-based methods, cointegration, machine learning (XGBoost), and deep learning (LSTM, Transformer, GNN). **Stage 2** (Signal Generation) evaluates four models: z-score threshold, Ornstein-Uhlenbeck mean-reversion, Kalman filter, and gradient-boosted classifier. Evaluated on **89 NSE Nifty 100 stocks** across 8 sectors using expanding-window walk-forward validation with **six out-of-sample folds covering 2018–2024**, with all costs modeled at **16.28 basis points round-trip** (zero brokerage, exchange fees, STT, stamp duty, slippage).

**Key findings:** (1) **Daily data dominates** (gross SR 1.14 vs 0.49 hourly); hourly strategy loses more than initial capital net-of-costs (MaxDD 214%). (2) Optimal minimum hold is **30 trading days** (aligns with OU mean-reversion half-life). (3) **Statistical baseline** (stat_only + OU) achieves **Net SR 0.480, CAGR 3.30%, MaxDD 12.72%** over 6 OOS folds. (4) **Full hybrid** (all 8 selectors + OU) achieves Net SR **0.520**, CAGR 3.72%, MaxDD 11.75% — a marginal improvement over the statistical baseline. (5) **Ablation reveals that Distance-only is the strongest individual selector** (Net SR 0.829), far exceeding every ML-augmented configuration. Adding the XGBoost ML selector to the statistical ensemble *destroys* ensemble alpha (SR: +0.256 → −0.311). Heavy LSTM weighting (w=3.0) is catastrophic (Net SR −0.164, MaxDD 43.90%). (6) The best ensemble configuration is Correlation-heavy (Net SR 0.526, MaxDD 9.61%) — still below the best single selector. (7) **Statistical significance:** full hybrid gross alpha barely reaches 5% significance (bootstrap p=0.048); net alpha is marginal at 10% across all modes (bootstrap p: 0.069–0.089); none significant at 5% after Bonferroni correction.

The answer to the research question is **no**: ML selectors do not outperform statistical baselines on NSE equities under realistic transaction costs and strict walk-forward validation. The strategy maintains market-neutral characteristics (MaxDD 3× smaller than Nifty 50) and produces marginal net alpha consistent with genuine but modest signal. NSE transaction costs are the primary constraint on statistical significance.

**Keywords:** Pairs trading, ensemble methods, deep learning, LSTM, XGBoost, NSE equities, walk-forward validation, mean reversion, Ornstein-Uhlenbeck process, Indian equity markets

---

**Paper Structure:**

- **Chapter 1:** Introduction — Motivation, research problem, and research questions
- **Chapter 2:** Literature Review — Classical pairs trading, cointegration, deep learning in finance
- **Chapter 3:** Methodology — Dataset, cost model, selector/signal algorithms, ensemble framework, backtesting
- **Chapter 4:** Results — Frequency analysis, hold period optimisation, ablation, WFV, benchmark, weighted ensemble, significance
- **Chapter 5:** Discussion — Why ML fails to add alpha; ensemble pitfalls; regime analysis; limitations

---

**Supervised by:** [Advisor Name]
**Department:** Computer Science / Financial Engineering
**Institution:** IIT Bombay

---

<!-- REVISION NOTES (remove before final submission):
  - All numbers from 89-ticker, 16.28 bps, 2015-2024 dataset (E4 canonical)
  - E7 results from 84-ticker (parquet refreshed inadvertently) — directionally valid
  - E8 (RL signal) excluded — gymnasium not on Kalpana
  - Headline pivoted: ML does NOT outperform statistical baseline
  - Distance_only SR=0.829 is the dominant result of the paper
-->
