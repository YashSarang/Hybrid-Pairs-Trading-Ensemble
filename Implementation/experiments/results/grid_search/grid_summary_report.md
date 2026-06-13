# E4 Ensemble Weight Space Search — Results Report
> Generated: 2026-06-13 02:34

## Overview
- Total configurations evaluated: **36**
- Selectors in search space: Correlation, Distance, Cointegration, Combined, ML, LSTM, Transformer, GNN
- S2 signal model: OU-only (fixed)
- Universe: 89-ticker NSE Nifty 100
- OOS period: 2018–2024 (6-fold expanding WFV)
- Transaction costs: 16.28 bps round-trip (IndianCosts)

---

## Table E4-Full: All Configurations Ranked by OOS Net Sharpe

| Rank | Configuration | N-Sel | Net SR | ±Std | Gross SR | Net CAGR | MaxDD | Trades | Folds+ |
|------|---------------|-------|--------|------|----------|----------|-------|--------|--------|
| 1 | Corr+Coint | 2 | 0.7259 | 1.252 | 0.7984 | 5.47% | 8.37% | 460 | 83% |
| 2 | ML | 1 | 0.6103 | 0.968 | 0.6557 | 6.55% | 11.33% | 424 | 83% |
| 3 | Coint+ML | 2 | 0.5898 | 0.735 | 0.6442 | 7.30% | 9.64% | 506 | 67% |
| 4 | Comb+ML | 2 | 0.5784 | 0.645 | 0.6328 | 7.60% | 9.64% | 469 | 67% |
| 5 | Dist+ML | 2 | 0.5702 | 0.809 | 0.6220 | 6.62% | 7.26% | 422 | 83% |
| 6 | Corr+Dist | 2 | 0.4977 | 0.662 | 0.5527 | 4.97% | 7.10% | 417 | 83% |
| 7 | Coint+Comb | 2 | 0.4852 | 1.165 | 0.5365 | 5.02% | 10.55% | 470 | 67% |
| 8 | Dist | 1 | 0.4444 | 0.832 | 0.4953 | 4.61% | 7.62% | 410 | 83% |
| 9 | Dist+LSTM | 2 | 0.4385 | 0.833 | 0.4910 | 4.49% | 7.52% | 412 | 83% |
| 10 | Dist+Trans | 2 | 0.4385 | 0.833 | 0.4910 | 4.49% | 7.52% | 412 | 83% |
| 11 | ML+LSTM | 2 | 0.4355 | 0.406 | 0.4728 | 6.06% | 14.15% | 396 | 83% |
| 12 | ML+Trans | 2 | 0.3797 | 1.226 | 0.4156 | 3.75% | 18.00% | 436 | 50% |
| 13 | Corr+Comb | 2 | 0.3354 | 0.706 | 0.3932 | 3.36% | 6.99% | 453 | 67% |
| 14 | Coint+Trans | 2 | 0.3178 | 0.791 | 0.3594 | 4.49% | 11.28% | 412 | 67% |
| 15 | Coint | 1 | 0.1665 | 0.925 | 0.2142 | 1.04% | 14.31% | 461 | 50% |
| 16 | Comb+LSTM | 2 | 0.1149 | 0.952 | 0.1539 | -0.10% | 18.24% | 431 | 50% |
| 17 | Comb | 1 | 0.1092 | 0.435 | 0.1529 | 1.16% | 13.88% | 417 | 50% |
| 18 | Coint+LSTM | 2 | 0.0995 | 1.269 | 0.1379 | 1.16% | 18.01% | 413 | 50% |
| 19 | Trans | 1 | 0.0409 | 1.354 | 0.0722 | -0.64% | 18.46% | 381 | 50% |
| 20 | LSTM+Trans | 2 | 0.0183 | 1.380 | 0.0457 | -1.16% | 19.19% | 367 | 50% |
| 21 | Dist+Comb | 2 | -0.0872 | 0.924 | -0.0380 | -0.38% | 10.85% | 411 | 50% |
| 22 | Dist+Coint | 2 | -0.0953 | 0.934 | -0.0456 | -0.58% | 11.60% | 408 | 33% |
| 23 | GNN | 1 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 24 | Corr+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 25 | Dist+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 26 | Coint+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 27 | Comb+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 28 | ML+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 29 | LSTM+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 30 | Trans+GNN | 2 | -0.1211 | 0.680 | -0.0904 | -2.51% | 20.10% | 391 | 33% |
| 31 | Corr+ML | 2 | -0.1386 | 0.603 | -0.0767 | -2.13% | 11.20% | 460 | 67% |
| 32 | Corr+LSTM | 2 | -0.2195 | 0.887 | -0.1562 | -2.27% | 10.82% | 434 | 33% |
| 33 | Corr+Trans | 2 | -0.2321 | 0.391 | -0.1631 | -2.04% | 10.00% | 449 | 33% |
| 34 | Corr | 1 | -0.2339 | 0.730 | -0.1714 | -2.25% | 9.38% | 439 | 33% |
| 35 | Comb+Trans | 2 | -0.2858 | 0.846 | -0.2434 | -2.64% | 18.09% | 431 | 17% |
| 36 | LSTM | 1 | -1.0342 | 1.312 | -1.0054 | -18.92% | 30.67% | 439 | 17% |

---

## Standalone Benchmarks (E4.S)

| Selector | Net SR | Gross SR | Net CAGR | Trades | Result |
|----------|--------|----------|----------|--------|--------|
| ML | 0.6103 | 0.6557 | 6.55% | 424 | ✅ Positive |
| Dist | 0.4444 | 0.4953 | 4.61% | 410 | ✅ Positive |
| Coint | 0.1665 | 0.2142 | 1.04% | 461 | ✅ Positive |
| Comb | 0.1092 | 0.1529 | 1.16% | 417 | ✅ Positive |
| Trans | 0.0409 | 0.0722 | -0.64% | 381 | ✅ Positive |
| GNN | -0.1211 | -0.0904 | -2.51% | 391 | ❌ Negative |
| Corr | -0.2339 | -0.1714 | -2.25% | 439 | ❌ Negative |
| LSTM | -1.0342 | -1.0054 | -18.92% | 439 | ❌ Negative |

---

## Best Pairwise Ensemble (E4.W2)

**Best 2-selector combination:** `Corr+Coint`
- Mean OOS Net SR: **0.7259**
- Net CAGR: 5.47%
- Std of fold SR: 1.2519

### All Pairwise Configurations Ranked

| Rank | Pair | Net SR | ±Std | Net CAGR |
|------|------|--------|------|----------|
| 1 | Corr+Coint | 0.7259 | 1.2519 | 5.47% |
| 2 | Coint+ML | 0.5898 | 0.7349 | 7.30% |
| 3 | Comb+ML | 0.5784 | 0.6449 | 7.60% |
| 4 | Dist+ML | 0.5702 | 0.8088 | 6.62% |
| 5 | Corr+Dist | 0.4977 | 0.6618 | 4.97% |
| 6 | Coint+Comb | 0.4852 | 1.1654 | 5.02% |
| 7 | Dist+LSTM | 0.4385 | 0.8330 | 4.49% |
| 8 | Dist+Trans | 0.4385 | 0.8330 | 4.49% |
| 9 | ML+LSTM | 0.4355 | 0.4060 | 6.06% |
| 10 | ML+Trans | 0.3797 | 1.2257 | 3.75% |
| 11 | Corr+Comb | 0.3354 | 0.7061 | 3.36% |
| 12 | Coint+Trans | 0.3178 | 0.7906 | 4.49% |
| 13 | Comb+LSTM | 0.1149 | 0.9520 | -0.10% |
| 14 | Coint+LSTM | 0.0995 | 1.2690 | 1.16% |
| 15 | LSTM+Trans | 0.0183 | 1.3796 | -1.16% |
| 16 | Dist+Comb | -0.0872 | 0.9240 | -0.38% |
| 17 | Dist+Coint | -0.0953 | 0.9345 | -0.58% |
| 18 | Corr+GNN | -0.1211 | 0.6800 | -2.51% |
| 19 | Dist+GNN | -0.1211 | 0.6800 | -2.51% |
| 20 | Coint+GNN | -0.1211 | 0.6800 | -2.51% |
| 21 | Comb+GNN | -0.1211 | 0.6800 | -2.51% |
| 22 | ML+GNN | -0.1211 | 0.6800 | -2.51% |
| 23 | LSTM+GNN | -0.1211 | 0.6800 | -2.51% |
| 24 | Trans+GNN | -0.1211 | 0.6800 | -2.51% |
| 25 | Corr+ML | -0.1386 | 0.6034 | -2.13% |
| 26 | Corr+LSTM | -0.2195 | 0.8873 | -2.27% |
| 27 | Corr+Trans | -0.2321 | 0.3912 | -2.04% |
| 28 | Comb+Trans | -0.2858 | 0.8457 | -2.64% |

---

## Key Findings

- **Best configuration:** `Corr+Coint` (Net SR = 0.7259)
- **Config C (Corr+LSTM):** Net SR = -0.2195
- **Delta (best vs Config C):** +0.9454 SR points
- **Standalone selectors with positive Net SR:** 5/8 (ML, Dist, Coint, Comb, Trans)

> **Interpretation:** If the best configuration across all C(8,2)+C(8,3) combinations is not significantly different from Config C (Corr+LSTM equal-weight) by DM test, this confirms the parsimony principle: equal-weight heuristics are near-optimal in this sample, and exhaustive weight search provides no significant benefit.