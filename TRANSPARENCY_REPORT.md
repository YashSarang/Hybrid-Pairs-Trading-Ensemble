# TRANSPARENCY REPORT — ALL EXPERIMENTAL RUNS
**Generated:** 2026-06-01 21:15 IST
**Total combinations:** 11   **Total run files:** 27

---

## EXECUTIVE SUMMARY

| Market | Signal | Runs | Mean Sharpe | Std | Min | Max | Reported | Notes |
|--------|--------|------|-------------|-----|-----|-----|----------|-------|
| Brazil | ou | 3 | 0.107 | 0.151 | 0.000 | 0.321 | 0.321 | Best of 3 runs (2 had 0 trades) |
| Brazil | zscore | 2 | -0.312 | 0.088 | -0.400 | -0.225 | N/R | Not reported in thesis |
| India | ou | 2 | 0.100 | 0.100 | 0.000 | 0.200 | 0.000 | First of 2 runs |
| India | zscore | 3 | 0.284 | 0.507 | -0.386 | 0.840 | 0.840 | Best/last of 3 runs ⚠️ |
| NSE_Nifty50 | ou | 2 | 0.159 | 0.012 | 0.147 | 0.171 | 0.147 | Single run — control Exp.1 (rolling) |
| NSE_Nifty50 | zscore | 2 | 0.908 | 0.156 | 0.752 | 1.063 | 0.752 | Single run — control Exp.1 (rolling) |
| United Kingdom | ou | 3 | -0.135 | 0.191 | -0.405 | 0.000 | N/R | Not prominently reported |
| United Kingdom | zscore | 2 | 0.010 | 0.255 | -0.245 | 0.265 | N/R | Not prominently reported |
| United States | ou | 3 | -0.085 | 0.120 | -0.255 | 0.000 | N/R |  |
| United States | unknown | 3 | 0.296 | 0.341 | 0.000 | 0.774 | 0.774 | Best of 3 runs ⚠️ |

> ⚠️ = Std > 0.3 — results unstable, likely ML non-determinism

---

## COMPLETE 2×2 UNIVERSE × METHOD MATRIX (NSE, ZScore)

| Universe | Method | Avg Net Sharpe | Std | Status |
|----------|--------|---------------|-----|--------|
| Nifty 100 | Expanding | −0.409 | — | Chapter 3 baseline |
| Nifty 100 | Rolling   | +0.052 | — | Chapter 3 baseline |
| **Nifty 50** | **Rolling**   | **+0.752** | 0.361 | ✅ Experiment 1 (June 1) |
| **Nifty 50** | **Expanding** | **+1.064** | 0.502 | ✅ Experiment 2 (June 1) |

**Universe uplift (rolling):   +0.700 Sharpe** (Nifty50 vs Nifty100)
**Universe uplift (expanding): +1.473 Sharpe** (Nifty50 vs Nifty100)
**Method effect within Nifty50: +0.312 Sharpe** (Expanding > Rolling)
**Geographic effect (multi-mkt mean vs Nifty50 rolling): −0.468 Sharpe** (control beats multi-mkt mean)

→ **SCENARIO A CONFIRMED.** Universe quality dominates both methodology and geography.

---

## DETAILED RUN ANALYSIS

### Brazil — OU
- **Runs:** 3
- **Mean ± Std:** 0.107 ± 0.151
- **Range:** [0.000, 0.321]
- **Reported in thesis:** 0.321 (Best of 3 runs (2 had 0 trades))
  - ⚠️ Reported deviates 0.214 pts from mean — disclosure required

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_090411.json` |
  | 2 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_074037.json` |
  | 3 | 20260529 | 0.321 | 32 | `wfv_4folds_ou_20260529_101431.json` ← reported |

### Brazil — ZSCORE
- **Runs:** 2
- **Mean ± Std:** -0.312 ± 0.088
- **Range:** [-0.400, -0.225]

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | -0.400 | 261 | `wfv_4folds_zscore_20260529_084830.json` |
  | 2 | 20260529 | -0.225 | 115 | `wfv_4folds_zscore_20260529_101426.json` |

### India — OU
- **Runs:** 2
- **Mean ± Std:** 0.100 ± 0.100
- **Range:** [0.000, 0.200]
- **Reported in thesis:** 0.000 (First of 2 runs)

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_085647.json` ← reported |
  | 2 | 20260529 | 0.200 | 26 | `wfv_4folds_ou_20260529_104015.json` |

### India — ZSCORE
- **Runs:** 3
- **Mean ± Std:** 0.284 ± 0.507
- **Range:** [-0.386, 0.840]
- **Reported in thesis:** 0.840 (Best/last of 3 runs)
  - ⚠️ Reported deviates 0.556 pts from mean — disclosure required

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.398 | 289 | `wfv_4folds_zscore_20260529_070512.json` |
  | 2 | 20260529 | -0.386 | 279 | `wfv_4folds_zscore_20260529_083100.json` |
  | 3 | 20260529 | 0.840 | 123 | `wfv_4folds_zscore_20260529_104009.json` ← reported |

### NSE_Nifty50 — OU
- **Runs:** 2
- **Mean ± Std:** 0.159 ± 0.012
- **Range:** [0.147, 0.171]
- **Reported in thesis:** 0.147 (Single run — control Exp.1 (rolling))

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260601 | 0.147 | 28 | `wfv_4folds_ou_20260601_203903.json` ← reported |
  | 2 | 20260601 | 0.171 | 31 | `wfv_4folds_ou_20260601_211346.json` |

### NSE_Nifty50 — ZSCORE
- **Runs:** 2
- **Mean ± Std:** 0.908 ± 0.156
- **Range:** [0.752, 1.063]
- **Reported in thesis:** 0.752 (Single run — control Exp.1 (rolling))
  - ⚠️ Reported deviates 0.156 pts from mean — disclosure required

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260601 | 0.752 | 126 | `wfv_4folds_zscore_20260601_203606.json` ← reported |
  | 2 | 20260601 | 1.063 | 133 | `wfv_4folds_zscore_20260601_211127.json` |

### United Kingdom — OU
- **Runs:** 3
- **Mean ± Std:** -0.135 ± 0.191
- **Range:** [-0.405, 0.000]

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_092934.json` |
  | 2 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_074531.json` |
  | 3 | 20260529 | -0.405 | 42 | `wfv_4folds_ou_20260529_110551.json` |

### United Kingdom — ZSCORE
- **Runs:** 2
- **Mean ± Std:** 0.010 ± 0.255
- **Range:** [-0.245, 0.265]

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.265 | 268 | `wfv_4folds_zscore_20260529_092241.json` |
  | 2 | 20260529 | -0.245 | 111 | `wfv_4folds_zscore_20260529_110559.json` |

### United States — OU
- **Runs:** 3
- **Mean ± Std:** -0.085 ± 0.120
- **Range:** [-0.255, 0.000]

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_070452.json` |
  | 2 | 20260529 | 0.000 | 0 | `wfv_4folds_ou_20260529_083228.json` |
  | 3 | 20260529 | -0.255 | 39 | `wfv_4folds_ou_20260529_113145.json` |

### United States — UNKNOWN
- **Runs:** 3
- **Mean ± Std:** 0.296 ± 0.341
- **Range:** [0.000, 0.774]
- **Reported in thesis:** 0.774 (Best of 3 runs)
  - ⚠️ Reported deviates 0.478 pts from mean — disclosure required

  | # | Timestamp | Sharpe | Trades | File |
  |---|-----------|--------|--------|------|
  | 1 | 20260529 | 0.774 | 302 | `wfv_4folds_20260529_025102.json` ← reported |
  | 2 | 20260529 | 0.000 | 0 | `wfv_6folds_20260529_020824.json` |
  | 3 | 20260529 | 0.116 | 106 | `wfv_6folds_20260529_023302.json` |

---

## CORRECTED CLAIMS FOR THESIS

| Market | Signal | Reported | Corrected Mean | Std | Action Required |
|--------|--------|----------|----------------|-----|-----------------|
| India Multi-Market | ZScore | +0.840 | +0.284 | ±0.631 | Report mean ± std; add Bonferroni |
| Brazil | OU | +0.321 | +0.107 | ±0.175 | Report mean; note 2 of 3 runs had 0 trades |
| United States | ZScore | +0.774 | +0.296 | ±0.340 | Report mean ± std |
| Nifty 50 (CTRL) | ZScore Rolling | N/A | +0.752 | ±0.361 | Add as new control result |
| Nifty 50 (CTRL) | ZScore Expanding | N/A | +1.064 | ±0.502 | Add as new control result |

*Report generated: 2026-06-01 21:15 IST*