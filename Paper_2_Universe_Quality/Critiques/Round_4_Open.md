---
status: ALL RESOLVED
round: 4
verdict: N/A — all items addressed as of 2026-06-04
last_updated: 2026-06-04
open_count: 0
---

# Round 4 — Critique Tracker

> Round 4 was a self-audit conducted against all surviving R1-R3 issues plus new issues identified during thesis draft review on 2026-06-04. All items resolved in the same session.

## FATAL (7/7 resolved)

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| F1 | VAE/copula in hypothesis/abstract | ✅ | Removed from Ch1, abstract, Ch3 |
| F2 | 8 selectors (should be 7, CNNSelector disabled) | ✅ | Fixed in Ch1, Ch3, Ch4, abstract |
| F3 | 16x multiplier uncaveated | ✅ | Caveated or replaced with 5.5x honest mean |
| F4 | 1.7x geographic claim | ✅ | Retracted; labeled cherry-picked in Ch1 |
| F5 | US ZScore = -0.297 (wrong) | ✅ | Corrected to +0.774 (exploratory n=1) |
| F6 | Geographic alpha dominates narrative | ✅ | Replaced with universe quality dominates throughout |
| F7 | Fund launch recommendation in Ch5 | ✅ | Removed; replaced with academic disclaimer |

## MAJOR (8/8 resolved)

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| M1 | Ch2: 192 model runs | ✅ | Already correct (112 = 7x4x4) in draft |
| M2 | Ch4 Table 4.1.2: 8 selectors | ✅ | Updated to 7 active (CNNSelector disabled) |
| M3 | Selector ablation table missing | ✅ | Section 4.4.12 added from real result files |
| M4 | ML overfitting disclosure absent | ✅ | Section 4.4.13 added; 3 lines of evidence |
| M5 | Brazil 8.4 vs 30 bps contradiction | ✅ | Disclosure note added; lower-bound caveat explicit |
| M6 | Section 4.3.4 missing (Liew and Wu) | ✅ | Section exists at Ch4 line 260 |
| M7 | Avellaneda NSE 0% stationarity not addressed | ✅ | Ch2 bridging explanation present |
| M8 | +0.284 vs +0.840 gap unexplained | ✅ | Ch4 Section 4.2 bridge paragraph present |

## MODERATE (8/8 resolved)

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| MOD1 | Abstract internally contradictory | ✅ | Abstract rewritten; all contradictions removed |
| MOD2 | 6-fold vs 4-fold inconsistency | ✅ | Ch1 roadmap clarified: 6-fold=Nifty100 baseline, 4-fold=Nifty50 primary |
| MOD3 | Universe quality claim from single NSE comparison | ✅ | Bounded in Ch5; cross-market replication = future work |
| MOD4 | Krauss 2017 misattributed for LSTM autoencoders | ✅ | Ch2 corrected |
| MOD5 | Survivorship bias contradiction | ✅ | Ch3 Section 3.2.1: mild look-ahead bias disclosed |
| MOD6 | VIX 2024 table self-contradicts | ✅ | Explicit caveat note added Ch4 Section 4.4.3a |
| MOD7 | Three different cost figures (16.4/16.28/16.5 bps) | ✅ | Ch1 fixed to 16.28 bps; all instances consistent |
| MOD8 | Flag emojis in academic text | ✅ | Removed from Ch4 and Ch5 |

## MINOR (6/6 resolved)

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| MIN1 | Brazil OU cherry-pick not flagged in narrative | ✅ | Ch4 note added; honest mean +0.107 stated |
| MIN2 | Ch5 closing generic | ✅ | Closing references specific Nifty50 finding |
| MIN3 | OU zero-folds pattern (n=1 effective) | ✅ | Abstract notes hypothesis-generating only |
| MIN4 | Gatev 1999 vs 2006 inconsistency | ✅ | Ch3 standardised to Gatev et al. (2006) |
| MIN5 | 2014-2025 date (dataset starts 2016) | ✅ | Ch1 corrected to 2016-2025 |
| MIN6 | Gross Sharpe threshold 0.60 vs 0.90 conflict | ✅ | Ch5 Section 5.1.2 note clarifies both configs |

## Summary

| Severity | Total | Resolved | Open |
|----------|-------|----------|------|
| FATAL | 7 | 7 | 0 |
| MAJOR | 8 | 8 | 0 |
| MODERATE | 8 | 8 | 0 |
| MINOR | 6 | 6 | 0 |
| **Total** | **29** | **29** | **0** |

Acceptance probability post-R4: JFM ~55%, QF ~70%
