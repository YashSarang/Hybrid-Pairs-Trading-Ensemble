# KnowledgeGraph — Root (Two-Paper Project)
# Last updated: 2026-06-04

## Repository Structure
This repo contains TWO papers. Each has its own subfolder with its own KnowledgeGraph.

| Folder | Paper | Status |
|--------|-------|--------|
| Paper_1_Hybrid_Ensemble/ | "Hybrid Ensemble Pairs Trading: Do ML Selectors Outperform Statistical Baselines?" | PLAN ONLY — not yet written |
| Paper_2_Universe_Quality/ | "Universe Quality Dominates Methodology: A Multi-Market Ensemble Pairs Trading Study" | SUBMISSION READY |

## Navigation
- Paper 1 plan: Paper_1_Hybrid_Ensemble/Paper_1_Plan.md
- Paper 1 KG: Paper_1_Hybrid_Ensemble/KnowledgeGraph/KnowledgeGraph.md
- Paper 2 KG: Paper_2_Universe_Quality/KnowledgeGraph/KnowledgeGraph.md
- Paper 2 critiques: Paper_2_Universe_Quality/Critiques/INDEX.md (start here)

## Shared Infrastructure
- Implementation/core/: shared engine (selectors, backtest, ensemble, entry, data)
- Implementation/experimental-ablation/: Paper 2 experiments
- Implementation/experiments/: Paper 1 experiments (E1-E8, need cost recalculation)
- KnowledgeGraph/TokensInstructions.md: token efficiency rules

---

# Knowledge Graph — Hybrid Pairs Trading Ensemble

> **For AI coding agents:** Read this file first every session (~800 tokens). Load graph/*.json only if your task touches that area. Never load source files speculatively.

**Last updated:** 2026-06-04 IST

---

## THESIS STATUS — ROUND 4 REVISION IN PROGRESS

| Field | Value |
|-------|-------|
| Round 4 verdict | REJECT — Fundamental coherence failure |
| Open issues | 29 (7 FATAL, 8 MAJOR, 8 MODERATE, 6 MINOR) |
| JFM estimate | 20% → 50% after all FATAL+ablation fixed |
| QF estimate | 35% → 65% after all FATAL+ablation fixed |
| Headline result | NSE Nifty 50, stat-only, ZScore, rolling: **+0.752** (95% CI [+0.422, +1.082], p=0.036) |
| Only sig. result | Yes — the ONLY statistically significant result in the study |
| Active selectors | 7 (Correlation, Distance, Cointegration, Combined, LSTM AE, Transformer, GNN) — CNNSelector DISABLED |

### Framing
> Make Nifty 50 statistical-only the headline. ML ensemble = exploratory negative finding.

### 2×2 Matrix
| Universe | Method | Signal | Sharpe |
|----------|--------|--------|--------|
| Nifty 100 | Expanding | ZScore | −0.409 |
| Nifty 100 | Rolling | ZScore | +0.052 |
| **Nifty 50** | **Rolling** | **ZScore** | **+0.752** ← HEADLINE |
| Nifty 50 | Rolling | OU | +0.147 |
| India Multi-Mkt | Rolling | ZScore | +0.284 mean / +0.840 best-of-3 |

---

## PRIORITISED FIX LIST — ROUND 4

### FATAL (fix first — all 7 block resubmission)
1. Abstract: remove VAE + copula → replace with '4 stat + 3 ML = 7 active selectors'
2. Abstract: US ZScore −0.297 → +0.774 (exploratory, n=1, regime-contingent); fix geographic hierarchy
3. Global: '8-selector/4 ML' → '7 active/3 ML (CNNSelector disabled)' — abstract×3, Ch2×4, Ch4 Table 4.1.2, Ch5×2
4. Global: remove 16× multiplier + +0.840 as headline → +0.284 mean / 5.5× / CPU +0.353–+0.484 — Ch3 §3.6.7-9, Ch4 §4.3.1/4.3.3/4.7, Ch5 §5.1.1/5.2/5.6
5. Ch5 §5.1.1+5.2: remove '1.7× geographic > methodology' (Ch4 §4.2.2 already fixed — propagate)
6. Ch5 §5.1.1/5.1.2/5.6: remove 'geographic alpha is LARGE and REAL' → universe quality narrative
7. ML overfitting: add loss curves for one fold OR frame ML as exploratory negative finding

### MAJOR (after FATAL)
8. Run selector ablation: Nifty 50 4-fold rolling, stat-only vs ML-only vs combined
9. Ch2 §2.2.3: '192 model runs' → 7×4×4=112; update runtime
10. Ch5 §5.5.3: remove fund launch recommendation → research prerequisites
11. Ch4 Table 4.1.2: '8 selectors' → '7 active (CNNSelector disabled)'
12. Add US ZScore (+0.774, n=1, exploratory) to Table 4.2.1 + Appendix A
13. Replace 'proves/proving' → 'suggests/consistent with' throughout
14. 16.4 bps → 16.28 bps in Ch2 §2.3.1/2.3.3/2.5, Ch4 §4.1.3/4.3.1/4.5.3
15. Ch2: clarify '4-fold multi-market / 6-fold NSE baseline' in §2.1.3/2.2.3/2.4/2.5

### MODERATE (8) — see Documentation/PROJECT_NARRATIVE.md for full list (items 16–23)
### MINOR (6) — see Documentation/PROJECT_NARRATIVE.md for full list (items 24–29)

---

## Project Overview

| Field | Value |
|-------|-------|
| Name | Hybrid Pairs Trading Ensemble — M.S. by Research Thesis |
| Universe | NSE Nifty 50 (35 tickers, 8 sectors) primary; Nifty 100, US S&P 500, Brazil B3, UK FTSE 100 secondary |
| Pipeline | Stage 1 pair selection (7 selectors) → Stage 2 signal (ZScore/OU) → Vectorized backtest |
| Costs | India 16.28 bps r/t; US 2.74 bps; Brazil 8.4 bps; UK 8.0 bps |
| Cluster | SLURM — IIT Bombay Kalpana (account: cminds_anandi, partition: cn3_anandi) |
| Venv | `Implementation/.venv` (Python 3.12) |

## Key Decisions

| Decision | Result |
|----------|--------|
| Daily (1D) data | E1: Gross SR 1.14 daily vs 0.49 hourly |
| min_hold=30 bars | E2 optimum; locked in config.py |
| OU-only signal | E3: +0.359 vs ensemble −0.189 (stat-only) |
| Equal-weight ensemble harmful | E3: LSTM_only +0.305 > 8-selector ensemble −0.719 (full mode) |
| MLSelector label wrong | Momentum label ≠ mean-reversion quality; MLSelector weight=0 in E7 |
| CPU-only TF | Required for determinism; GPU variance = 1.226 |
| Transformer Lambda bug fixed | `_PositionalEncodingLayer` replaces Lambda closure (2026-04-06) |

## Active State

- **Status:** Round 4 REJECT — 29 open issues, revision in progress
- **Current focus:** Fix 7 FATAL issues before any other work
- **Next experiment needed:** Selector ablation (MAJOR-8) — stat-only vs ML-only vs combined on Nifty 50 4-fold rolling
- **Thesis files:** `Implementation/thesis_drafts/` (chapters 1–5 + abstract)
- **Experiment runner:** `Implementation/experimental-ablation/scripts/run_multi_market_wfv.py`
- **All results:** `Implementation/experimental-ablation/results/`

## Graph Update Rules
- Update Active State after every session
- Update graph/*.json after: new modules, experiments, decisions, results
- Target: this file ≤200 lines; move detail into graph/*.json if it grows
- See TokensGraphing.md for full maintenance protocol

## Which Graph File for Each Task
| Task | Load |
|------|------|
| Selector/signal changes | graph/modules.json |
| Experiment scripts | graph/experiments.json |
| Deps/cluster/env | graph/build-and-config.json |
| Tricky changes / edge cases | graph/decisions-and-gotchas.json |
| Token cost of any file | graph/token-cost-map.json |

## Never Load (without specific need)
- `core/selectors_ml.py` (39KB/~10k tokens)
- `app.py` (60KB/~15k tokens)
- `core/predictions.py` (21KB/~5k tokens)
- `experiments/results/` JSON blobs
