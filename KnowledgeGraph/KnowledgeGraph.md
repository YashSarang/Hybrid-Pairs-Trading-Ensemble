# Knowledge Graph — Hybrid Pairs Trading Ensemble

> **For AI coding agents:** Read this file at the start of every session. It is the cheapest way (~800 tokens) to understand the full codebase. Do not read raw source files until you know you need them.

**Last updated:** 2026-06-09 IST

---

## 📋 THESIS STATUS — ALL ROUNDS RESOLVED (2026-06-02)

### Peer Review Summary
- **Round 4 critique date:** 2026-06-09 — ALL 29 ISSUES RESOLVED ✅
- **Ablation study:** Complete — Sections 3.3.3 + 3.3.4 written with Job 8492 data
- **Coherence pass:** Complete — abstract, Ch2, Ch4, Ch5 all updated
- **Open issues:** 0 blocking / 0 FATAL
- **Journal acceptance estimates:** JFM: 55% | QF: 70%
- **Recommendation:** Ready for submission after figure regeneration (Figure 4.1, 4.3 with CI error bars)

### Primary Finding (Statistically Significant)
**NSE Nifty 50, statistical-only (ZScore, rolling, 4 selectors):**
- Net Sharpe: **+0.752** (95% CI [+0.422, +1.082], p = 0.036)
- Fold-by-fold: +1.127, +0.218, +0.627, +1.036 (126 trades total)
- **Only statistically significant result in the entire study**

### Key Framing Insight
> **Invert the thesis:** Make NSE Nifty 50 statistical-only the headline result. The ML ensemble is an exploratory negative finding — not the centrepiece. This reframing is honest and more publishable.

### 2×2 Comparison Matrix
| Universe | Method | Signal | Sharpe | Status |
|----------|--------|--------|--------|--------|
| Nifty 100 | Expanding | ZScore | -0.409 | ✅ Chapter 3 |
| Nifty 100 | Rolling | ZScore | +0.052 | ✅ Chapter 3 |
| **Nifty 50 (CTRL)** | **Rolling** | **ZScore** | **+0.752** | ✅ **Headline result** |
| Nifty 50 (CTRL) | Rolling | OU | +0.147 | ✅ Chapter 3 |
| India Multi-Mkt | Rolling | ZScore | +0.840 | ⚠️ Best of 3 runs |
| India Multi-Mkt | Rolling | ZScore | +0.284 | ⚠️ Mean of 3 runs |

---

## 🔥 PRIORITISED FIX LIST — ROUND 4 (29 Open Issues)

### FATAL — Fix All Before Anything Else (7)

1. **Abstract: remove Variational Autoencoders and copula** — replace selector description with '4 statistical (Correlation, Distance, Cointegration, Combined) + 3 ML (LSTM autoencoder, Transformer, GNN), 7 total active selectors'
2. **Abstract: fix US ZScore** — change -0.297 to +0.774 (exploratory, n=1, regime-contingent). Revise 'India >> Brazil > UK > US' hierarchy accordingly.
3. **Global: replace ALL '8-selector/4 ML selectors'** — missed in Round 3. Locations: abstract (3), Ch2 (4), Ch4 Table 4.1.2, Ch5 (2). Replace with '7 active selectors / 3 ML selectors (CNNSelector disabled)'.
4. **Global: remove 16x multiplier / +0.840 as headline** — Ch3 Sec 3.6.7/3.6.8/3.6.9, Ch4 Sec 4.3.1/4.3.3/4.7, Ch5 Sec 5.1.1/5.2/5.6. Replace with +0.284 mean / 5.5x honest multiplier / CPU range +0.353–+0.484.
5. **Ch5 Sec 5.1.1 + 5.2: remove '1.7x geographic > methodology'** — Ch4 Sec 4.2.2 already has the correction; propagate it to Ch5.
6. **Ch5 Sec 5.1.1 heading + 5.1.2 + 5.6: remove 'geographic alpha is LARGE and REAL'** — replace with universe quality narrative throughout Ch5.
7. **ML overfitting: add training diagnostics** — add loss curves for one representative fold, OR explicitly frame ML as exploratory negative finding (statistically-only = +0.752, ensemble = +0.284, conclusion: ML adds noise on 12-month windows).

### MAJOR — Fix After FATAL (8)

8. **Selector ablation table** — run Nifty 50 4-fold rolling: statistical-only vs ML-only vs combined. This is the thesis's core methodological claim and is currently untested.
9. **Ch2 Sec 2.2.3: fix 192 model runs** — correct: 7×4×4=112. Update runtime claim.
10. **Ch5 Sec 5.5.3: remove fund launch recommendation** — replace with research prerequisites for commercialisation.
11. **Ch4 Table 4.1.2: '8 selectors' → '7 active selectors (CNNSelector disabled)'**
12. **Add US ZScore (+0.774, n=1, exploratory) to Table 4.2.1 and Appendix A Table A.1**
13. **Replace 'proves/proving' with 'suggests/is consistent with'** throughout abstract, Ch4, Ch5.
14. **16.4 bps → 16.28 bps in Ch2 (Sec 2.3.1, 2.3.3, 2.5) and Ch4 (Sec 4.1.3, 4.3.1, 4.5.3)**
15. **Ch2: add '4-fold for multi-market / 6-fold for NSE baseline'** clarification in Sec 2.1.3, 2.2.3, 2.4, 2.5.

### MODERATE (8)

16. Explain or remove 'results/ou' and 'results/unknown' artefacts in STATISTICAL_ANALYSIS.md
17. Ch4 Sec 4.7: replace +0.840/16x with +0.284 mean in chapter conclusions
18. Ch5 Sec 5.3.2 + 5.5.1: replace '+0.840' future targets with CPU-deterministic +0.353–+0.484
19. Ch2 Sec 2.3.3: cross-ref 'Section 4.3' → 'Section 4.3.4'
20. Ch3 References: Gatev 1999 → 2006
21. Ch5 RQ2: add note that gross Sharpe threshold is configuration-dependent (+0.60 rolling vs +0.90 general)
22. Ch4 lines 501-506: delete integration notes / TODO comments
23. STATISTICAL_ANALYSIS.md: remove 'results/unknown' duplicate of 'US/unknown'; relabel as 'US/ZScore (n=1 valid, 2 failed)'

### MINOR (6)

24. Ch3 Sec 3.6.6: rephrase 'production deployment' → 'for reproducibility in future research'
25. Ch5 Appendix B Table B.1: add Bonferroni-corrected p=0.640 row
26. Add 'Nath & Brooks (2015)' to reference list (cited Ch2 Sec 2.3.3)
27. Add 'Bhootra & Hur (2013)' to reference list (cited Ch4 Sec 4.3.1)
28. Ch5 Sec 5.2: 'Gatev 1999' → 'Gatev 2006'
29. Add NSE Nifty 50 Rolling OU control (+0.147, n=1) to Table 4.2.1

---

**If items 1–7 (FATAL) + 8 (ablation) + 7 (ML diagnostics) resolved → JFM: 50%, QF: 65%**

### MAJOR — Fix After FATAL (9 — note: original critique listed 9 MAJOR; items 6–11 are the first 6)

6. **Fix CNNSelector** — change all '8-selector' to '7-selector' throughout the thesis.
7. **Reconcile Brazil cost** — 8.4 bps used in backtest vs 30 bps cited in Chapter 2. Add footnote explaining discrepancy.
8. **Write Section 4.3.4** — Liew & Wu contradiction resolution.
9. **Add ML overfitting diagnostic note** to Chapter 3.
10. **Add selector ablation table** or at minimum a disclosure paragraph.
11. **Fix period-confounded methodology vs geography comparison.**

### MODERATE (7)

12. Fix Krauss 2017 misattribution.
13. Consolidate India cost to **16.28 bps** everywhere (remove 16.4 and 16.5 references).
14. Fix plain-language abstract date — '2014-2025' → correct study period.
15. Add caveat to VIX regime table acknowledging 2024 contradicts the causal story.
16. Fix OU results disclosure — note n=1 effective fold.

### MINOR (5)

17. Fix `results/ou` 5-zero runs — explain in transparency report.
18. Standardise Gatev citation to **2006** throughout.
19. Consolidate gross Sharpe threshold to one figure.

---

## How to Use This Knowledge Graph

The `KnowledgeGraph/graph/` folder contains structured JSON files that cache the repo's architecture. Load them instead of source files.

### Session Start Protocol (All Agents)

```
Step 1 — Always: Read KnowledgeGraph/KnowledgeGraph.md          (~800 tokens)
Step 2 — If needed: Read the relevant graph/ JSON files         (~200–600 tokens each)
Step 3 — Only if the task requires it: Read the specific source file
```

**Never** load source files speculatively at session start. Load graph files first, then load only the source file your task touches.

### Which Graph File to Load for Each Task

| Task type                                          | Load this graph file                        |
| -------------------------------------------------- | ------------------------------------------- |
| Adding/modifying a selector or signal model        | `graph/modules.json`                        |
| Running or building an experiment script           | `graph/experiments.json`                    |
| Changing dependencies, env, or cluster jobs        | `graph/build-and-config.json`               |
| Any change that feels tricky or has edge cases     | `graph/decisions-and-gotchas.json`          |
| Deciding whether to read a source file             | `graph/token-cost-map.json`                 |

### Files to Never Load

Check `graph/decisions-and-gotchas.json → doNotLoad` or `graph/token-cost-map.json → NEVER_LOAD` before reading any file. Do not load:

- `Implementation/core/selectors_ml.py` (39KB / ~10k tokens) — only load if modifying DL selectors
- `Implementation/app.py` (60KB / ~15k tokens) — only load for UI/Streamlit changes
- `Implementation/core/predictions.py` (21KB / ~5k tokens) — only load for real-time prediction work
- `experiments/results/` — JSON outputs, never need to be read in full

---

## Project Overview

| Field          | Value                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------- |
| **Name**       | Hybrid Pairs Trading Ensemble — M.S. by Research Thesis                                      |
| **Domain**     | Indian NSE equities (35 large-cap stocks, 8 sectors, Nifty 100)                             |
| **Framework**  | Python, Streamlit UI (`app.py`), UI-agnostic `core/` library                                |
| **Cluster**    | CMInDS Kalpana cluster — SLURM jobs in `Implementation/jobs/`                               |
| **Activate**   | `source /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/.venv/bin/activate` |
| **Run app**    | `streamlit run app.py` (from `Implementation/`)                                              |
| **Data**       | yfinance, NSE `.NS` tickers, daily `1D` frequency, 2016–2026                                |

---

## Architecture Map

```
KnowledgeGraph/                ← Context cache for agents (this folder)
├── KnowledgeGraph.md          ← You are here
├── TokensInstructions.md      ← Universal session efficiency rules
├── TokensGraphing.md          ← Universal graph maintenance protocol
└── graph/                     ← Structured JSON cache files
    ├── modules.json           ← All core/ modules, selectors, signal models
    ├── experiments.json       ← All experiment scripts + results summary
    ├── build-and-config.json  ← Dependencies, cluster config, env setup
    ├── decisions-and-gotchas.json ← Known bugs, key decisions, do-not-load list
    └── token-cost-map.json    ← Token cost for every significant file

Literature-Review/             ← Paper reproductions & verification
├── README.md                  ← Comprehensive catalog of all papers
└── yyyy-*TypeOfModel-Paper*/  ← 11 paper implementations (1987-2021)
    ├── paper.pdf              ← Original paper
    ├── reproduction.py        ← Standalone reproduction code
    ├── results.json           ← Our results vs claimed results
    └── README.md              ← Paper-specific documentation

Implementation/                ← All source code lives here
├── CLAUDE.md                  ← Agent-oriented architecture overview
├── app.py                     ← Streamlit UI + orchestration (~1,450 lines, 60KB)
├── requirements.txt           ← Python deps
├── core/                      ← UI-agnostic library
│   ├── data.py                ← DataConfig, yfinance fetching
│   ├── selectors.py           ← Re-exports all selectors
│   ├── selectors_base.py      ← Base class for selectors
│   ├── selectors_statistical.py ← Correlation, Distance, Cointegration, Combined
│   ├── selectors_ml.py        ← XGBoost, LSTM, Transformer, GNN (39KB — LARGE)
│   ├── entry.py               ← Stage 2 signal models (ZScore, OU, Kalman, ML)
│   ├── ensemble.py            ← Weighted score/signal combination
│   ├── backtest.py            ← Vectorized backtester, BacktestConfig, IndianCosts
│   ├── reports.py             ← ReportManager (Repository pattern)
│   └── predictions.py        ← Real-time prediction engine (21KB)
├── experiments/               ← Reproducible research scripts
│   ├── config.py              ← Universe, DEFAULT_MIN_HOLD=30, RANDOM_SEED=42
│   ├── freq_comparison.py     ← E1: daily vs hourly
│   ├── hold_period_sweep.py   ← E2: min hold optimization
│   ├── ablation.py            ← E3: per-selector attribution
│   ├── walk_forward.py        ← E4: headline WFV experiment (28KB — key file)
│   ├── benchmark_comparison.py ← E5: vs Nifty 50/Bank/IT
│   ├── significance_tests.py  ← E6: bootstrap CI + Bonferroni
│   └── results/               ← Timestamped JSON outputs (never load raw)
├── jobs/                      ← SLURM job scripts for cluster
├── logs/                      ← Cluster run output logs
├── reports/                   ← Streamlit UI session backtest outputs
├── data_cache/                ← Cached yfinance downloads
├── Plan.md                    ← Full thesis roadmap + experiment specs (source of truth)
├── Research.md                ← Full experiment log + all results (source of truth)
├── Currently_Doing.md         ← Cluster execution guide
├── scripts.md                 ← All run commands (local + cluster)
└── DevNotes.md                ← Scratch notes (low signal)
```

---

## Key Decisions (Quick Reference)

> Full detail in `graph/decisions-and-gotchas.json`

| Decision                                | Implication                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| Daily (1D) data for main experiments    | E1: Gross SR 1.14 (daily) vs 0.49 (hourly); daily is the right choice          |
| `DEFAULT_MIN_HOLD = 30` trading days   | E2 sweep result; locked in `experiments/config.py`; all experiments use it      |
| 35 NSE large-cap stocks, 8 sectors     | M&M.NS replaces TATAMOTORS.NS; universe fixed before any backtest               |
| OU-only S2 is the best signal model    | E3: OU_only Net SR +0.359; MLSignal overfit (Net SR -0.401)                     |
| Equal-weight ensemble underperforms    | E3 finding: ensemble Net SR -0.189 < Cointegration_only +0.119 (stat_only)     |
| `IndianCosts` for realistic NSE costs  | brokerage 3bp + STT 10bp + GST 18% + stamp 1bp + slippage 2bp/leg             |
| Expanding-window WFV (6 folds)        | Train 2016→year-1, Test each year 2020–2025; OOS evaluation only              |
| MLSelector label mis-specification     | MLSelector weight = 0.0 in E7; momentum label ≠ mean-reversion quality         |
| Transformer Lambda+GPU bug fixed       | `_PositionalEncodingLayer` in `selectors_ml.py` fixed 2026-04-06              |
| `RANDOM_SEED = 42` fixed everywhere    | Required for academic reproducibility                                           |

---

## Active State

> **Update this section after every significant session.**

- **Status:** Round 3 critique complete — revision phase
- **Headline result:** NSE Nifty 50 statistical-only ZScore: Net SR **+0.752** (95% CI [+0.422, +1.082], p=0.036) — only statistically significant result
- **Latest:** Round 3 peer review complete (2026-06-09)
  - 26 open issues: 5 FATAL, 9 MAJOR, 7 MODERATE, 5 MINOR
  - JFM acceptance: 35%, QF: 50%
  - Recommendation: Reject with Invitation to Resubmit
  - Key action: Invert thesis framing — Nifty 50 stat-only is the headline; ML ensemble is exploratory negative finding
- **Current focus:** Address FATAL issues (see Prioritised Fix List above)

---

## Graph Update Rules

- Update the `Active State` section above after every session.
- Update the relevant `graph/*.json` file after: new modules, new experiment scripts, dependency changes, new results, key decisions.
- Run the **Graph Health Check** (in `TokensGraphing.md`) at the start of any new sprint or after a long break.
- **Target size for this file:** ≤200 lines. Move detail into `graph/*.json` if it grows beyond that.
