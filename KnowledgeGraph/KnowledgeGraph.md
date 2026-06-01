# Knowledge Graph — Hybrid Pairs Trading Ensemble

> **For AI coding agents:** Read this file at the start of every session. It is the cheapest way (~800 tokens) to understand the full codebase. Do not read raw source files until you know you need them.

**Last updated:** 2026-06-01 21:20 IST

---

## 🚨 REPRODUCIBILITY CRISIS — CURRENT STATUS (2026-06-01)

**Control experiment COMPLETE (Job 8459).** Results below determine thesis narrative.

### NSE Nifty 50 + Rolling Windows (4 statistical selectors, no ML)
| Signal | Avg Net Sharpe | Std | Verdict |
|--------|---------------|-----|---------|
| ZScore | **+0.752 ± 0.361** | high variance | **SCENARIO A threshold MET (≥0.70)** |
| OU     | +0.147 ± 0.255 | 3 of 4 folds: 0 trades | OU underperforms on Nifty 50 |

ZScore fold-by-fold: +1.127, +0.218, +0.627, +1.036 (126 trades total)

### 2×2 Comparison Matrix
| Universe | Method | Signal | Sharpe | Status |
|----------|--------|--------|--------|--------|
| Nifty 100 | Expanding | ZScore | -0.409 | ✅ Chapter 3 |
| Nifty 100 | Rolling | ZScore | +0.052 | ✅ Chapter 3 |
| **Nifty 50 (CTRL)** | **Rolling** | **ZScore** | **+0.752** | ✅ **NEW — June 1** |
| Nifty 50 (CTRL) | Rolling | OU | +0.147 | ✅ NEW — June 1 |
| India Multi-Mkt | Rolling | ZScore | +0.840 | ⚠️ Best of 3 runs |
| India Multi-Mkt | Rolling | ZScore | +0.284 | ⚠️ Mean of 3 runs |

**Universe uplift (Nifty50 vs 100, ZScore rolling): +0.700**
**Remaining geographic effect (vs mean): −0.468 (i.e. India multi-market mean < Nifty 50 control)**

### ⚠️ CRITICAL CAVEAT
Control experiment used **4 statistical selectors only** (no ML/LSTM/GNN).
Original India multi-market result used **8 selectors including LSTM, GNN, Transformer**.
Fair comparison requires running NSE Nifty 50 with full 8-selector ensemble.
The ML selectors hang on the cluster due to TF import on CUDA-less node.

### Scenario Determination
**SCENARIO A: Universe Quality Dominates — CONFIRMED ON BOTH METHODS (June 1, 2026)**
- NSE Nifty 50 Rolling ZScore:   **+0.752 ± 0.361** (threshold ≥ 0.70: ✅)
- NSE Nifty 50 Expanding ZScore: **+1.064 ± 0.502** (threshold ≥ 0.70: ✅)
- Universe uplift (rolling):   +0.700 Sharpe vs Nifty 100
- Universe uplift (expanding): +1.473 Sharpe vs Nifty 100
- Method effect within Nifty 50: +0.312 (expanding > rolling)
- Geographic effect (multi-mkt mean vs control): **−0.468** — control beats multi-mkt mean
- New thesis narrative: **"Universe Selection Dominates Methodology and Geography"**
- Transparency report written: `TRANSPARENCY_REPORT.md`
- All results: `results/nse_nifty50/` (rolling) and `results/nse_nifty50_expanding/` (expanding)

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

- **Status:** Thesis write-up phase + Code verification complete
- **Headline result:** E7 Config C (LSTM+Corr): Net Return **+17.66%**, Net SR **+0.510** | E8 RL (PPO): Underperforms statistical baseline due to expected data starvation.
- **Latest:** ERROR RESOLUTION COMPLETE (2026-05-26)
  - ✅ Fixed 2 bare except clauses in app.py  
  - ✅ Installed missing dependencies (plotly, joblib)
  - ✅ Created comprehensive end-to-end test (test_complete_workflow.py)
  - ✅ Verified full workflow: data → pair selection → signal generation → backtest → results
  - ✅ All systems operational and production-ready
- **Previous:** Literature-Review/ system established (2026-05-26)
  - Comprehensive catalog of 11 major pairs trading papers (1987-2021)
  - PCA-OU reproduction complete: METHOD FAILS ON NSE (0% tradeable stocks)
  - High-value negative result strengthens thesis contribution
- **Current focus:** 
  - Thesis writing (Abstract ✅, Chapter 6 in progress)
  - Literature review reproductions (1/11 complete: PCA-OU)
  - Codebase production-ready for thesis submission
- **Known blockers:** None
- **Last significant change:** 
  - 2026-05-26: Created Literature-Review/ folder with 11 paper implementations; comprehensive review document with reproduction status tracking
  - 2026-05-06: Audited codebase, fixed double-charging transaction cost bug (BUG-08); regenerated all results

---

## Graph Update Rules

- Update the `Active State` section above after every session.
- Update the relevant `graph/*.json` file after: new modules, new experiment scripts, dependency changes, new results, key decisions.
- Run the **Graph Health Check** (in `TokensGraphing.md`) at the start of any new sprint or after a long break.
- **Target size for this file:** ≤200 lines. Move detail into `graph/*.json` if it grows beyond that.
