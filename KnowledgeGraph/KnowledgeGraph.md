# Knowledge Graph — Hybrid Pairs Trading Ensemble

> **For AI coding agents:** Read this file at the start of every session. It is the cheapest way (~800 tokens) to understand the full codebase. Do not read raw source files until you know you need them.

**Last updated:** 2026-05-06

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

- **Status:** Active research — Phase E (E7 Weighted Ensemble) is NEXT
- **Headline result:** E4 (stat_only + ou_only): Full-OOS Net Sharpe **+0.359** (Gross 0.627)
- **Current focus:** E7 Weighted Ensemble WFV — `experiments/weighted_ensemble.py` not yet built
- **Known blockers:** Transformer_only single-fold re-run pending (validate Lambda bug fix)
- **Last significant change:** E4 full-mode completed (+0.067 Net SR); E3 full-mode completed (LSTM_only best: +0.305); Transformer bug fixed (2026-04-06)

---

## Graph Update Rules

- Update the `Active State` section above after every session.
- Update the relevant `graph/*.json` file after: new modules, new experiment scripts, dependency changes, new results, key decisions.
- Run the **Graph Health Check** (in `TokensGraphing.md`) at the start of any new sprint or after a long break.
- **Target size for this file:** ≤200 lines. Move detail into `graph/*.json` if it grows beyond that.
