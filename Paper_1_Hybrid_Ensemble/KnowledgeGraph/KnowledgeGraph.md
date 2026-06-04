# Hybrid Pairs Trading — Paper 1 Knowledge Graph
*Last updated: 2026-06-05 | Universe: 89 NSE Nifty100 | Costs: 16.28 bps | CPU-only ML*

---

## HOW TO USE THIS FILE
1. Read this file (~1000 tokens) at the start of every session
2. Load only the specific graph/*.json node relevant to your task (see table below)
3. Never speculatively load large source files — check token-cost-map.json first
4. If idle >5 min, compact the session before resuming (cache expiry = 10-15x cost spike)

| Task | Load |
|------|------|
| Running/checking experiments | graph/experiments.json |
| Module/selector/signal changes | graph/modules.json |
| Env/cluster/SLURM changes | graph/build-and-config.json |
| Known bugs / tricky changes | graph/decisions-and-gotchas.json |
| Token budget planning | graph/token-cost-map.json |

---

## PROJECT IDENTITY
**Paper 1:** "Hybrid Ensemble Pairs Trading: Do ML Selectors Outperform Statistical Baselines?"
**Paper 2:** Significance/contribution paper — SUBMISSION READY (all 29 critiques resolved 2026-06-04)
**Thesis:** IIT Bombay PhD, Yash Sarang
**Repo:** `D:/Code/Hybrid-Pairs-Trading-Ensemble/` (local) | `~/Hybrid-Pairs-Trading-Ensemble` (Kalpana)

---

## CURRENT EXPERIMENT STATUS

| Exp | Name | Status | Key Result |
|-----|------|--------|-----------|
| E1 | Freq Comparison (daily vs hourly) | SUBMITTED job 8737 | PENDING |
| E2 | Hold Period Sweep | NOT RERUN (old result locked) | min_hold=30 days |
| E3 | Ablation (3 modes) | SUBMITTED job 8734 (prev 8704 CANCELLED: wrong universe) | PENDING |
| E4 | Walk-Forward Validation | COMPLETE jobs 8693-8699 | stat_only SR 0.480 / full SR 0.653 |
| E5 | Benchmark vs Nifty50 | COMPLETE job 8699 | Strategy SR 0.550 vs Nifty50 SR 0.720 |
| E6 | Significance Tests | SUBMITTED job 8735 (all 3 modes) | stat_only p=0.086 (prior run); stat_ml/full PENDING |
| E7 | Weighted Ensemble | SUBMITTED job 8736 (3 configs) | PENDING |
| E8 | RL Signal | BLOCKED — gymnasium not on Kalpana | — |

---

## KEY RESULTS (89-ticker, 16.28 bps, CPU-only)

**E4 Walk-Forward (6 folds, 2018-2024):**
- stat_only + ou_only: Net SR **0.480**, CAGR 3.30%, MaxDD 12.72%, 473 trades
- stat_ml + ou_only: Net SR **0.431**
- full hybrid (best): Net SR **0.653**, CAGR 4.51%, MaxDD 10.43%
- s2=all (stat_only): Net SR 0.312, MaxDD 19.32% — OU-only clearly superior

**E4 Fold breakdown (stat_only ou_only):**
Fold2018: SR 0.021 | Fold2019: 0.462 | Fold2020: 0.572 | Fold2021: 1.972 | Fold2022: -0.707 | Fold2023-24: 0.564
Aggregate: Net SR 0.481 ±0.802 | Full OOS: SR 0.480, CAGR 3.30%, MaxDD 12.72%

**E5 Benchmark:**
- Strategy SR 0.550 vs Nifty50 SR 0.720
- Strategy MaxDD 12.28% vs Nifty50 MaxDD 38.44%
- Strategy CAGR 3.76% vs Nifty50 CAGR 12.84%
- Verdict: underperforms on returns, materially lower drawdown (market-neutral)

**E6 Significance (stat_only ou_only, 1725 obs):**
- Bootstrap 95% CI: **[-0.209, +1.154]** | p(SR≤0) = **0.086**
- Newey-West: t=1.300, p=0.097 one-sided, lags=8
- **Marginally significant at 10%, NOT conventional 5%**
- NOTE: Paper 2 headline uses NSE Nifty50 ZScore SR +0.752 (p=0.036) — that is primary

---

## SETUP

| Parameter | Value |
|-----------|-------|
| Universe | 89 NSE Nifty100 tickers |
| Dropped | TATAMOTORS, BERGERPAINTS, VODAFONEIDEA, LTIM, ADANITRANS |
| Date range | 2015-01-01 to 2024-12-31 |
| WFV folds | 6, expanding: test years 2018/2019/2020/2021/2022/2023-24 |
| Transaction costs | 16.28 bps round-trip (D12) |
| ML mode | CPU-only, CUDA_VISIBLE_DEVICES='', TF_DETERMINISTIC_OPS=1, seed=42 (D13) |
| min_hold | 30 trading days (D2/D17) |
| Parquet cache | Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet |

---

## FILE MAP

```
Paper_1_Hybrid_Ensemble/
  Implementation/
    core/                     Stage 1 selectors + Stage 2 signals + backtest engine
      backtest.py             IndianCosts (16.28 bps), backtest_pairs, BUG-08 fixed
      selectors_statistical.py  Correlation, Distance, Cointegration, Combined
      selectors_ml.py         LSTM, Transformer, GNN, MLSelector (LARGE — 10k tokens)
      entry.py                OUThreshold, ZScore, Kalman, MLSignal
      entry_rl.py             RL signal — class-guarded (gymnasium missing)
      data.py                 DataConfig, Parquet cache, yfinance
      ensemble.py             Weighted combination Stage1/Stage2
      reports.py              ReportManager, BenchmarkComparison
      predictions.py          Live prediction (LARGE — 5.5k tokens)
    experiments/
      config.py               89 tickers, dates, seed, min_hold
      walk_forward.py         E4 WFV script (LARGE — 7k tokens)
      ablation.py             E3 ablation (LARGE — 6.5k tokens)
      significance_tests.py   E6
      benchmark_comparison.py E5
      freq_comparison.py      E1 (not rerun on 89-ticker universe)
      slurm/                  ACTIVE SLURM scripts
      results/                Current results from 89-ticker runs
      results/archive_old_universe/  OLD 35-ticker results — do not cite
      data/nse_nifty100/      Parquet cache
    ARCHIVED_SLURM/           OLD job scripts — do not use
    reports/
      abstract.md             NEEDS REVISION (old 35-ticker numbers)
      chapter1_introduction.md  NEEDS REVISION
      chapter2_literature_review.md  NEEDS REVISION (numbers only)
      chapter3_methodology.md  NEEDS REVISION (universe table, fold table)
      chapter4_results.md     NEEDS REVISION (all tables)
      chapter5_discussion.md  NEEDS REVISION (cost model + numbers)
    Decisions.md              D1-D22 full decision log
    Plan.md                   Full research roadmap
  KnowledgeGraph/
    KnowledgeGraph.md         THIS FILE
    graph/
      experiments.json        E1-E8 status + results
      modules.json            All modules, selectors, signals
      build-and-config.json   Env, cluster, run commands
      decisions-and-gotchas.json  D1-D22, BUG-01 to BUG-10, checklist
      token-cost-map.json     File sizes and load guidance
    TokensInstructions.md     Universal token efficiency guide
    TokensGraphing.md         Graph query patterns
    Prompt_to_initialise.md   Session start prompt
  Literature-Review/
    README.md                 12 papers covered — needs update with 89-ticker numbers
    2010-PCA-OU-Avellaneda-StatArb/  Negative result — valid, architecture-independent
  Documentation/
    DATA_APPENDIX.md          921-line data documentation
    TRADING_COST_UPDATE_FINAL_REPORT.md  22.9->16.28 bps correction history
    PROJECT_NARRATIVE.md      High-level project story
  THESIS_COMPLETION_PLAN.md   Completion roadmap
  STREAMLIT_ENHANCEMENT_PLAN.md  Literature Review pages — NOT YET IMPLEMENTED
  Paper_1_Plan.md             Paper 1 outline
```

---

## PENDING REVISIONS (ordered by priority)

1. **Jobs running on Kalpana** — E3(8734), E6(8735), E7(8736), E1(8737). Pull results when complete.
   - E3 ~8h (3 modes: stat_only/stat_ml/full ablation)
   - E6 ~6h (3 modes significance tests)
   - E7 ~8h (3 weighted configs: E7-A stat_ml weighted, E7-B full weighted, E7-C stat_only+ou_only)
   - E1 ~4h (freq comparison daily vs hourly)
2. **After results arrive** — update Ch4 §4.5/§4.7 [[PLACEHOLDER]] with E3/E7 numbers
3. **Chapter rewrites** — all 5 chapters + abstract need number updates once E3/E7 results in
4. **Ch5 cost fix** — Section 5.1.1 uses 6 bps brokerage (legacy); align to 0 bps (16.28 bps total)
5. **Literature-Review README** — update ablation numbers once E3 done
6. **Streamlit LR pages** — implement STREAMLIT_ENHANCEMENT_PLAN.md (low priority)

---

## WORKFLOW RULES

**Code changes:**
NEVER edit code directly on Kalpana. Always:
1. Edit locally
2. `git push origin main`
3. `ssh kalpana 'cd ~/Hybrid-Pairs-Trading-Ensemble && git pull origin main'`
4. `sbatch experiments/slurm/<script>.sh`

**Results pull:**
On Kalpana: `git add -A && git commit -m "results: ..." && git push origin main`
Locally: `git pull origin main`

**SLURM:**
- Partition: `cn3_anandi`, Account: `cminds_anandi`, QOS: `anandi`
- Active scripts: `Implementation/experiments/slurm/`
- Archived (do not use): `Implementation/ARCHIVED_SLURM/`
