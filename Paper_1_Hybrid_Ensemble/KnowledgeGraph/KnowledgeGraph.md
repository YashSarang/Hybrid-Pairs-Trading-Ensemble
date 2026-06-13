# Hybrid Pairs Trading — Paper 1 Knowledge Graph
*Last updated: 2026-06-12 (docs + flow restructuring synced + robustness results added) | Universe: 89 NSE Nifty100 | Costs: 16.28 bps | CPU-only ML*

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
| E1 | Freq Comparison (stat_only baseline WFV) | COMPLETE job 8737 | mean net SR=0.343 ±1.021 (84-ticker data) |
| E2 | Hold Period Sweep | LOCKED (old result) | min_hold=30 days optimal |
| E3 | Ablation (stat_only + stat_ml) | COMPLETE job 8734 | stat_only: Distance SR=0.829 (best single), Ensemble SR=0.256; stat_ml ensemble SR=-0.311 |
| E4 | Walk-Forward Validation | COMPLETE jobs 8693-8699 | stat_only SR 0.480 / stat_ml SR 0.453 / full SR 0.519 ±0.061 |
| E4-Robustness | 35-ticker Nifty50 Matched WFV | COMPLETE (Kalpana job 9442) | stat_only (OU): mean net SR=0.920, full OOS Net SR=0.773; stat_ml (OU): mean net SR=0.954, full OOS Net SR=0.792 |
| E5 | Benchmark vs Nifty50 | COMPLETE job 8699 | Strategy SR 0.550 vs Nifty50 SR 0.720 |
| E6 | Significance Tests (all 3 modes) | COMPLETE job 8735 | None significant at 5%; full best: p=0.069 boot, p=0.076 NW |
| E4 Grid | Exhaustive Weight Space Search | COMPLETE | E4.S: ML SR=0.610 (best single); E4.W2: Corr+Coint SR=0.726 (best pair); DM tests p>0.45 (not significant) |
| E8 | RL Signal | BLOCKED — gymnasium not on Kalpana | — |

---

## DOCUMENTATION STATUS (LATEST)

- `Flow.md` is now restructured as the canonical narrative with:
  - glossary of short forms/jargon at top,
  - explicit Sharpe Ratio terminology and computed Sharpe deltas where derivable,
  - evidence-classified key design decisions (literature vs experiment vs engineering),
  - causal experiment sequence (why each experiment led to the next),
  - E4 data-availability caveat for missing per-mode fields in committed artifacts,
  - **Future Work Scope → Immediate — Experimentation** block,
  - new standalone section: **Ensemble Optimisation and Cross-Universe Consistency Plan**.
- `Implementation/reports/chapter2_literature_review.md` has matching top glossary/jargon definitions to keep terminology consistent with `Flow.md`.

---

## KEY RESULTS (89-ticker, 16.28 bps, CPU-only)

**E4 Walk-Forward (6 folds, 2018-2024, 89-ticker canonical):**
- stat_only + ou_only: Net SR **0.480** (p_boot=0.086, 95% CI [-0.209, +1.154]), CAGR 3.30%, MaxDD 12.72%, Trades 473 (cost drag: mean 0.56 pp)
- stat_ml + ou_only: Net SR **0.438** (p_boot=0.089, 95% CI [-0.194, +1.081]), CAGR 3.23%, MaxDD 10.10%, Trades 476 (cost drag: mean 0.57 pp)
- full hybrid + ou_only: Net SR **0.520** (p_boot=0.069, 95% CI [-0.171, +1.213]), CAGR 3.72%, MaxDD 11.75%, Trades 467 (cost drag: mean 0.56 pp)
- s2=all (stat_only): Net SR 0.340, MaxDD 19.32% — OU-only clearly superior

**E4-Robustness Walk-Forward (6 folds, 2018-2024, 32 of 35 Nifty50 matched-universe tickers):**
- stat_only + ou_only: mean Net SR **0.920 ±1.022** | Full OOS Net SR **0.773**, Net Ret 5.51%, Net MaxDD 10.81%, Trades 458 (cost drag: mean 0.53 pp)
- stat_only + no_ml: mean Net SR **0.484 ±1.074** | Full OOS Net SR **0.312**, Net Ret 2.95%, Net MaxDD 23.48%, Trades 1056 (cost drag: mean 1.27 pp)
- stat_ml + ou_only: mean Net SR **0.954 ±1.349** | Full OOS Net SR **0.792**, Net Ret 5.56%, Net MaxDD 13.86%, Trades 450 (cost drag: mean 0.54 pp)
- *Note:* Missing 3 tickers due to yfinance/parquet availability: NTPC, TATAMOTORS, GRASIM. 
- *Interpretation:* The Fold 5 (2022) drawdown is negative in both universes (-1.504 Net SR in 35t, -0.707 Net SR in 89t), confirming this is a market-wide macro phenomenon, not a universe-specific artifact. Overall performance is higher in the Nifty50 subset, corroborating Paper 2's universe quality thesis.

**E4 Fold breakdown (stat_only ou_only):**
Fold2018: SR 0.021 | Fold2019: 0.462 | Fold2020: 0.572 | Fold2021: 1.972 | Fold2022: -0.707 | Fold2023-24: 0.564
Aggregate: Net SR 0.481 ±0.802 | Full OOS: SR 0.480, CAGR 3.30%, MaxDD 12.72%

**E3 Ablation (stat_only mode):**
- Correlation_only: SR 0.160 | Distance_only: SR **0.829** (best single, std=1.063) | Cointegration_only: SR -0.088 | Combined_only: SR -0.223
- S1_Ensemble (stat_only): SR 0.256 — **below Distance_only**
- S2 signal: OU_only SR 0.283 (best) | ZScore SR -0.275 | Kalman SR -0.257 | ML SR -0.405

**E3 Ablation (stat_ml mode):**
- ML_only selector: SR 0.217 | S1_Ensemble (stat+ML): SR **-0.311** — adding ML hurts
- S2 signal: OU_only SR 0.060 (drops from 0.283 with different pair selection)

**E5 Benchmark:**
- Strategy SR 0.550 vs Nifty50 SR 0.720
- Strategy MaxDD 12.28% vs Nifty50 MaxDD 38.44%
- Strategy CAGR 3.76% vs Nifty50 CAGR 12.84%
- Verdict: underperforms on returns, materially lower drawdown (market-neutral)

**E6 Significance (all 3 modes, ou_only — NONE significant at 5%):**
- stat_only: net SR=0.468 CI=[-0.209,1.154] p_boot=0.086 NW t=1.300 p=0.097 (sig at 10%)
- stat_ml:   net SR=0.438 CI=[-0.194,1.081] p_boot=0.089 NW t=1.243 p=0.107 (NOT sig at 10%)
- full:      net SR=0.520 CI=[-0.171,1.213] p_boot=0.069 NW t=1.434 p=0.076 (sig at 10%)

**E4 Grid Exhaustive Weight Space Search (89-ticker canonical data, 16.28 bps, June 2026):**
- Standalone (E4.S): ML (XGBoost) Net SR **0.610**, Distance Net SR **0.444**, Cointegration Net SR **0.167**, Combined Net SR **0.109**, Transformer Net SR **0.041**, GNN Net SR **-0.121**, Correlation Net SR **-0.234**, LSTM Net SR **-1.034**
- Pairwise (E4.W2): Best pair is **Corr+Coint** Net SR **0.726** (CAGR 5.47%, MaxDD 8.37%), representing a powerful synergistic dual-filter. Next best is Coint+ML Net SR **0.590**.
- Significance: Pairwise DM tests ($h=30$) confirm that these weight space improvements are statistically non-significant (Corr+Coint vs stat_only p=0.649, ML vs stat_only p=0.479), supporting the parsimony principle.

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

1. **Write paper** — all experiments complete; focus on chapter/table integration
2. **CPU full-mode non-determinism** — document clearly; 6 runs gave SR 0.437–0.618 range
3. **E7 data drift hardening** — keep `fetch_paper1_data.py` removed from E7 path; enforce 89-ticker parquet before reruns
4. **Bootstrap CIs for E4 headline numbers** — add per-mode CI panel (not only aggregate significance outputs)
5. **Chapter rewrites** — Ch4 §4.5/§4.7 to include final ablation + weighted-ensemble evidence
6. **Ch5 cost fix** — Section 5.1.1 uses 6 bps brokerage (legacy); align to 0 bps (16.28 bps total)
7. **Abstract final numbers** — harmonize with latest canonical table and caveats
8. **Cross-universe robustness pack (Immediate — Experimentation)** — run matched 35-ticker E4/E5/E6 and report as robustness appendix (89-ticker remains primary)

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
