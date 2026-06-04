# Decisions Log

Every non-trivial design or methodological decision made in this project is recorded here with full reasoning. The goal is that any reader — including a thesis examiner, a collaborator, or a future AI assistant — can understand *why* the system is built the way it is, not just *what* it does.

Format: each entry has a **Decision**, **Context** (what problem it solves), **Reasoning** (why this choice over alternatives), and **Trade-offs** (what we gave up).

---

## D1 — Daily (1D) data as the primary frequency

**Decision:** All main experiments use end-of-day daily prices. Hourly and sub-hourly data are not used for the main strategy.

**Context:** Data frequency is a free parameter. Higher frequency means more observations, which could improve model training and signal resolution.

**Reasoning (empirical, from E1):**
- Gross Sharpe fell 57% from daily (1.14) to hourly (0.49) on the same universe and date window.
- Spread Hurst exponent rose from 0.190 (daily) to 0.251 (hourly), confirming that hourly spreads contain more microstructure noise and are less mean-reverting.
- At hourly frequency, 904 trades/year at the then-estimated round-trip cost (note: cost model was later corrected to 16.3 bps in May 2026) caused the strategy to lose more than its entire capital (Net Max DD = 214%).
- The hourly selector identified qualitatively different pairs (e.g., MARUTI-BRITANNIA, a cross-sector pair with no obvious economic relationship) compared to daily (TCS-WIPRO, TCS-INFY — well-established IT sector co-integrations). This suggests hourly selectors are picking up noise-driven statistical artifacts rather than genuine economic co-movement.

**Reasoning (theoretical):**
- NSE pairs trading is based on economic co-integration between firms. Economic relationships manifest at daily or lower frequency; intraday prices are dominated by order flow, bid-ask bounce, and inventory effects (Roll 1984, Hasbrouck 1993).
- The Ornstein-Uhlenbeck mean-reversion model assumes a continuous-time stochastic process. Discretising at high frequency introduces microstructure noise that violates the OU assumptions and inflates estimated reversion speed (Elliott et al. 2005).
- Sub-hourly data (5-min, 1-min) cannot be tested at all due to yfinance's 60-day history limit — insufficient for any meaningful backtest or model training.

**Trade-offs:**
- We sacrifice intraday signal resolution. A daily signal can only enter or exit once per day, regardless of intraday spread dynamics.
- yfinance daily data may have Adjusted Close issues around corporate actions. These are accepted as a known data quality limitation.

---

## D2 — 35-stock diverse NSE universe across 8 sectors

**Decision:** The universe is fixed at 35 Nifty-100 large-cap stocks covering Banking, IT, Auto, FMCG, Pharma, Energy, Metals, and Cement. Defined in `experiments/config.py`.

**Context:** Universe selection affects both pair quality and result credibility. Too narrow a universe (e.g., all banking stocks) would trivially find pairs but fail to test the ensemble's generalization. Too broad a universe would introduce illiquid stocks and combinatorial explosion.

**Reasoning:**
- 35 stocks yields 595 candidate pairs (35C2). After coverage filtering this is typically ~561 pairs — tractable for all 8 selectors without parallelization.
- All stocks are Nifty 100 constituents — sufficiently liquid at daily frequency to avoid ADV concerns at our capital scale (INR 10 lakh / 1 lakh per pair).
- 8 sectors ensures same-sector pairs (natural economic co-integration: HDFCBANK-ICICIBANK, TCS-INFY) AND cross-sector pairs (which test whether the ML/DL selectors find non-obvious relationships).
- Restricting to Nifty 100 avoids small-cap stocks where yfinance data quality degrades and market impact at our notional size would be unrealistic.

**Trade-offs:**
- 35 is smaller than typical academic studies (which use 500+ stocks) but necessary given CPU-only compute and the 8-selector pipeline. The paper will note this.
- M&M.NS replaced TATAMOTORS.NS due to a persistent yfinance `YFTzMissingError`. Both are auto-sector large-caps; the substitution does not materially change the universe character.

---

## D3 — Universe fixed before any backtest or experiment

**Decision:** The universe is locked in `experiments/config.py` and does not change between experiments. It was defined before any backtest results were seen.

**Context:** If the universe were adjusted after seeing results (e.g., adding stocks that performed well), it would introduce look-ahead bias and invalidate all findings.

**Reasoning:**
- Academic credibility requires the universe to be pre-specified. Changing the universe post-hoc to improve results is a form of data snooping.
- The universe selection criteria (Nifty 100, 8 sectors, liquid) are publicly observable and economically motivated — they do not depend on the backtest results.
- TATAMOTORS was replaced with M&M for a technical reason (yfinance data unavailability), not a performance reason. This is documented here to be transparent.

**Trade-offs:** None significant. The cost of a slightly sub-optimal universe is outweighed by the benefit of methodological cleanliness.

---

## D4 — Equal ensemble weights for all ablation and comparison experiments

**Decision:** For experiments E1 (frequency comparison), E2 (hold period sweep), and E3 (ablation), all active selectors and all signal models receive equal weight (1.0 each, normalized internally).

**Context:** The ensemble weighting is a free parameter. Optimized weights could improve performance, but if we used optimized weights in the comparison, we would be comparing a tuned ensemble against untuned individual models — an unfair comparison.

**Reasoning:**
- Equal weights isolate the structural benefit of combining multiple models from the benefit of weight optimization.
- Weight optimization on the training set would introduce an additional free parameter that complicates the ablation story. The paper's claim is "multiple diverse models > one model" — this is best tested with equal weights.
- Equal weights are the standard in ensemble learning literature when assessing the benefit of combining vs. using a single model.

**Trade-offs:**
- The ensemble's full potential is not demonstrated under equal weights. A Bayesian weight optimization or a validation-set weight search could improve net Sharpe further. This is explicitly noted in the paper as future work.

---

## D5 — Minimum hold period as a fixed methodological parameter, not a tuned hyperparameter

**Decision:** The optimal `min_hold_bars` value found in E2 (hold period sweep) is applied as a fixed default in `BacktestConfig` for all subsequent experiments. It is NOT tuned per-fold in the walk-forward validation.

**Context:** Any parameter tuned on the full dataset and then applied to the same data introduces look-ahead bias. The minimum hold period could be treated as a hyperparameter and optimized per fold, but this adds complexity and overfitting risk.

**Reasoning:**
- The minimum hold period is a structural/methodological constraint, not a signal model parameter. It is analogous to a "no day-trading" rule imposed by the strategy design, not something that should vary over time.
- Sweeping hold periods on the full 10-year dataset (E2) and then applying the optimal value to walk-forward folds is acceptable because:
  (a) The sweep is done on the FULL dataset before WFV begins, not on test data.
  (b) The hold period affects cost structure, not signal direction — it cannot "learn" the future.
  (c) The selected hold period is the one that makes the strategy economically viable; without it, the strategy is not worth evaluating at all.
- This approach is analogous to choosing a daily rebalancing frequency (vs weekly): it is a design choice made before any fold-level analysis.

**Trade-offs:**
- If market conditions change and the optimal hold period shifts (e.g., NSE transaction costs drop significantly), the fixed hold period would be suboptimal. This is a known limitation, noted in the thesis.

---

## D6 — `_apply_min_hold` blocks both exits and reversals (not only reversals)

**Decision:** The minimum hold period filter (`_apply_min_hold` in `core/backtest.py`) prevents both exits (non-zero -> 0) AND reversals (non-zero -> opposite) during the hold window.

**Context:** A more lenient alternative would block only reversals and allow exits at any time.

**Reasoning (from E1 data):**
- Signal reversal rate in E1 was 40.4% (daily). Examination of the trades log shows many patterns of the form +1 -> 0 -> +1 within 2–3 bars — rapid entry, exit, re-entry. These are NOT full reversals (+1 -> -1) but are equally costly (2 round-trips in 3 bars = approximately 33 bps cost using 16.3 bps per round-trip on a signal that barely moved).
- Blocking only reversals would leave this rapid-exit pattern unconstrained and would not substantially reduce turnover.
- Blocking all changes for min_hold bars eliminates both patterns with a single parameter.

**Exception:** The soft stop-loss fires AFTER `_apply_min_hold` and can still unconditionally force the signal to zero (via `breach_persist` override). Emergency exits are therefore always honoured.

**Trade-offs:**
- Forcing exit delays could in theory hold a losing position longer. In practice, the soft stop provides the safety valve for extreme z-score breaches. The min hold is only a "patience" constraint for normal signal noise.

---

## D7 — Expanding window walk-forward validation (not rolling)

**Decision:** Walk-forward validation (E4) uses an expanding training window: train on 2016–2019, test 2020; then train 2016–2020, test 2021; etc.

**Context:** Two alternatives exist: rolling window (fixed training length) and expanding window (training grows with time).

**Reasoning:**
- Expanding window is standard in financial econometrics. With expanding windows, the model benefits from all available history — matching how a practitioner would actually deploy the strategy.
- Rolling windows (fixed train size) would discard early data in later folds, wasting signal. For NSE data from 2016, the early period (demonetisation, market recovery) contains regime information not present in later data.
- Expanding windows produce conservative estimates: later folds have more training data, which generally improves model quality. If performance degrades in later folds despite more data, it is a credible sign of regime change or market adaptation.

**Trade-offs:**
- Expanding windows mean later folds have more training data than earlier folds, creating heterogeneous fold conditions. A rolling window would give uniform conditions. We accept this trade-off for the benefit of using all historical data.

---

## D8 — Selectors run once per WFV fold; signal models run per pair per fold

**Decision:** In walk-forward validation, both Stage 1 selectors (pair selection) AND Stage 2 signal models (entry/exit) are re-fit on each training window. Test-period signals use only parameters fit on training data.

**Context:** Some implementations fit selectors once on all data (simpler, but look-ahead biased). Others re-fit only signal models (partial solution).

**Reasoning:**
- Re-fitting selectors per fold is critical: a cointegration test or LSTM trained on the full 10-year window "knows" which pairs were cointegrated in the future, which is pure look-ahead bias.
- Re-fitting signal models per fold is equally critical: MLSignal and Kalman parameters must be estimated from past data only.
- The computational cost is high (8 selectors × 6 folds), but this is the only academically defensible approach.

**Trade-offs:**
- Slow. GNN/LSTM/Transformer re-training per fold may take 30–60 minutes per fold on CPU. stat_ml mode (skip DL selectors) is used for initial results; full mode for final paper numbers.

---

## D9 — TATAMOTORS.NS replaced with M&M.NS

**Decision:** Mahindra & Mahindra (`M&M.NS`) replaces Tata Motors (`TATAMOTORS.NS`) in the universe.

**Context:** TATAMOTORS.NS consistently returned `YFTzMissingError('possibly delisted; no timezone found')` across multiple runs and days.

**Reasoning:**
- TATAMOTORS is not delisted on NSE (it is a Nifty 50 constituent), but yfinance has a known intermittent issue with certain ticker timezone metadata. Since the error is consistent and prevents data download, the ticker is unreliable as a data source.
- M&M.NS (Mahindra & Mahindra) is a direct substitute: also a large-cap NSE automobile/conglomerate company, Nifty 100 constituent, similar market cap tier, liquid, and yfinance-reliable.
- The substitution preserves the universe's Auto sector representation (5 stocks) without altering the sector balance.
- This substitution was made for a purely technical, documented reason — not to improve backtest results.

**Trade-offs:** None material. The universe character (sector balance, liquidity profile, market-cap tier) is unchanged.

---

## D11 — Optimal minimum hold period = 30 trading days

**Decision:** `DEFAULT_MIN_HOLD = 30` in `experiments/config.py`. Applied to all experiments from E3 onwards.

**Context:** E2 (hold_period_sweep.py) swept min_hold_bars ∈ {0, 5, 10, 15, 20, 25, 30, 40} on the full 10-year daily dataset. Hold=30 produced the peak net Sharpe (0.481) and peak gross Sharpe (0.963).

**Reasoning — why 30 days is the optimum (not arbitrary):**
- At hold < 20 days: the signal layer over-trades relative to mean-reversion speed. Cost drag (approximately 1.6 pp per round-trip using 16.3 bps per trade) exceeds alpha per trade when compounded across excessive turnover.
- At hold = 20–30 days: positions are held long enough to realise mean-reversion alpha (Hurst 0.19, estimated OU half-life ~20–30 days) while cost drag falls to 2–3 pp/year.
- At hold = 40 days: positions are held past the mean-reversion half-life. A spread that has reverted and begun diverging in the other direction is still being held long — turning a winning trade into a losing one. Net Sharpe collapses from +0.481 to −0.239.
- This boundary (30 vs 40 days) is directly consistent with OU theory: the half-life of the mean-reversion process is ~log(2)/κ, where κ is the reversion speed. For Hurst ≈ 0.19 spreads, this gives half-life ≈ 20–30 bars, which is exactly where performance peaks.

**Reasoning — why this is a methodological parameter, not a tuned hyperparameter (see D5):**
- The hold period affects transaction cost structure, not signal direction. It cannot "learn" future prices.
- It was determined on the FULL 10-year dataset before any WFV fold-level analysis begins. This is analogous to choosing the rebalancing frequency.
- Using 30 days in WFV test folds does not cause look-ahead bias: no future price information flows backward through this parameter.
- Transparency: the parameter selection process is fully documented and reproducible.

**Trade-offs:**
- The 30-day hold was found using stat_only mode (4 selectors). When re-run with full mode (8 selectors, including LSTM/Transformer/GNN which may select different pairs with different half-lives), the optimal hold might shift. If full-mode results suggest a different optimal, we will update this decision with documentation.
- A 30-day minimum hold means the strategy can enter/exit each pair at most ~8 times per year. This is quite conservative. In live trading this would limit responsiveness to regime changes.

---

## D10 — `experiments/config.py` as single source of truth

**Decision:** All experiment parameters (universe, date ranges, weights, capital, periods_per_year, hold values) are defined once in `experiments/config.py`. All experiment scripts import from there.

**Context:** Without a central config, parameters drift across scripts and results become non-comparable.

**Reasoning:**
- A single source of truth ensures all experiments are comparable: same universe, same costs, same capital.
- Any parameter change is made once and propagates everywhere.
- Makes the experimental setup fully transparent to a reader: one file fully describes all fixed choices.

**Trade-offs:** Slight inflexibility — an experiment that needs a genuinely different universe must either import from config and override locally (acceptable) or define a separate config (also acceptable). The key constraint is: the override must be documented.


---

## D12 — Transaction cost model: 16.28 bps round-trip (NSE India)

**Decision:** All experiments use 16.28 bps round-trip per pair trade.

| Component | Rate |
|---|---|
| Brokerage | 5.00 bps |
| STT (one-way delivery) | 1.00 bps |
| NSE transaction charges | 0.335 bps |
| GST on brokerage | 0.90 bps |
| SEBI charges | 0.005 bps |
| Stamp duty | 0.015 bps |
| Slippage / market impact | 9.025 bps |
| **Total round-trip** | **16.28 bps** |

**Context:** Earlier E1-E8 runs (March-April 2026) used 22.5-22.9 bps. Corrected May 2026 after audit against NSE fee schedules and broker documentation.

**Reasoning:**
- STT for delivery trades is 0.1% ONE-WAY (not round-trip). Earlier model double-counted it.
- 9.025 bps slippage is intentionally conservative for Nifty 100 large-caps — strengthens any positive result by setting a high hurdle.
- Consistent with Paper 2 (also 16.28 bps) — enables direct cross-paper comparison.
- Old E1-E8 JSON results archived in experiments/results/backup_old_costs/. NOT used in paper.

**Trade-offs:** Sensitivity table (costs +/- 5 bps) will appear in robustness section.

---

## D13 — CPU-only deterministic ML runs (no GPU)

**Decision:** All ML training (LSTM, Transformer, GNN) uses CPU only. Env vars: `CUDA_VISIBLE_DEVICES=""`, `TF_DETERMINISTIC_OPS=1`, `PYTHONHASHSEED=42`.

**Evidence of the problem:** Three consecutive GPU runs with identical seed=42 produced net Sharpe: +0.398, -0.386, +0.840. Range = 1.226 from non-determinism alone — larger than the signal itself. This makes a single GPU run scientifically meaningless.

**Why it happens:** GPU cuDNN uses non-associative parallel reductions for speed. Result depends on thread scheduling, which varies run-to-run even with identical seeds (cuDNN documentation section 4.1). Cannot be fixed with seeds alone.

**Reasoning:**
- Reproducibility is a first-order requirement for academic publication. A reviewer must be able to reproduce any reported number.
- CPU-only produces bit-identical results across runs (verified on Kalpana, June 2026).
- Extra wall time (~5h per full WFV on 8-core CPU vs GPU) is worth it for reproducibility.
- Three CPU-only runs of the same config should produce identical output. If they differ, it indicates a code bug, not hardware non-determinism.

**Trade-offs:** Slower training limits hyperparameter search. Models kept architecturally simple (1-2 layer LSTM) to remain tractable. CPU results may differ from GPU in a live deployment — disclosed as limitation.

---

## D14 — NSE Nifty 100 universe expanded to ~95 tickers for Paper 1

**Decision:** Paper 1 uses ~95 Nifty 100 constituents (vs 35 in early experiments). Tickers with fewer than 1,250 trading days (approx 5 years) in the 2015-2024 window are dropped at fetch time. Final universe logged in `experiments/data/nse_nifty100/metadata.txt`.

**Reasoning:**
- C(35,2) = 595 candidate pairs. C(95,2) = 4,465 pairs. With 95 tickers, ensemble selector DISAGREEMENT is meaningfully testable: do different selectors choose different pairs? With 35 tickers, overlap is too high to test this.
- 95 tickers covers all 13 NSE sectors proportionally. 35-ticker universe left several sectors under-represented, limiting the generalisability claim.
- Academic papers on Indian pairs trading use 50-200 stocks (e.g., Kakushadze and Serur 2018, Pratap and Tiwari 2022). 95 is within a credible range.
- All Nifty 100 stocks are liquid enough (market cap > INR 10,000 cr) to make the 16.28 bps cost model realistic.

**Survivorship bias disclosure:** We use current Nifty 100 constituents. Stocks delisted or removed from the index between 2015-2024 are not included. This upward-biases results. Disclosed as a limitation. Consistent with most published pairs trading work. Future work should use point-in-time index data (Bloomberg, Refinitiv).

**Trade-offs:** Survivorship bias. Mitigated by disclosure and comparison with out-of-sample folds.

---

## D15 — Pre-downloaded Parquet cache; no yfinance at experiment runtime

**Decision:** All Paper 1 experiments read prices from `experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet`, fetched once by `fetch_paper1_data.py`. yfinance is never called during SLURM jobs.

**Reasoning:**
- Previous runs called yfinance per job, causing: Yahoo 429 rate-limit errors, job failures from network timeouts, non-reproducible results (Yahoo retroactively adjusts split/dividend-adjusted prices without version control).
- Fixed Parquet = bit-identical prices across all runs and all collaborators. Eliminates an entire class of reproducibility problems.
- Parquet load is ~10x faster than 95 sequential yfinance API calls.
- Paper 2 experimental-ablation already proved this pattern stable across all Kalpana SLURM runs.

**Implementation:**
- `PAPER1_DATA_PATH` env var exported in every SLURM script.
- `DataConfig.parquet_path` field in `core/data.py` reads the var and bypasses yfinance entirely.
- yfinance path retained as fallback for backward compatibility (Paper 2 / Streamlit app unaffected).

---

## D16 — Date range 2015-01-01 to 2024-12-31 (10 full years)

**Decision:** Paper 1 uses 2015-2024 (10 years). Paper 2 used 2020-2025 (5 years).

**Reasoning:**
- ML models (LSTM, Transformer, GNN) require substantial training data. 5 years x 250 days = 1,250 observations is marginal for sequence models. 10 years x 250 = 2,500 observations is defensible.
- 10 years captures a full market cycle: demonetisation (Nov 2016), GST shock (Jul 2017), IL&FS crisis (Sep 2018), pre-COVID bull run (2019), COVID crash (Mar 2020), V-shaped recovery (Apr-Dec 2020), global rate hike cycle (2022-2024). Regime diversity is critical for testing robustness.
- Pre-2015 yfinance data for NSE has quality issues (missing data, suspicious adjusted-close jumps) for many Nifty 100 tickers. 2015 is a natural data quality boundary.
- End date 2024-12-31: keeps the dataset clean and closed. 2025 data is partial/live and should not be included in a fixed-dataset study.

**Trade-offs:** 10-year backtest may overstate live performance if 2015-2024 was unusually favourable for Indian equity pairs. Addressed via fold-level trend analysis (does Sharpe degrade in later folds?) and robustness checks.

---

## D17 — Walk-forward validation: 6-fold expanding for Paper 1 vs 4-fold rolling for Paper 2

**Decision:** Paper 1 = 6-fold expanding WFV. Paper 2 = 4-fold rolling WFV. Intentionally different designs for different research questions.

**Paper 1 fold design (6-fold expanding, 2015-2024):**

| Fold | Train | Test |
|---|---|---|
| F1 | 2015-2017 | 2018 |
| F2 | 2015-2018 | 2019 |
| F3 | 2015-2019 | 2020 |
| F4 | 2015-2020 | 2021 |
| F5 | 2015-2021 | 2022 |
| F6 | 2015-2022 | 2023-2024 |

**Reasoning:**
- Paper 1 question = MODEL COMPARISON (does ensemble > single-strategy?). Expanding WFV is standard for model comparison — all models see the same data sequence, comparison is within-fold-fair.
- Paper 2 question = CROSS-MARKET GENERALIZATION (does universe quality dominate?). Rolling WFV controls for calendar period, isolating the universe effect from time-period effects.
- Expanding WFV allows ML models to train on progressively larger datasets — their intended operating condition. A fixed rolling window would artificially handicap ML selectors relative to stat selectors.
- 6 folds over 10 years gives 1 test year per fold (except F6 = 2 years) — sufficient statistical power per fold without over-fragmenting.

**Important note:** Paper 2 found that expanding WFV on Nifty 50 produced worse results than rolling. This is NOT an indictment of expanding WFV as a design — it is a finding that Nifty 50 strategy performance degrades when ML sees more training data (consistent with the overfitting narrative). Paper 1 will replicate this test explicitly on Nifty 100 as a robustness check.

**Trade-offs:** Heterogeneous fold sizes (see D7). Later folds have more training data, creating non-uniform conditions. Accepted trade-off for using all historical data.

---

## D18 — CNNSelector disabled: presented as explicit negative result

**Decision:** CNNSelector weight=0.0 in all Paper 1 experiments. NOT silently dropped — disclosed fully in methodology section and Appendix B.

**Evidence:** E3 ablation: adding CNNSelector to the 7-selector ensemble caused net Sharpe to fall from ~+0.481 to ~-0.12. Removing it restored performance.

**Reasoning:**
- Silently omitting CNN would create a reviewer-detectable inconsistency: introduction claims 8 selectors, results show 7.
- An explicit negative result is more honest and more interesting. It demonstrates that ablation was performed rigorously and that not all ML architectures are suitable for pairs selection.
- Likely mechanism (not yet formally tested): CNN receptive fields capture short-term autocorrelation and momentum patterns. Pairs trading requires MEAN-REVERSION selectors. Momentum selection is antagonistic to mean-reversion strategy. CNN was selecting the wrong pairs.
- Disclosure text: "We initially designed 8 selectors. CNNSelector was disabled following E3 ablation, which showed consistent ensemble degradation (net SR: +0.481 to -0.12). Individual CNN results and architectural analysis are reported in Appendix B."

**Trade-offs:** Reduces headline from 8-selector to 7-selector ensemble. This is stronger — it shows honest ablation.

---

## D19 — ZScore as primary signal, OU as secondary

**Decision:** Z-Score signal (entry |z| > 2.0, exit |z| < 0.5) is the primary signal for all headline results. OU signal reported as secondary comparison. RL signal is exploratory only, not in main results.

**Reasoning:**
- ZScore is the most widely used signal in pairs trading literature (Gatev et al. 2006, Avellaneda and Lee 2010, Elliott et al. 2005). Using it as the primary signal maximises comparability with prior work.
- ZScore is fully interpretable: every trade can be audited by inspecting the spread z-score. OU requires understanding MLE parameter estimation; RL is a black box. Interpretability matters for a methodology paper.
- Paper 2 empirically confirmed: ZScore net Sharpe +0.752 vs OU slightly lower on NSE Nifty 50. ZScore's simplicity is its strength — it does not overfit OU parameters on short training windows.
- RL signal is too unstable for publication at this stage: GPU non-determinism compounds with RL reward shaping instability. Even CPU-only RL shows high variance across random seeds.

**Trade-offs:** OU is theoretically superior when OU parameters are well-estimated. In practice, estimation noise on short training windows makes OU parameters unreliable. Reporting both allows the paper to make this observation empirically.

---

## D20 — Statistical testing: block bootstrap CIs + Diebold-Mariano + Bonferroni

**Decision:** All Sharpe ratios carry bootstrap 95% CIs (1,000 resamples, block size=20 trading days). Pairwise comparisons use Diebold-Mariano test on daily PnL. Multiple comparisons corrected via Bonferroni (alpha = 0.05 / N).

**Reasoning:**
- **Block bootstrap not iid bootstrap:** Daily returns are autocorrelated (momentum, volatility clustering). iid bootstrap destroys autocorrelation and understates variance of the Sharpe estimator — producing falsely narrow CIs. Block bootstrap preserves short-range autocorrelation (Politis and Romano 1994).
- **Block size = 20 (~one trading month):** Standard choice. Sensitivity to block size 10/20/30 reported as robustness check.
- **Diebold-Mariano:** Directly tests whether strategy A and B have statistically distinguishable daily PnL. More appropriate than comparing overlapping CIs.
- **Why not Jobson-Korkie Sharpe t-test:** Requires joint normality of returns. NSE returns have excess kurtosis ~4-8. Bootstrap is distribution-free and more appropriate for fat-tailed data.
- **Bonferroni:** Controls Family-Wise Error Rate conservatively. With ~10 pairwise comparisons (each single-selector vs ensemble), threshold = 0.005. Results that survive Bonferroni are credible. More powerful corrections (Holm-Bonferroni, BH) can be added in revision if needed.

**Trade-offs:** Block bootstrap CIs are wider than parametric CIs. This is correct and intentional. If a result only survives narrow parametric CIs, it is fragile. We want findings that survive conservative testing.

---

## D21 — Two-paper structure: Paper 1 (ensemble value) vs Paper 2 (universe quality)

**Paper 1 — Hybrid Ensemble Pairs Trading:** Does the ML + statistical hybrid ensemble outperform single-strategy statistical baselines on NSE? Universe: Nifty 100, 2015-2024, expanding WFV. Status: Experiments in progress.

**Paper 2 — Universe Quality Dominates Methodology:** Is performance driven more by universe selection than selector methodology? Universe: NSE Nifty 50 + multi-market (India/US/UK/Brazil), rolling WFV. Status: SUBMISSION READY (2026-06-04, all 29 Round 4 critiques resolved).

**Why split into two papers:**
- Fundamentally different RQs requiring different experimental designs (expanding vs rolling WFV, single-market vs multi-market, ensemble-vs-single vs universe-vs-method).
- Combining both into one paper would create an unwieldy methodology section and muddle the central argument of each.
- Natural narrative arc: Paper 1 = "we built an ensemble, does it beat single-strategy?" Paper 2 = "we stress-tested it across markets and found a deeper finding: universe quality dominates the method." The two papers cite each other and form a coherent series.

**Sequencing:** Paper 2 submitted first (already ready). Paper 1 follows once experiments complete and writing is done. Paper 1 will reference Paper 2 in conclusions.

---

## D22 — SLURM job dependency chaining

**Decision:** E1 + E3 + E4a + E4b run in parallel (immediate). E4c runs independently. E5 depends on E4a. E6 depends on E4a + E4b + E4c.

**Reasoning:**
- E1 (frequency comparison) and E3 (ablation) are cheap (<4h, stat-only). No reason to queue them.
- E4a (stat_only WFV) and E4b (stat_ml WFV) are medium cost (~6h each). Parallelising halves wall time with no memory contention risk.
- E4c (full hybrid WFV with LSTM+Transformer+GNN, ~24h CPU) runs alone to avoid memory contention with other jobs.
- E5 (benchmark comparison) requires E4a output to compute relative metrics (ensemble vs buy-and-hold vs single-selector).
- E6 (significance tests, DM test, bootstrap CIs) requires ALL E4 variants to be complete before any cross-config comparison is valid.
- Total wall time: ~30h vs ~48h sequential.

**Failure recovery:** If E4a fails, SLURM marks E5 and E6 as dependency-failed. Re-submit only E4a; E5 and E6 will re-queue automatically with their original dependencies.
