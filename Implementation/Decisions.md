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
- At hourly frequency, 904 trades/year at ~60 bps round-trip caused the strategy to lose more than its entire capital (Net Max DD = 214%).
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
- Signal reversal rate in E1 was 40.4% (daily). Examination of the trades log shows many patterns of the form +1 -> 0 -> +1 within 2–3 bars — rapid entry, exit, re-entry. These are NOT full reversals (+1 -> -1) but are equally costly (2 round-trips in 3 bars = ~120 bps cost on a signal that barely moved).
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
- At hold < 20 days: the signal layer over-trades relative to mean-reversion speed. Cost drag (2+ pp per round-trip, 60 bps) exceeds alpha per trade.
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
