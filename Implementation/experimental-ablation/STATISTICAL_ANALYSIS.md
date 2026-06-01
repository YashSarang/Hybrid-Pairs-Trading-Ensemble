# STATISTICAL ANALYSIS REPORT
## Bootstrap CI, Bonferroni Correction, Outlier Analysis, UK Failure Analysis

---
## 1. All Experiments: Full Transparency Table

Each row = one (market, signal) pair. Columns: n_runs | fold-level Sharpe values | mean | std | best | cherry-pick delta.

**Brazil / ou** — 3 run(s)
  Run means: ['+0.000', '+0.000', '+0.321']
  Mean=+0.107, Std=+0.185, Best=+0.321, Cherry-pick delta=+0.214

**Brazil / zscore** — 2 run(s)
  Run means: ['-0.400', '-0.225']
  Mean=-0.312, Std=+0.124, Best=-0.225, Cherry-pick delta=+0.088

**India / ou** — 2 run(s)
  Run means: ['+0.000', '+0.200']
  Mean=+0.100, Std=+0.141, Best=+0.200, Cherry-pick delta=+0.100

**India / zscore** — 3 run(s)
  Run means: ['+0.398', '-0.386', '+0.840']
  Mean=+0.284, Std=+0.621, Best=+0.840, Cherry-pick delta=+0.556

**NSE_Nifty50_Expanding / ou** — 1 run(s)
  Run means: ['+0.171']
  Mean=+0.171, Std=  N/A , Best=+0.171, Cherry-pick delta=+0.000

**NSE_Nifty50_Expanding / zscore** — 1 run(s)
  Run means: ['+1.064']
  Mean=+1.064, Std=  N/A , Best=+1.064, Cherry-pick delta=+0.000

**NSE_Nifty50_Rolling / ou** — 1 run(s)
  Run means: ['+0.147']
  Mean=+0.147, Std=  N/A , Best=+0.147, Cherry-pick delta=+0.000

**NSE_Nifty50_Rolling / zscore** — 1 run(s)
  Run means: ['+0.752']
  Mean=+0.752, Std=  N/A , Best=+0.752, Cherry-pick delta=+0.000

**UK / ou** — 3 run(s)
  Run means: ['+0.000', '+0.000', '-0.405']
  Mean=-0.135, Std=+0.234, Best=+0.000, Cherry-pick delta=+0.135

**UK / zscore** — 2 run(s)
  Run means: ['+0.265', '-0.245']
  Mean=+0.010, Std=+0.361, Best=+0.265, Cherry-pick delta=+0.255

**US / ou** — 3 run(s)
  Run means: ['+0.000', '+0.000', '-0.254']
  Mean=-0.085, Std=+0.147, Best=+0.000, Cherry-pick delta=+0.085

**US / unknown** — 3 run(s)
  Run means: ['+0.774', '+0.000', '+0.116']
  Mean=+0.296, Std=+0.417, Best=+0.774, Cherry-pick delta=+0.477

**results / ou** — 5 run(s)
  Run means: ['+0.000', '+0.000', '+0.000', '+0.000', '+0.000']
  Mean=+0.000, Std=+0.000, Best=+0.000, Cherry-pick delta=+0.000

**results / unknown** — 3 run(s)
  Run means: ['+0.774', '+0.000', '+0.116']
  Mean=+0.296, Std=+0.417, Best=+0.774, Cherry-pick delta=+0.477

**results / zscore** — 4 run(s)
  Run means: ['+0.398', '-0.386', '-0.400', '+0.265']
  Mean=-0.031, Std=+0.422, Best=+0.398, Cherry-pick delta=+0.429

---
## 2. Bootstrap Confidence Intervals (95%, 10,000 resamples)

Applied to fold-level Sharpe ratios for the canonical/final run of each key experiment.

**NSE Nifty 50 Rolling ZScore (control)**
  Final run folds: ['+1.127', '+0.218', '+0.627', '+1.036']
  Mean=+0.752, Std=+0.417, SE=+0.209
  95% Bootstrap CI (final run): [+0.422, +1.082]
  95% Bootstrap CI (all 4 folds pooled): [+0.422, +1.082]
  t-statistic vs 0: t=+3.605 (n=4 folds)
  Cohen's d: +1.802

**NSE Nifty 50 Rolling OU (control)**
  Final run folds: ['+0.000', '+0.000', '+0.000', '+0.588']
  Mean=+0.147, Std=+0.294, SE=+0.147
  95% Bootstrap CI (final run): [+0.000, +0.441]
  95% Bootstrap CI (all 4 folds pooled): [+0.000, +0.441]
  t-statistic vs 0: t=+1.000 (n=4 folds)
  Cohen's d: +0.500

**NSE Nifty 50 Expanding ZScore (control)**
  Final run folds: ['+1.127', '+0.233', '+1.347', '+1.547']
  Mean=+1.064, Std=+0.580, SE=+0.290
  95% Bootstrap CI (final run): [+0.511, +1.447]
  95% Bootstrap CI (all 4 folds pooled): [+0.511, +1.447]
  t-statistic vs 0: t=+3.669 (n=4 folds)
  Cohen's d: +1.835

**NSE Nifty 50 Expanding OU (control)**
  Final run folds: ['+0.000', '+0.000', '+0.000', '+0.684']
  Mean=+0.171, Std=+0.342, SE=+0.171
  95% Bootstrap CI (final run): [+0.000, +0.513]
  95% Bootstrap CI (all 4 folds pooled): [+0.000, +0.513]
  t-statistic vs 0: t=+1.000 (n=4 folds)
  Cohen's d: +0.500

**India Multi-Market ZScore (all 3 runs)**
  Final run folds: ['+0.604', '-0.080', '+1.996', '+0.840']
  Mean=+0.840, Std=+0.864, SE=+0.432
  95% Bootstrap CI (final run): [+0.150, +1.648]
  95% Bootstrap CI (all 12 folds pooled): [-0.207, +0.758]
  t-statistic vs 0: t=+1.945 (n=4 folds)
  Cohen's d: +0.973

**India Multi-Market OU**
  Final run folds: ['+0.000', '+0.000', '+0.000', '+0.800']
  Mean=+0.200, Std=+0.400, SE=+0.200
  95% Bootstrap CI (final run): [+0.000, +0.600]
  95% Bootstrap CI (all 8 folds pooled): [+0.000, +0.300]
  t-statistic vs 0: t=+1.000 (n=4 folds)
  Cohen's d: +0.500

**Brazil OU**
  Final run folds: ['+0.000', '+0.000', '+0.000', '+1.284']
  Mean=+0.321, Std=+0.642, SE=+0.321
  95% Bootstrap CI (final run): [+0.000, +0.963]
  95% Bootstrap CI (all 12 folds pooled): [+0.000, +0.321]
  t-statistic vs 0: t=+1.000 (n=4 folds)
  Cohen's d: +0.500

**UK ZScore**
  Final run folds: ['-1.022', '-0.249', '+0.967', '-0.677']
  Mean=-0.245, Std=+0.868, SE=+0.434
  95% Bootstrap CI (final run): [-0.849, +0.556]
  95% Bootstrap CI (all 8 folds pooled): [-0.439, +0.466]
  t-statistic vs 0: t=-0.565 (n=4 folds)
  Cohen's d: -0.282

---
## 3. Multiple Testing Correction (Bonferroni)

Tests performed in Chapter 3: Expanding vs Rolling (2 methodologies).
Tests performed in Chapter 4: 4 markets × 2 signals = 8 comparisons vs baseline.

Bonferroni family-wise alpha = 0.05

**Chapter 3 — Nifty 100: Expanding vs Rolling (2 tests)**
  Reported p-value (raw): 0.320 (thesis section 3.6)
  Bonferroni-corrected alpha: 0.05 / 2 = 0.025
  Corrected p-value: 0.320 × 2 = 0.640
  **RESULT: NOT SIGNIFICANT** (p=0.640 >> 0.025)
  Cohen's d = 0.45 → small effect; needs n≥64 for 80% power

**Chapter 4 — Multi-Market: 8 experiments vs Rolling NSE baseline (8 tests)**
  Bonferroni-corrected alpha: 0.05 / 8 = 0.00625

  India ZScore: mean=+0.284, vs baseline +0.052, diff=+0.232 — p-value requires Nifty100 fold data (not available here)
  India OU: mean=+0.100, vs baseline +0.052, diff=+0.048 — p-value requires Nifty100 fold data (not available here)
  Brazil ZScore: mean=-0.312, vs baseline +0.052, diff=-0.364 — p-value requires Nifty100 fold data (not available here)
  Brazil OU: mean=+0.107, vs baseline +0.052, diff=+0.055 — p-value requires Nifty100 fold data (not available here)
  US ZScore: mean=N/A (no data)
  US OU: mean=-0.085, vs baseline +0.052, diff=-0.137 — p-value requires Nifty100 fold data (not available here)
  UK ZScore: mean=+0.010, vs baseline +0.052, diff=-0.042 — p-value requires Nifty100 fold data (not available here)
  UK OU: mean=-0.135, vs baseline +0.052, diff=-0.187 — p-value requires Nifty100 fold data (not available here)

  **NOTE:** Formal Bonferroni p-values for Ch4 require fold-level Nifty 100 data.
  Qualitative: Only India ZScore (mean +0.284) shows positive diff (+0.232) but high variance (std=0.631) means it will not survive Bonferroni at n=4.

---
## 4. Outlier Analysis — India ZScore Fold 3 (+1.996)

**India ZScore best run** (104009): ['+0.604', '-0.080', '+1.996', '+0.840']
  Mean WITH outlier (fold 3 = +1.996): +0.840
  95% CI WITH outlier: [+0.150, +1.648]
  Z-score of fold 3 within run: +1.338 sigma (threshold > 2.0 = outlier)
  Mean WITHOUT fold 3: +0.455
  95% CI WITHOUT fold 3: [-0.080, +0.840]
  **Impact of outlier:** Removing fold 3 drops mean from +0.840 to +0.455 (-0.385)
  **16x multiplier vs NSE baseline:** 16.2x WITH outlier, 8.7x WITHOUT outlier

  **CONCLUSION:** The 16x claim is driven by fold 3 (+1.996 = +1.6 sigma outlier).
  Outlier-robust mean = +0.455 → 8.8x multiplier (not 16x).
  Per-run mean across all 3 India ZScore runs = +0.284 → 5.5x multiplier (honest number).

---
## 5. UK Failure Analysis

**UK ZScore — fold-level Sharpe (all runs):**
  Run 1: ['+0.110', '+0.044', '-0.075', '+0.982']  mean=+0.265
  Run 2: ['-1.022', '-0.249', '+0.967', '-0.677']  mean=-0.245

**UK OU — fold-level Sharpe (all runs):**
  Run 1: ['+0.000', '+0.000', '+0.000', '+0.000']  mean=+0.000
  Run 2: ['+0.000', '+0.000', '+0.000', '+0.000']  mean=+0.000
  Run 3: ['-1.272', '+0.000', '+0.000', '-0.346']  mean=-0.405

**Pattern analysis:**
  - UK ZScore run 1: positive drift (+0.266 mean) — appears functional
  - UK ZScore run 2: negative (-0.245 mean) — high variance, different ML selection
  - UK OU runs 1 and 2: ALL ZEROS — no trades executed at all
  - UK OU run 3: -0.405 mean — trades execute but lose money

**Root cause hypothesis (data-driven):**
  1. ZERO TRADES in OU runs 1 & 2 → pairs failed OU stationarity or half-life filter
     UK equity pairs may lack mean-reversion at the 126-day lookback scale
  2. High fold variance in ZScore (range: -1.022 to +0.981) → regime sensitivity
     UK had Brexit referendum 2016 (pre-sample), COVID 2020 (fold 1), but also
     high 2022 inflation shock (fold 3 = -0.075 fold mean)
  3. MISSING: Need to check cointegration pass rates per market to confirm hypothesis

**What thesis needs (per critique #8):**
  - [ ] Cointegration pass rate by fold per market (requires raw pair scores from result JSONs)
  - [ ] Correlation matrix of UK pairs vs India pairs (requires price data)
  - [ ] Sector composition: UK FTSE vs India Nifty 50 (can be done from configs)

---
## 6. Cross-Market Summary Table (Honest — All Runs, Mean ± Std)

This is the table that REPLACES the cherry-picked MULTI_MARKET_RESULTS.md table.

Format: market | signal | n_runs | mean Sharpe | std | 95% CI | best run | cherry-pick risk

  NSE Nifty 50 Rolling ZScore (control)
    n_runs=1 | mean=+0.752 | std=N/A | CI=[+0.422,+1.082] | best=+0.752 | cherry-pick_delta=+0.000 OK

  NSE Nifty 50 Rolling OU (control)
    n_runs=1 | mean=+0.147 | std=N/A | CI=[+0.000,+0.441] | best=+0.147 | cherry-pick_delta=+0.000 OK

  NSE Nifty 50 Expanding ZScore (control)
    n_runs=1 | mean=+1.064 | std=N/A | CI=[+0.511,+1.447] | best=+1.064 | cherry-pick_delta=+0.000 OK

  NSE Nifty 50 Expanding OU (control)
    n_runs=1 | mean=+0.171 | std=N/A | CI=[+0.000,+0.513] | best=+0.171 | cherry-pick_delta=+0.000 OK

  India Multi-Market ZScore (all 3 runs)
    n_runs=3 | mean=+0.284 | std=+0.621 | CI=[-0.207,+0.758] | best=+0.840 | cherry-pick_delta=+0.556 ⚠️  HIGH

  India Multi-Market OU
    n_runs=2 | mean=+0.100 | std=+0.141 | CI=[+0.000,+0.300] | best=+0.200 | cherry-pick_delta=+0.100 ⚠️  MOD

  Brazil OU
    n_runs=3 | mean=+0.107 | std=+0.185 | CI=[+0.000,+0.321] | best=+0.321 | cherry-pick_delta=+0.214 ⚠️  MOD

  UK ZScore
    n_runs=2 | mean=+0.010 | std=+0.361 | CI=[-0.439,+0.466] | best=+0.265 | cherry-pick_delta=+0.255 ⚠️  MOD

**NSE Nifty 50 Control (new, deterministic):**
  NSE Nifty 50 Rolling ZScore
    mean=+0.752 | std=+0.417 | CI=[+0.422,+1.082] | cherry-pick_delta=0 (single run, deterministic)

  NSE Nifty 50 Rolling OU
    mean=+0.147 | std=+0.294 | CI=[+0.000,+0.441] | cherry-pick_delta=0 (single run, deterministic)

  NSE Nifty 50 Expanding ZScore
    mean=+1.063 | std=+0.580 | CI=[+0.511,+1.447] | cherry-pick_delta=0 (single run, deterministic)

  NSE Nifty 50 Expanding OU
    mean=+0.171 | std=+0.342 | CI=[+0.000,+0.513] | cherry-pick_delta=0 (single run, deterministic)
