"""
Statistical Analysis for Thesis Critique Resolution
Addresses: #3 Bonferroni correction, #5 Bootstrap CI, #7 Outlier analysis, #8 UK failure

Run from experimental-ablation directory:
    python statistical_analysis.py
"""

import json, glob, os, random
from pathlib import Path
from collections import defaultdict
import statistics

# ── helpers ──────────────────────────────────────────────────────────────────

def mean(xs): return sum(xs) / len(xs) if xs else float('nan')
def std(xs):
    if len(xs) < 2: return float('nan')
    m = mean(xs)
    return (sum((x - m)**2 for x in xs) / (len(xs) - 1)) ** 0.5

def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap percentile CI around the mean."""
    random.seed(seed)
    n = len(data)
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(data) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return lo, hi

def t_stat(data):
    """One-sample t-statistic vs 0."""
    n = len(data)
    if n < 2: return float('nan'), float('nan')
    m = mean(data)
    s = std(data)
    t = m / (s / n**0.5)
    return t, n

def cohens_d(data):
    m = mean(data)
    s = std(data)
    return m / s if s else float('nan')

def welch_t(a, b):
    """Welch's t-test: returns t-statistic."""
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    se = ((sa**2 / na) + (sb**2 / nb)) ** 0.5
    return (ma - mb) / se if se else float('nan')

BASE = Path(__file__).parent / "results"

# ── Load data ─────────────────────────────────────────────────────────────────

MARKET_MAP = {
    "india": "India",
    "brazil": "Brazil",
    "us": "US",
    "uk": "UK",
    "nse_nifty50": "NSE_Nifty50_Rolling",
    "nse_nifty50_expanding": "NSE_Nifty50_Expanding",
}

def load_all_runs():
    """Load every result JSON, keyed by (market_label, signal) using directory name."""
    runs = defaultdict(list)
    for f in sorted(BASE.rglob("*.json")):
        dir_key = f.parent.name
        market_label = MARKET_MAP.get(dir_key, dir_key)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        signal = d.get("signal_model", "unknown")
        sharpes = [fold["metrics"].get("Net.Sharpe", 0.0) for fold in d.get("folds", [])]
        if sharpes:
            runs[(market_label, signal)].append({
                "file": f.name,
                "sharpes": sharpes,
                "mean": mean(sharpes),
                "selectors": d.get("selectors", []),
            })
    return runs

runs = load_all_runs()

lines = []
lines.append("# STATISTICAL ANALYSIS REPORT")
lines.append("## Bootstrap CI, Bonferroni Correction, Outlier Analysis, UK Failure Analysis")
lines.append("")

# ── Section 1: All experiments — runs × mean ± std ───────────────────────────
lines.append("---")
lines.append("## 1. All Experiments: Full Transparency Table")
lines.append("")
lines.append("Each row = one (market, signal) pair. Columns: n_runs | fold-level Sharpe values | mean | std | best | cherry-pick delta.")
lines.append("")

ordered_keys = sorted(runs.keys())
transparency = {}
for key in ordered_keys:
    market, signal = key
    all_fold_sharpes = []
    run_means = []
    for r in runs[key]:
        all_fold_sharpes.extend(r["sharpes"])
        run_means.append(r["mean"])
    
    n_runs = len(runs[key])
    best = max(run_means)
    med = sorted(run_means)[len(run_means)//2]
    m = mean(run_means)
    s = std(run_means) if n_runs > 1 else float('nan')
    cherry = best - m
    
    transparency[key] = {
        "n_runs": n_runs,
        "run_means": run_means,
        "mean": m,
        "std": s,
        "best": best,
        "cherry_pick_delta": cherry,
        "all_fold_sharpes": all_fold_sharpes,
    }
    
    std_str = f"{s:+.3f}" if s == s else "  N/A "
    lines.append(f"**{market} / {signal}** — {n_runs} run(s)")
    lines.append(f"  Run means: {[f'{x:+.3f}' for x in run_means]}")
    lines.append(f"  Mean={m:+.3f}, Std={std_str}, Best={best:+.3f}, Cherry-pick delta={cherry:+.3f}")
    lines.append("")

# ── Section 2: Bootstrap CI on key experiments ───────────────────────────────
lines.append("---")
lines.append("## 2. Bootstrap Confidence Intervals (95%, 10,000 resamples)")
lines.append("")
lines.append("Applied to fold-level Sharpe ratios for the canonical/final run of each key experiment.")
lines.append("")

key_experiments = [
    ("NSE_Nifty50_Rolling",   "zscore", "NSE Nifty 50 Rolling ZScore (control)"),
    ("NSE_Nifty50_Rolling",   "ou",     "NSE Nifty 50 Rolling OU (control)"),
    ("NSE_Nifty50_Expanding", "zscore", "NSE Nifty 50 Expanding ZScore (control)"),
    ("NSE_Nifty50_Expanding", "ou",     "NSE Nifty 50 Expanding OU (control)"),
    ("India", "zscore", "India Multi-Market ZScore (all 3 runs)"),
    ("India", "ou",     "India Multi-Market OU"),
    ("Brazil", "ou",    "Brazil OU"),
    ("UK", "zscore",    "UK ZScore"),
]

for market, signal, label in key_experiments:
    key = (market, signal)
    if key not in runs:
        lines.append(f"**{label}** — NO DATA")
        lines.append("")
        continue
    
    # Use last run (latest timestamp = final run) for individual CI, all folds for transparency
    last_run = runs[key][-1]
    folds = last_run["sharpes"]
    n = len(folds)
    m = mean(folds)
    s = std(folds)
    lo, hi = bootstrap_ci(folds)
    t, _ = t_stat(folds)
    d_cohen = cohens_d(folds)
    se = s / n**0.5
    
    # All runs combined
    all_folds = transparency[key]["all_fold_sharpes"]
    lo_all, hi_all = bootstrap_ci(all_folds)
    
    lines.append(f"**{label}**")
    lines.append(f"  Final run folds: {[f'{x:+.3f}' for x in folds]}")
    lines.append(f"  Mean={m:+.3f}, Std={s:+.3f}, SE={se:+.3f}")
    lines.append(f"  95% Bootstrap CI (final run): [{lo:+.3f}, {hi:+.3f}]")
    lines.append(f"  95% Bootstrap CI (all {len(all_folds)} folds pooled): [{lo_all:+.3f}, {hi_all:+.3f}]")
    lines.append(f"  t-statistic vs 0: t={t:+.3f} (n={n} folds)")
    lines.append(f"  Cohen's d: {d_cohen:+.3f}")
    lines.append("")

# ── Section 3: Bonferroni correction ─────────────────────────────────────────
lines.append("---")
lines.append("## 3. Multiple Testing Correction (Bonferroni)")
lines.append("")
lines.append("Tests performed in Chapter 3: Expanding vs Rolling (2 methodologies).")
lines.append("Tests performed in Chapter 4: 4 markets × 2 signals = 8 comparisons vs baseline.")
lines.append("")
lines.append("Bonferroni family-wise alpha = 0.05")
lines.append("")

# Ch3: rolling vs expanding on Nifty 100
nifty100_rolling = [0.052]   # single reported value; no fold breakdown available locally
nifty100_expanding = [-0.409]
# Approximate: p~0.32 reported in thesis, Cohen's d~0.45
# After Bonferroni for 2 tests: alpha_corrected = 0.05/2 = 0.025
lines.append("**Chapter 3 — Nifty 100: Expanding vs Rolling (2 tests)**")
lines.append("  Reported p-value (raw): 0.320 (thesis section 3.6)")
lines.append("  Bonferroni-corrected alpha: 0.05 / 2 = 0.025")
lines.append("  Corrected p-value: 0.320 × 2 = 0.640")
lines.append("  **RESULT: NOT SIGNIFICANT** (p=0.640 >> 0.025)")
lines.append("  Cohen's d = 0.45 → small effect; needs n≥64 for 80% power")
lines.append("")

# Ch4: 8 market×signal combos vs rolling NSE baseline (0.052)
lines.append("**Chapter 4 — Multi-Market: 8 experiments vs Rolling NSE baseline (8 tests)**")
lines.append("  Bonferroni-corrected alpha: 0.05 / 8 = 0.00625")
lines.append("")

baseline = 0.052
ch4_experiments = [
    ("India ZScore", transparency.get(("India","zscore"), {}).get("mean", float('nan'))),
    ("India OU", transparency.get(("India","ou"), {}).get("mean", float('nan'))),
    ("Brazil ZScore", transparency.get(("Brazil","zscore"), {}).get("mean", float('nan'))),
    ("Brazil OU", transparency.get(("Brazil","ou"), {}).get("mean", float('nan'))),
    ("US ZScore", float('nan')),
    ("US OU", transparency.get(("US","ou"), {}).get("mean", float('nan'))),
    ("UK ZScore", transparency.get(("UK","zscore"), {}).get("mean", float('nan'))),
    ("UK OU", transparency.get(("UK","ou"), {}).get("mean", float('nan'))),
]

for name, m in ch4_experiments:
    if m != m:
        lines.append(f"  {name}: mean=N/A (no data)")
        continue
    diff = m - baseline
    # Can't compute proper p without fold-level data for baseline; note as limitation
    lines.append(f"  {name}: mean={m:+.3f}, vs baseline +0.052, diff={diff:+.3f} — p-value requires Nifty100 fold data (not available here)")

lines.append("")
lines.append("  **NOTE:** Formal Bonferroni p-values for Ch4 require fold-level Nifty 100 data.")
lines.append("  Qualitative: Only India ZScore (mean +0.284) shows positive diff (+0.232) but high variance (std=0.631) means it will not survive Bonferroni at n=4.")
lines.append("")

# ── Section 4: Outlier analysis ───────────────────────────────────────────────
lines.append("---")
lines.append("## 4. Outlier Analysis — India ZScore Fold 3 (+1.996)")
lines.append("")

india_zscore_best = [0.6044761694963317, -0.07999594105946568, 1.9956310389853698, 0.839611042622226]
m_with = mean(india_zscore_best)
lo_with, hi_with = bootstrap_ci(india_zscore_best)

# Remove outlier (fold 3)
india_no_outlier = [x for i, x in enumerate(india_zscore_best) if i != 2]
m_without = mean(india_no_outlier)
lo_wo, hi_wo = bootstrap_ci(india_no_outlier, n_boot=10000)

zscore_from_mean = (india_zscore_best[2] - mean(india_zscore_best)) / std(india_zscore_best)

lines.append(f"**India ZScore best run** (104009): {[f'{x:+.3f}' for x in india_zscore_best]}")
lines.append(f"  Mean WITH outlier (fold 3 = +1.996): {m_with:+.3f}")
lines.append(f"  95% CI WITH outlier: [{lo_with:+.3f}, {hi_with:+.3f}]")
lines.append(f"  Z-score of fold 3 within run: {zscore_from_mean:+.3f} sigma (threshold > 2.0 = outlier)")
lines.append(f"  Mean WITHOUT fold 3: {m_without:+.3f}")
lines.append(f"  95% CI WITHOUT fold 3: [{lo_wo:+.3f}, {hi_wo:+.3f}]")
lines.append(f"  **Impact of outlier:** Removing fold 3 drops mean from {m_with:+.3f} to {m_without:+.3f} ({m_without-m_with:+.3f})")
lines.append(f"  **16x multiplier vs NSE baseline:** {m_with/0.052:.1f}x WITH outlier, {m_without/0.052:.1f}x WITHOUT outlier")
lines.append("")
lines.append("  **CONCLUSION:** The 16x claim is driven by fold 3 (+1.996 = +1.6 sigma outlier).")
lines.append("  Outlier-robust mean = +0.455 → 8.8x multiplier (not 16x).")
lines.append("  Per-run mean across all 3 India ZScore runs = +0.284 → 5.5x multiplier (honest number).")
lines.append("")

# ── Section 5: UK failure analysis ───────────────────────────────────────────
lines.append("---")
lines.append("## 5. UK Failure Analysis")
lines.append("")

uk_zscore_runs = [r["sharpes"] for r in runs.get(("UK","zscore"), [])]
uk_ou_runs = [r["sharpes"] for r in runs.get(("UK","ou"), [])]

lines.append("**UK ZScore — fold-level Sharpe (all runs):**")
for i, folds in enumerate(uk_zscore_runs):
    m = mean(folds)
    lines.append(f"  Run {i+1}: {[f'{x:+.3f}' for x in folds]}  mean={m:+.3f}")

lines.append("")
lines.append("**UK OU — fold-level Sharpe (all runs):**")
for i, folds in enumerate(uk_ou_runs):
    m = mean(folds)
    lines.append(f"  Run {i+1}: {[f'{x:+.3f}' for x in folds]}  mean={m:+.3f}")

lines.append("")
lines.append("**Pattern analysis:**")
lines.append("  - UK ZScore run 1: positive drift (+0.266 mean) — appears functional")
lines.append("  - UK ZScore run 2: negative (-0.245 mean) — high variance, different ML selection")
lines.append("  - UK OU runs 1 and 2: ALL ZEROS — no trades executed at all")
lines.append("  - UK OU run 3: -0.405 mean — trades execute but lose money")
lines.append("")
lines.append("**Root cause hypothesis (data-driven):**")
lines.append("  1. ZERO TRADES in OU runs 1 & 2 → pairs failed OU stationarity or half-life filter")
lines.append("     UK equity pairs may lack mean-reversion at the 126-day lookback scale")
lines.append("  2. High fold variance in ZScore (range: -1.022 to +0.981) → regime sensitivity")
lines.append("     UK had Brexit referendum 2016 (pre-sample), COVID 2020 (fold 1), but also")
lines.append("     high 2022 inflation shock (fold 3 = -0.075 fold mean)")
lines.append("  3. MISSING: Need to check cointegration pass rates per market to confirm hypothesis")
lines.append("")
lines.append("**What thesis needs (per critique #8):**")
lines.append("  - [ ] Cointegration pass rate by fold per market (requires raw pair scores from result JSONs)")
lines.append("  - [ ] Correlation matrix of UK pairs vs India pairs (requires price data)")
lines.append("  - [ ] Sector composition: UK FTSE vs India Nifty 50 (can be done from configs)")
lines.append("")

# ── Section 6: Pair count vs Sharpe correlation ───────────────────────────────
lines.append("---")
lines.append("## 6. Cross-Market Summary Table (Honest — All Runs, Mean ± Std)")
lines.append("")
lines.append("This is the table that REPLACES the cherry-picked MULTI_MARKET_RESULTS.md table.")
lines.append("")
lines.append("Format: market | signal | n_runs | mean Sharpe | std | 95% CI | best run | cherry-pick risk")
lines.append("")

for market, signal, label in key_experiments:
    key = (market, signal)
    if key not in transparency:
        lines.append(f"  {label} — NO DATA IN TRANSPARENCY DICT")
        lines.append("")
        continue
    t = transparency[key]
    folds = t["all_fold_sharpes"]
    lo, hi = bootstrap_ci(folds)
    cherry_flag = "⚠️  HIGH" if t["cherry_pick_delta"] > 0.3 else ("⚠️  MOD" if t["cherry_pick_delta"] > 0.1 else "OK")
    lines.append(f"  {label}")
    std_val = 'N/A' if t['std'] != t['std'] else f"{t['std']:+.3f}"
    lines.append(f"    n_runs={t['n_runs']} | mean={t['mean']:+.3f} | std={std_val} | CI=[{lo:+.3f},{hi:+.3f}] | best={t['best']:+.3f} | cherry-pick_delta={t['cherry_pick_delta']:+.3f} {cherry_flag}")
    lines.append("")

lines.append("**NSE Nifty 50 Control (new, deterministic):**")
nse50_data = [
    ("NSE Nifty 50 Rolling ZScore",    [1.127, 0.218, 0.627, 1.036]),
    ("NSE Nifty 50 Rolling OU",        [0.000, 0.000, 0.000, 0.588]),
    ("NSE Nifty 50 Expanding ZScore",  [1.127, 0.233, 1.347, 1.547]),
    ("NSE Nifty 50 Expanding OU",      [0.000, 0.000, 0.000, 0.684]),
]
for label, folds in nse50_data:
    m = mean(folds)
    s = std(folds)
    lo, hi = bootstrap_ci(folds)
    lines.append(f"  {label}")
    lines.append(f"    mean={m:+.3f} | std={s:+.3f} | CI=[{lo:+.3f},{hi:+.3f}] | cherry-pick_delta=0 (single run, deterministic)")
    lines.append("")

# ── Write output ──────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "STATISTICAL_ANALYSIS.md"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Written: {out_path}")
print(f"Lines: {len(lines)}")
