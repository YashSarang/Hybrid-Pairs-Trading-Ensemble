"""
Figure CI Data — generates JSON for use by Streamlit/figure plotting code.
Provides mean ± std ± 95% CI for all key experiments for use in bar charts.
"""

import json, random
from pathlib import Path

def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    random.seed(seed)
    n = len(data)
    boot_means = sorted([sum(random.choices(data, k=n)) / n for _ in range(n_boot)])
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return lo, hi

def mean(xs): return sum(xs) / len(xs)
def std(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m)**2 for x in xs) / (len(xs) - 1)) ** 0.5

# Key experiments with fold-level Sharpe data
experiments = {
    "NSE Nifty 100 Expanding": {"folds": [-0.409], "label": "Nifty 100 Expanding", "color": "#d62728"},
    "NSE Nifty 100 Rolling":   {"folds": [0.052], "label": "Nifty 100 Rolling", "color": "#ff7f0e"},
    "NSE Nifty 50 Rolling":    {"folds": [1.127, 0.218, 0.627, 1.036], "label": "Nifty 50 Rolling (Control)", "color": "#2ca02c"},
    "NSE Nifty 50 Expanding":  {"folds": [1.127, 0.233, 1.347, 1.547], "label": "Nifty 50 Expanding (Control)", "color": "#1f77b4"},
    "India ZScore (mean)":     {"folds": [0.398, -0.386, 0.840], "run_means": True, "label": "India ZScore (3-run mean)", "color": "#9467bd"},
    "India ZScore (best)":     {"folds": [0.604, -0.080, 1.996, 0.840], "label": "India ZScore (best run)", "color": "#8c564b"},
    "Brazil OU (mean)":        {"folds": [0.000, 0.000, 0.321], "run_means": True, "label": "Brazil OU (3-run mean)", "color": "#e377c2"},
    "UK ZScore (mean)":        {"folds": [0.265, -0.245], "run_means": True, "label": "UK ZScore (2-run mean)", "color": "#7f7f7f"},
    "US OU (mean)":            {"folds": [0.000, 0.000, -0.254], "run_means": True, "label": "US OU (3-run mean)", "color": "#bcbd22"},
}

results = {}
for key, exp in experiments.items():
    folds = exp["folds"]
    m = mean(folds)
    s = std(folds)
    lo, hi = bootstrap_ci(folds)
    se = s / len(folds) ** 0.5
    results[key] = {
        "label": exp["label"],
        "color": exp["color"],
        "mean": round(m, 3),
        "std": round(s, 3),
        "se": round(se, 3),
        "ci_lo": round(lo, 3),
        "ci_hi": round(hi, 3),
        "n_folds": len(folds),
        "folds": folds,
        "run_means_not_fold_sharpes": exp.get("run_means", False),
    }

out = Path(__file__).parent / "figure_ci_data.json"
out.write_text(json.dumps(results, indent=2))
print(f"Written: {out}")
for k, v in results.items():
    print(f"  {v['label']}: mean={v['mean']:+.3f}, CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]")
