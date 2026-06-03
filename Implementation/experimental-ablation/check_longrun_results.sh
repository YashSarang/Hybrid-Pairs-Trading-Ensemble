#!/usr/bin/env bash
# Monitor SLURM job 8650 on kalpana and compute stats when done

SSH_HOST="yash.sarang@kalpana.minds.iitb.ac.in"
RESULTS_DIR="~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50_longrun"

# Check if job is still running
JOB_STATUS=$(ssh "$SSH_HOST" "squeue --job 8650 --noheader 2>/dev/null")

if [ -n "$JOB_STATUS" ]; then
    echo "Job 8650 still running"
    exit 0
fi

echo "Job 8650 completed. Collecting results..."

# Find the latest JSON file and compute stats
ssh "$SSH_HOST" python3 << 'PYEOF'
import os, json, glob, sys
import numpy as np
from scipy import stats

results_dir = os.path.expanduser(
    "~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50_longrun"
)

json_files = glob.glob(os.path.join(results_dir, "*.json"))
if not json_files:
    print("ERROR: No JSON files found in", results_dir)
    sys.exit(1)

latest = max(json_files, key=os.path.getmtime)
print(f"Latest JSON: {latest}")

with open(latest) as f:
    data = json.load(f)

# Extract Net.Sharpe for all folds
sharpe_values = []
for key, val in data.items():
    if isinstance(val, dict):
        # Try common key patterns
        for sk in ["Net.Sharpe", "net_sharpe", "sharpe", "Sharpe"]:
            if sk in val:
                sharpe_values.append(float(val[sk]))
                break

if not sharpe_values:
    # Try flat structure
    if "Net.Sharpe" in data:
        sharpe_values = [float(v) for v in data["Net.Sharpe"]] if isinstance(data["Net.Sharpe"], list) else [float(data["Net.Sharpe"])]

if not sharpe_values:
    print("ERROR: Could not find Net.Sharpe values. JSON keys:", list(data.keys())[:10])
    print("Full JSON structure (first 500 chars):", str(data)[:500])
    sys.exit(1)

x = np.array(sharpe_values)
n = len(x)
mean = x.mean()
std = x.std(ddof=1)

# t-test (H0: mean=0)
t_stat, p_val = stats.ttest_1samp(x, 0)

# HAC Newey-West t-stat
try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    ones = np.ones((n, 1))
    model = sm.OLS(x, ones).fit()
    hac_cov = cov_hac(model, nlags=int(np.floor(4 * (n/100)**(2/9))))
    hac_se = np.sqrt(hac_cov[0, 0])
    hac_t = mean / hac_se
    from scipy.stats import t as t_dist
    hac_p = 2 * t_dist.sf(abs(hac_t), df=n-1)
except Exception as e:
    hac_t, hac_p = float('nan'), float('nan')
    print(f"HAC warning: {e}")

# 95% Bootstrap CI
np.random.seed(42)
boot_means = [np.random.choice(x, size=n, replace=True).mean() for _ in range(10000)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

pos_folds = int((x > 0).sum())

print(f"\n{'='*50}")
print(f"SLURM Job 8650 — NSE Nifty50 Long-Run Results")
print(f"{'='*50}")
print(f"File: {os.path.basename(latest)}")
print(f"Folds: {n}")
print(f"Net.Sharpe values: {x.tolist()}")
print(f"\n--- Statistics ---")
print(f"Mean:              {mean:.4f}")
print(f"Std:               {std:.4f}")
print(f"t-stat:            {t_stat:.4f}")
print(f"p-value:           {p_val:.4f}")
print(f"HAC NW t-stat:     {hac_t:.4f}")
print(f"HAC NW p-value:    {hac_p:.4f}")
print(f"95% Bootstrap CI:  [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Positive folds:    {pos_folds}/{n} ({100*pos_folds/n:.1f}%)")
print(f"{'='*50}")
PYEOF
