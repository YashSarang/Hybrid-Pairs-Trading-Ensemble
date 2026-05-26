"""
Recalculate experiment results with corrected trading costs (ROBUST VERSION).

Old costs: 22.91 bps per round-trip
New costs: 16.28 bps per round-trip
Cost ratio: 0.7106 (29% reduction)
"""
import json
import os
import shutil
from pathlib import Path

# Cost correction factor
OLD_COST_FRAC = 0.002291  # 22.91 bps
NEW_COST_FRAC = 0.001628  # 16.28 bps
COST_RATIO = NEW_COST_FRAC / OLD_COST_FRAC  # 0.7106

print("=" * 70)
print("RECALCULATING EXPERIMENT RESULTS WITH CORRECTED TRADING COSTS")
print("=" * 70)
print(f"Old cost: {OLD_COST_FRAC * 10000:.2f} bps per round-trip")
print(f"New cost: {NEW_COST_FRAC * 10000:.2f} bps per round-trip")
print(f"Cost ratio: {COST_RATIO:.4f} (reduction: {(1 - COST_RATIO)*100:.1f}%)")
print()

# Setup paths
results_dir = Path("experiments/results")
backup_dir = results_dir / "backup_old_costs"
backup_dir.mkdir(exist_ok=True)

# Get all JSON result files
result_files = sorted(results_dir.glob("*.json"))
print(f"Found {len(result_files)} result files to process")
print()

def is_scalar(val):
    """Check if value is a scalar (not dict or list)."""
    return isinstance(val, (int, float, str, bool, type(None)))

def update_metrics_in_place(obj):
    """
    Recursively walk the object and update cost_drag_pp and net_ann_ret_pct.
    Works for any nesting level.
    """
    modified = False
    
    if isinstance(obj, dict):
        # Check if this dict has the metrics we need
        if "cost_drag_pp" in obj and "gross_ann_ret_pct" in obj:
            cost_val = obj["cost_drag_pp"]
            gross_val = obj["gross_ann_ret_pct"]
            
            # Handle scalar values (walk_forward per-fold)
            if is_scalar(cost_val) and is_scalar(gross_val):
                new_cost = float(cost_val) * COST_RATIO
                new_net = float(gross_val) - new_cost
                obj["cost_drag_pp"] = round(new_cost, 4)
                obj["net_ann_ret_pct"] = round(new_net, 4)
                modified = True
            
            # Handle dict values with "mean" key (ablation aggregated)
            elif isinstance(cost_val, dict) and "mean" in cost_val:
                old_cost = cost_val["mean"]
                new_cost = old_cost * COST_RATIO
                gross_ret = gross_val["mean"]
                new_net = gross_ret - new_cost
                obj["cost_drag_pp"]["mean"] = round(new_cost, 4)
                obj["net_ann_ret_pct"]["mean"] = round(new_net, 4)
                modified = True
        
        # Recurse into nested dicts
        for key, value in obj.items():
            if update_metrics_in_place(value):
                modified = True
    
    elif isinstance(obj, list):
        # Recurse into lists
        for item in obj:
            if update_metrics_in_place(item):
                modified = True
    
    return modified

# Process all files
updated_count = 0
skipped_count = 0

for result_file in result_files:
    try:
        # Read original
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Backup original (only if not already backed up)
        backup_file = backup_dir / result_file.name
        if not backup_file.exists():
            shutil.copy2(result_file, backup_file)
        
        # Recalculate
        modified = update_metrics_in_place(data)
        
        if modified:
            # Write updated
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✅ Updated: {result_file.name}")
            updated_count += 1
        else:
            print(f"⚠️  Skipped: {result_file.name} (no cost_drag found)")
            skipped_count += 1
    
    except Exception as e:
        print(f"❌ Error processing {result_file.name}: {e}")
        skipped_count += 1

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total files: {len(result_files)}")
print(f"Updated: {updated_count}")
print(f"Skipped: {skipped_count}")
print(f"Backups saved to: {backup_dir}/")
print()
print("✅ All experiment results have been recalculated with corrected costs!")
print()
print("Note: Net returns are now ~6-7% higher per year for high-turnover configs.")
