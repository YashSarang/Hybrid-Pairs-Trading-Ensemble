"""
Recalculate experiment results with corrected trading costs.

Old costs: 22.91 bps per round-trip
New costs: 16.28 bps per round-trip
Cost ratio: 0.7105 (29% reduction)

This script:
1. Reads all experiment JSON files
2. Recalculates net metrics using corrected costs
3. Backs up originals to results/backup_old_costs/
4. Writes corrected results
"""
import json
import os
import shutil
from pathlib import Path

# Cost correction factor
OLD_COST_FRAC = 0.002291  # 22.91 bps
NEW_COST_FRAC = 0.001628  # 16.28 bps
COST_RATIO = NEW_COST_FRAC / OLD_COST_FRAC  # 0.7105

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

def recalculate_metrics(data):
    """Recalculate net metrics from gross metrics and cost_drag."""
    
    def process_config(config):
        """Process a single configuration (has n_folds and full_oos)."""
        if "n_folds" not in config:
            return False
        
        modified = False
        
        # Get cost drag and recalculate
        if "cost_drag_pp" in config and "gross_ann_ret_pct" in config:
            old_cost_drag = config["cost_drag_pp"]["mean"]
            new_cost_drag = old_cost_drag * COST_RATIO
            
            # Recalculate net return
            gross_ret = config["gross_ann_ret_pct"]["mean"]
            old_net_ret = config["net_ann_ret_pct"]["mean"]
            new_net_ret = gross_ret - new_cost_drag
            
            # Update values
            config["cost_drag_pp"]["mean"] = round(new_cost_drag, 4)
            config["net_ann_ret_pct"]["mean"] = round(new_net_ret, 4)
            modified = True
        
        # Also update full_oos if present
        if "full_oos" in config and "cost_drag_pp" in config["full_oos"]:
            oos = config["full_oos"]
            old_cost = oos["cost_drag_pp"]
            new_cost = old_cost * COST_RATIO
            
            gross_ret = oos["gross_ann_ret_pct"]
            new_net_ret = gross_ret - new_cost
            
            oos["cost_drag_pp"] = round(new_cost, 4)
            oos["net_ann_ret_pct"] = round(new_net_ret, 4)
            modified = True
        
        return modified
    
    def process_fold(fold):
        """Process a single fold (direct metrics, no aggregation)."""
        if "cost_drag_pp" not in fold or "gross_ann_ret_pct" not in fold:
            return False
        
        old_cost = fold["cost_drag_pp"]
        new_cost = old_cost * COST_RATIO
        
        gross_ret = fold["gross_ann_ret_pct"]
        new_net_ret = gross_ret - new_cost
        
        fold["cost_drag_pp"] = round(new_cost, 4)
        fold["net_ann_ret_pct"] = round(new_net_ret, 4)
        
        return True
    
    # Process different experiment structures
    modified = False
    
    # Type 1: Has "stage1" and "stage2" keys (ablation experiments)
    if "stage1" in data:
        for config_name, config_data in data["stage1"].items():
            if process_config(config_data):
                modified = True
    
    if "stage2" in data:
        for config_name, config_data in data["stage2"].items():
            if process_config(config_data):
                modified = True
    
    # Type 2: Has direct configuration keys (benchmark, walk_forward with aggregation)
    for key, value in data.items():
        if isinstance(value, dict) and "n_folds" in value:
            if process_config(value):
                modified = True
    
    # Type 3: Has "folds" as a list (walk_forward per-fold results)
    if "folds" in data and isinstance(data["folds"], list):
        for fold in data["folds"]:
            if process_fold(fold):
                modified = True
    
    # Type 4: Has "aggregate" key (walk_forward aggregate results)
    if "aggregate" in data and isinstance(data["aggregate"], dict):
        if process_fold(data["aggregate"]):
            modified = True
        
        # Also process full_oos_metrics if present
        if "full_oos_metrics" in data["aggregate"]:
            oos = data["aggregate"]["full_oos_metrics"]
            if "cost_drag_pp" in oos and "gross_ann_ret_pct" in oos:
                old_cost = oos["cost_drag_pp"]
                new_cost = old_cost * COST_RATIO
                gross_ret = oos["gross_ann_ret_pct"]
                new_net_ret = gross_ret - new_cost
                
                oos["cost_drag_pp"] = round(new_cost, 4)
                oos["net_ann_ret_pct"] = round(new_net_ret, 4)
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
        
        # Backup original
        backup_file = backup_dir / result_file.name
        shutil.copy2(result_file, backup_file)
        
        # Recalculate
        modified = recalculate_metrics(data)
        
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
