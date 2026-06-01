import json, glob

for market_dir in ['uk', 'india', 'brazil', 'us', 'nse_nifty50', 'nse_nifty50_expanding']:
    files = sorted(glob.glob(f'results/{market_dir}/wfv_4folds_zscore*.json'))
    if not files:
        files = sorted(glob.glob(f'results/{market_dir}/*.json'))
    if not files:
        continue
    f = files[-1]
    d = json.load(open(f))
    print(f"=== {market_dir.upper()} === signal={d.get('signal_model','?')}")
    for fold in d.get('folds', []):
        coint = fold.get('selector_scores', {}).get('CointegrationSelector', {})
        n_coint = len([v for v in coint.values() if v > 0]) if coint else 0
        n_total = len(coint) if coint else 0
        pct = f"{n_coint/n_total*100:.0f}%" if n_total else "N/A"
        sharpe = fold['metrics'].get('Net.Sharpe', 0)
        n_pairs = len(fold.get('pairs', []))
        print(f"  Fold {fold['fold']}: coint_pass={n_coint}/{n_total} ({pct}), selected={n_pairs} pairs, sharpe={sharpe:.3f}")
    print()
