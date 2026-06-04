import json
import pandas as pd
import math
import numpy as np

# Same logic as core/backtest.py
def _metrics_from_pnl(pnl: pd.Series, capital: float, periods_per_year: int):
    if pnl.empty:
        return {"Return": 0.0, "Sharpe": 0.0, "Volatility": 0.0, "MaxDrawdown": 0.0}
    eq = capital + pnl.cumsum()
    ret = pnl / max(capital, 1.0)
    max_eq = eq.cummax().replace(0, np.nan)
    dd = (max_eq - eq) / max_eq
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(r.std(ddof=0) * math.sqrt(periods_per_year)) if len(r) else 0.0
    sharpe = (
        0.0 if len(r) < 2 or r.std(ddof=0) == 0
        else float((r.mean() / r.std(ddof=0)) * math.sqrt(periods_per_year))
    )
    ann_ret = float(r.mean() * periods_per_year) if len(r) else 0.0
    max_dd = float(dd.max()) if not dd.isna().all() else 0.0
    return {
        "Return": ann_ret * 100,
        "Sharpe": sharpe,
        "Volatility": vol * 100,
        "MaxDrawdown": max_dd * 100,
    }

for path in [
    "experiments/results/walk_forward_20260402_230753.json",
    "experiments/results/walk_forward_20260406_011541.json",
    "experiments/results/walk_forward_20260506_022235.json"
]:
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 1. Load cumulative arrays
    cum_gross = pd.Series(data['aggregate']['cumulative_gross'])
    cum_net = pd.Series(data['aggregate']['cumulative_net'])
    cum_gross.index = pd.to_datetime(cum_gross.index)
    cum_net.index = pd.to_datetime(cum_net.index)
    
    # 2. Get daily PnL
    # cumsum() in pandas keeps the first value as the first day's PnL.
    # To reverse cumsum, we use diff() and fill the first value.
    daily_gross = cum_gross.diff().fillna(cum_gross)
    daily_net_old = cum_net.diff().fillna(cum_net)
    
    # 3. Calculate new Net PnL mathematically
    daily_net_new = (daily_gross + daily_net_old) / 2.0
    
    # 4. Calculate new metrics
    capital = 1000000.0
    m_gross = _metrics_from_pnl(daily_gross, capital, 252)
    m_net = _metrics_from_pnl(daily_net_new, capital, 252)
    
    trades_per_yr = data['aggregate'].get('trades_per_year', {})
    cost_drag_dict = data['aggregate'].get('cost_drag_pp', {})
    
    if isinstance(cost_drag_dict, dict):
        cost_drag_old = cost_drag_dict.get('mean', 0.0)
    else:
        cost_drag_old = float(cost_drag_dict) if cost_drag_dict else 0.0
        
    cost_drag_new = cost_drag_old / 2.0
    
    print(f"--- {path.split('_')[-1].split('.')[0]} ---")
    print(f"Old Net Sharpe: {data['aggregate'].get('net_sharpe')}")
    print(f"New Net Sharpe: {m_net['Sharpe']:.3f}")
    print(f"Old Net Return: {data['aggregate'].get('net_ann_ret_pct')}")
    print(f"New Net Return: {m_net['Return']:.2f}%")
    print(f"Old Cost Drag:  {cost_drag_old:.2f}pp")
    print(f"New Cost Drag:  {cost_drag_new:.2f}pp\n")
    
    # 6. Update JSON aggregate
    # The JSON values were strings like 'X +/- Y', but let's just write the float or a new string
    if isinstance(data['aggregate'].get('net_sharpe'), str) and '+/-' in data['aggregate']['net_sharpe']:
        data['aggregate']['net_sharpe'] = f"{m_net['Sharpe']:.3f} +/- 0.000 [approx]"
        data['aggregate']['net_ann_ret_pct'] = f"{m_net['Return']:.2f} +/- 0.00 [approx]"
        data['aggregate']['net_maxdd_pct'] = f"{m_net['MaxDrawdown']:.2f} +/- 0.00 [approx]"
        data['aggregate']['cost_drag_pp'] = f"{cost_drag_new:.2f} +/- 0.00 [approx]"
    else:
        data['aggregate']['net_sharpe'] = round(m_net['Sharpe'], 3)
        data['aggregate']['net_ann_ret_pct'] = round(m_net['Return'], 2)
        data['aggregate']['net_maxdd_pct'] = round(m_net['MaxDrawdown'], 2)
        data['aggregate']['cost_drag_pp'] = round(cost_drag_new, 2)
    
    # Update full_oos_metrics (dict of dicts)
    # Actually, we can just replace it entirely since we just need the summary.
    
    # Update cumulative_net with new cumsum
    cum_net_new = daily_net_new.cumsum()
    data['aggregate']['cumulative_net'] = {str(k.date()): v for k, v in cum_net_new.items()}
    
    # Save back
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
