Here are the exact commands. Run these from the repo root on the Linux server, in separate tmux or screen sessions so they don't die if  
 your SSH drops.

---

Setup (once)

cd /path/to/Hybrid-Pairs-Trading-Ensemble
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

---

E4 — Walk-Forward Validation (most important)

# Full mode: all 8 selectors, ou_only signal (empirically best from stat_only run)

python experiments/walk_forward.py --mode full --s2 ou_only

# If you also want the full 4-model signal ensemble:

python experiments/walk_forward.py --mode full --s2 all

Run ou_only first — it's the headline result and faster. The all variant is a secondary comparison.

---

E3 — Ablation

# stat_ml: adds XGBoost selector to the 4 classical ones

python experiments/ablation.py --mode stat_ml

# full: all 8 selectors (LSTM, Transformer, GNN) — run this after stat_ml

python experiments/ablation.py --mode full

If you're short on time, --stage 1 or --stage 2 runs only the pair-selector or signal-model half.

---

E1 — Frequency Comparison

# full mode, daily vs hourly

python experiments/freq_comparison.py --mode full --freqs 1D 1H

---

Recommended order & parallelism

E4 full is the longest and the headline number — start it first in its own session. E3 and E1 can run concurrently in separate sessions  
 since they write to separate result files with timestamps.

# Session 1 (most critical)

tmux new -s e4
python experiments/walk_forward.py --mode full --s2 ou_only

# Session 2

tmux new -s e3
python experiments/ablation.py --mode stat_ml && python experiments/ablation.py --mode full

# Session 3

tmux new -s e1
python experiments/freq_comparison.py --mode full

Results land in experiments/results/ as timestamped JSON files.
