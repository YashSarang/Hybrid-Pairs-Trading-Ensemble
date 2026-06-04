# Streamlit Backtesting Engine

This folder contains the interactive backtesting engine built on Streamlit.
It is **separate from the research experiment scripts** in `experiments/`.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app — Simulator, Predictions, Reports pages |
| `literature_data.json` | Paper catalog data for the Literature Review page (planned) |
| `LITERATURE_REVIEW_FEATURE.md` | Spec for the Literature Review page feature |
| `LITERATURE_REVIEW_USAGE.md` | Usage guide for the LR feature |
| `STREAMLIT_ENHANCEMENT_PLAN.md` | Full enhancement roadmap for the Streamlit UI |

## How to run

```bash
cd Implementation/app
streamlit run app.py
```

## Current pages (implemented)
- **Simulator** — Interactive strategy parameter tuning with live backtest
- **Predictions** — ML signal predictions and pair scoring
- **Reports** — Results display from completed experiment JSONs

## Planned pages (see STREAMLIT_ENHANCEMENT_PLAN.md)
- **Literature Review** — Interactive paper browser using `literature_data.json`
- **Experiment Dashboard** — E1-E8 results viewer

## Notes
- App imports core modules from `../core/` — run from `Implementation/app/` or set `PYTHONPATH=..`
- Results are read from `../experiments/results/` (current universe only, not archive)
- `literature_data.json` is NOT auto-generated — it was hand-curated from the literature review
