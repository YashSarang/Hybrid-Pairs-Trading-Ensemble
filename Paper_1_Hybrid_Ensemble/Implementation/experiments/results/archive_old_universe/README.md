# ARCHIVED — Old Universe Results

These 46 result files are from the OLD experiment setup and must NOT be used for any paper claims.

## Why archived
- Universe: 35 NSE stocks (current = 89 Nifty100 tickers)
- Date range: 2016-2026 (current = 2015-2024)
- WFV folds: test years 2020-2025 (current = 2018-2024)
- Transaction costs: mix of 22.9 bps and partially-corrected 16.3 bps runs
- Some runs had BUG-01 through BUG-08 unfixed

## Key numbers from old universe (DO NOT CITE)
- Config C (LSTM+Correlation, OU): Net SR 0.510, CAGR 17.66%, MaxDD 3.78%
- stat_only Net SR: 0.436
- Equal-weight full ensemble: Net SR -0.660
- Ablation LSTM best: +0.341, GNN worst: -0.448

## Current valid results
Results from the 89-ticker, 2015-2024, 16.28 bps, CPU-only setup are stored in
`experiments/results/` (root level) as they come in from Kalpana jobs 8701+.
