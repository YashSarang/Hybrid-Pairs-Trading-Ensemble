# Usage Guide: Report Management & Benchmark Comparison

## Overview

The Pairs Trading application now includes comprehensive report management with persistent storage and benchmark comparison features.

## Features

### 1. Automatic Report Saving

Every simulation run is automatically saved to disk with complete details:

- **Location**: `reports/<YYYYMMDD_HHMMSS>/`
- **What's saved**:
  - All parameters (universe, weights, backtest config)
  - Performance metrics (Gross & Net)
  - Time series data (equity curves, P&L, turnover)
  - Complete trade log

### 2. Reports Page

Access saved reports through the "Reports" page in the sidebar.

#### Features:

- **Browse all runs**: Sorted by date (newest first)
- **View parameters**: Expandable section showing all configuration details
- **Performance metrics**: Visual display of key metrics
- **Benchmark comparison**: Compare with Indian indices
- **Download data**: Export trades and equity curves as CSV
- **Delete reports**: Remove old runs to save space

### 3. Benchmark Comparison

Compare your strategy returns with major Indian market indices.

#### Available Indices:

- **NIFTY 50** (^NSEI)
- **NIFTY 100** (^CNX100)
- **NIFTY 200** (^CNX200)
- **NIFTY 500** (^CNX500)
- **SENSEX** (^BSESN)
- **NIFTY BANK** (^NSEBANK)
- **NIFTY IT** (^CNXIT)

#### Comparison Metrics:

- **Strategy Return**: Your strategy's total return
- **Benchmark Return**: Index's total return over the same period
- **Excess Return**: Strategy return minus benchmark return
- **Information Ratio**: Risk-adjusted excess return
- **Tracking Error**: Volatility of return differences

#### How to Use:

1. Navigate to the "Reports" page
2. Select a saved report
3. Check "Compare with Index"
4. Select your desired index
5. Choose "Net Returns" or "Gross Returns"
6. View comparison metrics and chart

### 4. Workflow Example

#### Running a Simulation:

1. Configure your universe (Manual, CSV, or Index)
2. Set Stage 1 weights (Pair Selection)
3. Set Stage 2 weights (Entry/Exit)
4. Configure backtest parameters and costs
5. Click "Run Simulation"
6. Report is automatically saved with a unique ID

#### Viewing Reports:

1. Go to "Reports" page
2. Select a report from the dropdown
3. Expand "Run Parameters" to see configuration
4. View performance metrics
5. Enable benchmark comparison if desired
6. Download data or delete report as needed

## Report Structure

Each report directory contains:

```
reports/
└── 20250131_143022/
    ├── metadata.json          # Run configuration
    ├── metrics.json           # Performance metrics
    ├── params.json            # Detailed parameters
    ├── equity_gross.csv       # Gross equity curve
    ├── equity_net.csv         # Net equity curve
    ├── pnl_gross.csv          # Gross P&L
    ├── pnl_net.csv            # Net P&L
    ├── turnover.csv           # Position changes
    └── trades.csv             # Trade log
```

## Tips

### Storage Management

- Reports are saved permanently to disk
- Delete old reports you no longer need
- Each report typically uses 10-100 KB depending on trade count

### Benchmark Comparison

- Requires internet connection to fetch index data
- Data is fetched from Yahoo Finance
- Comparison aligns dates automatically
- Works best with daily frequency data

### Parameter Tracking

- All parameters are saved automatically
- No need to manually record settings
- Easy to reproduce past runs
- Compare different configurations side-by-side

## Troubleshooting

### "Failed to fetch benchmark data"

- Check internet connection
- Verify date range overlaps with index data availability
- Try a different index

### "Report not found"

- Report may have been deleted
- Check `reports/` directory exists
- Verify report ID is correct

### Large report files

- Trade logs can be large for high-frequency strategies
- Consider deleting old reports
- Download important reports before deletion

## Advanced Usage

### Comparing Multiple Runs

1. Run multiple simulations with different parameters
2. Note the Run IDs
3. Open each report and compare metrics
4. Use benchmark comparison to see relative performance

### Exporting for Analysis

1. Select a report
2. Download trades CSV
3. Download equity curves CSV
4. Import into your analysis tool (Excel, Python, R, etc.)

### Reproducing Results

1. Open a saved report
2. View "Run Parameters"
3. Configure simulator with same parameters
4. Run simulation to reproduce results

## Best Practices

1. **Descriptive universes**: Use meaningful ticker lists for easy identification
2. **Regular cleanup**: Delete test runs and failed experiments
3. **Benchmark early**: Compare with indices to validate strategy
4. **Document changes**: Note any manual adjustments in external documentation
5. **Backup important runs**: Copy report directories for critical results
