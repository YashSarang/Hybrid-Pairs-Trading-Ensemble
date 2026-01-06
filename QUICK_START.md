# Quick Start: Report Management Features

## Running Your First Simulation with Reports

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Configure Your Simulation

- **Universe**: Enter NSE tickers (e.g., RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK)
- **Stage 1 Weights**: Adjust pair selection model weights
- **Stage 2 Weights**: Adjust entry/exit model weights
- **Backtest Config**: Set capital, costs, and parameters

### 3. Run Simulation

- Click "Run Simulation" button
- Wait for backtest to complete
- Report is automatically saved!
- Note the Run ID shown in success message

### 4. View Your Report

- Navigate to "Reports" page (sidebar)
- Select your report from dropdown
- View performance metrics
- Expand "Run Parameters" to see configuration

### 5. Compare with Benchmark (Optional)

- Check "Compare with Index"
- Select index (e.g., NIFTY 50)
- Choose Net or Gross returns
- View comparison metrics and chart

### 6. Download Data (Optional)

- Click "Download Trades CSV" for trade log
- Click "Download Equity CSV" for equity curves
- Open in Excel, Python, or your analysis tool

## Example Workflow

### Scenario: Testing Different Pair Selection Strategies

**Run 1: Correlation-Heavy**

1. Set Correlation weight to 0.8
2. Set other Stage 1 weights to 0.05 each
3. Run simulation
4. Note Run ID: `20250131_143022`

**Run 2: Cointegration-Heavy**

1. Set Cointegration weight to 0.8
2. Set other Stage 1 weights to 0.05 each
3. Run simulation
4. Note Run ID: `20250131_143156`

**Compare Results**

1. Go to Reports page
2. View Run 1, note Net Sharpe
3. View Run 2, note Net Sharpe
4. Compare with NIFTY 50 for both
5. Determine which strategy performs better

## Tips for Success

### Universe Selection

- Start with 5-10 liquid stocks
- Use stocks from same sector for better pairs
- Ensure sufficient historical data (2+ years)

### Parameter Tuning

- Start with default weights
- Adjust one parameter at a time
- Save each run to compare results
- Use benchmark comparison to validate

### Report Management

- Delete test runs to save space
- Keep successful strategies for reference
- Download important results before deletion
- Review parameters before reproducing runs

## Common Use Cases

### 1. Strategy Development

- Run multiple configurations
- Compare performance metrics
- Identify best parameter combinations
- Validate against benchmarks

### 2. Performance Monitoring

- Track strategy over time
- Compare different time periods
- Monitor against market indices
- Analyze trade patterns

### 3. Risk Analysis

- Review max drawdown across runs
- Compare volatility metrics
- Analyze tracking error vs benchmarks
- Evaluate cost impact (Gross vs Net)

### 4. Reporting

- Download trades for compliance
- Export equity curves for presentations
- Share Run IDs with team members
- Document parameter choices

## Troubleshooting

### "No saved reports yet"

- Run at least one simulation first
- Check `reports/` directory exists
- Verify write permissions

### "Failed to fetch benchmark data"

- Check internet connection
- Try different index
- Verify date range has data
- Wait and retry (Yahoo Finance rate limits)

### Report not loading

- Verify Run ID is correct
- Check report directory exists
- Ensure all CSV files present
- Try deleting and re-running

### Large file sizes

- Normal for high-frequency strategies
- Delete old reports regularly
- Consider shorter backtest periods
- Archive important reports externally

## Next Steps

1. **Experiment**: Try different universes and parameters
2. **Compare**: Use benchmark comparison to validate strategies
3. **Analyze**: Download data for deeper analysis
4. **Optimize**: Iterate based on results
5. **Deploy**: Use best-performing configurations

## Support

For detailed information:

- **USAGE_GUIDE.md**: Complete feature documentation
- **IMPLEMENTATION_SUMMARY.md**: Technical details
- **reports/README.md**: Report structure reference

Happy trading! 🚀
