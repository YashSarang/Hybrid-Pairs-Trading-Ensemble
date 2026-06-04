# Literature Review Feature - User Guide

## Overview
The Literature Review feature provides an interactive interface to explore all 11 papers from the literature review, with detailed comparisons of claimed results vs. actual NSE implementation results.

## Accessing the Feature

### Step 1: Launch the App
```bash
cd D:/code/Hybrid-Pairs-Trading-Ensemble/Implementation
streamlit run app.py
```

### Step 2: Navigate to Literature Review
- In the left sidebar, you'll see the navigation radio buttons
- Click on **"Literature Review"** (4th option after Simulator, Predictions, Reports)

## Main Literature Review Page

### Overview Dashboard (Top)
Four key metrics displayed:
- **Total Papers**: 11
- **Reproduced**: 8 papers successfully implemented
- **Failed**: 1 paper (PCA-OU) that failed on NSE
- **Year Range**: 1987-2021

### Filtering Section
Three dropdown filters to narrow down papers:

1. **Status Filter**
   - REPRODUCED (8 papers)
   - FAILED (1 paper)
   - DOCUMENTED (1 paper)
   - PLANNED (1 paper)

2. **Category Filter**
   - Classical Statistical (5 papers)
   - Machine Learning (2 papers)
   - Deep Learning (3 papers)
   - Reinforcement Learning (1 paper)

3. **Method Filter**
   - COINTEGRATION
   - DISTANCE
   - OU (Ornstein-Uhlenbeck)
   - PCA-OU
   - ML (Machine Learning)
   - LSTM
   - TRANSFORMER
   - GNN (Graph Neural Network)
   - RL (Reinforcement Learning)
   - MULTI-CRITERIA

### Visualizations (Side by Side)

**Left: Status Distribution** (Bar Chart)
Shows count of papers by status (reproduced, failed, documented, planned)

**Right: Timeline** (Line Chart)
Shows distribution of papers by publication year from 1987 to 2021

### Papers Overview Table
Interactive sortable table showing:
- Year
- Title (truncated to 50 chars)
- Authors (first author + "et al.")
- Method
- Status
- Our Sharpe (NSE implementation result)

### Paper Details Dropdown
- Select any paper from dropdown
- Automatically expands to show full details
- Click "Paper Details & Results Comparison" expander

## Paper Details View

### Metadata Section (Two Columns)

**Left Column:**
- Authors (full names)
- Journal (full citation)
- Category

**Right Column:**
- Year
- Method (uppercase)
- Status (color-coded):
  - REPRODUCED (green)
  - FAILED (red)
  - DOCUMENTED (orange)
  - PLANNED (gray)

### Results Comparison (Two Columns)

**Left: Claimed Results (Original Paper)**
- Shows metrics reported in the original publication
- Market context (US, UK, etc.)
- Time period studied
- Description of methodology

**Right: Our Results (NSE Implementation)**
- Shows actual performance on NSE stocks
- Sharpe ratio achieved
- Trades per year
- Key findings or issues

### Performance Visualization
For papers with numeric results:
- Bar chart comparing Claimed vs NSE performance
- Sharpe Ratio comparison
- Return % comparison (if available)

### Implementation Notes
- Observations from reproduction attempt
- Why method works or fails on NSE
- Integration with core library

## Special Feature: PCA-OU Failure Analysis

When viewing the **Avellaneda & Lee (2010)** paper, a special section appears:

### Critical Negative Result Analysis

**Three Key Metrics:**
- Success Rate: **0%** (delta: -100%)
- Tradeable Stocks: **0 / 35**
- Test Periods: **5 years**

**Error Box (Red):**
Shows the critical finding that PCA-OU achieves 0% success rate on NSE, with explanation that all 35 stocks failed the half-life constraint.

**Info Box (Blue):**
Research implication highlighting that this negative result strengthens the LSTM+Correlation ensemble contribution (Net SR +0.451 vs 0 tradeable opportunities).

**Full Failure Analysis Report (Expandable):**
- Complete NEGATIVE_RESULT.md content
- Emoji-stripped for professional display
- Detailed breakdown of why the method fails
- Comparison tables (US vs NSE)
- Research implications
- Thesis recommendations

## Key Insights Section (Bottom)

### Left Column: Successful Methods on NSE
Top 5 methods ranked by Sharpe ratio:
1. LSTM: Sharpe 0.231 (2018) - Best deep learning method
2. OU: Sharpe 0.145 (2005) - Best classical method
3. MULTI-CRITERIA: Sharpe 0.134 (2021)
4. ENGLE-GRANGER: Sharpe 0.119 (1987)
5. ML: Sharpe 0.112 (2017)

### Right Column: Failed/Problematic Methods
Methods that don't work on NSE:
- PCA-OU: FAILED (2010) - 0% success rate
- TRANSFORMER: REPRODUCED (2021) - Negative Sharpe (-0.045)

## Use Cases

### 1. Research Review
- Quickly see which methods have been implemented
- Compare original claims to actual NSE performance
- Identify gaps (documented/planned papers)

### 2. Method Selection
- Choose proven methods for NSE (LSTM, OU, Multi-criteria)
- Avoid problematic methods (Transformer, PCA-OU)
- Understand why certain methods fail

### 3. Thesis Support
- Document reproduction efforts
- Show critical negative results (PCA-OU)
- Validate novel ensemble approach
- Demonstrate market-specific behavior

### 4. Interactive Exploration
- Filter papers by category or method
- Compare across time periods
- Visualize research timeline
- Export insights for presentations

## Tips

1. **Start with Overview**: Use filters to narrow down to your area of interest
2. **Check Status**: Focus on "reproduced" papers for validated results
3. **Compare Sharpe Ratios**: Sort by "Our Sharpe" to see best performers
4. **Read Failure Analysis**: PCA-OU case study is valuable research contribution
5. **Use Key Insights**: Quick summary of what works/doesn't work on NSE

## Navigation Flow

```
Literature Review (Main Page)
├── Overview Metrics
├── Filters (Status/Category/Method)
├── Visualizations (Charts)
├── Papers Table
├── Paper Selection Dropdown
│   └── Paper Details View
│       ├── Metadata
│       ├── Results Comparison
│       ├── Performance Charts
│       ├── Special Sections (PCA-OU)
│       └── Implementation Notes
└── Key Insights Summary
```

## Integration with Other Pages

### From Literature Review → Simulator
1. Review papers to understand methods
2. Note successful methods (LSTM, OU)
3. Go to Simulator
4. Configure ensemble with proven methods
5. Run backtest with confidence

### From Literature Review → Reports
1. Run simulations with different method weights
2. Go to Reports to analyze results
3. Return to Literature Review to compare
4. Validate against historical reproductions

### From Literature Review → Predictions
1. Understand which models work best
2. Go to Predictions for real-time recommendations
3. Trust predictions knowing models are validated
4. Reference literature for method details

## Professional Features

### No Emojis
All status indicators use Streamlit color markdown:
- `:green[REPRODUCED]` - Successfully implemented
- `:red[FAILED]` - Failed reproduction
- `:orange[DOCUMENTED]` - Documented only
- `:gray[PLANNED]` - Planned for future

### Dark Theme Compatibility
All new components match existing dark theme:
- Background: #0f172a
- Sidebar: #1e293b
- Metrics: #1e293b with #334155 borders
- Charts inherit app theme colors

### Streamlit Components
- `st.dataframe()`: Sortable, searchable tables
- `st.bar_chart()`: Clean bar visualizations
- `st.line_chart()`: Timeline trends
- `st.expander()`: Collapsible sections
- `st.metric()`: Key statistics
- `st.multiselect()`: Multi-criteria filtering
- `st.selectbox()`: Paper selection

## Data Freshness

The feature loads data from:
- `Implementation/literature_data.json` (11 papers)
- `Literature-Review/2010-PCA-OU-Avellaneda-StatArb/NEGATIVE_RESULT.md`

Both files are version-controlled and update with new reproductions.

---

**Ready to Explore?**
Launch the app with `streamlit run app.py` and select "Literature Review" from the sidebar!
