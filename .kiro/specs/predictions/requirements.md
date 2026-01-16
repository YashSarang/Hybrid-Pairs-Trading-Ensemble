# Predictions Section Requirements

## Overview

Create a predictions section that provides trading recommendations based on the current strategy configuration and market conditions. This feature will help users identify potential pairs and trades without running full backtests.

## User Stories

### Primary User Stories

**US-1: Real-time Pair Recommendations**

- As a trader, I want to see recommended pairs based on my current strategy settings
- So that I can identify potential trading opportunities without running full backtests
- **Acceptance Criteria:**
  - Display top 10-15 recommended pairs based on current Stage 1 weights
  - Show pair scores and ranking methodology
  - Update recommendations when strategy weights change
  - Include confidence indicators for each recommendation

**US-2: Current Market Signal Analysis**

- As a trader, I want to see current entry/exit signals for recommended pairs
- So that I can make informed trading decisions based on real-time market conditions
- **Acceptance Criteria:**
  - Show current z-scores, OU signals, and Kalman signals for top pairs
  - Display signal strength and direction (long/short/neutral)
  - Include time since last signal change
  - Provide signal confidence levels

**US-3: Forward-Looking Trade Suggestions**

- As a trader, I want specific trade recommendations with entry/exit levels
- So that I can execute trades with clear risk management parameters
- **Acceptance Criteria:**
  - Suggest specific entry prices/levels for recommended pairs
  - Provide stop-loss and take-profit recommendations
  - Include position sizing suggestions based on current capital settings
  - Show expected holding period estimates

### Secondary User Stories

**US-4: Market Regime Analysis**

- As a trader, I want to understand current market conditions
- So that I can adjust my strategy accordingly
- **Acceptance Criteria:**
  - Display market volatility indicators
  - Show correlation regime (high/low correlation environment)
  - Include trend/mean-reversion regime indicators
  - Provide strategy suitability recommendations

**US-5: Risk Assessment**

- As a trader, I want to see risk metrics for recommended trades
- So that I can manage portfolio risk effectively
- **Acceptance Criteria:**
  - Show estimated maximum drawdown for each recommendation
  - Display correlation between recommended pairs
  - Include sector/industry diversification analysis
  - Provide portfolio-level risk metrics

## Technical Requirements

### Data Requirements

**TR-1: Real-time Data Integration**

- Use current market data (latest available prices)
- Implement efficient data fetching for predictions
- Cache recent data to minimize API calls
- Handle market hours and data availability

**TR-2: Historical Context**

- Use configurable lookback periods for signal generation
- Maintain consistency with backtest methodology
- Store intermediate calculations for performance

### Algorithm Requirements

**AR-1: Prediction Engine Architecture**

```
Prediction Engine
├── Market Data Fetcher
├── Pair Scorer (Stage 1 ensemble)
├── Signal Generator (Stage 2 ensemble)
├── Risk Assessor
└── Recommendation Formatter
```

**AR-2: Scoring Methodology**

- Reuse existing Stage 1 selectors with current weights
- Apply real-time data to scoring algorithms
- Rank pairs by ensemble score
- Filter by minimum data requirements

**AR-3: Signal Generation**

- Apply Stage 2 models to top-ranked pairs
- Generate current signal strength and direction
- Calculate confidence intervals
- Estimate signal persistence

### UI Requirements

**UI-1: Predictions Page Layout**

```
Predictions Page
├── Market Overview Panel
├── Top Pairs Recommendations Table
├── Signal Analysis Charts
├── Trade Suggestions Panel
└── Risk Metrics Dashboard
```

**UI-2: Interactive Features**

- Refresh button for latest data
- Configurable number of recommendations (5-20)
- Expandable details for each pair
- Export recommendations to CSV

**UI-3: Visualization Requirements**

- Current spread charts for top pairs
- Signal strength indicators (gauges/bars)
- Risk-return scatter plots
- Time series of recent signals

## Implementation Approach

### Phase 1: Core Prediction Engine

1. Create `core/predictions.py` module
2. Implement `PredictionEngine` class
3. Add real-time data fetching capabilities
4. Integrate with existing Stage 1/2 models

### Phase 2: UI Integration

1. Add "🔮 Predictions" page to main navigation
2. Create prediction dashboard layout
3. Implement real-time data refresh
4. Add export functionality

### Phase 3: Advanced Features

1. Add market regime analysis
2. Implement risk assessment tools
3. Create trade suggestion algorithms
4. Add performance tracking for predictions

## Data Flow

```
User Configuration (Weights, Universe)
    ↓
Market Data Fetch (Latest Prices)
    ↓
Stage 1: Pair Scoring (Top N pairs)
    ↓
Stage 2: Signal Generation (Current signals)
    ↓
Risk Assessment (Portfolio metrics)
    ↓
Recommendation Formatting (UI display)
```

## Success Metrics

### Functional Metrics

- **Accuracy**: Prediction signals align with subsequent backtest results
- **Performance**: Page loads within 3 seconds with fresh data
- **Usability**: Users can understand and act on recommendations
- **Reliability**: System handles market data failures gracefully

### Business Metrics

- **Adoption**: Users regularly check predictions page
- **Actionability**: Recommendations lead to profitable trades
- **Efficiency**: Reduces time from analysis to trade execution

## Risk Considerations

### Technical Risks

- **Data Quality**: Stale or incorrect market data
- **Performance**: Slow real-time calculations
- **Reliability**: API failures during market hours

### Business Risks

- **Over-reliance**: Users may blindly follow recommendations
- **Market Changes**: Predictions may not adapt to regime changes
- **Regulatory**: Providing trading advice implications

## Acceptance Criteria Summary

### Must Have (MVP)

- [ ] Display top 10 pair recommendations based on current strategy
- [ ] Show current signal strength and direction for each pair
- [ ] Provide refresh functionality for latest data
- [ ] Include basic risk metrics (volatility, correlation)
- [ ] Export recommendations to CSV

### Should Have

- [ ] Market regime indicators
- [ ] Trade entry/exit level suggestions
- [ ] Position sizing recommendations
- [ ] Signal confidence intervals
- [ ] Historical prediction performance tracking

### Could Have

- [ ] Real-time alerts for signal changes
- [ ] Integration with broker APIs
- [ ] Advanced risk analytics
- [ ] Machine learning prediction improvements
- [ ] Mobile-responsive design

## Dependencies

### Internal Dependencies

- `core.data`: Market data fetching
- `core.selectors`: Pair selection models
- `core.entry`: Signal generation models
- `core.ensemble`: Weight combination logic
- `app.py`: UI integration

### External Dependencies

- `yfinance`: Real-time market data
- `streamlit`: UI components
- `pandas`: Data manipulation
- `numpy`: Numerical calculations

## Timeline Estimate

- **Phase 1 (Core Engine)**: 2-3 days
- **Phase 2 (UI Integration)**: 1-2 days
- **Phase 3 (Advanced Features)**: 2-3 days
- **Testing & Refinement**: 1-2 days

**Total Estimate**: 6-10 days

## Questions for Clarification

1. **Prediction Horizon**: What time horizon should predictions cover? (intraday, daily, weekly?)
2. **Update Frequency**: How often should predictions refresh? (real-time, hourly, daily?)
3. **Universe Scope**: Should predictions use the same universe as the last backtest or allow separate configuration?
4. **Risk Tolerance**: What level of risk assessment detail is needed?
5. **Integration**: Should predictions integrate with existing reports or be standalone?
6. **Alerts**: Do users want notification capabilities for signal changes?

## Next Steps

1. **User Clarification**: Get answers to the questions above
2. **Technical Design**: Create detailed technical specification
3. **Prototype Development**: Build MVP version for user feedback
4. **Iterative Refinement**: Enhance based on user testing
5. **Production Deployment**: Full feature rollout
