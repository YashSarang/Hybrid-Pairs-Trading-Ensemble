# Architecture: Report Management System

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐              ┌──────────────┐           │
│  │  Simulator   │              │   Reports    │           │
│  │     Page     │              │     Page     │           │
│  └──────┬───────┘              └──────┬───────┘           │
│         │                              │                   │
│         │ Run Simulation               │ View/Compare      │
│         │                              │                   │
└─────────┼──────────────────────────────┼───────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│              core/reports.py (Report Management)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  ReportManager   │         │ BenchmarkComparison│       │
│  ├──────────────────┤         ├──────────────────┤        │
│  │ • save_report()  │         │ • fetch_index()  │        │
│  │ • load_report()  │         │ • compare()      │        │
│  │ • list_reports() │         │                  │        │
│  │ • delete_report()│         │                  │        │
│  └────────┬─────────┘         └────────┬─────────┘        │
│           │                            │                   │
└───────────┼────────────────────────────┼───────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐    ┌──────────────────────┐
│   File System         │    │   Yahoo Finance      │
│   (reports/)          │    │   (yfinance)         │
├───────────────────────┤    ├──────────────────────┤
│ • metadata.json       │    │ • NIFTY 50           │
│ • metrics.json        │    │ • SENSEX             │
│ • params.json         │    │ • NIFTY BANK         │
│ • equity_gross.csv    │    │ • etc.               │
│ • equity_net.csv      │    │                      │
│ • pnl_gross.csv       │    │                      │
│ • pnl_net.csv         │    │                      │
│ • turnover.csv        │    │                      │
│ • trades.csv          │    │                      │
└───────────────────────┘    └──────────────────────┘
```

## Data Flow

### 1. Simulation Run Flow

```
User Configures Parameters
         │
         ▼
User Clicks "Run Simulation"
         │
         ▼
Backtest Engine Executes
         │
         ▼
BacktestResult Generated
         │
         ▼
ReportManager.save_report()
         │
         ├─► Create unique Run ID (timestamp)
         ├─► Create directory: reports/<run_id>/
         ├─► Save metadata.json
         ├─► Save metrics.json
         ├─► Save params.json
         ├─► Save equity_gross.csv
         ├─► Save equity_net.csv
         ├─► Save pnl_gross.csv
         ├─► Save pnl_net.csv
         ├─► Save turnover.csv
         └─► Save trades.csv
         │
         ▼
Success Message Displayed
```

### 2. Report Viewing Flow

```
User Opens Reports Page
         │
         ▼
ReportManager.list_reports()
         │
         ├─► Scan reports/ directory
         ├─► Read metadata.json from each
         └─► Return sorted list
         │
         ▼
User Selects Report
         │
         ▼
ReportManager.load_report(run_id)
         │
         ├─► Load metadata.json
         ├─► Load metrics.json
         ├─► Load params.json
         ├─► Load equity_gross.csv
         ├─► Load equity_net.csv
         ├─► Load pnl_gross.csv
         ├─► Load pnl_net.csv
         ├─► Load turnover.csv
         └─► Load trades.csv
         │
         ▼
Display Report in UI
```

### 3. Benchmark Comparison Flow

```
User Enables "Compare with Index"
         │
         ▼
User Selects Index (e.g., NIFTY 50)
         │
         ▼
BenchmarkComparison.fetch_index_returns()
         │
         ├─► Get date range from strategy
         ├─► Fetch index data from Yahoo Finance
         └─► Calculate cumulative returns
         │
         ▼
BenchmarkComparison.compare_with_benchmark()
         │
         ├─► Align dates
         ├─► Calculate excess returns
         ├─► Calculate information ratio
         ├─► Calculate tracking error
         └─► Prepare comparison data
         │
         ▼
Display Comparison Metrics & Chart
```

## Component Responsibilities

### app.py (UI Layer)

**Responsibilities:**

- User interface rendering
- User input collection
- Orchestrating workflow
- Displaying results

**Key Functions:**

- `simulator_page()`: Main simulation interface
- `render_reports_page()`: Report viewing interface
- `get_report_manager()`: Singleton access to ReportManager

### core/reports.py (Business Logic)

**Responsibilities:**

- Report persistence
- Report retrieval
- Benchmark data fetching
- Performance comparison

**Key Classes:**

- `ReportManager`: CRUD operations for reports
- `BenchmarkComparison`: Index comparison logic
- `ReportMetadata`: Data structure for metadata

### File System (Storage Layer)

**Responsibilities:**

- Persistent storage
- Data organization
- File management

**Structure:**

```
reports/
├── 20250131_143022/
│   ├── metadata.json
│   ├── metrics.json
│   ├── params.json
│   ├── equity_gross.csv
│   ├── equity_net.csv
│   ├── pnl_gross.csv
│   ├── pnl_net.csv
│   ├── turnover.csv
│   └── trades.csv
├── 20250131_145633/
│   └── ...
└── README.md
```

## Design Patterns Used

### 1. Singleton Pattern

```python
def get_report_manager():
    if "report_manager" not in st.session_state:
        st.session_state["report_manager"] = ReportManager()
    return st.session_state["report_manager"]
```

**Purpose:** Single instance of ReportManager across app

### 2. Data Class Pattern

```python
@dataclass
class ReportMetadata:
    run_id: str
    timestamp: str
    universe: List[str]
    # ...
```

**Purpose:** Structured data with type hints

### 3. Repository Pattern

```python
class ReportManager:
    def save_report(...)
    def load_report(...)
    def list_reports(...)
    def delete_report(...)
```

**Purpose:** Abstraction over data storage

### 4. Strategy Pattern

```python
class BenchmarkComparison:
    INDIAN_INDICES = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        # ...
    }
```

**Purpose:** Flexible benchmark selection

## Error Handling Strategy

### Graceful Degradation

```python
try:
    comparison = BenchmarkComparison.compare_with_benchmark(...)
except Exception as e:
    st.error(f"Failed to fetch benchmark data: {e}")
```

### Validation

```python
if not run_dir.exists():
    raise ValueError(f"Report {run_id} not found")
```

### User Feedback

```python
st.success(f"✅ Report saved successfully! Run ID: {run_id}")
st.info("View this report in the 'Reports' page.")
```

## Performance Considerations

### Lazy Loading

- Reports loaded only when selected
- Benchmark data fetched only when enabled
- Trade logs displayed with pagination

### Efficient Storage

- JSON for metadata (human-readable, small)
- CSV for time series (efficient, portable)
- Separate files for different data types

### Caching

- ReportManager singleton cached in session state
- Report list cached until refresh
- Benchmark data could be cached (future enhancement)

## Security Considerations

### Input Validation

- Run IDs validated before file operations
- Directory traversal prevented
- File existence checked before operations

### Data Isolation

- Each report in separate directory
- No cross-report data access
- Clean separation of concerns

### Error Messages

- No sensitive information exposed
- User-friendly error messages
- Detailed logging for debugging

## Scalability

### Current Limits

- **Reports**: Unlimited (disk space dependent)
- **Report Size**: ~10-100 KB typical
- **Load Time**: < 1 second per report
- **Concurrent Users**: Streamlit session-based

### Future Scaling Options

- Database backend (PostgreSQL, MongoDB)
- Cloud storage (S3, Azure Blob)
- Caching layer (Redis)
- API layer for multi-user access

## Testing Strategy

### Unit Tests (Recommended)

```python
def test_save_report():
    mgr = ReportManager()
    run_id = mgr.save_report(...)
    assert run_id is not None
    assert Path(f"reports/{run_id}").exists()

def test_load_report():
    mgr = ReportManager()
    report = mgr.load_report(run_id)
    assert "metadata" in report
    assert "metrics" in report

def test_benchmark_comparison():
    comparison = BenchmarkComparison.compare_with_benchmark(...)
    assert "excess_return" in comparison
    assert "information_ratio" in comparison
```

### Integration Tests (Recommended)

- End-to-end simulation run
- Report save and load cycle
- Benchmark comparison with real data
- UI interaction testing

## Monitoring & Observability

### Metrics to Track

- Number of reports saved
- Report load times
- Benchmark fetch success rate
- Storage usage
- User interactions

### Logging Points

- Report save operations
- Report load operations
- Benchmark fetch attempts
- Error occurrences
- User actions

## Maintenance

### Regular Tasks

- Clean up old reports
- Monitor disk usage
- Update benchmark indices
- Review error logs
- Update documentation

### Backup Strategy

- Regular backup of reports/ directory
- Version control for code
- Documentation updates
- Configuration backups

## Conclusion

The report management system follows clean architecture principles with:

- ✅ Clear separation of concerns
- ✅ Modular design
- ✅ Extensible structure
- ✅ Robust error handling
- ✅ Efficient storage
- ✅ User-friendly interface

Ready for production use with room for future enhancements.
