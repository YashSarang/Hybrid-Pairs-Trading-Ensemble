# Architecture: Report Management System

## Data Flow

## Component Responsibilities

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

````python
st.success(f"✅ Report saved successfully! Run ID: {run_id}")
st.info("View this report in the 'Reports' page.")
```;

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
````

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
