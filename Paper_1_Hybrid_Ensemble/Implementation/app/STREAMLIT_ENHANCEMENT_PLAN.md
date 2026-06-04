# Streamlit App Enhancement Plan

**Date:** 2026-05-26  
**Goal:** Add Literature Review pages with paper analysis, maintain all existing features

---

## Current App Structure

### Existing Pages (3)
1. **Simulator** — Main backtesting interface
2. **Predictions** — Real-time predictions
3. **Reports** — Saved evaluation results

### Existing Features to Preserve
- All selector models (8 types)
- All signal models (4 types)
- Cost configuration (4 presets)
- NSE stock reference dropdown
- Report saving/loading
- Benchmark comparison
- All visualizations

---

## Proposed Enhancement

### New Pages (2)

#### 1. Literature Review (Overview)
**Purpose:** Summary catalog of all pairs trading papers

**Content:**
- Chronological timeline of papers (1987-2021)
- Status badges (Reproduced / In Progress / Planned)
- Key findings summary table
- Links to individual paper analyses
- NSE vs US cost comparison

**Layout:**
- Filter by status/year/method type
- Sortable table
- Click paper → navigate to detail page

#### 2. Paper Analysis (Detail)
**Purpose:** Deep dive into individual paper reproduction

**Content per paper:**
- Paper metadata (authors, year, journal, citations)
- Methodology summary
- Claimed results (US data)
- Our reproduction results (NSE data)
- Side-by-side comparison
- Key findings / lessons learned
- Implementation status

**Special Feature for PCA-OU (only fully reproduced paper):**
- Interactive parameter sweep
- Live reproduction run
- Diagnostic plots (ADF p-values, half-life distribution)

---

## Implementation Approach

### Step 1: Extract Literature Data
Parse `Literature-Review/README.md` into structured JSON:
```json
{
  "papers": [
    {
      "id": "engle-granger-1987",
      "year": 1987,
      "title": "...",
      "authors": "...",
      "status": "reproduced",
      "method": "cointegration",
      "claimed_results": {...},
      "our_results": {...},
      "folder": "1987-Statistical-EngleGranger-Cointegration"
    }
  ]
}
```

### Step 2: Create Literature Review Page
- `render_literature_page()` function
- Display table with filtering
- Link to detail pages

### Step 3: Create Paper Detail Pages
- `render_paper_detail(paper_id)` function
- Load paper data from JSON
- Show comparison visualizations
- Link to reproduction code

### Step 4: Add Navigation
Update sidebar radio to:
```python
["Simulator", "Predictions", "Reports", "Literature Review"]
```

### Step 5: Preserve All Existing Features
- No changes to existing page logic
- Only add new navigation option
- Keep all imports and helpers

---

## Data Requirements

### Files to Create
1. `literature_data.json` — Structured paper catalog
2. Helper functions in `core/literature.py` (optional)

### Files to Parse
1. `Literature-Review/README.md` — Main catalog
2. `Literature-Review/*/README.md` — Individual paper details
3. `Literature-Review/*/NEGATIVE_RESULT.md` — Failure analysis

---

## Visualization Components

### Overview Page
- Timeline chart (year vs papers)
- Status pie chart
- Method type distribution
- Performance comparison table

### Detail Page
- Results comparison chart (claimed vs actual)
- Parameter sensitivity plots (for PCA-OU)
- Diagnostic plots (ADF, half-life, etc.)

---

## Testing Strategy

1. Verify all existing pages still work
2. Test navigation between pages
3. Test filter/sort on literature page
4. Test paper detail loading
5. Verify no breaking changes to simulator

---

## Rollout Plan

### Phase 1: Data Extraction (10 min)
- Parse README.md
- Create literature_data.json

### Phase 2: Overview Page (15 min)
- Create render_literature_page()
- Add filtering/sorting
- Add navigation link

### Phase 3: Detail Page (15 min)
- Create render_paper_detail()
- Add visualizations
- Link to code

### Phase 4: Testing (10 min)
- Smoke test all pages
- Verify existing features
- Fix any issues

**Total Estimated Time:** 50 minutes

---

## Success Criteria

1. All existing features work unchanged
2. New pages load without errors
3. Literature data displays correctly
4. Navigation is intuitive
5. Code is clean and maintainable
6. Professional formatting (no emojis)
