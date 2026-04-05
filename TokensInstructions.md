# Token Efficiency Instructions

Practical guidance for keeping Claude Code sessions cheap and fast in this repo.

---

## 1. Base Context Reduction (Highest free win)

**Already done:** `ENABLE_TOOL_SEARCH` is enabled in `~/.claude/settings.json`. This defers loading tool schemas until needed, cutting base context by ~30-50% per turn.

**Still to do — disable unused MCP servers.** The following are loaded but irrelevant to trading research:
- `claude.ai Canva`
- `claude.ai Figma`
- `claude.ai Gmail`
- `claude.ai Google Calendar`

Go to Claude Code settings and remove these. Each MCP server injects schema tokens on every turn.

**Memory files stay lean.** The project memory at `~/.claude/projects/C--Code-Hybrid-Pairs-Trading-Ensemble/memory/` is currently 125 lines across 5 files — keep it that way. These are pointers to canonical files (`Research.md`, `Plan.md`), not content dumps.

---

## 2. Cache Expiry (Biggest hidden cost)

Claude Code caches the conversation up to the last message. If you go idle for more than 5 minutes, the cache expires and the entire context is re-billed at full cost.

**Rule:** If you step away for > 5 minutes, run `/compact` or `/clear` before your next prompt.

- `/compact` — summarizes and compresses the conversation, preserving continuity
- `/clear` — full reset; re-prompt with a short summary of where you left off

Failing to do this is the single biggest source of unexpected token spikes.

---

## 3. Session Length

Long sessions accumulate cost exponentially because every turn re-bills the full prior context.

**Rule:** Start a new session whenever you switch tasks. Natural breakpoints for this repo:

- Finished an experiment run and reviewed results
- Switching from coding to analysis (or vice versa)
- Coming back after any significant break

When starting fresh, you don't need to re-explain the project — the memory system handles that. Just reference what you were doing: `"continuing E7, full-mode WFV — see Research.md"`.

---

## 4. Avoid Redundant File Reads

Reading the same file multiple times in a session inflates context with duplicate content.

**Pattern to avoid:**
```
Read core/entry.py    # turn 3
Read core/entry.py    # turn 7 (same content injected again)
```

**Better:** Ask Claude to refer back to what it already read. If you need a fresh look, use `/compact` first to compress prior reads out of the active context.

For large output files (experiment CSVs, logs), never dump the raw file. Ask for targeted summaries: `"show me the Net Sharpe column from the latest WFV output"`.

---

## 5. Tool Output Noise

Running scripts via Bash can flood context with verbose output.

**For experiment runs:** Pipe output to a log file and read targeted lines:
```bash
python -m experiments.run_e7 > logs/e7_run.log 2>&1
# then: "show me the final fold summary from logs/e7_run.log"
```

**For log files in this repo:** The `logs/` directory already accumulates run output. Always use targeted reads (`tail -20`, grep for Sharpe/Returns) rather than full reads.

---

## 6. The 80/20 Summary

Do these three things and you capture most of the savings:

1. **ENABLE_TOOL_SEARCH is on** — already configured, no action needed.
2. **Avoid idle > 5 min** — run `/compact` or `/clear` before resuming.
3. **Start new sessions per task** — don't let a session span multiple experiment runs.

**Cost model to keep in mind:**

```
Cost per turn  ≈  (all prior context) × price_per_token
If cache expires:  Cost per turn  ≈  10× normal
```
