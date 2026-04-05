# Knowledge Graph — Maintenance Guide

The "knowledge graph" for this repo is the Claude Code memory system at:
```
~/.claude/projects/C--Code-Hybrid-Pairs-Trading-Ensemble/memory/
```

It is a set of structured markdown files that Claude loads at the start of every session, replacing the need to re-read large project files from scratch. This is the equivalent of "build the graph once, query it cheaply forever."

---

## Current Graph State (as of 2026-04-06)

| Memory File | What It Covers | Size |
|---|---|---|
| `project_goal.md` | Thesis goals, academic rigour requirements | 11 lines |
| `user_profile.md` | Role, domain knowledge, collaboration style | 7 lines |
| `research_findings.md` | All experiment results (E1-E6), key bugs fixed, thesis implications | 82 lines |
| `project_plan.md` | Phase roadmap, experiment priority order, current blocker | 17 lines |
| `MEMORY.md` | Index / pointer to all of the above | 8 lines |

**Total: 125 lines.** This is the right size — lean pointers + key facts, not full content dumps.

The canonical documents these memories point to:
- `Implementation/Research.md` — full experiment log (source of truth for results)
- `Implementation/Plan.md` — full thesis roadmap and experiment specs

---

## When to Update the Graph (Regraph Triggers)

Update the relevant memory file immediately after any of these events:

### High Priority (update within the same session)

| Event | Which file to update |
|---|---|
| Experiment completes and results are reviewed | `research_findings.md` |
| A major bug is found and fixed (especially one that changes results) | `research_findings.md` |
| The priority order or current blocker changes | `project_plan.md` |
| A phase is completed or a new phase starts | `project_plan.md` |

### Medium Priority (update at end of session)

| Event | Which file to update |
|---|---|
| New selector, signal model, or ensemble variant added | `project_plan.md` |
| Thesis framing or conclusions evolve based on findings | `project_goal.md` |
| You give Claude feedback that should persist across sessions | Create a `feedback_*.md` entry |

### Do Not Update For

- In-progress work or partial results (wait until the experiment completes)
- Code-level details that can be derived by reading the files directly
- Anything already documented in `CLAUDE.md`

---

## How to Update

To update a memory file, instruct Claude in-session:

```
"Update the research_findings memory with these E7 results: ..."
```

Or do it yourself by editing the file directly at:
```
~/.claude/projects/C--Code-Hybrid-Pairs-Trading-Ensemble/memory/<file>.md
```

To add a new memory (e.g. feedback), ask Claude:
```
"Remember that [X] — save this as feedback"
```

Claude will create a new file and add it to `MEMORY.md`.

---

## Why This Works (vs re-reading files)

Without memory:
- Every new session → re-read `Plan.md`, `Research.md`, `core/*.py` to get context
- ~3,000-8,000 tokens of re-reading per session

With memory:
- Session starts with 125 lines already loaded (~1,000 tokens)
- Claude knows the experiment history, current blocker, and thesis goals
- Only reads specific files when the task actually requires it

The savings compound across sessions. A project at this stage (E5+ complete, entering full-mode runs) has enough accumulated context that the memory system provides substantial value on every session.

---

## Graph Health Check

Run this check at the start of any new phase or after a long break:

1. Is `research_findings.md` current with the last completed experiment?
2. Does `project_plan.md` reflect the actual current priority?
3. Are there stale facts (e.g. "Full-mode WFV running" — is it still running or done)?

If any answer is no, update the relevant file before starting new work.
