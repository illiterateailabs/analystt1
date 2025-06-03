# Documentation Cleanup Plan  
_File: memory-bank/DOCUMENTATION_CLEANUP_PLAN.md_  
_Last updated: 03 Jun 2025 – in scope of PR #64_

---

## 1 ▪ Objective
Bring all project knowledge into **one coherent, up-to-date corpus** and delete stale or duplicated markdown files.  
Canonical docs will live under `memory-bank/` only.

---

## 2 ▪ New Consolidated Structure

| Category | Canonical Doc | Purpose |
|----------|---------------|---------|
| 🟢 Status | `MASTER_STATUS.md` | Single-page project health, phase, backlog, metrics |
| 🟢 Architecture | `TECHNICAL_ARCHITECTURE.md` | Definitive system design & data-flow reference |
| 🟢 Capabilities | `CAPABILITIES_CATALOG.md` | What the platform does today + analyst recipes |
| 🟠 Handoffs / History | `session-handoff-*.md` | Keep **latest three** sessions for traceability |
| 🟠 Research | `research/*.md` | Technical research notes (kept as appendix) |
| 🟢 Cleanup Plan | `DOCUMENTATION_CLEANUP_PLAN.md` | (_this file_) roadmap for retiring legacy docs |

All other docs are considered **legacy** once this plan merges.

---

## 3 ▪ Files Marked for Deletion

| Path | Reason |
|------|--------|
| `memory-bank/progress.md` | Superseded by `MASTER_STATUS.md` |
| `ROADMAP.md` | Content merged into `MASTER_STATUS.md` backlog section |
| `memory-bank/CURRENT_STATUS_AND_GAP_ANALYSIS.md` | Incorporated into status & architecture docs |
| `memory-bank/SYSTEM_ARCHITECTURE_VISUAL.md` | Diagram replaced by §1 in `TECHNICAL_ARCHITECTURE.md` |
| `memory-bank/WORKFLOW_CAPABILITIES_ANALYSIS.md` | Duplicate of new capabilities catalog |
| `memory-bank/CAPABILITIES_QUICK_REFERENCE.md` | Duplicate |
| `IMMEDIATE_ACTION_PLAN.md` | Actions completed (historic) |
| `P0_P1_IMPLEMENTATION_PLAN.md` | Tasks delivered; now tracked in backlog |
| `P0_P1_TIMELINE.md` | Timeline consolidated |
| `CRITICAL_FIXES.md` | Fixes merged; archived via PR descriptions |
| `PHASE1_IMPLEMENTATION_PLAN.md` | Phase complete |
| `PHASE1_DEPENDENCY_ANALYSIS.md` | Obsolete |
| `theplan.md` | Early draft, no longer relevant |
| `CI_DEPENDENCY_FIX.md` and `FIX_GITLAB_CI_DEPENDENCY_CONFLICT.md` | Problem resolved; details recorded in commit history |
| `CI_DEPENDENCY_OPTIMIZATION.md` | Same as above |
| _Older session-handoff files older than 3 most recent_ | Retain only latest 3 for brevity |

_Total slated for removal: 18 files._

---

## 4 ▪ Files to Archive (Read-Only, Keep)

* `memory-bank/research/*.md` – valuable background studies  
* `session-handoff-2025-06-02-integration-fixes.md`, `session-handoff-2025-06-02-gnn.md`, `session-handoff-2025-05-31-critical-fixes.md` – latest three handoffs  
* CONTRIBUTING, LICENSE, TESTING_GUIDE etc. – still current for dev workflow

---

## 5 ▪ Implementation Steps

1. **Merge PR #64** (conflict-resolution branch).  
2. Create branch `docs/cleanup`.  
3. Delete files listed in §3; commit message `docs: remove legacy markdown files (consolidated)`.  
4. Run `docs_linter` CI job to verify no broken intra-repo links.  
5. Update repository README links to point to new canonical docs.  
6. Merge `docs/cleanup` to `main`.

---

## 6 ▪ Verification Checklist

- [ ] `grep -R "progress.md" --exclude=*.py` returns no hits  
- [ ] `mkdocs build` or equivalent passes (if docs site used)  
- [ ] CI green, test coverage unchanged  
- [ ] Links in README & front-matter sections resolve

---

## 7 ▪ Ownership & Timeline

* **Owner:** Marian Stanescu  
* **Planned PR date:** 04 Jun 2025  
* **Target merge:** within 24 h after review

---

### End of Document
