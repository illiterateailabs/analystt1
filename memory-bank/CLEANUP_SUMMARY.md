# Memory-Bank Cleanup Summary  
**File:** `memory-bank/CLEANUP_SUMMARY.md`  
**Last updated:** 2025-07-07  

This document records the rationalisation of the `memory-bank` directory.  
It explains what was consolidated, what was archived / deleted, and the new ground-rules for documentation going forward.

---

## 1 · Objectives  
1. Remove duplication that had accumulated across status and TODO files.  
2. Provide a single source of truth for current project state and roadmap.  
3. Preserve historical material in an `archive/` sub-folder for traceability.  

---

## 2 · Consolidated Files  

| New File | Replaces | Notes |
|----------|----------|-------|
| **PROJECT_STATUS.md** | `MASTER_STATUS.md`, `IMPLEMENTATION_STATUS_2025-06-23.md`, `STATUS_UPDATE_2025-06-23.md` | Unified, versioned status report. Update this only. |
| **PHASE_2_ROADMAP.md** | `TODO_ROADMAP.md`, `TODO_2025-06-23.md`, `TODO_PERSONAL_PROJECT_FOCUS.md`, `TODO_SPRINT1.md` | Single actionable backlog for Phase 2. |
| **CLEANUP_SUMMARY.md** | _new_ | Records cleanup decisions (this file). |

---

## 3 · Files Moved to `archive/`  

The following documents remain valuable for historical reference but are **no longer authoritative**.  
They were moved verbatim to `memory-bank/archive/`:

```
MASTER_STATUS.md
IMPLEMENTATION_STATUS_2025-06-23.md
STATUS_UPDATE_2025-06-23.md
TODO_ROADMAP.md
TODO_2025-06-23.md
TODO_PERSONAL_PROJECT_FOCUS.md
TODO_SPRINT1.md
```

---

## 4 · Files Deleted (Redundant / Superseded)  

None at this stage. All displaced files were archived rather than deleted to preserve audit history.

---

## 5 · Unaffected Files  

The following documents remain unchanged:

* `CAPABILITIES_CATALOG.md` – feature inventory  
* `TECHNICAL_ARCHITECTURE.md` – system architecture  
* `GRAPH_ANALYSIS_IDEAS.md` – idea backlog  
* Research papers under `memory-bank/research/`  
* Integration & cookbook docs (`ADD_NEW_DATA_SOURCE_COOKBOOK.md`, etc.)  

---

## 6 · New Directory Structure (Effective 2025-07-07)

```
memory-bank/
├── PROJECT_STATUS.md          # Single source of truth for project status
├── PHASE_2_ROADMAP.md         # Canonical TODO / roadmap
├── CLEANUP_SUMMARY.md         # You are here
├── CAPABILITIES_CATALOG.md
├── TECHNICAL_ARCHITECTURE.md
├── GRAPH_ANALYSIS_IDEAS.md
├── ADD_NEW_DATA_SOURCE_COOKBOOK.md
├── SIM_API_INTEGRATION_PLAN.md
├── PHASE_6_INTEGRATION_PLAN.md
├── MODERNIZATION_ACHIEVEMENTS.md
├── research/                  # Research & design spikes
│   └── …
└── archive/                   # Historical, read-only docs
    ├── MASTER_STATUS.md
    ├── IMPLEMENTATION_STATUS_2025-06-23.md
    ├── STATUS_UPDATE_2025-06-23.md
    ├── TODO_ROADMAP.md
    ├── TODO_2025-06-23.md
    ├── TODO_PERSONAL_PROJECT_FOCUS.md
    └── TODO_SPRINT1.md
```

---

## 7 · Ground Rules  

1. **Update, don’t fork:**  
   • Project status → edit `PROJECT_STATUS.md`  
   • Roadmap / backlog → edit `PHASE_2_ROADMAP.md`

2. **Archive, don’t delete:** Outdated docs should be moved to `archive/` with a timestamped name if needed again.

3. **One document per theme:** Avoid creating new status or TODO files; append to the canonical ones instead.

4. **Version & date stamp:** Add `Last updated:` header to every modified memory-bank file.

---

## 8 · Next Steps  

* Verify CI checks referencing old paths – update to new files if required.  
* Inform contributors of the new documentation policy.  
* Schedule quarterly reviews to ensure the memory-bank remains tidy.

---

*End of cleanup summary.*
