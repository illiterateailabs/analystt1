# Memory Bank – Index & Navigation  
_Last updated: 03 Jun 2025_

Welcome to the **Memory Bank**, the single-source knowledge base for the **Analystt1** project.  
All living documentation is kept here; anything outside this directory is either code, tests, or legacy docs scheduled for removal.

---

## 📖 Core Reference Docs

| File | Purpose |
|------|---------|
| **MASTER_STATUS.md** | One-page project health, roadmap & metrics |
| **TECHNICAL_ARCHITECTURE.md** | Canonical system design & data-flow |
| **CAPABILITIES_CATALOG.md** | What Analystt1 can do today, with usage recipes |
| **SESSION_HISTORY_2025.md** | Consolidated timeline of major 2025 milestones |
| **DOCUMENTATION_CLEANUP_PLAN.md** | Roadmap for retiring legacy markdown files (post-PR #64) |

---

## 📚 Historical Handoff Files  
(kept for traceability – latest three only)

* `session-handoff-2025-06-02-integration-fixes.md`  
* `session-handoff-2025-06-02-gnn.md`  
* `session-handoff-2025-05-31-critical-fixes.md`

Older handoff notes have been archived outside of the default clone depth.

---

## 🔬 Research Notes

All background studies and design spikes live under `research/`:

* `research/crewai-analystagent-factory.md` – CrewAI factory design experiments  
* `research/crewai-analystagent.md` – Early agent prompting strategies  
* `research/crypto-multichain-apis.md` – On-chain data source comparison  
* `research/gemini-llm-provider-design.md` – Gemini integration deep-dive  

_These docs are informative, not authoritative. Always cross-check with the core reference docs._

---

## 🗑️ Legacy Docs

Files like `progress.md`, `ROADMAP.md`, `CURRENT_STATUS_AND_GAP_ANALYSIS.md`, etc. are **deprecated**.  
They will be deleted once **PR #64** merges and the cleanup plan executes.  
Until removal they remain outside the Memory Bank to avoid confusion.

---

## ✍️ Updating the Memory Bank

1. **Edit** the relevant core doc (status, architecture, capabilities).  
2. Commit with message `docs: update <file> – <short summary>`.  
3. No other markdown files should be introduced outside this directory without prior discussion.

---

> “A single source of truth reduces overhead and accelerates coordinated progress.”  
> — Project Principle #1
