# SYSTEM_INTEGRATION_GAPS.md  
_Closing the Loop – From Disjointed Modules to a Cohesive Platform_  
_Last updated : 02 Jun 2025_

---

## 1 · Executive Summary

Although Analystt1’s components (FastAPI backend, CrewAI engine, Next.js UI, Neo4j graph, Redis, Gemini) functioned independently, six **critical wiring gaps** prevented the product from operating end-to-end. This document captures each gap, the technical root cause, and the fix delivered in PR #63.

| # | Gap (Pre-Fix) | Impact | Status |
|---|---------------|--------|--------|
| 1 | **Template Creation ➜ Execution** not possible | Analysts could design templates but could not run them | ✅ Hot-reload & runnable |
| 2 | **Tool Result Propagation** lost between agents | ML/Code outputs (e.g., CodeGenTool) never reached downstream tasks | ✅ Shared context layer |
| 3 | **Frontend Results Display** missing | Investigation outputs invisible to users | ✅ `/analysis/[taskId]` UI + API |
| 4 | **PolicyDocsTool** heuristics only | Compliance checks unreliable | ✅ Redis-Vector RAG w/ Gemini |
| 5 | **Task / Result APIs** absent | Dashboard couldn’t list or fetch executions | ✅ `/crew/tasks` & `/crew/{id}/result` |
| 6 | **Security Enforcement Gaps** | Risk of unauthorised crew runs | ✅ RBAC verified & hardened |

---

## 2 · Detailed Gap Analysis & Fixes

### 2.1 Template Creation ↔ Crew Execution
* **Root Cause** : Crews were loaded at startup from YAML; newly created templates weren’t discovered until a full restart.  
* **Fix** :  
  * Added `CrewFactory.reload()` that rescans `backend/agents/configs/crews/*.yaml`.  
  * Templates API now triggers a reload after CRUD operations, making new templates executable within seconds.  

### 2.2 Tool Result Propagation
* **Root Cause** : `Crew.kickoff()` discarded intermediate artefacts; no mutable context shared across agents.  
* **Fix** : Introduced a **Context Propagation Layer** – a `context` dict stored in `RUNNING_CREWS[task_id]["context"]` and injected into agent inputs. Tools (e.g., CodeGenTool) push structured results (JSON, images) into this dict for downstream consumption.

### 2.3 Frontend Results Display
* **Root Cause** : UI lacked a route and backend lacked an endpoint to fetch completed task data.  
* **Fix** :  
  * New API `GET /crew/{task_id}/result` returns outputs, artefacts, metadata.  
  * New Next.js page `/analysis/[taskId]` visualises executive summary, markdown report, graph, charts, downloadable assets.

### 2.4 PolicyDocsTool Enhancement
* **Root Cause** : Tool performed keyword search over static list → low recall & false positives.  
* **Fix** : Implemented full **Retrieval-Augmented Generation (RAG)**:  
  * Gemini embeddings ➜ Redis `FT.CREATE` vector index (cosine similarity).  
  * Chunked policy corpus (AML, KYC, sanctions).  
  * Fallback to keyword when vector store unavailable.

### 2.5 Task Management & Result APIs
* **Root Cause** : No persistence of running crews; dashboard couldn’t show status.  
* **Fix** :  
  * `RUNNING_CREWS` global dict tracks lifecycle (`STARTING → RUNNING → COMPLETED/ERROR/PAUSED`).  
  * Endpoints:  
    * `GET /crew/tasks` – list & filter tasks  
    * `GET /crew/{task_id}/result` – fetch final artefacts  

### 2.6 Security Enforcement Gaps
* **Root Cause** : Some new endpoints bypassed `require_roles`; JWT blacklist lived in memory only.  
* **Fix** :  
  * Added RBAC guards to all new crew/analysis/template routes.  
  * Confirmed JWT blacklist integration; Redis persistence slated for next migration (`appendonly yes`).

---

## 3 · Architectural Improvements Achieved

* **Dynamic Config Hot-Reload** – system adapts to YAML changes without restart.  
* **Shared Execution Context** – enables rich multi-agent data flow & artefact passing.  
* **RAG Layer** – first prod use of Gemini + Redis vector search.  
* **Task Tracking Abstraction** – groundwork for DB-backed workflow persistence.  
* **Modular Tool Registry** – `get_all_tools()` centralises discovery & hot-plugging.  
* **Full-stack Visibility** – analysts can now create → run → view results in one session.

---

## 4 · Timeline & Validation

| Date | Milestone | Validation |
|------|-----------|------------|
| 01 Jun 2025 | Gap analysis completed | Docs: _CURRENT_STATUS_AND_GAP_ANALYSIS.md_ |
| 02 Jun 2025 | Implementation sprint | Unit + integration tests (coverage still ≈ 50 %) |
| 02 Jun 2025 | PR #63 opened | CI green; manual QA on hot-reload, RAG, UI |
| 03 Jun 2025 | **Merge & deploy planned** | P0 checklist below |

---

## 5 · Next P0 Actions

1. **Merge PR #63** (this document) → deploy staging.  
2. **Alembic migration** for `hitl_reviews` table (persist pause state).  
3. **Redis AOF** (`appendonly yes`) for JWT blacklist durability.  
4. Smoke-test template creation → execution → results workflow.

_Once these are complete analystt1 is considered **Phase-4 production-ready**._

---
