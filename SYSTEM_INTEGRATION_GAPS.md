# SYSTEM_INTEGRATION_GAPS.md
_Last updated: 02 Jun 2025_

## 1 · Executive Summary 🚨
Analystt1’s components (FastAPI backend, CrewAI engine, Next.js UI, Neo4j layer) **work in isolation** but **fail to operate as a single product**.  
Six critical wiring gaps prevent production readiness:

| # | Integration Failure | Business Risk |
|---|---------------------|---------------|
| 1 | **Template Creation ↔ Crew Execution** | Users create templates but cannot run them, blocking the flagship feature. |
| 2 | **Tool Result Propagation** (CodeGenTool, PolicyDocsTool) | Investigations miss ML insights → false negatives/positives. |
| 3 | **Frontend Results Display** | Analysts receive no investigation output → unusable UI. |
| 4 | **HITL State Persistence** | Compliance reviews lost on restart → audit violation. |
| 5 | **Security Enforcement Gaps** (RBAC & JWT revocation) | Unauthorised crew runs & stale tokens → regulatory exposure. |
| 6 | **Real-Time Updates** | 30 s workflows feel frozen; poor UX and uncertainty. |

## 2 · Current Data-Flow & Breakpoints
```text
Frontend (Next.js)
  │ 1. POST /templates         ──✔ creates YAML
  │ 2. POST /crew/run          ──✖ template not linked
  │
FastAPI API
  ├─ /api/v1/templates.py      (stores YAML)
  ├─ /api/v1/crew.py           (executes static crews)
  │      └─  RBAC missing here ⚠
  │
CrewFactory (backend/agents/factory.py)
  │  ├─ loads DEFAULT_*.yaml only
  │  ├─ runs agents sequentially
  │  └─ CodeGenTool──►E2B Exec──►result ⬅(lost here)⬅
  │
Agents / Tools
  │  ├─ fraud_pattern_hunter
  │  ├─ compliance_checker ──► PolicyDocsTool (stub) ⚠
  │  └─ report_writer (no ML charts)
  │
State Layer
  │  └─ RUNNING_CREWS = {}  (memory only) ⚠ loses data on restart
  │
Redis (JWT blacklist not persistent) ⚠
Neo4j  ✔
Gemini ✔
```
_Breakpoints marked ⚠ or ✖ above._

## 3 · Gap Deep-Dive

| # | Gap | What’s Broken / Missing | User Impact | Technical Root Cause | Proposed Fix |
|---|-----|-------------------------|-------------|----------------------|--------------|
| 1 | Template ↔ Execution | Newly created templates (YAML) never loaded when `/crew/run` is called. | Users think they can run custom workflows but nothing happens. | `CrewFactory.get_available_crews()` reads only `DEFAULT_CREW_CONFIGS`, not disk. | • Extend CrewFactory to watch `backend/agents/configs/crews/*.yaml` (hot-reload already exists).<br>• Accept `crew_name` that matches user templates.<br>• Add lookup in `/crew/run` to call `CrewFactory.run_crew` for dynamic templates. |
| 2 | Tool Result Flow | CodeGenTool output JSON/figures lost; PolicyDocsTool returns stub. | Reports lack ML charts/compliance snippets → incomplete investigations. | `CodeGenTool.execute()` returns to CrewFactory but not attached to shared context; `PolicyDocsTool` has placeholder logic. | • Pass CodeGenTool result into shared `context` dict appended to TaskOutput.<br>• Refactor `report_writer` to read context for `analysis_assets`.<br>• Implement vector retrieval in `PolicyDocsTool` (Redis + Gemini embeddings). |
| 3 | Frontend Results View | No page renders crew results. | Analysts cannot see findings. | Dashboard calls `/crew/tasks` but no results endpoint/UI. | • Add `/api/v1/crew/{task_id}/result` endpoint.<br>• Build `frontend/src/app/analysis/[task_id]/page.tsx` to display markdown report + graph PNG. |
| 4 | HITL Persistence | Pause/resume state stored in RAM. | If pod restarts, review queue disappears. | `RUNNING_CREWS` dict in factory; no DB table. | • Create `hitl_reviews` table (id, task_id, state_json, status).<br>• Persist on pause, reload on resume. |
| 5 | Security Gaps | `/crew/run` lacks RBAC; JWT blacklist in Redis not persisted. | Potential abuse & stale sessions. | Decorator present but disabled; Redis configured volatile. | • Add `@require_roles(RoleSets.ANALYSTS_AND_ADMIN)` in `backend/api/v1/crew.py` lines 37 & 79.<br>• Switch Redis to `appendonly yes` or RDB save; re-hydrate blacklist on start. |
| 6 | Real-Time Updates | Long runs give no progress in UI. | User thinks system hung. | Only REST; no push channel. | • Add FastAPI WebSocket `/ws/tasks/{task_id}` broadcasting TaskStatus events.<br>• Use SWR or React hook in dashboard to subscribe. |

## 4 · Fix Prioritisation Matrix

| Priority | Gap | Impact | Effort | Rationale |
|----------|-----|--------|--------|-----------|
| **P0** | 1, 5 | 🔴 High | ⚪ Medium | Blocks core feature & poses security risk |
| **P1** | 2, 3 | 🟠 High | 🟠 High | Delivers usable results & insights |
| **P2** | 4 | 🟡 Medium | 🟠 Medium | Compliance / reliability |
| **P3** | 6 | 🟡 Medium | ⚪ Low – WebSocket infra simple | UX polish |

## 5 · File Paths & Code Touch-Points

| Component | Files to Modify / Add |
|-----------|-----------------------|
| CrewFactory | `backend/agents/factory.py` (dynamic YAML loader, context propagation) |
| Crew API | `backend/api/v1/crew.py` (RBAC decorator, new result endpoint) |
| Templates API | `backend/api/v1/templates.py` (trigger reload after save) |
| Tools | `backend/agents/tools/code_gen_tool.py`, `policy_docs_tool.py`, `report_writer` logic |
| HITL Persistence | `backend/models/hitl_review.py`, Alembic migration, `CrewFactory.pause_crew` / `resume_crew` |
| Security | `backend/auth/jwt_handler.py`, Redis config in `docker-compose.yml` |
| Frontend | `frontend/src/app/analysis/[task_id]/page.tsx`, hooks in `frontend/src/lib/api.ts`, WebSocket client |
| Real-Time | New file `backend/api/v1/ws.py` (WebSocket router), integrate in `main.py` |

## 6 · Implementation Roadmap

### Phase 0 (≤ 1 day)  
1. **Re-enable RBAC** on `/crew/run` (single-line change).  
2. **Configure Redis AOF** for JWT persistence; reload blacklist at startup.

### Phase 1 (2-3 days)  
1. Extend **CrewFactory**:  
   - Load all YAML in `configs/crews/` on startup & on templates API `POST`.  
   - Accept custom `crew_name` in `run_crew`.  
2. **Templates API**: after create/update/delete, call `CrewFactory.reload()`.

### Phase 2 (3-4 days)  
1. **CodeGenTool propagation**: attach `execution_result` to `TaskOutput.context`.  
2. **Report Writer**: render charts/images from context.  
3. Implement **PolicyDocsTool** vector search via Redis (use existing embedding utility).

### Phase 3 (4-5 days)  
1. **Result Endpoint** `/crew/{task_id}/result` returning markdown + assets.  
2. **Frontend Analysis View** page; wire dashboard links.

### Phase 4 (3 days)  
1. **HITL DB persistence**: new model, migrate, update pause/resume logic.  
2. **WebSocket/SSE** broadcast for task progress; React hooks to consume.

### Phase 5 (Testing & Hardening, 2 days)  
- Write pytest integration tests for new flows.  
- Playwright E2E for template → execution → result UI.  
- Update docs & bump coverage to 55 %.

---

Closing these gaps will transform analystt1 from a collection of promising modules into a **cohesive, secure, and user-friendly financial crime analysis platform ready for production deployment**.
