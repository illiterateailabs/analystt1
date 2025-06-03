# TECHNICAL_ARCHITECTURE.md  
_Analystt1 – Canonical Technical Reference_  
_Last updated: **03 Jun 2025 – commit `ab99807` (PR #64)**_

---

## 1 ▪ High-Level Overview

```
┌───────────────────┐        HTTPS         ┌────────────────────────┐
│     Frontend      │  ─────────────────▶  │ FastAPI Back-End       │
│  (Next.js + MUI)  │   JSON/WS/SSE        │  • Auth & RBAC         │
└───────────────────┘                      │  • Crew / Templates    │
        ▲                                  │  • Tool Gateway        │
        │                                  └─────────┬──────────────┘
        │                                          Async I/O
        │                            ┌────────────────┴────────────────┐
        │                            │   CrewAI Engine (CrewFactory)   │
        │                            │  • Agent & Tool registry        │
        │   WebSockets (P1)          │  • RUNNING_CREWS task tracker   │
        │                            │  • Context propagation          │
        │                            └────────────────┬────────────────┘
        │                         calls via           │
        │                         Python APIs         │
┌───────┴────────┐  Cypher   ┌──────────────┐   REST / gRPC   ┌────────────┐
│   Neo4j 5.15   │◀──────────┤  GraphQuery  │                 │  e2b.dev   │
└────────────────┘           └──────────────┘<───────────────▶│ Sandboxes  │
   │  APOC, GDS                                Code / PIP      └────────────┘
   │                                                  ▲
   │  Embeddings / Cypher                             │
┌──┴────────────┐        Vector search     ┌──────────┴─────────┐
│  Gemini LLM   │  <────────────────────── │    Redis 7         │
└───────────────┘     (Policy RAG)         └────────────────────┘
        ▲                                            │
        │ JWT / TLS                                   │
┌───────┴───────────┐                                │
│  Postgres 15      │  Alembic migrations            │
└───────────────────┘                                ▼
                                          Prometheus Metrics / OTel (P2)
```

---

## 2 ▪ Component Breakdown

| Layer | Module / Dir | Key Classes & Files | Purpose |
|-------|--------------|---------------------|---------|
| **Frontend** | `frontend/src` | `app/analysis/[taskId]/page.tsx` | Auth, dashboard, template wizard, results UI |
| **API** | `backend/api/v1` | `crew.py`, `templates.py`, `analysis.py` | Crew run/pause/resume, task list & result, template CRUD |
| **Factory** | `backend/agents/factory.py` | `CrewFactory`, `get_all_tools`, `RUNNING_CREWS` | Central builder; hot-reload configs; shared context |
| **Agents** | `backend/agents/configs` | YAML files per agent/crew | Declarative definitions (role, goal, tools) |
| **Tools** | `backend/agents/tools` | `GraphQueryTool`, `CodeGenTool`, `SandboxExecTool`, `PolicyDocsTool`, `GNN*Tool` | Encapsulate external capabilities |
| **LLM** | `backend/integrations/gemini_client.py`, `agents/llm.py` | `GeminiClient`, `GeminiLLMProvider` | Text & code generation, embedding |
| **Execution** | `backend/integrations/e2b_client.py` | Secure Python execution in Firecracker VMs |
| **Data** | `backend/integrations/neo4j_client.py` | Async driver helpers, Cypher execution |
| **Security** | `backend/auth` | `jwt_handler.py`, `rbac.py` | JWT issuance/blacklist, role decorator |
| **Persistence** | `alembic/versions`, `backend/database.py` | DB models (User, HITL to-do), migrations |
| **Observability** | `backend/core/metrics.py`, Prometheus exporter | LLM token & cost counters, crew execution timers |
| **Memory Bank** | `memory-bank/*.md` | `MASTER_STATUS.md`, this doc | Single source doc store |

---

## 3 ▪ Data Flow

### 3.1 Template → Execution → Results

1. **Template Creation**  
   Frontend wizard → `POST /api/v1/templates` (FastAPI)  
   • YAML written under `backend/agents/configs/crews/`  
   • `CrewFactory.reload()` hot-reloads config cache.

2. **Crew Execution**  
   Analyst calls `POST /api/v1/crew/run` with `crew_name` & optional inputs.  
   • Factory generates `task_id`, adds entry in `RUNNING_CREWS`.  
   • Agents iterate tasks; context dict `_context` travels via `kickoff_with_context`.  
   • Tools (e.g., CodeGenTool) push artefacts into `_context`.

3. **Tool Invocations**  
   • _GraphQueryTool_ runs Cypher via Neo4j; returns JSON subgraphs.  
   • _CodeGenTool_ → Gemini code → e2b sandbox → executes → returns stdout, JSON, PNG (base64) → stored in context.  
   • _PolicyDocsTool_ embeds query, vector search Redis → Gemini answer.

4. **Completion**  
   Crew returns `TaskOutput`; `RUNNING_CREWS` updated to `COMPLETED`.  
   Analyst polls `GET /api/v1/crew/{task_id}/result` → receives raw output, report (if generated), visualizations, metadata.

5. **UI Display**  
   React page fetches result, renders executive summary, markdown report, vis-network graph, gallery, risk/confidence chips.

### 3.2 Authentication

```
Login ➜ /auth/login
   ↳ bcrypt pwd check
   ↳ access/refresh JWT
           ↓
Every API call with Authorization: Bearer <access>
   ↳ FastAPI dependency verifies JWT + Redis blacklist
   ↳ @require_roles checks RBAC scopes
```

Refresh tokens stored HTTP-only; blacklist persisted (P0).

---

## 4 ▪ Tool Registry & Extensibility

`get_all_tools()` instantiates core + crypto + GNN tools.  
Adding a tool:

1. `class MyTool(BaseTool): ...` in `backend/agents/tools`.  
2. Append to `get_all_tools()` or dynamic discover via entry-points (P2).  
3. Reference tool ID in agent YAML; hot-reload picks it up.

---

## 5 ▪ Runtime Lifecycle

| Phase | Action |
|-------|--------|
| **Startup** | FastAPI mounts routers; CrewFactory singleton lazily created; gRPC (future) servers start |
| **Connect** | Each crew run triggers `CrewFactory.connect()` → Neo4j driver init |
| **Execution** | Agents sequential/hierarchical per YAML; metrics collected; context propagated |
| **Pause / HITL** | `POST /crew/pause` sets state=PAUSED; review stored; resume continues same context |
| **Shutdown** | `CrewFactory.close()` closes Neo4j, terminates sandboxes |

---

## 6 ▪ Deployment & CI

* **Docker Compose (dev)**: `docker-compose.yml` spins Neo4j, Postgres, Redis, backend (hot-reload), frontend.  
* **Docker Compose (prod)**: `docker-compose.prod.yml` – Nginx static frontend, gunicorn Uvicorn workers (GPU image WIP).  
* **CI Pipeline** (GitHub Actions):  
  * Ruff + Black + isort  
  * mypy static typing  
  * pytest (banked creds mocked)  
  * Build Docker images  
  * Push artefacts (future).

---

## 7 ▪ Security Layers

1. **RBAC Decorator** – endpoint guard; roles in JWT.  
2. **JWT Blacklist** – Redis set + AOF (pending).  
3. **SandboxExecTool** – Firecracker micro-VM isolation, 30 s CPU / 512 MB mem caps.  
4. **CORS & HTTPS** – enforced in FastAPI + Nginx.  
5. **Audit Trails** – Run events in RUNNING_CREWS (persist to Postgres P1).  
6. **Data Governance** – PolicyDocsTool cross-checks findings vs AML/OFAC guidelines.

---

## 8 ▪ Observability

* **Prometheus Exporter** – LLM token usage, cost (USD), crew duration, task counts.  
* **Structured Logging** – Loguru JSON lines; log level via env.  
* **OTel Instrumentation** – TODO (Phase 5).  
* **Frontend Sentry** – TODO.

---

## 9 ▪ File & Document Governance

* This document + `MASTER_STATUS.md` are the _only_ canonical docs.  
* All other scattered .md files (progress, roadmap, etc.) are legacy; will be deleted after PR #64 merge.  
* Updates: bump versions & dates in §1 and relevant tables.

---

## 10 ▪ Future Evolution

* **Graph Transformer** & heterogeneous graph support (HGT, HAN).  
* **Real-time SSE/WebSocket progress** integrated with RUNNING_CREWS.  
* **Tenant isolation** – row-level security + per-tenant Neo4j DBMS.  
* **Model Registry** – MLflow / S3 for GNN artefacts.  
* **Plugin Marketplace** – community tools auto-discoverable via MCP.

---

_© 2025 IlliterateAI Labs – built by Marian Stanescu & Factory Droids_
