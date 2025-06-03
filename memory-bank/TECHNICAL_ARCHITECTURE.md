# TECHNICAL_ARCHITECTURE.md  
_Analystt1 – Canonical Technical Reference_  
_Last updated: **03 Jun 2025 – commit `ab99807` (PR #64)**_

---

## 1 ▪ High-Level Overview

```
┌───────────────────┐        HTTPS/WS        ┌────────────────────────┐
│     Frontend      │  ───────────────────▶  │    FastAPI Back-End    │
│  (Next.js + MUI)  │                       │  • Auth & RBAC         │
└───────────────────┘                       │  • Crew / Templates    │
        ▲                                   │  • Tool Gateway        │
        │                                   └─────────┬──────────────┘
        │                                           Async I/O
        │                             ┌──────────────┴──────────────┐
        │                             │   CrewAI Engine (Factory)   │
        │                             │  • Agent & Tool registry    │
        │      WebSockets/SSE (P1)    │  • RUNNING_CREWS tracker    │
        │                             │  • Context propagation      │
        │                             └──────────────┬──────────────┘
        │                          Python calls      │
        │                                             ▼
        │                        ┌──────────────────────────────┐
        │                        │  Tools & External Services   │
        │                        │  (GraphQuery, GNN, RAG, …)   │
        │                        └──────────────────────────────┘
Services: Neo4j • Postgres • Redis • E2B Sandboxes • Gemini API • Prometheus
```

---

## 2 ▪ Component Breakdown

| Layer | Component | Notes |
|-------|-----------|-------|
| **Frontend** | Next.js 14 / React 18, MUI v5 | Wizard UIs (templates, investigation run), analysis dashboard, auth pages |
| **API** | FastAPI 0.111 | Versioned under `/api/v1/*`; fully async; integrated Prometheus middleware |
| | `auth.py` | JWT issuance, refresh, blacklist (Redis) |
| | `templates.py` | CRUD + AI suggestions (Gemini) |
| | `analysis.py` | Task submission, status, results retrieval |
| | `crew.py` | Hot-reload & runtime management |
| **CrewAI Engine** | `CrewFactory` + `agents/*` | Loads YAML configs & user templates, instantiates agents & tools |
| **Tools** | GraphQueryTool, GraphQLQueryTool, CodeGenTool (E2B), GNNFraudDetectionTool, PatternLibraryTool, PolicyDocsTool (RAG), SandboxExecTool, etc. | All implement common interface; discoverable via ToolFactory |
| **Data Stores** | Neo4j 5.15 (graph), Postgres 15 (relational, Alembic), Redis 7 (cache, JWT, vector store), Minio/S3 (artifacts) |
| **External** | Gemini Flash/Pro (LLM), E2B secure containers, Optuna (tuning) |
| **Observability** | Prometheus metrics exporter, structured Loguru logs, future OpenTelemetry traces |

---

## 3 ▪ Detailed Data Flow

1. **User action** (template create / run investigation) from browser → Frontend fetch / mutate.
2. **FastAPI** validates JWT (RBAC) → routes to corresponding service:
   - **Template** CRUD hits `templates.py` → persists YAML in `backend/agents/configs/*` → `CrewFactory.reload()` for hot-availability.
   - **Analysis** run hits `analysis.py` → enqueues new crew in `RUNNING_CREWS` Gauge, returns `taskId`.
3. **CrewAI Engine** executes agents sequentially/parallel:
   - Each **Agent** selects **Tools**; outputs structured JSON or artifacts.
   - **Context propagation**: shared dict passed through all agents; CodeGenTool results, GNN predictions, etc., become inputs for later steps.
4. **Results persistence**:
   - Intermediate JSON in Redis (TTL), final artefacts (plots, HTML) in S3 path `analyses/{taskId}/`.
   - DB rows (Postgres) for HITL reviews (`hitl_reviews` table – P0 migration).
5. **Frontend polling / (P1) WebSocket** fetches `/analysis/{taskId}` until `status=done`, then renders graphs & reports.
6. **Metrics** pushed: crew durations, LLM tokens, cost, errors → Prometheus scrape at `/metrics`.

---

## 4 ▪ Tool Ecosystem & Agent Configurations

| Tool | Purpose | Key Deps |
|------|---------|----------|
| **GraphQueryTool** | Parameterized Cypher queries, sub-graph extraction | neo4j-python |
| **GNNFraudDetectionTool** | GCN, GAT, GraphSAGE inference; pattern detection | PyTorch + DGL |
| **GNNTrainingTool** | Train & tune models via Optuna; saves `.pt` artefacts | CUDA optional |
| **CodeGenTool** | Generates & executes Python/SQL in sandbox (E2B) | e2b.dev API |
| **PolicyDocsTool** | RAG over AML/KYC docs; Redis vector store | Gemini embeddings |
| **PatternLibraryTool** | Yaml-defined fraud patterns; matches graph motifs | pydantic |
| **SandboxExecTool** | Generic code execution with isolation | E2B |
| **Neo4jSchemaTool** | Auto-migrate graph schema from YAML | neo4j |

Agents are defined YAML-first; key agent roles:

- `investigator`: orchestrates workflow
- `graph_analyst`: runs GraphQuery & GNN tools
- `code_writer`: uses CodeGenTool for visualizations
- `compliance_checker`: invokes PolicyDocsTool
- `report_writer`: collates context dict → Markdown/HTML report

---

## 5 ▪ Context Propagation Mechanism

```python
# Simplified excerpt
context: dict[str, Any] = {}
for agent in crew.agents:
    result = await agent.run(context)
    context.update(result)           # 🔑 Key line – global mutable context
```

• **RUNNING_CREWS** (in‐memory dict) stores `taskId → context` pointer.  
• Completed contexts serialized to S3 & Postgres for audit.  
• Prevents “lost tool outputs” problem fixed in PR #63.

---

## 6 ▪ Security

1. **JWT Auth** – Access & refresh tokens (HS256); blacklist set in Redis with AOF (`appendonly yes`) for durability.  
2. **RBAC** – Role scopes (`analyst`, `admin`) enforced via FastAPI `Depends`.  
3. **Sandboxed Code Execution** – All arbitrary code runs in e2b.dev secure containers; no host access.  
4. **Database Least Privilege** – App role with limited grants; Alembic migrations tracked.  
5. **Secrets** – Supplied via Docker/CI env vars; never committed.  
6. **Compliance** – PolicyDocsTool auto-flags sanctions/AML violations; HITL review table stores approvals.

---

## 7 ▪ Observability

| Aspect | Implementation |
|--------|----------------|
| **Metrics** | `backend/core/metrics.py` – Prometheus Counters, Histograms, Gauges (LLM tokens, cost, crew durations, HITL reviews) |
| **Logging** | Loguru JSON lines, log level via `LOG_LEVEL` env; aggregated by docker-compose. |
| **Tracing** | OpenTelemetry instrumentation planned (Phase 5). |
| **Frontend** | React error boundaries; Sentry SDK TODO. |

---

## 8 ▪ Deployment Footprint

| Compose Service | CPU | RAM | Notes |
|-----------------|-----|-----|-------|
| backend | 1 vCPU | 1 GiB | FastAPI + workers |
| frontend | 0.5 vCPU | 512 MiB | Next.js prod |
| neo4j | 2 vCPU | 2 GiB | Heap 1 GiB, pagecache 512 MiB |
| postgres | 1 vCPU | 1 GiB | WAL, `shared_buffers=256M` |
| redis | 0.5 vCPU | 512 MiB | AOF on, `maxmemory 256M` |
| prometheus (optional) | 0.5 vCPU | 512 MiB | External scrape |

GPU image (P1) will extend backend with CUDA 12, PyTorch.

---

## 9 ▪ Future Evolution

* **WebSocket/SSE progress feed** – Task events → UI real-time.  
* **GPU CI Job** – nightly GNN baseline evaluation.  
* **Graph Transformer & heterogeneous graphs** – HGT, HAN experimentation.  
* **OpenTelemetry + Grafana/Loki** – full tracing and log dashboards.  
* **Multi-tenant onboarding** – tenant_id column + Neo4j DBMS per tenant.

---

© 2025 IlliterateAI Labs • Built by Marian Stanescu & Factory Droids  
_Every architectural change **must** be reflected in this document._
