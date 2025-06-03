# CAPABILITIES_CATALOG.md  
_Analystt1 – Comprehensive Capability Reference_  
_Last updated: **03 Jun 2025 – commit `ab99807` (PR #64)**_

---

## 1 ▪ Problems Analystt1 Solves

| Domain Challenge | How Analystt1 Addresses It |
|------------------|---------------------------|
| Fragmented fraud investigations across fiat & crypto channels | Unified graph-driven data model (Neo4j) + multi-chain API ingestion |
| Slow manual workflow design | **AI-powered Template System** generates end-to-end investigations in minutes |
| Hidden relational fraud patterns | **Graph Analytics + GNN** surface complex rings, mixers, layering tactics |
| Code/ML results lost between tools | **Context propagation layer** ensures every agent sees prior outputs |
| Disconnected compliance checks | **PolicyDocsTool (RAG)** cross-references AML/KYC policy corpus in real time |
| Risky on-prem execution of untrusted code | **E2B Sandboxes** isolate CodeGenTool runs |
| Auditability & human oversight | **HITL reviews** stored in Postgres; metrics exported to Prometheus |
| Token leakage & key rotation pain | **JWT blacklist with Redis AOF** guarantees revocation durability |

---

## 2 ▪ End-to-End Workflow

```
 Analyst ⤏ Browser UI
     │ 1. create / pick template
     ▼
Frontend Wizard (React)
     │ POST /api/v1/templates
     ▼
 FastAPI Templates API      ──▶  Gemini Suggestion Engine
     │ 2. run investigation
     ▼
 Analysis API  ──▶  CrewFactory.reload()
     │ enqueue    │                     (hot-reload)
     ▼            ▼
 RUNNING_CREWS  (in-memory) ──▶  CrewAI Engine
     │                               │
     │ 3. agents execute             ▼
     │                               Tools Layer
     │                               ├─ GraphQueryTool → Neo4j
     │                               ├─ GNNFraudDetectionTool → PyTorch/DGL
     │                               ├─ CodeGenTool → e2b.dev
     │                               └─ PolicyDocsTool → Redis vector
     │                              
     │ 4. context propagated ◄───────┘
     ▼
 Results persisted (S3 / Postgres) ──▶  HITL Review UI (if required)
     │
     ▼
Frontend Dashboard `/analysis/{taskId}` renders graphs, charts, risk scores
```

---

## 3 ▪ Functional Capability Map

| Category | Capability | Status | Key Files/Modules |
|----------|------------|--------|-------------------|
| **Template System** | 6-step wizard, AI suggestions, YAML storage, hot-reload | ✅ | `backend/api/v1/templates.py`, `frontend/src/components/templates/*`, `agents/factory.py` |
| **Graph Analytics** | Parameterized Cypher queries, sub-graph extraction, pattern library matching | ✅ | `graph_query_tool.py`, `pattern_library_tool.py`, `backend/neo4j_client.py` |
| **Machine Learning / GNN** | Fraud detection with GCN/GAT/GraphSAGE, Optuna tuning, model registry stub | ✅ (Phase 4) | `gnn_fraud_detection_tool.py`, `gnn_training_tool.py` |
| **Code Generation & Execution** | LLM-powered Python/SQL code, isolated run, artifact capture (plots, HTML) | ✅ | `code_gen_tool.py`, E2B integration |
| **Compliance & RAG** | PolicyDocsTool semantic search over AML/KYC corpus, Gemini embeddings, vector Redis | ✅ | `policy_docs_tool.py` |
| **Crypto Investigation** | Multi-chain APIs, anomaly tool, random TX generator | ✅ | `crypto_anomaly_tool.py`, `random_tx_generator_tool.py` |
| **Security** | JWT auth/refresh, RBAC, Redis blacklist with AOF, sandbox exec, GDPR purge helper | ✅ (AOF pending P0) | `auth/*.py`, `redis` service config |
| **Human-in-the-Loop** | Pause, resume, approve, reject review endpoints; metrics; Postgres storage | 🟡 (migration P0) | `metrics.py`, forthcoming `hitl_reviews` table |
| **Observability** | Prometheus metrics (LLM tokens, cost, crew durations, HITL), structured logs | ✅ | `backend/core/metrics.py` |
| **Real-time UX** | WebSocket/SSE progress feed | 🟡 P1 | planned `ws_progress.py`, frontend hooks |
| **MCP Integration** | Echo & Graph servers, dynamic tool hot-plug | ✅ Phase 0 | `mcp/client.py`, `mcp_servers/*` |

Legend: ✅ implemented • 🟡 planned/in-progress • 🔲 future

---

## 4 ▪ API Examples

### Authentication

```bash
# Login
curl -X POST /api/v1/auth/login \
     -d '{"username":"alice","password":"secret"}'
# -> { "access_token":"...", "refresh_token":"..." }
```

### Create Investigation Template

```bash
curl -X POST /api/v1/templates \
     -H "Authorization: Bearer $TOKEN" \
     -F "yaml=@aml_investigation.yaml"
```

### Run Analysis Task

```bash
curl -X POST /api/v1/analysis \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"template":"aml_investigation"}'
# -> { "task_id":"123e..." }
```

### Fetch Results

```bash
curl /api/v1/analysis/123e/results \
     -H "Authorization: Bearer $TOKEN"
```

_HITL endpoints_: `/api/v1/hitl/{taskId}/approve`, `/reject`, `/pause`.

---

## 5 ▪ Integration Patterns

| Pattern | Description |
|---------|-------------|
| **CrewAI Orchestration** | YAML → `CrewFactory` → dynamic agents/tools instantiation |
| **Context Propagation** | Shared dict passed through agents; eliminates brittle I/O edges |
| **Model Context Protocol (MCP)** | Tool processes can run as separate servers; client abstracts transport |
| **Redis Vector Search** | Gemini embeddings stored with `HNSW` index (`redis-py 5.x`) |
| **Append-Only Redis** | `appendonly yes` ensures JWT blacklist survives restarts |
| **Alembic Migrations** | `alembic revision --autogenerate` tracks Postgres schema (init + HITL) |
| **E2B Sandbox Isolation** | Code execution in remote container, artifacts streamed back |
| **Prometheus Pull Model** | `/metrics` endpoint scraped; optional Pushgateway for batch jobs |

---

## 6 ▪ Roadmap (Phase 4 → 6)

| Phase | Focus | Target Date | Key Deliverables |
|-------|-------|-------------|------------------|
| **4 – Advanced AI Features** | Integration fixes, template system, docs consolidation | ✅ Jun 2025 | PR #64 – #66 merged |
| **4.x Blockers (P0)** | _See below_ | **EOW 03 Jun 2025** | ALEMBIC `hitl_reviews` migration, Redis AOF, smoke test |
| **5 – Real-Time & GPU** | WebSocket progress, 55 % test coverage, GPU image & CI job | mid Jun 2025 | live updates, CUDA Dockerfile, nightly GNN tests |
| **6 – Observability & Scaling** | OpenTelemetry traces, tenant onboarding wizard, Graph Transformer | Jul 2025 | OTel ➜ Grafana, multi-tenant mode, HGT experiments |

---

## 7 ▪ Current P0 Tasks

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | **Enable Redis AOF for JWT blacklist persistence** | DevOps | 🔄 verifying in all envs |
| 2 | **Alembic migration: `hitl_reviews` table** | Backend | 🚧 to-do |
| 3 | **End-to-end smoke test** (template → execution → UI) | QA | ⏳ |
| 4 | **Merge PR #64** (integration fixes + template) | ✅ | done |
| 5 | **Merge PR #65 & #66** (docs consolidation & cleanup) | ✅ | done |

---

## 8 ▪ Extending Analystt1

1. **Add a new Tool**  
   • Implement class in `backend/agents/tools/` adhering to `ToolProtocol`  
   • Register in `tools/__init__.py` & `ToolFactory`  
   • Reference name in agent YAML

2. **Create a new Template**  
   • POST YAML via `/api/v1/templates` or use Frontend wizard  
   • Hot-reload makes it instantly available

3. **Onboard another Data Source**  
   • Write ETL script, store raw in S3, nodes/edges in Neo4j  
   • Extend GraphQueryTool config with new labels / relationships

---

## 9 ▪ References & Links

* `MASTER_STATUS.md` – project health & backlog  
* `TECHNICAL_ARCHITECTURE.md` – in-depth system design  
* `TESTING_GUIDE.md` – how to run & extend tests  
* CrewAI docs – https://github.com/joaomdmoura/crewai

---

© 2025 IlliterateAI Labs – Built by Marian Stanescu & Factory Droids  
_All new capabilities must be recorded in this catalog._
