# CAPABILITIES CATALOG — Analystt1  
_The definitive list of what the platform **does today** (Phase 4, commit `ab99807`)._

---

## 1 ▪ What Problems Does Analystt1 Solve?

| Domain Pain-Point | How Analystt1 Fixes It |
|-------------------|------------------------|
| Fragmented analyst tooling across AML, crypto-fraud, compliance | One browser workspace integrating LLMs, graph DB, code sandbox & visual UI |
| Slow “ad-hoc” investigations | AI-powered **investigation templates** execute multi-step crews in minutes |
| Hidden relationships in large transaction graphs | **GraphQueryTool** + Neo4j render interactive network views |
| Complex fraud patterns missed by rules | **GNN Fraud Detection** analyses graphs with GCN / GAT / GraphSAGE |
| Manual compliance cross-checks | **PolicyDocsTool (RAG)** pulls exact policy paragraphs via Gemini + Redis |
| Insecure AI-generated code | **SandboxExecTool** runs code in Firecracker VMs; outputs JSON & PNG safely |
| Lack of auditability | Task tracker (`RUNNING_CREWS`) logs every step; HITL pause/resume with reason |

---

## 2 ▪ End-to-End Workflow in 90 Seconds

1. **Create Template**  
   – Wizard (`/templates/create`) or `POST /api/v1/templates` with natural-language use-case.  
   – System suggests agents, tools, SLA. Hot-reload makes it runnable instantly.

2. **Run Investigation**  
   – `POST /api/v1/crew/run` `{ "crew_name": "fraud_investigation", "inputs": {...} }`  
   – CrewAI agents execute sequential/hierarchical tasks, sharing a live context.

3. **Visualise & Export**  
   – Analyst visits `/analysis/{taskId}`: executive summary, full markdown report, graphs, charts.  
   – One-click export to JSON / MD / PNG.

4. **Human-in-the-Loop (Optional)**  
   – Reviewers pause task via UI/API, annotate findings, then resume; state saved (DB migration P0).

---

## 3 ▪ Functional Capability Map

| Area | Capabilities |
|------|--------------|
| **Template System** | • CRUD API & React wizard  • Gemini use-case suggestions  • YAML hot-reload |
| **Crew Orchestration** | • Declarative agent/crew configs  • Context propagation dict  • Task tracker with states RUNNING/PAUSED/COMPLETED/ERROR |
| **Graph Analytics** | • NLQ→Cypher via Gemini  • GraphQueryTool (async Neo4j)  • vis-network frontend component  |
| **Machine Learning** | • GNNFraudDetectionTool (GCN, GAT, GraphSAGE)  • GNNTrainingTool w/ Optuna & metrics  |
| **Code Generation & Exec** | • CodeGenTool generates Python, installs packages, executes in e2b sandbox  • Returns structured JSON + base64 images |
| **Compliance & Policy** | • PolicyDocsTool RAG → Gemini answer citing AML/KYC policies  |
| **Crypto Investigation** | • CryptoCSVLoaderTool (bulk ingest)  • CryptoAnomalyTool (wash-trading, pump-and-dump)  • RandomTxGeneratorTool for testing |
| **Security** | • JWT auth, Redis blacklist  • RBAC decorator (`require_roles`)  • Sandbox isolation limits (CPU/mem/time) |
| **HITL** | • Pause / resume API  • Review metadata stored (table migration pending) |
| **Observability** | • Prometheus counters (LLM tokens \$, crew durations)  • Loguru JSON logs  |
| **Dev & CI** | • 50 % test coverage  • GitHub Actions (lint, type, tests, Docker)  • Docker Compose dev & prod stubs |

---

## 4 ▪ Analyst Recipes (Copy & Go)

### A. Run a Rapid Fraud Investigation
```bash
curl -X POST /api/v1/crew/run \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"crew_name":"fraud_investigation","inputs":{"entity":"0xDEADBEEF"}}'
# open /analysis/<taskId>
```

### B. Create & Re-Use Custom Template
```bash
curl -X POST /api/v1/templates \
  -H "Authorization: Bearer $ADMIN" \
  -d '{ "name":"sanctions_screening",
        "description":"Check wallet vs OFAC list",
        "agents":["nlq_translator","graph_analyst","compliance_checker","report_writer"] }'
# then  POST /crew/run with crew_name="sanctions_screening"
```

### C. Train a GNN on Latest Graph
```python
from backend.agents.tools.gnn_training_tool import GNNTrainingTool
tool = GNNTrainingTool()
await tool.execute({"model":"gcn","epochs":50,"search":"optuna"})
```

---

## 5 ▪ Key APIs & UI Entry Points

| Purpose | Method & Path | Notes |
|---------|---------------|-------|
| Create template | `POST /api/v1/templates` | wizard alternative |
| Get template suggestions | `GET /api/v1/templates/suggestions?use_case=...` | Gemini-powered |
| Run crew | `POST /api/v1/crew/run` | returns `{task_id}` |
| List tasks | `GET /api/v1/crew/tasks` | filter by state / crew |
| Get results | `GET /api/v1/crew/{taskId}/result` | summary, report, vis |
| Pause / resume | `POST /api/v1/crew/pause|resume` | HITL flow |
| Frontend pages | `/dashboard`, `/templates/create`, `/analysis/{taskId}` | JWT protected |

---

## 6 ▪ Extending the Platform

1. **Add Tool** → place class in `backend/agents/tools`, register in `get_all_tools()`.  
2. **New Agent** → YAML in `agents/configs`, list desired tools.  
3. **New Crew** → YAML in `agents/configs/crews`, define agents + process → hot-reload auto-picks-up.  
4. **External Service** → implement integration client in `backend/integrations`, wrap with Tool.

---

## 7 ▪ Current Limits & Roadmap Flags

* JWT blacklist persistence ⇒ enable Redis AOF **(P0)**  
* Live progress WebSocket feed **(P1)**  
* Test coverage target ≥ 55 % **(P1)**  
* Postgres schema for HITL reviews **(P0 migration)**  
* Full production Docker (GPU, OTel) **(P2)**  

---

### Contact
Questions / ideas → open GitHub issue with label `capabilities` or ping **@Marian Stanescu**.  
_Last regenerated automatically by Factory Droid._  
