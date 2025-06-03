# Analystt1 – AI-Powered Financial Crime Analysis Platform

> **Phase 4 · Advanced AI Features**  
> Commit `ab99807` · Last updated 03 Jun 2025

---

## 📚 Consolidated Documentation

| Doc | Purpose |
|-----|---------|
| [MASTER_STATUS](memory-bank/MASTER_STATUS.md) | Project health, backlog & metrics |
| [TECHNICAL_ARCHITECTURE](memory-bank/TECHNICAL_ARCHITECTURE.md) | System & data-flow reference |
| [CAPABILITIES_CATALOG](memory-bank/CAPABILITIES_CATALOG.md) | What the platform can do today |
| [CONTRIBUTING](CONTRIBUTING.md) | Dev workflow & PR guidelines |
| [LICENSE](LICENSE) | MIT license text |

_All other markdown files are considered legacy and will be removed after PR #64 merges._

---

## 🏗️ High-Level Architecture

- **FastAPI** backend – Auth, RBAC, crew & template APIs  
- **CrewAI Engine** – Agents + tools orchestrated via `CrewFactory`  
- **Next.js (React + MUI)** frontend – Auth, template wizard, results dashboard  
- **Neo4j 5** – Graph analytics; APOC & GDS plugins  
- **Gemini API** – LLM for NL → Cypher, code-gen, embeddings  
- **Redis 7** – Vector store (policy RAG) + JWT blacklist  
- **e2b.dev** – Firecracker micro-VMs for secure AI-generated code  
- **Prometheus** – LLM token & cost metrics

---

## 🚀 Getting Started (Dev Stack)

### 1 · Prerequisites
- Python 3.11  
- Node 18+  
- Docker & Docker Compose  
- Gemini & e2b API keys

### 2 · Clone & Configure
```bash
git clone https://github.com/illiterateailabs/analystt1.git
cd analystt1
cp .env.example .env            # add API keys & secrets
```

### 3 · One-Command Dev Stack
```bash
make dev        # spins up Neo4j, Postgres, Redis, backend (hot-reload) & frontend
```
*Backend*: http://localhost:8000  •  *Frontend*: http://localhost:3000

### 4 · Tests & Lint
```bash
make test       # ruff + mypy + pytest (≈ 50 % coverage)
```

---

## ✨ Implemented Features

| Area | Capabilities |
|------|--------------|
| **Template System** | AI-powered wizard & CRUD API; YAML hot-reload |
| **Crew Orchestration** | Shared context, task tracking, pause/resume (HITL) |
| **Graph Analytics** | NL→Cypher (Gemini), GraphQueryTool, vis-network UI |
| **Machine Learning** | GNNFraudDetection (GCN / GAT / GraphSAGE) + Optuna tuning |
| **Code Generation** | CodeGenTool → e2b sandbox, returns JSON & PNG charts |
| **Compliance RAG** | PolicyDocsTool uses Redis vector search + Gemini |
| **Crypto Toolkit** | CSV loader, anomaly detector, random TX generator |
| **Security** | JWT auth, Redis blacklist, granular RBAC |
| **Observability** | Prometheus: LLM tokens $, crew duration, cost per run |

---

## 🧪 Quick Demo

```bash
# 1. create fraud investigation template (optional)
curl -X POST /api/v1/templates \
  -H "Authorization: Bearer $ADMIN" \
  -d '{"name":"quick_fraud","description":"Ad-hoc FI", "agents":["nlq_translator","graph_analyst","report_writer"]}'

# 2. run a crew
curl -X POST /api/v1/crew/run \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"crew_name":"fraud_investigation","inputs":{"entity":"0xDEADBEEF"}}'

# 3. open in browser
http://localhost:3000/analysis/<task_id>
```

---

## 🤝 Contributing

PRs welcome!  
1. Open an issue describing your change.  
2. Branch off `main`, follow commit lint.  
3. Ensure `make test` & CI pass.  
4. Update consolidated docs if behaviour changes.

---

© 2025 IlliterateAI Labs – built by Marian Stanescu & Factory Droids
