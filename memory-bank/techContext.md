# techContext.md – Technical Context & Setup

_Last updated: **30 May 2025**_

---

## 1. Stack Overview

| Layer | Technology | Version | Role |
|-------|------------|---------|------|
| **Language** | Python | 3.9 / 3.10 / 3.11 (tested) | Backend, agents, tooling |
|  | TypeScript / React (Next.js) | 14.x | Front-end UI |
| **Web Framework** | FastAPI | 0.104.1 | REST/API gateway |
| **Orchestration** | CrewAI | 0.5.0 | Multi-agent engine |
| **LLM** | Google Generative AI (Gemini) | ≥ 0.3.0 SDK | Chat/completion + function calls |
| **Graph DB** | Neo4j | 5.15.0 (Enterprise) | Transaction & entity graph |
| **Graph Algorithms** | Neo4j GDS | 2.x (in DB) | PageRank, centrality, etc. |
| **Vector Store** | Redis (future) / Chroma | Redis 7.2 (dev) | RAG memory, embeddings |
| **Secure Code Exec** | e2b sandboxes | 0.15.0 | Python execution in VM |
| **Runtime** | Docker / docker-compose | 24.x | Containerisation |
| **CI** | GitHub Actions | n/a | Lint, mypy, pytest, docker-build |
| **Observability** | structlog + python-json-logger | 2.0.7 | JSON logs |
|  | Prometheus | scrape | Metrics `/metrics` endpoint |

---

## 2. Development Setup

1. **Clone & bootstrap**

```bash
git clone <repo>
cd analyst-agent-illiterateai
make setup          # creates .env & venv, installs deps
```

2. **Run dev stack**

```bash
make start-dev      # docker compose up (backend reload, frontend dev)
```

3. **Execute tests / lint**

```bash
make lint           # ruff
make type-check     # mypy
make test-coverage  # pytest + coverage htmlcov/
```

4. **Hot reload**  
`uvicorn backend.main:app --reload` (inside container via docker-compose dev profile).

> _Tip:_ `make pre-commit-install` to enable local hooks (ruff, black, mypy).

---

## 3. Environment Configuration (.env keys)

| Variable | Purpose |
|----------|---------|
| `APP_NAME`, `APP_VERSION` | App metadata (health endpoint) |
| `SECRET_KEY` | JWT signing (backend/auth) |
| `GOOGLE_API_KEY` | Gemini LLM access |
| `E2B_API_KEY`, `E2B_TEMPLATE_ID` | Sandbox exec |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Graph DB |
| `CORS_ORIGINS` | Allowed origins for FastAPI |
| `REQUIRE_NEO4J` | Fail startup if Neo4j unreachable (`true/false`) |
| `LOG_LEVEL` | DEBUG / INFO / ERROR |
| `DUNE_API_KEY`, `ETHERSCAN_API_KEY` … | (optional) crypto tool APIs |

All settings loaded via **Pydantic Settings** (`backend/config.py`).

---

## 4. Dependency Catalogue (pip)

| Group | Package(s) | Why We Need It |
|-------|------------|----------------|
| **Core API** | fastapi, uvicorn, pydantic-settings | HTTP server & config |
| **Agents** | crewai\[tools\], crewai-tools, chromadb | Orchestration + vector memory |
| **LLM** | google-generativeai | Gemini SDK |
| **Graph** | neo4j, py2neo | Bolt driver & utility |
| **ML / Data** | pandas, numpy, scipy, scikit-learn, torch, torch-geometric, dgl | Analytics & GNN |
| **Sandbox** | e2b | Remote VM exec |
| **Security** | python-jose, passlib | JWT & hashing |
| **Logging** | structlog, python-json-logger, sentry-sdk | Structured logs & tracing |
| **Testing** | pytest, pytest-asyncio, pytest-cov, pytest-env | CI test suite |
| **Quality** | ruff, black, isort, mypy, pre-commit | Static analysis & style |
| **Async I/O** | aiohttp, aiofiles, websockets | Non-blocking calls |
| **Visualization** | matplotlib, plotly, seaborn | Image/HTML graphs (report_writer) |

_All versions pinned in `requirements.txt`; update via `pip-tools` in future._

---

## 5. Tool Usage Patterns

| Tool Class | Wraps | Typical Call Flow |
|------------|-------|-------------------|
| `GraphQueryTool` | `Neo4jClient.run_query()` | Cypher text → dict result |
| `Neo4jSchemaTool` | DB meta queries | Provides labels / rels for NLQ |
| `SandboxExecTool` | `E2BClient.execute_code()` | Python string → stdout/stderr |
| `CodeGenTool` | Gemini function-call | Prompt → Python code |
| `PatternLibraryTool` | YAML motifs store | Motif ID → Cypher template |
| `PolicyDocsTool` | RAG over regs (Redis) | Question → citation answer |

Agents receive tool instances via **CrewFactory** dependency injection.

---

## 6. Docker & Deployment

### Compose Profiles
| Profile | Purpose | Services |
|---------|---------|----------|
| **dev** | Hot-reload, bind mounts | backend, frontend, neo4j, redis, postgres |
| **prod** | Secure, non-root, pinned tags | backend, frontend, neo4j, redis, postgres |

```
docker-compose up --profile prod -d
```

### Backend Image

* **Dockerfile** multi-stage:  
  1. `python:3.11-slim` builder – install deps  
  2. Copy app ➜ non-root user  
* Healthcheck `curl /health`

### Frontend Image

* Next.js build → `next start`  
* Nginx side-car for static assets (prod).

### Deployment Workflow (CI/CD)

1. **GitHub Actions**  
   * Install deps → lint → mypy → pytest matrix → coverage  
   * Buildx backend & frontend images (`--cache type=gha`)  
2. **Push** to container registry (future step).  
3. **Deploy** via Docker Compose or Kubernetes Helm.

> _Kubernetes note:_ health probes reuse `/health`, secrets injected via k8s Secrets, Prometheus scrape annotation enabled.

---

## 7. Technical Constraints & Guidelines

* **Sequential crew** for MVP – maximise auditability (reg-tech requirement).  
* **100 % reproducibility** – Task-IDs and build SHA included in every health JSON.  
* **No direct DB writes from LLM** – all mutative operations go through validated service layer.  
* **HITL mandatory** for compliance_checker – system must pause until reviewer approves.  
* **Cost limits** – keep average Gemini token spend ‹ $0.01 per enrichment; monitor via AgentOps (Phase 3).  
* **Security** – never expose E2B sandbox tokens in logs; enforce `REQUIRE_NEO4J=true` in prod.

---

*Maintaining this Technical Context ensures any future contributor (or post-reset me) can spin up the stack, understand dependencies, and deploy with confidence.*  
