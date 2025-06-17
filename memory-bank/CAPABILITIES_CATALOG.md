# Capabilities Catalog  
*File `memory-bank/CAPABILITIES_CATALOG.md` · last updated 2025-06-17*

A consolidated index of **features, tools and integrations** that power the Analyst Augmentation Agent.  
Use this catalog to quickly assess what is already live, what is under active development, and what is on the roadmap.

Legend | Meaning  
-------|---------  
✓ Implemented | Feature is live in `main`  
🚧 In-progress | Work started (open PR / flagged in backlog)  
🛠 Planned | Approved roadmap item, no code yet  
❌ Deprecated | Superseded / removed

---

## 1 · Core Platform

| Capability | Module / Path | Status | Notes |
|------------|---------------|--------|-------|
| FastAPI REST API | `backend/main.py` + `backend/api/v1/*` | ✓ | Versioned under `/api/v1` |
| Async Postgres ORM | `backend/database.py` (SQLAlchemy 2 async) | ✓ | StaticPool (dev) / QueuePool (prod) |
| Auth & RBAC (JWT) | `backend/auth/*` | ✓ | HS256; cookie rotation 🛠 |
| CrewAI Orchestration | `backend/agents/*` | ✓ | Multi-agent workflows |
| Task Progress WebSockets | `backend/api/v1/ws_progress.py` | ✓ | Real-time task updates |

---

## 2 · AI & Analysis

| Capability | Module / Tool | Status | Notes |
|------------|---------------|--------|-------|
| Gemini Text Generation | `GeminiClient` (`backend/integrations/gemini_client.py`) | ✓ | 1.5-pro preview |
| Gemini Vision (Image) | same as above | ✓ | `/chat/analyze-image` endpoint |
| Cypher Generation (NLQ) | `GeminiClient.generate_cypher_query` | ✓ | Converts natural language → Cypher |
| Fraud ML Toolkit (XGBoost removed) | `backend/agents/tools/fraud_ml_tool.py` | ✓ | Traditional classifiers |
| GNN Fraud Detection | `backend/agents/tools/gnn_*` | 🚧 | Models training pipeline exists; serving API pending |
| Time-series Anomaly Detection | `backend/agents/tools/crypto_anomaly_tool.py` (ADTK) | ✓ | Crypto market patterns |
| Code-Execution Sandbox | `e2b.dev` via `e2b_client.py` | ✓ | Runs python notebooks safely |

---

## 3 · Graph & Data Storage

| Capability | Module / Path | Status | Notes |
|------------|---------------|--------|-------|
| Neo4j 5 Driver (Bolt) | `backend/integrations/neo4j_client.py` | ✓ | Singleton pool; schema auto-init |
| Graph Schema Introspection API | `GET /graph/schema` | ✓ | Exposes labels, rel types, indexes |
| Cypher Query API | `POST /graph/query` | ✓ | Raw Cypher exec |
| Natural-Language Graph Query | `POST /graph/nlq` | ✓ | NLQ → Cypher |
| Graph Centrality Metrics | `POST /graph/centrality` | 🛠 | Planned GDS algorithms endpoint |
| Graph-based Entity Storage | `_store_entities_in_graph` helper | 🚧 | Basic node insert; rel logic TODO |

---

## 4 · Workflow & HITL

| Capability | Module / Path | Status | Notes |
|------------|---------------|--------|-------|
| Crew Pause / Resume | `backend/api/v1/crew.py` | ✓ | Awaiting human review |
| Compliance Review Webhooks | `backend/api/v1/webhooks.py` | ✓ | Slack/Email/Teams + custom URL |
| Review Callback Handling | same file | ✓ | Updates task status |
| Persistent Review Storage | — | 🛠 | Planned Postgres schema `hitl_reviews` |
| Conversation Persistence | `backend/api/v1/chat.py` | 🚧 | In-memory → Postgres migration 🛠 |

---

## 5 · Security & Compliance

| Capability | Module | Status | Notes |
|------------|--------|--------|-------|
| JWT Access & Refresh | `auth/jwt_handler.py` | ✓ | Refresh rotation roadmap 🛠 |
| Role-based Access Control | `auth/rbac.py` | ✓ | Fine-grained scopes |
| Rate Limiting | SlowAPI (config pending) | 🛠 | Not yet enabled |
| Secrets via `.env` | `.env.example` | ✓ | Docker secrets integration roadmap 🛠 |
| Security Scanning CI | Bandit, Safety, npm-audit | ✓ | GH Actions job `security-scan` |

---

## 6 · Observability & DevOps

| Capability | Module / Tool | Status | Notes |
|------------|---------------|--------|-------|
| Structured JSON Logging | `backend/core/logging.py` (structlog) | ✓ | Console + file |
| Prometheus Metrics | `backend/core/metrics.py` | ✓ | `/metrics` endpoint |
| Sentry Error Reporting | Integration scaffold | 🚧 | DSN placeholder |
| GitHub Actions CI Matrix | `.github/workflows/ci.yml` | ✓ | Python & Node versions |
| Code Coverage Upload | Codecov | ✓ | Separate flags: backend / frontend |
| CodeQL Static Analysis | GH `codeql-analysis` job | ✓ | Python + JS |

---

## 7 · Front-end UI

| Capability | Path | Status | Notes |
|------------|------|--------|-------|
| Next.js 14 App Router | `frontend/src/app` | ✓ | TS, Tailwind |
| Auth Pages (Login/Register) | `frontend/src/app/login`, `/register` | ✓ | Hooks to JWT API |
| Analysis Dashboard | `/analysis`, `/dashboard` | ✓ | Graph vis + risk scoring |
| Prompt Management UI | `/prompts` | ✓ | Agent prompt CRUD |
| React Query v5 Integration | global provider | ✓ | Server-state cache |
| ESLint + Prettier | config files | ✓ | CI gate |
| Unit / Component Tests | Jest + RTL | 🚧 | Scaffold ready (coverage goal 70 %) |
| E2E Tests (Playwright) | — | 🛠 | Roadmap Q2 |

---

## 8 · Testing & Quality Gates

| Capability | Tool | Status | Notes |
|------------|------|--------|-------|
| Backend Unit & Integration Tests | Pytest (+asyncio, cov) | ✓ | ~58 % coverage |
| Frontend Unit Tests | Jest + RTL | 🚧 | Seed tests committed |
| API Contract Tests | Pytest ‑ `test_api_*` | ✓ | FastAPI testclient |
| Lint / Type-check Gates | Ruff, Mypy, ESLint, tsc | ✓ | Fail build on error |
| CI Security Gates | Bandit, Safety, npm-audit | ✓ | High severity blocking |

---

## 9 · Deprecated / Removed

| Item | Reason | Replacement |
|------|--------|-------------|
| `xgboost` dep | Large wheels, CI timeouts | Standard scikit-learn models |
| `spacy`, `transformers` | Gemini handles NLP | — |
| `web3`, `eth-account` | No on-chain tx signing required | — |

---

*Maintain this catalog with every significant feature merge.  
Add new rows, adjust status, and remove deprecated items to keep the team aligned.*  
