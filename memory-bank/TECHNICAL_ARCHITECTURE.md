# Technical Architecture  
*File `memory-bank/TECHNICAL_ARCHITECTURE.md` – updated 2025-06-17*

This document captures the **implemented** architecture of the Analyst Augmentation Agent after the _Critical Fixes (PR #71)_ merge.

---

## 1 · High-Level Overview

```
              ┌───────────────────────────┐
              │  React / Next.js 14 App   │
              │ (TypeScript + Tailwind)   │
              └──────────────┬────────────┘
                             │ HTTPS / WS
                             ▼
              ┌───────────────────────────┐
              │ FastAPI Backend (Py 3.11) │
              │  ├── Auth & RBAC          │
              │  ├── Chat & Analysis APIs │
              │  ├── CrewAI Orchestrator  │
              │  ├── Graph API (Cypher)   │
              │  ├── Webhooks (HITL)      │
              │  └── WS Progress Streams  │
              └──────────────┬────────────┘
       ┌──────────┬──────────┼───────────┬─────────┐
       │          │          │           │         │
       ▼          ▼          ▼           ▼         ▼
 PostgreSQL   Neo4j 5    Redis 7   Google Gemini  e2b.dev
  (asyncpg)   (Bolt)     (Cache)    (LLM API)   (Sandbox)
```

---

## 2 · Technology Stack

| Layer | Tech | Notes |
|-------|------|-------|
| **Frontend** | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, @tanstack/react-query v5 | Jest + RTL for tests, ESLint + Prettier |
| **Backend** | Python 3.11, FastAPI 0.111, SQLAlchemy 2 (async), Pydantic 2, CrewAI 0.119 | Ruff, Mypy, Pytest |
| **Graph** | Neo4j 5.15 (community), APOC + GDS plugins | Async driver, singleton pool |
| **Relational** | PostgreSQL 15 (alpine) | QueuePool connection pooling |
| **Cache / Rate-limit** | Redis 7 | Prepared for SlowAPI throttling |
| **LLM** | Google Gemini 1.5-pro | Vision + text endpoints |
| **Sandbox** | e2b.dev “python-data-science” template | Safe code exec |
| **CI/CD** | GitHub Actions matrix, Codecov, Bandit/Safety, npm-audit, CodeQL | PR gates |
| **Containerisation** | Docker, docker-compose for dev; prod images to container registry | K8s Helm chart (planned) |

---

## 3 · Component Breakdown

### 3.1 Backend Services
| Module | Path | Responsibility |
|--------|------|----------------|
| **Auth** | `backend/auth` | JWT generation, role enforcement |
| **API v1** | `backend/api/v1` | REST endpoints (`/auth`, `/chat`, `/analysis`, `/graph`, `/crew`, `/prompts`, `/webhooks`) |
| **Integrations** | `backend/integrations` | `neo4j_client.py`, `gemini_client.py`, `e2b_client.py` |
| **Crew Engine** | `backend/agents/*` | Multi-agent workflows, tool plugins |
| **Core** | `backend/core` | Logging, events, metrics (Prometheus) |
| **Database** | `backend/database.py` | Async engine, session maker, pooling strategy |
| **Main App** | `backend/main.py` | FastAPI app factory, startup/shutdown hooks, CORS |

### 3.2 Frontend
| Area | Path | Highlights |
|------|------|-----------|
| **App Router** | `frontend/src/app` | Pages: `/login`, `/analysis`, `/dashboard`, etc. |
| **Shared Libs** | `frontend/src/lib` | `api.ts` Axios wrapper, auth, constants |
| **State** | React Query v5 | Server-state caching |
| **Testing** | Jest, Testing-Library | Coverage threshold ≥ 70 % (goal) |

---

## 4 · Data-Flow Scenarios

### 4.1 User Chat with Graph Query
1. **Browser** → `POST /api/v1/chat/message`  
2. Backend passes message to **Gemini** for intent detection & (optional) Cypher generation.  
3. If Cypher generated → **Neo4j** queried, results streamed back.  
4. Assistant reply composed and returned; conversation stored in PostgreSQL _(pending)_ / in-memory (current).  
5. WS endpoint `/api/v1/ws/progress/{task}` streams task states.

### 4.2 CrewAI Workflow with HITL
1. Frontend triggers `POST /crew/run` – crew tasks execute in async workers.  
2. When policy rule matched, backend calls `/webhooks/notify/compliance-review`.  
3. Slack/Teams/email endpoints receive payload; reviewer hits callback URL.  
4. Crew resumes via `/crew/resume/{task}` with reviewer verdict.  
5. Results persisted, Neo4j updated.

### 4.3 Image Analysis
1. FE uploads image → `/chat/analyze-image` multipart.  
2. Gemini Vision returns caption + entities.  
3. Entities optionally inserted as Neo4j nodes.  
4. Response includes analysis, stored graph IDs, base64 visualisations.

---

## 5 · Deployment Architecture

### 5.1 Development (docker-compose)
* Services: `backend`, `frontend`, `postgres`, `neo4j`, `redis`, `jupyter`  
* Hot-reload enabled (`uvicorn --reload`, `next dev`)  
* `.env` mounted; secrets local.

### 5.2 CI Pipeline
1. **Checkout** → install deps (cache)  
2. **Backend Job**: ruff → mypy → pytest + coverage → Bandit / Safety  
3. **Frontend Job**: ESLint → type-check → Jest + coverage → npm-audit  
4. **Upload artefacts** to Codecov & CodeQL.

### 5.3 Production (target)
* Container images pushed to registry.  
* K8s (Helm) or Docker Swarm cluster:  
  * **backend** deployment (N replicas, Horizontal Pod Autoscaling)  
  * **frontend** served via Nginx sidecar or Edge CDN  
  * **postgres** & **neo4j** managed DBaaS or StatefulSets  
  * **redis** for caching/rate-limit  
* Traffic ingress through TLS-terminated load balancer.  
* Observability stack: Prometheus + Grafana dashboards, Sentry for errors.

---

## 6 · Cross-Cutting Concerns

* **Security**: CORS locked to FE hosts; secrets via env; JWT HS256; upcoming httpOnly cookies.  
* **Observability**: Structured JSON logs → log aggregator; Prometheus metrics exposed at `/metrics`; TODO: OpenTelemetry traces.  
* **Scalability**: Async FastAPI, pooled DB drivers, stateless containers; Neo4j & Postgres scale vertically; queue workers (future).  
* **Resilience**: Health probes (`/health`, `/health/neo4j`); graceful shutdown; retries for webhooks with exponential back-off.

---

## 7 · Future Enhancements

1. Persist conversations & reviews to PostgreSQL (Alembic 003).  
2. Introduce task queue (Celery / Huey or built-in CrewAI runner) for long-running jobs.  
3. Migrate Auth tokens to secure cookies + refresh-rotation.  
4. Enable Sentry + OTEL full trace.  
5. E2E Playwright suite covering chat → graph → analysis flows.  

---

_This document supersedes all earlier architecture drafts. Update on each major architectural change._  
