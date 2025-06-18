# Master Status – Analyst Augmentation Agent  

_File: `memory-bank/MASTER_STATUS.md` – updated 2025-06-18_

---

## 1 · Project Snapshot
| Item | Value |
|------|-------|
| **Current Version** | **1.1.0-beta** (“Sim-Data cut”) |
| **Latest Commit** | `9680149d` (🚀 Sim API integration) |
| **Deployed Envs** | • Dev (Docker Compose) ✅ • CI preview (GH Actions) ✅ • Prod (staging cluster) ⏳ awaiting QA |
| **Maintainers** | Backend @Daniel-Wurth • Frontend @UI-Lead • DevOps @Ops-Guru |

---

## 2 · Current Functionality

| Domain | Status | Notes |
|--------|--------|-------|
| **Auth / RBAC** | ✅ | JWT (HS256) with role scopes; secrets centralised in `.env` |
| **Sim API Ingestion** | ✅ Backend · ⚠️ FE wiring | Balances & Activity tools merged; Graph events emit; UI still uses mocks |
| **Chat & Image Analysis** | ✅ | Gemini 1.5-pro; persistent convo log TODO |
| **CrewAI Workflow** | ✅ | Pause / Resume, HITL webhooks, task progress WS |
| **Graph API** | ✅ | Cypher exec, NLQ → Cypher, schema introspection |
| **Data Stores** | ✅ | PostgreSQL 15 (async SQLAlchemy), Neo4j 5 |
| **Observability** | ⚠️ | Prometheus metrics exporting; Sentry wiring TODO |
| **Frontend UI** | ⚠️ | Next 14 App Router; Sim data binding pending; tests scaffolded |
| **CI / Security** | ✅ | GH Actions matrix (Py 3.9-3.11, Node 18-20) + Bandit / Safety / npm-audit |

---

## 3 · Deployment Status
| Environment | Image Tags | Last Deploy | Health |
|-------------|------------|-------------|--------|
| **Dev-Compose** | `backend:sim-data` `frontend:sim-data` | 2025-06-18 | Passing |
| **GH CI Preview** | ephemeral on-push | 2025-06-18 | All checks green |
| **Staging Cluster** | pending | — | — |

_Key changes in this cut_  
* SimClient, BalancesTool, ActivityTool with tenacity retry & metrics  
* Graph events (`wallet_balances`, `wallet_activity`) flowing to Neo4j  
* `SIM_API_KEY` + `SIM_API_URL` in `.env.example`  
* Memory-bank plan added (`SIM_API_INTEGRATION_PLAN.md`)

---

## 4 · Quality & Coverage
| Suite | Tests | Coverage | Trend |
|-------|-------|----------|-------|
| **Backend (pytest)** | 512 | **60 %** statements | ▲ +2 % |
| **Frontend (Jest/RTL)** | 12 | 5 % | ▲ +4 % |
| **Integration E2E** | 0 | — | Planned |

Static Analysis  
* **Ruff** – 0 errors  
* **Mypy** – clean on `backend/`  
* **ESLint** – 28 warnings (accessibility)

---

## 5 · Known Issues / Risks
1. Conversation & webhook data still in-memory → loss on restart  
2. Access / Refresh JWTs use localStorage (XSS risk) – move to httpOnly cookies  
3. Frontend still displays **mock** balances / activity → user confusion  
4. Sentry DSN placeholder; error telemetry disabled in prod  
5. Bundle not yet tree-shaken; large initial JS (≈ 1 MB)

---

## 6 · Next Priorities (Q2 · Sprint 6)

| Priority | Epic / Task | Owner |
|----------|-------------|-------|
| **P0** | Wire Sim **Balances & Activity** into UI components | Frontend |
| **P0** | Migrate conversations & HITL reviews to PostgreSQL (Alembic 003) | Backend |
| **P1** | Graph enrichment job for Sim data (Neo4j loader) | Data Graph |
| **P1** | Enable refresh-token rotation & httpOnly cookie auth | Backend |
| **P2** | Finish FE test harness (reach 70 % coverage) | Frontend |
| **P2** | Integrate Sentry & OTEL traces end-to-end | DevOps |
| **P3** | Add e2e Playwright suite (chat + analysis flow) | QA |
| **P3** | Optimize FE bundle (code-splitting, RSC) | Frontend |

---

## 7 · Recent Changelog (since v1.0.0-beta)

* **2025-06-18 – Sim Data integration (#79, #sim-tools)**  
  * Added `SimClient`, `sim_balances_tool.py`, `sim_activity_tool.py`  
  * Graph events + Pydantic schemas + retry / metrics  
  * Memory-bank `SIM_API_INTEGRATION_PLAN.md`  
* **2025-06-17 – Critical-fixes PR #71 merged**  
  * Fixed Neo4j import & driver singleton, added `.env.example`, removed dup `config_jwt.py`, ESLint/Prettier/Jest scaffolding  
* **2025-06-10 – GNN fraud-detection tools integrated (#68)**  
* **2025-06-03 – HITL webhook system MVP (#64)**  

---

## 8 · Glossary
| Term | Meaning |
|------|---------|
| **Sim APIs** | Dune’s real-time multichain blockchain data service |
| **HITL** | Human-In-The-Loop review |
| **CrewAI** | Multi-agent orchestration framework |
| **NLQ** | Natural-Language-to-Cypher query generation |

---
