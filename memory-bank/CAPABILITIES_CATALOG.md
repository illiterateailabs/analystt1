# Capabilities Catalog  
*File: `memory-bank/CAPABILITIES_CATALOG.md` · last updated 2025-06-20*

A structured reference of **all functional capabilities** in the Analystt1 platform.  
Use this catalog to discover existing features, understand integration points, and track roadmap status.

Legend | Meaning  
-------|---------  
✓ Implemented | Feature is live in `main` / `mina`  
🚧 In-progress | Work started (open PR / flagged)  
🔧 Planned | Approved roadmap item, no code yet  
✗ Deprecated | Superseded / removed  

---

## 1 · Blockchain Data Analysis  

| Capability | Module / Path | Status | Notes |
|------------|---------------|--------|-------|
| Multi-chain Balances | `sim_balances_tool.py` | ✓ | 60 + EVM chains |
| Chronological Activity | `sim_activity_tool.py` | ✓ | Transfers, swaps, approvals, calls |
| NFT Collectibles | `sim_collectibles_tool.py` | ✓ | ERC721 / 1155, OpenSea enrichment |
| Transaction Details | `sim_graph_ingestion_tool.py` + Sim `/transactions` proxied | ✓ | Raw tx for forensic drill-down |
| Token Metadata & Liquidity | `sim_token_info_tool.py` | ✓ | Price, decimals, pool_size |
| Token Holder Distribution | `sim_token_holders_tool.py` | ✓ | Whale concentration analysis |
| Solana Balances / Tx | `sim_svm_balances_tool.py` | ✓ | SVM beta routes |

---

## 2 · 🐋 Whale Movement Tracking (**NEW MAJOR FEATURE**)  

| Sub-Capability | Module / Path | Status | Notes |
|----------------|---------------|--------|-------|
| WhaleDetectionTool | `backend/agents/tools/whale_detection_tool.py` | ✓ | Tier1 / Tier2 / Active classification |
| Large Movement Feed | `WhaleDashboard.tsx` | ✓ | ≥ $100 k tx feed (configurable) |
| Coordination Pattern Detection | same tool | ✓ | DISTRIBUTION / ACCUMULATION / CIRCULAR |
| Real-time Monitoring API | `backend/api/v1/whale_endpoints.py` `/monitor` | ✓ | Alerts with confidence scores |
| Whale Statistics | `/whale/stats` | ✓ | Tier counts, total value, chain split |
| Graph Integration | event `GraphAddEvent(type='whale_detection')` | ✓ | Nodes & edges for whale analytics |

---

## 3 · Risk Assessment Tools  

| Capability | Module | Status | Notes |
|------------|--------|--------|-------|
| Wallet Risk Score | `/sim/risk-score/{wallet}` | ✓ | Liquidity, velocity, approvals |
| Token Liquidity Flagging | part of Balances API | ✓ | `low_liquidity` boolean + pool_size |
| Whale Risk Level | Whale detection stats | ✓ | Tier-based & behaviour risk |

---

## 4 · Fraud Detection Patterns  

| Pattern | Detection Logic | Module / Tool | Status |
|---------|-----------------|---------------|--------|
| Low-liquidity Dump | Low pool_size + large sell | Risk Score + Activity | ✓ |
| Peel-Chain Structuring | Deep send chains | Pattern library YAML + graph query | ✓ |
| NFT Wash Trading | Same NFT round-trips | Collectibles + Activity | ✓ |
| Whale Concentration | Top-10 holders > 90 % | Token-Holders tool | ✓ |
| Circular Tx | A→B→C→A cycles (≥3) | WhaleDetectionTool | ✓ |
| Bridge Abuse | Outflow chain A → inflow chain B < 5 min | Multi-chain Balances + Activity | 🔧 |

---

## 5 · Data Sources & Integrations  

| Source / Service | Purpose | Status |
|------------------|---------|--------|
| **Sim APIs** | Real-time on-chain data (EVM + SVM) | ✓ |
| OpenSea API | NFT image / metadata enrichment | ✓ |
| Gemini LLM | Code & query generation, explanations | ✓ |
| Neo4j 5 | Graph persistence & analytics | ✓ |
| PostgreSQL | Auth, conversations, HITL reviews | ✓ |
| Prometheus | Metrics collection | ✓ |
| Sentry | Error telemetry | 🚧 scaffold |

---

## 6 · Analytics & Visualization  

| Capability | Path / Component | Status | Notes |
|------------|-----------------|--------|-------|
| WalletAnalysisPanel | `WalletAnalysisPanel.tsx` | ✓ | Tokens / Activity / NFTs / Risk |
| WhaleDashboard | `WhaleDashboard.tsx` | ✓ | Overview, Whales, Movements, Coordination |
| Progress Visualization | `TaskProgress.tsx` | ✓ | Real-time WebSocket updates |
| Graph Explorer (future) | planned d3 / cytoscape view | 🔧 | Q3 roadmap |

---

## 7 · User Interface Capabilities  

| Feature | Path | Status | Notes |
|---------|------|--------|-------|
| Next.js 14 App Router | `frontend/src/app` | ✓ | SSR + static export |
| Auth Pages | `/login`, `/register` | ✓ | JWT flow |
| Dashboard & Analysis | `/dashboard`, `/analysis` | ✓ | Core workflows |
| Prompt Manager | `/prompts` | ✓ | CrewAI prompt CRUD |
| Whale Tracker Tab | `/analysis` top-level Tabs | ✓ | Real-time tracking |
| Responsive Design | TailwindCSS | ✓ | Mobile-ready |
| Unit Tests (RTL) | `frontend/**/__tests__` | ✓ 70 % coverage |

---

## 8 · API Capabilities  

| Endpoint Group | Swagger Tag | Status |
|----------------|-------------|--------|
| Auth & JWT | Authentication | ✓ |
| Chat & LLM | Chat | ✓ |
| Analysis & Code | Analysis | ✓ |
| Whale Tracking | Whale | ✓ |
| Graph Ops | Graph | ✓ |
| Crew Management | Crew | ✓ |
| Prompts & Templates | Prompts | ✓ |
| Webhooks | Webhooks | ✓ |
| WebSockets (task progress) | WS | ✓ |

---

## 9 · Security & Compliance  

| Feature | Implementation | Status |
|---------|----------------|--------|
| JWT Access & Refresh | `auth/jwt_handler.py` | ✓ |
| RBAC Decorators | `auth/rbac.py` | ✓ |
| CSRF-safe Cookies | `auth/secure_cookies.py` | ✓ |
| Rate Limiting | SlowAPI middleware | ✓ default 100/min |
| Secrets Management | `.env` + Docker secrets roadmap | ✓ / 🔧 |
| Dependency Scanning | Bandit, Safety, npm-audit | ✓ |
| Static Analysis | Ruff, Mypy, ESLint, TS | ✓ |
| Error Monitoring | Sentry SDK (backend) | 🚧 |

---

## 10 · Performance & Scalability  

| Aspect | Mechanism | Status | Notes |
|--------|-----------|--------|-------|
| Async I/O | FastAPI + httpx + async SQLAlchemy | ✓ |
| Background Jobs | `sim_graph_job.py` (async) | ✓ |
| Pagination | Cursor-based across Sim APIs | ✓ |
| Retry / Backoff | Tenacity + custom logic | ✓ |
| WebSockets | Real-time progress & alerts | ✓ |
| CI Matrix | Py 3.10/3.11 & Node 18/20 | ✓ |
| Dockerised Services | `docker-compose.yml` | ✓ |
| Horizontal Scaling | Containers + Neo4j cluster-ready | 🔧 roadmap |

---

*Maintain this catalog on every significant merge to keep the entire team aligned on **what the platform can do** and **where it is headed**.*  
