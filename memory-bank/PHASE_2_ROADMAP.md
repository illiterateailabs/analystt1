# Phase 2 Feature Enhancement Roadmap  
**File:** `memory-bank/PHASE_2_ROADMAP.md`  **Version:** v2.0-draft  **Last updated:** 2025-07-07  

---

## 🎯 Phase 2 Goal  
Deliver the next wave of functionality that turns Analyst Droid One into a SaaS-ready, real-time fraud analysis platform:

| # | Objective | Outcome |
|---|-----------|---------|
| 1 | Advanced Graph Algorithms | Deeper, faster, and more explainable fraud pattern discovery |
| 2 | Real-Time Streaming | Sub-second monitoring of on-chain transactions & anomalies |
| 3 | ML Model Integration | Automated, continuously-learning risk scoring service |
| 4 | Multi-Tenant Architecture | Secure isolation & self-service onboarding for multiple customers |

Target duration: **6 weeks** (2025-07-08 → 2025-08-19)  
Release tag: **v2.0.0-beta**

---

## 🗺️ High-Level Timeline  

| Week | Theme Focus | Key Deliverables |
|------|-------------|------------------|
| **0 (Prep)** | Planning & scaffolds | Design docs, branch `phase-2`, CI matrix update |
| **1** | Graph Algorithms (part 1) | GAT model PoC, Neo4j GDS community detection |
| **2** | Graph Algorithms (part 2) | Production GAT service, risk-propagation algorithm, tests |
| **3** | Real-Time Streaming | Kafka/Redis Streams ingestion, `/ws/tx_stream`, dashboard widget |
| **4** | ML Risk Service | Ensemble model API, Celery retraining tasks, model registry |
| **5** | Multi-Tenant Core | Tenant context propagation, RBAC scopes, Neo4j multi-DB |
| **6** | Tenant UX & Hardening | Tenant admin UI, Helm chart updates, load/perf tests, docs |

*Continuous:* QA automation, Grafana dashboards, documentation.

---

## 1 · Advanced Graph Algorithms

### 1.1 Epic Breakdown
| ID | Task | Tech Details | Owner | Done When |
|----|------|--------------|-------|-----------|
| GA-1 | Implement Graph Attention Network (GAT) layer in `gnn_fraud_detection_tool.py` | PyTorch Geometric; input from Neo4j via DataLoader; export risk score | ML / Graph | Model F1 ≥ 0.82 on test set |
| GA-2 | Add community detection via Neo4j GDS (`louvain`, `label-prop`) | Cypher procedures; store `:Community` nodes + metrics | Graph Team | Communities persisted & visualised |
| GA-3 | Risk-propagation algorithm | Propagate risk along high-value edges; write as APOC custom proc | Graph Team | Risk scores cached, Prom metric exposed |
| GA-4 | Integrate algorithms into CrewAI toolchain | New `advanced_graph_tool.py`; expose via `/api/v1/graph/advanced` | Backend Team | Endpoint returns JSON & Cypher explanation |

### 1.2 Dependencies
* Neo4j ≥ 5.15 with GDS plugin  
* GPU runner in CI for GAT tests (GitHub Actions matrix)

---

## 2 · Real-Time Streaming

### 2.1 Epic Breakdown
| ID | Task | Tech Details | Owner | Done When |
|----|------|--------------|-------|-----------|
| RT-1 | Select streaming stack | Compare Kafka vs Redis Streams; decide & document | Infra Lead | ADR merged |
| RT-2 | Build `stream_ingestor` service | Python async worker reads SIM WebSocket → streams to topic | Backend | Up to 5 k TPS on dev machine |
| RT-3 | WebSocket relay endpoint `/ws/tx_stream` | FastAPI endpoint subscribes to topic; filter by tenant | Backend | Frontend receives <1 sec latency |
| RT-4 | Frontend live dashboard widget | Next.js component using Zustand; heat-map & alerts | FE Team | Widget renders with mock + live data |
| RT-5 | Alert rules & metrics | Prometheus: `tx_stream_lag_seconds`, `anomaly_rate` | DevOps | Grafana panel + alert >5 s lag |

---

## 3 · ML Model Integration

### 3.1 Epic Breakdown
| ID | Task | Tech Details | Owner | Done When |
|----|------|--------------|-------|-----------|
| ML-1 | Ensemble risk scoring service | FastAPI micro-service; XGBoost + lightGBM ensemble | ML Team | `/risk/score` returns JSON with ≤100 ms latency |
| ML-2 | Celery retraining pipeline | Daily job pulls labeled data, retrains, pushes model to registry | ML Team | Model artefact in S3 + version tag |
| ML-3 | Model registry & CI checks | Use MLflow; GitHub action to test model metrics drift | DevOps | PR blocked if AUC < previous |
| ML-4 | Integrate risk scores into EvidenceBundle | Update schema; CrewAI agents consume service | Backend | Evidence JSON includes `risk_score` field |

### 3.2 Security
* Service behind internal network; JWT service-to-service tokens  
* Rate-limit via back-pressure middleware

---

## 4 · Multi-Tenant Architecture

### 4.1 Epic Breakdown
| ID | Task | Tech Details | Owner | Done When |
|----|------|--------------|-------|-----------|
| MT-1 | Tenant context propagation | Add `X-Tenant-ID` header; Pydantic `TenantContext` dependency | Backend | All APIs enforce tenant filter |
| MT-2 | Database isolation strategy | Neo4j multi-database (`tenant_$id`); Postgres schemas | DB Team | Data leak tests pass |
| MT-3 | Auth & RBAC scopes | Extend JWT claims (`tenant_id`, `role`) | Auth Team | Auth tests green |
| MT-4 | Tenant admin UI | Next.js pages: tenant list, usage, billing hooks | FE Team | UI passes Cypress tests |
| MT-5 | Billing & quota hooks | Provider registry per tenant; Prom counter `tenant_api_cost_total` | DevOps | Budget alerts fire correctly |
| MT-6 | Helm chart multi-tenant values | Template extra databases, env vars | Infra Lead | Helm lint & kind deploy succeed |

---

## 5 · Cross-Cutting Tasks

| Area | Task |
|------|------|
| **Observability** | Expand OTEL spans to new services; Grafana dashboards for each objective |
| **QA / Testing** | Add Playwright e2e for streaming + tenant switch; load test with `scripts/end_to_end_load_test.py` |
| **Docs** | Update `TECHNICAL_ARCHITECTURE.md`, write `ADD_NEW_TENANT_COOKBOOK.md`, upgrade README badges |
| **Security** | TLS everywhere, SlowAPI rate-limit on new endpoints, Snyk scan in CI |
| **Release** | Tag `v2.0.0-beta`, changelog, migration guide |

---

## ✅ Definition of Done
1. All epic tasks complete & merged into `main`  
2. CI pipeline green, test coverage ≥80 %  
3. Grafana dashboards deployed, alerts configured  
4. Documentation updated (memory-bank + README)  
5. Demo video recorded for stakeholders  

---

## 🚩 Risks & Mitigation
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Kafka operational complexity | Medium | Fallback plan: Redis Streams; use managed Kafka for prod |
| GPU shortages in CI | Low | Conditional job; skip GPU tests if runner unavailable |
| Multi-DB Neo4j licensing limits | Medium | Validate with OSS edition; alternative: label-based isolation |
| Model drift false positives | Medium | Add drift monitoring & rollback pipeline |

---

## 📚 References
* `PROJECT_STATUS.md` – current system status  
* Neo4j GDS docs, PyG documentation  
* ADR-12 — Streaming Stack Decision  
* ADR-13 — Tenant Isolation Strategy  

*Author: Factory Assistant · generated 2025-07-07*
