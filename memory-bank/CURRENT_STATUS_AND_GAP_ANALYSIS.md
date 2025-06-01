# CURRENT_STATUS_AND_GAP_ANALYSIS.md  
_Last updated: 01 Jun 2025_

The purpose of this document is to give any contributor an **at-a-glance view** of what the Analyst Augmentation Agent _can_ do today, what is only _partially_ available, and where **documented plans exceed implementation**. Use the ✅ / 🟡 / ❌ indicators throughout:

* ✅ **Working** – implemented, tested, runs end-to-end  
* 🟡 **Partial** – code stub or MVP exists but feature incomplete / un-wired  
* ❌ **Missing** – only described in docs or backlog  

---

## 1 · Executive Snapshot

| Layer                    | Status | Notes (evidence) |
|--------------------------|--------|------------------|
| **Backend FastAPI**      | ✅ | Boots, health endpoints, CORS, global error handlers |
| **CrewAI Engine**        | ✅ | Sequential crews run; metrics logged |
| **Core Agents (5)**      | ✅ | `nlq_translator`, `graph_analyst`, `fraud_pattern_hunter`, `compliance_checker`, `report_writer` YAML configs present |
| **Custom Tools (11)**    | 🟡 | All tool modules exist; `CodeGenTool` exec flow & `PolicyDocsTool` inference path need finishing |
| **Pattern Library (30+ motifs)** | ✅ | YAML schema + matching engine, unit tests 95 % cov. |
| **Authentication & RBAC**| 🟡 | JWT login/register live; Alembic migrations + Redis blacklist pending |
| **HITL Workflow**        | ✅ | Pause/resume endpoints, backend state, UI review component |
| **Neo4j Graph Layer**    | ✅ | Async driver, schema init, CSV loader, GDS integration |
| **Crypto Anomaly ML**    | 🟡 | `CryptoAnomalyTool` implemented; model tuning & prod wiring TODO |
| **Frontend UI**          | 🟡 | Auth pages & GraphVisualization done; analysis dashboard & HITL e2e missing |
| **CI/CD (GitHub/GitLab)**| ✅ | Ruff, mypy, pytest matrix, Docker build pass |
| **Prod Docker/Compose**  | 🟡 | Dev profile solid; prod images running but k8s/Helm not yet |
| **Observability**        | 🟡 | Prometheus metrics present; Loki/OTel traces planned |
| **Test Coverage 50 %**   | ✅ | Meets current gate; target 55 % |

---

## 2 · Detailed Capability Matrix

### 2.1 Back-End API

| API Area | Endpoints | Status | Limitations |
|----------|-----------|--------|-------------|
| Auth     | `/auth/login`, `/auth/register`, refresh | ✅ | Redis blacklist not enforced (token revocation) |
| Crew     | `/crew/run`, `/crew/pause`, `/crew/resume` | ✅ | RBAC guard missing on `run` (security gap) |
| Analysis | `/analysis/*` | 🟡 | Stubs exist, not wired to agents yet |
| Graph    | `/graph/query`, `/graph/upload` | ✅ | Schema validation lenient |
| Prompts  | `/prompts` CRUD | ✅ | No optimistic concurrency checks |
| Webhooks | Compliance HITL | ✅ | Only local HTTPS tested |

### 2.2 Agents & Tools

| Component | Purpose | Status | Gap |
|-----------|---------|--------|-----|
| `nlq_translator` | NL ➜ Cypher | ✅ | Needs advanced schema awareness |
| `graph_analyst`  | Multi-hop + GDS | ✅ | Path-length cap hard-coded |
| `fraud_pattern_hunter` | Motif match + anomaly score | ✅ | Graph ML integration pending |
| `compliance_checker` | Policy doc cross-check + HITL | ✅ | PolicyDocsTool rules stubbed (🟡) |
| `report_writer` | Narrative Markdown | ✅ | TemplateEngineTool partial (🟡) |
| `CodeGenTool` + `SandboxExecTool` | Auto-Python + safe exec | 🟡 | Execution result not routed back to crew |
| `PolicyDocsTool` | Retrieve AML policy snippets | 🟡 | Retrieval logic + embeddings missing |

### 2.3 Front-End

| Area | Status | Missing |
|------|--------|---------|
| Auth UI (Login/Register) | ✅ | Form validation UX polish |
| Dashboard / Prompt Editor | ✅ | Real-time SSE updates |
| Graph Visualization | ✅ | Schema validation, legend filter UI |
| HITL Review Panel | 🟡 | Integration test + decision audit trail |
| Analysis Results View | ❌ | Not yet built |

### 2.4 Data & ML

| Capability | Status | Notes |
|------------|--------|-------|
| CSV → Neo4j loader | ✅ | Multi-thread ingest, metrics |
| Time-series anomaly (ADTK) | 🟡 | Model parameters defaults only |
| Fraud ML classification | 🟡 | RandomForest/XGB notebooks; not productionised |
| Risk propagation scoring | ❌ | Planned Phase 3 |

### 2.5 DevOps & Quality

| Item | Status | Notes |
|------|--------|-------|
| Docker Dev Compose | ✅ | One-liner up |
| Docker Prod Images  | 🟡 | Multistage built; k8s charts TBD |
| CI Pipeline | ✅ | 3× Python versions, Neo4j service |
| GitLab Mirror | 🟡 | `git push gitlab` manual; CI config warning harmless |
| Test Coverage ≥ 55 % | ❌ | Currently 50 %, target unmet |
| Sentry / Loki logging | ❌ | Not integrated |

---

## 3 · “Should Do” vs “Can Do” Examples

| Use Case | Expected Behaviour (Docs) | Actual Behaviour (Today) | Indicator |
|----------|---------------------------|--------------------------|-----------|
| _Fraud Investigation_ | NL query → Cypher → Graph → Pattern + ML → Compliance HITL → Report in ≤ 30 s | Works end-to-end; ML optional step returns placeholder score | ✅ |
| _Real-Time Alert Enrichment_ | SLA < 5 s, light pipeline (no sandbox) | Pipeline exists but no SLA enforcement; benchmark 6-7 s avg | 🟡 |
| _Policy Compliance Check_ | Auto-cross reference to policy docs, flag redacted terms | Checker pauses for HITL but uses hard-coded rules, no doc retrieval | 🟡 |
| _Automated Python Analytics_ | Agent generates code, sandbox executes, results merged into report | Code executes; result JSON not fed back → report lacks charts | 🟡 |
| _Advanced Schema Awareness_ | NLQ understands temporal / risk propagation schema | Only base node/rel labels; temporal unsupported | ❌ |
| _Frontend Analysis Dashboard_ | Rich summary cards, graph filter, export CSV | Not implemented | ❌ |
| _Observability_ | Traces + logs across agents, dashboards | Only Prometheus counters | 🟡 |

---

## 4 · Key Gaps & Recommended Next Actions

| Priority | Gap | Impact | Action |
|----------|-----|--------|--------|
| P0 | `CodeGenTool` result integration | Reports missing ML charts | Finish exec→crew glue; test |
| P0 | RBAC protection on `/crew/run` | Security exposure | Apply `require_roles(['analyst','admin'])` |
| P0 | Alembic migration for `users` | Prod auth fails cold start | Generate & commit revision |
| P1 | Redis token blacklist | Logout / revoke broken | Implement token store & tests |
| P1 | PolicyDocsTool retrieval | Compliance automation weak | Embed docs in Redis vector, semantic search |
| P1 | Front-end Analysis View | UX incomplete | Build results page, connect to API |
| P1 | Test coverage ≥ 55 % | CI gate upcoming | Add tool/FE tests, Playwright E2E |
| P2 | Observability (OTel, Loki) | Hard to debug prod | Add FastAPI middleware, helm charts |
| P2 | Risk propagation model | Incomplete scoring | Design temporal schema, implement proc |

---

## 5 · Conclusion

The project is **functionally viable for supervised fraud investigations**. Core agents, graph tooling, pattern library, and HITL workflow run successfully.  
To progress toward **Phase 3 “Integrations & Ecosystem”**, focus on closing P0 / P1 gaps: production-grade security (RBAC, token revocation), complete tool execution paths, richer compliance automation, and finish the remaining UI & coverage work.

_This analysis should be updated at the end of every development cycle to maintain a reliable picture of system health and roadmap alignment._
