# CAPABILITIES QUICK REFERENCE  
_Last updated: 01 Jun 2025_

A one-page cheat-sheet of what the **Analyst Augmentation Agent** can (✅ CAN) and cannot (❌ CANNOT) do _today_. Use it to spot capability gaps at a glance.

---

## 1 · Authentication & Security
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Issue **JWT** access & refresh tokens via `/auth/login` & `/auth/register`. | Persist token blacklist to **Redis** (logout/revoke ineffective after restart). |
| Enforce **RBAC** on `/prompts`, `/graph`, HITL endpoints. | Guard `/crew/run` & `/analysis/*` (open to any authenticated user). |
| Hash passwords with **bcrypt**; store in PostgreSQL. | Run automatic Alembic migration for `users` table on deploy (manual step). |

---

## 2 · Data Integration & Storage
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Connect to **Neo4j 5.x** (Bolt + TLS), initialise base schema. | Handle multi-database Neo4j clusters (single DB only). |
| Load on-chain CSVs → Neo4j via `CryptoCSVLoaderTool` (~150k tx/min). | Stream blockchain APIs directly (Dune, Etherscan) – planned Phase 3. |
| Store user/auth data in **PostgreSQL** (asyncpg). | Auto-create/upgrade tables via migrations (pending revision). |

---

## 3 · Data Analysis (Agents & Tools)
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Translate NL → **Cypher** with `nlq_translator`. _Example_: “Show all transfers from 0xABC last 7 days”. | Understand temporal/risk propagation schema (simple labels only). |
| Execute graph queries & run **PageRank** with `graph_analyst`. | Run advanced GDS algos (betweenness, similarity). |
| Generate Python analytics in **e2b sandbox** (`CodeGenTool` + `SandboxExecTool`). | Feed sandbox results back into crew output. |

---

## 4 · Fraud Detection & Pattern Matching
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Match **30+** fraud motifs (wash trading, flash-loan, smurfing) via `PatternLibraryTool`. | Auto-update / download new motif packs (manual commit). |
| Detect time-series anomalies with `CryptoAnomalyTool` (ADTK). | Tune ADTK hyper-params per asset; integrate ML scores into risk. |
| Produce heuristic **risk score** (motif weight × count). | Combine ML classification (RandomForest/XGB) – integration pending. |

---

## 5 · Compliance & HITL
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Pause crew when `compliance_checker` flags sensitive content; resume via `/crew/resume`. | Retrieve AML policy excerpts via `PolicyDocsTool` (vector search stub). |
| Send generic JSON **webhook** to Slack/Email for review. | Persist HITL decision & comments to DB (in-memory only). |

---

## 6 · Reporting
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Render Markdown narrative in `report_writer` using **Jinja2** template. | Embed ML charts / sandbox outputs in report. |
| Return **graph JSON** for vis-network UI. | Export PDF / email report (planned Phase 3). |

---

## 7 · Frontend UI
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Login/Register pages with JWT handling. | Full **Analysis Results** dashboard (in design). |
| Interactive **GraphVisualization** (zoom, filter, PNG export). | Real-time SSE progress updates during long-running crews. |
| Prompt Editor for live agent prompt tweaks. | Persist UI settings per user. |

---

## 8 · API & Orchestration
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Launch predefined crews via `/crew/run` (`fraud_investigation`, `alert_enrichment`). | Dynamically compose crews from ad-hoc tool list. |
| Pause/Resume crews for HITL compliance. | Cancel/abort runaway crews mid-task. |
| Track Prometheus metrics: `llm_tokens_used_total`, `crew_task_duration_seconds`. | Provide OpenTelemetry traces or structured Loki logs. |

---

## 9 · DevOps & CI/CD
| ✅ CAN | ❌ CANNOT |
|--------|-----------|
| Run green **CI matrix** (Ruff, mypy strict, pytest on Py 3.9-3.11). | Enforce ≥ 55 % coverage gate (current 50 %). |
| Build multistage Docker images; dev compose spins up full stack. | Deploy via Helm/k8s with autoscaling & secrets. |

---

### Quick Examples

* **Tracing Funds**  
  `POST /crew/run` → returns risk-scored graph, _status: PAUSED_FOR_HITL_ if sensitive. ✅ Works end-to-end (report lacks ML charts).

* **Real-Time Alert Enrichment**  
  `POST /analysis/enrich` with alert JSON → returns enrichment in ~6-7 s. 🟡 SLA 5 s unmet.

---

**Bottom Line:** The system handles **investigation & enrichment workflows** with audit-friendly HITL, but needs **security hardening, richer ML integration, and UI/observability polish** to hit Phase 3 goals.
