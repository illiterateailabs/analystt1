# WORKFLOW_CAPABILITIES_ANALYSIS.md  
_Last updated: 01 Jun 2025_

This document contrasts **documented intent** with **current implementation** for the four critical workflows:

* Fraud Investigation  
* Real-Time Alert Enrichment  
* Compliance Review (HITL)  
* Report Generation  

Legend ✅ implemented 🟡 partial ❌ missing  

---

## 1 · Fraud Investigation Workflow

| Stage | Expected Behaviour (Docs) | Current Behaviour | Status | Notes / Limitations |
|-------|---------------------------|-------------------|--------|---------------------|
| 1. NL → Cypher (`nlq_translator`) | Accept arbitrary natural-language question; generate schema-aware Cypher incl. temporal filters | Generates valid Cypher for basic node/rel labels; ignores temporal/risk schema | 🟡 | Works for 1-3 hop patterns, fails on `:TRANSFERS*\*` star-expansion > 4 |
| 2. Graph Query (`graph_analyst`) | Execute Cypher, run GDS algorithms (PageRank, betweenness) | Executes query, optional PageRank; betweenness/stats reserved | 🟡 | Path length capped at 6; heavy queries may time out → 30 s agent limit |
| 3. Pattern Matching (`fraud_pattern_hunter`) | Match 30+ motifs & run anomaly scoring | Matches motifs; anomaly score stub returns `0-100` random seed | 🟡 | ML score integration pending (`FraudMLTool`) |
| 4. Sandbox ML (`CodeGenTool` + `SandboxExecTool`) | Auto-generate Python for bespoke analytics & execute securely | Python executes; result JSON not routed back to crew | 🟡 | Charts / dataframes unavailable to subsequent agents |
| 5. Compliance Check (`compliance_checker`) | Pause if sensitive findings, auto-cross-reference AML policy | Pause works; policy doc retrieval not wired | 🟡 | Hard-coded rule list only |
| 6. Report (`report_writer`) | Produce markdown narrative + graph JSON + risk score | Generates markdown & graph JSON; missing ML charts, attachment export | 🟡 | Uses Jinja template; placeholders remain for missing fields |

### Example Run

```
POST /api/v1/crew/run
Authorization: Bearer <JWT>
{
  "crew_name": "fraud_investigation",
  "input": "Trace funds from 0xABC over the last 30 days and summarise risk"
}
```

**Current 200 Response (trimmed):**

```json
{
  "success": true,
  "task_id": "task_12345",
  "graph_data": { "nodes": 234, "edges": 456 },
  "fraud_patterns": ["WASH_TRADING", "FLASH_LOAN"],
  "risk_score": 74,
  "status": "PAUSED_FOR_HITL",
  "compliance_link": "/hitl/review/task_12345"
}
```

Limitations:  
* Risk score is heuristic (motif count × weight) – no ML.  
* If query > 30 s the agent aborts with `error: TIMEOUT exceeded`.

---

## 2 · Real-Time Alert Enrichment Workflow

| Stage | Expected | Current | Status | Limitations |
|-------|----------|---------|--------|-------------|
| Alert Intake | Receive alert JSON from monitoring system | Works via `/api/v1/analysis/enrich` | ✅ | JWT service account required |
| NL → Cypher | Fixed prompt template focusing on entity & timeframe | Same as Fraud workflow; run with shorter context | 🟡 | Strips description text > 500 chars |
| Graph Query | Single pass, lightweight query; SLA < 5 s | Typical runtime 6-7 s avg | 🟡 | Occasional 429 Gemini hits add latency |
| Pattern Fast-Match | Top-10 motifs only, no ML | Implemented | ✅ | —
| Response Push | Return enrichment JSON | Implemented | ✅ | —

### Example API Call

```
POST /api/v1/analysis/enrich
Content-Type: application/json
Authorization: Bearer <SERVICE_JWT>
{
  "alert_id": "alert-987",
  "entity": "0xDEADBEEF",
  "time_window_hours": 6
}
```

Sample Response:

```json
{
  "alert_id": "alert-987",
  "enriched": true,
  "risk_score": 58,
  "evidence_summary": "Detected possible wash trading (3 hops)",
  "graph_excerpt": { "...": "..." },
  "latency_ms": 6543
}
```

Limitations:  
* SLA _not_ guaranteed; needs caching & lighter prompts.  
* No streaming/SSE yet – caller must poll.

---

## 3 · Compliance Review (HITL) Workflow

| Stage | Expected | Current | Status | Limitations |
|-------|----------|---------|--------|-------------|
| Pause Trigger | If findings classified as **sensitive** per policy | Trigger on hard-coded keywords list | 🟡 | False positives/negatives possible |
| Webhook Notify | Send Slack/Email webhook to compliance queue | Works with generic JSON webhook | ✅ | Only tested locally |
| Review UI | Analyst sees findings, can **Approve / Reject / Comment** | UI loads, actions PATCH `/crew/resume` | 🟡 | Comments not persisted to DB |
| Auto-Resume | POST `/crew/resume` continues tasks | Implemented | ✅ | —
| Audit Log | Store decision + user + timestamp | Not persisted (in-memory) | ❌ | Required for audit trail |

### Example Resume Call

```
PATCH /api/v1/crew/resume
Authorization: Bearer <JWT_compliance>
{
  "task_id": "task_12345",
  "approved": true,
  "comment": "Looks good, release report"
}
```

Returns:

```json
{ "success": true, "status": "RESUMED" }
```

---

## 4 · Report Generation Workflow

| Stage | Expected | Current | Status | Limitations |
|-------|----------|---------|--------|-------------|
| Data Aggregation | Collect graph, ML, pattern & compliance notes | Graph + pattern yes; ML & compliance notes partial | 🟡 | Missing charts, sanction-list cross-refs |
| Template Rendering (`TemplateEngineTool`) | Jinja2 template with dynamic sections | Renders markdown; placeholder sections left blank | 🟡 | Needs conditional blocks |
| Export Formats | Markdown, PDF, CSV attachments | Markdown only | ❌ | PDF export planned Phase 3 |
| Delivery | Return via API & optional email | API returns markdown & JSON | ✅ | No email yet |

### Example Output Snippet

```
### Executive Summary
Risk Score: **74 / 100 (High)**  
Identified patterns: Wash Trading, Flash Loan

### Graph Snapshot
![graph](data:image/png;base64,...)

### Key Findings
1. Address 0xABC received 4 clustered deposits...
```

Limitations:  
* Image is referenced but base64 not included until code-gen integration is finished.  
* Tables auto-wrap poorly in PDF conversion (missing).

---

## 5 · Cross-Workflow Dependency Map

```
Fraud Investigation ─┬─▶ Compliance Review ──┬─▶ Report Generation
                     └───────────────────────┘
Real-Time Alert Enrichment ─┬─▶ Report Generation (lite)
```

Current blockers propagate: e.g., incomplete PolicyDocsTool affects both **Fraud Investigation** and **Compliance Review**.

---

## 6 · Summary of Major Gaps

| Gap | Affects Workflows | Priority |
|-----|-------------------|----------|
| PolicyDocsTool retrieval logic | Fraud Investigation, Compliance Review | P1 |
| CodeGenTool result ingestion | Fraud Investigation, Report Generation | P0 |
| Audit log persistence | Compliance Review | P1 |
| SLA optimisation & caching | Real-Time Alert Enrichment | P1 |
| Export to PDF & email | Report Generation | P2 |

_Closing the P0-P1 gaps will unlock fully automated investigations with auditable compliance trails and analyst-ready reports._  
