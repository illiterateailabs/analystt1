# systemPatterns.md – System & Design Patterns

---

## 1. High-Level Architecture

```
┌────────────┐   REST   ┌──────────────┐   Crew orchestrates  ┌────────────┐
│  Frontend  │ ───────► │   FastAPI    │ ────────────────────►│  CrewAI    │
│  Next.js   │          │  Backend     │                     │  Engine    │
└────────────┘           │  (Python)   │◄────────────────────┘            │
                         │             │          Task ctx / results      │
                         └────┬────────┘                                   │
                              │ async/await                                │
                              ▼                                            │
                   ┌──────────────────────┐                                │
                   │  Tools  (Ports)      │                                │
                   │  • GraphQueryTool    │─ Bolt → Neo4j                  │
                   │  • PatternLibrary    │                                │
                   │  • SandboxExecTool   │─ gRPC → e2b                    │
                   │  • PolicyDocsTool    │─ RAG → Redis/Vector            │
                   │  • Etc.             …                                 │
                   └──────────────────────┘                                │
                              │                                            │
                              ▼                                            │
                        Google Gemini (HTTP)  /  External MCP servers
```

* **Presentation** – Next.js UI & external clients hit REST endpoints.  
* **Service Layer** – FastAPI routes validate, authenticate (JWT), launch crew tasks.  
* **Application Core** – CrewAI orchestrates agents in a **sequential** process for MVP; tasks pass explicit context.  
* **Infrastructure Adapters (Tools)** – Each external system wrapped as `BaseTool` (Ports-and-Adapters).  
* **Data Stores & Services** – Neo4j (graph), Redis (vector memory/events), e2b sandboxes, Gemini LLM API, Prometheus, S3 (artefacts).  

## 2. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Sequential crew for Phase-2** | Deterministic, auditable execution path required by AML regulators. |
| **Tool abstraction (Ports/Adapters)** | Swappable integrations, easier unit-testing (mock tools). |
| **GeminiLLMProvider subclass** | Unlock Gemini function-calling & cost metrics in CrewAI. |
| **Try/Except optional-tool imports** | Prevent missing crypto tools from breaking CI/runtime. |
| **e2b sandbox for code gen** | Safely run LLM-generated Python; isolates exploits. |
| **HITL pause/resume via webhook** | Compliance_checker must obtain human approval. |
| **Pydantic Settings + env** | 12-factor, Docker & k8s friendly configuration. |
| **Structured logging (structlog + JSON)** | Machine-queryable logs; link Task-ID to trace-ID. |
| **CI fast-fail order** | Lint ➜ mypy ➜ install ➜ pytest; short feedback loop. |

## 3. Design Patterns in Use

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Factory** | `backend/agents/factory.py` | Centralises tool init, agent cache, crew creation. |
| **Ports & Adapters** | Tools layer | Keeps core logic independent of external libs. |
| **Builder (YAML)** | `memory-bank/crews/*.yaml` | Declarative crew/agent/task config. |
| **Decorator – Retry/Rate-Limit** | Neo4jClient / GeminiClient | Auto-retry transient errors, respect rate caps. |
| **Strategy (PatternLibrary)** | Motif → Cypher conversion strategies (LLM vs Template). |
| **Observer / Metrics** | Prometheus counters per task & agent. |
| **Mediator** | CrewAI engine mediates agent interactions. |
| **HITL Circuit-Breaker** | compliance_checker pauses execution awaiting human input. |

## 4. Component Relationships

* **FastAPI → CrewFactory** – Each `/api/v1/crew/run` call fetches or builds a crew instance from Factory.
* **CrewFactory → Tools** – Injects tool instances into agents according to YAML config.
* **Agents**  
  * `nlq_translator` – Gemini + Neo4jSchemaTool  
  * `graph_analyst` – GraphQueryTool  
  * `fraud_pattern_hunter` – PatternLibraryTool (+ SandboxExecTool for ML)  
  * `sandbox_coder` – CodeGenTool + SandboxExecTool  
  * `compliance_checker` – PolicyDocsTool + HITL  
  * `report_writer` – TemplateEngineTool  
* **Tools → External Services** – Database queries, code exec, API calls.
* **Redis bus** – future event stream / SSE streaming to UI.

## 5. Critical Implementation Paths

### Fraud Investigation (Phase-2 MVP)

1. **POST /crew/run** – Analyst question arrives.  
2. **CrewFactory** loads `fraud_investigation.yaml`; sequential tasks:  
   1. `nlq_translator` → Cypher  
   2. `graph_analyst` → Neo4j result + GDS  
   3. `fraud_pattern_hunter` → motif matching + anomaly score  
   4. `sandbox_coder` (optional ML)  
   5. `compliance_checker` → HITL pause if sensitive  
   6. `report_writer` → markdown, risk score, graph JSON  
3. **FastAPI returns** 200 + JSON; UI renders graph visual.

### Real-Time Alert Enrichment (RT-AE)

Similar path but with light-weight GDS and no sandbox step; SLA < 5 s enforced via agent max_iter, prompt size limits, aggressive caching.

## 6. Communication Flows

| Direction | Protocol | Payload |
|-----------|----------|---------|
| **Frontend ↔ Backend** | HTTPS/JSON | Crew run request & SSE (future). |
| **Backend ↔ Neo4j** | Bolt+TLS | Cypher queries, GDS procedures. |
| **Backend ↔ Gemini** | HTTPS (REST) | JSON chat messages, tool calls. |
| **Backend ↔ e2b** | gRPC/HTTPS | Code bytes, exec params, stdout/stderr. |
| **Backend ↔ Redis** | RESP | Vector embeddings, publish/subscribe. |
| **Backend ↔ Prometheus** | HTTP scrape | Metrics exposition `/metrics`. |

## 7. Error Handling Patterns

| Layer | Pattern |
|-------|---------|
| **FastAPI** | Global `http_exception_handler`, `general_exception_handler`; returns JSON with `request_id`. |
| **CrewFactory** | Try/except around external service connect; returns `{success: False, error}` to caller. |
| **Tools** | Internal retry (exponential backoff) for transient network/Neo4j errors; raise custom `ToolError`. |
| **GeminiLLMProvider** | Categorise API errors (429, 5xx) -> Crew error; track token cost even on failure. |
| **SandboxExecTool** | Time-box exec, capture non-zero exit codes, surface `stderr` in agent output. |
| **HITL** | If human rejects, crew returns `success: False, status: "REJECTED_BY_REVIEWER"`. |
| **CI Pipeline** | Fail-fast: dependency install error aborts matrix; lint & mypy fail gate merges. |
| **Observability** | Prometheus counter `crew_errors_total{agent="*",type="*"}` with labels for alerting. |

---

*Maintaining these system patterns ensures onboarding — and my post-reset self — can grasp architecture, design trade-offs and critical paths at a glance.*  
