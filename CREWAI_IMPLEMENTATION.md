# CrewAI Multi-Agent System – Implementation Guide

Repository: `analyst-agent-illiterateai`  
File: **CREWAI_IMPLEMENTATION.md**  
Last updated: 2025-05-30  

---

## 1. Purpose

This document is the single source of truth for the **CrewAI-powered multi-agent extension** that augments the Analyst Agent.  
It explains:

* High-level architecture and data-flow  
* All agents, their goals, tools and configuration knobs  
* Custom tools and how they wrap external services (Neo4j, e2b, Google GenAI)  
* API surface (`/api/v1/crew/*`) with request / response payloads  
* End-to-end examples to help you run, debug and extend the system  

If you want to **add a new agent**, **write a new fraud pattern**, or simply **call the crew from your frontend** – start here.

---

## 2. Architectural Overview

```
+──────────+    NL Query   +──────────+         +──────────+
|  Frontend| ───────────▶ | /crew/run | ──────▶ | CrewFactory|
+──────────+              +──────────+         +────┬──────+
                                                     │creates
                                                     ▼
                                          +──────────────────────+
                                          |   Crew (Process)     |
                                          |  orchestrator_manager|
                                          +─────────┬────────────+
                                                    │tasks
            +--------------------+------+------+----+------+------+--------------------+
            |                    |             |           |             |              |
            ▼                    ▼             ▼           ▼             ▼              ▼
      nlq_translator      graph_analyst  fraud_pattern  sandbox_   compliance_   report_
                                            _hunter       coder      checker      writer
            │                    │             │           │             │              │
            ▼                    ▼             ▼           ▼             ▼              ▼
  Neo4jSchemaTool        GraphQueryTool  PatternLibrary  CodeGen    PolicyDocs    TemplateEngine
                                          + Gemini LLM     +e2b
```

1. **CrewFactory** instantiates agents, tools and tasks based on YAML / default configs.  
2. A **Crew** runs in **sequential** or **hierarchical** mode, coordinated by `orchestrator_manager`.  
3. Each agent owns an **LLM** (Google GenAI via `GeminiLLMProvider`) and a **tool belt**.  
4. Tools abstract external services; they are callable from LLM function-calls or imperative code.  
5. Final artefacts (reports, Python notebooks, graphs) return to the REST caller.

---

## 3. Components in Detail

### 3.1 Agents

| ID | Role | Goal | Key Tools | Notes |
|----|------|------|-----------|-------|
| **orchestrator_manager** | Workflow Coordinator | Kick-off, SLA, quality gate | – | Not a dynamic planner in sequential crews; can become manager_agent in hierarchical crews |
| **nlq_translator** | NL → Cypher Specialist | Convert analyst questions to Cypher | `neo4j_schema_tool`, `graph_query_tool` | Uses Gemini function-calling to produce safe queries |
| **graph_analyst** | Graph Data Scientist | Execute queries, run GDS | `graph_query_tool`, `sandbox_exec_tool` | Streams results + analytics |
| **fraud_pattern_hunter** | Pattern Detection | Detect known / unknown patterns | `pattern_library_tool`, `graph_query_tool` | Hybrid rule + LLM conversion |
| **sandbox_coder** | Code Generator | Produce & run Python in e2b | `code_gen_tool`, `sandbox_exec_tool` | Installs libs on the fly |
| **compliance_checker** | AML Officer | Validate output vs policy | `policy_docs_tool` | HITL capable (pauses crew) |
| **report_writer** | Intelligence Author | Create executive report | `template_engine_tool`, `graph_query_tool` | Markdown / HTML / JSON |
| **red_team_adversary** | Fraud Simulator (stretch) | Generate synthetic attacks | `random_tx_generator_tool`, `sandbox_exec_tool` | Used in red/blue crews |

Agent YAMLs live under `backend/agents/configs/` (auto-generated defaults shipped).

### 3.2 Tools (Python path `backend/agents/tools/`)

| Tool | Wraps | Purpose |
|------|-------|---------|
| **GraphQueryTool** | Neo4j driver | Async Cypher with safe JSON output |
| **Neo4jSchemaTool** | Neo4j driver | Introspect labels, rels, constraints |
| **PatternLibraryTool** | File system + Gemini | Convert YAML/JSON fraud motifs to Cypher |
| **SandboxExecTool** | e2b SDK | Secure code execution, file IO |
| **CodeGenTool** | Gemini | Produce secure, linted Python |
| **PolicyDocsTool** | In-mem / future RAG | Search AML/KYC text or generate answer via LLM |
| **TemplateEngineTool** | Jinja2 | Render Markdown/HTML/JSON reports |
| **RandomTxGeneratorTool** | N/A | Produce synthetic transactions for tests |

### 3.3 LLM Provider

`GeminiLLMProvider` subclasses `crewai.llm.BaseLLM` and delegates to **google-genai** (`from google import genai`).  
It supports:

* Async generation  
* Function calling (`tools=` arg)  
* Custom temperature / max tokens per call  
* Automatic retries with exponential back-off  

---

## 4. REST API

All endpoints are versioned under `/api/v1/crew`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run` | Run a crew (sync or async) |
| `GET`  | `/status/{task_id}` | Check async execution status |
| `GET`  | `/crews` | List available crews |
| `GET`  | `/crews/{crew_name}` | Detailed crew metadata |
| `GET`  | `/agents` | List agents (optionally by crew) |
| `GET`  | `/agents/{agent_id}` | Detailed agent metadata |

### 4.1 Run Crew (sync)

```bash
curl -X POST http://localhost:8000/api/v1/crew/run \
     -H "Authorization: Bearer <JWT>" \
     -H "Content-Type: application/json" \
     -d '{
           "crew_name": "fraud_investigation",
           "inputs": {
             "question": "Trace funds from Wallet 0xABC and summarise risk."
           },
           "async_execution": false
         }'
```

**Response**

```json
{
  "success": true,
  "crew_name": "fraud_investigation",
  "result": {
    "report_markdown": "...",
    "graph_visual": "base64-png",
    "risk_score": 0.87
  }
}
```

### 4.2 Async Execution

Pass `"async_execution": true` – you receive a `task_id`, then poll `/status/{task_id}`.

---

## 5. Setup & Deployment

### 5.1 Prerequisites

* Python 3.10+  
* Docker (if using `docker-compose.yml`)  
* API keys in `.env`  
  * `GOOGLE_API_KEY` → Gemini (google-genai)  
  * `E2B_API_KEY` → e2b sandboxes  

### 5.2 Install

```bash
pip install -U -q "google-genai"      # latest GenAI SDK
pip install 'crewai[tools]'           # multi-agent framework
pip install -r requirements.txt       # rest of backend deps
```

### 5.3 Run services

```bash
./scripts/setup.sh   # optional helper – builds images
./scripts/start.sh   # starts backend, frontend, Neo4j
open http://localhost:3000
```

Health check: `http://localhost:8000/health`

---

## 6. Extending the System

### 6.1 Add a New Pattern

1. Create `backend/agents/patterns/<category>/<id>.yaml` (see pattern README).  
2. Restart backend (hot-reload picks it up).  
3. Convert via PatternLibraryTool or run crew – pattern becomes available.

### 6.2 Add a New Agent

1. Add config in `backend/agents/configs/<agent_id>.yaml`.  
2. Implement custom tool(s) if needed.  
3. Reference agent in a crew YAML or modify `DEFAULT_CREW_CONFIGS` in `agents/config.py`.

### 6.3 Switch to Hierarchical Crew

1. Set `process_type: hierarchical` in the crew config.  
2. Optionally define `manager_llm` or keep `orchestrator_manager` as `manager`.  
3. Add delegation prompts to enable dynamic tasking.

---

## 7. Testing

* **Pytest** suite lives under `tests/` – includes mocks for GenAI, Neo4j, e2b.  
* Run `pytest -q --cov=backend` – target ≥ 70 % coverage.  
* End-to-end Playwright tests (TODO for Phase 3).

---

## 8. Observability & Ops

| Metric | Source | Where |
|--------|--------|-------|
| Task executions / agent | Prometheus Counter | `/metrics` (enabled in Milestone 5) |
| LLM token & cost | AgentOps / Langtrace (future) | Agent traces |
| Logs | structlog JSON | stdout or Loki |
| Errors | Sentry | DSN in `.env` |

---

## 9. Roadmap Snapshot

* **Phase 2 MVP** (current) – end-to-end fraud investigation + alert enrichment crews.  
* **Phase 3** – Hierarchical sub-crews, SSE streaming, automated CI/CD.  
* **Phase 4** – AgentOps observability, federated cross-company collaboration.  

See `ROADMAP.md` for full plan.

---

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: google.genai` | old SDK installed | `pip install -U google-genai` |
| 401 on `/crew/run` | Missing or invalid JWT | Hit `/api/v1/auth/login` first |
| Neo4j `ServiceUnavailable` | DB not up / wrong creds | Check `docker-compose`, `.env` |
| LLM calls fail with 429 | Rate limit | Increase `max_rpm`, add retries |

---

## 11. Contributing

1. Fork ➜ feature branch ➜ PR.  
2. Follow **Black** & **isort** (`./scripts/format.sh`).  
3. Add / update tests (`pytest`).  
4. Describe new agents or tools in this document.

---

**Enjoy building powerful multi-agent financial crime solutions with CrewAI!**  
Questions → discussions / issues on GitHub.
