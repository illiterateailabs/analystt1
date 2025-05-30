# Multi-Agent Implementation Plan  

*Integration of CrewAI into analyst-augmentation-agent*  

**Filename:** `MULTI_AGENT_IMPLEMENTATION_PLAN.md`  

**Author:** Factory Assistant – 2025-05-29  


---


## 0. Why CrewAI?


| Benefit | Impact on Current Stack |

|---------|------------------------|

| Role-based agents & task routing | Mirrors human analyst org chart → clearer reasoning paths |

| Built-in memory + reflection | Long investigations retain context beyond single prompt |

| Tool abstraction | Wrap Neo4j, e2b sandboxes, external MCP tools as first-class “Tools” |

| Deterministic & autonomous workflows | Choose sequential (auditable) or autonomous (exploratory) modes per use case |


---


## 1. Target Use-cases for Phase 2


1. **Complex Fraud Case Investigation** – multi-hop graph queries, pattern detection, summarisation.  

2. **Real-time Alert Enrichment** – ingest an alert, gather supporting evidence, risk-score, recommend action.  

3. **Red-Team vs Blue-Team Simulation** (stretch) – adversarial agents generate synthetic fraud scenarios.


---


## 2. Crew Design


### 2.1 Agent Roles


| Agent ID | Primary Goal | Key Tools | Notes |

|----------|--------------|-----------|-------|

| `orchestrator_manager` | Break user request into sub-tasks, assign agents, ensure SLA | Internal memory | 1 per crew |

| `nlq_translator` | Convert NL → Cypher / SQL | GeminiClient, Neo4jSchemaTool | Already similar to existing GPT→Cypher |

| `graph_analyst` | Execute Cypher, run GDS algorithms, return structured result | Neo4jClient | Heavy graph workloads |

| `fraud_pattern_hunter` | Search for known/unknown patterns, anomaly scoring | Neo4jClient, PatternLibrary | Uses templates + unsupervised algorithms |

| `sandbox_coder` | Generate & run Python code in e2b VM for data munging/ML | GeminiClient, E2BClient | Installs libs on-the-fly |

| `compliance_checker` | Ensure outputs align with AML regulations, format SAR sections | PolicyDocs, Gemini | RBAC: must approve sensitive outputs |

| `report_writer` | Produce executive narrative, markdown, PPT slides | Gemini, TemplateEngine | Multimodal output |

| `red_team_adversary` (optional) | Simulate fraudster behaviour, probe defences | e2bClient, RandomTxGenerator | Runs against staging data |


### 2.2 Crew Composition Example (YAML)


```yaml

# crewai/crews/fraud_investigation.yaml

crew_name: fraud_investigation

manager: orchestrator_manager        # task decomposition

process_type: sequential             # deterministic & auditable

agents:

  - id: nlq_translator

  - id: graph_analyst

  - id: fraud_pattern_hunter

  - id: sandbox_coder

  - id: compliance_checker

  - id: report_writer

```


---


## 3. Integration Points


| Layer | Implementation |

|-------|----------------|

| **LLM Back-end** | Use existing `GeminiClient`; register with CrewAI via custom `GeminiLLMProvider` class |

| **Graph Access** | Wrap `Neo4jClient.run_query()` as a CrewAI `Tool` (`GraphQueryTool`) |

| **e2b Sandboxes** | Wrap `E2BClient.execute_code()` as `SandboxExecTool` (returns stdout / artefacts) |

| **MCP Tools** (Phase 3) | Use `crewai-mcp-toolbox` to auto-wrap external MCP servers |


---


## 4. Step-by-Step Implementation


### Milestone 1 — Skeleton Crew (1–2 days)


1. `pip install "crewai[tools]"` – add to requirements.  

2. Create `backend/agents/` package.  

3. Implement minimal agents (`orchestrator_manager`, `nlq_translator`, `graph_analyst`) in YAML.  

4. Register **GraphQueryTool** that calls Neo4j.


```python

# backend/agents/tools/graph_query_tool.py

from crewai_tools import BaseTool

from backend.integrations.neo4j_client import Neo4jClient

class GraphQueryTool(BaseTool):

    name = "graph_query_tool"

    description = "Run Cypher on Neo4j and return JSON."

    async def _run(self, query: str):

        return await Neo4jClient.get_global().run_query(query)

```


### Milestone 2 — Tool Wrappers & Memory (3–4 days)


- Wrap **GeminiLLMProvider** (`crewai.llm.BaseLLM`) using `GeminiClient.generate_text`.  

- Create **SandboxExecTool** & **CodeGenTool** (Gemini generates code, SandboxExec runs it).  

- Enable vector memory (Redis) for agents that need long context.


### Milestone 3 — Fraud Pattern Library (1 week)


- Build JSON / YAML definitions of fraud motifs (e.g., circular transfers).  

- `fraud_pattern_hunter` loads library, converts to Cypher subgraphs, runs via tool.  

- Score results → pass to `report_writer`.


### Milestone 4 — Reporting Workflow (2 days)


- `report_writer` uses Gemini to combine agent outputs.  

- Return Markdown + JSON summary.  

- Expose `/api/v1/crew/run` endpoint in FastAPI:


```python

@app.post("/api/v1/crew/run")

async def run_crew(request: CrewRequest):

    crew = await CrewFactory.load("fraud_investigation")

    result = await crew.kickoff(inputs=request.dict())

    return {"result": result}

```


### Milestone 5 — CI, Tests, Observability (1 week)


- Unit tests: mock Gemini / Neo4j calls.  

- Integrate **CrewAI logging** into existing structlog.  

- Prometheus counter: tasks executed per agent.  


### Milestone 6 — Red/Blue Team Extension (stretch)


- Implement `red_team_adversary` + `blue_team_monitor`.  

- Use **alternating crews** inside simulation harness.


---


## 5. Code Snippet: Agent Declaration (Python alternative)


```python

from crewai import Agent, Task, Crew

from backend.agents.tools import GraphQueryTool, SandboxExecTool


nlq = Agent(

    id="nlq_translator",

    role="NLQ-to-Cypher Specialist",

    goal="Translate analyst questions into optimized Cypher",

    model="gemini-1.5-pro",

    tools=[GraphQueryTool()],

    max_iter=3,

)


graph_analyst = Agent(

    id="graph_analyst",

    role="Graph Data Scientist",

    goal="Execute queries and run community detection",

    model="gemini-1.5-pro",

    tools=[GraphQueryTool(), SandboxExecTool()],

    memory=True,

)


crew = Crew(

    manager=Agent(id="mgr", role="Investigation Manager", model="gemini-1.5-pro"),

    agents=[nlq, graph_analyst],

    process="sequential",

)

```


---


## 6. Roadmap & Timeline


| Week | Deliverable | Success Criteria |

|------|-------------|------------------|

| 1 | Milestone 1 | Demo: NLQ → graph results via crew endpoint |

| 2 | Milestone 2 | Sandbox code generation runs Python in VM |

| 3–4 | Milestone 3 | Pattern hunter finds sample money-laundering ring |

| 5 | Milestone 4 | Markdown + PPT fraud report auto-generated |

| 6 | Milestone 5 | CI tests >70 % pass, Prometheus metrics |

| 7+ | Milestone 6 | Red/Blue simulation, MCP tool intake |


---


## 7. Definition of Done (Phase 2 MVP)


1. CrewAI endpoints live under `/api/v1/crew/*`.  

2. Analysts can ask: **“Trace funds from Wallet 0xABC and summarise risk.”**  

3. System returns: graph visual, risk score, narrative report.  

4. All tasks logged, reproducible via task ID.  

5. Test coverage ≥ 70 %, CI green.  


---


## 8. Future Enhancements


- Hierarchical crews (manager → sub-crew per entity).  

- Streaming mode (SSE) for incremental responses.  

- AgentOps observability (CrewAI + LangTrace).  

- Federated crews across multiple companies for collaborative AML.  


---


> **Ready to build!** Start with Milestone 1 skeleton agents; scaffold in `backend/agents/` and plug into existing FastAPI. This will immediately unlock visible multi-agent power and set the stage for advanced financial crime intelligence.   
