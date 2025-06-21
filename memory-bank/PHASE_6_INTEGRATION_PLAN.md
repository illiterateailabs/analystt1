# 📋 Phase 6: CrewAI Integration & Wiring Plan  
*Bridging CrewAI with our full-stack infrastructure*  

---

## 1. Executive Summary  
Phases 0-5 delivered the technical foundation: provider registry, Graph-Aware RAG, EvidenceBundle, HITL review, code-gen tooling, observability and hardening.  
**Phase 6** closes the final gap—wiring CrewAI agents and workflows to **use** these capabilities end-to-end, exposing a unified execution API for both UI and CI/CD consumers.

---

## 2. Current Integration Gaps  

| # | Gap | Impact |
|---|-----|--------|
| G-1 | **No Crew execution endpoint** (`/api/v1/crew/execute`) | Front-end & external callers cannot run crews |
| G-2 | **Tools not registered with CrewAI** | Agents cannot invoke SIM, Graph, RAG, Evidence tools |
| G-3 | **No RAG context provider in agents** | Responses lack graph-grounded context |
| G-4 | **EvidenceBundles not produced** | Findings lack structured, auditable output |
| G-5 | **HITL pause/resume not wired** | Human review flow unavailable during crew runs |

---

## 3. Phase Goal & Success Criteria  

**Goal:** Deliver a production-ready CrewAI layer fully integrated with tools, RAG, Evidence, and HITL, triggerable via REST/WebSocket.  

Success when:  
1. `POST /api/v1/crew/execute` returns structured results & EvidenceBundle.  
2. Any crew YAML can reference auto-discovered tools and they execute successfully.  
3. Agent answers cite RAG context; queries traceable in `explain_cypher`.  
4. HITL reviewers can pause, comment, resume from UI or API.  
5. All new paths covered by tests & traced by OTEL spans.

---

## 4. Work Breakdown & Task List  

### Task 6-1 Crew Execution API  
| Item | Details |
|------|---------|
| **Files** | `backend/api/v1/crew.py` (new), register in `main.py` |
| **Steps** | 1. Define request/response Pydantic models (crew_id, workflow, inputs).<br>2. Instantiate crew via `custom_crew.create_crew(crew_id)`. <br>3. Run workflow async; stream logs via WebSocket; return `{"result": ..., "evidence_bundle_id": ...}`. |
| **Tests** | `tests/test_api_crew.py` for 200 & error cases. |
| **Metrics** | Prometheus counter `crew_execution_total`, duration histogram. |

### Task 6-2 Tool ↔ Crew Wiring  
| Item | Details |
|------|---------|
| **Files** | `backend/agents/custom_crew.py`, `backend/api/v1/tools.py` |
| **Steps** | 1. On crew init, load `global_tool_registry` populated by tool auto-discovery.<br>2. Inject tool instances into each agent’s `.tools` list.<br>3. Ensure JSON schema for function-calling exposed to LLM provider. |
| **Tests** | `tests/test_crew_factory.py`, ensure tool count >0 for crew. |

### Task 6-3 RAG Context Provider  
| Item | Details |
|------|---------|
| **Files** | `backend/agents/tools/graph_rag_tool.py` (wrapper), update crew YAML (`graph_rag_tool`). |
| **Steps** | 1. Create `GraphRagTool` with methods `search_context`, `embed_and_store`.<br>2. Agents call tool for context retrieval prior to response.<br>3. Attach retrieved context as `agent.context`. |
| **Tests** | Integration test calls crew and asserts `graph_context` present in output. |

### Task 6-4 EvidenceBundle Output  
| Item | Details |
|------|---------|
| **Files** | `backend/agents/tools/evidence_tool.py`, modify `custom_crew.py` finish hook. |
| **Steps** | 1. Initialize `EvidenceBundle` at crew start; pass to `EvidenceTool`.<br>2. Agents add evidence during execution.<br>3. On crew completion, store bundle in DB / Redis; return ID in API. |
| **Tests** | `tests/test_evidence_bundle.py` assert bundle fields populated. |

### Task 6-5 HITL Pause/Resume  
| Item | Details |
|------|---------|
| **Files** | `backend/api/v1/hitl.py`, `backend/agents/custom_crew.py` |
| **Steps** | 1. Define `@hitl_point` decorator for tasks needing review.<br>2. When hit, create review row & WebSocket event; set crew state to `PAUSED`.<br>3. `/api/v1/hitl/approve` resumes task with feedback.<br>4. Add timeout auto-approve config. |
| **Tests** | `tests/test_hitl.py` covers pause & resume flow. |

---

## 5. Milestones & Timeline  

| Week | Deliverable |
|------|-------------|
| **W 0** | Task 6-1 API endpoint live |
| **W 1** | Task 6-2 tool wiring + unit tests |
| **W 2** | Task 6-3 RAG tool; agents show context |
| **W 3** | Task 6-4 EvidenceBundle persisted & returned |
| **W 3** | Task 6-5 HITL integration; end-to-end demo |
| **W 4** | Phase 6 regression tests, docs, Grafana panels update |

---

## 6. Ownership Matrix  

| Area | Owner | Reviewers |
|------|-------|-----------|
| API design & docs | Backend Lead | Dev Rel |
| Crew refactor | AI Platform Eng | QA |
| RAG tool | Data Science Eng | Backend Lead |
| Evidence integration | AI Platform Eng | Compliance SME |
| HITL flow | Frontend + Backend | Product |

---

## 7. Risks & Mitigations  

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM function-calling schema mismatch | Agents fail | Contract tests in CI |
| Evidence bundle bloat | Memory issues | Paginate + store in object store |
| HITL latency stalls crews | SLA breach | Auto-timeout fallback |
| Tool version drift | Errors | Pin versions in registry.yml |

---

## 8. Definition of Done Checklist  

- [ ] `/api/v1/crew/execute` documented in OpenAPI  
- [ ] Crews list >0 tools at runtime  
- [ ] RAG context key present in agent outputs  
- [ ] EvidenceBundle ID returned & retrievable via `/api/v1/evidence/{id}`  
- [ ] HITL pause/resume demoed in staging  
- [ ] Unit + integration tests ≥ 90 % pass  
- [ ] OTEL traces visible in Jaeger for crew run  
- [ ] Memory-bank docs & README updated  

---

*Drafted by Factory Droid • Last updated 2025-06-22*  
