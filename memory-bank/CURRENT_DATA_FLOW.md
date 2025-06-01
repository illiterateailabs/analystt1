# CURRENT_DATA_FLOW.md
_Last updated: 01 Jun 2025_

## Current Implementation Data Flow

This document visualizes the **actual data flow** through the system as currently implemented, focusing on the Fraud Investigation workflow. The sequence diagram shows real API calls, agent interactions, and highlights where the flow breaks or has gaps.

sequenceDiagram
    participant FE as Frontend (Next.js)
    participant API as FastAPI Backend
    participant CF as CrewFactory
    participant NLQ as nlq_translator
    participant GA as graph_analyst
    participant FPH as fraud_pattern_hunter
    participant CGT as CodeGenTool (Sandbox)
    participant CC as compliance_checker
    participant HITL as Human Reviewer
    participant RW as report_writer
    participant Neo4j as Neo4j Database
    participant Gemini as Gemini API
    participant E2B as e2b Sandbox

    Note over FE,E2B: Fraud Investigation Workflow (Current Implementation)
    
    FE->>+API: POST /api/v1/crew/run<br>JWT + {crew_name: "fraud_investigation", input: "query"}
    Note right of API: ⚠️ Missing RBAC check
    API->>+CF: get_or_create_crew("fraud_investigation")
    CF-->>-API: crew instance
    
    API->>+CF: run_crew(input="query")
    Note right of CF: Sequential execution starts
    
    CF->>+NLQ: Task 1: Convert NL to Cypher (5-8s)
    NLQ->>+Gemini: LLM API call
    Gemini-->>-NLQ: Response
    NLQ->>+Neo4j: Check schema (Neo4jSchemaTool)
    Neo4j-->>-NLQ: Schema metadata
    NLQ-->>-CF: Cypher query
    Note right of NLQ: ⚠️ Limited to basic schema<br>No temporal understanding
    
    CF->>+GA: Task 2: Execute graph query (3-15s)
    GA->>+Neo4j: Run Cypher + PageRank
    Neo4j-->>-GA: Graph results
    GA-->>-CF: Node/edge data + metrics
    Note right of GA: ⚠️ Path length capped at 6<br>Heavy queries may timeout
    
    CF->>+FPH: Task 3: Match fraud patterns (2-5s)
    FPH->>+Neo4j: Pattern matching via PatternLibraryTool
    Neo4j-->>-FPH: Matching subgraphs
    FPH-->>-CF: Patterns + heuristic risk score
    Note right of FPH: ⚠️ ML integration partial<br>Uses pattern count × weight
    
    CF->>+CGT: Task 4: Generate & run Python (optional)
    CGT->>+Gemini: Generate Python code
    Gemini-->>-CGT: Python script
    CGT->>+E2B: Execute in sandbox
    E2B-->>-CGT: Execution result
    CGT-->>-CF: Code + result
    Note right of CGT: ❌ BROKEN: Results not fed<br>back to subsequent agents
    
    CF->>+CC: Task 5: Compliance check (1-3s)
    CC->>+Gemini: Check for sensitive content
    Gemini-->>-CC: Sensitivity assessment
    
    alt Sensitive content detected
        CC-->>CF: PAUSE request
        CF-->>API: {status: "PAUSED_FOR_HITL", task_id: "xyz"}
        API-->>FE: 200 OK + pause status + review link
        FE->>HITL: Display review UI
        Note over HITL: Human reviews findings
        HITL->>+FE: Approve/Reject decision
        FE->>+API: PATCH /api/v1/crew/resume<br>{task_id: "xyz", approved: true}
        API->>+CF: resume_crew(task_id)
        CF->>+CC: Resume from pause
        CC-->>-CF: Continue with approval
    else No sensitive content
        CC-->>-CF: Proceed (no pause)
    end
    
    Note right of CC: ⚠️ Uses hard-coded rules<br>No PolicyDocsTool retrieval
    
    CF->>+RW: Task 6: Generate report (2-4s)
    RW->>+Gemini: Draft narrative
    Gemini-->>-RW: Report text
    RW-->>-CF: Markdown + graph JSON
    Note right of RW: ⚠️ Missing ML charts<br>No PDF export
    
    CF-->>-API: Complete crew result
    API-->>-FE: 200 OK + full response
    
    Note over FE,E2B: Total time: 15-35s (+ HITL review time if paused)

## Key Observations

1. **Authentication & Security**
   - ❌ Missing RBAC check on `/crew/run` endpoint (security gap)
   - ✅ JWT validation works
   - 🟡 Token blacklist not persistent (Redis integration pending)

2. **Agent Execution**
   - ✅ Sequential execution ensures auditability
   - ✅ All 5 core agents operational
   - 🟡 Agent timeouts can occur on complex queries (30s limit)

3. **Critical Gaps**
   - ❌ **CodeGenTool** results not fed back to subsequent agents
   - 🟡 **PolicyDocsTool** uses hard-coded rules instead of vector retrieval
   - 🟡 ML integration partial (heuristic risk scoring)

4. **HITL Workflow**
   - ✅ Pause/resume mechanism works
   - ✅ Human review UI exists
   - 🟡 Approval decisions not persisted to database

5. **Performance**
   - 🟡 Total flow: 15-35s (excluding HITL time)
   - 🟡 Real-time alert enrichment: 6-7s (target: <5s)

## Implementation Notes

- The system successfully completes end-to-end flows but has gaps in ML integration, policy retrieval, and result persistence.
- The most critical fix needed is the CodeGenTool result integration, which breaks the flow of ML-generated insights to the final report.
- RBAC protection on `/crew/run` is a P0 security gap that should be addressed immediately.
- The current implementation meets basic requirements for supervised fraud investigations but needs enhancements for full automation.
