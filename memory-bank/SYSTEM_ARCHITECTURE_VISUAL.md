# SYSTEM_ARCHITECTURE_VISUAL.md
_Last updated: 01 Jun 2025_

## System Architecture with Implementation Status

This diagram provides a visual representation of the Analyst Agent system architecture, with clear indicators showing implementation status:

- **Solid lines & Green** (✅): Fully implemented and working
- **Dashed lines & Yellow** (🟡): Partially implemented or MVP exists
- **Dotted lines & Red** (❌): Not implemented yet (documentation only)

flowchart TD
    %% Main Components
    UI[Frontend Next.js] --> |REST API| API[Backend FastAPI]
    API --> CrewEngine[CrewAI Engine]
    CrewEngine --> Agents[Agent Layer]
    Agents --> Tools[Tool Layer]
    
    %% Agents
    Agents --> NLQ[nlq_translator]
    Agents --> GA[graph_analyst]
    Agents --> FPH[fraud_pattern_hunter]
    Agents --> CC[compliance_checker]
    Agents --> RW[report_writer]
    
    %% Tools
    Tools --> GQT[GraphQueryTool]
    Tools --> PLT[PatternLibraryTool]
    Tools --> SET[SandboxExecTool]
    Tools --> PDT[PolicyDocsTool]
    Tools --> TET[TemplateEngineTool]
    Tools --> CAT[CryptoAnomalyTool]
    Tools --> CCLT[CryptoCSVLoaderTool]
    Tools --> FMT[FraudMLTool]
    
    %% External Services
    GQT --> Neo4j[Neo4j Graph DB]
    PLT --> Neo4j
    CCLT --> Neo4j
    SET --> E2B[e2b Sandbox]
    API --> Gemini[Google Gemini API]
    CrewEngine --> Gemini
    PDT -.-> Redis[Redis Vector Store]
    API --> Postgres[PostgreSQL]
    CC --> HITL[Human-in-the-Loop]
    
    %% Frontend Components
    UI --> AuthUI[Auth UI]
    UI --> DashUI[Dashboard UI]
    UI --> GraphUI[Graph Visualization]
    UI --> HITLUI[HITL Review UI]
    UI -.-> AnalysisUI[Analysis Results UI]
    
    %% DevOps & Monitoring
    API --> Prom[Prometheus Metrics]
    API -.-> Loki[Loki Logs]
    API -.-> OTel[OpenTelemetry Traces]
    
    %% Data Flows
    subgraph Workflow
        direction LR
        NLQuery[Natural Language Query] --> CypherQuery[Cypher Query]
        CypherQuery --> GraphResults[Graph Results]
        GraphResults --> PatternMatching[Pattern Matching]
        PatternMatching --> ComplianceCheck[Compliance Check]
        ComplianceCheck --> Report[Final Report]
    end
    
    %% Styling
    classDef implemented fill:#47956f,color:white,stroke:#333,stroke-width:2px
    classDef partial fill:#de953e,color:white,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    classDef missing fill:#8b251e,color:white,stroke:#333,stroke-width:2px,stroke-dasharray: 2 2
    
    %% Apply Classes - Implemented (Green)
    class UI,API,CrewEngine,Agents,NLQ,GA,FPH,CC,RW,GQT,PLT,SET,CCLT,Neo4j,E2B,Gemini,HITL,AuthUI,GraphUI,Prom implemented
    
    %% Apply Classes - Partial (Yellow)
    class Tools,CAT,TET,PDT,FMT,DashUI,HITLUI,Postgres,Redis partial
    
    %% Apply Classes - Missing (Red)
    class AnalysisUI,Loki,OTel missing
    
    %% Workflow styling
    class NLQuery,CypherQuery,GraphResults,PatternMatching,ComplianceCheck,Report implemented

## Architecture Notes

1. **Frontend Layer**:
   - ✅ Authentication UI (login/register) complete
   - ✅ Graph Visualization component with interactive features
   - 🟡 Dashboard UI partially implemented (prompt management only)
   - 🟡 HITL Review UI component exists but lacks full integration
   - ❌ Analysis Results View not yet built

2. **Backend Layer**:
   - ✅ FastAPI with health endpoints, CORS, error handling
   - ✅ JWT Auth with RBAC (role-based access control)
   - ✅ CrewAI engine with sequential agent execution
   - 🟡 Database migrations pending for users table

3. **Agent Layer**:
   - ✅ All 5 core agents implemented with YAML configs
   - ✅ Agent Prompt Management System for runtime editing

4. **Tool Layer**:
   - ✅ GraphQueryTool for Neo4j integration
   - ✅ PatternLibraryTool with 30+ fraud patterns
   - ✅ SandboxExecTool for secure code execution
   - 🟡 PolicyDocsTool exists but lacks vector retrieval
   - 🟡 TemplateEngineTool partially implemented
   - 🟡 CryptoAnomalyTool implemented but needs tuning
   - 🟡 FraudMLTool implemented but not fully integrated

5. **External Services**:
   - ✅ Neo4j Graph Database with schema initialization
   - ✅ e2b Sandbox for secure code execution
   - ✅ Google Gemini API integration (Flash & Pro)
   - 🟡 PostgreSQL for user data (missing migrations)
   - 🟡 Redis for token blacklist (partially implemented)

6. **Observability**:
   - ✅ Prometheus metrics for LLM tokens, costs, durations
   - ❌ Loki logs integration planned
   - ❌ OpenTelemetry traces planned

## Implementation Gaps & Priorities

1. **P0 (Critical):**
   - Complete `CodeGenTool` execution result integration
   - Apply RBAC protection to `/crew/run` endpoint
   - Generate Alembic migration for users table

2. **P1 (High):**
   - Implement Redis token blacklist
   - Complete PolicyDocsTool retrieval functionality
   - Build frontend Analysis Results View
   - Increase test coverage to ≥55%

3. **P2 (Medium):**
   - Add observability integrations (OTel, Loki)
   - Implement risk propagation model
   - Create production Kubernetes/Helm charts
