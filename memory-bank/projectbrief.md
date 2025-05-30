# Project Brief – Analyst Augmentation Agent

## 1. Project Name & Purpose
**Analyst Augmentation Agent**  
An AI-powered, multi-agent platform that amplifies the speed, depth and accuracy of financial-crime analysts in both crypto and traditional finance.  
It combines Large-Language-Model reasoning (Google Gemini), graph analytics (Neo4j + GDS), secure code execution (e2b sand-box) and CrewAI orchestration to:

* trace funds & entities across complex transaction graphs  
* detect known and novel fraud / AML patterns  
* enrich real-time alerts with contextual evidence & risk scores  
* generate executive-ready narratives and artefacts

## 2. Core Requirements & Goals
| Area | Requirement / Goal |
|------|--------------------|
| Architecture | CrewAI-based multi-agent crews with role-specific tools |
| LLM | Custom **GeminiLLMProvider** supporting function calls & cost tracking |
| Data | Neo4j 5.x graph database, GDS algorithms, Chroma/Redis vector memory |
| Tooling | GraphQueryTool, PatternLibraryTool, SandboxExecTool, CodeGenTool, PolicyDocsTool |
| Workflows | • Complex Fraud Case Investigation<br>• Real-time Alert Enrichment |
| Security | e2b isolated code execution, HITL for compliance outputs, secrets management |
| Quality | CI (lint, type-check, pytest), ≥ 30 % coverage for MVP, structured logging, Prometheus metrics |
| Outputs | API `/api/v1/crew/*`; returns graph visual (JSON for front-end), risk score, markdown report |
| Performance | < 5 s average latency for alert enrichment (Phase 2 target) |
| Compliance | All sensitive decisions reviewed via HITL workflow (pause / approve / resume) |

## 3. Key Stakeholders
* **illiterate ai** – Product owner & lead developer  
* **Financial-Crime Analysts** – Primary users, drive requirements  
* **Compliance & Legal Teams** – Ensure regulatory alignment (AML, KYC, SAR)  
* **DevOps / Infrastructure** – Maintain deployments, observability, secrets  
* **Executive Sponsors** – Measure ROI, approve budgets  
* **Regulators / Auditors** – External reviewers of audit trails & model governance

## 4. Success Criteria
1. **Phase 2 MVP** delivers end-to-end flow:  
   * Input: “Trace funds from wallet 0xABC and summarise risk.”  
   * Output: Graph visual, risk score, narrative markdown; all tasks logged & reproducible via Task-ID.  
2. CI pipeline green on main with lint, mypy, pytest (≥ 30 % coverage).  
3. < 5 s P95 latency for alert enrichment (test dataset).  
4. HITL approvals captured for every compliance_checker decision.  
5. Positive analyst feedback (> 80 % tasks completed faster vs baseline).  
6. No critical security incidents in sandbox or tool layer.

## 5. Scope Boundaries
### In Scope
* Multi-agent orchestration (CrewAI) for the two primary workflows  
* Custom tools & integrations listed under Core Requirements  
* FastAPI backend, Next.js front-end skeleton, Dockerised deployment  
* Unit & integration testing, basic observability (structlog, Prometheus)  
* Documentation & Memory Bank maintenance

### Out of Scope (Phases ≥ 3)
* Federated cross-company AML collaboration  
* Advanced red/blue simulations with high-fidelity synthetic data  
* Full PPT/slide generation, external BI integrations  
* Production-grade role-based access control & multitenancy

## 6. Phase Definitions
| Phase | Description | Target Duration | Key Deliverables |
|-------|-------------|-----------------|------------------|
| **Phase 1 – Baseline Infrastructure** | FastAPI skeleton, health probes, Neo4j client, JWT auth, Docker, Makefile | ✅ *Completed* | Running backend & basic endpoints |
| **Phase 2 – MVP Multi-Agent** | Implement core crew (`fraud_investigation`, `alert_enrichment`), tools, GeminiLLMProvider, PatternLibrary, HITL, CI + tests | ~6-8 weeks | Definition of Done criteria (see §4) |
| **Phase 3 – Tool Expansion & Observability** | MCP tool adapters, advanced monitoring (AgentOps / Langtrace), streaming SSE responses | TBD | Additional crews & live dashboards |
| **Phase 4 – Advanced Simulation & Federation** | Red/Blue team, sophisticated scenario generator, cross-org federated crews | TBD | Prototype collaborative AML network |

---

*This **projectbrief.md** is the single source of truth for project scope. Any scope change must be reflected here before implementation proceeds.*  
