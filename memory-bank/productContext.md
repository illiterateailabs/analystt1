# Product Context – Analyst Augmentation Agent

## 1. Why This Project Exists  
Financial-crime teams face an explosion of data (on-chain & traditional finance), stricter AML regulations, and soaring alert volumes. Human analysts struggle to **trace complex fund flows, detect sophisticated fraud patterns, and meet tight Service-Level Agreements**. Existing tools are siloed, rule-based, and require labor-intensive graph queries. The Analyst Augmentation Agent is built to **amplify analysts with AI-driven graph reasoning, machine learning and automated reporting** while preserving auditability and human oversight.

## 2. Problems It Solves  
| Pain Point | Impact | Solution Provided |
|------------|--------|-------------------|
| Manual multi-hop graph queries are slow & error-prone | Hours-days per case | Natural-language→Cypher translation & automated execution |
| Alert fatigue & high false-positive rate | Wasted analyst time | Real-time enrichment with contextual evidence & risk scoring |
| Disconnected tools (databases, ML notebooks) | Context switching, data leakage | Unified multi-agent workflow with first-class tools (Neo4j, e2b sandboxes, RAG policy docs) |
| Regulatory need for audit trails & SAR accuracy | Fines, reputational risk | Deterministic sequential crews, HITL compliance checker, reproducible task IDs |
| Shortage of skilled graph/ML engineers | Bottlenecks | AI agents generate & run Python analytics in secure sandboxes |

## 3. How It Should Work (User Perspective)  
1. **Ask** – Analyst types a question or drops an alert ID:  
   _“Trace funds from wallet 0xABC over the last 30 days and summarise risk.”_  
2. **Orchestrate** – The system spins up a _fraud_investigation_ crew:  
   - `nlq_translator` converts the request to Cypher.  
   - `graph_analyst` executes multi-hop queries & GDS algorithms.  
   - `fraud_pattern_hunter` matches known motifs & runs anomaly scoring.  
   - `compliance_checker` validates findings against AML policy (HITL pause if needed).  
3. **Deliver** – Within seconds the UI displays:  
   - Interactive graph visual highlighting risky paths.  
   - Numeric risk score & key drivers.  
   - Markdown narrative + downloadable JSON summary.  
4. **Audit** – Clicking “Details” reveals every agent task, timestamps, token usage and links to replay via Task-ID.

## 4. User Experience Goals  
* **Speed** – < 5 s median response for alert enrichment; < 30 s for deep investigations.  
* **Transparency** – Clear reasoning path, accessible logs, explain-able risk scoring.  
* **Interactivity** – Zoomable graph, drill-down on entities, what-if re-queries.  
* **Trust & Control** – Human approval before sensitive SAR sections leave sandbox.  
* **Low Cognitive Load** – Natural-language input/output, no Cypher or Python required.  

## 5. Key Workflows & Use Cases  
| Workflow | Primary Actors | Outcome | Frequency |
|----------|----------------|---------|-----------|
| **Complex Fraud Case Investigation** | Human analyst + fraud_investigation crew | Full funds-flow trace, pattern analysis, SAR draft | Ad-hoc, high value cases |
| **Real-Time Alert Enrichment** | Monitoring system → alert_enrichment crew | Enriched alert with evidence & risk score pushed back to case-manager | Thousands/day |
| **Red/Blue Team Simulation** (stretch) | red_team_adversary vs blue_team_monitor | Synthetic fraud scenarios & defense gap analysis | Quarterly exercises |
| **Compliance Review** | compliance_checker + HITL human | Approved narrative & SAR section export | Every investigation |

## 6. Business Value Proposition  
* **70 %+ Reduction in investigation time** via automated graph & ML tasks.  
* **False-positive rate down by 30 %** from richer, immediate context.  
* **Regulatory resilience** – auditable AI workflow, HITL checkpoints, policy-aligned outputs.  
* **Cost savings** – lower analyst hours, no need for separate ML notebooks or DIY scripts.  
* **Competitive edge** – faster onboarding of new assets/jurisdictions; adaptable agent/tool framework for emerging threats.  

## 7. Success Metrics  
- Mean time-to-evidence for high-risk alerts ‹ 5 s  
- Analyst satisfaction (CSAT) ≥ 4.5/5  
- Task reproducibility rate 100 % (given Task-ID)  
- Regulatory findings with zero critical observations in audit  
- Monthly LLM cost per enriched alert ‹ $0.01  

---

*This Product Context complements **projectbrief.md** and drives user-centric design, feature prioritisation, and value measurement.*  
