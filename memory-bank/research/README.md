# 📚 Research Index  
_This file indexes all documents in `memory-bank/research/` and sets the standard for future research notes._

---

## 1. Purpose  

The **research** folder captures deep-dive investigations, design spikes, competitive analyses, and API explorations that inform the Analyst Augmentation Agent roadmap.  
This `README.md` offers:

1. A categorized list of **current research papers** with one-line summaries.  
2. A **template & guidelines** for adding new research documents so knowledge stays structured and discoverable.

---

## 2. Current Research Library  

| # | Document | Category | Summary |
|---|----------|----------|---------|
| 1 | `crewai-analystagent-factory.md` | **Multi-Agent Architecture** | Factory-oriented blueprint for orchestrating CrewAI agents inside the Analyst Agent. |
| 2 | `crewai-analystagent.md` | **Multi-Agent Architecture** | Core concepts of CrewAI roles, goals, and tool chaining tailored to fraud analysis. |
| 3 | `gemini-llm-provider-design.md` | **LLM & Prompt Engineering** | Design decisions for integrating Google Gemini (text + vision) as primary LLM provider. |
| 4 | `crypto-multichain-apis.md` | **Blockchain / Crypto Data** | Survey of multichain data providers (Dune, Covalent, Moralis) and rate-limit comparisons. |

### Category Legend
| Category | Scope |
|----------|-------|
| **Multi-Agent Architecture** | CrewAI patterns, role design, task routing, agent collaboration. |
| **LLM & Prompt Engineering** | Provider evaluations, prompt templates, context-window strategies. |
| **Blockchain / Crypto Data** | On-chain analytics APIs, indexers, schema mapping, rate limits. |
| **Observability & DevOps** | Logging, tracing, CI/CD, infra benchmarking. |
| **Security & Compliance** | Threat models, data-privacy constraints, regulatory research. |

---

## 3. Contributing New Research  

### 3.1 File Naming Convention
```
yyyy-mm-dd-key-topic.md
```
*Examples:*  
`2025-07-02-open-telemetry-evaluation.md`  
`2025-07-15-llm-context-window-benchmarks.md`

### 3.2 Front-Matter Header (optional)
```yaml
title: OpenTelemetry Evaluation
author: Jane Doe
date: 2025-07-02
category: Observability & DevOps
status: draft   # draft | reviewed | adopted
```

### 3.3 Recommended Document Outline
1. **Problem / Question** – What are we trying to learn or decide?  
2. **Background** – Existing knowledge or constraints.  
3. **Investigation & Findings** – Experiments, data, diagrams, pros/cons.  
4. **Recommendation** – Clear suggestion or next step.  
5. **Appendix** – Raw data, links, references.

### 3.4 Review Workflow
1. Commit the new markdown file under `memory-bank/research/`.  
2. Update the table in Section 2 with a one-line summary.  
3. Open a pull request and request review from the **Domain Lead** (see `MASTER_STATUS.md`).  
4. Once merged, update `status:` front-matter from `draft` → `reviewed` or `adopted`.

---

## 4. House-Keeping Rules  

* Keep research documents **short & focused** — big reports can be split.  
* When conclusions drive code changes, reference the research doc in the PR description.  
* Outdated docs can be moved to `/archive` subfolder to declutter the index.

---

_Last updated: 2025-06-17_  
Maintainer: **Documentation Lead** (@docs-maintainer)  
