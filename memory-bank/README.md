# Memory-Bank Documentation Hub

Welcome to the **Memory-Bank** &mdash; the living knowledge base for the Analyst Augmentation Agent project (`anal-ist1`).  
This folder centralises **architecture diagrams, design notes, research, hand-off logs and operational status records** so that every contributor can quickly find the “why” behind the code.

---

## 📑 Quick Links

| Domain | Document | Purpose |
| ------ | -------- | ------- |
| **Project Status** | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Canonical health dashboard (road-map, risks, release notes). |
| **Phase 2 Roadmap (history)** | [`PHASE_2_ROADMAP.md`](PHASE_2_ROADMAP.md) | Completed feature-wave log & timeline. |
| **Architecture** | [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md) | Component breakdown of the FastAPI + Next.js + Neo4j/PostgreSQL stack. |
| **Capabilities** | [`CAPABILITIES_CATALOG.md`](CAPABILITIES_CATALOG.md) | Catalogue of built-in analysis & fraud-detection capabilities. |
| **Docs House-keeping** | [`CLEANUP_SUMMARY.md`](CLEANUP_SUMMARY.md) | Rationale & rules for doc consolidation. |
| **Code ↔ Doc Audit** | [`CODEBASE_VERIFICATION.md`](CODEBASE_VERIFICATION.md) | Truth-table of documented vs implemented features. |
| **Sessions / Hand-offs** | `session-handoff-YYYY-MM-DD-*.md` | Daily transfer logs to preserve decision context across shifts. |
| **Research** | [`research/`](research) | External notes, competitive analysis, API deep-dives. |
| **Docs Improvement Plans** | `DOCUMENTATION_*` files | Planned clean-ups & doc debt tasks. |

> Looking for **code**? See `/backend`, `/frontend` and `/tests` at project root.  
> Looking for **run instructions**? See repository [`README.md`](../README.md).

---

## 🏗️ Current Architecture Snapshot (July 2025 &nbsp;· v2.0.0-beta)

```
Users ⟶ Next.js Front-End (TS/React 18)
               │
               ▼
 FastAPI Backend (Python 3.11)
    ├─ Authentication & RBAC
    ├─ CrewAI Workflow Engine
    ├─ Chat & Image Analysis (Gemini)
    ├─ Graph Endpoints (Cypher, NLQ)
    ├─ Advanced Graph (GAT, Communities, Risk-Prop)
    ├─ HITL Webhooks & Reviews
    ├─ ML Risk-Scoring Service (ensemble + SHAP)
    ├─ Streaming Relay (Redis Streams ➜ WebSocket)
    └─ Observability (Prometheus, Sentry)

Persistent Stores
    ├─ PostgreSQL 15 (async SQLAlchemy)
    └─ Neo4j 5 (Graph DB + GDS)

Async Workers / Sandbox
    └─ e2b.dev for code execution

Multi-Tenant Isolation
    ├─ Field / label filters (OSS default)
    └─ Optional multi-DB / schema strategy
```

For deeper detail read [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md).

---

## 🗂️ Folder Structure

```
memory-bank/
├─ PROJECT_STATUS.md             # Project health dashboard
├─ PHASE_2_ROADMAP.md            # Historical roadmap (feature wave 2)
├─ CLEANUP_SUMMARY.md            # Rationale for doc consolidation
├─ CODEBASE_VERIFICATION.md      # Docs ↔ code truth table
├─ CAPABILITIES_CATALOG.md        # Feature index
├─ DOCUMENTATION_*.md             # Doc cleanup & plans
├─ TECHNICAL_ARCHITECTURE.md      # System diagrams & flows
├─ research/                      # External research notes
├─ session-handoff-*.md           # Shift-change / on-call logs
├─ archive/                       # Historical, read-only docs
└─ (additional reference files)
```

Obsolete or duplicated docs have been removed; if you find outdated material open an issue or PR.

---

## ✍️ Contributing to Documentation

1. **Keep it source-controlled.** Always place docs in this folder so they version with code.  
2. **Prefer Markdown.** Use headings and tables for readability.  
3. **Timestamp logs.** Name hand-off files `session-handoff-YYYY-MM-DD-topic.md`.  
4. **Cross-link generously.** Link code files with the `@` mention in Factory or relative paths.  
5. **Review like code.** Submit a PR; reviewers look for clarity, accuracy and redundancy removal.

---

## 🔄 Update Cadence

| Doc Type | Owner | Refresh Cycle |
| -------- | ----- | ------------- |
| **PROJECT_STATUS.md** | Tech Lead | Weekly (Friday) |
| **Architecture & Capabilities** | Lead Architect | After major feature merge |
| **PHASE_2_ROADMAP.md** | PM | Frozen – historical record |
| **Hand-off Logs** | On-call / Pair | End of shift |
| **Research Notes** | Feature Squad | Ad-hoc |

---

## 🤝 Need Help?

* **Slack**: `#analyst-agent`  
* **Issues**: tag with `documentation` label  
* **Maintainers**: @Daniel-Wurth (Backend), @UI-Lead (Frontend), @Data-Graph (Graph/Neo4j)

Let’s keep knowledge fresh and discoverable.  
Happy documenting! 📚
