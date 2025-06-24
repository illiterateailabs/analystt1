# Personal-Project TODO – Analyst Droid One  
_Focus: Fun features & usable tooling (ignore enterprise ops)_  
_Last updated: 2025-06-23_

Legend  
• **Priority** – P0 (now) · P1 (next) · P2 (later)  
• **Effort** – S (≤2 h) · M (½–1 day) · L (multi-day)  
• **Status** – ☐ todo · ☐⧗ in-progress · ☑ done  

---

## P0 – Core Functionality & Quick Wins
| # | Task | Effort | Status | Notes |
|---|------|--------|--------|-------|
| 0-1 | **Expose `/health/workers` endpoint** from `worker_monitor.py` via API v1 | S | ☑ | Completed in commit `154bc34` – returns queue depth & worker count |
| 0-2 | **Emit SIM client cost metrics** (`external_api_credit_used_total`) | M | ☑ | Implemented in commit `185e191` – SIM client + tests & budget guard |
| 0-3 | **Finish Graph-Aware RAG “Explain-with-Cypher”** | M | ☑ | Completed in commit `92c2aef` – citations & evidence bundles integrated |
| 0-4 | **Smoke-test suite**: chat → graph → background job flow | M | ☑ | Completed in commit `e453c4` – full `test_smoke_flow.py` & runner script |

## P1 – Feature Epics (Pick & Build)
| # | Epic | Effort | Status | Notes |
|---|------|--------|--------|-------|
| 1-1 | ⚡ **DeFi Stress-Test What-If** (AI explains stress scenarios) | L | ☐ | Celery task + prompt template |
| 1-2 | 🔍 **Anomaly Hunting Fraud Detector** (GNN + heuristics) | L | ☑ | Completed in commit `9405715` – full anomaly service, API & tasks |
| 1-3 | 🌉 **Cross-Chain Liquidity Migration Predictor** | L | ☐ | Time-series model + LLM narrative |

_Start implementation with #1-1 once P0 items pass tests._

> 🎉 **All P0 tasks are now complete!** Move on to P1 epics.

> 🥳 **Milestone:** First P1 epic finished – Anomaly Hunting Fraud Detector is live!

## P2 – Polish & Delight
| # | Task | Effort | Status | Notes |
|---|------|--------|--------|-------|
| 2-1 | Chat UI progress indicator for Celery tasks | M | ☑ | Completed in commit `eb08bfa` – WebSocket real-time updates |
| 2-2 | Graph visualisation tweaks – colour fraud scores | M | ☑ | Completed in commit `ef7bd5e` – enhanced D3/Vis graph colouring |
| 2-3 | Data ingestion demo script (`scripts/demo_ingest.py`) | M | ☑ | Completed in commit `eb08bfa` – demo generator & fraud showcase |
| 2-4 | Sample fraud scenario dataset & README walkthrough | S | ☐ | Helps new users reproduce demo |

---

### Working Guidelines
1. Push straight to **main**; document each win here (update Status column).  
2. Keep tasks small & testable – aim for green smoke tests at all times.  
3. After each completed item, commit with message `feat(todo-id): …` and mark ☑ here.  

_Have fun & build cool stuff!_
