# Session Handoff – 31 May 2025 (RBAC, metrics, graph MVP)

## What was delivered this session

| Item | PR | Notes |
|------|----|-------|
| Prometheus LLM token & cost counters | #23 | `GeminiClient` now updates `llm_tokens_used_total`, `llm_cost_usd_total` |
| Front-end GraphVisualization MVP | #25 | vis-network canvas, label-based colours, PNG export |
| End-to-end smoke test | #26 | `/crew/run` mocked, asserts Prometheus counters increment |
| RBAC – Phase 1 | _(this PR)_ | `/prompts/*` admin-only, `/graph/*` analyst or admin; `request.state.user` set for rate-limiting |

---

## Next recommended steps

1. Extend RBAC to `/crew/run`, `/analysis/*` and add unit tests (401 / 403 / 200 cases).  
2. Finish cost-telemetry widget in the front-end (`Header` component).  
3. Lock dependencies with **uv / poetry** to stabilise CI.

---

## Open questions

* **Role claim source** – currently expects `user_data.role` in JWT. Decide on issuer mapping or external IdP.  
* **Rate-limit store** is in-memory; migrate to Redis for multi-instance deployment.

–– **Droid session end** ––
