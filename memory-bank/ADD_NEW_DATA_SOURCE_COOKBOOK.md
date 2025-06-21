# 📚 Developer Cookbook: Adding a New Data Source

Welcome to the definitive guide for integrating **any** external data source (API, stream, or feed) into Coding-Analyst-Droid.  
Follow the steps below to go from *idea ➜ production-ready provider* in a single afternoon.

---

## 0. Prerequisites

• Docker / Docker-Compose running  
• Local Neo4j & Redis from `docker-compose up -d`  
• Python ≥ 3.10 with virtualenv  
• `poetry install` (or `pip -r requirements.txt`)  
• Environment variables copied from `.env.example`

---

## 1️⃣ Step-by-Step Workflow

| # | Action | Tools/Files |
|---|--------|-------------|
| 1 | Draft provider config (JSON/YAML) | `provider_config.json` |
| 2 | Run scaffold generator | `python scripts/new_provider_scaffold.py --config provider_config.json` |
| 3 | Fill in API-client methods | `backend/integrations/<provider>_client.py` |
| 4 | Implement tool logic | `backend/agents/tools/<provider>/<tool>.py` |
| 5 | Wire data → graph | use `Neo4jLoader` helpers |
| 6 | Add metrics / cost tags (optional) | Prometheus counters already injected |
| 7 | Run integration test matrix | `python scripts/integration_test_matrix.py --provider <provider>` |
| 8 | Generate Grafana dashboards | `python scripts/generate_grafana_dashboard.py --provider <provider>` |
| 9 | Create PR + CI green | GitHub Actions auto-runs test matrix |
|10 | Deploy & monitor | Docker / K8s + Grafana + Sentry |

---

## 2️⃣ Provider Config Template

`provider_config.json`

```json
{
  "name": "example_blockchain_api",
  "provider_type": "rest",
  "base_url": "https://api.example.com/v1",
  "auth_method": "api_key",
  "data_types": ["blockchain_transactions", "wallet_balances"],
  "description": "Example API for blockchain data",
  "rate_limit": { "requests_per_minute": 60 },
  "endpoints": {
    "get_transactions": "/transactions",
    "get_balance": "/balances/{address}"
  },
  "tools": [
    {
      "name": "example_transactions_tool",
      "description": "Retrieve transactions",
      "endpoints": ["get_transactions"]
    }
  ]
}
```

Run:

```bash
python scripts/new_provider_scaffold.py --config provider_config.json
```

---

## 3️⃣ Code Implementation Highlights

### 3.1 Client Snippet (REST)

Located at `backend/integrations/example_blockchain_api_client.py`

```python
class ExampleBlockchainApiClient:
    async def get_transactions(self, address: str, limit: int = 10):
        url = f"{self.base_url}/transactions"
        return await self._make_request("GET", url, params={"address": address, "limit": limit})
```

### 3.2 Tool Snippet

`backend/agents/tools/example_blockchain_api/example_transactions_tool.py`

```python
class ExampleTransactionsTool(AbstractApiTool):
    name = "example_transactions_tool"
    provider_id = "example_blockchain_api"

    async def _execute(self, request: ExampleTransactionsToolRequest):
        data = await self.client.get_transactions(request.address, request.limit)
        # optional: ingest into Neo4j
        await self._ingest_transactions(data["transactions"])
        return data
```

---

## 4️⃣ Testing & Validation

1. **Unit Tests** – auto-generated in `tests/test_<provider>_client.py`  
2. **Tool Tests** – `tests/test_<tool>.py` covers `_execute` logic  
3. **Integration Matrix** – run

   ```bash
   python scripts/integration_test_matrix.py --provider example_blockchain_api --verbose
   ```

   ✓ Verifies API mocks, Neo4j ingestion, metrics, rate-limit handling.

4. **Static & Type Checks**

   ```bash
   ruff check .
   mypy backend/integrations/example_blockchain_api_client.py
   ```

---

## 5️⃣ Integration Touch-Points

| System | How it’s wired |
|--------|----------------|
| Neo4j  | Use `Neo4jLoader.ingest_*` helpers inside tool `_execute` |
| Redis  | `AbstractApiTool` provides transparent caching (`cache_ttl`) |
| Prometheus | Counters & histograms auto-labelled with `provider` / `endpoint` |
| Sentry | Exceptions bubble to `sentry_config.py` automatically |
| OpenTelemetry | HTTPX, Redis, Neo4j spans auto-instrumented |

---

## 6️⃣ Best Practices & Pitfalls

✔  **Keep provider IDs snake_case** – matches env vars & registry keys  
✔  **Pagination** – loop internally; expose `limit` param to callers  
✔  **Cost Tracking** – set `credit_type` labels if the API charges tokens/credits  
✔  **Rate Limits** – honour `Retry-After`; scaffold includes retry/back-off  
❌  **Don’t block event loop** – all client calls are async  
❌  **Avoid large response bodies in Neo4j** – extract only needed fields

---

## 7️⃣ Configuration & Environment

1. **API Keys**

   Add to `.env`

   ```
   EXAMPLE_BLOCKCHAIN_API_KEY=your_key_here
   ```

2. **Registry**

   `backend/providers/registry.yaml` auto-updated – review `rate_limit`, `retry`, `cost_rules`.

3. **Docker Compose**

   Nothing extra unless the provider needs local mock service; add service under `docker-compose.yml`.

---

## 8️⃣ Monitoring & Observability

Metric | Description | Example Query
------ |-------------|--------------
`external_api_calls_total` | Counter per provider/endpoint | rate(external_api_calls_total{provider="example_blockchain_api"}[5m])
`external_api_duration_seconds` | Histogram (latency) | histogram_quantile(0.9, ...)
`external_api_credit_used_total` | Cost counter | increase(external_api_credit_used_total{provider="example_blockchain_api"}[1h])

Generate dashboards:

```bash
python scripts/generate_grafana_dashboard.py --provider example_blockchain_api --output-dir grafana_dashboards
```

---

## 9️⃣ Deployment Checklist

- [ ] **Secrets** in Kubernetes `Secret` or CI vault  
- [ ] **Prometheus scrape** includes `/metrics` on backend pod  
- [ ] **Grafana dashboards** imported  
- [ ] **Back-pressure middleware** thresholds tuned (`BUDGET_CHECK_INTERVAL_SECONDS`, budgets)  
- [ ] **OTEL_EXPORTER_TYPE** & endpoint configured for tracing  
- [ ] **CI** passes unit, integration matrix, lint, type checks  

---

## 🔧 10️⃣ Troubleshooting Guide

Problem | Likely Cause | Fix
------- | ------------ | ---
`Provider not found in registry` | Scaffold not run / typo | Ensure `id` matches provider folder
`Missing API key` | Env var not set | Add `EXAMPLE_BLOCKCHAIN_API_KEY`
`429 Too Many Requests` | Rate limit hit | Check scaffold retry; Grafana rate-limit dashboard
`neo4j.exceptions.ClientError` | Bad Cypher | Validate ingestion queries; enable `explain_cypher` prototype
`external_api_credit_used_total` stays 0 | Cost rules not configured | Add `cost_rules` in registry.yaml

---

## 🔄 Examples by Data Source Type

### REST (JSON)

• Use `provider_type: rest`  
• Endpoints defined as path strings  
• Client uses `_make_request`

### GraphQL

• `provider_type: graphql`  
• Endpoints map to GraphQL queries; auto-generated client uses `_execute_query`

### WebSocket / Streaming

• `provider_type: websocket`  
• Endpoints define channels & subscribe messages  
• Client manages reconnects & pushes messages to callbacks

---

## 🎉 You’re Done!

Merge your PR ➜ CI green ➜ dashboards live ➜ **new data flowing into the graph** 🚀
