# Add New Data Source Cookbook

**A step-by-step guide for integrating new blockchain data providers into the Analyst Augmentation Agent**

*Updated with real examples from Covalent & Moralis integrations (June 2025)*

---

## Overview

This cookbook explains how to add *any* blockchain data provider to the `droid101` project while preserving our unified pattern:

| You will build | Outcome |
| -------------- | ------- |
| **Provider Client** | Async REST / GraphQL client with auth, retries & Prometheus metrics |
| **Tool Integration** | `AbstractApiTool` subclass so Crew agents can consume the API |
| **Registry Entry** | Provider config block with rate-limits, retry & cost tracking |
| **Integration Tests** | Neo4j-mocked test suite + optional real-API smoke tests |
| **Documentation** | Cookbook update + sprint checklist entry |

---

## Prerequisites

* Python 3.8+ environment  
* Familiarity with async/await & HTTP APIs  
* API key (or free-tier token) for the new provider  
* Access to provider docs (endpoints, auth, limits)

---

## Step 1 – Generate Boilerplate with the Scaffold

The one-stop script lives at `scripts/new_provider_scaffold.py` (≈ 1 800 LOC).

```bash
python scripts/new_provider_scaffold.py
```

Follow the prompts:

| Prompt | Example (Covalent) |
| ------ | ----------------- |
| Provider id | `covalent` |
| Provider name | Covalent |
| Base URL | https://api.covalenthq.com/v1 |
| Auth method | `api_key` |
| Data types | `blockchain_transactions,wallet_balances,token_transfers,nft_data` |
| Rate limits | 100 req/min, 10 000 req/day |
| Endpoints | `/ {chain_id} /address/ {address} /balances_v2/` etc. |

**Tip:** For non-interactive use supply `--config provider.json` with the same fields.

---

## Step 2 – Inspect Generated Files

```
├── covalent_config.json
├── backend/integrations/covalent_client.py
├── backend/agents/tools/covalent_balances_tool.py
└── tests/test_covalent_integration.py
```

### What each part does

1. **Client** – Handles auth header / basic-auth, retries + back-off, metrics.  
2. **Tool** – Validates params (`pydantic`), maps them to client methods, surfaces to CrewAI.  
3. **Tests** – Real-API (if key set) + Neo4j mock ingest + validation errors.

---

## Step 3 – Add Provider to the Registry

Open `backend/providers/registry.yaml` and append:

```yaml
# -------------------------------------------------------------------------- #
# N. Covalent (Multi-chain Data)
# -------------------------------------------------------------------------- #
- id: "covalent"
  name: "Covalent"
  description: "Covalent unified API for multi-chain blockchain data."
  auth:
    api_key_env_var: "COVALENT_API_KEY"
  budget:
    monthly_usd: 0.0            # Free tier
  rate_limits:
    requests_per_minute: 100
    requests_per_day: 10000
  cost_rules:
    default_cost_per_request: 0.0
  retry_policy:
    attempts: 3
    backoff_factor: 1.0
```

Repeat for Moralis:

```yaml
- id: "moralis"
  name: "Moralis"
  description: "Moralis Web3 API for NFT metadata and multi-chain data."
  auth:
    api_key_env_var: "MORALIS_API_KEY"
  rate_limits:
    requests_per_minute: 120
    requests_per_day: 25000
```

---

## Step 4 – Customise as Needed

### Auth patterns

* **X-API-Key header** (Moralis)

```python
if self.api_key:
    self.headers["X-API-Key"] = self.api_key
```

* **Basic auth** (Covalent)

```python
if self.api_key:
    self.auth = (self.api_key, "")
```

### Tool routing examples

*Default on `token_id` presence*  

```python
if "token_id" in params:
    return await self.get_nft_metadata(params)
return await self.get_wallet_nfts(params)
```

*Explicit `method` parameter*  

```python
method = params.pop("method", "get_wallet_nfts")
return await getattr(self, method)(params)
```

---

## Step 5 – Run Tests

```bash
# single provider
python scripts/integration_test_matrix.py --provider covalent --verbose

# all providers
python scripts/integration_test_matrix.py
```

Set env-vars first:

```bash
export COVALENT_API_KEY="xxx"
export MORALIS_API_KEY="yyy"
```

The matrix script spins a mock transport but will hit live APIs when keys exist.

---

## Step 6 – Document & Push

1. Update `memory-bank/TODO_SPRINT*.md` → mark task ✅  
2. Append a short entry under *Phase 4 – Extensibility Hooks* in `MASTER_STATUS.md`.  
3. Commit:

```bash
git add backend/memory-bank providers tests
git commit -m "🔥 Covalent provider integration - COOKED & PUSHED"
git push
```

---

## Advanced Patterns

| Scenario | Cookbook Tip |
| -------- | ------------ |
| Multiple tools per provider | Create `backend/agents/tools/{provider}/` package with `__init__.py` |
| GraphQL providers | Scaffold supports `--type graphql`, generates `_execute_query()` |
| WebSockets | Scaffold supports `--type websocket`, adds `_subscribe()` helpers |
| Custom Neo4j ingest | Place logic in `scripts/integration_test_matrix.py` provider hook |

---

## Troubleshooting Checklist

| Symptom | Fix |
| ------- | --- |
| `ImportError backend.integrations.{provider}_client` | Run scaffold or add file to `__init__.py` |
| Auth 401 | Ensure `ENV_VAR` matches `registry.yaml` |
| Neo4j ingestion silent | Confirm you called `neo4j_loader.ingest_*` or crafted `_execute_query` |
| Rate-limit loops | Check provider headers (`Retry-After`) and `retry_config` |

---

## Success Metrics

* **CI** green (unit + integration)  
* **Prometheus** `external_api_call_total{provider="new"}` increments  
* **Graph** nodes/relationships created for sample wallet  
* **CrewAI** can call the new tool via chat (manual smoke test)  

---

### Real-World Examples

| Provider | Free Tier | Specialisation | Graph Pattern |
| -------- | --------- | -------------- | ------------- |
| Covalent | 1 000 calls/day | Token balances & tx | `Address-HOLDS-Token` |
| Moralis | ~40 k CU/month | NFT metadata & ownership | `Address-OWNS-NFT` |

---

*Last updated: 28 Jun 2025 – Factory Droid*  
“**Cook & push to GitHub**” 🌶️🚀
