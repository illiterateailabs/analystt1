# Sim API Integration Plan  
_Analyst Augmentation Agent – v1.0_

> Location: `memory-bank/SIM_API_INTEGRATION_PLAN.md`  
> Last updated: 2025-06-18

---

## 1 Goals

| Objective | Success Metric |
|-----------|----------------|
| Ingest **real-time multichain data** via Sim APIs | Balances/Activity latency < 2 s |
| Enrich **graph & ML pipeline** with on-chain signals | +10 % F1 on fraud classifier |
| Replace UI mocks with **live wallet data** | All KPI cards populated |
| Enable **LLM–driven investigations** through CrewAI tools | 1-click “Fetch wallet activity” in chat |

---

## 2 Endpoint-to-Component Mapping

| Sim Endpoint | Backend Agent / Tool | Neo4j Graph Mapping | UI Surface |
|--------------|----------------------|---------------------|------------|
| **Balances** `/v1/evm/balances/{addr}` | `balances_tool.py` (new) → `BalancesAgent` | `(Wallet)-[:OWNS_BALANCE]->(TokenBalance)` {valueUsd} | Dashboard KPI cards, Wallet panel |
| **Activity** `/v1/evm/activity/{addr}` | `activity_tool.py` → `ActivityAgent` | `(:Tx)<-[:SIGNED]-(Wallet)` + typed edges `SENT/RECEIVED/CALL` | “Activity” tab timeline |
| **Collectibles** `/v1/evm/collectibles/{addr}` | `collectibles_tool.py` → `NFTAgent` | `(Wallet)-[:OWNS_NFT]->(NFT)` | Collectibles gallery |
| **Transactions** `/v1/evm/transactions/{addr}` | `transactions_tool.py` | Full raw tx node `(TxRaw)` for forensic drill-down | Hidden dev panel / JSON viewer |
| **Token-Info** `/v1/evm/token-info/{token}` | `token_info_tool.py` | `(Token)-[:PRICE]->(USDC)` attrs {priceUsd, poolSize} | Token badge (price + liquidity) |
| **Token-Holders** `/v1/evm/token-holders/{cid}/{token}` | `holders_tool.py` | `(Wallet)-[:HOLDS_%]->(Token)` plus distribution stats property on Token | Holder distribution widget |
| **Supported-Chains**  | Shared util for dropdown autogen | — | Settings → Chain selector |
| **SVM Balances/Tx**  | Same pattern via `svm_balances_tool.py` | `chain:"solana"` labels | Cross-chain toggle |

---

## 3 Technical Implementation

### 3.1 Backend (FastAPI + CrewAI)

```
backend/agents/
├─ tools/
│   ├─ balances_tool.py      # wraps Sim call + schema validation
│   ├─ activity_tool.py
│   └─ … (others)
├─ balances_agent.py         # calls tool, pushes events
└─ …
```

* **Tool pattern**  
  ```python
  class BalancesTool(BaseTool):
      name, description = …
      def run(self, wallet: str, limit: int = 100):
          url = f"{SIM_URL}/v1/evm/balances/{wallet}?limit={limit}&metadata=url,logo"
          return req.get(url, headers=SIM_HEADERS).json()
  ```
* **CrewAI workflow**  
  1. Chat → Function-call _get_wallet_balances_  
  2. `BalancesAgent` executes tool  
  3. Emits `GraphAddEvent` → `events.py`  
  4. WebSocket pushes to UI, Graph updated.

### 3.2 Graph (Neo4j 5)

Cypher ingestion template (run in async worker):

```cypher
UNWIND $balances AS b
MERGE (w:Wallet {address:$wallet})
MERGE (t:Token {address:b.address, chain:b.chain})
MERGE (w)-[r:OWNS_BALANCE]->(tb:TokenBalance {
  id: apoc.create.uuid(), block_time: timestamp()
})
SET tb.amount = b.amount,
    tb.valueUsd = b.value_usd,
    r.last_seen = timestamp()
```

### 3.3 Frontend (Next 14 + Zustand)

* `useBalancesQuery(wallet)` built with **@tanstack/react-query**.  
* Zustand slice `balancesSlice` merges into existing investigation store.  
* Replace mocked components:
  * `components/dashboard/KPICard.tsx` → real data
  * `components/activity/Timeline.tsx` → Sim Activity feed  
* Risk badge uses `token_info.pool_size` for **low-liquidity warning**.

---

## 4 Priority Roadmap

| Order | Feature | Endpoints | Effort | Owner |
|-------|---------|-----------|--------|-------|
| P0 | **Wallet Balances + Activity ingestion** | Balances, Activity | 2 d BE / 1 d FE | Backend + UI |
| P1 | **Graph enrichment & ML features** | Balances, Activity, Token-Info | 1 d BE | Data Graph |
| P2 | **NFT Collectibles gallery** | Collectibles | 0.5 d FE | Frontend |
| P3 | **Liquidity-aware spam filter** | Token-Info | 0.5 d BE | Backend |
| P4 | **Whale / Sybil detection widgets** | Token-Holders | 2 d BE + FE | Fraud-ML |
| P5 | **Solana support** | SVM Balances/Tx | 1 d BE | Backend |

---

## 5 Fraud-Detection Enhancements

| Pattern | Data Needed (Sim) | Detection Logic |
|---------|-------------------|-----------------|
| **Low-liquidity dump** | Token-Info.pool_size, Activity | tkn `low_liquidity == true` **AND** wallet sells > $10k in 24 h |
| **Peel chain structuring** | Activity (send/receive) | Depth > 10 sequential txs with decreasing amounts |
| **NFT wash trading** | Collectibles + Activity | Same wallet ↔ same counterparty swaps same NFT ≥ 3× |
| **Whale concentration** | Token-Holders | Top-10 hold > 90 % supply AND recent price spike |
| **Cross-chain bridge abuse** | Balances (multi-chain) | Rapid drain on Chain A, fund appearance Chain B within 5 min |

Graph-GNN retraining to incorporate new edge types will follow P2.

---

## 6 Security & Ops

* Store **`SIM_API_KEY`** in `backend/.env` and staged secrets in GH Actions.  
* Rate-limit: wrap tools with exponential back-off on `429`.  
* Cloudflare Worker proxy possible but **backend only** (no browser key leak).  
* Observability: add Sim call_duration metric to Prometheus.

---

## 7 Documentation Updates

* Update `TECHNICAL_ARCHITECTURE.md` data-flow diagram.  
* Add “Sim Data Ingestion” section to `CAPABILITIES_CATALOG.md`.  
* Include troubleshooting tips from Sim **Error Handling** guide.

---

## 8 Completion Definition of Done

- [ ] All P0 endpoints live in Dev & CI preview  
- [ ] Graph nodes/edges appear for test wallet `0xd8da…`  
- [ ] UI KPI cards show non-zero values  
- [ ] CrewAI chat tool “Get wallet balances” returns Sim data  
- [ ] Added unit tests ≥ 80 % for tool wrappers  
- [ ] Memory-bank docs updated & MR approved

---

_“Cook & Push — but always measure the hash rate.”_  
— AAA Core Team
