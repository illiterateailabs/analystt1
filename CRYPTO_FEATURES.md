# Crypto-Analysis Feature Pack

Welcome to the **Crypto Feature Pack** – a turnkey extension that transforms the Analyst-Augmentation Agent into a **multi-chain crypto / DeFi investigation platform**.

---

## 1. What’s Inside 🚀

| Layer | New Item | Purpose |
|-------|----------|---------|
| **Tools** | `dune_analytics_tool`, `defillama_tool`, `etherscan_tool` | Connect to the major public data platforms for on-chain analytics, DeFi metrics and block-explorer look-ups |
| **Agents** | `crypto_data_collector`, `blockchain_detective`, `defi_analyst`, `whale_tracker`, `protocol_investigator` | Specialist LLM agents that leverage the new tools |
| **Crew** | `crypto_investigation` | End-to-end workflow that orchestrates all crypto agents and produces a markdown report |
| **Config** | `CryptoConfig` & extensive `.env.example` additions | Centralised storage of API keys and rate-limit settings |
| **Docs** | *this file* | Detailed HOW-TO & reference |

---

## 2. Tool Reference

| Tool | Primary API | Highlights | Example `action` / helper |
|------|-------------|-----------|---------------------------|
| **DuneAnalyticsTool** | Dune API v1 (SQL) | • Multi-chain curated data<br>• Custom SQL execution<br>• Helper methods: wallet, DeFi, whale analysis | `execute_query(query_id=12345, parameters={...})` |
| **DefiLlamaTool** | DefiLlama REST / Yields / Stablecoins | • TVL & yield tracking<br>• Protocol comparison<br>• Chain dominance | `find_top_yield_opportunities(min_apy=8)` |
| **EtherscanTool** | Etherscan-family REST (multi-chain) | • Address & token forensics<br>• Contract source/ABI fetch<br>• Related-address graphing | `analyze_wallet(address=0x..., include_nfts=true)` |

All tools are **async**, rate-limit aware, and can **store results to Neo4j** for graph queries downstream.

---

## 3. New Agents & Their Super-Powers

| Agent ID | Key Tools | Core Skill |
|----------|-----------|-----------|
| `crypto_data_collector` | Dune, DefiLlama, Etherscan | Aggregates & normalises data from multiple APIs |
| `blockchain_detective` | Etherscan, Dune | Traces transactions, clusters addresses, follows cross-chain hops |
| `defi_analyst` | DefiLlama, Dune, Etherscan | Evaluates DeFi TVL, yields, protocol health & risk |
| `whale_tracker` | Etherscan, Dune | Monitors large-value transfers, exchange inflows/outflows |
| `protocol_investigator` | Etherscan, Dune, Sandbox | Dissects smart-contract interactions, identifies exploits |

---

## 4. The `crypto_investigation` Crew Workflow

1. **Collect** – `crypto_data_collector` pulls raw data.  
2. **Trace** – `blockchain_detective` builds address relationship graphs.  
3. **DeFi Lens** – `defi_analyst` analyses protocol exposure & yield history.  
4. **Whale Watch** – `whale_tracker` flags large-holder behaviour.  
5. **Contract Deep-Dive** – `protocol_investigator` inspects ABI & function calls.  
6. **Report** – `report_writer` turns findings into an actionable markdown brief.

Kick it off:

```bash
curl -X POST http://localhost:8000/api/v1/crew/run \
  -H "Authorization: Bearer <JWT>" \
  -d '{ "crew_name": "crypto_investigation", "inputs": { "target_wallet": "0x…", "hours": 24 } }'
```

---

## 5. API-Key Setup  🔑

Add keys to `.env` (see template):

| Variable | Needed For | Free Tier? |
|----------|------------|-----------|
| `DUNE_API_KEY` | DuneAnalyticsTool | ✔️ |
| `ETHERSCAN_API_KEY` | Ethereum via EtherscanTool | ✔️ (5 RPS) |
| `BSCSCAN_API_KEY`, `POLYGONSCAN_API_KEY`, … | Other chains in EtherscanTool | ✔️ |
| *(optional)* `COINGECKO_API_KEY`, `MORALIS_API_KEY`, `COVALENT_API_KEY` | Future integrations | ✔️ / paid |

No key? — tools still load but will return **error strings** in responses.

---

## 6. Quick-Start for Crypto Analysis

```bash
# 1. Provide keys
cp .env.example .env
# ↳ fill DUNE_API_KEY and ETHERSCAN_API_KEY (others optional)

# 2. Build & launch
make start-dev   # docker-compose backend+frontend+dbs

# 3. Verify
make health          # /health, /health/crew show new crypto tools
make crew-run name=crypto_investigation  # helper alias

# 4. Explore results
http://localhost:8000/docs      # interactive API
http://localhost:7474           # Neo4j browser for graphs
```

---

## 7. Supported Networks & Data Sources

| Network / Source | Tool | Coverage |
|------------------|------|----------|
| Ethereum Mainnet / Testnets | EtherscanTool, DuneAnalyticsTool | Blocks, txs, contracts |
| Binance Smart Chain | EtherscanTool (BSCScan) | Blocks, txs, BEP-20 |
| Polygon | EtherscanTool (PolygonScan), Dune | Blocks, txs, tokens |
| Arbitrum / Optimism | EtherscanTool variants, Dune | Layer-2 txs |
| Avalanche, Fantom, … | EtherscanTool variants | Blocks, txs |
| **200+ Chains** | Covalent (future) | Raw decoded events |
| DeFi protocols (TVL/metrics) | DefiLlamaTool | 1 000+ protocols |
| SQL datasets (DEX trades, bridges, MEV) | DuneAnalyticsTool | Curated & custom |

---

## 8. Example NLQ Flow

*“Which wallets received **≥ 100 ETH** from Tornado Cash in the last 24 h and deposited to Binance?”*

1. **User** asks in chat.  
2. `nlq_translator` builds a Cypher query.  
3. `blockchain_detective` enriches with Etherscan transfers.  
4. `defi_analyst` cross-references CeFi deposit addresses from Dune.  
5. `report_writer` returns list + risk commentary.

---

## 9. Extending Further 🛠️

*Ideas*  
- Add **QuickNode RPC** for raw mempool streams.  
- Integrate **The Graph** subgraph queries into `crypto_data_collector`.  
- Hook **Covalent** for 200+ chain uniform‐event ingestion.  
- Build a **real-time alert agent** subscribing to Moralis Streams.

---

Happy hunting 🔎  
*— Analyst Augmentation Agent team*
