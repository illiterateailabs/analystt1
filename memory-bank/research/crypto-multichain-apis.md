# Multi-Chain Crypto Data APIs – GraphQL-Focused Cheat-Sheet  
_Last reviewed: 01 Jun 2025_

This note distills the 2025 market report into an **actionable playbook** for analyst-agent-illiterateai.  
We only track vendors that (a) expose a GraphQL interface _today_ or via subgraphs, and (b) deliver data useful for **fraud detection / on-chain investigations**.

| Rank | Service | Native GraphQL? | Chains* | Fraud-Detection Highlights | Indicative Pricing** |
|------|---------|-----------------|---------|---------------------------|----------------------|
| 1 | **The Graph Protocol** | ✅ (Subgraphs) | 90+ | Custom subgraphs; decoded events; on-chain state snapshots; real-time subscriptions | 100 k free queries/mo then pay-per-GRT (~\$0.04 / 1 k) |
| 2 | **QuickNode – GraphAPI** | ✅ | 65+ | Archive queries; NFT & token endpoints; WebSocket streams for anomaly triggers | Free 10 M credits; Pro \$49+ (credits) |
| 3 | **Bitquery** | ✅ | 40+ | Unified schema incl. DEX trades, address flows, Coinpath® money-trail | Free dev tier; >100 k req ≈ \$99/mo |
| 4 | **Chainstack – Subgraphs** | ✅ (Elastic & custom) | 25+ | Pre-built DeFi subgraphs (Uniswap, Aave, Curve); real-time GraphQL | 3 M RU free; \$49 / 20 M RU |
| 5 | **Covalent** | REST (SQL-like)*** | 100+ | Full historical wallets, decoded logs, XY=K DEX data; JSON easy to convert | Premium 50 k credits \$50/mo |
| 6 | **Alchemy** | ➖ (REST/WS) | 30+ | Transfers API (full address history), Trace API (internal tx), NFT spam filter | 100 M CU free; PAYG \$0.000001/CU |
| 7 | **Ankr – Advanced APIs** | ➖ (REST) | 55+ | Multi-chain query in one call; holder distributions; token price | 30 r/m free; \$10 /100 M credits |
| 8 | **CryptoAPIs.io** | ➖ (REST) | 30+ | AddressHistory (genesis); exchange order books; webhook alerts | Starter 300 M credits \$49/mo |

\* _Mainnet count; testnets excluded_  
\** _Public list prices – enterprise deals vary_  
\*** _Covalent offers SQL-style REST; GraphQL proxy possible via Postman Transformer_

---

## 1. Data Features That Matter for Fraud Detection

| Capability | The Graph | QuickNode | Bitquery | Chainstack | Notes |
|------------|-----------|-----------|----------|------------|-------|
| Historical full-node / archive | Via subgraph indexing | ✅ | ✅ | ✅ | Use for retro investigations |
| Decoded contract events | ✅ | Partial (NFT/Token) | ✅ | ✅ | Saves manual ABI decode |
| Cross-chain wallet history | Community subgraphs | ✅ (Transfers API) | ✅ | — | Needed to follow laundering paths |
| DEX trade traces | Custom subgraph | ✅ | ✅ | Pre-built XY=K | Identify wash-trades / spoofing |
| Real-time WebSocket push | `subscribe { … }` | Streams API | GraphQL subscriptions | GraphQL subscriptions | Alert pipeline for pumps |
| Money-flow graph | Build via subgraphs | Must aggregate | Coinpath® ready | Build via NetworkX | Critical for layering detection |

---

## 2. Integration Recommendations

1. **GraphQLQueryTool presets**  
   ```python
   THE_GRAPH = "https://api.thegraph.com/subgraphs/name/{slug}"
   BITQUERY  = "https://graphql.bitquery.io"
   QUICKNODE = "https://your-endpoint.quicknode.com/graphql"
   ```

2. **Auth & rate-limit env vars**  
   ```
   export THEGRAPH_KEY=<GRT_token>
   export BITQUERY_KEY=<bitquery>
   export QUICKNODE_KEY=<api_key>
   ```

3. **Back-off wrapper**  
   - The Graph: 100 req/min  
   - Bitquery: 60 req/min  
   - QuickNode: credit burn per call – track with Prometheus label `graphql_endpoint`.

4. **Cache immutable queries**  
   - Block/tx lookups => Redis TTL ∞  
   - Token metadata => 6 h TTL  
   - Subgraph responses for last 24 h => 5 min TTL

5. **Fraud-pattern data sources**  
   | Pattern | Primary Source | Query Sketch |
   |---------|----------------|--------------|
   | Structuring (many <10 k) | Bitquery | `transactions(where:{value_lt:10e18, address:"…"})` |
   | Wash-trading NFT | Chainstack Uniswap subgraph | Fetch swaps, group by wallet pairs |
   | Flash-loan attack | The Graph (Aave) | Subscription on `FlashLoan` events |

6. **Cost control**  
   - Prefer The Graph (subgraph) for heavy historical scans (cheap GRT)  
   - Use QuickNode Streams only for hot addresses (credits expensive)  
   - Batch queries (Bitquery supports 10 queries/mutation)

---

## 3. Pricing Cheatsheet (budget 2025)

| Monthly volume | Cheapest mix |
|----------------|--------------|
| ≤ 100 k queries | The Graph free tier + Bitquery dev |
| 1 M – 5 M | Chainstack Growth (\$49) + Bitquery Basic (\$99) |
| High-freq alerts | QuickNode Build (\$49) Streams + The Graph paid |
| Enterprise | Combine Covalent Platinum + Blockdaemon |

---

## 4. Next Implementation Tasks

1. **Add endpoint registry** (`settings.graphql_endpoints`) for easy switch.
2. **Write Bitquery -> Neo4j loader** for Coinpath® trails.
3. **Prototype live pump-&-dump detector** using QuickNode Streams → anomaly queue.
4. **Cost monitor** – extend Prometheus counter `graphql_credits_used_total`.

---

_This file is a living reference – update when new providers or pricing changes._  
