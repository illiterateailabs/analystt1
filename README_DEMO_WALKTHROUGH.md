# Analyst Droid One – Demo Walkthrough 🚀

Welcome! This guide walks you through **re-creating the full fraud-detection demo** and showcases every major capability of the Analyst Droid One platform – from data ingestion to real-time anomaly alerts and visual investigations.

---

## 1. Value-at-a-Glance 💡

| Super-power | What it means | Where you’ll see it |
|-------------|--------------|---------------------|
| 💚 Health & Observability | `/api/v1/health/*` endpoints, Prometheus metrics | Grafana / browser |
| 💸 Cost Guard | Tracks SIM / Gemini spend, budget enforcement | `/metrics`, budget dashboard |
| 🔗 Graph-Aware RAG | “Explain-with-Cypher” – LLM answers + Cypher citations | Chat panel |
| 🧠 GNN + Heuristics | Automated anomaly hunting, fraud pattern library | Alerts panel |
| 🔴 Real-time UI | WebSocket task progress & live alerts | Toasts / progress bars |
| 🌈 Visual Graph | Risk-coloured nodes, fraud-typed edges | Graph tab |
| 📚 Evidence Bundles | Auto-generated provenance & narratives | Investigation drawer |

**In short:** _import data → detect fraud in seconds → drill into evidence with explainable queries_.

---

## 2. Quick Start (10 min)

### 2.1 Prerequisites

| Tool | Version |
|------|---------|
| Docker / Docker Compose | ≥ 24 |
| Git | ≥ 2.3 |
| Make (optional) | for convenience |

### 2.2 Run the stack

```bash
git clone https://github.com/illiterateailabs/obebibbeautifulanalyst.git
cd obebibbeautifulanalyst

# spin everything up
docker compose up -d --build
# watch the logs (optional)
docker compose logs -f backend worker
```

Once you see `🌞 Celery worker monitoring background task started.` the platform is ready.

---

## 3. Load the Demo Dataset

The repository ships with a **one-click generator** producing wash-trading, smurfing, layering and high-frequency scenarios.

```bash
docker compose exec backend bash
# inside the container
python scripts/demo_ingest.py --addresses 150 --transactions 1500 --verbose
exit
```

The script:
1. Generates ~150 addresses + 1 500 transactions (normal + fraud)
2. Pushes them into Neo4j
3. Adds risk labels & patterns
4. Stores handy samples in Redis
5. Automatically runs anomaly detection and creates evidence bundles

Expected tail output:

```
✓ Generated wash trading pairs with 30 transactions
✓ Ingested 150 addresses
✓ Ingested 1500 transactions
✓ Created evidence bundle for anomaly ...
✓ Demo Complete – Happy fraud hunting!
```

---

## 4. Explore the Platform

Open http://localhost:8000  
You’ll land on the **Dashboard**.

| Screenshot | What to look for |
|------------|------------------|
| ![dashboard](./assets/dashboard.png) | System health green, spend gauges, recent alerts |
| ![graphs](./assets/graph_colours.png) | Red nodes = high risk, dashed pink edges = wash trades |
| ![progress](./assets/task_progress.gif) | Toast showing live Celery progress |

*(If images don’t render in GitHub, view them in `/assets` folder or in the web UI.)*

---

## 5. Full Fraud-Detection Walk-Through

| # | Action | UI / API | What happens |
|---|--------|----------|--------------|
| 1 | “Ingest Demo” button (or run `demo_ingest.py`) | Scripts tab | Loads sample data |
| 2 | **Health** → *Workers Healthy* | `/api/v1/health/workers` | Redis queue depths & active workers |
| 3 | **Run Detector** → Address `0xabc…` | `/anomaly/detect/single` | Celery task spawns → WebSocket updates |
| 4 | Toast “57 % – Detecting wash trading…” | Bottom right | Progress indicator via `/ws/tasks/{id}` |
| 5 | Alert pops “Wash Trading (HIGH)” | Live Alerts panel | Evidence bundle auto-created |
| 6 | Click **View Graph** | Graph tab | Nodes coloured by risk, wash-trade edges dashed pink |
| 7 | Click suspicious node | Modal shows risk score 0.88, 3 anomalies |
| 8 | **Investigate** → opens Investigation view | `/investigation/{id}` | Narrative + citations (Cypher code block) |
| 9 | View Citation | Chat / narrative | ```cypher MATCH (a)-[tx]->(b)…``` |
|10 | Export PNG | Graph controls | High-res image with legend |

---

## 6. Real-Time Features Under the Hood ⚙️

1. **ConnectionManager** streams Celery events → WebSocket `/ws/progress`.
2. Each background task joins a room `task_{id}`; UI auto-subscribes and updates a progress bar.
3. Anomaly alerts broadcast on `/ws/alerts`; red toast appears instantly.
4. Graph component polls `/anomaly/entities/.../results` for badge counts.

---

## 7. Diving Deeper – Investigations & Evidence

1. Every anomaly ⇒ `EvidenceBundle` with:
   - `AnomalyEvidence` (severity, score, details)
   - Raw graph elements
   - Auto-synthesised narrative
2. Bundles export to **HTML / PDF** in one click.
3. Chain-of-Custody: evidence `provenance_link` → `cypher:query:{id}` so an auditor can rerun queries.

---

## 8. Reproducing & Extending

| Want to… | Do this |
|----------|---------|
| Re-run demo from scratch | `docker compose down -v && docker compose up -d --build` |
| Load live Ethereum data | Adapt `scripts/new_provider_scaffold.py` |
| Add your own fraud pattern | Edit Redis key `anomaly:patterns` or call `/anomaly/strategies` |
| Build a stress-test scenario | (Next epic) See `TODO_PERSONAL_PROJECT_FOCUS.md` #1-1 |

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Celery workers UNHEALTHY` | `docker compose restart worker` |
| Graph empty | Ensure `demo_ingest.py` finished, check Neo4j logs |
| Budget exceeded message | Adjust limits in `providers/registry.yaml` |
| WebSocket 1006 | Check reverse-proxy WebSocket config |

---

## 10. Conclusion 🎯

Analyst Droid One **ingests blockchain data, detects complex fraud in real-time, and explains it with verifiable evidence** – all in minutes.  
Use this walkthrough to demo the power, then plug in your own chain, tune patterns, and ship production-grade fraud detection.

Happy hunting! 🕵️‍♀️🚀
