# DEPENDENCY_CLEANUP.md  
_Last updated: 01 Jun 2025_

Speeding-up CI and reducing operational complexity starts with trimming heavyweight or unused Python packages. This document lists **what can go**, why, the functional impact, lighter alternatives, expected CI savings, and a safe removal playbook.

---

## 1 · Removable Dependencies

| Package | Size* | Why It Can Go | Current Usage Trace |
|---------|-------|--------------|---------------------|
| **spacy** | 675 MB | NLP now handled by Gemini 2.5; no local models invoked in code or tests | Only referenced in legacy docs |
| **transformers** | 524 MB | Same as above; no Hugging Face models used | Not imported anywhere |
| **torch / tensorflow / keras** (transitive) | 1-2 GB | Pulled in by `transformers`; none required after removal | Not imported |
| **py2neo** | 90 MB | Deprecated; replaced by official neo4j-python-driver (`neo4j`) | Already removed in code – stay out of `requirements.txt` |
| **crewai-tools** | 60 MB | Conflicts with CrewAI ≥0.119; custom tools cover required functionality | Only in old docs |
| **statsmodels** | 43 MB | Heavy; only needed for advanced ARIMA/ETS not on roadmap | Imported nowhere |
| **xgboost** | 38 MB | Large wheel; scikit-learn suffices for Phase-3 ML | Not imported; ML PoC uses RandomForest |
| **yfinance** | 22 MB | TradFi market data fetcher; not referenced in code | Placeholder for future |
| **alpha-vantage** | 19 MB | Same as above | Unused |
| **chromadb** (server) | 70 MB | Only pulled for unused `crewai-tools` vector DB; we rely on Redis + Neo4j | Not directly imported |

_*Compressed wheel sizes on PyPI_

Total potential wheel download saved per CI run: **≈ 2.5 GB**

---

## 2 · Impact Analysis

| Functionality Area | Impact if Removed | Mitigation / Existing Tool |
|--------------------|-------------------|----------------------------|
| NLP tasks | None – all prompts handled by Gemini API | Keep Gemini integration |
| ML models (deep learning) | None – current fraud ML uses scikit-learn | Use scikit-learn, xgboost optional via Docker arg if ever needed |
| Time-series forecasting | Minimal; ADTK & pandas cover anomaly detection | For ARIMA later, lightweight `pmdarima` can be added on-demand |
| Financial market data | `yfinance/alpha-vantage` unused; fetch via GraphQL or REST when needed | Add back selectively per feature branch |
| Vector DB | `chromadb` unnecessary; RedisSearch or Neo4j native vectors planned | Keep Redis |

---

## 3 · Alternative Approaches

1. **Gemini for NLP**  
   – Use function-calling + JSON schemas instead of spaCy pipelines.  
2. **Scikit-learn + ADTK** for classical ML & anomaly detection  
   – Cover 90 % of fraud-detection needs without heavyweight DL stacks.  
3. **GraphQLQueryTool** for external crypto data  
   – Replaces `yfinance` / `alpha-vantage` for on-chain & market feeds.  
4. **Redis Search** for lightweight vector similarity  
   – Avoids `chromadb`, fits existing Redis instance.

---

## 4 · Estimated CI Time Savings

| Stage | Before | After Cleanup | Δ |
|-------|--------|---------------|---|
| `pip install` (cold) | ~35-45 min (wheel downloads & builds) | ~7-10 min | **-30 min** |
| Dependency resolution | Up to 25 min backtracking | <5 min | **-20 min** |
| Docker build size | ~3.2 GB | ~900 MB | **-2.3 GB** |
| Overall job runtime | 55-65 min → frequent 60 min hard-timeouts | ~25-30 min | **> 50 % faster** |

---

## 5 · Step-by-Step Removal Plan

| Step | Action | Owner | Time |
|------|--------|-------|------|
| 1 | **Branch** `chore/deps-cleanup` off `main` | backend-dev | 5 m |
| 2 | Edit `requirements.txt` & `constraints.txt` – remove listed packages | backend-dev | 10 m |
| 3 | Search codebase (`grep -R`) for removed imports; delete or replace | backend-dev | 20 m |
| 4 | Update **Dockerfile** to skip OS deps for spaCy / torch | dev-ops | 10 m |
| 5 | Run `pipdeptree` & `pip-audit` – verify no hidden deps pull them back | backend-dev | 10 m |
| 6 | Push branch → **CI Dry-Run** (`--no-cache --dry-run`) | dev-ops | 30 m |
| 7 | Fix any failing imports/tests (expect none) | backend-dev | 15 m |
| 8 | Open PR “deps: slim build by removing unused heavy libs” | backend-dev | 5 m |
| 9 | Merge after green CI | maintainer | — |
|10 | Monitor next three CI runs & prod images for regressions | dev-ops | ongoing |

Total effort: **≈ 2 hours**.

---

### Roll-back Plan
If any feature unexpectedly needs a removed library:
1. Re-add specific dep in a scoped feature branch (`feat/xgboost-poC`)  
2. Gate through optional extras (`pip install .\[xgboost\]`)  
3. Ensure tests & CI time budget remain acceptable.

---

### Next Optimisations
- **Switch pip → `uv`** (after cleanup) for 30-40 % faster installs  
- Cache **`~/.cache/pip`** and **`/root/.cache/pip`** between CI jobs  
- Use **multi-stage Docker build** with `pip install --no-deps` for pure runtime image.

---

*Removing unused heavyweight libraries is the single biggest lever to keep CI under 30 minutes and prevent future timeout headaches.*  
