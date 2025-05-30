# Tests Directory Guide

Welcome to **`/tests`** – the automated quality-gate for the Analyst-Agent backend.  
This short guide covers **how the test suite is organised, run, and extended**.

---

## 1. Structure & Organisation

```
tests/
├── README.md         ← you are here
├── test_auth.py      ← authentication & RBAC
├── test_integrations.py
├── test_api_analysis.py
├── test_api_chat.py
├── test_api_crew.py
├── test_api_graph.py
└── test_crew_factory.py
```

| Layer                        | File(s)                               | Purpose |
| ---------------------------- | ------------------------------------- | ------- |
| **Unit tests**               | `test_auth.py`, parts of `test_crew_factory.py` | Pure Python logic, no I/O |
| **Integration tests**        | `test_integrations.py`, API tests     | External services mocked with `unittest.mock` |
| **API (FastAPI) tests**      | `test_api_*`                          | Route happy-paths and edge cases using `fastapi.testclient` |
| **Behaviour / CrewAI tests** | `test_crew_factory.py`                | Crew creation, task wiring, error handling |

All tests are **async-aware** (`pytest-asyncio`) and grouped by functionality rather than module path to keep navigation intuitive.

---

## 2. Running the Test Suite

### Local (recommended)

```bash
# Create & activate venv
make deps-backend          # or pip install -r requirements.txt
# Run everything
make test                  # wrapper for `pytest`
# Or manually
pytest -n auto             # parallel run if pytest-xdist installed
```

### With coverage

```bash
pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

### Inside Docker

```bash
docker compose run --rm backend pytest
```

### Make targets

| Target        | Command                                   |
| ------------- | ----------------------------------------- |
| `make test`   | `pytest` with default opts                |
| `make test-backend` | `pytest backend` (subset)           |
| `make clean`  | prune caches (`__pycache__`, coverage, etc.) |

---

## 3. Coverage Targets

| Stage                 | % Covered | Notes |
| --------------------- | --------- | ----- |
| **Quick-Win (current)** | ≥ **30 %** | Ensures baseline confidence |
| Phase 2               | ≥ 50 %    | Core logic, error paths     |
| Beta / Release        | **≥ 70 %** | Critical & happy-path parity |

⚠️ Coverage is not a silver bullet – focus on **meaningful assertions** and failure scenarios.

---

## 4. Writing New Tests – Guidelines

1. **Follow the AAA pattern**  
   *Arrange – Act – Assert* for readability.
2. **Prefer async tests** (`pytest.mark.asyncio`) when the code is async.
3. **Isolate side-effects**  
   - Use `unittest.mock.patch` or `patch.object` to stub network, DB, or AI calls.  
   - Avoid real API keys – the CI replaces them with dummy values.
4. **Name clearly**  
   `test_<unit>_<scenario>()`, e.g. `test_gemini_generate_text_error`.
5. **One behaviour per test** – granular failures speed diagnosis.
6. **Avoid slow sleeps / waits** – mock timeouts instead.
7. **Use fixtures** (see below) to share setup, never global state.

---

## 5. Mocking Patterns

| Pattern                              | Example |
| ------------------------------------ | ------- |
| **`patch("package.module.Class")`**  | Replace GeminiClient with dummy |
| **`AsyncMock`** for coroutines       | `generate_text = AsyncMock()` |
| **Property mocks** (`PropertyMock`)  | Stubbing `.model` attr |
| **Context fixtures** for FastAPI app | Inject mocked clients into `app.state` |
| **Side-effects & return values**     | `mock.run_query.side_effect = Exception("DB Err")` |

These keep tests **fast, deterministic, and offline**.

---

## 6. Shared Fixtures

| Fixture name          | Scope | Provides |
| --------------------- | ----- | -------- |
| `access_token`        | func  | Valid JWT for auth-required routes |
| `auth_headers`        | func  | `{"Authorization": "Bearer …"}` dict |
| `mock_gemini_client`  | func  | Stubbed Gemini SDK (text / image / cypher) |
| `mock_neo4j_client`   | func  | Async Neo4j driver replacement |
| `mock_e2b_client`     | func  | Stubbed sandbox execution |
| `mock_crew_factory`   | func  | Lightweight CrewFactory surrogate |
| `test_user_data`      | func  | Canonical user payload |

Use them via dependency injection or direct parameterisation.

---

## 7. CI / CD Integration

* **GitHub Actions** – see `.github/workflows/ci.yml`  
  - Matrix tests (Py 3.9–3.11)  
  - Lint (`ruff`), type-check (`mypy`), unit tests (`pytest --cov`)  
  - Coverage uploaded to Codecov  
  - Docker images built after green tests

* **Fail-fast**: any test failure or coverage regression blocks the PR.

* **Pre-commit hooks**: run `pre-commit install` to execute lint & tests pre-push.

---

## 8. Contributing

1. Create feature branch: `git checkout -b tests/<feature>`
2. Add / update tests in `tests/` mirroring new code paths.
3. Ensure `pytest -q` passes **locally**.
4. Commit with meaningful message: `test: add edge case for XYZ`.
5. Push & open PR – GitHub Actions will validate automatically.

Enjoy testing – every assertion makes the analyst-agent smarter 🛡️
