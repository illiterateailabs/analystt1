# TESTING_GUIDE.md  
Analyst-Agent Testing Handbook

_Last updated: 31 May 2025_

---

## 1 · Running the Test Suite

| Environment | Command | Notes |
|-------------|---------|-------|
| Local venv  | `pytest -q` | Fastest. Requires Neo4j running (`docker compose up -d neo4j`). |
| Dev Docker  | `docker compose run --rm backend pytest -q` | Uses the same image CI builds. |
| Makefile    | `make test` | Lint → mypy → pytest with coverage. |
| GitHub CI   | Automatic on PRs (`.github/workflows/ci.yml`) | Matrix Python 3.9-3.11 + Neo4j service. |

### Extra flags

```
pytest -vv                 # verbose
pytest tests/test_tools.py # run subset
pytest --ff                # failed-first
pytest --cov=backend --cov=tests
```

---

## 2 · What Each Test File Covers

| File | Scope |
|------|-------|
| **tests/test_api_\*.py** | FastAPI endpoint contracts (chat, graph, analysis, crew). |
| **tests/test_auth.py** | JWT encode/decode, expiry, password hashing. |
| **tests/test_rbac.py** | `require_roles` decorator & dependency helper across 401 / 403 / 200 paths. |
| **tests/test_crew_factory.py** | Tool initialisation, agent/crew caching, error paths. |
| **tests/test_agent_configs.py** | YAML validation for all default agents; CrewFactory integration sanity. |
| **tests/test_tools.py** | Unit tests for each tool: TemplateEngineTool, PolicyDocsTool, CodeGenTool, GraphQueryTool, SandboxExecTool, PatternLibraryTool, Neo4jSchemaTool. |
| **tests/test_crew_integration.py** | RBAC + API + CrewFactory wiring; mocks external services. |
| **tests/test_full_integration.py** | End-to-end fraud-investigation workflow including HITL pause/resume and task status tracking. |
| **tests/test_integrations.py** | Thin smoke tests for Gemini, Neo4j, e2b wrappers (skipped if env vars missing). |
| **tests/test_end_to_end.py** | Legacy happy-path covering `/crew/run` → graph result → Prometheus metrics. |

> TIP: run `pytest -q tests/test_tools.py` first; they are fast and isolate most regressions.

---

## 3 · Coverage Targets

| Phase | Minimum Line Coverage | Focus Areas |
|-------|----------------------|-------------|
| Phase 2 MVP | **50 %** overall, **75 %** for `backend/agents/tools` and `backend/auth` | Security-critical code, tool logic |
| Phase 3 | 65 % | Graph algorithms, streaming |
| GA | 80 % | All public modules |

CI fails if coverage < target (`pytest --cov --cov-fail-under=50` in workflow).

---

## 4 · Common Test Failures & Fixes

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| `ImportError: No module named 'jinja2'` | New dev env missing optional deps | `pip install -r requirements.txt` (or add to constraints). |
| `Neo4j connection refused` in integration tests | Docker Neo4j not up / password mismatch | `docker compose up -d neo4j` and ensure `NEO4J_PASSWORD=analyst123`. |
| `401 Not authenticated` on API tests | Auth middleware not setting `request.state.user` | Verify `Authorization: Bearer …` header in test client. |
| `403 Access denied` unexpectedly | Incorrect role in token | Regenerate token with correct `role` claim (`create_access_token`). |
| `resolution-too-deep` during `pip install` | New dep added without pin | Add pin to `constraints.txt` & regenerate lock. |
| `jinja2.exceptions.TemplateNotFound` in TemplateEngineTool tests | Template dir path changed | Call `TemplateEngineTool(template_dir=Path("..."))` in test or update default path. |
| Coverage drop below gate | New code lacks tests | Add tests or mark experimental code with `# pragma: no cover`. |

---

## 5 · Adding New Tests

1. **Name** the file `tests/test_<module>.py`.  
2. **Isolate** external services with `unittest.mock` or fixtures in `tests/conftest.py`.  
3. **Prefer async** functions + `pytest.mark.asyncio` for code that awaits.  
4. **Assert JSON contracts** (`json.loads(resp)` then field checks).  
5. **Measure coverage**: run `pytest --cov`. Ensure coverage delta ≤ ‑2 %.  
6. **CI compatibility**: tests must run in ~60 s on GitHub’s default runners. Mock heavy tasks (Gemini, Neo4j algorithm loops).  
7. **Mark slow/optional** with `@pytest.mark.skipif` when env vars missing (e.g., real API keys).  
8. **Document** non-obvious fixtures at top of file; keep each test idempotent and network-free.

Template:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_my_tool_basic():
    from backend.agents.tools.my_tool import MyTool
    tool = MyTool()

    with patch.object(tool, "external_call", AsyncMock(return_value="ok")):
        result = await tool._arun(param="value")
        result_json = json.loads(result)
        assert result_json["success"]
```

---

### Where to Look for Examples

- `tests/test_tools.py` shows **unit** style with extensive mocking.
- `tests/test_crew_integration.py` shows **integration** style hitting FastAPI in-memory.
- `tests/test_full_integration.py` demonstrates **end-to-end** patterns and HITL workflow simulation.

---

## 6 · Helpful Pytest Commands Cheat-Sheet

```
pytest -k "rbac and not slow"          # keyword filter
pytest -m "not integration"            # marker filter
pytest --maxfail=2                     # stop after 2 failures
pytest --durations=10                  # slowest 10 tests
pytest --cov-report=term-missing:skip-covered
```

Keep tests fast, deterministic, and clearly named—future-you will thank you.  
