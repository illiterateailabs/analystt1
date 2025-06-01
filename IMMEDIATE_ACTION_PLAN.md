# IMMEDIATE_ACTION_PLAN.md  
_Start fixing P0 & P1 gaps • Updated 01 Jun 2025_

Follow this playbook **top-down**. Each step is designed to be executed right now on a fresh checkout of `main`.

---

## 0 · Prerequisites

```bash
git switch -c fix/P0-quick-wins
poetry install            # or `pip install -r requirements.txt`
cp .env.example .env      # fill in local secrets
docker compose up -d neo4j redis postgres
```

Confirm health:

```bash
curl http://localhost:8000/health
```

---

## 1 · Quick Wins (≤ 3 h total)

### 1.1 Add RBAC Guard to Sensitive Routes 🟢

**File:** `backend/api/v1/crew.py`

```diff
-from fastapi import APIRouter, Depends
+from fastapi import APIRouter, Depends
+from backend.auth.dependencies import require_roles
 router = APIRouter()

 @router.post("/run")
-async def run_crew(req: CrewRunRequest):
+async def run_crew(
+    req: CrewRunRequest,
+    _=Depends(require_roles(["analyst", "admin"]))  # NEW
+):
     ...
```

Repeat for every `/analysis/*` route:

```python
# backend/api/v1/analysis.py
router = APIRouter()

@router.post("/enrich")
async def enrich_alert(
    req: AlertEnrichRequest,
    _ = Depends(require_roles(["analyst", "admin"]))
):
    ...
```

**Smoke test**

```bash
pytest tests/test_rbac.py -k "crew_run"
```

Expect **401/403** for missing/insufficient roles.

---

### 1.2 Generate Alembic Migration for `users` Table 🟢

```bash
alembic revision --autogenerate -m "users table"   # creates alembic/versions/<hash>_users_table.py
alembic upgrade head                               # apply locally
pytest tests/test_auth.py::test_register_login     # should pass
```

**CI hook**  
Add to `.github/workflows/ci.yml` before test step:

```yaml
- name: Run migrations
  run: alembic upgrade head
```

---

### 1.3 Add Red-Green Test for CodeGenTool Result Path 🟢

Create `tests/test_codegen_integration.py`

```python
import pytest
from backend.agents.tools.code_gen_tool import CodeGenTool

@pytest.mark.asyncio
async def test_codegen_result_merges_into_context():
    tool = CodeGenTool()
    # monkeypatch sandbox exec to return deterministic result
    async def fake_exec(*_, **__):
        return {"result": 42}
    tool._execute_in_sandbox = fake_exec

    res = await tool.run({"question": "2+40"})
    assert res["result"] == 42, "Result not propagated"
```

Run:

```bash
pytest tests/test_codegen_integration.py  # Expect FAILURE right now
```

Leave it failing ‑ this sets the stage for P0-1 below.

Commit quick-wins:

```bash
git add backend tests alembic
git commit -m "P0 quick wins: RBAC guard + users migration + failing CodeGen test"
```

Push branch and open PR to keep CI green on quick-wins.

---

## 2 · Critical P0 Tasks

> Switch to a new branch **after** merging quick-wins:  
> `git switch -c fix/P0-1-codegen-results`

### 2.1 Integrate CodeGenTool Results

1. **Define contract**

```python
# backend/agents/tools/code_gen_tool.py
@dataclass
class CodeGenResult:
    stdout: str
    artifacts: dict[str, str]  # name: base64
    result: Any
```

2. **Return merged context**

```python
crew_context["codegen"] = result.dict()
```

3. **Update report_writer prompt** to include `{{codegen.result}}`.

4. **Make failing test pass**

```bash
pytest tests/test_codegen_integration.py  # should go green
```

### 2.2 Enforce RBAC Tests

Add in `tests/test_rbac.py`

```python
def test_crew_run_forbidden(client, jwt_analyst, jwt_viewer):
    res = client.post("/api/v1/crew/run", headers={"Authorization": f"Bearer {jwt_viewer}"})
    assert res.status_code == 403
```

### 2.3 Wire Alembic in Docker

*`docker-compose.yml`*

```yaml
command:
  bash -c "alembic upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000"
```

---

## 3 · High-Priority P1 Starters

(continue in dedicated branches after P0 merged)

| Task | Kick-off Command |
|------|------------------|
| Redis blacklist | `git switch -c feat/P1-redis-blacklist` |
| PolicyDocsTool vectors | `git switch -c feat/P1-policydocs-vector` |
| Front-end Analysis View | `git switch -c feat/P1-analysis-view` |
| Raise coverage | integrate with above tasks |

---

## 4 · Verification Checklist

```bash
# Backend
pytest -q                       # green
coverage run -m pytest
coverage report                 # ≥ 51 % now, 55 % goal P1

# Local app
docker compose up -d
curl -X POST http://localhost:8000/api/v1/crew/run \
     -H "Authorization: Bearer $JWT_ANALYST" \
     -d '{"crew_name":"fraud_investigation","input":"trace funds"}'
```

Ensure response contains **risk_score** and any **codegen** output.

---

## 5 · Commit & PR Policy

1. **One PR per P0/P1 task**  
2. Must pass: Ruff, mypy --strict, pytest, coverage gate  
3. Update **Memory Bank** (`progress.md`, `activeContext.md`) after merge

---

_Execute steps 1.1–1.3 today; deploy migration & RBAC fix to staging. Start P0-1 tomorrow morning._  
**Time to first impact: < 3 hours.**  
