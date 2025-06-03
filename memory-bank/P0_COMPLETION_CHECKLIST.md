# P0 Completion Checklist  
_Last updated: **03 Jun 2025**_

Use this single page to execute and verify every **P0 blocker** before Phase-4 freeze.

---

## 0 ▪ Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔄 | In progress |
| ✅ | Completed / validated |

Tick boxes as you go.  

---

## 1 ▪ Documentation Files Creation

| Doc | Path | Status |
|-----|------|--------|
| MASTER_STATUS.md | `memory-bank/MASTER_STATUS.md` | ⬜ |
| TECHNICAL_ARCHITECTURE.md | `memory-bank/TECHNICAL_ARCHITECTURE.md` | ⬜ |
| CAPABILITIES_CATALOG.md | `memory-bank/CAPABILITIES_CATALOG.md` | ⬜ |

### Steps

1. Confirm files exist and render correctly in GitHub.  
2. Verify TOC links & section anchors.  
3. Remove _legacy_ markdown listed in `DOCUMENTATION_CLEANUP_PLAN.md`.

**Command**

```bash
gh repo clone illiterateailabs/analystt1
cd analystt1
ls memory-bank | grep -E 'MASTER_STATUS|TECHNICAL_ARCHITECTURE|CAPABILITIES_CATALOG'
```

**Expected Outcome**

* Three filenames echoed; opening in browser shows latest commit hash & date.  
* `git status` clean (no untracked stray docs).

---

## 2 ▪ Alembic Migration – `hitl_reviews`

| Item | Status |
|------|--------|
| Migration file `002_add_hitl_reviews_table.py` exists | ⬜ |
| Migrates successfully on local DB | ⬜ |
| Table visible in Postgres | ⬜ |

### Instructions

```bash
# 1. Generate & review migration (already created)
alembic upgrade head

# 2. Inspect schema
psql -h localhost -U analyst -d analyst_agent -c '\d+ hitl_reviews'
```

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | default uuid_generate_v4() |
| task_id | UUID | analysis task reference |
| review_type | varchar(50) | e.g. compliance |
| risk_level | varchar(20) | high/medium/low |
| status | varchar(20) | pending/approved/rejected |
| reviewer_id | UUID FK → users.id | nullable |
| comments | text | optional |
| review_started_at / completed_at | timestamptz | nullable |
| created_at / updated_at | timestamptz | triggers update |

**Validation Criteria**

* `alembic history` shows revision `002`.  
* Query returns _0 rows_ but correct columns.  
* Inserting dummy row succeeds.

---

## 3 ▪ Redis AOF Persistence for JWT Blacklist

| Check | Status |
|-------|--------|
| `appendonly yes` present in **all** compose files / Helm charts | ⬜ |
| Redis starts with `AOF enabled` log line | ⬜ |
| Blacklisted token survives container restart | ⬜ |

### Verification Steps

1. **Runtime check**

```bash
docker-compose up -d redis
docker exec analyst-agent-redis redis-cli -a $REDIS_PASSWORD CONFIG GET appendonly
# -> 1) "appendonly"  2) "yes"
```

2. **Durability test**

```bash
TOKEN="dummy.jwt"
redis-cli -a $REDIS_PASSWORD SETEX blacklist:$TOKEN 3600 1
docker restart analyst-agent-redis
redis-cli -a $REDIS_PASSWORD GET blacklist:$TOKEN  # expect "1"
```

**Expected Outcome**

* After restart the `GET` returns `1`.  
* No data loss message in Redis logs.

---

## 4 ▪ End-to-End Smoke Test

| Stage | Status |
|-------|--------|
| Template created via API | ⬜ |
| Crew executes & finishes | ⬜ |
| Results retrievable via API | ⬜ |
| Frontend `/analysis/{taskId}` renders graphs | ⬜ |

### Procedure

```bash
# 1. Start full stack
make dev   # or docker-compose up -d

# 2. Auth
TOKEN=$(curl -sX POST http://localhost:8000/api/v1/auth/login \
        -d '{"username":"alice","password":"secret"}' | jq -r .access_token)

# 3. Create template
curl -X POST http://localhost:8000/api/v1/templates \
     -H "Authorization: Bearer $TOKEN" \
     -F "yaml=@examples/aml_investigation.yaml"

# 4. Run analysis
TASK=$(curl -sX POST http://localhost:8000/api/v1/analysis \
        -H "Authorization: Bearer $TOKEN" \
        -d '{"template":"aml_investigation"}' | jq -r .task_id)

# 5. Poll status (max 5 min)
watch -n5 "curl -s -H 'Authorization: Bearer $TOKEN' \
      http://localhost:8000/api/v1/analysis/$TASK/status"

# 6. Fetch results
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/analysis/$TASK/results > results.json
```

**Validation Criteria**

* `/status` transitions → `running` → `done` with no error.  
* `results.json` contains `"risk_score"` and at least one `"visualizations"` entry.  
* Browser page `http://localhost:3000/analysis/$TASK` shows graphs & summary.

---

## 5 ▪ Completion Gate

Tick all boxes above → push checklist → tag release `v0.4.0-rc`.  

```bash
git add memory-bank/P0_COMPLETION_CHECKLIST.md
git commit -m "docs: ✅ P0 blockers completed"
git push origin droid/p0-blockers-complete
gh release create v0.4.0-rc --title "Phase-4 Release Candidate"
```

✅ **Project is now clear of P0 blockers and ready for P1 work.**
