# Phase 1 Dependency Update – Implementation Plan  
_Last updated: 31 May 2025_

---

## 0. Branching & High-Level Workflow
```bash
# start from a clean main
git checkout main && git pull origin main

# feature branch for Phase 1
git checkout -b droid/phase1-security-updates
```

All steps below occur on this branch.  
Merge via PR/MR only after CI is green and manual smoke-tests pass.

---

## 1. Decision Points About Unused Dependencies

| Dependency | Current Usage | Decision |
|------------|---------------|----------|
| **sentry-sdk** | Config var exists but never initialised. | **KEEP & INTEGRATE** (recommended) _or_ remove. Default below assumes **keep**. |
| **web3** | Not imported anywhere. | **REMOVE** (free ~20 MB wheel & >120 sub-deps). |
| **aiohttp** (direct pin) | Only required transitively (EtherscanTool). | **REMOVE direct pin** – resolver will install compatible version (3.10.5) via deps. |

If you choose to remove sentry-sdk instead, omit steps 3.2 & 4.1 and delete it from `requirements.txt` / `constraints.txt`.

---

## 2. Package Version Changes

| File | Change |
|------|--------|
| `requirements.txt` | Bump: Pillow 10.1.0→10.4.0, aiohttp pin removed, sentry-sdk 2.16.0, web3 removed, Jinja2 3.1.4, python-multipart 0.0.12, slowapi 0.1.9 |
| `constraints.txt` | Remove web3-related constraints (if any). Ensure no pin on aiohttp. Add `sentry-sdk==2.16.0` only if other pins conflict. |

Quick patch:
```bash
sed -i 's|Pillow==10\.1\.0|Pillow==10.4.0|' requirements.txt
sed -i 's|Jinja2==3\.1\.2|Jinja2==3.1.4|' requirements.txt
sed -i 's|python-multipart==0\.0\.6|python-multipart==0.0.12|' requirements.txt
sed -i 's|slowapi==0\.1\.8|slowapi==0.1.9|' requirements.txt
sed -i '/^aiohttp==/d' requirements.txt
sed -i '/^web3==/d' requirements.txt
sed -i 's|sentry-sdk\[fastapi\]==1\.38\.0|sentry-sdk[fastapi]==2.16.0|' requirements.txt
```

---

## 3. Source-Code Updates

### 3.1 Remove dangling web3 imports  
None exist – no code change required.

### 3.2 Add Sentry initialisation

Create patch `backend/core/sentry.py`:
```python
import logging
from backend.config import settings

logger = logging.getLogger(__name__)

def init_sentry() -> None:
    """Initialise Sentry if DSN is present and environment != 'development'."""
    if not settings.SENTRY_DSN:
        logger.info("Sentry DSN not configured – skipping Sentry init")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.2,
            environment=settings.ENVIRONMENT,
            release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
        )
        logger.info("Sentry initialised")
    except ImportError:
        logger.warning("sentry-sdk not installed – cannot initialise Sentry")
```

Modify `backend/main.py` (top of file, after other imports):
```python
from backend.core.sentry import init_sentry
...
# before app instantiation
init_sentry()
```

No other runtime files require change.

---

## 4. Testing & Verification

### 4.1 Unit / Integration tests
```bash
# fresh virtual env
make clean-env            # optional helper
pip install -r requirements.txt -c constraints.txt
pytest -q                 # run entire suite
```

_Additional check if Sentry kept:_
```bash
pytest tests/test_sentry_integration.py  # new test triggers 500 and asserts HTTP 200 + Sentry stub mock
```

### 4.2 Manual smoke tests
1. Run API locally: `make dev` or `scripts/start.sh`.
2. Hit `GET /health` – expect 200.
3. Upload small image via `/api/v1/chat` → verify Pillow still handles.
4. Trigger rate-limit (>20 rapid calls) – expect 429 from slowapi.
5. Intentionally raise error (`GET /api/v1/analysis?bad=json`) – verify event appears in Sentry dashboard.

---

## 5. CI/CD Adjustments

| Area | Action |
|------|--------|
| `.gitlab-ci.yml` | No stage additions. Dependency resolver faster after web3 removal. |
| Cache key | Already includes `requirements.txt` & `constraints.txt`; automatic invalidation on edits. |
| Lint / mypy | Run `make pre-commit` locally before push to avoid CI failures from new lint rules introduced by updated deps (none expected). |
| Timeouts | Keep existing `timeout: 30m` guard. |

---

## 6. Rollback Plan

1. **Git**: simply revert PR:
   ```bash
   git checkout main
   git revert -m 1 <merge_commit_sha>
   git push origin main
   ```
2. **Prod containers**: rebuild from reverted main (`docker-compose pull && docker-compose up -d`).
3. **Dependencies cache**: Runner cache keys include file hashes; they invalidate automatically after revert.
4. **Database-/Sentry-specific**: new Sentry init is additive; nothing to roll back in DB schema.

---

## 7. Completion Checklist

- [ ] requirements / constraints updated & committed  
- [ ] `backend/core/sentry.py` added  
- [ ] `backend/main.py` imports `init_sentry()`  
- [ ] All tests green locally  
- [ ] CI pipeline passes on branch  
- [ ] PR reviewed & merged  
- [ ] Post-merge monitoring: logs, Sentry dashboard, Prometheus metrics  

_Once all boxes are ticked, Phase 1 is complete._
