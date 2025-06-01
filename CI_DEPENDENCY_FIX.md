# CI_DEPENDENCY_FIX.md  
_Last updated: 01 Jun 2025_

A pocket-guide for keeping our CI pipeline green when `pip`’s resolver melts down.

---

## 1 · The `httpx` Conflict (Real-World Case)

**Symptom**  
CI failed with:

```
ERROR: Cannot install … and httpx==0.27.0 because these package versions have conflicting dependencies.

fastapi 0.111.0   → httpx>=0.23.0
chromadb 0.5.23   → httpx>=0.27.0
google-genai 1.18 → httpx>=0.28.1,<1.0.0
```

**Root cause**  
We had **`httpx==0.27.0`** pinned.  
`google-genai 1.18.0` requires **≥ 0.28.1**.

**Fix**  
Bump to the smallest compatible version for *all* dependants:

```diff
- httpx==0.27.0
+ httpx==0.28.1
```

No other packages were affected:  
`fastapi` & `chromadb` still satisfy their lower bounds.

---

## 2 · General `pip` Resolution Tips

| Trick | Why it helps |
|-------|--------------|
| **Read the traceback** | `pip` lists every incompatible requirement. Scroll up! |
| **Constrain, don’t pin, transitive deps** | Pin only direct deps in *requirements.txt*.  Use *constraints.txt* for the rest. |
| **`pip install --dry-run -r requirements.txt`** | Shows the solver result without downloading wheels (PEP 668). |
| **`pipdeptree --reverse <pkg>`** | Visualise who drags an old version in. |
| **`pip check` after install** | Verifies runtime compatibility. |
| **Cache wheels** | Add `pip cache dir` to CI cache to avoid repeated downloads. |
| **Use `--no-deps` in Docker multi-stage** | First copy *requirements.txt* → install, then project code; keeps layer invalidations minimal. |

---

## 3 · Identifying the Offending Package

1. **Look for “The conflict is caused by:”** – pip spells it out.  
2. **Find the tightest specifier** – exact (`==`) or narrow range causes most pain.  
3. **Walk the tree**  
   ```
   pipdeptree --warn silence | less
   ```
   Find who pins the bad version.  
4. **Test the hypothesis** – in a venv:  
   ```bash
   pip install "troublesome-lib>=X.Y"
   pip install -r requirements.txt
   ```
   If it now resolves → you found the culprit.

---

## 4 · Version-Pinning Best Practices

| Rule | Example |
|------|---------|
| **Pin direct runtime deps** in `requirements.txt` | `fastapi==0.111.0` |
| **Use semver caps for libs you don’t control** | `sqlalchemy>=2.0,<2.1` |
| **Guide the solver with `constraints.txt`** | keep protobuf / grpcio in sync |
| **Group heavyweight extras behind opt-in** | `pip install .[xgboost]` |
| **Regularly update & audit** | `pip-review --local` monthly |
| **Fail fast in CI** – `pip install --no-cache-dir --constraint constraints.txt` |

---

### TL;DR Flowchart

1. ❌ _Pip fails_ → read conflicting list.  
2. 🔎 Identify **narrow pin** or **outdated lower bound**.  
3. 🛠  Bump or unpin in *requirements.txt* or *constraints.txt*.  
4. ✅ `pip install --dry-run -r requirements.txt` passes.  
5. 🎉 Commit & watch CI run in < 30 min again.

_End of file._
