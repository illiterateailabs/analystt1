# CI Dependency Fix — `resolution-too-deep` Error

_Last updated: 31 May 2025_

---

## 1. Problem Description

* **Symptom in CI**  
  GitHub Actions failed during the Docker-build stage:

  ```
  error: resolution-too-deep
  × Dependency resolution exceeded maximum depth
  ╰─> Pip cannot resolve the current dependencies as the dependency graph is too complex for pip to solve efficiently.
  ```

* **Impact**  
  • Docker images were not produced → pipeline red ❌  
  • All jobs depending on the backend image (integration tests, deploy previews) were blocked.

---

## 2. Root-Cause Analysis

| Item | Details |
|------|---------|
| Trigger | `pip install -r requirements.txt` inside Dockerfile |
| Offending package | **`confection`** (transitive dependency of **spaCy 3.7.2**) |
| Why pip failed | New resolver back-tracked across **hundreds** of versions pulled in by spaCy / transformers, exceeded default back-tracking depth (200 k) and aborted. |
| Hidden factor | requirements.txt had _no explicit pins_ for transitive libs. Each CI run could attempt a different (and possibly conflicting) set of versions released upstream ↔ non-deterministic builds. |

---

## 3. Solution Implemented

### 3.1 Dependency Pinning

1. **Downgraded spaCy**  
   `spacy==3.7.2` ⟶ `spacy==3.6.1` (latest version _before_ the confection conflict emerged).

2. **Created `constraints.txt`**  
   Explicit upper / lower bounds for problematic transitive deps:

   ```
   confection>=0.1.3,<1.0.0
   catalogue>=2.0.6,<2.1.0
   thinc>=8.1.0,<8.3.0
   ...
   ```

3. **Updated requirements.txt**

   ```
   spacy==3.6.1
   transformers==4.35.2
   confection>=0.1.3,<1.0.0  # explicit
   ```

### 3.2 Build & CI Pipeline Changes

* **Dockerfile**
  * Two-phase install:  
    `pip install --no-deps …` (metadata only) → full install with `--constraint constraints.txt`.
  * Added resolver tuning (`backtrack-limit=5000`, retries, timeout).

* **GitHub Actions**
  * All steps (`lint`, `type-check`, `tests`, `docker-build`) install with  
    `pip install --constraint constraints.txt -r requirements.txt`.

Result: **CI pipeline is green** and Docker images build reproducibly.

---

## 4. Alternative Approach — UV Package Manager

`uv` (https://astral.sh/uv) is a drop-in replacement for pip that features a Rust-based resolver and dramatically faster, deterministic installs.

* **Prototype**: `Dockerfile.uv`
  1. Installs uv once (`curl | sh`).
  2. Uses `uv pip install` to resolve & install all deps.
* **Benefits**
  * Handles large dependency graphs without resolution-too-deep errors.
  * 2-5× faster cold builds.
* **How to test**

  ```bash
  docker build -f Dockerfile.uv -t analyst-agent-backend:uv .
  ```

If future dependency storms appear, switch the main build to UV by renaming the file or adjusting CI.

---

## 5. Preventive Measures

| Measure | Action |
|---------|--------|
| **Lock transitive dependencies** | Keep `constraints.txt` updated; regenerate quarterly with `pip-compile --generate-hashes`. |
| **Deterministic builds** | Always pass `--constraint constraints.txt` in local & CI installs. |
| **Monitor upstream releases** | Dependabot alerts plus weekly `pip index versions package` check for spaCy, confection, transformers. |
| **CI gate** | Fail PR if Docker build or dependency diff changes SHA of lockfile. |
| **Consider UV** | Evaluate UV in staging for one month; if stable, promote to default installer. |

---

### TL;DR for New Contributors

1. Run `make dev` → docker-compose uses the pinned images.  
2. Add new libs to `requirements.txt`, then run `pip-compile --upgrade --output-file constraints.txt requirements.txt`.  
3. Verify `github/workflows/ci.yml` still passes locally with `act` or push a branch.
