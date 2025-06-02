# CI Dependency Optimisation – Root-Cause & Remediation

Pipeline stalls **> 60 min** on the *Install dependencies* step for every job.  
This note documents why and proposes incremental fixes that **do not** affect production runtime.

---

## 1  Why the install step is slow

| Factor | Evidence in repo | Impact |
|--------|-----------------|--------|
| **Huge dependency graph** (≈ 140 pkgs) inc. full ML/​SciPy stack | `requirements.txt` pins `numpy`, `pandas`, `scipy`, `scikit-learn` for **every** matrix job | 8 000 + wheels download, long resolver time |
| **Google Vertex AI libs** (`google-cloud-aiplatform`) pull 60-70 extra packages | Observed in `pip install` logs (multiple backtracking rounds) | Adds minutes of resolver work |
| `pip` SAT solver re-evaluates graph *three* times (one per Python version) | No wheel cache reused across jobs | Redundant downloads |
| Tools (`ruff`, `black`, `mypy`) installed *alongside* heavy deps even for lint jobs | Single `pip install -r requirements.txt` used in **all** jobs | ~600 MB wheels for tasks that only need linters |
| No lock file ➜ resolver backtracking | `--constraint constraints.txt` helps but still wide ranges | Extra 10-15 min |

---

## 2  Quick wins (< 30 min change)

| Step | Snippet / Action | Benefit |
|------|------------------|---------|
| **Cache wheel directory across jobs** | ```yaml\n- name: Restore pip cache\n  uses: actions/cache@v4\n  with:\n    path: ~/.cache/pip\n    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt','constraints.txt') }}\n``` | Avoids redownloading wheels (-5-10 min/job) |
| **Prefer binary wheels** | Add `--prefer-binary` to install commands | Skips source builds |
| **Slim deps for lint/type jobs** | Create `requirements-ci-min.txt` containing only:\n```\nblack==23.11.0\nisort==5.12.0\nruff==0.1.5\nmypy==1.7.1\n```\nand use it in *lint* & *type-check* jobs | Removes heavy ML stack (-15 min for those jobs) |

Combined quick wins usually cut **≈ 20 min per job**.

---

## 3  Medium fixes (1-2 h)

| Action | How | Expected gain |
|--------|-----|---------------|
| **Freeze full lock** | Generate `requirements-lock.txt` using `pip-compile` (pip-tools). In CI:<br>`pip install -r requirements-lock.txt --no-deps --prefer-binary` | Resolver time → ~0, install ≈ 3-5 min |
| **Move heavy ML stack to test job only** | Keep `requirements-ci-min.txt` for lint/type; install full lockfile only in `test` matrix | Saves 500 MB downloads on other jobs |
| **Matrix collapse on push** | Run tests on 3.11 for PRs, full matrix nightly | –2 duplicate installs per commit |
| **(Optional) Wheel mirror** | Host wheels on GHCR or internal artifact store; pass custom `--index-url` | Speed boost for large deps |

---

## 4  Long term (≈ 1 day) – Pre-built Docker image

1. Weekly workflow builds `analyst-runtime` image with *all* dependencies and caches wheels.  
2. CI jobs run inside that container:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    container: ghcr.io/<org>/analyst-runtime:latest
```

Runtime per job drops to ~5 min; no per-run resolver work.

---

## 5  Proposed **minimal PR** (low risk)

1. Add pip cache step to every job.  
2. Commit `requirements-ci-min.txt`.  
3. Update workflow YAML:  
   * lint & type-check → `pip install -r requirements-ci-min.txt --prefer-binary`  
   * keep full install only for `test` + `docker-build`.

No application code changes.

---

## 6  Next steps

| Owner | Task | ETA |
|-------|------|-----|
| CI maintainer | Implement Quick wins & minimal dep split | **This week** |
| Dev-Ops | Evaluate Docker image approach | Later |

> With caching **and** dependency split we typically bring the install phase **under 6 min** per job, eliminating 1 h timeouts.
