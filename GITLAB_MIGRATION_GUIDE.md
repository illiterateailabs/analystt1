# GitLab Migration Guide  
Migrating `analyst-agent-illiterateai` from **GitHub** ➜ **GitLab**

_Last updated : 31 May 2025_

---

## 0 · Why Move?  
* **CI minutes exhausted on GitHub** – free tier is limited (2 000 min/mo).  
* **GitLab 60-day Ultimate trial** gives **50 000 CI minutes**, advanced security scans, and self-hosted runners.  
* Consolidates issues, CI/CD, and container registry in one place.

---

## 1 · Push Code to GitLab

### 1.1 Create the target project  
1. Log in to GitLab → **➕ New project** → _Import project → “Repository by URL”_.  
2. Enter  
   ```
   https://github.com/illiterateailabs/analyst-agent-illiterateai.git
   ```  
   Choose **Private** visibility, click **Create project**.

> Alternatively, create an empty repo then push manually:

### 1.2 Manual mirror push  
```bash
# Clone existing repo with all branches & tags
git clone --mirror git@github.com:illiterateailabs/analyst-agent-illiterateai.git
cd analyst-agent-illiterateai.git

# Add GitLab remote (replace namespace as needed)
git remote add gitlab git@gitlab.com:illiterateailabs-group/illiterateailabs-project.git

# Push everything
git push gitlab --mirror
```

*☑️  All branches (e.g. `droid/complete-implementation-gaps`, `droid/ml-fraud-detection`) and tags are now on GitLab.*

---

## 2 · Set Up GitLab CI/CD

### 2.1 Add `.gitlab-ci.yml`  
The file is already committed on `main`. GitLab auto-detects and starts pipelines.

### 2.2 Configure CI/CD variables (Settings ➜ CI/CD ➜ Variables)

| Variable | Purpose | Scope |
|----------|---------|-------|
| `GOOGLE_API_KEY` | Gemini 2.5 models | **Masked, Protected** |
| `E2B_API_KEY` | e2b sandbox | Masked |
| `E2B_TEMPLATE_ID` | Sandbox template (e.g. `python3-default`) | Optional |
| `JWT_SECRET_KEY` | Token signing key | Masked |
| `DOCKER_AUTH_TOKEN` | (If pushing to GitLab registry) | Protected |
| `PIP_CACHE_DIR` | Cached by default | — |

> In Ultimate trial the **environments** feature lets you scope variables per branch.

### 2.3 Optional Runner setup  
* Using shared runners consumes trial minutes.  
* Add **self-hosted runner** (Settings ➜ CI/CD ➜ Runners) to avoid limits.

---

## 3 · Repository Settings Checklist

| Location | Action | Why |
|----------|--------|-----|
| Settings ➜ General ➜ Visibility | Keep **Private** | Secrets & financial data |
| Settings ➜ Repository ➜ Mirroring | Set **Push** mirror back to GitHub (optional) | Dual-home code while trial lasts |
| Settings ➜ Repository ➜ Default branch | Switch to `main` | Matches GitHub |
| Settings ➜ Integrations ➜ Slack/Mattermost | Add webhooks | Pipeline notifications |
| Packages ➜ Container Registry | Enable | Store backend/front-end images |

---

## 4 · Update References

1. **Badges** in `README.md`  
   ```md
   ![GitLab pipeline](https://gitlab.com/<group>/<project>/badges/main/pipeline.svg)
   ```
2. **CI configuration mentions** (`.github/workflows` → remove or mark legacy).  
3. **Links** inside docs/tutorials → point to GitLab project or raw file URLs.  
4. **Secrets docs** (`techContext.md`, `.env.example`) – note GitLab variable UI.  
5. **Issue/PR templates** → convert to **Issue templates** & **Merge Request templates**.

---

## 5 · GitHub Actions vs GitLab CI

| Feature | GitHub Actions | GitLab CI |
|---------|----------------|-----------|
| Config file | `.github/workflows/*.yml` | `.gitlab-ci.yml` (single entrypoint) |
| Matrix builds | `strategy.matrix` | `parallel: matrix:` |
| Services (DBs) | `services:` | Same keyword |
| Secrets | _Actions → Repository secrets_ | **CI/CD variables** (masked, scoped) |
| Artifacts | `actions/upload-artifact` | Built-in `artifacts:` |
| Manual jobs | `workflow_dispatch` | `when: manual` |
| Environments | Separate concept | Built-in with `environment:` |
| Minutes | 2 000 / mo (public) | 50 000 in Ultimate trial |

Conversion already handled in the provided `.gitlab-ci.yml`:

| GitHub Job | GitLab Stage | Notes |
|------------|--------------|-------|
| `lint` | `lint` | Python 3.11 image |
| `type-check` | `type-check` | mypy |
| `test` (matrix) | `test` (parallel matrix) | Neo4j service preserved |
| `docker-build` | `docker-build` | Docker-in-Docker |

---

## 6 · Maximizing the 60-Day Trial

| Tip | Benefit |
|-----|---------|
| **Enable Ultimate features** (Security scans, Code Quality) | Adds SAST/DAST reports for free |
| Use **Auto-DevOps** template on a throw-away branch | Free performance testing |
| Spin up **Review Apps** for each MR | Live front-end previews |
| Add **Push mirroring** back to GitHub | Maintains presence for external collaborators |
| Register a small **self-hosted runner** | Zero minute consumption for heavy tests |
| Track **CI minute usage** in _Settings ➜ Usage Quotas_ | Avoid surprises |
| Plan a **post-trial strategy** (self-hosted GitLab CE or return to GH) | Continual builds |

---

## 7 · Post-Migration Smoke Test

```bash
# Clone from GitLab
git clone git@gitlab.com:illiterateailabs-group/illiterateailabs-project.git
cd illiterateailabs-project

# Trigger pipeline manually (if needed)
curl --request POST \
     --form token=$CI_TRIGGER_TOKEN \
     --form ref=main \
     https://gitlab.com/api/v4/projects/<id>/trigger/pipeline
```

Expect stages: **lint → type-check → test (3×) → docker-build** all green.  
Neo4j service should pass health checks; coverage report uploads.

---

## 8 · Clean-Up

- Disable GitHub Actions in `Settings ➜ Actions` to stop minute charges.  
- Archive GitHub repo or leave read-only.  
- Delete obsolete GitHub secrets.

---

👋  You are now ready to run unlimited pipelines (for 60 days) on GitLab. Enjoy the extra horsepower!  
For questions, open an **Issue** in the GitLab project and mention `@illiterate-ai`.  
