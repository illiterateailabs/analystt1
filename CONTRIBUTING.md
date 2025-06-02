# Contributing to **analystt1**

First off — thanks for taking the time to contribute!  
This guide explains the tooling, workflow and quality standards we follow so that changes land quickly and safely.

---

## 1 · How We Work – High-Level Flow

```
Fork → Branch → Code & Test → Commit → Push → Pull Request → Review → Merge
```

1. **Every change happens through a Pull Request (PR)** – even docs.  
2. **CI must be green** – tests, lint and type-checks run automatically.  
3. **At least one approving review** is required before merge to `main`.

---

## 2 · Developer Environment

| Step | Command / Notes |
|------|-----------------|
| Clone repo | `git clone https://github.com/illiterateailabs/analystt1.git` |
| Create Python venv | `python -m venv venv && source venv/bin/activate` |
| Install backend deps | `pip install -r requirements.txt` |
| Install frontend deps | `cd frontend && npm install && cd ..` |
| Copy env file | `cp .env.example .env` – fill in API keys |
| Start services | `docker-compose up -d neo4j postgres redis` |
| Run app locally | `make dev` (spins up backend & frontend) |
| Pre-commit hooks | `pre-commit install` (runs lint, codespell, mdformat) |
| Tests | `pytest -q && npm run test` |

---

## 3 · Branching Strategy

| Branch prefix | Use for | CI target |
|---------------|---------|-----------|
| `feat/<name>` | New features / epics | Full matrix |
| `fix/<name>`  | Bug fixes | Affected jobs |
| `docs/<name>` | Docs only | Lint + link check |
| `hotfix/<name>` | Critical production issues | Fast-track |

Guidelines:

* Branch off **`main`**; keep them short-lived (< 1 week).  
* **Rebase** onto `main` before opening / updating a PR – avoid merge commits.  
* Delete the branch after the PR is merged.

---

## 4 · Commit Style (Conventional Commits)

```
<type>(scope): <subject>
<blank line>
<body>  # optional
```

| Type        | When to use                                  |
|-------------|----------------------------------------------|
| `feat`      | New feature or capability                    |
| `fix`       | Bug fix                                      |
| `docs`      | Documentation only                           |
| `refactor`  | Code change that neither fixes a bug nor adds a feature |
| `test`      | Adding or updating tests                     |
| `chore`     | Build tasks, CI, tooling                     |

Examples:

```
feat(neo4j): add APOC pathfinding support
fix(auth): correct JWT expiry validation
docs(roadmap): mark phase-3 as complete
```

---

## 5 · Pull Request Checklist

Before hitting **Create PR** confirm that:

- [ ] Branch rebased on latest `main`
- [ ] Follows commit style
- [ ] **All tests pass** (`pytest`, `npm run test`)
- [ ] `black`, `ruff`, `mypy`, `markdownlint` clean
- [ ] Added / updated tests for new behavior
- [ ] Updated docs (`README`, `memory-bank`, etc.)
- [ ] Linked related Issue (e.g. `Closes #123`)
- [ ] No secrets or personal data committed
- [ ] CI status **green**

PR description template:

```markdown
### What / Why
<concise description>

### How
<key implementation points>

### Validation
- [ ] unit tests
- [ ] manual QA steps

### Related Issues
Closes #
```

---

## 6 · Review & Merge

* Reviewers focus on **logic, security, performance, readability**.  
* Address comments with additional commits – avoid force-pushing after review starts unless asked.  
* Squash-and-merge is the default strategy; commit titles become release notes.  

---

## 7 · Code of Conduct

We follow the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).  
By participating, you agree to uphold a respectful and inclusive community.

---

## 8 · Need Help?

* Open a **Discussion** or **Issue** with the **question** label.  
* Ping us on the project Slack `#analystt1-dev`.  
Happy hacking! 🚀
