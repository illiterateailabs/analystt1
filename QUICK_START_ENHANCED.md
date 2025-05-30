# 🚀 Analyst Augmentation Agent – Quick Start (Enhanced Edition)

Welcome to the enhanced version of the Analyst Augmentation Agent.  
This guide walks you through:

1. Local setup & environment configuration  
2. Running the new CI/CD pipeline  
3. Launching services (backend, frontend, Neo4j)  
4. Executing the test-suite & linting  
5. Using the new JWT authentication endpoints  

> **Prerequisites**  
> • Python 3.9 – 3.11 • Node 18+ • Docker & Docker-Compose • GitHub account (for CI)  

---

## 1 ▸ Clone & bootstrap

```bash
git clone https://github.com/ms81labs/analyst-augmentation-agent.git
cd analyst-augmentation-agent
./scripts/setup.sh        # creates venv, installs deps, copies .env.example → .env
```

Open `.env` and set **at minimum**:

| Key | Description |
|-----|-------------|
| `SECRET_KEY` | long random string for JWT signing |
| `GOOGLE_API_KEY` | Gemini API key |
| `E2B_API_KEY` | e2b.dev key |

Other values already match `docker-compose.yml`.

---

## 2 ▸ Run Services Locally

```bash
# Start databases & optional Jupyter
docker-compose up -d neo4j postgres redis

# Backend (auto-reload)
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend
npm run dev
```

Now browse:  
• Frontend → http://localhost:3000  
• API docs → http://localhost:8000/docs  
• Neo4j Browser → http://localhost:7474 (neo4j / analyst123)

---

## 3 ▸ CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs **on every PR & push**:

| Job | What it does |
|-----|--------------|
| `python-lint-test` | installs deps, `flake8`, `black`, `isort`, `mypy`, `pytest` (+ coverage) |
| `frontend-lint-build` | `npm ci`, ESLint, type-check, production build |
| `security-scan` | `pip-audit`, Safety, Bandit, `npm audit` |
| `docker-build` | (main/develop) builds & pushes backend + frontend images to GH CR |

### Enabling

1. Push project to GitHub.  
2. In repo **Settings → Secrets & variables → Actions** add if needed:  
   * `CR_PAT` – token for GitHub Container Registry (if different from default)  
3. Open **Actions** tab to watch jobs.

---

## 4 ▸ Testing & Linting Locally

```bash
# Activate venv first
pytest                       # full Python test-suite
pytest -m unit               # just unit tests
pytest --cov=backend         # coverage report

flake8 backend               # style
black --check backend        # formatting
isort --check-only backend   # import order
mypy backend                 # static types

# Frontend
cd frontend
npm run lint
npm run type-check
```

---

## 5 ▸ JWT Authentication Cheat-Sheet

### 5.1 Register (user self-service or admin)

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
        "email":"alice@example.com",
        "password":"Alice123!",
        "full_name":"Alice Analyst",
        "role":"analyst"
      }'
```

### 5.2 Obtain Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=alice@example.com&password=Alice123!" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

Response:

```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 5.3 Authenticated Request

```bash
ACCESS=eyJhbGciOi...           # paste from previous step
curl http://localhost:8000/api/v1/chat/ask \
  -H "Authorization: Bearer $ACCESS" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

Rate-limit headers (`X-RateLimit-*`) are automatically returned.

### 5.4 Refresh Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"<refresh_token_here>"}'
```

### 5.5 Logout

Client-side—discard tokens. (Server-side blacklist TBD.)

---

## 6 ▸ Docker Images (optional)

```bash
# Build & run backend
docker build -t analyst-agent-backend .
docker run -p 8000:8000 --env-file .env analyst-agent-backend

# Build & run frontend
cd frontend
docker build -t analyst-agent-frontend .
docker run -p 3000:80 analyst-agent-frontend
```

---

## 7 ▸ Troubleshooting

| Symptom | Fix |
|---------|-----|
| `jwt.InvalidSignatureError` | Ensure `SECRET_KEY` matches backend instance |
| `401 Not authenticated` | Include `Authorization: Bearer <token>` header |
| `429 Rate limit exceeded` | Wait or lower request frequency |
| `neo4j.ServiceUnavailable` | Confirm Neo4j container up & `NEO4J_PASSWORD` correct |

---

### 🎉 You’re all set!

Secure CI, JWT auth, and a robust test-suite are now in place—happy hacking!  
For deeper architectural details see [`docs/`](docs) and the roadmap (`ROADMAP.md`).
