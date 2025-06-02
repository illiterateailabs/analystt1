# Analyst's Augmentation Agent

An AI-powered system that revolutionizes analyst workflows across finance and research domains by integrating multimodal understanding, graph analytics, secure code execution, and standardized tool use.

## 🏗️ Architecture

- **FastAPI** – async Python backend, REST API surface  
- **Next.js** – React-based frontend for chat, graph & dashboards  
- **Gemini API** – multimodal LLM for reasoning, code generation, and NLU  
- **Neo4j** – graph database for complex interconnected data analysis  
- **e2b.dev** – secure cloud execution environment for AI-generated code  
- **MCP** – Model Context Protocol for standardized tool interaction  
- **Python** – primary integration & orchestration layer  

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (for Neo4j **and** Postgres & Redis via `docker-compose`)
- Google Cloud API key (Gemini)
- e2b.dev API key

### Installation

1. **Clone and set up environment**:
```bash
git clone <repo-url>
cd analystt1
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install frontend dependencies**:
```bash
cd frontend
npm install
cd ..
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Start services (Neo4j, Postgres, Redis)**:
```bash
docker-compose up -d neo4j postgres redis
```

5. **Run the application**:
```bash
# Backend
python -m uvicorn backend.main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend && npm run dev
```

## 📁 Project Structure

```
analystt1/
├── backend/                 # FastAPI backend
│   ├── core/               # Core orchestration & reasoning
│   ├── integrations/       # External service integrations
│   ├── models/             # Data models & schemas
│   ├── services/           # Business logic services
│   └── api/                # API endpoints
├── frontend/               # React (Next.js) frontend
├── neo4j/                  # Neo4j configuration & scripts
├── e2b_sandboxes/          # e2b.dev sandbox templates
├── mcp_tools/             # MCP tool implementations
├── tests/                 # Test suites
└── docs/                  # Documentation
```

## 🔧 Development Phases

- **Phase 1**: Core Foundation & PoC ✅ (Completed)
- **Phase 2**: MVP for Financial Crime Analysis ✅
- **Phase 3**: Integrations & Ecosystem (Completed)
- **Phase 4**: Advanced AI & "Data Detective" (Current)
- **Phase 5**: Ongoing Evolution

## 📊 Features

### Current
- [x] Gemini API integration
- [x] FastAPI backend & Next.js UI
- [x] Neo4j graph database setup
- [x] e2b.dev secure execution
- [x] Basic NLQ-to-Cypher translation
- [x] JWT + RBAC security
- [x] Prometheus metrics

### Planned
- [ ] Advanced fraud detection modules
- [ ] Multimodal document analysis
- [ ] MCP tool ecosystem expansion
- [ ] Graph analytics & pattern recognition
- [ ] AI-assisted hypothesis generation

## 🛡️ Security & Ethics

- Secure API key management
- Isolated code execution via e2b.dev
- Data privacy compliance (GDPR/CCPA)
- Explainable AI (XAI) integration
- Bias mitigation strategies

## 📚 Documentation

- [Architecture Overview](memory-bank/systemPatterns.md)
- API Reference – TODO
- Development Guide – TODO
- Deployment Guide – TODO

## 🤝 Contributing

See our forthcoming [CONTRIBUTING.md](CONTRIBUTING.md) for development practices and guidelines.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
