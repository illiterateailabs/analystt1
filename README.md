# Analyst's Augmentation Agent

An AI-powered system that revolutionizes analyst workflows across finance and research domains by integrating multimodal understanding, graph analytics, secure code execution, and standardized tool use.

## 🏗️ Architecture

- **Gemini API**: Multimodal LLM for reasoning, code generation, and NLU
- **Neo4j**: Graph database for complex interconnected data analysis
- **e2b.dev**: Secure cloud execution environment for AI-generated code
- **MCP**: Model Context Protocol for standardized tool interaction
- **Python**: Primary integration layer

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (for Neo4j)
- Google Cloud API key (Gemini)
- e2b.dev API key

### Installation

1. **Clone and setup environment**:
```bash
git clone <repo-url>
cd eCommGeminiIntegration
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install frontend dependencies**:
```bash
cd frontend
npm install
cd ..
```

3. **Setup environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Start Neo4j**:
```bash
docker-compose up -d neo4j
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
eCommGeminiIntegration/
├── backend/                 # FastAPI backend
│   ├── core/               # Core orchestration & reasoning
│   ├── integrations/       # External service integrations
│   ├── models/             # Data models & schemas
│   ├── services/           # Business logic services
│   └── api/                # API endpoints
├── frontend/               # React frontend
├── neo4j/                  # Neo4j configuration & scripts
├── e2b_sandboxes/         # e2b.dev sandbox templates
├── mcp_tools/             # MCP tool implementations
├── tests/                 # Test suites
└── docs/                  # Documentation
```

## 🔧 Development Phases

- **Phase 1**: Core Foundation & PoC ✅ (Current)
- **Phase 2**: MVP for Financial Crime Analysis
- **Phase 3**: Feature Enrichment & MCP Integration
- **Phase 4**: Advanced AI & "Data Detective" Features
- **Phase 5**: Ongoing Evolution

## 📊 Features

### Current (Phase 1)
- [x] Gemini API integration
- [x] Neo4j graph database setup
- [x] e2b.dev secure execution
- [x] Basic NLQ-to-Cypher translation
- [x] Minimal web interface

### Planned
- [ ] Advanced fraud detection modules
- [ ] Multimodal document analysis
- [ ] MCP tool ecosystem
- [ ] Graph analytics & pattern recognition
- [ ] AI-assisted hypothesis generation

## 🛡️ Security & Ethics

- Secure API key management
- Isolated code execution via e2b.dev
- Data privacy compliance (GDPR/CCPA)
- Explainable AI (XAI) integration
- Bias mitigation strategies

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for development practices and guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
