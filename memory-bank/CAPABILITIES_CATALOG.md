# Capabilities Catalog — **v1.9.0-alpha**
*Feature inventory updated 2025-06-22*

## 🔍 Fraud Detection & Analysis

### Advanced Pattern Recognition
- **GNN Fraud Detection**: Graph Neural Network-based suspicious activity identification
- **Whale Tracking**: Large transaction monitoring and behavioral analysis  
- **Structuring Detection**: Anti-money laundering pattern recognition
- **Cross-Chain Identity**: Multi-blockchain entity correlation and tracking
- **Anomaly Detection**: Statistical outlier identification in transaction patterns

### Investigation Tools
- **Transaction Flow Analysis**: End-to-end money trail reconstruction
- **Entity Clustering**: Automated grouping of related addresses/entities
- **Risk Scoring**: ML-based risk assessment for addresses and transactions
- **Pattern Library**: Configurable YAML-based fraud motif definitions
- **Temporal Analysis**: Time-series pattern recognition for suspicious behavior

## 📊 Data Intelligence & Visualization

### Real-Time Data Processing
- **Multi-Chain Ingestion**: Supports Ethereum, Bitcoin, and other major chains
- **SIM API Integration**: Real-time blockchain data streaming
- **Balance Tracking**: Live wallet balance monitoring across chains
- **Token Analysis**: ERC-20/ERC-721 token holder and transfer analysis
- **Activity Monitoring**: Real-time transaction and contract interaction tracking
- **Prometheus Metrics**: `/metrics` endpoint exposing rich business & infra metrics
- **Sentry Error Tracking**: Centralised exception capture & performance tracing
- **Typed EventBus**: Structured pub/sub event system with async & batching

### Graph Analytics
- **Neo4j Integration**: Advanced graph database for relationship analysis
- **Cypher Query Engine**: Complex graph pattern matching capabilities
- **Subgraph Analysis**: Focused analysis on specific network segments
- **Centrality Metrics**: Network importance and influence calculations
- **Path Analysis**: Shortest path and connectivity analysis between entities
- **Graph-Aware RAG**:  
  - Vector embeddings for nodes / edges / paths / subgraphs  
  - Redis Vector store with semantic similarity search  
  - Query expansion, re-ranking & hybrid search strategies  
  - Cypher-backed context retrieval for LLM prompts

### Visualization & Reporting
- **Interactive Graph Visualization**: Dynamic network relationship displays
- **Dashboard Analytics**: Real-time metrics and KPI monitoring
- **Evidence Bundling**: Structured investigation result packaging
- **Export Capabilities**: Multiple format support for analysis results
- **Audit Trail**: Comprehensive investigation history tracking
- **Evidence Management**:  
  - Standard `EvidenceBundle` (narrative / evidence / raw)  
  - Confidence & quality scoring, uncertainty metrics  
  - Export to JSON / HTML / PDF / DataFrame  
  - Provenance tracking & chain-of-custody audit events

## 🤖 AI-Powered Analysis

### Multi-Agent System
- **CrewAI Framework**: Coordinated multi-agent analytical workflows
- **Specialized Agents**: Domain-specific agents for different analysis types
- **Hierarchical Planning**: Multi-level task decomposition and execution
- **Context Awareness**: Persistent memory and cross-session learning
- **Human-in-the-Loop**: Interactive review and approval workflows
- **Enhanced Crew Platform**:  
  - YAML crew configs with agents / tasks / workflows  
  - `CREW_MODE` toggle (sequential / hierarchical / planning)  
  - HITL pause / resume integration
- **Graph-Aware RAG Service**: Seamless LLM context retrieval grounded in graph data

### LLM Integration
- **Google Gemini**: Advanced reasoning and natural language processing
- **Contextual Analysis**: Domain-aware interpretation of blockchain data
- **Report Generation**: Automated narrative creation from analytical findings
- **Query Translation**: Natural language to technical query conversion
- **Explanation Generation**: Human-readable explanations of complex patterns
- **Explain-with-Cypher (Proto)**: Captures & cites Cypher queries used for answers *(80 % complete)*

## 🛠️ Platform Infrastructure

### API & Integration
- **RESTful API**: Comprehensive /api/v1/* endpoint coverage
- **WebSocket Support**: Real-time bidirectional communication
- **GraphQL Queries**: Flexible data retrieval interface
- **Webhook Integration**: External system notification capabilities
- **MCP Protocol**: Model Context Protocol for extensible integrations
- **Tool Auto-Discovery**: Startup scan of `agents/tools` → `/api/v1/tools` registry  
  - Health checks, metadata, schemas & MCP manifest  
  - Sync/async execution endpoints  
- **Provider Registry**: Central YAML for all external data providers (auth, rate-limit, retry)
- **Redis Tiering**: DB 0 = cache, DB 1 = vector store (+ Pub/Sub Pipelines)
- **Prometheus + Sentry Hooks**: First-class observability baked in

### Security & Authentication
- **JWT Authentication**: Secure token-based user authentication
- **Role-Based Access Control**: Granular permission management
- **Secure Cookie Handling**: Session management with security best practices
- **API Rate Limiting**: Protection against abuse and overuse
- **Input Validation**: Comprehensive request sanitization
- **Back-Pressure Middleware (Planned)**: Automatic throttling when provider budget low

### Development & Operations
- **Docker Compose**: Complete containerized development environment
- **CI/CD Pipeline**: Automated testing and deployment workflows
- **Code Quality**: Automated linting, type checking, and formatting
- **Testing Suite**: Comprehensive unit, integration, and E2E testing
- **Monitoring Ready**: Structured logging and metrics collection
- **OpenTelemetry (Planned)**: Distributed tracing across API & agents

## 📈 Simulation & Modeling

### Sandbox Environment
- **E2B Integration**: Secure code execution and testing environment
- **Transaction Simulation**: What-if analysis for investigation scenarios
- **Balance Projection**: Predictive modeling for account behaviors
- **Network Effect Modeling**: Impact analysis of interventions
- **Scenario Planning**: Multiple outcome evaluation capabilities

### Data Generation
- **Random Transaction Generator**: Synthetic data creation for testing
- **Network Topology Generation**: Artificial graph structure creation
- **Stress Testing**: Load simulation for performance validation
- **Edge Case Generation**: Boundary condition testing capabilities
- **Mock Data Providers**: Testing infrastructure for development

## 🎯 Specialized Domains

### Cryptocurrency Analysis
- **Multi-Chain Support**: Cross-blockchain analytical capabilities
- **DeFi Protocol Analysis**: Decentralized finance pattern recognition
- **Token Economics**: Tokenomics analysis and modeling
- **Mining Pool Analysis**: Proof-of-work network analysis
- **Staking Analysis**: Proof-of-stake network behavior analysis

### Compliance & Regulatory
- **AML Pattern Detection**: Anti-money laundering compliance checking
- **Regulatory Reporting**: Structured report generation for authorities
- **Risk Assessment**: Compliance risk scoring and management
- **Audit Trail**: Complete investigation history for regulatory review
- **Policy Engine**: Configurable rule-based compliance checking

---

## 🚀 Extensibility Features

### Plugin Architecture
- **Tool Auto-Discovery**: Automatic recognition of new analysis tools
- **Provider Registry**: Pluggable external data source integration
- **Custom Crew Definitions**: YAML-based workflow configuration
- **Template Engine**: Reusable analysis and report templates
- **Hook System**: Extensible event-driven architecture
- **Human-in-the-Loop (HITL) System**:  
  - `/api/v1/hitl` endpoints for review queue, templates, stats  
  - WebSocket push notifications & webhook ingestion  
  - Auto-timeout with fallback actions

### Integration Capabilities
- **API-First Design**: All features accessible via programmatic interface
- **Webhook Support**: Real-time notifications to external systems
- **Export Formats**: Multiple output formats for downstream systems
- **Database Flexibility**: Multiple backend storage options
- **Cloud-Native**: Kubernetes and container orchestration ready

---
*Catalog reflects capabilities delivered through **Phase 3 (v1.9.0-alpha)** – further expansion scheduled for Phases 4-5.*
