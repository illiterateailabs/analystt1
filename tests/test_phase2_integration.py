import pytest
import httpx
import asyncio
import websockets
import json
import os
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Any, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.integrations.neo4j_client import Neo4jClient
from backend.core.redis_client import get_redis_client
from backend.ml.registry import get_model_registry, create_dummy_model
from backend.tenancy import TenantIsolationLevel, DEFAULT_TENANT, SYSTEM_TENANT

# Set environment variables for testing
os.environ["FASTAPI_DEBUG"] = "true"
os.environ["ENVIRONMENT"] = "test"
os.environ["APP_VERSION"] = "test-v1.0"
os.environ["ENABLE_METRICS"] = "false"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes-only"
os.environ["JWT_SECRET"] = "test-jwt-secret-key-for-testing-purposes-only"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_analyst_droid"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["ML_MODEL_STORAGE"] = "local"
os.environ["ML_MODEL_DIR"] = "test_models"
os.environ["STREAM_BACKEND"] = "redis"
os.environ["DEFAULT_TENANT"] = "default"
os.environ["TENANT_ISOLATION_LEVEL"] = "field" # Use field-based for simpler testing

# Ensure test model directory exists and is clean
TEST_MODEL_DIR = "test_models"
if os.path.exists(TEST_MODEL_DIR):
    import shutil
    shutil.rmtree(TEST_MODEL_DIR)
os.makedirs(TEST_MODEL_DIR, exist_ok=True)

# --- Fixtures ---

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="session")
async def db_engine():
    """Provides an async SQLAlchemy engine for the test database."""
    engine = create_async_engine(os.environ["DATABASE_URL"])
    yield engine
    await engine.dispose()

@pytest.fixture(scope="session")
async def setup_test_db(db_engine):
    """Sets up and tears down the test database for migrations."""
    async with db_engine.begin() as conn:
        # Drop all tables to ensure a clean state for migration tests
        await conn.run_sync(lambda sync_conn: sync_conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")))
        await conn.run_sync(lambda sync_conn: sync_conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")))

    # Run migrations
    # This part would typically involve calling Alembic programmatically
    # For simplicity in this test, we'll rely on the app's startup event to create tables
    # and then manually add the tenant table if needed for specific tests.
    # In a real scenario, you'd run `alembic upgrade head` here.
    # For now, we'll let the app's startup create the initial tables, and then test the tenant migration.
    pass

@pytest.fixture(scope="function")
async def client(setup_test_db) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an httpx client for testing FastAPI endpoints."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        # Ensure app startup events run
        await app.router.startup()
        # Create dummy model for ML scoring tests
        await create_dummy_model()
        yield ac
        # Ensure app shutdown events run
        await app.router.shutdown()

@pytest.fixture(scope="function")
async def neo4j_client() -> AsyncGenerator[Neo4jClient, None]:
    """Provides a Neo4j client for testing."""
    client = Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"]
    )
    await client.connect()
    # Clear Neo4j for a clean test state
    await client.query("MATCH (n) DETACH DELETE n")
    yield client
    await client.close()

@pytest.fixture(scope="function")
async def redis_client():
    """Provides a Redis client for testing."""
    client = get_redis_client()
    await client.flushdb() # Clear Redis for a clean test state
    yield client
    await client.close()

@pytest.fixture
def auth_headers():
    """Provides authentication headers for an admin user."""
    # In a real app, you'd generate a valid JWT token here.
    # For integration tests, a mock token or a simple bypass might be used.
    # Assuming a simple mock for now, as auth is handled by backend.auth.dependencies.
    # The RoleChecker expects a user dict, so we'll simulate that.
    return {"Authorization": "Bearer mock_admin_token"}

# --- Helper Functions ---

async def create_test_user(client: httpx.AsyncClient, username: str, password: str, role: str = "analyst"):
    """Helper to create a user for testing authentication."""
    response = await client.post("/api/v1/auth/register", json={"username": username, "password": password, "role": role})
    assert response.status_code == 200
    return response.json()

async def get_auth_token(client: httpx.AsyncClient, username: str, password: str):
    """Helper to get an auth token."""
    response = await client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}

# --- Tests ---

@pytest.mark.asyncio
async def test_api_root_health(client: httpx.AsyncClient):
    """Test the root and health endpoints."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_database_migrations(db_engine):
    """
    Test that the multi-tenancy and model_metadata tables/columns are created.
    This relies on the app's startup event to run migrations.
    """
    async with db_engine.connect() as conn:
        # Check for 'tenants' table
        result = await conn.execute(text("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'tenants');"))
        assert result.scalar() is True, "tenants table should exist"

        # Check for 'model_metadata' table
        result = await conn.execute(text("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'model_metadata');"))
        assert result.scalar() is True, "model_metadata table should exist"

        # Check for 'tenant_id' column in 'users' table
        result = await conn.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='tenant_id');"))
        assert result.scalar() is True, "tenant_id column should exist in users table"

        # Check for 'tenant_id' column in 'conversations' table
        result = await conn.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conversations' AND column_name='tenant_id');"))
        assert result.scalar() is True, "tenant_id column should exist in conversations table"

        # Check for 'tenant_id' column in 'hitl_reviews' table
        result = await conn.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='hitl_reviews' AND column_name='tenant_id');"))
        assert result.scalar() is True, "tenant_id column should exist in hitl_reviews table"

@pytest.mark.asyncio
async def test_environment_configuration(client: httpx.AsyncClient):
    """Test that new environment variables are correctly loaded."""
    # This is implicitly tested by the app starting up without errors
    # and by the behavior of the new features.
    # We can add a simple check for a known env var if exposed.
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["environment"] == "test"
    assert response.json()["version"] == "test-v1.0"

@pytest.mark.asyncio
async def test_advanced_graph_community_detection(client: httpx.AsyncClient, neo4j_client: Neo4jClient, auth_headers: Dict[str, str]):
    """Test community detection endpoint."""
    # Populate Neo4j with some test data
    await neo4j_client.query("""
        CREATE (w1:Wallet {id: 'w1', risk_score: 0.8})
        CREATE (w2:Wallet {id: 'w2', risk_score: 0.7})
        CREATE (w3:Wallet {id: 'w3', risk_score: 0.2})
        CREATE (w4:Wallet {id: 'w4', risk_score: 0.1})
        CREATE (w5:Wallet {id: 'w5', risk_score: 0.9})
        CREATE (w1)-[:SENDS_TO {amount_usd: 1000}]->(w2)
        CREATE (w2)-[:SENDS_TO {amount_usd: 500}]->(w1)
        CREATE (w3)-[:SENDS_TO {amount_usd: 100}]->(w4)
        CREATE (w5)-[:SENDS_TO {amount_usd: 2000}]->(w1)
    """)

    response = await client.post(
        "/api/v1/advanced-graph/community/detect",
        headers=auth_headers,
        json={
            "node_type": "Wallet",
            "algorithm": "louvain",
            "store_results": True,
            "min_community_size": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["operation"] == "community_detection"
    assert "community_count" in data["result"]
    assert data["result"]["community_count"] >= 1 # At least one community should be found

@pytest.mark.asyncio
async def test_advanced_graph_risk_propagation(client: httpx.AsyncClient, neo4j_client: Neo4jClient, auth_headers: Dict[str, str]):
    """Test risk propagation endpoint."""
    # Populate Neo4j with some test data
    await neo4j_client.query("""
        CREATE (w1:Wallet {id: 'w1', risk_score: 0.9})
        CREATE (w2:Wallet {id: 'w2', risk_score: 0.5})
        CREATE (w3:Wallet {id: 'w3', risk_score: 0.1})
        CREATE (w1)-[:SENDS_TO {amount_usd: 10000}]->(w2)
        CREATE (w2)-[:SENDS_TO {amount_usd: 5000}]->(w3)
    """)

    response = await client.post(
        "/api/v1/advanced-graph/risk/propagate",
        headers=auth_headers,
        json={
            "seed_nodes": ["w1"],
            "node_type": "Wallet",
            "risk_property": "risk_score",
            "decay_factor": 0.5,
            "max_hops": 2,
            "update_scores": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["operation"] == "risk_propagation"
    assert "high_risk_nodes" in data["result"]
    assert len(data["result"]["high_risk_nodes"]) >= 1 # w1 should be high risk

@pytest.mark.asyncio
async def test_ml_scoring_transaction(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Test ML scoring for a single transaction."""
    transaction_data = {
        "id": "tx123",
        "timestamp": datetime.now().isoformat(),
        "from_address": "0xabc",
        "to_address": "0xdef",
        "amount": 100.0,
        "amount_usd": 1500.0,
        "chain_id": "ethereum",
        "transaction_type": "transfer"
    }
    response = await client.post(
        "/api/v1/ml-scoring/transaction",
        headers=auth_headers,
        json={"transaction_data": transaction_data}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "risk_score" in data["result"]
    assert 0 <= data["result"]["risk_score"] <= 1
    assert data["result"]["model_version"] == "mock_v1"

@pytest.mark.asyncio
async def test_ml_scoring_entity(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Test ML scoring for an entity."""
    entity_data = {
        "id": "wallet456",
        "address": "0x1234567890abcdef",
        "balance_usd": 50000.0,
        "transaction_count": 250
    }
    response = await client.post(
        "/api/v1/ml-scoring/entity",
        headers=auth_headers,
        json={"entity_data": entity_data, "entity_type": "Wallet"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "risk_score" in data["result"]
    assert 0 <= data["result"]["risk_score"] <= 1
    assert data["result"]["entity_id"] == "wallet456"

@pytest.mark.asyncio
async def test_ml_scoring_subgraph(client: httpx.AsyncClient, neo4j_client: Neo4jClient, auth_headers: Dict[str, str]):
    """Test ML scoring for a subgraph."""
    # Populate Neo4j with some test data
    await neo4j_client.query("""
        CREATE (w1:Wallet {id: 'sub_w1', risk_score: 0.8})
        CREATE (w2:Wallet {id: 'sub_w2', risk_score: 0.7})
        CREATE (w3:Wallet {id: 'sub_w3', risk_score: 0.2})
        CREATE (w1)-[:SENDS_TO {amount_usd: 100}]->(w2)
        CREATE (w2)-[:SENDS_TO {amount_usd: 50}]->(w3)
    """)

    response = await client.post(
        "/api/v1/ml-scoring/subgraph",
        headers=auth_headers,
        json={"node_ids": ["sub_w1", "sub_w2"], "node_type": "Wallet", "n_hops": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "subgraph_risk_score" in data["result"]
    assert "node_scores" in data["result"]
    assert len(data["result"]["node_scores"]) >= 2 # Should score at least the requested nodes

@pytest.mark.asyncio
async def test_multi_tenant_isolation_header(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Test multi-tenant isolation using X-Tenant-ID header."""
    tenant1_headers = auth_headers.copy()
    tenant1_headers["X-Tenant-ID"] = "tenant1"

    tenant2_headers = auth_headers.copy()
    tenant2_headers["X-Tenant-ID"] = "tenant2"

    # Create a user for tenant1
    user1_data = await create_test_user(client, "user1_tenant1", "password123", "analyst")
    user1_token = await get_auth_token(client, "user1_tenant1", "password123")
    user1_headers = user1_token.copy()
    user1_headers["X-Tenant-ID"] = "tenant1"

    # Create a user for tenant2
    user2_data = await create_test_user(client, "user2_tenant2", "password123", "analyst")
    user2_token = await get_auth_token(client, "user2_tenant2", "password123")
    user2_headers = user2_token.copy()
    user2_headers["X-Tenant-ID"] = "tenant2"

    # Test that a user from tenant1 cannot access data from tenant2 (mocked by endpoint behavior)
    # For this integration test, we'll check if the tenant ID is correctly propagated.
    # The actual data isolation logic is within the service layer, which is mocked here.
    # We can verify the middleware sets the tenant ID in the response header.
    response = await client.get("/api/v1/auth/me", headers=user1_headers)
    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == "tenant1"

    response = await client.get("/api/v1/auth/me", headers=user2_headers)
    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == "tenant2"

    # Test default tenant behavior
    response = await client.get("/api/v1/auth/me", headers=auth_headers) # No X-Tenant-ID
    assert response.status_code == 200
    assert response.headers.get("x-tenant-id") == DEFAULT_TENANT

@pytest.mark.asyncio
async def test_streaming_websocket_connection(client: httpx.AsyncClient, redis_client, auth_headers: Dict[str, str]):
    """Test the WebSocket connection for transaction streaming."""
    # Simulate a transaction being added to the Redis stream
    async def produce_message(tenant_id: str, chain_id: str, is_high_risk: bool):
        stream_name = f"tx_stream:{tenant_id}:tx_stream"
        message = {
            "id": f"tx_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "from_address": "0xproducer",
            "to_address": "0xconsumer",
            "amount": 100.0,
            "amount_usd": 1000.0,
            "chain_id": chain_id,
            "transaction_type": "transfer",
            "risk_score": 0.9 if is_high_risk else 0.1,
            "is_high_risk": is_high_risk,
            "tenant_id": tenant_id
        }
        await redis_client.xadd(stream_name, message)

    # Test with a specific tenant and filter
    test_tenant_id = "test_tenant_ws"
    test_chain_id = "ethereum"

    # Start WebSocket client
    async with websockets.connect(
        f"ws://localhost:8000/api/v1/ws/tx_stream?tenant_id={test_tenant_id}&chain_id={test_chain_id}&high_risk_only=true",
        extra_headers=auth_headers # Pass auth headers if needed by the WS endpoint
    ) as websocket:
        # Give some time for the connection to establish
        await asyncio.sleep(0.5)

        # Produce a high-risk message for the correct tenant and chain
        await produce_message(test_tenant_id, test_chain_id, True)
        await produce_message(test_tenant_id, "polygon", True) # Wrong chain
        await produce_message(test_tenant_id, test_chain_id, False) # Not high risk
        await produce_message("other_tenant", test_chain_id, True) # Wrong tenant

        received_messages = []
        try:
            # Wait for messages, with a timeout
            for _ in range(2): # Expecting at least one message
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                received_messages.append(json.loads(message))
        except asyncio.TimeoutError:
            pass # No more messages expected or timeout reached

        assert len(received_messages) == 1
        assert received_messages[0]["tenant_id"] == test_tenant_id
        assert received_messages[0]["chain_id"] == test_chain_id
        assert received_messages[0]["is_high_risk"] is True

        # Test connection without filters (should receive all from default tenant)
        async with websockets.connect(
            f"ws://localhost:8000/api/v1/ws/tx_stream?tenant_id={DEFAULT_TENANT}",
            extra_headers=auth_headers
        ) as websocket_default:
            await asyncio.sleep(0.5)
            await produce_message(DEFAULT_TENANT, "bitcoin", False)
            message = await asyncio.wait_for(websocket_default.recv(), timeout=5)
            assert json.loads(message)["tenant_id"] == DEFAULT_TENANT

@pytest.mark.asyncio
async def test_service_initialization_and_teardown(client: httpx.AsyncClient):
    """
    Test that services initialize and shut down correctly.
    This is primarily covered by the `client` fixture's startup/shutdown.
    We can add a simple check to ensure a service is available after startup.
    """
    # Check if ML model registry is initialized and has the dummy model
    model_registry = get_model_registry()
    latest_models = await model_registry.get_latest_models()
    assert len(latest_models) >= 1
    assert latest_models[0].version == "dummy_v1"

    # Check if Redis client is available
    redis_client_instance = get_redis_client()
    assert await redis_client_instance.ping() is True

    # Check if Neo4j client is available (implicitly by advanced graph tests)

    # No explicit teardown test needed as the fixture handles it.

@pytest.mark.asyncio
async def test_error_handling_invalid_input(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Test error handling for invalid input to ML scoring endpoint."""
    response = await client.post(
        "/api/v1/ml-scoring/transaction",
        headers=auth_headers,
        json={"transaction_data": {"invalid_field": "value"}} # Missing required fields
    )
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation error)
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_error_handling_unauthorized_access(client: httpx.AsyncClient):
    """Test unauthorized access to protected endpoints."""
    response = await client.post(
        "/api/v1/ml-scoring/transaction",
        json={"transaction_data": {"id": "tx1", "amount_usd": 100}}
    )
    assert response.status_code == 401 # Unauthorized

@pytest.mark.asyncio
async def test_performance_baseline_ml_scoring(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Basic performance baseline test for ML scoring."""
    transaction_data = {
        "id": "perf_tx",
        "timestamp": datetime.now().isoformat(),
        "from_address": "0xabc",
        "to_address": "0xdef",
        "amount": 100.0,
        "amount_usd": 1500.0,
        "chain_id": "ethereum",
        "transaction_type": "transfer"
    }
    start_time = datetime.now()
    response = await client.post(
        "/api/v1/ml-scoring/transaction",
        headers=auth_headers,
        json={"transaction_data": transaction_data}
    )
    end_time = datetime.now()
    assert response.status_code == 200
    duration = (end_time - start_time).total_seconds()
    print(f"ML Scoring (single transaction) took: {duration:.4f} seconds")
    assert duration < 0.5 # Expect a fast response for mock service

@pytest.mark.asyncio
async def test_performance_baseline_batch_ml_scoring(client: httpx.AsyncClient, auth_headers: Dict[str, str]):
    """Basic performance baseline test for batch ML scoring."""
    transactions = []
    for i in range(50): # Test with 50 transactions
        transactions.append({
            "id": f"batch_tx_{i}",
            "timestamp": datetime.now().isoformat(),
            "from_address": f"0xabc{i}",
            "to_address": f"0xdef{i}",
            "amount": 100.0 + i,
            "amount_usd": 1500.0 + i * 10,
            "chain_id": "ethereum",
            "transaction_type": "transfer"
        })
    start_time = datetime.now()
    response = await client.post(
        "/api/v1/ml-scoring/batch/transactions",
        headers=auth_headers,
        json={"transactions": transactions}
    )
    end_time = datetime.now()
    assert response.status_code == 200
    assert len(response.json()) == 50
    duration = (end_time - start_time).total_seconds()
    print(f"ML Scoring (batch of 50) took: {duration:.4f} seconds")
    assert duration < 1.0 # Expect a fast response for mock service
