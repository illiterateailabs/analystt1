"""
Neo4j Database Client Integration

This module provides a robust, asynchronous client for interacting with the Neo4j
graph database. It is designed for high performance and is fully integrated with
the application's core systems.

Key Features:
- Asynchronous query execution using the official neo4j driver.
- Centralized configuration loaded from the provider registry.
- Integration with the BackpressureMiddleware is not directly applied here, as
  self-hosted databases typically don't have the same budget/rate-limit constraints
  as external APIs. However, performance is tracked.
- Comprehensive observability through Prometheus metrics (`DatabaseMetrics`) and
  OpenTelemetry tracing (`@trace`).
- Helper methods for common graph operations and schema management.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from backend.core.metrics import DatabaseMetrics
from backend.core.telemetry import trace
from backend.providers import get_provider

logger = logging.getLogger(__name__)


class Neo4jClient:
    """An asynchronous client for the Neo4j graph database."""

    def __init__(self):
        """
        Initializes the Neo4jClient.

        Loads all necessary configuration from the provider registry.
        """
        provider_config = get_provider("neo4j")
        if not provider_config:
            raise ValueError("Neo4j provider configuration not found in registry.")

        self.uri = provider_config.get("connection_uri")
        auth_config = provider_config.get("auth", {})
        self.username = os.getenv(auth_config.get("username_env_var"))
        self.password = os.getenv(auth_config.get("password_env_var"))
        self.database = provider_config.get("database_name", "neo4j")

        pool_config = provider_config.get("connection_pool", {})
        self.max_pool_size = pool_config.get("max_connections", 100)
        self.acquisition_timeout = pool_config.get("acquisition_timeout_seconds", 60)

        if not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j connection details (URI, USERNAME, PASSWORD) are not fully configured.")

        self.driver: Optional[AsyncDriver] = None
        logger.info("Neo4jClient initialized with configuration from provider registry.")

    @trace(name="neo4j.connect")
    async def connect(self):
        """
        Establishes and verifies the connection to the Neo4j database.
        """
        if self.driver:
            return

        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=self.max_pool_size,
                connection_acquisition_timeout=self.acquisition_timeout,
            )
            await self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            await self._initialize_schema()
        except (AuthError, ServiceUnavailable) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Neo4j connection: {e}")
            raise

    @trace(name="neo4j.close")
    async def close(self):
        """Closes the connection to the Neo4j database."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed.")

    @DatabaseMetrics.track_operation(database="neo4j", operation="execute_query")
    @trace(name="neo4j.execute_query")
    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query and returns the results.

        Args:
            query: The Cypher query string to execute.
            parameters: A dictionary of parameters to bind to the query.

        Returns:
            A list of dictionaries, where each dictionary represents a result record.
        """
        if not self.driver:
            await self.connect()

        params = parameters or {}
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, params)
                records = await result.data()
                logger.debug(f"Executed Cypher query successfully. Query: {query[:100]}...")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}\nQuery: {query}\nParams: {params}")
            raise

    @trace(name="neo4j.initialize_schema")
    async def _initialize_schema(self):
        """
        Ensures necessary constraints and indexes exist in the database.
        This method is idempotent.
        """
        logger.info("Initializing Neo4j schema: ensuring constraints and indexes exist.")
        # Define constraints for unique identifiers
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT wallet_address_unique IF NOT EXISTS FOR (w:Wallet) REQUIRE w.address IS UNIQUE",
            "CREATE CONSTRAINT transaction_hash_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.hash IS UNIQUE",
            "CREATE CONSTRAINT block_hash_unique IF NOT EXISTS FOR (b:Block) REQUIRE b.hash IS UNIQUE",
        ]
        # Define indexes for frequently queried properties
        indexes = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX transaction_value_index IF NOT EXISTS FOR (t:Transaction) ON (t.value_usd)",
            "CREATE INDEX wallet_chain_index IF NOT EXISTS FOR (w:Wallet) ON (w.chain)",
        ]

        try:
            for constraint in constraints:
                await self.execute_query(constraint)
            logger.debug("Successfully applied Neo4j constraints.")

            for index in indexes:
                await self.execute_query(index)
            logger.debug("Successfully applied Neo4j indexes.")
            logger.info("Neo4j schema initialization complete.")
        except Exception as e:
            # It's okay if this fails (e.g., permissions), but we should log it.
            logger.error(f"Could not initialize Neo4j schema (constraints/indexes): {e}")

    @trace(name="neo4j.get_schema_info")
    async def get_schema_info(self) -> Dict[str, Any]:
        """Retrieves and returns the schema of the graph database."""
        try:
            labels_query = "CALL db.labels() YIELD label"
            rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType"
            
            labels_result = await self.execute_query(labels_query)
            rel_types_result = await self.execute_query(rel_types_query)

            schema_info = {
                "labels": [record["label"] for record in labels_result],
                "relationship_types": [record["relationshipType"] for record in rel_types_result],
            }
            logger.debug("Successfully retrieved Neo4j schema info.")
            return schema_info
        except Exception as e:
            logger.error(f"Failed to retrieve Neo4j schema information: {e}")
            return {"labels": [], "relationship_types": []}

