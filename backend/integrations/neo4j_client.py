"""Neo4j database client for graph operations."""

import logging
from typing import Dict, List, Optional, Any, Union
import asyncio
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

# Use the global application settings object instead of a standalone Neo4j
# config class. This prevents drift and ensures a single source of truth.
from backend.config import settings


logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async Neo4j client for graph database operations."""
    
    def __init__(self):
        """Initialize the Neo4j client."""
        # Keep a reference to the global settings for convenience.
        self.config = settings
        self.driver: Optional[AsyncDriver] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(
                    self.config.NEO4J_USERNAME,
                    self.config.NEO4J_PASSWORD,
                ),
            )
                max_connection_lifetime=getattr(
                    self.config, "NEO4J_MAX_CONNECTION_LIFETIME", 3600
                ),
                max_connection_pool_size=getattr(
                    self.config, "NEO4J_MAX_CONNECTION_POOL_SIZE", 10
                ),
                connection_acquisition_timeout=getattr(
                    self.config, "NEO4J_CONNECTION_ACQUISITION_TIMEOUT", 30
                ),
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            self._connected = True
            
            logger.info(f"Connected to Neo4j at {self.config.URI}")
            
            # Initialize schema
            await self._initialize_schema()
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise
    
    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected and self.driver is not None
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None):
        """Create an async session context manager."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Neo4j")
        
        session = self.driver.session(database=database or self.config.NEO4J_DATABASE)
        try:
            yield session
        finally:
            await session.close()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        try:
            async with self.session(database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                
                logger.debug(f"Executed query: {query[:100]}... | Records: {len(records)}")
                return records
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write query in a transaction."""
        try:
            async with self.session(database) as session:
                result = await session.execute_write(
                    self._execute_query_tx, query, parameters or {}
                )
                return result
                
        except Exception as e:
            logger.error(f"Error executing write query: {e}")
            raise
    
    @staticmethod
    async def _execute_query_tx(tx, query: str, parameters: Dict[str, Any]):
        """Execute query within a transaction."""
        result = await tx.run(query, parameters)
        return await result.data()
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        try:
            # Get node labels
            labels_result = await self.execute_query("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            # Get relationship types
            rel_types_result = await self.execute_query("CALL db.relationshipTypes()")
            relationship_types = [record["relationshipType"] for record in rel_types_result]
            
            # Get property keys
            prop_keys_result = await self.execute_query("CALL db.propertyKeys()")
            property_keys = [record["propertyKey"] for record in prop_keys_result]
            
            # Get constraints
            constraints_result = await self.execute_query("SHOW CONSTRAINTS")
            constraints = constraints_result
            
            # Get indexes
            indexes_result = await self.execute_query("SHOW INDEXES")
            indexes = indexes_result
            
            schema_info = {
                "labels": labels,
                "relationship_types": relationship_types,
                "property_keys": property_keys,
                "constraints": constraints,
                "indexes": indexes,
                "node_count": await self._get_node_count(),
                "relationship_count": await self._get_relationship_count()
            }
            
            logger.debug(f"Retrieved schema info: {len(labels)} labels, {len(relationship_types)} rel types")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            raise
    
    async def _get_node_count(self) -> int:
        """Get total node count."""
        result = await self.execute_query("MATCH (n) RETURN count(n) as count")
        return result[0]["count"] if result else 0
    
    async def _get_relationship_count(self) -> int:
        """Get total relationship count."""
        result = await self.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        return result[0]["count"] if result else 0
    
    async def create_node(
        self,
        labels: Union[str, List[str]],
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new node."""
        if isinstance(labels, str):
            labels = [labels]
        
        labels_str = ":".join(labels)
        query = f"CREATE (n:{labels_str} $properties) RETURN n"
        
        result = await self.execute_write_query(query, {"properties": properties})
        return result[0]["n"] if result else {}
    
    async def create_relationship(
        self,
        from_node_id: int,
        to_node_id: int,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a relationship between two nodes."""
        query = """
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:%s $properties]->(b)
        RETURN r
        """ % relationship_type
        
        params = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "properties": properties or {}
        }
        
        result = await self.execute_write_query(query, params)
        return result[0]["r"] if result else {}
    
    async def find_nodes(
        self,
        labels: Optional[Union[str, List[str]]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria."""
        query_parts = ["MATCH (n"]
        
        if labels:
            if isinstance(labels, str):
                labels = [labels]
            query_parts.append(":" + ":".join(labels))
        
        query_parts.append(")")
        
        if properties:
            where_clauses = []
            for key, value in properties.items():
                where_clauses.append(f"n.{key} = ${key}")
            
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        query_parts.append(f"RETURN n LIMIT {limit}")
        
        query = " ".join(query_parts)
        result = await self.execute_query(query, properties or {})
        
        return [record["n"] for record in result]
    
    async def _initialize_schema(self) -> None:
        """Initialize the database schema with constraints and indexes."""
        try:
            # Create constraints for unique identifiers
            constraints = [
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT organization_id_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
                "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    await self.execute_write_query(constraint)
                except Exception as e:
                    # Constraint might already exist
                    logger.debug(f"Constraint creation skipped: {e}")
            
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX transaction_amount_index IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
                "CREATE INDEX transaction_date_index IF NOT EXISTS FOR (t:Transaction) ON (t.date)",
                "CREATE INDEX document_content_index IF NOT EXISTS FOR (d:Document) ON (d.content)",
            ]
            
            for index in indexes:
                try:
                    await self.execute_write_query(index)
                except Exception as e:
                    logger.debug(f"Index creation skipped: {e}")
            
            logger.info("Schema initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            # Don't raise - schema initialization is not critical for basic functionality
