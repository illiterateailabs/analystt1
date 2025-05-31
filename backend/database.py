"""
Database connection and session management for the Analyst Agent application.

This module sets up the SQLAlchemy engine, sessionmaker, and provides
helper functions for asynchronous database interactions using asyncpg.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from backend.config import settings

logger = logging.getLogger(__name__)

# Database URL from environment settings
DATABASE_URL = settings.DATABASE_URL

# Create an asynchronous SQLAlchemy engine
# The poolclass=NullPool is used to prevent connection pooling issues in some async contexts
# and to ensure connections are closed properly after each use, especially in serverless
# or short-lived request environments.
engine = create_async_engine(
    DATABASE_URL,
    echo=settings.DEBUG,  # Echo SQL statements if debug is enabled
    poolclass=NullPool,
    # Future=True enables 2.0 style API for Session and other components
    future=True
)

# Create a sessionmaker for asynchronous sessions
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False # Prevents SQLAlchemy from expiring objects after commit
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get an asynchronous database session.

    Yields:
        AsyncSession: An asynchronous SQLAlchemy session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}", exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()

async def test_db_connection():
    """
    Tests the database connection by executing a simple query.
    """
    try:
        async with engine.connect() as connection:
            result = await connection.execute("SELECT 1")
            if result.scalar_one() == 1:
                logger.info("Successfully connected to the database.")
                return {"success": True, "message": "Database connected successfully."}
            else:
                logger.error("Database connection test failed: Unexpected query result.")
                return {"success": False, "message": "Database connection test failed: Unexpected query result."}
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}", exc_info=True)
        return {"success": False, "message": f"Failed to connect to the database: {str(e)}"}
