"""
Application configuration settings for the Analyst Agent.

This module uses Pydantic Settings to load configuration from environment
variables, providing a centralized and type-safe way to manage application
settings, API keys, database connections, and more.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, SecretStr, PostgresDsn, AnyHttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class AppSettings(BaseSettings):
    """
    General application settings.
    """
    APP_NAME: str = Field("Analyst Augmentation Agent", description="Name of the application")
    APP_VERSION: str = Field("1.0.0", description="Application version")
    ENVIRONMENT: str = Field("development", description="Application environment (development, staging, production)")
    DEBUG: bool = Field(True, description="Enable debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    LOG_FILE_PATH: Optional[str] = Field(None, description="Path to log file")
    HOST: str = Field("0.0.0.0", description="Host to bind the application to")
    PORT: int = Field(8000, description="Port to bind the application to")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DatabaseSettings(BaseSettings):
    """
    Database connection settings.
    """
    DATABASE_URL: Optional[PostgresDsn] = Field(
        None, 
        description="Database URL in the format: postgresql+asyncpg://user:password@host:port/dbname"
    )
    DATABASE_POOL_SIZE: int = Field(5, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(10, description="Maximum number of connections to overflow")
    DATABASE_POOL_RECYCLE: int = Field(3600, description="Recycle connections after this many seconds")
    DATABASE_ECHO: bool = Field(False, description="Echo SQL queries for debugging")

    @validator("DATABASE_URL", pre=True)
    def assemble_db_url(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            # If a complete URL is provided, use it
            return v
        
        # For backward compatibility, support individual components
        user = os.getenv("POSTGRES_USER", "analyst")
        password = os.getenv("POSTGRES_PASSWORD", "analyst123")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "analyst")
        
        # Construct the async PostgreSQL URL
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Neo4jSettings(BaseSettings):
    """
    Neo4j graph database settings.
    """
    NEO4J_URI: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    NEO4J_USERNAME: str = Field("neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field("analyst123", description="Neo4j password")
    NEO4J_DATABASE: str = Field("neo4j", description="Neo4j database name")
    REQUIRE_NEO4J: bool = Field(True, description="Whether Neo4j is required for the application")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    """
    Redis connection settings.
    """
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    REDIS_PASSWORD: Optional[str] = Field(None, description="Optional Redis password")
    REDIS_DB: int = Field(0, description="Redis database number")
    REDIS_MAX_CONNECTIONS: int = Field(10, description="Maximum connections in Redis pool")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class SecuritySettings(BaseSettings):
    """
    Security-related settings, including JWT and CORS.
    """
    SECRET_KEY: str = Field("development_secret_key_change_in_production", description="Secret key for general encryption")
    JWT_SECRET_KEY: str = Field("jwt_secret_key_change_in_production", description="Secret key for JWT encoding")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRATION_MINUTES: int = Field(60, description="JWT expiration time in minutes")
    JWT_REFRESH_EXPIRATION_MINUTES: int = Field(10080, description="JWT refresh token expiration time in minutes (7 days)")
    JWT_AUDIENCE: str = Field("analyst-agent-api", description="JWT audience")
    JWT_ISSUER: str = Field("analyst-agent", description="JWT issuer")
    CORS_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"],
        description="List of allowed CORS origins"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ExternalAPISettings(BaseSettings):
    """
    Settings for external API keys and services.
    """
    GOOGLE_API_KEY: Optional[str] = Field(None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field("gemini-1.5-pro-preview-05-14", description="Default Gemini model to use") # Updated to a valid model
    MODEL: str = Field("gemini/gemini-1.5-pro-preview-05-14", description="CrewAI LLM model configuration") # Updated to a valid model
    SERPER_API_KEY: Optional[str] = Field(None, description="Serper API key for web search")
    E2B_API_KEY: Optional[str] = Field(None, description="e2b.dev API key for sandbox execution")
    E2B_TEMPLATE_ID: str = Field("python-data-science", description="e2b.dev template ID")
    
    # Financial data API keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None, description="Alpha Vantage API key for financial data")
    POLYGON_API_KEY: Optional[str] = Field(None, description="Polygon.io API key for market data")
    NEWS_API_KEY: Optional[str] = Field(None, description="News API key for news data")
    
    # Blockchain API keys
    DUNE_API_KEY: Optional[str] = Field(None, description="Dune Analytics API key")
    ETHERSCAN_API_KEY: Optional[str] = Field(None, description="Etherscan API key")
    BSCSCAN_API_KEY: Optional[str] = Field(None, description="BSCScan API key")
    POLYGONSCAN_API_KEY: Optional[str] = Field(None, description="PolygonScan API key")
    ARBISCAN_API_KEY: Optional[str] = Field(None, description="Arbiscan API key")
    OPTIMISM_API_KEY: Optional[str] = Field(None, description="Optimism Explorer API key")
    FTMSCAN_API_KEY: Optional[str] = Field(None, description="FTMScan API key")
    SNOWTRACE_API_KEY: Optional[str] = Field(None, description="Snowtrace API key")
    COINGECKO_API_KEY: Optional[str] = Field(None, description="CoinGecko API key")
    MORALIS_API_KEY: Optional[str] = Field(None, description="Moralis API key")
    COVALENT_API_KEY: Optional[str] = Field(None, description="Covalent API key")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """
    Observability settings for logging, metrics, and tracing.
    """
    SENTRY_DSN: Optional[str] = Field(None, description="Sentry DSN for error tracking")
    ENABLE_PROMETHEUS: bool = Field(True, description="Enable Prometheus metrics")
    METRICS_PATH: str = Field("/metrics", description="Path for Prometheus metrics endpoint")
    ENABLE_STRUCTURED_LOGGING: bool = Field(True, description="Enable structured logging (JSON format)")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(
    AppSettings,
    DatabaseSettings,
    Neo4jSettings,
    RedisSettings,
    SecuritySettings,
    ExternalAPISettings,
    ObservabilitySettings,
):
    """
    Main settings class that combines all setting categories.
    """
    # Add any cross-category validators or settings here
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate that environment is one of the allowed values."""
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {', '.join(allowed)}")
        return v.lower()

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create a global settings instance
settings = Settings()

# Log loaded configuration in debug mode
if settings.DEBUG:
    # Filter out sensitive values like API keys and passwords
    safe_settings = {
        k: "***REDACTED***" if any(sensitive in k.lower() for sensitive in ["key", "secret", "password", "token", "redis_password"])
        else v
        for k, v in settings.dict().items()
    }
    logger.debug(f"Loaded settings: {safe_settings}")
