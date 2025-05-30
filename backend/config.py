"""Configuration management for the Analyst's Augmentation Agent."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = Field(default="Analyst Augmentation Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )

    # JWT Authentication
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=60, env="JWT_EXPIRATION_MINUTES")  # 1 hour
    jwt_refresh_expiration_minutes: int = Field(default=10080, env="JWT_REFRESH_EXPIRATION_MINUTES")  # 7 days
    jwt_audience: str = Field(default="analyst-agent-api", env="JWT_AUDIENCE")
    jwt_issuer: str = Field(default="analyst-agent", env="JWT_ISSUER")

    # Google Gemini API
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")

    # e2b.dev Configuration
    e2b_api_key: str = Field(..., env="E2B_API_KEY")
    e2b_template_id: str = Field(default="python-data-science", env="E2B_TEMPLATE_ID")

    # Database Configuration (Application State)
    database_url: str = Field(default="sqlite:///./app.db", env="DATABASE_URL")

    # External APIs (for MCP tools)
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")

    # Monitoring & Logging
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    log_file_path: str = Field(default="./logs/app.log", env="LOG_FILE_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Derived configurations
class GeminiConfig:
    """Gemini API configuration."""

    API_KEY = settings.google_api_key
    MODEL = settings.gemini_model

    # Generation parameters
    TEMPERATURE = 0.1
    TOP_P = 0.8
    TOP_K = 40
    MAX_OUTPUT_TOKENS = 8192

    # Safety settings
    SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]


class Neo4jConfig:
    """Neo4j configuration."""

    URI = settings.neo4j_uri
    USERNAME = settings.neo4j_username
    PASSWORD = settings.neo4j_password
    DATABASE = settings.neo4j_database

    # Connection settings
    MAX_CONNECTION_LIFETIME = 30 * 60  # 30 minutes
    MAX_CONNECTION_POOL_SIZE = 50
    CONNECTION_ACQUISITION_TIMEOUT = 60  # 60 seconds


class E2BConfig:
    """e2b.dev configuration."""

    API_KEY = settings.e2b_api_key
    TEMPLATE_ID = settings.e2b_template_id

    # Sandbox settings
    TIMEOUT = 300  # 5 minutes
    MEMORY_MB = 2048
    CPU_COUNT = 2


class JWTConfig:
    """JWT Authentication configuration."""
    
    ALGORITHM = settings.jwt_algorithm
    EXPIRATION_MINUTES = settings.jwt_expiration_minutes
    REFRESH_EXPIRATION_MINUTES = settings.jwt_refresh_expiration_minutes
    AUDIENCE = settings.jwt_audience
    ISSUER = settings.jwt_issuer
    SECRET_KEY = settings.secret_key


# Export commonly used configurations
__all__ = [
    "settings",
    "GeminiConfig",
    "Neo4jConfig",
    "E2BConfig",
    "JWTConfig"
]
