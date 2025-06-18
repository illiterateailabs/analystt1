"""
Sim API Configuration

Configuration settings for Sim APIs integration, including authentication and endpoints.
These settings can be imported into the main config.py file.

Environment Variables:
    SIM_API_KEY: API key for authenticating with Sim APIs
    SIM_API_URL: Base URL for Sim API endpoints (defaults to https://api.sim.dune.com)
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field, validator

class SimSettings(BaseSettings):
    """
    Configuration settings for Sim APIs.
    
    Attributes:
        SIM_API_KEY: API key for authenticating with Sim APIs
        SIM_API_URL: Base URL for Sim API endpoints
    """
    
    SIM_API_KEY: str = Field(
        ...,  # Required field (no default)
        description="API key for authenticating with Sim APIs"
    )
    
    SIM_API_URL: str = Field(
        "https://api.sim.dune.com",
        description="Base URL for Sim API endpoints"
    )
    
    @validator('SIM_API_KEY')
    def validate_api_key(cls, v: str) -> str:
        """Validate that the API key is provided and non-empty."""
        if not v or not v.strip():
            raise ValueError("SIM_API_KEY must be provided")
        return v
    
    @validator('SIM_API_URL')
    def validate_api_url(cls, v: str) -> str:
        """Validate that the API URL is a proper URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("SIM_API_URL must be a valid HTTP or HTTPS URL")
        return v.rstrip('/')  # Remove trailing slash for consistency
    
    class Config:
        """Pydantic config for SimSettings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
sim_settings = SimSettings()

# Export settings as variables for easy import
SIM_API_KEY = sim_settings.SIM_API_KEY
SIM_API_URL = sim_settings.SIM_API_URL
