"""
Database models for the Analyst Agent application.

This package defines SQLAlchemy models for various entities in the application,
such as users, sessions, and other domain-specific data.
"""

from backend.models.user import Base, User

__all__ = [
    "Base",
    "User",
]
