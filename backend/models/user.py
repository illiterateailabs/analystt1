"""
SQLAlchemy User model for the Analyst Agent application.

This module defines the User model, including fields for authentication,
user roles, and timestamps. It also provides methods for password
hashing and verification using bcrypt.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from passlib.hash import bcrypt

Base = declarative_base()

class User(Base):
    """
    Represents a user in the system.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, default="analyst", nullable=False)  # e.g., "admin", "analyst", "compliance"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Add a unique index for email for faster lookups and to enforce uniqueness
    __table_args__ = (
        Index('ix_users_email', 'email', unique=True),
    )

    def __repr__(self):
        return f"<User(id='{self.id}', email='{self.email}', role='{self.role}')>"

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hashes a plain text password using bcrypt.
        """
        return bcrypt.hash(password)

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verifies a plain text password against a hashed password.
        """
        try:
            return bcrypt.verify(password, hashed_password)
        except ValueError:
            # This can happen if the hashed_password format is invalid
            return False
