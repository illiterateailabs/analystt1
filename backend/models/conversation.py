"""
SQLAlchemy models for conversations and messages.

This module defines the database models for storing chat conversations
and their associated messages, replacing the previous in-memory storage.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, DateTime, ForeignKey, Index, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Conversation(Base):
    """
    Represents a chat conversation between a user and the assistant.
    
    A conversation contains multiple messages and tracks metadata like
    creation time, last update, and title.
    """
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,  # Allow anonymous conversations
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Conversation",
        server_default="New Conversation",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        onupdate=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        server_default="true",
        nullable=False,
    )
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Message.timestamp",
    )
    user = relationship("User", back_populates="conversations", lazy="selectin")
    
    # Indexes
    __table_args__ = (
        Index("ix_conversations_created_at", created_at.desc()),
        Index("ix_conversations_user_id_created_at", user_id, created_at.desc()),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title='{self.title}'>"
    
    @property
    def message_count(self) -> int:
        """Get the number of messages in this conversation."""
        return len(self.messages)
    
    @property
    def last_message(self) -> Optional["Message"]:
        """Get the most recent message in this conversation."""
        if not self.messages:
            return None
        return self.messages[-1]
    
    def to_dict(self) -> dict:
        """Convert the conversation to a dictionary representation."""
        return {
            "conversation_id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "messages": [message.to_dict() for message in self.messages],
            "is_active": self.is_active,
        }


class Message(Base):
    """
    Represents a single message within a conversation.
    
    Messages can be from either the user or the assistant and contain
    the message content and metadata.
    """
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(
        String(10000),  # Limit message length to 10,000 chars
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
    )
    metadata: Mapped[Optional[dict]] = mapped_column(
        nullable=True,  # Allow storing additional message metadata
    )
    
    # Relationships
    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages",
        lazy="selectin",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_messages_conversation_id_timestamp", conversation_id, timestamp),
    )
    
    def __repr__(self) -> str:
        return f"<Message id={self.id} role='{self.role}' conversation_id={self.conversation_id}>"
    
    def to_dict(self) -> dict:
        """Convert the message to a dictionary representation."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
