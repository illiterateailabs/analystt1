"""
Add conversations and messages tables.

Revision ID: 003_add_conversations_tables
Revises: 002_add_hitl_reviews_table
Create Date: 2025-06-17 10:30:00.000000

This migration adds the conversations and messages tables to the database schema,
replacing the previous in-memory storage with persistent PostgreSQL tables.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '003_add_conversations_tables'
down_revision: Union[str, None] = '002_add_hitl_reviews_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create conversations and messages tables.
    """
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', UUID(as_uuid=False), server_default=sa.text('gen_random_uuid()'), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=False), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('title', sa.String(255), nullable=False, server_default='New Conversation'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    )
    
    # Create indexes for conversations table
    op.create_index('ix_conversations_created_at', 'conversations', ['created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('ix_conversations_user_id_created_at', 'conversations', ['user_id', 'created_at'], postgresql_ops={'created_at': 'DESC'})
    
    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', UUID(as_uuid=False), server_default=sa.text('gen_random_uuid()'), primary_key=True),
        sa.Column('conversation_id', UUID(as_uuid=False), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('content', sa.String(10000), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('metadata', JSONB, nullable=True),
    )
    
    # Create indexes for messages table
    op.create_index('ix_messages_role', 'messages', ['role'])
    op.create_index('ix_messages_conversation_id_timestamp', 'messages', ['conversation_id', 'timestamp'])


def downgrade() -> None:
    """
    Drop messages and conversations tables.
    
    Note: Drop messages first due to foreign key constraints.
    """
    op.drop_table('messages')
    op.drop_table('conversations')
