"""
Create hitl_reviews table for human-in-the-loop review tracking.

Revision ID: 002
Revises: 001_initial_user_table
Create Date: 2025-06-03 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001_initial_user_table'
branch_labels = None
depends_on = None


def upgrade():
    # Create hitl_reviews table
    op.create_table(
        'hitl_reviews',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False),
        sa.Column('task_id', UUID(as_uuid=True), nullable=False, comment='Reference to analysis task ID'),
        sa.Column('review_type', sa.String(50), nullable=False, comment='Type of review (compliance, risk, fraud)'),
        sa.Column('risk_level', sa.String(20), nullable=False, comment='Risk level (high, medium, low)'),
        sa.Column('status', sa.String(20), nullable=False, default='pending', 
                  comment='Status (pending, approved, rejected, expired)'),
        sa.Column('reviewer_id', UUID(as_uuid=True), nullable=True, comment='User ID of the reviewer'),
        sa.Column('comments', sa.Text, nullable=True, comment='Review comments or notes'),
        sa.Column('review_started_at', sa.DateTime(timezone=True), nullable=True, 
                  comment='When review was started'),
        sa.Column('review_completed_at', sa.DateTime(timezone=True), nullable=True, 
                  comment='When review was completed'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                  onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['reviewer_id'], ['users.id'], name='fk_hitl_reviews_reviewer'),
        schema=None
    )
    
    # Create indexes for common query patterns
    op.create_index('ix_hitl_reviews_task_id', 'hitl_reviews', ['task_id'])
    op.create_index('ix_hitl_reviews_status', 'hitl_reviews', ['status'])
    op.create_index('ix_hitl_reviews_reviewer_id', 'hitl_reviews', ['reviewer_id'])
    op.create_index('ix_hitl_reviews_review_type_risk_level', 'hitl_reviews', ['review_type', 'risk_level'])
    
    # Create a trigger to update updated_at timestamp
    op.execute("""
    CREATE OR REPLACE FUNCTION update_hitl_reviews_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    
    CREATE TRIGGER trigger_update_hitl_reviews_updated_at
    BEFORE UPDATE ON hitl_reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_hitl_reviews_updated_at();
    """)


def downgrade():
    # Drop the trigger and function first
    op.execute("DROP TRIGGER IF EXISTS trigger_update_hitl_reviews_updated_at ON hitl_reviews;")
    op.execute("DROP FUNCTION IF EXISTS update_hitl_reviews_updated_at;")
    
    # Drop the table
    op.drop_table('hitl_reviews')
