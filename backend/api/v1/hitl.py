"""
Human-in-the-Loop (HITL) API Endpoints

This module provides a comprehensive HITL system for agent review and intervention:
- Review request creation and management
- Approval/rejection workflows
- Pause/resume agent execution
- Review queue management and prioritization
- Notification system integration
- Review templates and forms
- Timeout and fallback handling
- WebSocket support for real-time updates

The HITL system allows human experts to review and intervene in agent operations,
providing oversight, quality control, and expert input for critical decisions.
"""

import asyncio
import enum
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, verify_permissions
from backend.core.events import publish_event
from backend.core.metrics import ApiMetrics
from backend.core.redis_client import RedisClient, RedisDb
from backend.database import get_db
from backend.models.user import User

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Redis client for pub/sub and caching
redis_client = None


class ReviewStatus(str, enum.Enum):
    """Status of a HITL review."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    FEEDBACK_PROVIDED = "feedback_provided"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ReviewType(str, enum.Enum):
    """Type of HITL review."""
    APPROVAL = "approval"  # Simple approve/reject
    FEEDBACK = "feedback"  # Provide feedback
    EDIT = "edit"  # Edit the output


class ReviewPriority(str, enum.Enum):
    """Priority of a HITL review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, enum.Enum):
    """Notification channel for HITL reviews."""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"


class ReviewRequest(BaseModel):
    """Model for a HITL review request."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    crew_id: str
    task_id: str
    agent_id: str
    user_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    task_name: str
    task_output: Dict[str, Any]
    confidence: float = 0.0
    review_type: ReviewType = ReviewType.APPROVAL
    priority: ReviewPriority = ReviewPriority.MEDIUM
    status: ReviewStatus = ReviewStatus.PENDING
    timeout_minutes: Optional[int] = None
    timeout_at: Optional[str] = None
    fallback_action: Optional[str] = None
    template_id: Optional[str] = None
    form_fields: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "crew_id": "fraud_detection",
                "task_id": "pattern_analysis",
                "agent_id": "pattern_analyst",
                "task_name": "Analyze Fraud Patterns",
                "task_output": {
                    "detected_patterns": [
                        {"pattern": "wash_trading", "confidence": 0.85}
                    ],
                    "confidence": 0.85
                },
                "confidence": 0.85,
                "review_type": "approval",
                "priority": "high",
                "timeout_minutes": 30,
                "fallback_action": "proceed"
            }
        }
    
    @validator('timeout_at', always=True)
    def set_timeout_at(cls, v, values):
        """Set timeout_at based on timeout_minutes if not provided."""
        if v is None and values.get('timeout_minutes'):
            timeout = datetime.now() + timedelta(minutes=values['timeout_minutes'])
            return timeout.isoformat()
        return v


class ReviewResponse(BaseModel):
    """Model for a HITL review response."""
    review_id: str
    user_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: ReviewStatus
    feedback: Optional[str] = None
    edited_output: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "review_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user123",
                "timestamp": "2025-06-21T12:34:56.789Z",
                "status": "approved",
                "feedback": "The pattern detection looks accurate."
            }
        }


class ReviewDetail(BaseModel):
    """Detailed view of a HITL review."""
    request: ReviewRequest
    response: Optional[ReviewResponse] = None
    
    class Config:
        schema_extra = {
            "example": {
                "request": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "crew_id": "fraud_detection",
                    "task_id": "pattern_analysis",
                    "agent_id": "pattern_analyst",
                    "task_name": "Analyze Fraud Patterns",
                    "task_output": {
                        "detected_patterns": [
                            {"pattern": "wash_trading", "confidence": 0.85}
                        ],
                        "confidence": 0.85
                    },
                    "confidence": 0.85,
                    "review_type": "approval",
                    "priority": "high",
                    "status": "pending",
                    "timeout_minutes": 30,
                    "timeout_at": "2025-06-21T13:34:56.789Z",
                    "fallback_action": "proceed"
                },
                "response": None
            }
        }


class ReviewTemplate(BaseModel):
    """Template for HITL review forms."""
    id: str
    name: str
    description: str
    fields: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "fraud_review",
                "name": "Fraud Review Template",
                "description": "Template for reviewing fraud detection results",
                "fields": [
                    {
                        "name": "confidence_score",
                        "label": "Confidence Score",
                        "type": "number",
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "required": True
                    },
                    {
                        "name": "feedback",
                        "label": "Feedback",
                        "type": "textarea",
                        "required": False
                    }
                ]
            }
        }


class NotificationConfig(BaseModel):
    """Configuration for HITL notifications."""
    channels: List[NotificationChannel]
    webhook_url: Optional[str] = None
    email_recipients: Optional[List[str]] = None
    slack_webhook: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "channels": ["websocket", "email"],
                "email_recipients": ["analyst@example.com"]
            }
        }


class ReviewStats(BaseModel):
    """Statistics about HITL reviews."""
    total: int = 0
    pending: int = 0
    in_progress: int = 0
    approved: int = 0
    rejected: int = 0
    feedback_provided: int = 0
    timed_out: int = 0
    cancelled: int = 0
    average_response_time_seconds: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "total": 100,
                "pending": 5,
                "in_progress": 2,
                "approved": 75,
                "rejected": 10,
                "feedback_provided": 5,
                "timed_out": 3,
                "cancelled": 0,
                "average_response_time_seconds": 256.5
            }
        }


class PauseRequest(BaseModel):
    """Request to pause agent execution."""
    crew_id: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    reason: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "crew_id": "fraud_detection",
                "task_id": "pattern_analysis",
                "reason": "Manual review needed"
            }
        }


class ResumeRequest(BaseModel):
    """Request to resume agent execution."""
    crew_id: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "crew_id": "fraud_detection",
                "task_id": "pattern_analysis"
            }
        }


# WebSocket connection manager
class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a WebSocket client."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect a WebSocket client."""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_json(message)


# Initialize connection manager
manager = ConnectionManager()


# Database operations for HITL reviews
class HITLReviewRepository:
    """Repository for HITL review database operations."""
    
    @staticmethod
    def create_review(db: Session, review: ReviewRequest) -> ReviewRequest:
        """
        Create a new HITL review.
        
        Args:
            db: Database session
            review: Review request
            
        Returns:
            Created review
        """
        from backend.models.user import HITLReview
        
        # Create database model
        db_review = HITLReview(
            id=review.id,
            crew_id=review.crew_id,
            task_id=review.task_id,
            agent_id=review.agent_id,
            user_id=review.user_id,
            timestamp=review.timestamp,
            task_name=review.task_name,
            task_output=json.dumps(review.task_output),
            confidence=review.confidence,
            review_type=review.review_type,
            priority=review.priority,
            status=review.status,
            timeout_minutes=review.timeout_minutes,
            timeout_at=review.timeout_at,
            fallback_action=review.fallback_action,
            template_id=review.template_id,
            form_fields=json.dumps(review.form_fields) if review.form_fields else None,
        )
        
        # Add to database
        db.add(db_review)
        db.commit()
        db.refresh(db_review)
        
        # Convert back to Pydantic model
        return review
    
    @staticmethod
    def get_review(db: Session, review_id: str) -> Optional[ReviewDetail]:
        """
        Get a HITL review by ID.
        
        Args:
            db: Database session
            review_id: Review ID
            
        Returns:
            Review detail or None if not found
        """
        from backend.models.user import HITLReview, HITLReviewResponse
        
        # Get review from database
        db_review = db.query(HITLReview).filter(HITLReview.id == review_id).first()
        if not db_review:
            return None
        
        # Convert to Pydantic model
        request = ReviewRequest(
            id=db_review.id,
            crew_id=db_review.crew_id,
            task_id=db_review.task_id,
            agent_id=db_review.agent_id,
            user_id=db_review.user_id,
            timestamp=db_review.timestamp,
            task_name=db_review.task_name,
            task_output=json.loads(db_review.task_output),
            confidence=db_review.confidence,
            review_type=db_review.review_type,
            priority=db_review.priority,
            status=db_review.status,
            timeout_minutes=db_review.timeout_minutes,
            timeout_at=db_review.timeout_at,
            fallback_action=db_review.fallback_action,
            template_id=db_review.template_id,
            form_fields=json.loads(db_review.form_fields) if db_review.form_fields else None,
        )
        
        # Get response if available
        db_response = db.query(HITLReviewResponse).filter(
            HITLReviewResponse.review_id == review_id
        ).first()
        
        response = None
        if db_response:
            response = ReviewResponse(
                review_id=db_response.review_id,
                user_id=db_response.user_id,
                timestamp=db_response.timestamp,
                status=db_response.status,
                feedback=db_response.feedback,
                edited_output=json.loads(db_response.edited_output) if db_response.edited_output else None,
            )
        
        # Create review detail
        return ReviewDetail(request=request, response=response)
    
    @staticmethod
    def update_review_status(
        db: Session,
        review_id: str,
        status: ReviewStatus,
    ) -> Optional[ReviewRequest]:
        """
        Update a HITL review status.
        
        Args:
            db: Database session
            review_id: Review ID
            status: New status
            
        Returns:
            Updated review or None if not found
        """
        from backend.models.user import HITLReview
        
        # Get review from database
        db_review = db.query(HITLReview).filter(HITLReview.id == review_id).first()
        if not db_review:
            return None
        
        # Update status
        db_review.status = status
        db.commit()
        db.refresh(db_review)
        
        # Convert to Pydantic model
        return ReviewRequest(
            id=db_review.id,
            crew_id=db_review.crew_id,
            task_id=db_review.task_id,
            agent_id=db_review.agent_id,
            user_id=db_review.user_id,
            timestamp=db_review.timestamp,
            task_name=db_review.task_name,
            task_output=json.loads(db_review.task_output),
            confidence=db_review.confidence,
            review_type=db_review.review_type,
            priority=db_review.priority,
            status=db_review.status,
            timeout_minutes=db_review.timeout_minutes,
            timeout_at=db_review.timeout_at,
            fallback_action=db_review.fallback_action,
            template_id=db_review.template_id,
            form_fields=json.loads(db_review.form_fields) if db_review.form_fields else None,
        )
    
    @staticmethod
    def create_review_response(
        db: Session,
        response: ReviewResponse,
    ) -> ReviewResponse:
        """
        Create a HITL review response.
        
        Args:
            db: Database session
            response: Review response
            
        Returns:
            Created response
        """
        from backend.models.user import HITLReviewResponse
        
        # Create database model
        db_response = HITLReviewResponse(
            review_id=response.review_id,
            user_id=response.user_id,
            timestamp=response.timestamp,
            status=response.status,
            feedback=response.feedback,
            edited_output=json.dumps(response.edited_output) if response.edited_output else None,
        )
        
        # Add to database
        db.add(db_response)
        db.commit()
        db.refresh(db_response)
        
        # Update review status
        HITLReviewRepository.update_review_status(db, response.review_id, response.status)
        
        # Convert back to Pydantic model
        return response
    
    @staticmethod
    def list_reviews(
        db: Session,
        status: Optional[ReviewStatus] = None,
        crew_id: Optional[str] = None,
        priority: Optional[ReviewPriority] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ReviewRequest]:
        """
        List HITL reviews with optional filtering.
        
        Args:
            db: Database session
            status: Filter by status
            crew_id: Filter by crew ID
            priority: Filter by priority
            limit: Maximum number of reviews to return
            offset: Offset for pagination
            
        Returns:
            List of reviews
        """
        from backend.models.user import HITLReview
        
        # Build query
        query = db.query(HITLReview)
        
        # Apply filters
        if status:
            query = query.filter(HITLReview.status == status)
        if crew_id:
            query = query.filter(HITLReview.crew_id == crew_id)
        if priority:
            query = query.filter(HITLReview.priority == priority)
        
        # Apply pagination
        query = query.order_by(HITLReview.timestamp.desc())
        query = query.limit(limit).offset(offset)
        
        # Execute query
        db_reviews = query.all()
        
        # Convert to Pydantic models
        reviews = []
        for db_review in db_reviews:
            reviews.append(ReviewRequest(
                id=db_review.id,
                crew_id=db_review.crew_id,
                task_id=db_review.task_id,
                agent_id=db_review.agent_id,
                user_id=db_review.user_id,
                timestamp=db_review.timestamp,
                task_name=db_review.task_name,
                task_output=json.loads(db_review.task_output),
                confidence=db_review.confidence,
                review_type=db_review.review_type,
                priority=db_review.priority,
                status=db_review.status,
                timeout_minutes=db_review.timeout_minutes,
                timeout_at=db_review.timeout_at,
                fallback_action=db_review.fallback_action,
                template_id=db_review.template_id,
                form_fields=json.loads(db_review.form_fields) if db_review.form_fields else None,
            ))
        
        return reviews
    
    @staticmethod
    def get_review_stats(db: Session, crew_id: Optional[str] = None) -> ReviewStats:
        """
        Get statistics about HITL reviews.
        
        Args:
            db: Database session
            crew_id: Filter by crew ID
            
        Returns:
            Review statistics
        """
        from backend.models.user import HITLReview, HITLReviewResponse
        from sqlalchemy import func
        
        # Base query
        query = db.query(HITLReview.status, func.count(HITLReview.id))
        
        # Apply crew filter if provided
        if crew_id:
            query = query.filter(HITLReview.crew_id == crew_id)
        
        # Group by status and execute
        status_counts = dict(query.group_by(HITLReview.status).all())
        
        # Calculate total
        total = sum(status_counts.values())
        
        # Calculate average response time
        avg_response_time = None
        if total > 0:
            # Join reviews and responses
            time_query = db.query(
                func.avg(
                    func.extract('epoch', func.timestamp(HITLReviewResponse.timestamp)) - 
                    func.extract('epoch', func.timestamp(HITLReview.timestamp))
                )
            ).join(
                HITLReviewResponse,
                HITLReview.id == HITLReviewResponse.review_id
            )
            
            # Apply crew filter if provided
            if crew_id:
                time_query = time_query.filter(HITLReview.crew_id == crew_id)
            
            # Execute query
            avg_response_time = time_query.scalar()
        
        # Create stats
        return ReviewStats(
            total=total,
            pending=status_counts.get(ReviewStatus.PENDING, 0),
            in_progress=status_counts.get(ReviewStatus.IN_PROGRESS, 0),
            approved=status_counts.get(ReviewStatus.APPROVED, 0),
            rejected=status_counts.get(ReviewStatus.REJECTED, 0),
            feedback_provided=status_counts.get(ReviewStatus.FEEDBACK_PROVIDED, 0),
            timed_out=status_counts.get(ReviewStatus.TIMED_OUT, 0),
            cancelled=status_counts.get(ReviewStatus.CANCELLED, 0),
            average_response_time_seconds=avg_response_time,
        )


# Initialize Redis client
def get_redis_client() -> RedisClient:
    """
    Get or initialize the Redis client.
    
    Returns:
        Redis client
    """
    global redis_client
    if redis_client is None:
        try:
            from backend.config import settings
            redis_client = RedisClient(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
            )
        except Exception as e:
            logger.error(f"Error initializing Redis client: {e}")
            # Create a dummy client for testing
            redis_client = RedisClient(
                host="localhost",
                port=6379,
            )
    
    return redis_client


# Background task for handling review timeouts
async def check_review_timeouts(db: Session):
    """
    Check for timed out reviews and handle them.
    
    Args:
        db: Database session
    """
    from backend.models.user import HITLReview
    
    # Get current time
    now = datetime.now().isoformat()
    
    # Find reviews that have timed out
    timed_out_reviews = db.query(HITLReview).filter(
        HITLReview.status == ReviewStatus.PENDING,
        HITLReview.timeout_at < now,
    ).all()
    
    # Handle each timed out review
    for db_review in timed_out_reviews:
        # Update status to timed out
        db_review.status = ReviewStatus.TIMED_OUT
        
        # Apply fallback action if specified
        fallback_action = db_review.fallback_action
        if fallback_action == "proceed":
            # Auto-approve the review
            from backend.models.user import HITLReviewResponse
            
            # Create auto-response
            db_response = HITLReviewResponse(
                review_id=db_review.id,
                user_id="system",
                timestamp=datetime.now().isoformat(),
                status=ReviewStatus.APPROVED,
                feedback="Automatically approved due to timeout",
            )
            
            db.add(db_response)
        
        elif fallback_action == "abort":
            # Auto-reject the review
            from backend.models.user import HITLReviewResponse
            
            # Create auto-response
            db_response = HITLReviewResponse(
                review_id=db_review.id,
                user_id="system",
                timestamp=datetime.now().isoformat(),
                status=ReviewStatus.REJECTED,
                feedback="Automatically rejected due to timeout",
            )
            
            db.add(db_response)
        
        # Commit changes
        db.commit()
        
        # Send notification
        review = ReviewRequest(
            id=db_review.id,
            crew_id=db_review.crew_id,
            task_id=db_review.task_id,
            agent_id=db_review.agent_id,
            user_id=db_review.user_id,
            timestamp=db_review.timestamp,
            task_name=db_review.task_name,
            task_output=json.loads(db_review.task_output),
            confidence=db_review.confidence,
            review_type=db_review.review_type,
            priority=db_review.priority,
            status=ReviewStatus.TIMED_OUT,
            timeout_minutes=db_review.timeout_minutes,
            timeout_at=db_review.timeout_at,
            fallback_action=db_review.fallback_action,
            template_id=db_review.template_id,
            form_fields=json.loads(db_review.form_fields) if db_review.form_fields else None,
        )
        
        await send_notifications(review)
        
        # Publish event
        publish_event("hitl_review_timeout", {
            "review_id": db_review.id,
            "crew_id": db_review.crew_id,
            "task_id": db_review.task_id,
            "fallback_action": fallback_action,
        })


# Function to send notifications
async def send_notifications(review: ReviewRequest) -> None:
    """
    Send notifications for a review.
    
    Args:
        review: Review request
    """
    # Send WebSocket notification
    notification = {
        "type": "review_update",
        "review_id": review.id,
        "crew_id": review.crew_id,
        "task_id": review.task_id,
        "status": review.status,
        "priority": review.priority,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Broadcast to all connected clients
    await manager.broadcast(notification)
    
    # Send webhook notification if configured
    webhook_url = None
    try:
        from backend.config import settings
        webhook_url = settings.HITL_WEBHOOK_URL
    except Exception:
        pass
    
    if webhook_url:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json=notification,
                    timeout=5.0,
                )
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    # Send email notification if configured
    email_recipients = None
    try:
        from backend.config import settings
        email_recipients = settings.HITL_EMAIL_RECIPIENTS
    except Exception:
        pass
    
    if email_recipients:
        try:
            # This would integrate with an email service
            logger.info(f"Would send email to {email_recipients}")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")


# Start background task for checking timeouts
@router.on_event("startup")
async def start_timeout_checker():
    """Start background task for checking timeouts."""
    async def timeout_checker():
        while True:
            try:
                # Get database session
                db = next(get_db())
                
                # Check timeouts
                await check_review_timeouts(db)
                
                # Close session
                db.close()
            except Exception as e:
                logger.error(f"Error checking review timeouts: {e}")
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute
    
    # Start background task
    asyncio.create_task(timeout_checker())


@router.post(
    "/reviews",
    response_model=ReviewRequest,
    status_code=status.HTTP_201_CREATED,
    summary="Create review request",
    description="Create a new HITL review request",
)
async def create_review(
    review: ReviewRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewRequest:
    """
    Create a new HITL review request.
    
    Args:
        review: Review request
        background_tasks: Background tasks
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created review
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:create")
    
    # Set user ID if not provided
    if not review.user_id:
        review.user_id = current_user.id
    
    # Create review in database
    created_review = HITLReviewRepository.create_review(db, review)
    
    # Send notifications in background
    background_tasks.add_task(send_notifications, created_review)
    
    # Publish event
    publish_event("hitl_review_created", {
        "review_id": created_review.id,
        "crew_id": created_review.crew_id,
        "task_id": created_review.task_id,
        "priority": created_review.priority,
        "user_id": current_user.id,
    })
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/reviews",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return created_review


@router.get(
    "/reviews/{review_id}",
    response_model=ReviewDetail,
    summary="Get review",
    description="Get a HITL review by ID",
)
async def get_review(
    review_id: str = Path(..., description="Review ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewDetail:
    """
    Get a HITL review by ID.
    
    Args:
        review_id: Review ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Review detail
        
    Raises:
        HTTPException: If review is not found
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:read")
    
    # Get review from database
    review = HITLReviewRepository.get_review(db, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review not found: {review_id}",
        )
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/hitl/reviews/{review_id}",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return review


@router.get(
    "/reviews",
    response_model=List[ReviewRequest],
    summary="List reviews",
    description="List HITL reviews with optional filtering",
)
async def list_reviews(
    status: Optional[ReviewStatus] = Query(None, description="Filter by status"),
    crew_id: Optional[str] = Query(None, description="Filter by crew ID"),
    priority: Optional[ReviewPriority] = Query(None, description="Filter by priority"),
    limit: int = Query(100, description="Maximum number of reviews to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ReviewRequest]:
    """
    List HITL reviews with optional filtering.
    
    Args:
        status: Filter by status
        crew_id: Filter by crew ID
        priority: Filter by priority
        limit: Maximum number of reviews to return
        offset: Offset for pagination
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        List of reviews
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:list")
    
    # Get reviews from database
    reviews = HITLReviewRepository.list_reviews(
        db=db,
        status=status,
        crew_id=crew_id,
        priority=priority,
        limit=limit,
        offset=offset,
    )
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/reviews",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return reviews


@router.post(
    "/reviews/{review_id}/respond",
    response_model=ReviewResponse,
    summary="Respond to review",
    description="Submit a response to a HITL review",
)
async def respond_to_review(
    response: ReviewResponse,
    review_id: str = Path(..., description="Review ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewResponse:
    """
    Submit a response to a HITL review.
    
    Args:
        response: Review response
        review_id: Review ID
        background_tasks: Background tasks
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created response
        
    Raises:
        HTTPException: If review is not found or already responded to
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:respond")
    
    # Override review ID and user ID
    response.review_id = review_id
    response.user_id = current_user.id
    
    # Check if review exists
    review_detail = HITLReviewRepository.get_review(db, review_id)
    if not review_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review not found: {review_id}",
        )
    
    # Check if review already has a response
    if review_detail.response:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Review already has a response: {review_id}",
        )
    
    # Create response in database
    created_response = HITLReviewRepository.create_review_response(db, response)
    
    # Send notifications in background
    review = review_detail.request
    review.status = response.status
    background_tasks.add_task(send_notifications, review)
    
    # Resume agent execution if approved
    if response.status == ReviewStatus.APPROVED:
        # This would integrate with the agent execution system
        # For now, just publish an event
        publish_event("hitl_review_approved", {
            "review_id": review_id,
            "crew_id": review.crew_id,
            "task_id": review.task_id,
            "user_id": current_user.id,
        })
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/hitl/reviews/{review_id}/respond",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return created_response


@router.get(
    "/stats",
    response_model=ReviewStats,
    summary="Get review statistics",
    description="Get statistics about HITL reviews",
)
async def get_stats(
    crew_id: Optional[str] = Query(None, description="Filter by crew ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewStats:
    """
    Get statistics about HITL reviews.
    
    Args:
        crew_id: Filter by crew ID
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Review statistics
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:stats")
    
    # Get stats from database
    stats = HITLReviewRepository.get_review_stats(db, crew_id)
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/stats",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return stats


@router.post(
    "/pause",
    status_code=status.HTTP_200_OK,
    summary="Pause agent execution",
    description="Pause agent execution for manual review",
)
async def pause_execution(
    request: PauseRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Pause agent execution for manual review.
    
    Args:
        request: Pause request
        current_user: Current authenticated user
        
    Returns:
        Pause status
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:pause")
    
    # This would integrate with the agent execution system
    # For now, just publish an event
    publish_event("hitl_pause_execution", {
        "crew_id": request.crew_id,
        "task_id": request.task_id,
        "agent_id": request.agent_id,
        "reason": request.reason,
        "user_id": current_user.id,
    })
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/pause",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return {
        "status": "paused",
        "crew_id": request.crew_id,
        "task_id": request.task_id,
        "agent_id": request.agent_id,
        "timestamp": datetime.now().isoformat(),
    }


@router.post(
    "/resume",
    status_code=status.HTTP_200_OK,
    summary="Resume agent execution",
    description="Resume previously paused agent execution",
)
async def resume_execution(
    request: ResumeRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Resume previously paused agent execution.
    
    Args:
        request: Resume request
        current_user: Current authenticated user
        
    Returns:
        Resume status
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:resume")
    
    # This would integrate with the agent execution system
    # For now, just publish an event
    publish_event("hitl_resume_execution", {
        "crew_id": request.crew_id,
        "task_id": request.task_id,
        "agent_id": request.agent_id,
        "user_id": current_user.id,
    })
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/resume",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return {
        "status": "resumed",
        "crew_id": request.crew_id,
        "task_id": request.task_id,
        "agent_id": request.agent_id,
        "timestamp": datetime.now().isoformat(),
    }


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
        user_id: User ID
    """
    await manager.connect(websocket, user_id)
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            
            # Process message
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    })
                
                elif message.get("type") == "subscribe":
                    # Subscribe to specific reviews or crews
                    # This would be implemented with Redis pub/sub
                    pass
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type",
                        "timestamp": datetime.now().isoformat(),
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.now().isoformat(),
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


@router.get(
    "/templates",
    response_model=List[ReviewTemplate],
    summary="List review templates",
    description="List available review templates",
)
async def list_templates(
    current_user: User = Depends(get_current_user),
) -> List[ReviewTemplate]:
    """
    List available review templates.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of review templates
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:read")
    
    # This would normally load templates from a database or file system
    # For now, return some example templates
    templates = [
        ReviewTemplate(
            id="fraud_review",
            name="Fraud Review Template",
            description="Template for reviewing fraud detection results",
            fields=[
                {
                    "name": "confidence_score",
                    "label": "Confidence Score",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "required": True,
                },
                {
                    "name": "feedback",
                    "label": "Feedback",
                    "type": "textarea",
                    "required": False,
                },
            ],
        ),
        ReviewTemplate(
            id="pattern_review",
            name="Pattern Review Template",
            description="Template for reviewing pattern detection results",
            fields=[
                {
                    "name": "pattern_accuracy",
                    "label": "Pattern Accuracy",
                    "type": "select",
                    "options": ["High", "Medium", "Low"],
                    "required": True,
                },
                {
                    "name": "false_positives",
                    "label": "False Positives",
                    "type": "number",
                    "min": 0,
                    "required": True,
                },
                {
                    "name": "comments",
                    "label": "Comments",
                    "type": "textarea",
                    "required": False,
                },
            ],
        ),
    ]
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/templates",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return templates


@router.get(
    "/templates/{template_id}",
    response_model=ReviewTemplate,
    summary="Get review template",
    description="Get a review template by ID",
)
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    current_user: User = Depends(get_current_user),
) -> ReviewTemplate:
    """
    Get a review template by ID.
    
    Args:
        template_id: Template ID
        current_user: Current authenticated user
        
    Returns:
        Review template
        
    Raises:
        HTTPException: If template is not found
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:read")
    
    # This would normally load the template from a database or file system
    # For now, return an example template
    if template_id == "fraud_review":
        template = ReviewTemplate(
            id="fraud_review",
            name="Fraud Review Template",
            description="Template for reviewing fraud detection results",
            fields=[
                {
                    "name": "confidence_score",
                    "label": "Confidence Score",
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "required": True,
                },
                {
                    "name": "feedback",
                    "label": "Feedback",
                    "type": "textarea",
                    "required": False,
                },
            ],
        )
    elif template_id == "pattern_review":
        template = ReviewTemplate(
            id="pattern_review",
            name="Pattern Review Template",
            description="Template for reviewing pattern detection results",
            fields=[
                {
                    "name": "pattern_accuracy",
                    "label": "Pattern Accuracy",
                    "type": "select",
                    "options": ["High", "Medium", "Low"],
                    "required": True,
                },
                {
                    "name": "false_positives",
                    "label": "False Positives",
                    "type": "number",
                    "min": 0,
                    "required": True,
                },
                {
                    "name": "comments",
                    "label": "Comments",
                    "type": "textarea",
                    "required": False,
                },
            ],
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/hitl/templates/{template_id}",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return template


@router.post(
    "/notify",
    status_code=status.HTTP_200_OK,
    summary="Send notification",
    description="Send a notification to HITL reviewers",
)
async def send_notification(
    message: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a notification to HITL reviewers.
    
    Args:
        message: Notification message
        current_user: Current authenticated user
        
    Returns:
        Notification status
    """
    # Verify permissions
    verify_permissions(current_user, "hitl:notify")
    
    # Add timestamp and sender
    notification = {
        **message,
        "timestamp": datetime.now().isoformat(),
        "sender": current_user.id,
    }
    
    # Broadcast to all connected clients
    await manager.broadcast(notification)
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/hitl/notify",
        func=lambda: None,
        environment="development",
        version="1.8.0-beta",
    )()
    
    return {
        "status": "sent",
        "timestamp": notification["timestamp"],
    }


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="HITL webhook",
    description="Webhook endpoint for external HITL integrations",
)
async def hitl_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Webhook endpoint for external HITL integrations.
    
    Args:
        request: HTTP request
        background_tasks: Background tasks
        db: Database session
        
    Returns:
        Webhook status
    """
    # Parse webhook payload
    try:
        payload = await request.json()
        
        # Check for required fields
        if "type" not in payload:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Missing type field in webhook payload"},
            )
        
        # Handle different webhook types
        if payload["type"] == "review_request":
            # Create review request
            if "review" not in payload:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Missing review field in webhook payload"},
                )
            
            # Create review
            review_data = payload["review"]
            review = ReviewRequest(**review_data)
            
            # Save to database
            created_review = HITLReviewRepository.create_review(db, review)
            
            # Send notifications in background
            background_tasks.add_task(send_notifications, created_review)
            
            return {
                "status": "created",
                "review_id": created_review.id,
            }
        
        elif payload["type"] == "review_response":
            # Process review response
            if "response" not in payload:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Missing response field in webhook payload"},
                )
            
            # Create response
            response_data = payload["response"]
            response = ReviewResponse(**response_data)
            
            # Save to database
            created_response = HITLReviewRepository.create_review_response(db, response)
            
            # Get review
            review_detail = HITLReviewRepository.get_review(db, response.review_id)
            if review_detail:
                # Send notifications in background
                review = review_detail.request
                review.status = response.status
                background_tasks.add_task(send_notifications, review)
            
            return {
                "status": "processed",
                "review_id": response.review_id,
            }
        
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": f"Unknown webhook type: {payload['type']}"},
            )
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Error processing webhook: {str(e)}"},
        )


# Helper function for integrating with the crew execution system
def create_hitl_callback(db: Session) -> callable:
    """
    Create a HITL callback function for the crew execution system.
    
    Args:
        db: Database session
        
    Returns:
        HITL callback function
    """
    async def hitl_callback(review_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        HITL callback function for the crew execution system.
        
        Args:
            review_request: Review request data
            
        Returns:
            Review response data
        """
        # Create review request
        review = ReviewRequest(**review_request)
        
        # Save to database
        created_review = HITLReviewRepository.create_review(db, review)
        
        # Send notifications
        await send_notifications(created_review)
        
        # Wait for response
        max_wait_seconds = 300  # 5 minutes
        poll_interval_seconds = 2
        waited_seconds = 0
        
        while waited_seconds < max_wait_seconds:
            # Check if review has a response
            review_detail = HITLReviewRepository.get_review(db, created_review.id)
            if review_detail and review_detail.response:
                # Return response
                return {
                    "status": review_detail.response.status,
                    "feedback": review_detail.response.feedback,
                    "edited_output": review_detail.response.edited_output,
                }
            
            # Wait before checking again
            await asyncio.sleep(poll_interval_seconds)
            waited_seconds += poll_interval_seconds
        
        # Timed out waiting for response
        # Apply fallback action
        fallback_action = review.fallback_action or "proceed"
        
        if fallback_action == "proceed":
            # Auto-approve
            return {
                "status": ReviewStatus.APPROVED,
                "feedback": "Automatically approved due to timeout",
            }
        else:
            # Auto-reject
            return {
                "status": ReviewStatus.REJECTED,
                "feedback": "Automatically rejected due to timeout",
            }
    
    return hitl_callback
