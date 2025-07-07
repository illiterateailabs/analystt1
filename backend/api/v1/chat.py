"""Chat API endpoints for natural language interaction."""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File
from pydantic import BaseModel, Field
import base64

# Import SlowAPI for rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.database import get_db  # DB session dependency
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.models.conversation import Conversation, Message
from fastapi import Query

# OpenTelemetry tracing
from backend.core.telemetry import trace

logger = logging.getLogger(__name__)
router = APIRouter()

from datetime import datetime
import uuid

# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    # Hard-limit message length to mitigate prompt-injection / DoS
    message: str = Field(
        ...,
        description="User message",
        min_length=1,
        max_length=2_000,
    )
    context: Optional[str] = Field(
        None,
        description="Additional context that should be passed to the LLM",
        max_length=4_000,
    )
    # Validate that any supplied conversation_id is a proper UUID
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation ID for context (UUID v4)",
        regex=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
    )
    include_graph_data: bool = Field(
        False,
        description="Set to true if the client wants the LLM to use graph context",
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    cypher_query: Optional[str] = Field(None, description="Generated Cypher query if applicable")
    graph_results: Optional[List[Dict[str, Any]]] = Field(None, description="Graph query results")
    execution_details: Optional[Dict[str, Any]] = Field(None, description="Code execution details")


class ImageAnalysisRequest(BaseModel):
    prompt: str = Field(default="Analyze this image and describe what you see.")
    extract_entities: bool = Field(False, description="Extract entities for graph storage")


class ImageAnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Image analysis result")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted entities")
    graph_updates: Optional[Dict[str, Any]]] = Field(None, description="Graph database updates")


# Dependency functions
async def get_gemini_client(request: Request) -> GeminiClient:
    """Get Gemini client from app state."""
    return request.app.state.gemini


async def get_neo4j_client(request: Request) -> Neo4jClient:
    """Get Neo4j client from app state."""
    # During unit tests or early startup the client might not exist.
    return getattr(request.app.state, "neo4j", None)


async def get_e2b_client(request: Request) -> E2BClient:
    """Get e2b client from app state."""
    return request.app.state.e2b


async def get_limiter(request: Request) -> Limiter:
    """Get the global rate limiter from app state."""
    return request.app.state.limiter


@router.post("/message", response_model=ChatResponse)
@trace()  # OpenTelemetry span for full request handling
# Add stricter rate limit for chat endpoint: 5 requests per 10 seconds
async def send_message(
    request: ChatRequest,
    req: Request,
    gemini: GeminiClient = Depends(get_gemini_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    e2b: E2BClient = Depends(get_e2b_client),
    db: AsyncSession = Depends(get_db),
    limiter: Limiter = Depends(get_limiter),
):
    """Process a chat message and return AI response."""
    # Apply rate limiting at runtime
    await limiter.check_request(
        req, 
        key_func=get_remote_address, 
        rate="5/10seconds"  # Stricter limit: 5 requests per 10 seconds
    )
    
    try:
        logger.info(f"Processing chat message: {request.message[:100]}...")
        
        # ------------------------------------------------------------------ #
        # Ensure conversation exists (DB persistence)
        # ------------------------------------------------------------------ #
        conversation: Optional[Conversation] = None
        conversation_id: str

        if request.conversation_id:
            # Fetch existing conversation
            stmt = select(Conversation).where(Conversation.id == request.conversation_id)
            result = await db.execute(stmt)
            conversation = result.scalar_one_or_none()

        if conversation is None:
            # Create new conversation (assign title later if needed)
            conversation = Conversation(
                title="New Conversation",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(conversation)
            await db.flush()  # Populate conversation.id

        conversation_id = conversation.id

        response_data = {
            "conversation_id": conversation_id,
            "response": "",
            "cypher_query": None,
            "graph_results": None,
            "execution_details": None
        }
        
        # Check if the message requires graph database interaction
        if request.include_graph_data or any(keyword in request.message.lower() 
                                           for keyword in ["find", "search", "query", "show", "list", "analyze"]):
            
            # Get schema context
            schema_info = await neo4j.get_schema_info()
            schema_context = f"""
Graph Database Schema:
- Node Labels: {', '.join(schema_info['labels'])}
- Relationship Types: {', '.join(schema_info['relationship_types'])}
- Property Keys: {', '.join(schema_info['property_keys'])}
- Total Nodes: {schema_info['node_count']}
- Total Relationships: {schema_info['relationship_count']}
"""
            
            # Check if this looks like a query that needs Cypher
            if any(keyword in request.message.lower() 
                   for keyword in ["find", "search", "show", "list", "count", "match"]):
                
                # Generate Cypher query
                cypher_query = await gemini.generate_cypher_query(
                    request.message,
                    schema_context
                )
                
                response_data["cypher_query"] = cypher_query
                
                # Execute the query
                try:
                    graph_results = await neo4j.execute_query(cypher_query)
                    response_data["graph_results"] = graph_results
                    
                    # Generate explanation of results
                    explanation = await gemini.explain_results(
                        request.message,
                        graph_results,
                        context=request.context
                    )
                    response_data["response"] = explanation
                    
                except Exception as e:
                    logger.error(f"Error executing Cypher query: {e}")
                    response_data["response"] = f"I generated a query but encountered an error executing it: {str(e)}"
            
            else:
                # General question with graph context
                full_context = schema_context
                if request.context:
                    full_context += f"\n\nAdditional Context: {request.context}"
                
                response = await gemini.generate_text(
                    request.message,
                    context=full_context
                )
                response_data["response"] = response
        
        else:
            # Simple text generation without graph context
            response = await gemini.generate_text(
                request.message,
                context=request.context
            )
            response_data["response"] = response
        
        logger.info("Chat message processed successfully")

        # ------------------------------------------------------------------ #
        # Persist messages to DB
        # ------------------------------------------------------------------ #
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            timestamp=datetime.utcnow(),
        )
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_data["response"],
            timestamp=datetime.utcnow(),
        )
        db.add_all([user_msg, assistant_msg])

        # Update conversation.updated_at
        conversation.updated_at = datetime.utcnow()

        # Commit transaction
        await db.commit()

        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        # Rollback DB changes on error
        try:
            await db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
@trace()  # Trace image-analysis pipeline
# Add stricter rate limit for image analysis endpoint: 3 requests per 30 seconds
async def analyze_image(
    file: UploadFile = File(...),
    request: ImageAnalysisRequest = ImageAnalysisRequest(),
    req: Request = Request,
    gemini: GeminiClient = Depends(get_gemini_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    limiter: Limiter = Depends(get_limiter),
):
    """Analyze an uploaded image using Gemini's vision capabilities."""
    # Apply rate limiting at runtime
    await limiter.check_request(
        req, 
        key_func=get_remote_address, 
        rate="3/30seconds"  # Even stricter limit for image analysis: 3 requests per 30 seconds
    )
    
    try:
        logger.info(f"Analyzing image: {file.filename}")
        
        # Read image data
        image_data = await file.read()
        
        # Analyze image with Gemini
        analysis = await gemini.analyze_image(image_data, request.prompt)
        
        response_data = {
            "analysis": analysis,
            "entities": None,
            "graph_updates": None
        }
        
        # Extract entities if requested
        if request.extract_entities:
            entity_extraction_prompt = f"""
Based on this image analysis: {analysis}

Extract structured entities that could be stored in a graph database. 
Return a JSON list of entities with the following format:
[
    {{
        "type": "entity_type",
        "name": "entity_name", 
        "properties": {{"key": "value"}},
        "relationships": [
            {{"type": "relationship_type", "target": "target_entity"}}
        ]
    }}
]

Focus on entities like people, organizations, locations, documents, transactions, etc.
"""
            
            entities_response = await gemini.generate_text(entity_extraction_prompt)
            
            # Try to parse entities (basic implementation)
            try:
                import json
                entities = json.loads(entities_response)
                response_data["entities"] = entities
                
                # Store entities in graph database
                graph_updates = await _store_entities_in_graph(entities, neo4j)
                response_data["graph_updates"] = graph_updates
                
            except json.JSONDecodeError:
                logger.warning("Could not parse extracted entities as JSON")
                response_data["entities"] = []
        
        logger.info("Image analysis completed successfully")
        return ImageAnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Conversation management endpoints (DB-backed)
# --------------------------------------------------------------------------- #

async def _get_conversation_or_404(conversation_id: str, db: AsyncSession) -> Conversation:
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    return conversation


@router.get("/conversation/{conversation_id}")
# Add moderate rate limit for conversation retrieval: 20 requests per minute
async def get_conversation(
    conversation_id: str, 
    req: Request,
    db: AsyncSession = Depends(get_db),
    limiter: Limiter = Depends(get_limiter),
):
    """Retrieve a conversation from the database."""
    # Apply rate limiting at runtime
    await limiter.check_request(
        req, 
        key_func=get_remote_address, 
        rate="20/minute"  # More lenient limit for read operations
    )
    
    conversation = await _get_conversation_or_404(conversation_id, db)
    return conversation.to_dict()


@router.delete("/conversation/{conversation_id}")
# Add moderate rate limit for conversation deletion: 10 requests per minute
async def delete_conversation(
    conversation_id: str, 
    req: Request,
    db: AsyncSession = Depends(get_db),
    limiter: Limiter = Depends(get_limiter),
):
    """Delete a conversation and its messages from the database."""
    # Apply rate limiting at runtime
    await limiter.check_request(
        req, 
        key_func=get_remote_address, 
        rate="10/minute"  # Moderate limit for deletion operations
    )
    
    conversation = await _get_conversation_or_404(conversation_id, db)
    try:
        await db.delete(conversation)
        await db.commit()
        return {"success": True, "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


# --------------------------------------------------------------------------- #
# List conversations (paginated)
# --------------------------------------------------------------------------- #


def _conversation_meta(convo: Conversation) -> Dict[str, Any]:
    """Return lightweight conversation metadata for listings."""
    return {
        "conversation_id": convo.id,
        "title": convo.title,
        "created_at": convo.created_at.isoformat(),
        "updated_at": convo.updated_at.isoformat(),
        "message_count": len(convo.messages),
        "is_active": convo.is_active,
        "user_id": convo.user_id,
    }


@router.get("/conversations")
# Add moderate rate limit for listing conversations: 20 requests per minute
async def list_conversations(
    limit: int = Query(20, ge=1, le=100, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Items to skip"),
    user_id: Optional[str] = Query(
        None,
        regex=r"^[0-9a-fA-F\-]{36}$",
        description="Filter by owner user_id (UUID)",
    ),
    req: Request = Request,
    db: AsyncSession = Depends(get_db),
    limiter: Limiter = Depends(get_limiter),
):
    """
    List conversations with optional **pagination** and **user filtering**.

    Results are ordered by `created_at` descending.
    """
    # Apply rate limiting at runtime
    await limiter.check_request(
        req, 
        key_func=get_remote_address, 
        rate="20/minute"  # More lenient limit for read operations
    )
    
    try:
        base_stmt = select(Conversation)
        if user_id:
            base_stmt = base_stmt.where(Conversation.user_id == user_id)

        # Get total count for pagination
        total_stmt = base_stmt.with_only_columns(func.count()).order_by(None)
        total_result = await db.execute(total_stmt)
        total: int = total_result.scalar_one()

        # Apply ordering, limit, offset
        stmt = (
            base_stmt.order_by(Conversation.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        conversations = result.scalars().unique().all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "conversations": [_conversation_meta(c) for c in conversations],
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")


async def _store_entities_in_graph(entities: List[Dict[str, Any]], neo4j: Neo4jClient) -> Dict[str, Any]:
    """Store extracted entities in the graph database."""
    try:
        created_nodes = 0
        created_relationships = 0
        
        for entity in entities:
            # Create node
            node_labels = [entity.get("type", "Entity")]
            node_properties = {
                "name": entity.get("name", "Unknown"),
                **entity.get("properties", {})
            }
            
            await neo4j.create_node(node_labels, node_properties)
            created_nodes += 1
            
            # TODO: Create relationships (requires more complex logic)
            # This would need to find existing nodes and create relationships
        
        return {
            "created_nodes": created_nodes,
            "created_relationships": created_relationships,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error storing entities in graph: {e}")
        return {
            "created_nodes": 0,
            "created_relationships": 0,
            "status": "error",
            "error": str(e)
        }
