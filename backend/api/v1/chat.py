"""Chat API endpoints for natural language interaction."""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File
from pydantic import BaseModel, Field
import base64

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    context: Optional[str] = Field(None, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    include_graph_data: bool = Field(False, description="Include graph database context")


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
    graph_updates: Optional[Dict[str, Any]] = Field(None, description="Graph database updates")


# Dependency functions
async def get_gemini_client(request: Request) -> GeminiClient:
    """Get Gemini client from app state."""
    return request.app.state.gemini


async def get_neo4j_client(request: Request) -> Neo4jClient:
    """Get Neo4j client from app state."""
    return request.app.state.neo4j


async def get_e2b_client(request: Request) -> E2BClient:
    """Get e2b client from app state."""
    return request.app.state.e2b


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    e2b: E2BClient = Depends(get_e2b_client)
):
    """Process a chat message and return AI response."""
    try:
        logger.info(f"Processing chat message: {request.message[:100]}...")
        
        response_data = {
            "conversation_id": request.conversation_id or "default",
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
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    request: ImageAnalysisRequest = ImageAnalysisRequest(),
    gemini: GeminiClient = Depends(get_gemini_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Analyze an uploaded image using Gemini's vision capabilities."""
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


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history (placeholder for future implementation)."""
    # TODO: Implement conversation storage and retrieval
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "created_at": None,
        "updated_at": None
    }


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation (placeholder for future implementation)."""
    # TODO: Implement conversation deletion
    return {"message": f"Conversation {conversation_id} deleted"}


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
