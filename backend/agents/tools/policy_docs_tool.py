"""
PolicyDocsTool for retrieving AML policy and regulatory information using vector search.

This tool provides CrewAI agents with access to policy documents and regulatory
information related to Anti-Money Laundering (AML), Know Your Customer (KYC),
and other financial compliance requirements. It implements a RAG (Retrieval
Augmented Generation) system using Gemini embeddings and Redis vector storage.
"""

import json
import logging
import uuid
import os
from typing import Any, Dict, List, Optional, Union, Tuple
import re
from datetime import datetime

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.config import settings

# Import Redis with vector search capabilities
try:
    import redis
    from redis.commands.search.field import TextField, TagField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis vector search not available, falling back to basic search")

logger = logging.getLogger(__name__)


class PolicyDocument(BaseModel):
    """Model for a policy document."""
    
    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    content: str = Field(..., description="Full content of the document")
    type: str = Field(..., description="Type of document (e.g., 'aml', 'kyc', 'sanctions')")
    source: Optional[str] = Field(None, description="Source of the document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Model for a chunk of a policy document."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    title: str = Field(..., description="Title of the parent document")
    content: str = Field(..., description="Content of this chunk")
    type: str = Field(..., description="Type of document")
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyQueryInput(BaseModel):
    """Input model for policy document queries."""
    
    query: str = Field(
        ...,
        description="The query or topic to search for in policy documents"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document to search (e.g., 'aml', 'kyc', 'sanctions')"
    )
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of results to return"
    )
    similarity_threshold: Optional[float] = Field(
        default=0.7,
        description="Minimum similarity score (0-1) for results"
    )


class PolicyDocumentInput(BaseModel):
    """Input model for adding or updating policy documents."""
    
    title: str = Field(..., description="Title of the document")
    content: str = Field(..., description="Full content of the document")
    type: str = Field(..., description="Type of document (e.g., 'aml', 'kyc', 'sanctions')")
    source: Optional[str] = Field(None, description="Source of the document")
    document_id: Optional[str] = Field(None, description="Document ID (for updates)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PolicyDocsTool(BaseTool):
    """
    Tool for retrieving AML policy and regulatory information using vector search.
    
    This tool allows agents to query policy documents and regulatory information
    related to Anti-Money Laundering (AML), Know Your Customer (KYC), and other
    financial compliance requirements. It implements a RAG (Retrieval Augmented
    Generation) system using Gemini embeddings and Redis vector storage.
    """
    
    name: str = "policy_docs_tool"
    description: str = """
    Retrieve policy and regulatory information for financial compliance using semantic search.
    
    Use this tool when you need to:
    - Find specific AML regulatory requirements
    - Check KYC compliance procedures
    - Retrieve information about sanctions and watchlists
    - Get guidance on Suspicious Activity Report (SAR) filing
    - Access regulatory definitions and thresholds
    
    Example queries:
    - "What are the requirements for filing a SAR?"
    - "What is the definition of a Politically Exposed Person (PEP)?"
    - "What are the record-keeping requirements for transaction monitoring?"
    - "What are the thresholds for mandatory reporting of cash transactions?"
    """
    args_schema: type[BaseModel] = PolicyQueryInput
    
    # Vector dimension for Gemini embeddings
    VECTOR_DIM = 768
    
    # Redis index name
    INDEX_NAME = "policy_docs_idx"
    
    # Prefix for document keys in Redis
    DOC_PREFIX = "policy:doc:"
    CHUNK_PREFIX = "policy:chunk:"
    
    # Chunk size for document splitting
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    def __init__(
        self, 
        gemini_client: Optional[GeminiClient] = None,
        redis_url: Optional[str] = None,
        initialize_default_docs: bool = True
    ):
        """
        Initialize the PolicyDocsTool.
        
        Args:
            gemini_client: Optional GeminiClient instance. If not provided,
                          a new client will be created.
            redis_url: Optional Redis URL. If not provided, will use settings.
            initialize_default_docs: Whether to initialize default policy documents.
        """
        super().__init__()
        self.gemini_client = gemini_client or GeminiClient()
        
        # Initialize Redis client if available
        self.redis_client = None
        self.vector_search_available = False
        
        if REDIS_AVAILABLE:
            try:
                redis_url = redis_url or settings.redis_url
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=5.0
                )
                # Test connection
                self.redis_client.ping()
                
                # Initialize vector index
                self._initialize_vector_index()
                self.vector_search_available = True
                logger.info("Redis vector search initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis vector search: {e}")
                self.redis_client = None
                self.vector_search_available = False
        
        # Fallback in-memory policy documents
        self.policy_docs = {}
        
        # Initialize default policy documents if requested
        if initialize_default_docs:
            self._initialize_default_docs()
    
    def _initialize_vector_index(self):
        """Initialize the vector search index in Redis."""
        if not self.redis_client:
            return
        
        try:
            # Check if index already exists
            try:
                self.redis_client.ft(self.INDEX_NAME).info()
                logger.info(f"Vector index {self.INDEX_NAME} already exists")
                return
            except:
                # Index doesn't exist, create it
                pass
            
            # Define schema for the index
            schema = [
                TextField("title"),
                TextField("content"),
                TagField("doc_id"),
                TagField("type"),
                TagField("chunk_index"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                )
            ]
            
            # Create the index
            self.redis_client.ft(self.INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.CHUNK_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            logger.info(f"Created vector index {self.INDEX_NAME}")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            self.vector_search_available = False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Gemini.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as list of floats
        """
        try:
            # Use Gemini to generate embedding
            embedding = await self.gemini_client.generate_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return empty embedding as fallback
            return [0.0] * self.VECTOR_DIM
    
    def _chunk_document(self, document: PolicyDocument) -> List[DocumentChunk]:
        """
        Split a document into chunks for embedding.
        
        Args:
            document: Policy document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        chunks = []
        
        # Simple chunking by character count with overlap
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position with potential overlap
            end = min(start + self.CHUNK_SIZE, len(content))
            
            # If not at the end, try to find a good break point
            if end < len(content):
                # Look for paragraph break
                paragraph_break = content.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + self.CHUNK_SIZE // 2:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = content.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + self.CHUNK_SIZE // 2:
                        end = sentence_break + 2
                    else:
                        # Look for word break
                        word_break = content.rfind(" ", start, end)
                        if word_break != -1 and word_break > start + self.CHUNK_SIZE // 2:
                            end = word_break + 1
            
            # Create chunk
            chunk_content = content[start:end]
            chunk_id = f"{document.id}_chunk_{chunk_index}"
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                title=document.title,
                content=chunk_content,
                type=document.type,
                chunk_index=chunk_index,
                metadata={
                    "source": document.metadata.get("source"),
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                }
            ))
            
            # Move to next chunk with overlap
            start = end - self.CHUNK_OVERLAP if end < len(content) else end
            chunk_index += 1
        
        return chunks
    
    async def add_document(self, document_input: PolicyDocumentInput) -> str:
        """
        Add a new policy document to the store.
        
        Args:
            document_input: Document to add
            
        Returns:
            Document ID
        """
        # Create document ID if not provided
        doc_id = document_input.document_id or f"doc_{uuid.uuid4()}"
        
        # Create document
        document = PolicyDocument(
            id=doc_id,
            title=document_input.title,
            content=document_input.content,
            type=document_input.type,
            source=document_input.source,
            metadata=document_input.metadata or {}
        )
        
        # Store in memory (fallback)
        self.policy_docs[doc_id] = document.dict()
        
        # If vector search is available, store in Redis
        if self.vector_search_available and self.redis_client:
            try:
                # Store full document
                doc_key = f"{self.DOC_PREFIX}{doc_id}"
                self.redis_client.hset(
                    doc_key,
                    mapping={
                        "id": doc_id,
                        "title": document.title,
                        "content": document.content,
                        "type": document.type,
                        "source": document.source or "",
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
                        "metadata": json.dumps(document.metadata),
                    }
                )
                
                # Chunk document
                chunks = self._chunk_document(document)
                
                # Generate embeddings and store chunks
                for chunk in chunks:
                    # Generate embedding
                    chunk.embedding = await self._generate_embedding(chunk.content)
                    
                    # Store chunk with embedding
                    chunk_key = f"{self.CHUNK_PREFIX}{chunk.id}"
                    self.redis_client.hset(
                        chunk_key,
                        mapping={
                            "id": chunk.id,
                            "doc_id": chunk.document_id,
                            "title": chunk.title,
                            "content": chunk.content,
                            "type": chunk.type,
                            "chunk_index": str(chunk.chunk_index),
                            "embedding": json.dumps(chunk.embedding),
                            "metadata": json.dumps(chunk.metadata),
                        }
                    )
                
                logger.info(f"Added document {doc_id} with {len(chunks)} chunks to Redis")
            except Exception as e:
                logger.error(f"Error adding document to Redis: {e}")
        
        return doc_id
    
    async def update_document(self, document_input: PolicyDocumentInput) -> bool:
        """
        Update an existing policy document.
        
        Args:
            document_input: Document to update
            
        Returns:
            Success flag
        """
        if not document_input.document_id:
            raise ValueError("Document ID is required for updates")
        
        doc_id = document_input.document_id
        
        # Check if document exists
        if doc_id not in self.policy_docs and not (
            self.vector_search_available and 
            self.redis_client and 
            self.redis_client.exists(f"{self.DOC_PREFIX}{doc_id}")
        ):
            return False
        
        # Update in-memory version
        if doc_id in self.policy_docs:
            self.policy_docs[doc_id].update({
                "title": document_input.title,
                "content": document_input.content,
                "type": document_input.type,
                "source": document_input.source,
                "updated_at": datetime.utcnow().isoformat(),
                "metadata": document_input.metadata or self.policy_docs[doc_id].get("metadata", {})
            })
        
        # If vector search is available, update in Redis
        if self.vector_search_available and self.redis_client:
            try:
                # Delete existing chunks
                existing_chunks = self.redis_client.keys(f"{self.CHUNK_PREFIX}{doc_id}_chunk_*")
                if existing_chunks:
                    self.redis_client.delete(*existing_chunks)
                
                # Create updated document
                document = PolicyDocument(
                    id=doc_id,
                    title=document_input.title,
                    content=document_input.content,
                    type=document_input.type,
                    source=document_input.source,
                    created_at=datetime.fromisoformat(
                        self.policy_docs.get(doc_id, {}).get(
                            "created_at", 
                            datetime.utcnow().isoformat()
                        )
                    ),
                    updated_at=datetime.utcnow(),
                    metadata=document_input.metadata or self.policy_docs.get(doc_id, {}).get("metadata", {})
                )
                
                # Update full document
                doc_key = f"{self.DOC_PREFIX}{doc_id}"
                self.redis_client.hset(
                    doc_key,
                    mapping={
                        "id": doc_id,
                        "title": document.title,
                        "content": document.content,
                        "type": document.type,
                        "source": document.source or "",
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
                        "metadata": json.dumps(document.metadata),
                    }
                )
                
                # Chunk document and store with embeddings
                chunks = self._chunk_document(document)
                
                # Generate embeddings and store chunks
                for chunk in chunks:
                    # Generate embedding
                    chunk.embedding = await self._generate_embedding(chunk.content)
                    
                    # Store chunk with embedding
                    chunk_key = f"{self.CHUNK_PREFIX}{chunk.id}"
                    self.redis_client.hset(
                        chunk_key,
                        mapping={
                            "id": chunk.id,
                            "doc_id": chunk.document_id,
                            "title": chunk.title,
                            "content": chunk.content,
                            "type": chunk.type,
                            "chunk_index": str(chunk.chunk_index),
                            "embedding": json.dumps(chunk.embedding),
                            "metadata": json.dumps(chunk.metadata),
                        }
                    )
                
                logger.info(f"Updated document {doc_id} with {len(chunks)} chunks in Redis")
                return True
            except Exception as e:
                logger.error(f"Error updating document in Redis: {e}")
                return False
        
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a policy document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Success flag
        """
        # Remove from in-memory store
        if document_id in self.policy_docs:
            del self.policy_docs[document_id]
        
        # If vector search is available, delete from Redis
        if self.vector_search_available and self.redis_client:
            try:
                # Delete document
                doc_key = f"{self.DOC_PREFIX}{document_id}"
                self.redis_client.delete(doc_key)
                
                # Delete chunks
                chunk_keys = self.redis_client.keys(f"{self.CHUNK_PREFIX}{document_id}_chunk_*")
                if chunk_keys:
                    self.redis_client.delete(*chunk_keys)
                
                logger.info(f"Deleted document {document_id} from Redis")
                return True
            except Exception as e:
                logger.error(f"Error deleting document from Redis: {e}")
                return False
        
        return True
    
    async def _vector_search(
        self,
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query text
            document_type: Optional document type filter
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching chunks with similarity scores
        """
        if not self.vector_search_available or not self.redis_client:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Prepare Redis query
            redis_query = f"*"
            
            # Add type filter if specified
            if document_type:
                redis_query = f"@type:{{{document_type}}}"
            
            # Create vector query
            vector_query = Query(redis_query)\
                .dialect(2)\
                .return_fields("id", "doc_id", "title", "content", "type", "chunk_index")\
                .sort_by("__embedding_score")\
                .vectorize("embedding", query_embedding)\
                .limit(0, max_results * 2)  # Get more results for re-ranking
            
            # Execute search
            results = self.redis_client.ft(self.INDEX_NAME).search(vector_query)
            
            # Process results
            chunks = []
            for result in results.docs:
                # Calculate similarity score (1 - distance for cosine)
                similarity = 1 - float(result.__embedding_score)
                
                # Skip results below threshold
                if similarity < similarity_threshold:
                    continue
                
                chunks.append({
                    "id": result.id,
                    "doc_id": result.doc_id,
                    "title": result.title,
                    "content": result.content,
                    "type": result.type,
                    "chunk_index": int(result.chunk_index),
                    "similarity": similarity
                })
            
            # Re-rank results based on relevance
            chunks = self._rerank_results(query, chunks)
            
            # Limit to max_results
            chunks = chunks[:max_results]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank search results based on relevance to query.
        
        Args:
            query: Original query
            results: Search results to re-rank
            
        Returns:
            Re-ranked results
        """
        # Simple re-ranking based on keyword presence and similarity score
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for result in results:
            # Count keyword matches
            content_terms = set(re.findall(r'\w+', result["content"].lower()))
            title_terms = set(re.findall(r'\w+', result["title"].lower()))
            
            # Calculate term overlap
            content_overlap = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
            title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms) if query_terms else 0
            
            # Combine with similarity score
            # Weight: 60% vector similarity, 30% content overlap, 10% title overlap
            result["relevance_score"] = (
                0.6 * result["similarity"] +
                0.3 * content_overlap +
                0.1 * title_overlap
            )
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    async def _keyword_search(
        self,
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search (fallback when vector search is unavailable).
        
        Args:
            query: Query text
            document_type: Optional document type filter
            max_results: Maximum number of results
            
        Returns:
            List of matching documents
        """
        results = []
        
        # Normalize query for matching
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))
        
        for doc_id, doc in self.policy_docs.items():
            # Filter by document type if specified
            if document_type and doc["type"] != document_type:
                continue
            
            # Simple keyword matching
            title_lower = doc["title"].lower()
            content_lower = doc["content"].lower()
            
            # Check for direct substring match
            direct_match = query_lower in title_lower or query_lower in content_lower
            
            # Check for term overlap
            content_terms = set(re.findall(r'\w+', content_lower))
            title_terms = set(re.findall(r'\w+', title_lower))
            term_overlap = len(query_terms.intersection(content_terms.union(title_terms)))
            
            # Calculate relevance score
            if direct_match:
                relevance = 0.9  # High score for direct matches
            elif term_overlap > 0:
                # Score based on term overlap percentage
                relevance = 0.5 + (0.4 * term_overlap / len(query_terms))
            else:
                continue  # Skip if no match
            
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "type": doc["type"],
                "relevance": relevance
            })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:max_results]
    
    async def _generate_response_from_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a response based on retrieved chunks.
        
        Args:
            query: Original query
            chunks: Retrieved document chunks
            
        Returns:
            Generated response
        """
        if not chunks:
            return "No relevant information found in the policy documents."
        
        # Prepare context from chunks
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}"
            for chunk in chunks
        ])
        
        # Generate response using Gemini
        prompt = f"""
        You are a financial compliance expert. Based on the following policy document excerpts,
        provide an answer to this query: "{query}"
        
        Context:
        {context}
        
        Provide a concise, accurate response based only on the information in the policy documents.
        If the information is not available in the documents, state that clearly.
        """
        
        try:
            response = await self.gemini_client.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to simple concatenation
            return f"Based on the policy documents:\n\n{context}"
    
    def _merge_chunks_by_document(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge chunks from the same document.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of merged documents
        """
        # Group chunks by document
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = {
                    "id": doc_id,
                    "title": chunk["title"],
                    "type": chunk["type"],
                    "chunks": [],
                    "max_similarity": chunk["similarity"]
                }
            
            doc_chunks[doc_id]["chunks"].append(chunk)
            doc_chunks[doc_id]["max_similarity"] = max(
                doc_chunks[doc_id]["max_similarity"],
                chunk["similarity"]
            )
        
        # Merge chunks for each document
        merged_docs = []
        for doc_id, doc in doc_chunks.items():
            # Sort chunks by index
            sorted_chunks = sorted(doc["chunks"], key=lambda x: x["chunk_index"])
            
            # Merge content
            content = "\n".join([chunk["content"] for chunk in sorted_chunks])
            
            merged_docs.append({
                "id": doc_id,
                "title": doc["title"],
                "type": doc["type"],
                "content": content,
                "relevance": doc["max_similarity"],
                "chunk_count": len(doc["chunks"])
            })
        
        # Sort by relevance
        merged_docs.sort(key=lambda x: x["relevance"], reverse=True)
        return merged_docs
    
    async def _arun(
        self,
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        Search policy documents asynchronously.
        
        Args:
            query: The query or topic to search for
            document_type: Optional type of document to search
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            JSON string containing matching policy information
        """
        try:
            # Try vector search first if available
            chunks = []
            if self.vector_search_available:
                chunks = await self._vector_search(
                    query,
                    document_type,
                    max_results,
                    similarity_threshold
                )
            
            # Fall back to keyword search if no results or vector search unavailable
            if not chunks:
                logger.info("Falling back to keyword search")
                results = await self._keyword_search(query, document_type, max_results)
                
                # Return keyword search results
                if results:
                    # Generate response
                    response = await self._generate_response_from_chunks(query, results)
                    
                    return json.dumps({
                        "success": True,
                        "query": query,
                        "document_type": document_type,
                        "search_type": "keyword",
                        "results": results,
                        "count": len(results),
                        "generated_response": response
                    })
            else:
                # Merge chunks by document
                merged_docs = self._merge_chunks_by_document(chunks)
                
                # Generate response from chunks
                response = await self._generate_response_from_chunks(query, chunks)
                
                return json.dumps({
                    "success": True,
                    "query": query,
                    "document_type": document_type,
                    "search_type": "vector",
                    "results": merged_docs,
                    "count": len(merged_docs),
                    "generated_response": response
                })
            
            # If no results from either method, generate a response based on all documents
            if not chunks and not results:
                # Prepare context from all policy documents
                context = "\n\n".join([
                    f"Document: {doc['title']}\n{doc['content']}"
                    for doc in self.policy_docs.values()
                    if not document_type or doc["type"] == document_type
                ])
                
                # Generate a response based on the query and context
                prompt = f"""
                You are a financial compliance expert. Based on the following policy documents,
                provide an answer to this query: "{query}"
                
                Context:
                {context}
                
                Provide a concise, accurate response based only on the information in the policy documents.
                If the information is not available in the documents, state that clearly.
                """
                
                response = await self.gemini_client.generate_text(prompt)
                
                # Return the generated response
                return json.dumps({
                    "success": True,
                    "query": query,
                    "document_type": document_type,
                    "results": [],
                    "generated_response": response,
                    "note": "No exact matches found, generated response based on available policy information."
                })
            
        except Exception as e:
            logger.error(f"Error searching policy documents: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": query
            })
    
    def _run(
        self,
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> str:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(query, document_type, max_results, similarity_threshold)
        )
    
    def _initialize_default_docs(self):
        """Initialize default policy documents."""
        default_docs = {
            "aml_general": {
                "title": "Anti-Money Laundering General Guidelines",
                "type": "aml",
                "content": """
                Anti-Money Laundering (AML) refers to the laws, regulations, and procedures designed to prevent criminals from disguising illegally obtained funds as legitimate income. Financial institutions are required to monitor customers' transactions and report suspicious activities.
                
                Key requirements include:
                1. Customer Due Diligence (CDD)
                2. Transaction monitoring
                3. Suspicious Activity Reporting
                4. Record keeping
                5. Risk assessment
                6. Training programs
                
                Failure to comply with AML regulations can result in significant fines and penalties.
                """
            },
            "sar_filing": {
                "title": "Suspicious Activity Report (SAR) Filing Requirements",
                "type": "aml",
                "content": """
                Financial institutions must file a Suspicious Activity Report (SAR) when they detect a suspicious transaction or activity that might signal money laundering, fraud, or other criminal activity.
                
                Key SAR filing requirements:
                1. Filing deadline: 30 days from detection (45 days if additional investigation is needed)
                2. Mandatory fields: Customer information, activity details, suspicious activity categories
                3. Threshold: Generally, transactions of $5,000 or more require SAR filing if suspicious
                4. Confidentiality: SAR filings cannot be disclosed to the subject of the report
                5. Record retention: SARs and supporting documentation must be kept for 5 years
                
                Institutions must have clear procedures for identifying, investigating, and reporting suspicious activities.
                """
            },
            "kyc_procedures": {
                "title": "Know Your Customer (KYC) Procedures",
                "type": "kyc",
                "content": """
                Know Your Customer (KYC) procedures are a critical component of AML programs. They require financial institutions to verify the identity of their clients and assess their risk factors.
                
                Standard KYC procedures include:
                1. Customer Identification Program (CIP): Collecting and verifying customer identity information
                2. Customer Due Diligence (CDD): Understanding the nature and purpose of customer relationships
                3. Enhanced Due Diligence (EDD): Additional scrutiny for high-risk customers
                4. Ongoing monitoring: Regular reviews of customer activity and risk profiles
                5. Beneficial ownership identification: Identifying individuals who own 25% or more of a legal entity
                
                KYC procedures must be risk-based and proportionate to the customer's risk profile.
                """
            },
            "pep_definition": {
                "title": "Politically Exposed Person (PEP) Definition",
                "type": "kyc",
                "content": """
                A Politically Exposed Person (PEP) is an individual who is or has been entrusted with a prominent public function. Due to their position and influence, PEPs may present a higher risk for potential involvement in bribery and corruption.
                
                PEP categories typically include:
                1. Senior government officials (e.g., heads of state, ministers)
                2. Senior judicial officials
                3. Senior military officials
                4. Senior executives of state-owned corporations
                5. Important political party officials
                6. Family members and close associates of the above
                
                Financial institutions must apply Enhanced Due Diligence (EDD) to PEPs, including:
                1. Senior management approval for establishing business relationships
                2. Measures to establish source of wealth and source of funds
                3. Enhanced ongoing monitoring of the business relationship
                
                PEP status does not automatically imply involvement in illicit activities.
                """
            },
            "transaction_monitoring": {
                "title": "Transaction Monitoring Requirements",
                "type": "aml",
                "content": """
                Transaction monitoring is the process of reviewing and analyzing customer transactions to identify suspicious activities that might indicate money laundering, terrorist financing, or other financial crimes.
                
                Key requirements:
                1. Automated systems: Institutions must implement automated systems capable of detecting unusual patterns
                2. Risk-based approach: Monitoring intensity should correspond to customer risk profiles
                3. Alert investigation: Proper procedures for reviewing and investigating system alerts
                4. Documentation: Thorough documentation of monitoring processes and investigation outcomes
                5. Periodic review: Regular assessment and tuning of monitoring parameters
                6. Staffing: Adequate and trained personnel to review alerts
                
                Common red flags include:
                - Transactions just below reporting thresholds
                - Rapid movement of funds ("in-and-out" transactions)
                - Transactions with high-risk jurisdictions
                - Unusual cash activity
                - Transactions inconsistent with customer profile
                """
            },
            "sanctions_compliance": {
                "title": "Sanctions Compliance Guidelines",
                "type": "sanctions",
                "content": """
                Financial institutions must comply with various sanctions programs that restrict or prohibit dealings with specific countries, entities, and individuals.
                
                Key sanctions compliance requirements:
                1. Screening: Regular screening of customers and transactions against sanctions lists
                2. Blocking: Immediate freezing of funds or rejecting transactions as required
                3. Reporting: Timely reporting of blocked transactions to appropriate authorities
                4. Risk assessment: Evaluating sanctions risks in business activities and relationships
                5. Technology: Implementing effective screening systems with appropriate sensitivity settings
                6. Updates: Maintaining current sanctions data and program information
                
                Major sanctions programs include those administered by:
                - Office of Foreign Assets Control (OFAC)
                - United Nations Security Council
                - European Union
                - UK Office of Financial Sanctions Implementation (OFSI)
                
                Sanctions violations can result in severe civil and criminal penalties.
                """
            }
        }
        
        # Store in memory
        self.policy_docs = default_docs
        
        # If vector search is available, store in Redis
        if self.vector_search_available:
            import asyncio
            
            # Create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Add each document
            for doc_id, doc in default_docs.items():
                loop.run_until_complete(
                    self.add_document(
                        PolicyDocumentInput(
                            document_id=doc_id,
                            title=doc["title"],
                            content=doc["content"],
                            type=doc["type"]
                        )
                    )
                )
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Closed Redis connection")
            except:
                # Handle sync Redis client
                try:
                    self.redis_client.close()
                    logger.info("Closed Redis connection")
                except Exception as e:
                    logger.error(f"Error closing Redis connection: {e}")
