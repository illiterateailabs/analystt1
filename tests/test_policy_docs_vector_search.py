"""
Comprehensive tests for the PolicyDocsTool vector search functionality.

This module tests:
1. Initialization with and without Redis.
2. Document addition, updating, and deletion.
3. Vector search (mocked Redis) and fallback to keyword search.
4. Embedding generation (mocked Gemini) and document chunking.
5. Re-ranking and response generation (mocked Gemini).
6. Edge cases and error handling.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call

from backend.agents.tools.policy_docs_tool import (
    PolicyDocsTool,
    PolicyDocumentInput,
    PolicyQueryInput,
    PolicyDocument,
    DocumentChunk,
    REDIS_AVAILABLE
)
from backend.integrations.gemini_client import GeminiClient
from backend.config import settings

# Mock settings if redis_url is not present by default
if not hasattr(settings, 'redis_url'):
    settings.redis_url = "redis://mock-redis:6379/0"


@pytest.fixture
def mock_gemini_client_instance():
    """Fixture for a mocked GeminiClient instance."""
    client = MagicMock(spec=GeminiClient)
    client.generate_embedding = AsyncMock(return_value=[0.1] * PolicyDocsTool.VECTOR_DIM)
    client.generate_text = AsyncMock(return_value="Generated AI response based on context.")
    return client

@pytest.fixture
def mock_redis_ft_client():
    """Fixture for a mocked Redis FT (search) client."""
    ft_client = MagicMock()
    ft_client.info = MagicMock() # Mock info to simulate index check
    ft_client.create_index = MagicMock()
    
    # Mock search results
    mock_search_result_doc = MagicMock()
    mock_search_result_doc.id = "policy:chunk:doc1_chunk_0"
    mock_search_result_doc.doc_id = "doc1"
    mock_search_result_doc.title = "Test Document 1"
    mock_search_result_doc.content = "This is chunk 0 of test document 1."
    mock_search_result_doc.type = "aml"
    mock_search_result_doc.chunk_index = "0" # Redis stores as string
    mock_search_result_doc.__embedding_score = "0.2" # Cosine distance, stored as string

    mock_search_results = MagicMock()
    mock_search_results.docs = [mock_search_result_doc]
    mock_search_results.total = 1
    
    ft_client.search = MagicMock(return_value=mock_search_results)
    return ft_client

@pytest.fixture
def mock_redis_client_instance(mock_redis_ft_client):
    """Fixture for a mocked Redis client instance."""
    client = MagicMock() # Use MagicMock for flexibility if redis.Redis is not available
    if REDIS_AVAILABLE:
        # If redis is available, we can spec it for better type hinting,
        # but for environments where it's not, MagicMock is safer.
        try:
            import redis as redis_sync # Avoid conflict with redis.asyncio
            client = MagicMock(spec=redis_sync.Redis)
        except ImportError:
            pass # Stick with plain MagicMock

    client.ping = AsyncMock(return_value=True) # For async version, though PolicyDocsTool uses sync
    client.ft = MagicMock(return_value=mock_redis_ft_client)
    client.hset = MagicMock()
    client.exists = MagicMock(return_value=False)
    client.keys = MagicMock(return_value=[])
    client.delete = MagicMock()
    client.close = AsyncMock() # For async version
    
    # Synchronous ping for PolicyDocsTool's direct ping
    client.ping = MagicMock(return_value=True)
    return client


@pytest.fixture
async def policy_tool_with_redis(mock_gemini_client_instance, mock_redis_client_instance):
    """Fixture for PolicyDocsTool initialized with mocked Redis."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis module not available, skipping Redis-dependent tests.")
    
    with patch("backend.agents.tools.policy_docs_tool.redis.Redis.from_url", return_value=mock_redis_client_instance):
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance, initialize_default_docs=False)
        # Ensure index creation is attempted or info is checked
        if tool.vector_search_available:
             # If index doesn't exist, create_index is called. If it does, info is called.
            try:
                mock_redis_client_instance.ft().info.assert_called_once()
            except AssertionError:
                mock_redis_client_instance.ft().create_index.assert_called_once()
        yield tool
        await tool.close()


@pytest.fixture
async def policy_tool_without_redis(mock_gemini_client_instance):
    """Fixture for PolicyDocsTool initialized without Redis (fallback mode)."""
    with patch("backend.agents.tools.policy_docs_tool.REDIS_AVAILABLE", False):
        # Re-import or re-evaluate PolicyDocsTool if REDIS_AVAILABLE is checked at import time
        # For this test, patching it should be enough if PolicyDocsTool checks it dynamically.
        # If PolicyDocsTool's constructor directly tries to import redis and fails,
        # this test setup might need adjustment or the tool needs to handle ImportError gracefully.
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance, initialize_default_docs=False)
        yield tool
        await tool.close()


@pytest.mark.asyncio
async def test_initialization_with_redis_success(mock_gemini_client_instance, mock_redis_client_instance):
    """Test PolicyDocsTool initializes correctly when Redis is available."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis module not available")

    with patch("backend.agents.tools.policy_docs_tool.redis.Redis.from_url", return_value=mock_redis_client_instance):
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance, initialize_default_docs=False)
        assert tool.vector_search_available is True
        assert tool.redis_client == mock_redis_client_instance
        mock_redis_client_instance.ping.assert_called_once()
        # Check that index initialization was attempted
        assert mock_redis_client_instance.ft().info.called or mock_redis_client_instance.ft().create_index.called
        await tool.close()

@pytest.mark.asyncio
async def test_initialization_with_redis_failure(mock_gemini_client_instance):
    """Test PolicyDocsTool falls back gracefully if Redis connection fails."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis module not available")

    with patch("backend.agents.tools.policy_docs_tool.redis.Redis.from_url", side_effect=Exception("Redis connection error")):
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance, initialize_default_docs=False)
        assert tool.vector_search_available is False
        assert tool.redis_client is None
        await tool.close()

@pytest.mark.asyncio
async def test_initialization_without_redis_module(mock_gemini_client_instance):
    """Test PolicyDocsTool initializes in fallback mode if redis module is not installed."""
    with patch("backend.agents.tools.policy_docs_tool.REDIS_AVAILABLE", False):
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance, initialize_default_docs=False)
        assert tool.vector_search_available is False
        assert tool.redis_client is None
        await tool.close()


@pytest.mark.asyncio
async def test_add_document_with_redis(policy_tool_with_redis: PolicyDocsTool, mock_gemini_client_instance, mock_redis_client_instance):
    """Test adding a document when Redis is available."""
    doc_input = PolicyDocumentInput(
        title="AML Policy Update Q1",
        content="This is the content of the AML policy update for Q1. It includes new guidelines.",
        type="aml",
        source="Internal Memo"
    )
    doc_id = await policy_tool_with_redis.add_document(doc_input)

    assert doc_id is not None
    assert doc_id in policy_tool_with_redis.policy_docs # Check in-memory fallback
    
    # Check Gemini embedding calls (1 per chunk, assume 1 chunk for this short content)
    mock_gemini_client_instance.generate_embedding.assert_called()
    
    # Check Redis hset calls (1 for doc, 1 per chunk)
    mock_redis_client_instance.hset.assert_any_call(
        f"{PolicyDocsTool.DOC_PREFIX}{doc_id}", mapping=pytest.ANY
    )
    # Assuming 1 chunk for this content
    chunk_id_pattern = f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_0"
    
    # Check if hset was called for the chunk
    chunk_hset_called = False
    for call_args in mock_redis_client_instance.hset.call_args_list:
        if call_args[0][0] == chunk_id_pattern:
            chunk_hset_called = True
            # Verify embedding is part of the chunk data
            assert "embedding" in call_args[1]['mapping']
            break
    assert chunk_hset_called, f"hset for chunk {chunk_id_pattern} was not called"


@pytest.mark.asyncio
async def test_add_document_without_redis(policy_tool_without_redis: PolicyDocsTool):
    """Test adding a document when Redis is not available."""
    doc_input = PolicyDocumentInput(
        title="KYC Procedures",
        content="Detailed KYC procedures for new clients.",
        type="kyc"
    )
    doc_id = await policy_tool_without_redis.add_document(doc_input)
    assert doc_id is not None
    assert doc_id in policy_tool_without_redis.policy_docs
    assert policy_tool_without_redis.policy_docs[doc_id]["title"] == "KYC Procedures"
    # Embeddings should not be generated if Redis is not available for storing them
    policy_tool_without_redis.gemini_client.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_document_logic(policy_tool_with_redis: PolicyDocsTool):
    """Test the _chunk_document method."""
    long_content = ("This is sentence one. This is sentence two. \n\nThis is a new paragraph. "
                    "It has more words. " * int(PolicyDocsTool.CHUNK_SIZE / 20) ) # Ensure content is larger than CHUNK_SIZE
    doc = PolicyDocument(
        id="doc_chunk_test",
        title="Chunk Test Doc",
        content=long_content,
        type="test"
    )
    chunks = policy_tool_with_redis._chunk_document(doc)
    
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.document_id == "doc_chunk_test"
        assert chunk.chunk_index == i
        assert len(chunk.content) <= PolicyDocsTool.CHUNK_SIZE
        if i > 0: # Check overlap
            prev_chunk_end_index = chunks[i-1].content.rfind(chunk.content[:PolicyDocsTool.CHUNK_OVERLAP//2])
            assert prev_chunk_end_index != -1, "Chunk overlap not found or insufficient"


@pytest.mark.asyncio
async def test_vector_search_with_redis(policy_tool_with_redis: PolicyDocsTool, mock_gemini_client_instance, mock_redis_client_instance):
    """Test _vector_search method when Redis is available."""
    # Add a document first to ensure there's something to search (even if mocked)
    doc_input = PolicyDocumentInput(title="Searchable Doc", content="Content for vector search.", type="aml")
    await policy_tool_with_redis.add_document(doc_input)
    
    query = "find AML policies"
    results = await policy_tool_with_redis._vector_search(query, document_type="aml", max_results=3, similarity_threshold=0.6)
    
    mock_gemini_client_instance.generate_embedding.assert_called_with(query) # For query embedding
    mock_redis_client_instance.ft(PolicyDocsTool.INDEX_NAME).search.assert_called_once()
    
    assert len(results) <= 3 # Should be 1 based on mock_redis_ft_client
    if results:
        assert results[0]["title"] == "Test Document 1" # From mock_redis_ft_client
        assert results[0]["similarity"] >= 0.6
        assert results[0]["similarity"] == (1.0 - 0.2) # 1 - cosine distance


@pytest.mark.asyncio
async def test_keyword_search_fallback(policy_tool_without_redis: PolicyDocsTool):
    """Test _keyword_search method (used as fallback)."""
    doc_input = PolicyDocumentInput(title="Keyword Doc", content="This document contains specific keywords for testing.", type="kyc")
    await policy_tool_without_redis.add_document(doc_input)
    
    query = "specific keywords"
    results = await policy_tool_without_redis._keyword_search(query, document_type="kyc", max_results=1)
    
    assert len(results) == 1
    assert results[0]["title"] == "Keyword Doc"
    assert results[0]["relevance"] > 0.5 # Should have some relevance


@pytest.mark.asyncio
async def test_arun_vector_search_flow(policy_tool_with_redis: PolicyDocsTool, mock_gemini_client_instance):
    """Test the main _arun method flow when vector search yields results."""
     # Mock _vector_search to return specific chunks
    mock_chunks = [
        {"id": "chunk1", "doc_id": "doc1", "title": "AML Guidelines", "content": "Guideline A for AML.", "type": "aml", "similarity": 0.85, "chunk_index": 0},
        {"id": "chunk2", "doc_id": "doc1", "title": "AML Guidelines", "content": "Guideline B for AML.", "type": "aml", "similarity": 0.82, "chunk_index": 1}
    ]
    with patch.object(policy_tool_with_redis, "_vector_search", AsyncMock(return_value=mock_chunks)):
        result_json = await policy_tool_with_redis._arun(query="AML guidelines", document_type="aml")
    
    result = json.loads(result_json)
    assert result["success"] is True
    assert result["search_type"] == "vector"
    assert len(result["results"]) == 1 # Merged by document
    assert result["results"][0]["title"] == "AML Guidelines"
    assert "Guideline A" in result["results"][0]["content"]
    assert "Guideline B" in result["results"][0]["content"]
    assert "generated_response" in result
    mock_gemini_client_instance.generate_text.assert_called_once() # For generating final response


@pytest.mark.asyncio
async def test_arun_fallback_to_keyword_search(policy_tool_with_redis: PolicyDocsTool, mock_gemini_client_instance):
    """Test _arun falls back to keyword search if vector search yields no results."""
    # Mock _vector_search to return empty list
    with patch.object(policy_tool_with_redis, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(policy_tool_with_redis, "_keyword_search", AsyncMock(return_value=[
             {"id": "doc_kw", "title": "Keyword Match Doc", "content": "Content matching keywords.", "type": "aml", "relevance": 0.7}
         ])):
        result_json = await policy_tool_with_redis._arun(query="keywords", document_type="aml")
        
    result = json.loads(result_json)
    assert result["success"] is True
    assert result["search_type"] == "keyword"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Keyword Match Doc"
    mock_gemini_client_instance.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_arun_no_results_found(policy_tool_with_redis: PolicyDocsTool, mock_gemini_client_instance):
    """Test _arun when neither vector nor keyword search finds results."""
    # Add a default doc so it has something to generate a response from
    await policy_tool_with_redis.add_document(PolicyDocumentInput(title="Default", content="Default content", type="generic"))

    with patch.object(policy_tool_with_redis, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(policy_tool_with_redis, "_keyword_search", AsyncMock(return_value=[])):
        result_json = await policy_tool_with_redis._arun(query="obscure query", document_type="aml")
        
    result = json.loads(result_json)
    assert result["success"] is True
    assert len(result["results"]) == 0
    assert "note" in result
    assert "No exact matches found" in result["note"]
    mock_gemini_client_instance.generate_text.assert_called_once() # Gemini tries to answer based on general context


@pytest.mark.asyncio
async def test_rerank_results_logic(policy_tool_with_redis: PolicyDocsTool):
    """Test the _rerank_results method."""
    query = "important policy update"
    results_to_rerank = [
        {"id": "c1", "title": "Policy Update", "content": "An important update to policy.", "similarity": 0.9, "type":"generic"},
        {"id": "c2", "title": "Old Policy", "content": "Some old information.", "similarity": 0.8, "type":"generic"},
        {"id": "c3", "title": "Update Notice", "content": "Minor important notice.", "similarity": 0.85, "type":"generic"},
    ]
    reranked = policy_tool_with_redis._rerank_results(query, results_to_rerank)
    
    assert len(reranked) == 3
    # Expect c1 to be highest due to title and content match + high similarity
    assert reranked[0]["id"] == "c1"
    assert reranked[0]["relevance_score"] > reranked[1]["relevance_score"]
    assert reranked[1]["relevance_score"] > reranked[2]["relevance_score"]


@pytest.mark.asyncio
async def test_delete_document_with_redis(policy_tool_with_redis: PolicyDocsTool, mock_redis_client_instance):
    """Test deleting a document when Redis is available."""
    doc_input = PolicyDocumentInput(title="To Be Deleted", content="Delete me.", type="temp")
    doc_id = await policy_tool_with_redis.add_document(doc_input)
    
    # Mock Redis keys() to return chunks for this doc_id
    mock_redis_client_instance.keys.return_value = [f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_0"]
    
    deleted = await policy_tool_with_redis.delete_document(doc_id)
    assert deleted is True
    assert doc_id not in policy_tool_with_redis.policy_docs
    
    mock_redis_client_instance.delete.assert_any_call(f"{PolicyDocsTool.DOC_PREFIX}{doc_id}")
    mock_redis_client_instance.delete.assert_any_call(f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_0")

@pytest.mark.asyncio
async def test_update_document_with_redis(policy_tool_with_redis: PolicyDocsTool, mock_redis_client_instance, mock_gemini_client_instance):
    """Test updating a document when Redis is available."""
    doc_input_orig = PolicyDocumentInput(title="Original Doc", content="Original content.", type="orig")
    doc_id = await policy_tool_with_redis.add_document(doc_input_orig)
    
    # Reset mocks for embedding and hset for the update part
    mock_gemini_client_instance.generate_embedding.reset_mock()
    mock_redis_client_instance.hset.reset_mock()
    mock_redis_client_instance.keys.return_value = [f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_0"] # Simulate old chunk existing
    mock_redis_client_instance.delete.reset_mock() # Reset delete mock

    doc_input_updated = PolicyDocumentInput(
        document_id=doc_id,
        title="Updated Doc",
        content="Updated content here.",
        type="updated"
    )
    updated = await policy_tool_with_redis.update_document(doc_input_updated)
    assert updated is True
    assert policy_tool_with_redis.policy_docs[doc_id]["title"] == "Updated Doc"
    
    # Check old chunks deleted
    mock_redis_client_instance.delete.assert_called_with(f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_0")
    
    # Check new doc and chunks added
    mock_gemini_client_instance.generate_embedding.assert_called() # For new chunks
    mock_redis_client_instance.hset.assert_any_call(f"{PolicyDocsTool.DOC_PREFIX}{doc_id}", mapping=pytest.ANY)
    # Check if hset was called for the new chunk
    new_chunk_hset_called = False
    for call_args in mock_redis_client_instance.hset.call_args_list:
        if call_args[0][0].startswith(f"{PolicyDocsTool.CHUNK_PREFIX}{doc_id}_chunk_"): # New chunk ID will be similar
            new_chunk_hset_called = True
            break
    assert new_chunk_hset_called, "hset for new chunk was not called during update"

@pytest.mark.asyncio
async def test_initialize_default_docs_with_redis(mock_gemini_client_instance, mock_redis_client_instance):
    """Test that default documents are added to Redis on initialization if flag is True."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis module not available")
    
    with patch("backend.agents.tools.policy_docs_tool.redis.Redis.from_url", return_value=mock_redis_client_instance):
        # initialize_default_docs is True by default
        tool = PolicyDocsTool(gemini_client=mock_gemini_client_instance) 
        
        # Check that add_document (which calls hset and generate_embedding) was called for default docs
        # Number of default docs * (1 for doc + N for chunks)
        # Default docs have short content, likely 1 chunk each.
        num_default_docs = 6 # As per current default_docs in PolicyDocsTool
        
        # Check generate_embedding calls (1 per chunk)
        assert mock_gemini_client_instance.generate_embedding.call_count >= num_default_docs
        
        # Check hset calls (1 for doc + 1 per chunk)
        assert mock_redis_client_instance.hset.call_count >= num_default_docs * 2
        await tool.close()

@pytest.mark.asyncio
async def test_close_method_redis(policy_tool_with_redis: PolicyDocsTool, mock_redis_client_instance):
    """Test that the close method calls redis_client.close()."""
    await policy_tool_with_redis.close()
    # If redis_client is async, it's await close(). If sync, it's close().
    # The mock_redis_client_instance.close is an AsyncMock.
    mock_redis_client_instance.close.assert_called_once()

@pytest.mark.asyncio
async def test_close_method_no_redis(policy_tool_without_redis: PolicyDocsTool):
    """Test that the close method doesn't error if no redis_client."""
    # This test mainly ensures no exception is raised.
    await policy_tool_without_redis.close()
    # No specific assertion needed other than it completes without error.
    assert policy_tool_without_redis.redis_client is None

