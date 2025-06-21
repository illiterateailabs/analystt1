"""
Evidence Management Tool for CrewAI Agents

This module provides a comprehensive tool for CrewAI agents to interact with
the Evidence Management System. It allows agents to:
- Create and manage EvidenceBundles.
- Add various types of evidence (transactions, patterns, anomalies, etc.).
- Synthesize narratives from collected evidence.
- Enrich evidence with context from the Graph-Aware RAG service.
- Export evidence bundles to different formats.
- Filter and search through evidence.
- Track audit trails and evidence chains.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from backend.agents.tools.base_tool import AbstractApiTool, ApiError
from backend.core.evidence import (
    EvidenceBundle,
    EvidenceCategory,
    EvidenceSource,
    TransactionEvidence,
    PatternMatchEvidence,
    AnomalyEvidence,
    RelationshipEvidence,
    GraphElementEvidence,
    ExternalSourceEvidence,
    GraphElement,
    create_evidence_bundle,
    create_transaction_evidence,
    create_pattern_match_evidence,
    create_anomaly_evidence,
)
from backend.core.graph_rag import GraphRAG, GraphElementType, SearchResults
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.core.telemetry import trace_async_function

# Configure module logger
logger = logging.getLogger(__name__)


class CreateBundleRequest(BaseModel):
    """Request model for creating a new EvidenceBundle."""
    investigation_id: Optional[str] = Field(None, description="Unique ID for the investigation.")
    narrative: str = Field("", description="Initial narrative or summary for the bundle.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the bundle.")


class AddEvidenceRequest(BaseModel):
    """Base request model for adding any type of evidence."""
    category: EvidenceCategory = Field(..., description="Category of the evidence.")
    description: str = Field(..., description="A brief description of the evidence.")
    source: EvidenceSource = Field(..., description="The source from which the evidence was obtained.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0) for the evidence.")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw data associated with the evidence.")
    provenance_link: Optional[str] = Field(None, description="Link to the original source or provenance.")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the evidence.")
    parent_id: Optional[str] = Field(None, description="ID of a parent evidence item for chain tracking.")


class AddTransactionEvidenceRequest(AddEvidenceRequest):
    """Request model for adding transaction evidence."""
    category: EvidenceCategory = EvidenceCategory.TRANSACTION
    tx_hash: str
    chain: str
    from_address: str
    to_address: str
    amount: Optional[float] = None
    asset: Optional[str] = None


class AddPatternMatchEvidenceRequest(AddEvidenceRequest):
    """Request model for adding pattern match evidence."""
    category: EvidenceCategory = EvidenceCategory.PATTERN_MATCH
    pattern_name: str
    matched_entities: List[str]
    pattern_description: Optional[str] = None


class AddAnomalyEvidenceRequest(AddEvidenceRequest):
    """Request model for adding anomaly evidence."""
    category: EvidenceCategory = EvidenceCategory.ANOMALY
    anomaly_type: str
    severity: str
    affected_entities: List[str]


class AddRelationshipEvidenceRequest(AddEvidenceRequest):
    """Request model for adding relationship evidence."""
    category: EvidenceCategory = EvidenceCategory.ENTITY_RELATIONSHIP
    from_entity: str
    to_entity: str
    relationship_type: str


class AddGraphElementEvidenceRequest(AddEvidenceRequest):
    """Request model for adding graph element evidence."""
    category: EvidenceCategory = EvidenceCategory.GRAPH_ELEMENT
    element_id: str
    element_type: GraphElementType
    element_properties: Dict[str, Any] = Field(default_factory=dict)


class AddExternalSourceEvidenceRequest(AddEvidenceRequest):
    """Request model for adding external source evidence."""
    category: EvidenceCategory = EvidenceCategory.EXTERNAL_SOURCE
    source_name: str
    source_url: Optional[str] = None
    source_type: str


class SynthesizeNarrativeRequest(BaseModel):
    """Request model for synthesizing a narrative."""
    max_length: int = Field(1000, description="Maximum length of the synthesized narrative.")


class EnrichWithRagRequest(BaseModel):
    """Request model for enriching evidence with RAG."""
    query: str = Field(..., description="The query to use for RAG enrichment.")
    limit: int = Field(5, description="Maximum number of RAG results to add as evidence.")


class ExportBundleRequest(BaseModel):
    """Request model for exporting an EvidenceBundle."""
    format: str = Field(..., description="Export format (json, html, pdf, dataframe).")
    include_raw: bool = Field(False, description="Whether to include raw data in the export (for JSON).")


class FilterEvidenceRequest(BaseModel):
    """Request model for filtering evidence items."""
    category: Optional[EvidenceCategory] = Field(None, description="Filter by evidence category.")
    source: Optional[EvidenceSource] = Field(None, description="Filter by evidence source.")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence score.")
    min_quality: float = Field(0.0, ge=0.0, le=1.0, description="Minimum quality score.")
    tags: Optional[List[str]] = Field(None, description="Filter by tags.")


class SearchEvidenceRequest(BaseModel):
    """Request model for searching evidence items."""
    query: str = Field(..., description="Text query to search within evidence descriptions or raw data.")


class GetEvidenceChainRequest(BaseModel):
    """Request model for getting an evidence chain."""
    evidence_id: str = Field(..., description="ID of the evidence item to get the chain for.")


class EvidenceToolRequest(BaseModel):
    """Main request model for the EvidenceTool, supporting multiple operations."""
    operation: str = Field(..., description="The operation to perform: 'create_bundle', 'add_evidence', 'synthesize_narrative', 'enrich_with_rag', 'export_bundle', 'filter_evidence', 'search_evidence', 'get_summary', 'get_audit_trail', 'get_evidence_chain', 'calculate_overall_confidence', 'calculate_uncertainty'.")
    
    # Optional parameters for each operation
    create_bundle_params: Optional[CreateBundleRequest] = None
    add_evidence_params: Optional[Union[
        AddTransactionEvidenceRequest,
        AddPatternMatchEvidenceRequest,
        AddAnomalyEvidenceRequest,
        AddRelationshipEvidenceRequest,
        AddGraphElementEvidenceRequest,
        AddExternalSourceEvidenceRequest,
        AddEvidenceRequest # Generic fallback
    ]] = None
    synthesize_narrative_params: Optional[SynthesizeNarrativeRequest] = None
    enrich_with_rag_params: Optional[EnrichWithRagRequest] = None
    export_bundle_params: Optional[ExportBundleRequest] = None
    filter_evidence_params: Optional[FilterEvidenceRequest] = None
    search_evidence_params: Optional[SearchEvidenceRequest] = None
    get_evidence_chain_params: Optional[GetEvidenceChainRequest] = None

    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = [
            'create_bundle', 'add_evidence', 'synthesize_narrative', 'enrich_with_rag',
            'export_bundle', 'filter_evidence', 'search_evidence', 'get_summary',
            'get_audit_trail', 'get_evidence_chain', 'calculate_overall_confidence',
            'calculate_uncertainty'
        ]
        if v not in valid_operations:
            raise ValueError(f"Operation must be one of {valid_operations}.")
        return v

    @validator('add_evidence_params')
    def check_add_evidence_params(cls, v, values):
        if values.get('operation') == 'add_evidence' and v is None:
            raise ValueError("add_evidence_params must be provided for 'add_evidence' operation.")
        return v
    
    @validator('export_bundle_params')
    def check_export_bundle_params(cls, v, values):
        if values.get('operation') == 'export_bundle' and v is None:
            raise ValueError("export_bundle_params must be provided for 'export_bundle' operation.")
        return v

    @validator('enrich_with_rag_params')
    def check_enrich_with_rag_params(cls, v, values):
        if values.get('operation') == 'enrich_with_rag' and v is None:
            raise ValueError("enrich_with_rag_params must be provided for 'enrich_with_rag' operation.")
        return v

    @validator('filter_evidence_params')
    def check_filter_evidence_params(cls, v, values):
        if values.get('operation') == 'filter_evidence' and v is None:
            raise ValueError("filter_evidence_params must be provided for 'filter_evidence' operation.")
        return v

    @validator('search_evidence_params')
    def check_search_evidence_params(cls, v, values):
        if values.get('operation') == 'search_evidence' and v is None:
            raise ValueError("search_evidence_params must be provided for 'search_evidence' operation.")
        return v

    @validator('get_evidence_chain_params')
    def check_get_evidence_chain_params(cls, v, values):
        if values.get('operation') == 'get_evidence_chain' and v is None:
            raise ValueError("get_evidence_chain_params must be provided for 'get_evidence_chain' operation.")
        return v


class EvidenceTool(AbstractApiTool):
    """
    A comprehensive Evidence Management tool for CrewAI agents.
    Allows agents to create, manage, and interact with EvidenceBundles.
    """
    name = "evidence_tool"
    description = "Manages and synthesizes evidence for investigations, supporting various types of evidence, RAG enrichment, and export capabilities."
    provider_id = "internal_evidence_manager"  # Internal provider
    request_model = EvidenceToolRequest

    def __init__(self, evidence_bundle: Optional[EvidenceBundle] = None, graph_rag_service: Optional[GraphRAG] = None, redis_client: Optional[RedisClient] = None):
        """
        Initializes the EvidenceTool.
        Args:
            evidence_bundle: An existing EvidenceBundle instance to operate on. If None, a new one can be created.
            graph_rag_service: An instance of the GraphRAG service for enrichment.
            redis_client: An instance of the RedisClient for persistence.
        """
        super().__init__(provider_id=self.provider_id)
        self.evidence_bundle = evidence_bundle
        self.graph_rag_service = graph_rag_service
        self.redis_client = redis_client or RedisClient() # For persistence if needed

    @trace_async_function(span_name="evidence_tool.execute")
    async def _execute(self, request: EvidenceToolRequest) -> Dict[str, Any]:
        """
        Executes the requested EvidenceTool operation.
        Args:
            request: An instance of EvidenceToolRequest specifying the operation and parameters.
        Returns:
            A dictionary containing the result of the operation.
        Raises:
            ValueError: If the operation is unknown or parameters are missing.
            ApiError: For errors during the underlying EvidenceBundle operations.
        """
        try:
            if request.operation == 'create_bundle':
                if self.evidence_bundle:
                    logger.warning("An EvidenceBundle already exists for this tool instance. Creating a new one.")
                self.evidence_bundle = create_evidence_bundle(
                    narrative=request.create_bundle_params.narrative if request.create_bundle_params else "",
                    investigation_id=request.create_bundle_params.investigation_id if request.create_bundle_params else None,
                    metadata=request.create_bundle_params.metadata if request.create_bundle_params else None,
                    graph_rag_service=self.graph_rag_service
                )
                return {"status": "success", "message": "EvidenceBundle created.", "investigation_id": self.evidence_bundle.investigation_id}
            
            # Ensure bundle exists for other operations
            if not self.evidence_bundle:
                raise ValueError("No EvidenceBundle initialized. Call 'create_bundle' first.")

            if request.operation == 'add_evidence':
                params = request.add_evidence_params
                if not params: raise ValueError("add_evidence_params are required.")
                
                if params.category == EvidenceCategory.TRANSACTION:
                    # Cast to specific type
                    tx_params = params
                    evidence = create_transaction_evidence(
                        tx_hash=tx_params.tx_hash,
                        chain=tx_params.chain,
                        from_address=tx_params.from_address,
                        to_address=tx_params.to_address,
                        amount=tx_params.amount,
                        asset=tx_params.asset,
                        description=tx_params.description,
                        source=tx_params.source,
                        confidence=tx_params.confidence,
                        raw_data=tx_params.raw_data,
                        provenance_link=tx_params.provenance_link,
                        tags=tx_params.tags
                    )
                elif params.category == EvidenceCategory.PATTERN_MATCH:
                    # Cast to specific type
                    pattern_params = params
                    evidence = create_pattern_match_evidence(
                        pattern_name=pattern_params.pattern_name,
                        matched_entities=pattern_params.matched_entities,
                        pattern_description=pattern_params.pattern_description,
                        description=pattern_params.description,
                        source=pattern_params.source,
                        confidence=pattern_params.confidence,
                        raw_data=pattern_params.raw_data,
                        provenance_link=pattern_params.provenance_link,
                        tags=pattern_params.tags
                    )
                elif params.category == EvidenceCategory.ANOMALY:
                    # Cast to specific type
                    anomaly_params = params
                    evidence = create_anomaly_evidence(
                        anomaly_type=anomaly_params.anomaly_type,
                        severity=anomaly_params.severity,
                        affected_entities=anomaly_params.affected_entities,
                        description=anomaly_params.description,
                        source=anomaly_params.source,
                        confidence=anomaly_params.confidence,
                        raw_data=anomaly_params.raw_data,
                        provenance_link=anomaly_params.provenance_link,
                        tags=anomaly_params.tags
                    )
                elif params.category == EvidenceCategory.ENTITY_RELATIONSHIP:
                    # Cast to specific type
                    rel_params = params
                    evidence = RelationshipEvidence(
                        category=EvidenceCategory.ENTITY_RELATIONSHIP,
                        description=rel_params.description,
                        source=rel_params.source,
                        confidence=rel_params.confidence,
                        raw_data=rel_params.raw_data,
                        provenance_link=rel_params.provenance_link,
                        tags=rel_params.tags,
                        from_entity=rel_params.from_entity,
                        to_entity=rel_params.to_entity,
                        relationship_type=rel_params.relationship_type
                    )
                elif params.category == EvidenceCategory.GRAPH_ELEMENT:
                    # Cast to specific type
                    graph_params = params
                    evidence = GraphElementEvidence(
                        category=EvidenceCategory.GRAPH_ELEMENT,
                        description=graph_params.description,
                        source=graph_params.source,
                        confidence=graph_params.confidence,
                        raw_data=graph_params.raw_data,
                        provenance_link=graph_params.provenance_link,
                        tags=graph_params.tags,
                        element_id=graph_params.element_id,
                        element_type=graph_params.element_type,
                        element_properties=graph_params.element_properties
                    )
                elif params.category == EvidenceCategory.EXTERNAL_SOURCE:
                    # Cast to specific type
                    ext_params = params
                    evidence = ExternalSourceEvidence(
                        category=EvidenceCategory.EXTERNAL_SOURCE,
                        description=ext_params.description,
                        source=ext_params.source,
                        confidence=ext_params.confidence,
                        raw_data=ext_params.raw_data,
                        provenance_link=ext_params.provenance_link,
                        tags=ext_params.tags,
                        source_name=ext_params.source_name,
                        source_url=ext_params.source_url,
                        source_type=ext_params.source_type
                    )
                else:
                    # Generic evidence item
                    evidence = EvidenceItem(
                        category=params.category,
                        description=params.description,
                        source=params.source,
                        confidence=params.confidence,
                        raw_data=params.raw_data,
                        provenance_link=params.provenance_link,
                        tags=params.tags,
                        parent_id=params.parent_id
                    )
                
                # Add to bundle
                self.evidence_bundle.add_evidence(evidence)
                return {
                    "status": "success", 
                    "message": f"Evidence added: {evidence.id}", 
                    "evidence_id": evidence.id,
                    "category": evidence.category.value,
                    "quality_score": evidence.quality_score
                }
            
            elif request.operation == 'synthesize_narrative':
                params = request.synthesize_narrative_params
                max_length = params.max_length if params else 1000
                narrative = self.evidence_bundle.synthesize_narrative(max_length=max_length)
                return {
                    "status": "success",
                    "narrative": narrative,
                    "length": len(narrative)
                }
            
            elif request.operation == 'enrich_with_rag':
                params = request.enrich_with_rag_params
                if not params: raise ValueError("enrich_with_rag_params are required.")
                
                if not self.graph_rag_service:
                    raise ValueError("GraphRAG service not available for enrichment.")
                
                results = await self.evidence_bundle.enrich_with_rag(
                    query=params.query,
                    limit=params.limit
                )
                
                return {
                    "status": "success",
                    "query": params.query,
                    "results_count": len(results),
                    "results": [
                        {
                            "element_id": result.element_id,
                            "element_type": result.element_type.value,
                            "similarity": result.similarity,
                            "text": result.text
                        }
                        for result in results
                    ]
                }
            
            elif request.operation == 'export_bundle':
                params = request.export_bundle_params
                if not params: raise ValueError("export_bundle_params are required.")
                
                if params.format.lower() == 'json':
                    export_data = self.evidence_bundle.to_json(include_raw=params.include_raw)
                    return {
                        "status": "success",
                        "format": "json",
                        "data": export_data
                    }
                elif params.format.lower() == 'html':
                    export_data = self.evidence_bundle.to_html()
                    return {
                        "status": "success",
                        "format": "html",
                        "data": export_data
                    }
                elif params.format.lower() == 'pdf':
                    pdf_path = self.evidence_bundle.to_pdf()
                    return {
                        "status": "success",
                        "format": "pdf",
                        "path": pdf_path
                    }
                elif params.format.lower() == 'dataframe':
                    df = self.evidence_bundle.to_dataframe()
                    return {
                        "status": "success",
                        "format": "dataframe",
                        "rows": len(df),
                        "columns": list(df.columns),
                        "data": df.to_dict(orient='records')
                    }
                else:
                    raise ValueError(f"Unsupported export format: {params.format}")
            
            elif request.operation == 'filter_evidence':
                params = request.filter_evidence_params
                if not params: raise ValueError("filter_evidence_params are required.")
                
                filtered_items = self.evidence_bundle.filter_evidence(
                    category=params.category,
                    source=params.source,
                    min_confidence=params.min_confidence,
                    min_quality=params.min_quality,
                    tags=params.tags
                )
                
                return {
                    "status": "success",
                    "count": len(filtered_items),
                    "items": [
                        {
                            "id": item.id,
                            "category": item.category.value,
                            "description": item.description,
                            "source": item.source.value,
                            "confidence": item.confidence,
                            "quality_score": item.quality_score,
                            "timestamp": item.timestamp.isoformat()
                        }
                        for item in filtered_items
                    ]
                }
            
            elif request.operation == 'search_evidence':
                params = request.search_evidence_params
                if not params: raise ValueError("search_evidence_params are required.")
                
                search_results = self.evidence_bundle.search_evidence(params.query)
                
                return {
                    "status": "success",
                    "query": params.query,
                    "count": len(search_results),
                    "items": [
                        {
                            "id": item.id,
                            "category": item.category.value,
                            "description": item.description,
                            "source": item.source.value,
                            "confidence": item.confidence,
                            "quality_score": item.quality_score,
                            "timestamp": item.timestamp.isoformat()
                        }
                        for item in search_results
                    ]
                }
            
            elif request.operation == 'get_summary':
                summary = self.evidence_bundle.get_summary()
                return {
                    "status": "success",
                    "summary": summary
                }
            
            elif request.operation == 'get_audit_trail':
                audit_trail = self.evidence_bundle.audit_trail
                return {
                    "status": "success",
                    "count": len(audit_trail),
                    "audit_trail": [
                        {
                            "timestamp": event.timestamp.isoformat(),
                            "event_type": event.event_type,
                            "description": event.description,
                            "details": event.details
                        }
                        for event in audit_trail
                    ]
                }
            
            elif request.operation == 'get_evidence_chain':
                params = request.get_evidence_chain_params
                if not params: raise ValueError("get_evidence_chain_params are required.")
                
                chain = self.evidence_bundle.get_evidence_chain(params.evidence_id)
                
                return {
                    "status": "success",
                    "evidence_id": params.evidence_id,
                    "chain_length": len(chain),
                    "chain": [
                        {
                            "id": item.id,
                            "category": item.category.value,
                            "description": item.description,
                            "source": item.source.value,
                            "confidence": item.confidence,
                            "quality_score": item.quality_score,
                            "timestamp": item.timestamp.isoformat()
                        }
                        for item in chain
                    ]
                }
            
            elif request.operation == 'calculate_overall_confidence':
                confidence = self.evidence_bundle.calculate_overall_confidence()
                return {
                    "status": "success",
                    "overall_confidence": confidence
                }
            
            elif request.operation == 'calculate_uncertainty':
                uncertainty = self.evidence_bundle.calculate_uncertainty()
                return {
                    "status": "success",
                    "uncertainty": uncertainty
                }
            
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
        
        except Exception as e:
            logger.error(f"Error in EvidenceTool: {e}", exc_info=True)
            raise ApiError(f"Evidence operation failed: {e}", provider_id=self.provider_id, endpoint=request.operation)
