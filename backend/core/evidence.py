"""
Evidence Management System

This module provides a comprehensive system for managing and synthesizing evidence
in fraud investigations, supporting various evidence types, provenance tracking,
quality scoring, and integration with graph-aware RAG.

Features:
- Standardized EvidenceBundle object for aggregation
- Support for multiple evidence types (transactions, patterns, anomalies, relationships, raw graph elements)
- Evidence validation and quality scoring
- Aggregation and synthesis of evidence into a narrative
- Export capabilities to various formats
- Provenance tracking and audit trail
- Filtering and search capabilities
- Confidence scoring and uncertainty quantification
- Integration with Graph RAG for contextual evidence
"""

import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd
from pydantic import BaseModel, Field, validator

from backend.core.events import EventPriority, publish_event
from backend.core.graph_rag import (
    GraphElement, Node, Relationship, Path, Subgraph,
    GraphElementType, GraphRAG, SearchQuery, SearchResult
)
from backend.core.metrics import BusinessMetrics

# Configure module logger
logger = logging.getLogger(__name__)


class EvidenceCategory(str, Enum):
    """Categories of evidence for filtering and organization."""
    TRANSACTION = "transaction"
    PATTERN_MATCH = "pattern_match"
    ANOMALY = "anomaly"
    ENTITY_RELATIONSHIP = "entity_relationship"
    EXTERNAL_SOURCE = "external_source"
    GRAPH_ELEMENT = "graph_element"
    OTHER = "other"


class EvidenceSource(str, Enum):
    """Source of the evidence."""
    SIM_API = "sim_api"
    GRAPH_ANALYSIS = "graph_analysis"
    LLM_GENERATED = "llm_generated"
    HUMAN_INPUT = "human_input"
    EXTERNAL_TOOL = "external_tool"
    SYSTEM = "system"


class EvidenceItem(BaseModel):
    """Base model for a single piece of evidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: EvidenceCategory
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    source: EvidenceSource
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score (0.0-1.0)")
    raw_data: Optional[Dict[str, Any]] = None
    provenance_link: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None  # For evidence chain tracking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence item to dictionary for serialization."""
        data = self.dict()
        data["category"] = self.category.value
        data["source"] = self.source.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        """Create evidence item from dictionary."""
        data["category"] = EvidenceCategory(data["category"])
        data["source"] = EvidenceSource(data["source"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def calculate_quality_score(self) -> float:
        """
        Calculate quality score based on evidence attributes.
        
        Returns:
            Quality score (0.0-1.0)
        """
        # Start with base score
        score = 0.5
        
        # Adjust based on confidence
        score += self.confidence * 0.2
        
        # Adjust based on source reliability
        source_reliability = {
            EvidenceSource.HUMAN_INPUT: 0.9,
            EvidenceSource.EXTERNAL_TOOL: 0.8,
            EvidenceSource.SIM_API: 0.7,
            EvidenceSource.GRAPH_ANALYSIS: 0.7,
            EvidenceSource.LLM_GENERATED: 0.5,
            EvidenceSource.SYSTEM: 0.6,
        }
        score += source_reliability.get(self.source, 0.5) * 0.1
        
        # Adjust based on completeness
        has_raw_data = self.raw_data is not None and bool(self.raw_data)
        has_provenance = self.provenance_link is not None and bool(self.provenance_link)
        has_tags = bool(self.tags)
        
        completeness = 0.0
        if has_raw_data:
            completeness += 0.4
        if has_provenance:
            completeness += 0.4
        if has_tags:
            completeness += 0.2
        
        score += completeness * 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def update_quality_score(self) -> None:
        """Update the quality score based on evidence attributes."""
        self.quality_score = self.calculate_quality_score()


class TransactionEvidence(EvidenceItem):
    """Evidence related to a blockchain transaction."""
    category: EvidenceCategory = EvidenceCategory.TRANSACTION
    tx_hash: str
    chain: str
    from_address: str
    to_address: str
    amount: Optional[float] = None
    asset: Optional[str] = None
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 1.0 # Transactions are factual


class PatternMatchEvidence(EvidenceItem):
    """Evidence related to a detected fraud pattern."""
    category: EvidenceCategory = EvidenceCategory.PATTERN_MATCH
    pattern_name: str
    matched_entities: List[str] = Field(default_factory=list)
    pattern_description: Optional[str] = None
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 0.7 # Patterns have inherent uncertainty


class AnomalyEvidence(EvidenceItem):
    """Evidence related to an detected anomaly."""
    category: EvidenceCategory = EvidenceCategory.ANOMALY
    anomaly_type: str
    severity: str
    affected_entities: List[str] = Field(default_factory=list)
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 0.6 # Anomalies are often statistical


class RelationshipEvidence(EvidenceItem):
    """Evidence related to a relationship between entities."""
    category: EvidenceCategory = EvidenceCategory.ENTITY_RELATIONSHIP
    from_entity: str
    to_entity: str
    relationship_type: str
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 0.8 # Relationships can be strong or inferred


class GraphElementEvidence(EvidenceItem):
    """Evidence directly from a graph element (node, relationship, path, subgraph)."""
    category: EvidenceCategory = EvidenceCategory.GRAPH_ELEMENT
    element_id: str
    element_type: GraphElementType
    element_properties: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 0.9 # Graph elements are usually factual


class ExternalSourceEvidence(EvidenceItem):
    """Evidence from external sources like news, reports, or APIs."""
    category: EvidenceCategory = EvidenceCategory.EXTERNAL_SOURCE
    source_name: str
    source_url: Optional[str] = None
    source_type: str
    
    @validator('confidence', always=True)
    def set_default_confidence(cls, v):
        return v if v is not None else 0.6 # External sources vary in reliability


class AuditEvent(BaseModel):
    """Model for an audit event in the evidence bundle."""
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    user_id: Optional[str] = None
    description: str
    details: Optional[Dict[str, Any]] = None


class EvidenceBundle:
    """
    A comprehensive bundle of evidence for a fraud investigation.
    Aggregates various types of evidence, provides synthesis, and provenance.
    """
    def __init__(
        self,
        narrative: str = "",
        evidence_items: Optional[List[EvidenceItem]] = None,
        raw_data: Optional[List[Dict[str, Any]]] = None,
        audit_trail: Optional[List[AuditEvent]] = None,
        graph_rag_service: Optional[GraphRAG] = None,
        investigation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.narrative = narrative
        self.evidence_items: List[EvidenceItem] = evidence_items if evidence_items is not None else []
        self.raw_data: List[Dict[str, Any]] = raw_data if raw_data is not None else []
        self.audit_trail: List[AuditEvent] = audit_trail if audit_trail is not None else []
        self.graph_rag_service = graph_rag_service
        self.investigation_id = investigation_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.summary = {}
        self._update_summary()
        
        # Log creation
        self._log_audit_event("create_bundle", "Evidence bundle created")

    def add_evidence(self, item: EvidenceItem) -> None:
        """Adds a single evidence item to the bundle."""
        # Update quality score
        item.update_quality_score()
        
        # Add to evidence items
        self.evidence_items.append(item)
        self._update_summary()
        self._log_audit_event("add_evidence", f"Added evidence: {item.id}", {"evidence_id": item.id, "category": item.category.value})
        
        # Track metrics
        BusinessMetrics.track_event(
            event_type="evidence_added",
            category=item.category.value,
            source=item.source.value,
            confidence=item.confidence,
            quality=item.quality_score,
        )

    def add_raw_data(self, data: Dict[str, Any], description: str = "Raw data added") -> None:
        """Adds raw, unstructured data to the bundle."""
        self.raw_data.append(data)
        self._log_audit_event("add_raw_data", description, {"data_type": type(data).__name__})

    def add_graph_element(
        self, 
        element: GraphElement, 
        description: str = "", 
        source: EvidenceSource = EvidenceSource.GRAPH_ANALYSIS, 
        confidence: float = 0.9
    ) -> None:
        """Adds a graph element as evidence."""
        evidence_item = GraphElementEvidence(
            category=EvidenceCategory.GRAPH_ELEMENT,
            description=description or f"Graph element {element.id} of type {element.type.value}",
            source=source,
            confidence=confidence,
            raw_data=element.dict(),
            element_id=element.id,
            element_type=element.type,
            element_properties=element.properties
        )
        self.add_evidence(evidence_item)

    def _update_summary(self) -> None:
        """Internal method to update summary statistics."""
        self.summary = {
            "total_items": len(self.evidence_items),
            "categories": {cat.value: 0 for cat in EvidenceCategory},
            "sources": {src.value: 0 for src in EvidenceSource},
            "avg_confidence": 0.0,
            "avg_quality_score": 0.0,
            "raw_data_count": len(self.raw_data)
        }
        
        total_confidence = 0.0
        total_quality = 0.0

        for item in self.evidence_items:
            self.summary["categories"][item.category.value] += 1
            self.summary["sources"][item.source.value] += 1
            total_confidence += item.confidence
            total_quality += item.quality_score
        
        if self.evidence_items:
            self.summary["avg_confidence"] = total_confidence / len(self.evidence_items)
            self.summary["avg_quality_score"] = total_quality / len(self.evidence_items)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the evidence bundle."""
        return self.summary

    def filter_evidence(
        self,
        category: Optional[EvidenceCategory] = None,
        source: Optional[EvidenceSource] = None,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[EvidenceItem]:
        """Filters evidence items based on criteria."""
        filtered = []
        for item in self.evidence_items:
            if category and item.category != category:
                continue
            if source and item.source != source:
                continue
            if item.confidence < min_confidence:
                continue
            if item.quality_score < min_quality:
                continue
            if tags and not any(tag in item.tags for tag in tags):
                continue
            filtered.append(item)
        return filtered

    async def enrich_with_rag(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Uses Graph RAG to find and add contextual evidence.
        Returns a list of search results.
        """
        if not self.graph_rag_service:
            logger.warning("Graph RAG service not available for evidence enrichment")
            return []
        
        try:
            # Create search query
            search_query = SearchQuery(
                query=query,
                limit=limit,
                min_similarity=0.6,  # Lower threshold for better recall
                include_raw_elements=True,
            )
            
            # Execute search
            search_results = await self.graph_rag_service.search(search_query)
            
            # Add search results as evidence
            for result in search_results.results:
                if result.element:
                    # Add as graph element evidence
                    self.add_graph_element(
                        element=result.element,
                        description=f"Graph element found via RAG for query: {query}",
                        confidence=result.similarity,
                    )
            
            # Log the enrichment
            self._log_audit_event(
                "enrich_with_rag", 
                f"Enriched with RAG using query: {query}",
                {"query": query, "results_count": len(search_results.results)}
            )
            
            return search_results.results
        
        except Exception as e:
            logger.error(f"Error enriching evidence with RAG: {e}")
            return []

    def synthesize_narrative(self, max_length: int = 1000) -> str:
        """
        Synthesizes a narrative from the evidence items.
        
        Args:
            max_length: Maximum length of the narrative
            
        Returns:
            Synthesized narrative
        """
        # This would typically use an LLM for synthesis
        # For now, we'll create a simple summary
        
        # Start with a header
        narrative = f"# Evidence Summary for Investigation {self.investigation_id}\n\n"
        
        # Add summary statistics
        narrative += "## Summary Statistics\n\n"
        narrative += f"- Total Evidence Items: {self.summary['total_items']}\n"
        narrative += f"- Average Confidence: {self.summary['avg_confidence']:.2f}\n"
        narrative += f"- Average Quality Score: {self.summary['avg_quality_score']:.2f}\n\n"
        
        # Add category breakdown
        narrative += "## Evidence by Category\n\n"
        for category, count in self.summary["categories"].items():
            if count > 0:
                narrative += f"- {category.replace('_', ' ').title()}: {count}\n"
        
        # Add high-confidence evidence highlights
        narrative += "\n## Key Evidence Highlights\n\n"
        high_confidence = self.filter_evidence(min_confidence=0.8)
        for item in high_confidence[:5]:  # Limit to top 5
            narrative += f"- {item.description} (Confidence: {item.confidence:.2f})\n"
        
        # Add timestamp
        narrative += f"\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Truncate if too long
        if len(narrative) > max_length:
            narrative = narrative[:max_length - 3] + "..."
        
        # Update the narrative
        self.narrative = narrative
        
        # Log the synthesis
        self._log_audit_event("synthesize_narrative", "Narrative synthesized")
        
        return narrative

    def get_evidence_chain(self, evidence_id: str) -> List[EvidenceItem]:
        """
        Gets the chain of evidence for a specific item.
        
        Args:
            evidence_id: ID of the evidence item
            
        Returns:
            List of evidence items in the chain
        """
        # Find the target evidence item
        target_item = None
        for item in self.evidence_items:
            if item.id == evidence_id:
                target_item = item
                break
        
        if not target_item:
            return []
        
        # Build the chain
        chain = [target_item]
        
        # Follow parent links
        current_id = target_item.parent_id
        while current_id:
            # Find parent
            parent = None
            for item in self.evidence_items:
                if item.id == current_id:
                    parent = item
                    break
            
            if parent:
                chain.insert(0, parent)  # Add to beginning of chain
                current_id = parent.parent_id
            else:
                break  # Parent not found
        
        return chain

    def search_evidence(self, query: str) -> List[EvidenceItem]:
        """
        Searches evidence items for matching text.
        
        Args:
            query: Search query
            
        Returns:
            List of matching evidence items
        """
        results = []
        query = query.lower()
        
        # Search in descriptions
        for item in self.evidence_items:
            if query in item.description.lower():
                results.append(item)
                continue
            
            # Search in raw data if available
            if item.raw_data:
                raw_str = str(item.raw_data).lower()
                if query in raw_str:
                    results.append(item)
                    continue
            
            # Search in tags
            if any(query in tag.lower() for tag in item.tags):
                results.append(item)
                continue
        
        return results

    def calculate_overall_confidence(self) -> float:
        """
        Calculates the overall confidence score for the evidence bundle.
        
        Returns:
            Overall confidence score (0.0-1.0)
        """
        if not self.evidence_items:
            return 0.0
        
        # Weight by quality score
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in self.evidence_items:
            weight = item.quality_score
            weighted_sum += item.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight

    def calculate_uncertainty(self) -> Dict[str, Any]:
        """
        Calculates uncertainty metrics for the evidence bundle.
        
        Returns:
            Dictionary of uncertainty metrics
        """
        if not self.evidence_items:
            return {
                "overall_uncertainty": 1.0,
                "category_uncertainty": {},
                "source_uncertainty": {},
            }
        
        # Calculate overall uncertainty
        overall_confidence = self.calculate_overall_confidence()
        overall_uncertainty = 1.0 - overall_confidence
        
        # Calculate uncertainty by category
        category_uncertainty = {}
        for category in EvidenceCategory:
            items = self.filter_evidence(category=category)
            if items:
                avg_confidence = sum(item.confidence for item in items) / len(items)
                category_uncertainty[category.value] = 1.0 - avg_confidence
            else:
                category_uncertainty[category.value] = 1.0
        
        # Calculate uncertainty by source
        source_uncertainty = {}
        for source in EvidenceSource:
            items = self.filter_evidence(source=source)
            if items:
                avg_confidence = sum(item.confidence for item in items) / len(items)
                source_uncertainty[source.value] = 1.0 - avg_confidence
            else:
                source_uncertainty[source.value] = 1.0
        
        return {
            "overall_uncertainty": overall_uncertainty,
            "category_uncertainty": category_uncertainty,
            "source_uncertainty": source_uncertainty,
        }

    def _log_audit_event(self, event_type: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs an audit event to the audit trail.
        
        Args:
            event_type: Type of event
            description: Description of the event
            details: Additional details
        """
        event = AuditEvent(
            event_type=event_type,
            description=description,
            details=details,
        )
        self.audit_trail.append(event)
        
        # Also publish event to the event system
        publish_event(
            event_type=f"evidence.{event_type}",
            data={
                "investigation_id": self.investigation_id,
                "description": description,
                "details": details,
                "timestamp": event.timestamp.isoformat(),
                "priority": EventPriority.NORMAL.value,
            },
        )

    def to_json(self, include_raw: bool = False) -> str:
        """
        Exports the evidence bundle to JSON.
        
        Args:
            include_raw: Whether to include raw data
            
        Returns:
            JSON string
        """
        # Create a dictionary of the bundle
        data = {
            "investigation_id": self.investigation_id,
            "narrative": self.narrative,
            "summary": self.summary,
            "metadata": self.metadata,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "audit_trail": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "description": event.description,
                    "details": event.details,
                }
                for event in self.audit_trail
            ],
        }
        
        # Include raw data if requested
        if include_raw:
            data["raw_data"] = self.raw_data
        
        # Convert to JSON
        return json.dumps(data, indent=2)

    def to_html(self) -> str:
        """
        Exports the evidence bundle to HTML.
        
        Returns:
            HTML string
        """
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evidence Bundle: {self.investigation_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .evidence-item {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .high-confidence {{ background-color: #d5f5e3; }}
                .medium-confidence {{ background-color: #fdebd0; }}
                .low-confidence {{ background-color: #fadbd8; }}
                .metadata {{ color: #7f8c8d; font-size: 0.9em; }}
                .summary {{ background-color: #eaecee; padding: 15px; border-radius: 5px; }}
                .audit-trail {{ font-size: 0.8em; color: #555; }}
            </style>
        </head>
        <body>
            <h1>Evidence Bundle: {self.investigation_id}</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Evidence Items: {self.summary['total_items']}</p>
                <p>Average Confidence: {self.summary['avg_confidence']:.2f}</p>
                <p>Average Quality Score: {self.summary['avg_quality_score']:.2f}</p>
            </div>
            
            <h2>Narrative</h2>
            <div class="narrative">
                {self.narrative.replace('\n', '<br>')}
            </div>
            
            <h2>Evidence Items</h2>
        """
        
        # Add evidence items
        for item in self.evidence_items:
            # Determine confidence class
            if item.confidence >= 0.8:
                confidence_class = "high-confidence"
            elif item.confidence >= 0.5:
                confidence_class = "medium-confidence"
            else:
                confidence_class = "low-confidence"
            
            html += f"""
            <div class="evidence-item {confidence_class}">
                <h3>{item.description}</h3>
                <div class="metadata">
                    <p>ID: {item.id}</p>
                    <p>Category: {item.category.value}</p>
                    <p>Source: {item.source.value}</p>
                    <p>Confidence: {item.confidence:.2f}</p>
                    <p>Quality Score: {item.quality_score:.2f}</p>
                    <p>Timestamp: {item.timestamp.isoformat()}</p>
                    <p>Tags: {', '.join(item.tags) if item.tags else 'None'}</p>
                </div>
            </div>
            """
        
        # Add audit trail
        html += """
            <h2>Audit Trail</h2>
            <table class="audit-trail" border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Timestamp</th>
                    <th>Event Type</th>
                    <th>Description</th>
                </tr>
        """
        
        for event in self.audit_trail:
            html += f"""
                <tr>
                    <td>{event.timestamp.isoformat()}</td>
                    <td>{event.event_type}</td>
                    <td>{event.description}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="footer">
                <p>Generated on {}</p>
            </div>
        </body>
        </html>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html

    def to_pdf(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Exports the evidence bundle to PDF.
        
        Args:
            output_path: Path to save the PDF file
            
        Returns:
            Path to the PDF file or None if export failed
        """
        try:
            # This would typically use a PDF generation library
            # For now, we'll just generate HTML and mention the conversion
            html = self.to_html()
            
            # If output path not provided, use a default
            if not output_path:
                output_dir = Path("./exports")
                output_dir.mkdir(exist_ok=True)
                output_path = str(output_dir / f"evidence_bundle_{self.investigation_id}.pdf")
            
            # In a real implementation, we would convert HTML to PDF here
            # For example, using a library like weasyprint or pdfkit
            
            logger.info(f"PDF export would be saved to: {output_path}")
            
            # Log the export
            self._log_audit_event("export_pdf", f"Exported to PDF: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts evidence items to a pandas DataFrame.
        
        Returns:
            DataFrame of evidence items
        """
        # Convert evidence items to dictionaries
        data = []
        for item in self.evidence_items:
            item_dict = item.to_dict()
            # Flatten some nested structures for better DataFrame representation
            if "raw_data" in item_dict and item_dict["raw_data"]:
                item_dict["has_raw_data"] = True
            else:
                item_dict["has_raw_data"] = False
                item_dict.pop("raw_data", None)
            
            data.append(item_dict)
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
        else:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                "id", "category", "description", "timestamp", "source",
                "confidence", "quality_score", "provenance_link", "tags",
                "parent_id", "has_raw_data"
            ])
        
        return df

    def merge(self, other_bundle: "EvidenceBundle") -> "EvidenceBundle":
        """
        Merges another evidence bundle into this one.
        
        Args:
            other_bundle: Other evidence bundle to merge
            
        Returns:
            Self for chaining
        """
        # Add evidence items
        for item in other_bundle.evidence_items:
            if item.id not in {existing.id for existing in self.evidence_items}:
                self.evidence_items.append(item)
        
        # Add raw data
        self.raw_data.extend(other_bundle.raw_data)
        
        # Add audit trail
        self.audit_trail.extend(other_bundle.audit_trail)
        
        # Update summary
        self._update_summary()
        
        # Log the merge
        self._log_audit_event(
            "merge_bundle", 
            f"Merged with bundle: {other_bundle.investigation_id}",
            {"merged_bundle_id": other_bundle.investigation_id}
        )
        
        return self

    @classmethod
    def from_json(cls, json_str: str) -> "EvidenceBundle":
        """
        Creates an evidence bundle from JSON.
        
        Args:
            json_str: JSON string
            
        Returns:
            Evidence bundle
        """
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Create evidence items
            evidence_items = []
            for item_data in data.get("evidence_items", []):
                # Determine evidence type based on category
                category = item_data.get("category")
                
                if category == EvidenceCategory.TRANSACTION.value:
                    item = TransactionEvidence.from_dict(item_data)
                elif category == EvidenceCategory.PATTERN_MATCH.value:
                    item = PatternMatchEvidence.from_dict(item_data)
                elif category == EvidenceCategory.ANOMALY.value:
                    item = AnomalyEvidence.from_dict(item_data)
                elif category == EvidenceCategory.ENTITY_RELATIONSHIP.value:
                    item = RelationshipEvidence.from_dict(item_data)
                elif category == EvidenceCategory.GRAPH_ELEMENT.value:
                    item = GraphElementEvidence.from_dict(item_data)
                elif category == EvidenceCategory.EXTERNAL_SOURCE.value:
                    item = ExternalSourceEvidence.from_dict(item_data)
                else:
                    item = EvidenceItem.from_dict(item_data)
                
                evidence_items.append(item)
            
            # Create audit trail
            audit_trail = []
            for event_data in data.get("audit_trail", []):
                timestamp = datetime.fromisoformat(event_data["timestamp"])
                audit_trail.append(AuditEvent(
                    timestamp=timestamp,
                    event_type=event_data["event_type"],
                    description=event_data["description"],
                    details=event_data.get("details"),
                ))
            
            # Create bundle
            bundle = cls(
                narrative=data.get("narrative", ""),
                evidence_items=evidence_items,
                raw_data=data.get("raw_data", []),
                audit_trail=audit_trail,
                investigation_id=data.get("investigation_id"),
                metadata=data.get("metadata", {}),
            )
            
            return bundle
        
        except Exception as e:
            logger.error(f"Error creating evidence bundle from JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")


def create_evidence_bundle(
    narrative: str = "",
    investigation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    graph_rag_service: Optional[GraphRAG] = None,
) -> EvidenceBundle:
    """
    Creates a new evidence bundle.
    
    Args:
        narrative: Initial narrative
        investigation_id: Investigation ID
        metadata: Additional metadata
        graph_rag_service: Graph RAG service for contextual evidence
        
    Returns:
        New evidence bundle
    """
    return EvidenceBundle(
        narrative=narrative,
        investigation_id=investigation_id,
        metadata=metadata,
        graph_rag_service=graph_rag_service,
    )


def create_transaction_evidence(
    tx_hash: str,
    chain: str,
    from_address: str,
    to_address: str,
    amount: Optional[float] = None,
    asset: Optional[str] = None,
    description: Optional[str] = None,
    source: EvidenceSource = EvidenceSource.SIM_API,
    confidence: float = 1.0,
    raw_data: Optional[Dict[str, Any]] = None,
    provenance_link: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> TransactionEvidence:
    """
    Creates transaction evidence.
    
    Args:
        tx_hash: Transaction hash
        chain: Blockchain chain
        from_address: Sender address
        to_address: Recipient address
        amount: Transaction amount
        asset: Asset type
        description: Evidence description
        source: Evidence source
        confidence: Confidence score
        raw_data: Raw data
        provenance_link: Link to source data
        tags: Evidence tags
        
    Returns:
        Transaction evidence
    """
    # Generate description if not provided
    if description is None:
        asset_str = f" {asset}" if asset else ""
        amount_str = f" {amount}{asset_str}" if amount is not None else ""
        description = f"Transaction {tx_hash} from {from_address} to {to_address}{amount_str} on {chain}"
    
    return TransactionEvidence(
        tx_hash=tx_hash,
        chain=chain,
        from_address=from_address,
        to_address=to_address,
        amount=amount,
        asset=asset,
        description=description,
        source=source,
        confidence=confidence,
        raw_data=raw_data,
        provenance_link=provenance_link,
        tags=tags or [],
    )


def create_pattern_match_evidence(
    pattern_name: str,
    matched_entities: List[str],
    pattern_description: Optional[str] = None,
    description: Optional[str] = None,
    source: EvidenceSource = EvidenceSource.GRAPH_ANALYSIS,
    confidence: float = 0.7,
    raw_data: Optional[Dict[str, Any]] = None,
    provenance_link: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> PatternMatchEvidence:
    """
    Creates pattern match evidence.
    
    Args:
        pattern_name: Name of the pattern
        matched_entities: List of matched entities
        pattern_description: Description of the pattern
        description: Evidence description
        source: Evidence source
        confidence: Confidence score
        raw_data: Raw data
        provenance_link: Link to source data
        tags: Evidence tags
        
    Returns:
        Pattern match evidence
    """
    # Generate description if not provided
    if description is None:
        entity_str = ", ".join(matched_entities[:3])
        if len(matched_entities) > 3:
            entity_str += f", and {len(matched_entities) - 3} more"
        description = f"Pattern '{pattern_name}' matched entities: {entity_str}"
    
    return PatternMatchEvidence(
        pattern_name=pattern_name,
        matched_entities=matched_entities,
        pattern_description=pattern_description,
        description=description,
        source=source,
        confidence=confidence,
        raw_data=raw_data,
        provenance_link=provenance_link,
        tags=tags or [],
    )


def create_anomaly_evidence(
    anomaly_type: str,
    severity: str,
    affected_entities: List[str],
    description: Optional[str] = None,
    source: EvidenceSource = EvidenceSource.GRAPH_ANALYSIS,
    confidence: float = 0.6,
    raw_data: Optional[Dict[str, Any]] = None,
    provenance_link: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> AnomalyEvidence:
    """
    Creates anomaly evidence.
    
    Args:
        anomaly_type: Type of anomaly
        severity: Severity of the anomaly
        affected_entities: List of affected entities
        description: Evidence description
        source: Evidence source
        confidence: Confidence score
        raw_data: Raw data
        provenance_link: Link to source data
        tags: Evidence tags
        
    Returns:
        Anomaly evidence
    """
    # Generate description if not provided
    if description is None:
        entity_str = ", ".join(affected_entities[:3])
        if len(affected_entities) > 3:
            entity_str += f", and {len(affected_entities) - 3} more"
        description = f"{severity.capitalize()} {anomaly_type} anomaly detected in entities: {entity_str}"
    
    return AnomalyEvidence(
        anomaly_type=anomaly_type,
        severity=severity,
        affected_entities=affected_entities,
        description=description,
        source=source,
        confidence=confidence,
        raw_data=raw_data,
        provenance_link=provenance_link,
        tags=tags or [],
    )
