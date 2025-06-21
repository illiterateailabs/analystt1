"""
Fraud Detection Prompt Templates

This module provides a comprehensive prompt template system for fraud detection agents,
with support for dynamic prompt generation, context-awareness, multi-language support,
chain-specific variations, and evidence-based prompting.

Templates are organized by agent role and can be customized based on investigation state,
blockchain ecosystem, and available evidence.
"""

import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import yaml
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pydantic import BaseModel, Field, validator

from backend.core.events import publish_event

# Configure module logger
logger = logging.getLogger(__name__)

# Default template directories
TEMPLATES_DIR = Path(__file__).parent / "templates"
CUSTOM_TEMPLATES_DIR = Path(__file__).parent / "templates" / "custom"

# Ensure template directories exist
TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)
CUSTOM_TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)


class ChainType(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"
    SOLANA = "solana"
    BINANCE = "binance"
    UNKNOWN = "unknown"


class AgentRole(str, Enum):
    """Agent roles in fraud detection."""
    LEAD_INVESTIGATOR = "lead_investigator"
    PATTERN_ANALYST = "pattern_analyst"
    DATA_ANALYST = "data_analyst"
    REPORT_GENERATOR = "report_generator"


class PromptLanguage(str, Enum):
    """Supported languages for prompts."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class InvestigationStage(str, Enum):
    """Stages of a fraud investigation."""
    INITIAL = "initial"
    DATA_GATHERING = "data_gathering"
    PATTERN_ANALYSIS = "pattern_analysis"
    TRANSACTION_FLOW = "transaction_flow"
    ENTITY_CLUSTERING = "entity_clustering"
    EVIDENCE_COLLECTION = "evidence_collection"
    REPORT_GENERATION = "report_generation"
    FINAL = "final"


class EvidenceType(str, Enum):
    """Types of evidence in fraud investigations."""
    TRANSACTION = "transaction"
    PATTERN_MATCH = "pattern_match"
    ANOMALY = "anomaly"
    ENTITY_RELATIONSHIP = "entity_relationship"
    TEMPORAL_CORRELATION = "temporal_correlation"
    EXTERNAL_SOURCE = "external_source"


class PromptTemplate(BaseModel):
    """Base model for prompt templates."""
    name: str
    description: str
    template: str
    role: AgentRole
    language: PromptLanguage = PromptLanguage.ENGLISH
    chain_specific: Optional[ChainType] = None
    stage: Optional[InvestigationStage] = None
    requires_evidence: bool = False
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    
    @validator('template')
    def validate_template(cls, v):
        """Validate that the template contains valid Jinja2 syntax."""
        try:
            env = Environment(autoescape=select_autoescape())
            env.parse(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid template syntax: {e}")


class EvidenceBundle(BaseModel):
    """Bundle of evidence for fraud investigations."""
    transactions: List[Dict[str, Any]] = Field(default_factory=list)
    pattern_matches: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    entity_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    external_sources: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get_evidence_by_type(self, evidence_type: EvidenceType) -> List[Dict[str, Any]]:
        """Get evidence of a specific type."""
        if evidence_type == EvidenceType.TRANSACTION:
            return self.transactions
        elif evidence_type == EvidenceType.PATTERN_MATCH:
            return self.pattern_matches
        elif evidence_type == EvidenceType.ANOMALY:
            return self.anomalies
        elif evidence_type == EvidenceType.ENTITY_RELATIONSHIP:
            return self.entity_relationships
        elif evidence_type == EvidenceType.EXTERNAL_SOURCE:
            return self.external_sources
        else:
            return []
    
    def add_evidence(self, evidence_type: EvidenceType, evidence: Dict[str, Any]) -> None:
        """Add evidence to the bundle."""
        if evidence_type == EvidenceType.TRANSACTION:
            self.transactions.append(evidence)
        elif evidence_type == EvidenceType.PATTERN_MATCH:
            self.pattern_matches.append(evidence)
        elif evidence_type == EvidenceType.ANOMALY:
            self.anomalies.append(evidence)
        elif evidence_type == EvidenceType.ENTITY_RELATIONSHIP:
            self.entity_relationships.append(evidence)
        elif evidence_type == EvidenceType.EXTERNAL_SOURCE:
            self.external_sources.append(evidence)
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of the evidence bundle."""
        return {
            "transactions": len(self.transactions),
            "pattern_matches": len(self.pattern_matches),
            "anomalies": len(self.anomalies),
            "entity_relationships": len(self.entity_relationships),
            "external_sources": len(self.external_sources),
            "total": (
                len(self.transactions) +
                len(self.pattern_matches) +
                len(self.anomalies) +
                len(self.entity_relationships) +
                len(self.external_sources)
            )
        }


class PromptTemplateManager:
    """Manager for prompt templates with dynamic generation capabilities."""
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        custom_templates_dir: Optional[Path] = None,
        language: PromptLanguage = PromptLanguage.ENGLISH,
    ):
        """
        Initialize the prompt template manager.
        
        Args:
            templates_dir: Directory containing base templates
            custom_templates_dir: Directory containing custom templates
            language: Default language for templates
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.custom_templates_dir = custom_templates_dir or CUSTOM_TEMPLATES_DIR
        self.language = language
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader([
                self.templates_dir,
                self.custom_templates_dir,
            ]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Load templates
        self.templates = self._load_templates()
        
        logger.info(f"Loaded {len(self.templates)} prompt templates")
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """
        Load templates from template directories.
        
        Returns:
            Dictionary of templates by name
        """
        templates = {}
        
        # Load base templates
        base_templates = self._load_templates_from_dir(self.templates_dir)
        templates.update(base_templates)
        
        # Load custom templates (overriding base templates if needed)
        custom_templates = self._load_templates_from_dir(self.custom_templates_dir)
        templates.update(custom_templates)
        
        return templates
    
    def _load_templates_from_dir(self, directory: Path) -> Dict[str, PromptTemplate]:
        """
        Load templates from a directory.
        
        Args:
            directory: Directory containing templates
            
        Returns:
            Dictionary of templates by name
        """
        templates = {}
        
        if not directory.exists():
            return templates
        
        # Load YAML template definitions
        for yaml_file in directory.glob("*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    template_defs = yaml.safe_load(f)
                
                if not isinstance(template_defs, list):
                    template_defs = [template_defs]
                
                for template_def in template_defs:
                    try:
                        template = PromptTemplate(**template_def)
                        templates[template.name] = template
                        logger.debug(f"Loaded template: {template.name}")
                    except Exception as e:
                        logger.error(f"Error loading template from {yaml_file}: {e}")
            
            except Exception as e:
                logger.error(f"Error loading templates from {yaml_file}: {e}")
        
        # Load template files
        for template_file in directory.glob("*.j2"):
            try:
                template_name = template_file.stem
                
                # Skip if already loaded from YAML
                if template_name in templates:
                    continue
                
                with open(template_file, "r") as f:
                    template_content = f.read()
                
                # Extract metadata from template comments
                metadata = self._extract_template_metadata(template_content)
                
                # Create template
                template = PromptTemplate(
                    name=template_name,
                    description=metadata.get("description", f"Template: {template_name}"),
                    template=template_content,
                    role=metadata.get("role", AgentRole.LEAD_INVESTIGATOR),
                    language=metadata.get("language", PromptLanguage.ENGLISH),
                    chain_specific=metadata.get("chain_specific"),
                    stage=metadata.get("stage"),
                    requires_evidence=metadata.get("requires_evidence", False),
                    version=metadata.get("version", "1.0.0"),
                    tags=metadata.get("tags", []),
                )
                
                templates[template_name] = template
                logger.debug(f"Loaded template file: {template_name}")
            
            except Exception as e:
                logger.error(f"Error loading template file {template_file}: {e}")
        
        return templates
    
    def _extract_template_metadata(self, template_content: str) -> Dict[str, Any]:
        """
        Extract metadata from template comments.
        
        Args:
            template_content: Template content
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Look for metadata in comments at the beginning of the file
        # Format: {# key: value #}
        lines = template_content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{#") and line.endswith("#}"):
                # Extract key-value pair
                content = line[2:-2].strip()
                if ":" in content:
                    key, value = content.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Convert to appropriate type
                    if key == "role":
                        try:
                            value = AgentRole(value.lower())
                        except ValueError:
                            value = AgentRole.LEAD_INVESTIGATOR
                    elif key == "language":
                        try:
                            value = PromptLanguage(value.lower())
                        except ValueError:
                            value = PromptLanguage.ENGLISH
                    elif key == "chain_specific":
                        try:
                            value = ChainType(value.lower())
                        except ValueError:
                            value = None
                    elif key == "stage":
                        try:
                            value = InvestigationStage(value.lower())
                        except ValueError:
                            value = None
                    elif key == "requires_evidence":
                        value = value.lower() in ("true", "yes", "1")
                    elif key == "tags":
                        value = [tag.strip() for tag in value.split(",")]
                    
                    metadata[key] = value
            elif not line.startswith("{#"):
                # Stop processing when we reach non-comment lines
                break
        
        return metadata
    
    def get_template(
        self,
        template_name: str,
        role: Optional[AgentRole] = None,
        language: Optional[PromptLanguage] = None,
        chain: Optional[ChainType] = None,
        stage: Optional[InvestigationStage] = None,
    ) -> Optional[PromptTemplate]:
        """
        Get a template by name with optional filters.
        
        Args:
            template_name: Name of the template
            role: Filter by agent role
            language: Filter by language
            chain: Filter by blockchain
            stage: Filter by investigation stage
            
        Returns:
            Matching template or None if not found
        """
        # First try exact match
        if template_name in self.templates:
            template = self.templates[template_name]
            
            # Check if template matches filters
            if role and template.role != role:
                logger.debug(f"Template {template_name} role {template.role} doesn't match {role}")
                return None
            
            if language and template.language != language:
                logger.debug(f"Template {template_name} language {template.language} doesn't match {language}")
                return None
            
            if chain and template.chain_specific and template.chain_specific != chain:
                logger.debug(f"Template {template_name} chain {template.chain_specific} doesn't match {chain}")
                return None
            
            if stage and template.stage and template.stage != stage:
                logger.debug(f"Template {template_name} stage {template.stage} doesn't match {stage}")
                return None
            
            return template
        
        # Try to find template with role, language, chain, and stage
        template_variants = []
        
        for t in self.templates.values():
            if t.name.startswith(f"{template_name}_"):
                # Check if template matches filters
                if role and t.role != role:
                    continue
                
                if language and t.language != language:
                    continue
                
                if chain and t.chain_specific and t.chain_specific != chain:
                    continue
                
                if stage and t.stage and t.stage != stage:
                    continue
                
                template_variants.append(t)
        
        if template_variants:
            # Sort by specificity (more specific templates first)
            template_variants.sort(
                key=lambda t: (
                    t.role == role if role else False,
                    t.language == language if language else False,
                    t.chain_specific == chain if chain else False,
                    t.stage == stage if stage else False,
                ),
                reverse=True,
            )
            
            return template_variants[0]
        
        logger.warning(f"Template not found: {template_name}")
        return None
    
    def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        role: Optional[AgentRole] = None,
        language: Optional[PromptLanguage] = None,
        chain: Optional[ChainType] = None,
        stage: Optional[InvestigationStage] = None,
        evidence: Optional[EvidenceBundle] = None,
    ) -> str:
        """
        Render a template with variables.
        
        Args:
            template_name: Name of the template
            variables: Variables to substitute in the template
            role: Filter by agent role
            language: Filter by language
            chain: Filter by blockchain
            stage: Filter by investigation stage
            evidence: Evidence bundle for evidence-based prompting
            
        Returns:
            Rendered template
            
        Raises:
            ValueError: If template not found or rendering fails
        """
        # Get template
        template = self.get_template(
            template_name=template_name,
            role=role,
            language=language or self.language,
            chain=chain,
            stage=stage,
        )
        
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Check if template requires evidence
        if template.requires_evidence and not evidence:
            logger.warning(f"Template {template_name} requires evidence but none provided")
        
        # Prepare variables
        render_vars = variables.copy()
        
        # Add evidence if available
        if evidence:
            render_vars["evidence"] = evidence
            render_vars["evidence_summary"] = evidence.get_summary()
        
        # Add context variables
        render_vars["role"] = role.value if role else template.role.value
        render_vars["language"] = language.value if language else template.language.value
        render_vars["chain"] = chain.value if chain else (template.chain_specific.value if template.chain_specific else "unknown")
        render_vars["stage"] = stage.value if stage else (template.stage.value if template.stage else "unknown")
        
        try:
            # Render template
            jinja_template = self.env.from_string(template.template)
            rendered = jinja_template.render(**render_vars)
            
            return rendered
        
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise ValueError(f"Error rendering template {template_name}: {e}")
    
    def create_template(self, template: PromptTemplate) -> bool:
        """
        Create a new template.
        
        Args:
            template: Template to create
            
        Returns:
            True if template was created, False otherwise
        """
        if template.name in self.templates:
            logger.warning(f"Template {template.name} already exists")
            return False
        
        try:
            # Validate template
            template.validate_template(template.template)
            
            # Add to templates
            self.templates[template.name] = template
            
            # Save to custom templates directory
            self._save_template(template)
            
            logger.info(f"Created template: {template.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating template {template.name}: {e}")
            return False
    
    def update_template(self, template: PromptTemplate) -> bool:
        """
        Update an existing template.
        
        Args:
            template: Template to update
            
        Returns:
            True if template was updated, False otherwise
        """
        if template.name not in self.templates:
            logger.warning(f"Template {template.name} not found")
            return False
        
        try:
            # Validate template
            template.validate_template(template.template)
            
            # Update template
            self.templates[template.name] = template
            
            # Save to custom templates directory
            self._save_template(template)
            
            logger.info(f"Updated template: {template.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating template {template.name}: {e}")
            return False
    
    def delete_template(self, template_name: str) -> bool:
        """
        Delete a template.
        
        Args:
            template_name: Name of the template to delete
            
        Returns:
            True if template was deleted, False otherwise
        """
        if template_name not in self.templates:
            logger.warning(f"Template {template_name} not found")
            return False
        
        try:
            # Remove from templates
            template = self.templates.pop(template_name)
            
            # Remove from custom templates directory
            template_file = self.custom_templates_dir / f"{template_name}.yaml"
            if template_file.exists():
                template_file.unlink()
            
            logger.info(f"Deleted template: {template_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting template {template_name}: {e}")
            return False
    
    def _save_template(self, template: PromptTemplate) -> None:
        """
        Save a template to the custom templates directory.
        
        Args:
            template: Template to save
        """
        # Create custom templates directory if it doesn't exist
        self.custom_templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Save as YAML
        template_file = self.custom_templates_dir / f"{template.name}.yaml"
        
        with open(template_file, "w") as f:
            yaml.dump(template.dict(), f, default_flow_style=False)


# Base templates for different agent roles

LEAD_INVESTIGATOR_TEMPLATE = """
{# role: lead_investigator #}
{# description: Base template for the Lead Investigator agent #}
{# version: 1.0.0 #}
{# tags: base, investigator #}

You are a Lead Fraud Investigator specializing in blockchain analysis. Your task is to coordinate 
the investigation into potential fraud activity and synthesize findings from your team.

## Investigation Context
{{ context }}

## Current Stage: {{ stage }}

{% if evidence %}
## Evidence Summary
- Transactions: {{ evidence_summary.transactions }}
- Pattern Matches: {{ evidence_summary.pattern_matches }}
- Anomalies: {{ evidence_summary.anomalies }}
- Entity Relationships: {{ evidence_summary.entity_relationships }}
- External Sources: {{ evidence_summary.external_sources }}
{% endif %}

## Your Task
{{ task_description }}

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Instructions
1. Review the investigation context and available evidence
2. Coordinate with specialized agents as needed
3. Analyze patterns and connections in the data
4. Synthesize findings into a coherent narrative
5. Provide clear, actionable recommendations
6. Assign confidence levels to your conclusions

Remember to maintain a methodical approach and document your reasoning process.
"""

PATTERN_ANALYST_TEMPLATE = """
{# role: pattern_analyst #}
{# description: Base template for the Pattern Analyst agent #}
{# version: 1.0.0 #}
{# tags: base, analyst #}

You are a Fraud Pattern Analyst specializing in blockchain transaction patterns. Your task is to 
identify known fraud patterns and anomalies in the provided data.

## Analysis Context
{{ context }}

## Current Stage: {{ stage }}

{% if evidence %}
## Available Evidence
{% if evidence_summary.transactions > 0 %}
### Transactions ({{ evidence_summary.transactions }})
{% for tx in evidence.transactions[:5] %}
- {{ tx.tx_hash }}: {{ tx.from_address }} → {{ tx.to_address }}, {{ tx.value }} {{ tx.asset }}, {{ tx.timestamp }}
{% endfor %}
{% if evidence_summary.transactions > 5 %}...and {{ evidence_summary.transactions - 5 }} more{% endif %}
{% endif %}

{% if evidence_summary.pattern_matches > 0 %}
### Pattern Matches ({{ evidence_summary.pattern_matches }})
{% for match in evidence.pattern_matches[:5] %}
- {{ match.pattern_name }} (confidence: {{ match.confidence }}): {{ match.description }}
{% endfor %}
{% if evidence_summary.pattern_matches > 5 %}...and {{ evidence_summary.pattern_matches - 5 }} more{% endif %}
{% endif %}

{% if evidence_summary.anomalies > 0 %}
### Anomalies ({{ evidence_summary.anomalies }})
{% for anomaly in evidence.anomalies[:5] %}
- {{ anomaly.type }}: {{ anomaly.description }} (severity: {{ anomaly.severity }})
{% endfor %}
{% if evidence_summary.anomalies > 5 %}...and {{ evidence_summary.anomalies - 5 }} more{% endif %}
{% endif %}
{% endif %}

## Your Task
{{ task_description }}

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Known Patterns to Check
{% for pattern in patterns %}
- {{ pattern.name }}: {{ pattern.description }}
{% endfor %}

## Instructions
1. Analyze the transaction data for known fraud patterns
2. Identify anomalies that don't match known patterns
3. Calculate confidence scores for each pattern match
4. Provide detailed reasoning for your findings
5. Suggest follow-up analyses if needed

Be thorough in your analysis and provide specific examples from the data to support your conclusions.
"""

DATA_ANALYST_TEMPLATE = """
{# role: data_analyst #}
{# description: Base template for the Data Analyst agent #}
{# version: 1.0.0 #}
{# tags: base, analyst #}

You are an On-Chain Data Analyst specializing in blockchain data retrieval and processing. Your task is to 
gather and analyze raw blockchain data to support the fraud investigation.

## Data Context
{{ context }}

## Current Stage: {{ stage }}

## Your Task
{{ task_description }}

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Data Sources
{% for source in data_sources %}
- {{ source.name }}: {{ source.description }}
{% endfor %}

## Instructions
1. Retrieve relevant blockchain data based on the investigation parameters
2. Process and normalize the data for analysis
3. Identify key entities and relationships in the data
4. Track fund flows across addresses and chains
5. Provide structured data outputs for pattern analysis

Focus on data completeness and accuracy. Missing or incorrect data could lead to false conclusions.
"""

REPORT_GENERATOR_TEMPLATE = """
{# role: report_generator #}
{# description: Base template for the Report Generator agent #}
{# version: 1.0.0 #}
{# tags: base, report #}
{# requires_evidence: true #}

You are a Fraud Report Generator specializing in creating comprehensive fraud reports. Your task is to 
synthesize investigation findings into a clear, structured report with supporting evidence.

## Report Context
{{ context }}

## Current Stage: {{ stage }}

{% if evidence %}
## Evidence Summary
- Transactions: {{ evidence_summary.transactions }}
- Pattern Matches: {{ evidence_summary.pattern_matches }}
- Anomalies: {{ evidence_summary.anomalies }}
- Entity Relationships: {{ evidence_summary.entity_relationships }}
- External Sources: {{ evidence_summary.external_sources }}

## Key Evidence
{% if evidence_summary.pattern_matches > 0 %}
### Pattern Matches
{% for match in evidence.pattern_matches %}
- {{ match.pattern_name }} (confidence: {{ match.confidence }}): {{ match.description }}
{% endfor %}
{% endif %}

{% if evidence_summary.anomalies > 0 %}
### Anomalies
{% for anomaly in evidence.anomalies %}
- {{ anomaly.type }}: {{ anomaly.description }} (severity: {{ anomaly.severity }})
{% endfor %}
{% endif %}
{% endif %}

## Your Task
{{ task_description }}

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Report Requirements
- Executive Summary: Brief overview of findings
- Methodology: How the investigation was conducted
- Findings: Detailed analysis of fraud patterns and evidence
- Evidence: Supporting data and analysis
- Recommendations: Suggested actions based on findings
- Confidence Assessment: Evaluation of certainty in conclusions

## Instructions
1. Review all evidence and investigation findings
2. Organize findings into a coherent narrative
3. Include specific examples and evidence to support conclusions
4. Provide clear, actionable recommendations
5. Assess confidence levels for all findings
6. Format the report for the intended audience

Ensure the report is factual, objective, and supported by evidence. Avoid speculation and clearly 
distinguish between facts and interpretations.
"""

# Chain-specific template for Ethereum
ETHEREUM_PATTERN_ANALYST_TEMPLATE = """
{# role: pattern_analyst #}
{# description: Ethereum-specific template for the Pattern Analyst agent #}
{# version: 1.0.0 #}
{# tags: ethereum, analyst #}
{# chain_specific: ethereum #}

You are a Fraud Pattern Analyst specializing in Ethereum blockchain transaction patterns. Your task is to 
identify known fraud patterns and anomalies in the provided Ethereum data.

## Analysis Context
{{ context }}

## Current Stage: {{ stage }}

{% if evidence %}
## Available Evidence
{% if evidence_summary.transactions > 0 %}
### Transactions ({{ evidence_summary.transactions }})
{% for tx in evidence.transactions[:5] %}
- {{ tx.tx_hash }}: {{ tx.from_address }} → {{ tx.to_address }}, {{ tx.value }} ETH, {{ tx.timestamp }}
{% endfor %}
{% if evidence_summary.transactions > 5 %}...and {{ evidence_summary.transactions - 5 }} more{% endif %}
{% endif %}
{% endif %}

## Your Task
{{ task_description }}

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Ethereum-Specific Patterns to Check
- Flash Loan Attacks: Look for large loans and repayments in the same transaction
- MEV Exploitation: Check for sandwich attacks and front-running
- Reentrancy Attacks: Look for multiple calls to the same contract function
- Token Approval Exploits: Check for unlimited token approvals followed by transfers
- Honeypot Contracts: Look for contracts with hidden backdoors or traps
- Wash Trading: Check for circular trading patterns between related addresses
- Airdrop Farming: Look for accounts created solely to receive airdrops
- Phishing Contracts: Check for contracts impersonating legitimate protocols

## Instructions
1. Analyze the Ethereum transaction data for known fraud patterns
2. Pay special attention to smart contract interactions and internal transactions
3. Check for gas price anomalies that might indicate MEV or front-running
4. Look for ERC-20 and ERC-721 token transfers that might be suspicious
5. Calculate confidence scores for each pattern match
6. Provide detailed reasoning for your findings

Be thorough in your analysis and provide specific examples from the data to support your conclusions.
"""

# Create template manager with base templates
template_manager = PromptTemplateManager()

# Initialize with base templates if they don't exist
if not template_manager.templates:
    # Create base templates
    template_manager.create_template(PromptTemplate(
        name="lead_investigator_base",
        description="Base template for the Lead Investigator agent",
        template=LEAD_INVESTIGATOR_TEMPLATE,
        role=AgentRole.LEAD_INVESTIGATOR,
    ))
    
    template_manager.create_template(PromptTemplate(
        name="pattern_analyst_base",
        description="Base template for the Pattern Analyst agent",
        template=PATTERN_ANALYST_TEMPLATE,
        role=AgentRole.PATTERN_ANALYST,
    ))
    
    template_manager.create_template(PromptTemplate(
        name="data_analyst_base",
        description="Base template for the Data Analyst agent",
        template=DATA_ANALYST_TEMPLATE,
        role=AgentRole.DATA_ANALYST,
    ))
    
    template_manager.create_template(PromptTemplate(
        name="report_generator_base",
        description="Base template for the Report Generator agent",
        template=REPORT_GENERATOR_TEMPLATE,
        role=AgentRole.REPORT_GENERATOR,
        requires_evidence=True,
    ))
    
    # Create chain-specific templates
    template_manager.create_template(PromptTemplate(
        name="pattern_analyst_ethereum",
        description="Ethereum-specific template for the Pattern Analyst agent",
        template=ETHEREUM_PATTERN_ANALYST_TEMPLATE,
        role=AgentRole.PATTERN_ANALYST,
        chain_specific=ChainType.ETHEREUM,
    ))


def get_template_manager(
    language: PromptLanguage = PromptLanguage.ENGLISH,
    custom_templates_dir: Optional[Path] = None,
) -> PromptTemplateManager:
    """
    Get a template manager with the specified language.
    
    Args:
        language: Language for templates
        custom_templates_dir: Directory containing custom templates
        
    Returns:
        Template manager instance
    """
    return PromptTemplateManager(
        language=language,
        custom_templates_dir=custom_templates_dir,
    )


def get_agent_prompt(
    role: AgentRole,
    context: Dict[str, Any],
    task_description: str,
    language: PromptLanguage = PromptLanguage.ENGLISH,
    chain: Optional[ChainType] = None,
    stage: Optional[InvestigationStage] = None,
    evidence: Optional[EvidenceBundle] = None,
) -> str:
    """
    Get a prompt for an agent based on role and context.
    
    Args:
        role: Agent role
        context: Context variables
        task_description: Description of the task
        language: Language for the prompt
        chain: Blockchain network
        stage: Investigation stage
        evidence: Evidence bundle
        
    Returns:
        Rendered prompt
    """
    # Get template name based on role
    template_name = f"{role.value}_base"
    
    # Add variables
    variables = context.copy()
    variables["task_description"] = task_description
    
    # Render template
    try:
        return template_manager.render_template(
            template_name=template_name,
            variables=variables,
            role=role,
            language=language,
            chain=chain,
            stage=stage,
            evidence=evidence,
        )
    except ValueError:
        # Fall back to base template
        logger.warning(f"Template {template_name} not found, falling back to base template")
        
        # Use appropriate base template based on role
        if role == AgentRole.LEAD_INVESTIGATOR:
            template = LEAD_INVESTIGATOR_TEMPLATE
        elif role == AgentRole.PATTERN_ANALYST:
            template = PATTERN_ANALYST_TEMPLATE
        elif role == AgentRole.DATA_ANALYST:
            template = DATA_ANALYST_TEMPLATE
        elif role == AgentRole.REPORT_GENERATOR:
            template = REPORT_GENERATOR_TEMPLATE
        else:
            template = LEAD_INVESTIGATOR_TEMPLATE
        
        # Create Jinja2 template
        env = Environment(autoescape=select_autoescape())
        jinja_template = env.from_string(template)
        
        # Render template
        return jinja_template.render(**variables)


def get_crew_prompts(
    crew_config: Dict[str, Any],
    language: PromptLanguage = PromptLanguage.ENGLISH,
) -> Dict[str, str]:
    """
    Get prompts for all agents in a crew.
    
    Args:
        crew_config: Crew configuration
        language: Language for prompts
        
    Returns:
        Dictionary of prompts by agent ID
    """
    prompts = {}
    
    # Get agents from crew configuration
    agents = crew_config.get("agents", {})
    
    for agent_id, agent_config in agents.items():
        # Get agent role
        try:
            role = AgentRole(agent_id)
        except ValueError:
            # Use role from config or default to lead investigator
            role_str = agent_config.get("role", "").lower()
            if "investigator" in role_str:
                role = AgentRole.LEAD_INVESTIGATOR
            elif "pattern" in role_str:
                role = AgentRole.PATTERN_ANALYST
            elif "data" in role_str:
                role = AgentRole.DATA_ANALYST
            elif "report" in role_str:
                role = AgentRole.REPORT_GENERATOR
            else:
                role = AgentRole.LEAD_INVESTIGATOR
        
        # Get context from agent configuration
        context = {
            "context": agent_config.get("description", ""),
            "tools": agent_config.get("tools", []),
        }
        
        # Get task description from agent goals
        goals = agent_config.get("goals", [])
        task_description = "\n".join(f"- {goal}" for goal in goals)
        
        # Get prompt
        prompt = get_agent_prompt(
            role=role,
            context=context,
            task_description=task_description,
            language=language,
        )
        
        prompts[agent_id] = prompt
    
    return prompts


def create_evidence_bundle(
    transactions: Optional[List[Dict[str, Any]]] = None,
    pattern_matches: Optional[List[Dict[str, Any]]] = None,
    anomalies: Optional[List[Dict[str, Any]]] = None,
    entity_relationships: Optional[List[Dict[str, Any]]] = None,
    external_sources: Optional[List[Dict[str, Any]]] = None,
) -> EvidenceBundle:
    """
    Create an evidence bundle for fraud investigations.
    
    Args:
        transactions: List of transactions
        pattern_matches: List of pattern matches
        anomalies: List of anomalies
        entity_relationships: List of entity relationships
        external_sources: List of external sources
        
    Returns:
        Evidence bundle
    """
    return EvidenceBundle(
        transactions=transactions or [],
        pattern_matches=pattern_matches or [],
        anomalies=anomalies or [],
        entity_relationships=entity_relationships or [],
        external_sources=external_sources or [],
    )


def validate_template(template_content: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a template for syntax errors.
    
    Args:
        template_content: Template content
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        env = Environment(autoescape=select_autoescape())
        env.parse(template_content)
        return True, None
    except Exception as e:
        return False, str(e)


def test_template(
    template_content: str,
    variables: Dict[str, Any],
) -> Tuple[bool, Union[str, Exception]]:
    """
    Test a template with variables.
    
    Args:
        template_content: Template content
        variables: Variables to substitute
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        env = Environment(autoescape=select_autoescape())
        template = env.from_string(template_content)
        result = template.render(**variables)
        return True, result
    except Exception as e:
        return False, e
