"""
PolicyDocsTool for retrieving AML policy and regulatory information.

This tool provides CrewAI agents with access to policy documents and regulatory
information related to Anti-Money Laundering (AML), Know Your Customer (KYC),
and other financial compliance requirements. It serves as a placeholder that
can later be enhanced with a full RAG (Retrieval Augmented Generation) implementation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


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


class PolicyDocsTool(BaseTool):
    """
    Tool for retrieving AML policy and regulatory information.
    
    This tool allows agents to query policy documents and regulatory information
    related to Anti-Money Laundering (AML), Know Your Customer (KYC), and other
    financial compliance requirements. It serves as a placeholder that can later
    be enhanced with a full RAG implementation.
    """
    
    name: str = "policy_docs_tool"
    description: str = """
    Retrieve policy and regulatory information for financial compliance.
    
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
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize the PolicyDocsTool.
        
        Args:
            gemini_client: Optional GeminiClient instance. If not provided,
                          a new client will be created.
        """
        super().__init__()
        self.gemini_client = gemini_client or GeminiClient()
        
        # Placeholder for policy documents database
        # In a real implementation, this would be a vector database or similar
        self.policy_docs = self._initialize_policy_docs()
    
    def _initialize_policy_docs(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize placeholder policy documents.
        
        Returns:
            Dictionary of policy documents
        """
        return {
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
    
    async def _arun(
        self,
        query: str,
        document_type: Optional[str] = None,
        max_results: int = 5
    ) -> str:
        """
        Search policy documents asynchronously.
        
        Args:
            query: The query or topic to search for
            document_type: Optional type of document to search
            max_results: Maximum number of results to return
            
        Returns:
            JSON string containing matching policy information
        """
        try:
            # In a real implementation, this would use vector similarity search
            # For this placeholder, we'll use simple keyword matching
            
            results = []
            for doc_id, doc in self.policy_docs.items():
                # Filter by document type if specified
                if document_type and doc["type"] != document_type:
                    continue
                
                # Simple keyword matching (case insensitive)
                if (query.lower() in doc["title"].lower() or 
                    query.lower() in doc["content"].lower()):
                    results.append({
                        "id": doc_id,
                        "title": doc["title"],
                        "type": doc["type"],
                        "content": doc["content"],
                        "relevance": 0.85  # Placeholder relevance score
                    })
            
            # If no direct matches, use Gemini to generate a response
            if not results:
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
            
            # Limit results
            results = results[:max_results]
            
            return json.dumps({
                "success": True,
                "query": query,
                "document_type": document_type,
                "results": results,
                "count": len(results)
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
        max_results: int = 5
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
            self._arun(query, document_type, max_results)
        )
