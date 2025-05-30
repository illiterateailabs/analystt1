"""Analysis API endpoints for data analysis and code execution."""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class AnalysisRequest(BaseModel):
    task_description: str = Field(..., description="Description of the analysis task")
    data_source: Optional[str] = Field(None, description="Data source (graph, file, etc.)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")
    output_format: str = Field(default="json", description="Output format (json, csv, plot)")


class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Python code to execute")
    libraries: Optional[List[str]] = Field(None, description="Required libraries")
    timeout: Optional[int] = Field(300, description="Execution timeout in seconds")
    sandbox_id: Optional[str] = Field(None, description="Existing sandbox ID")


class AnalysisResponse(BaseModel):
    task_id: str = Field(..., description="Analysis task ID")
    status: str = Field(..., description="Analysis status")
    results: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    code_generated: Optional[str] = Field(None, description="Generated code")
    execution_details: Optional[Dict[str, Any]] = Field(None, description="Execution details")
    visualizations: Optional[List[str]] = Field(None, description="Generated visualization paths")


class CodeExecutionResponse(BaseModel):
    sandbox_id: str = Field(..., description="Sandbox ID")
    stdout: str = Field(..., description="Standard output")
    stderr: str = Field(..., description="Standard error")
    exit_code: int = Field(..., description="Exit code")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Execution success status")


# Dependency functions
async def get_gemini_client(request: Request) -> GeminiClient:
    return request.app.state.gemini


async def get_neo4j_client(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


async def get_e2b_client(request: Request) -> E2BClient:
    return request.app.state.e2b


@router.post("/execute-code", response_model=CodeExecutionResponse)
async def execute_code(
    request: CodeExecutionRequest,
    e2b: E2BClient = Depends(get_e2b_client)
):
    """Execute Python code in a secure sandbox."""
    try:
        logger.info("Executing Python code in sandbox")
        
        # Install required libraries if specified
        if request.libraries and request.sandbox_id:
            await e2b.install_packages(request.libraries, request.sandbox_id)
        
        # Execute the code
        result = await e2b.execute_python_code(
            request.code,
            sandbox_id=request.sandbox_id,
            timeout=request.timeout
        )
        
        logger.info(f"Code execution completed with exit code: {result['exit_code']}")
        return CodeExecutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
async def perform_analysis(
    request: AnalysisRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    e2b: E2BClient = Depends(get_e2b_client)
):
    """Perform data analysis using AI-generated code."""
    try:
        logger.info(f"Starting analysis: {request.task_description[:100]}...")
        
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Prepare context for code generation
        context = f"Task: {request.task_description}\n"
        
        if request.data_source == "graph" or "graph" in request.task_description.lower():
            # Get graph schema for context
            schema_info = await neo4j.get_schema_info()
            context += f"""
Graph Database Context:
- Available node labels: {', '.join(schema_info['labels'])}
- Available relationship types: {', '.join(schema_info['relationship_types'])}
- Total nodes: {schema_info['node_count']}
- Total relationships: {schema_info['relationship_count']}

Use the neo4j library to connect and query the database.
Connection details will be provided in the execution environment.
"""
        
        if request.parameters:
            context += f"\nParameters: {request.parameters}"
        
        # Generate Python code for the analysis
        libraries = [
            "pandas", "numpy", "matplotlib", "seaborn", "plotly", 
            "neo4j", "networkx", "scikit-learn"
        ]
        
        generated_code = await gemini.generate_python_code(
            request.task_description,
            context=context,
            libraries=libraries
        )
        
        # Create sandbox and execute code
        sandbox_id = await e2b.create_sandbox()
        
        try:
            # Install required libraries
            await e2b.install_packages(libraries, sandbox_id)
            
            # Prepare code with Neo4j connection if needed
            if request.data_source == "graph" or "neo4j" in generated_code.lower():
                # Add Neo4j connection setup
                neo4j_setup = f"""
# Neo4j connection setup
import os
from neo4j import GraphDatabase

NEO4J_URI = "bolt://host.docker.internal:7687"  # Adjust for sandbox environment
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "analyst123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

"""
                full_code = neo4j_setup + generated_code
            else:
                full_code = generated_code
            
            # Execute the analysis code
            execution_result = await e2b.execute_python_code(full_code, sandbox_id)
            
            # Parse results from stdout
            results = {}
            if execution_result["success"]:
                try:
                    # Try to parse JSON output
                    import json
                    results = json.loads(execution_result["stdout"])
                except:
                    # If not JSON, return as text
                    results = {"output": execution_result["stdout"]}
            
            # Check for generated visualizations
            visualizations = []
            files = await e2b.list_files(sandbox_id)
            for file in files:
                if any(ext in file for ext in ['.png', '.jpg', '.svg', '.html']):
                    visualizations.append(file)
            
            response_data = {
                "task_id": task_id,
                "status": "completed" if execution_result["success"] else "failed",
                "results": results,
                "code_generated": generated_code,
                "execution_details": execution_result,
                "visualizations": visualizations
            }
            
        finally:
            # Cleanup sandbox
            await e2b.close_sandbox(sandbox_id)
        
        logger.info(f"Analysis completed: {task_id}")
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error performing analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud-detection/patterns")
async def detect_fraud_patterns(
    pattern_type: str = "money_laundering",
    limit: int = 100,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    gemini: GeminiClient = Depends(get_gemini_client)
):
    """Detect fraud patterns in the graph database."""
    try:
        logger.info(f"Detecting fraud patterns: {pattern_type}")
        
        # Define pattern-specific queries
        pattern_queries = {
            "money_laundering": """
                MATCH (a:Person)-[:PERFORMED_TRANSACTION]->(t1:Transaction)-[:TO]->(b:Person),
                      (b)-[:PERFORMED_TRANSACTION]->(t2:Transaction)-[:TO]->(c:Person),
                      (c)-[:PERFORMED_TRANSACTION]->(t3:Transaction)-[:TO]->(a)
                WHERE t1.amount > 10000 AND t2.amount > 10000 AND t3.amount > 10000
                AND t1.date < t2.date < t3.date
                RETURN a, b, c, t1, t2, t3
                LIMIT $limit
            """,
            "circular_transactions": """
                MATCH path = (a:Person)-[:PERFORMED_TRANSACTION*3..5]->(a)
                WHERE ALL(r IN relationships(path) WHERE r.amount > 5000)
                RETURN path, length(path) as path_length
                LIMIT $limit
            """,
            "suspicious_velocity": """
                MATCH (p:Person)-[:PERFORMED_TRANSACTION]->(t:Transaction)
                WITH p, count(t) as transaction_count, 
                     sum(t.amount) as total_amount,
                     max(t.date) as last_transaction,
                     min(t.date) as first_transaction
                WHERE transaction_count > 50 
                AND total_amount > 1000000
                RETURN p, transaction_count, total_amount, 
                       duration.between(first_transaction, last_transaction) as time_span
                LIMIT $limit
            """
        }
        
        if pattern_type not in pattern_queries:
            raise HTTPException(status_code=400, detail=f"Unknown pattern type: {pattern_type}")
        
        # Execute the pattern detection query
        query = pattern_queries[pattern_type]
        results = await neo4j.execute_query(query, {"limit": limit})
        
        # Generate AI explanation of the patterns found
        explanation = await gemini.explain_results(
            f"Fraud pattern detection for {pattern_type}",
            results,
            context="This is a fraud detection analysis. Focus on suspicious patterns and risk indicators."
        )
        
        return {
            "pattern_type": pattern_type,
            "patterns_found": len(results),
            "results": results,
            "explanation": explanation,
            "query_used": query
        }
        
    except Exception as e:
        logger.error(f"Error detecting fraud patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sandbox/{sandbox_id}/files")
async def list_sandbox_files(
    sandbox_id: str,
    directory: str = ".",
    e2b: E2BClient = Depends(get_e2b_client)
):
    """List files in a sandbox directory."""
    try:
        files = await e2b.list_files(sandbox_id, directory)
        return {"sandbox_id": sandbox_id, "directory": directory, "files": files}
        
    except Exception as e:
        logger.error(f"Error listing sandbox files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sandbox/{sandbox_id}/download/{file_path:path}")
async def download_sandbox_file(
    sandbox_id: str,
    file_path: str,
    e2b: E2BClient = Depends(get_e2b_client)
):
    """Download a file from a sandbox."""
    try:
        content = await e2b.download_file(sandbox_id, file_path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"file_path": file_path, "content": content}
        
    except Exception as e:
        logger.error(f"Error downloading sandbox file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
