"""Analysis API endpoints for data analysis and code execution."""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.integrations.sim_client import SimClient, SimApiError
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.jobs.sim_graph_job import (
    run_sim_graph_ingestion_job,
    batch_sim_graph_ingestion_job,
)
from backend.api.v1.whale_endpoints import router as whale_router


logger = logging.getLogger(__name__)
router = APIRouter()
# --------------------------------------------------------------------------- #
#                   Include Whale Movement Tracker Endpoints                  #
# --------------------------------------------------------------------------- #
router.include_router(whale_router)


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

# --------------------------------------------------------------------------- #
#                              Graph Ingestion IO                              #
# --------------------------------------------------------------------------- #


class GraphIngestRequest(BaseModel):
    """Request model for single-wallet graph ingestion."""

    wallet_address: str = Field(
        ...,
        description="Blockchain wallet address to ingest (EVM or Solana).",
        examples=["0xd8da6bf26964af9d7eed9e03e53415d37aa96045"],
    )
    ingest_balances: bool = Field(True, description="Ingest token balances.")
    ingest_activity: bool = Field(True, description="Ingest activity/transactions.")
    limit_balances: int = Field(100, description="Max balances to fetch.")
    limit_activity: int = Field(50, description="Max activity records.")
    chain_ids: str = Field("all", description="Comma-separated chain IDs or 'all'.")
    create_schema: bool = Field(
        False,
        description="Create Neo4j constraints/indexes if not present (first run only).",
    )


class BatchGraphIngestRequest(GraphIngestRequest):
    """Request model for batch wallet ingestion."""

    wallet_addresses: List[str] = Field(
        ...,
        description="List of wallet addresses to ingest.",
        min_items=1,
    )


class IngestResponse(BaseModel):
    """Generic response returned by ingestion jobs (single or batch)."""

    status: str = Field(..., description="'completed', 'failed', or 'error'")
    details: Dict[str, Any] = Field(
        ...,
        description="Detailed counts and/or error messages from the job.",
    )


# Dependency functions
async def get_gemini_client(request: Request) -> GeminiClient:
    return request.app.state.gemini


async def get_neo4j_client(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


async def get_e2b_client(request: Request) -> E2BClient:
    return request.app.state.e2b

# Added dependency for Sim API client
async def get_sim_client(request: Request) -> SimClient:
    return request.app.state.sim


@router.post("/execute-code", response_model=CodeExecutionResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
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

# --------------------------------------------------------------------------- #
#                          Sim API Proxy Endpoints                            #
# --------------------------------------------------------------------------- #


@router.get("/sim/balances/{wallet}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_wallet_balances(
    wallet: str,
    limit: int = 100,
    chain_ids: str = "all",
    metadata: str = "url,logo",
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **token balances** for a wallet address
    via Sim APIs. Returns the raw response from Sim unchanged so the frontend
    maintains full flexibility.
    """
    logger.info(f"Sim balances request: wallet={wallet} limit={limit} chain_ids={chain_ids}")
    try:
        return sim.get_balances(wallet, limit=limit, chain_ids=chain_ids, metadata=metadata)
    except SimApiError as e:
        logger.error(f"Sim balances error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim balances")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/activity/{wallet}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_wallet_activity(
    wallet: str,
    limit: int = 25,
    offset: Optional[str] = None,
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **chronological activity** for a wallet
    address via Sim APIs.
    """
    logger.info(f"Sim activity request: wallet={wallet} limit={limit} offset={offset}")
    try:
        return sim.get_activity(wallet, limit=limit, offset=offset)
    except SimApiError as e:
        logger.error(f"Sim activity error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim activity")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/collectibles/{wallet}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_wallet_collectibles(
    wallet: str,
    limit: int = Query(50, ge=1, le=100),
    offset: Optional[str] = None,
    chain_ids: Optional[str] = None,
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **NFT collectibles** for a wallet address
    via Sim APIs.
    """
    logger.info(f"Sim collectibles request: wallet={wallet} limit={limit} offset={offset} chain_ids={chain_ids}")
    try:
        return sim.get_collectibles(wallet, limit=limit, offset=offset, chain_ids=chain_ids)
    except SimApiError as e:
        logger.error(f"Sim collectibles error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim collectibles")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/token-info/{token_address}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_token_information(
    token_address: str,
    chain_ids: str = Query(..., description="Comma-separated list of chain IDs (e.g., '1,137'). Mandatory."),
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **detailed token metadata and pricing** for a token
    via Sim APIs.
    """
    logger.info(f"Sim token info request: token_address={token_address} chain_ids={chain_ids}")
    try:
        return sim.get_token_info(token_address, chain_ids=chain_ids)
    except SimApiError as e:
        logger.error(f"Sim token info error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim token info")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/token-holders/{chain_id}/{token_address}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_token_holders(
    chain_id: str,
    token_address: str,
    limit: int = Query(100, ge=1, le=100),
    offset: Optional[str] = None,
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **token holder distribution** for a given token
    on a specific chain via Sim APIs.
    """
    logger.info(f"Sim token holders request: chain_id={chain_id} token_address={token_address} limit={limit} offset={offset}")
    try:
        return sim.get_token_holders(chain_id, token_address, limit=limit, offset=offset)
    except SimApiError as e:
        logger.error(f"Sim token holders error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim token holders")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/svm/balances/{wallet}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_svm_wallet_balances(
    wallet: str,
    limit: int = Query(50, ge=1, le=100),
    offset: Optional[str] = None,
    chains: str = "all",
    sim: SimClient = Depends(get_sim_client),
):
    """
    Proxy endpoint that retrieves **Solana (SVM) token balances** for a wallet address
    via Sim APIs.
    """
    logger.info(f"Sim SVM balances request: wallet={wallet} limit={limit} offset={offset} chains={chains}")
    try:
        return sim.get_svm_balances(wallet, limit=limit, offset=offset, chains=chains)
    except SimApiError as e:
        logger.error(f"Sim SVM balances error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calling Sim SVM balances")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sim/risk-score/{wallet}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_sim_wallet_risk_score(
    wallet: str,
    sim: SimClient = Depends(get_sim_client),
):
    """
    Calculates a risk score for a wallet based on its balances and activity.
    Analyzes factors like suspicious token holdings, transaction patterns,
    and liquidity risks to generate a comprehensive risk assessment.
    """
    logger.info(f"Calculating risk score for wallet: {wallet}")
    try:
        # Step 1: Fetch balances and activity data
        balances_data = sim.get_balances(wallet, limit=100, metadata="url,logo")
        activity_data = sim.get_activity(wallet, limit=50)
        
        balances = balances_data.get("balances", [])
        activities = activity_data.get("activity", [])
        
        # Step 2: Calculate risk factors and overall score
        risk_factors = []
        risk_score = 0
        
        # Balance-based risk factors
        total_value_usd = 0
        low_liquidity_tokens = 0
        high_value_tokens = 0
        
        for balance in balances:
            value_usd = float(balance.get("value_usd", 0))
            total_value_usd += value_usd
            
            # Check for low liquidity tokens
            if balance.get("low_liquidity", False):
                low_liquidity_tokens += 1
                if value_usd > 1000:
                    risk_factors.append(f"High value in low liquidity token: {balance.get('symbol', 'Unknown')} (${value_usd:.2f})")
                    risk_score += 2
            
            # Check for high value tokens
            if value_usd > 50000:
                high_value_tokens += 1
        
        # Calculate percentage of low liquidity tokens
        if balances:
            low_liquidity_percentage = (low_liquidity_tokens / len(balances)) * 100
            if low_liquidity_percentage > 50:
                risk_factors.append(f"{low_liquidity_percentage:.1f}% of tokens have low liquidity")
                risk_score += 3
        
        # Activity-based risk factors
        if activities:
            # Count transaction types
            tx_types = {}
            approvals = 0
            large_outflows = 0
            small_rapid_txs = 0
            recent_activity_count = 0
            
            # Track timestamps for velocity analysis
            timestamps = []
            
            for activity in activities:
                tx_type = activity.get("type", "unknown")
                tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
                
                # Check for approvals (potential security risk)
                if tx_type == "approve":
                    approvals += 1
                
                # Check for large outflows
                if tx_type == "send" and float(activity.get("value_usd", 0)) > 10000:
                    large_outflows += 1
                    
                # Track timestamp for velocity analysis
                if "block_time" in activity:
                    timestamps.append(activity["block_time"])
                    
                # Count recent activity (last 30 days)
                if "block_time" in activity:
                    import time
                    current_time = time.time()
                    block_time = activity["block_time"]
                    if (current_time - block_time) < (30 * 24 * 60 * 60):  # 30 days in seconds
                        recent_activity_count += 1
            
            # Assess approval risk
            if approvals > 5:
                risk_factors.append(f"High number of token approvals: {approvals}")
                risk_score += 2
            
            # Assess large outflow risk
            if large_outflows > 0:
                risk_factors.append(f"Large value outflows detected: {large_outflows}")
                risk_score += large_outflows
            
            # Assess transaction velocity
            if len(timestamps) >= 2:
                timestamps.sort()
                time_span = timestamps[-1] - timestamps[0]
                if time_span > 0:
                    tx_per_day = (len(timestamps) / time_span) * 86400  # Convert to per day
                    if tx_per_day > 20:
                        risk_factors.append(f"High transaction velocity: {tx_per_day:.1f} tx/day")
                        risk_score += 2
        
        # Calculate overall risk score (0-100 scale)
        normalized_risk_score = min(100, risk_score * 5)
        
        # Determine risk level
        risk_level = "LOW"
        if normalized_risk_score >= 75:
            risk_level = "HIGH"
        elif normalized_risk_score >= 40:
            risk_level = "MEDIUM"
        
        # Prepare response
        response = {
            "wallet_address": wallet,
            "risk_score": normalized_risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected."],
            "summary": {
                "total_value_usd": total_value_usd,
                "token_count": len(balances),
                "low_liquidity_tokens": low_liquidity_tokens,
                "high_value_tokens": high_value_tokens,
                "transaction_count": len(activities),
                "transaction_types": tx_types
            }
        }
        
        return response
        
    except SimApiError as e:
        logger.error(f"Risk score calculation error ({e.status_code}): {e}")
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error calculating risk score")
        raise HTTPException(status_code=500, detail="Internal server error")


# --------------------------------------------------------------------------- #
#                         Graph Ingestion Trigger APIs                        #
# --------------------------------------------------------------------------- #


@router.post("/graph/ingest-wallet", response_model=IngestResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def ingest_single_wallet(
    request: GraphIngestRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    sim: SimClient = Depends(get_sim_client),
):
    """
    Trigger a **single-wallet** graph-ingestion job.
    Loads balances and/or activity for the supplied wallet address from Sim
    and persists them as nodes / relationships in Neo4j.
    """
    logger.info(
        "Graph ingestion (single) requested for wallet %s [balances=%s activity=%s]",
        request.wallet_address,
        request.ingest_balances,
        request.ingest_activity,
    )
    try:
        result = await run_sim_graph_ingestion_job(
            wallet_address=request.wallet_address,
            neo4j_client=neo4j,
            sim_client=sim,
            ingest_balances=request.ingest_balances,
            ingest_activity=request.ingest_activity,
            limit_balances=request.limit_balances,
            limit_activity=request.limit_activity,
            chain_ids=request.chain_ids,
            create_schema=request.create_schema,
        )
        return IngestResponse(status="completed", details=result)
    except Exception as e:
        logger.exception("Graph ingestion (single) failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/batch-ingest", response_model=IngestResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def ingest_batch_wallets(
    request: BatchGraphIngestRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    sim: SimClient = Depends(get_sim_client),
):
    """
    Trigger a **batch** graph-ingestion job for multiple wallet addresses.
    Ideal for nightly syncs or bulk investigations.
    """
    logger.info(
        "Graph ingestion (batch) requested for %d wallets", len(request.wallet_addresses)
    )
    try:
        result = await batch_sim_graph_ingestion_job(
            wallet_addresses=request.wallet_addresses,
            neo4j_client=neo4j,
            sim_client=sim,
            ingest_balances=request.ingest_balances,
            ingest_activity=request.ingest_activity,
            limit_balances=request.limit_balances,
            limit_activity=request.limit_activity,
            chain_ids=request.chain_ids,
            create_schema=request.create_schema,
        )
        return IngestResponse(status="completed", details=result)
    except Exception as e:
        logger.exception("Graph ingestion (batch) failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/analyze", response_model=AnalysisResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
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
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
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
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
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
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
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
