"""
SimTokenHoldersTool - Fetch token holder distribution data from Sim APIs.

This tool retrieves a ranked list of token holders for a given ERC20 or ERC721 token
on a specific EVM chain using the Sim APIs. It provides insights into token distribution,
identifies whale concentrations, and assesses associated risks.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from collections import Counter

from crewai_tools import BaseTool
from pydantic import BaseModel, Field, validator

from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error

logger = logging.getLogger(__name__)


class SimTokenHoldersInput(BaseModel):
    """Input schema for SimTokenHoldersTool."""
    
    chain_id: str = Field(
        ...,
        description="The ID of the blockchain chain (e.g., '1' for Ethereum, '137' for Polygon)."
    )
    token_address: str = Field(
        ...,
        description="The contract address of the token (ERC20 or ERC721) to fetch holders for."
                    "For native tokens (e.g., ETH, MATIC), use '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'."
    )
    limit: int = Field(
        50,
        description="Maximum number of token holders to return (default: 50, max: 100)."
    )
    offset: Optional[str] = Field(
        None,
        description="Pagination cursor for fetching the next page of results."
    )


class SimTokenHoldersTool(BaseTool):
    """
    Tool for retrieving detailed token holder distribution for ERC20 and ERC721 tokens.
    
    This tool fetches a ranked list of token holders, their balances, and provides
    analysis on distribution patterns, including whale concentration and Gini coefficient.
    It's crucial for assessing decentralization, identifying potential market manipulation,
    and understanding ownership risks in financial crime analysis.
    """
    
    name: str = "sim_token_holders_tool"
    description: str = (
        "Fetch token holder distribution data for a given token on a specific EVM chain. "
        "Provides ranked list of holders, their balances, and analysis on distribution "
        "patterns, including whale concentration and Gini coefficient. "
        "Requires 'chain_id' and 'token_address'. "
        "For native tokens, use '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' as the token address."
    )
    
    sim_client: SimClient
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the SimTokenHoldersTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        chain_id: str,
        token_address: str,
        limit: int = 50,
        offset: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch token holder data for the specified token and chain.
        
        Args:
            chain_id: The ID of the blockchain chain.
            token_address: The contract address of the token.
            limit: Maximum number of token holders to return.
            offset: Pagination cursor for fetching next page of results.
            
        Returns:
            Dictionary containing token holder data, distribution analysis, and risk assessment.
            
        Raises:
            Exception: If there's an error fetching the token holder information.
        """
        try:
            record_tool_usage(self.name)
            
            params = {
                "limit": min(limit, 100)  # Cap at 100 to prevent abuse
            }
            if offset:
                params["offset"] = offset
            
            endpoint = f"/v1/evm/token-holders/{chain_id}/{token_address}"
            logger.info(f"Fetching token holders for {token_address} on chain {chain_id} with params {params}")
            response = await self.sim_client.get(endpoint, **params)
            
            holders_data = response.get("entries", [])
            next_offset = response.get("next_offset")
            
            total_supply = response.get("total_supply")
            total_holders = response.get("total_holders")
            
            # Analyze distribution
            distribution_analysis = self._analyze_distribution(holders_data, total_supply)
            risk_assessment = self._assess_concentration_risk(distribution_analysis)
            
            result = {
                "chain_id": chain_id,
                "token_address": token_address,
                "holders": holders_data,
                "count": len(holders_data),
                "total_supply": total_supply,
                "total_holders": total_holders,
                "next_offset": next_offset,
                "has_more": next_offset is not None,
                "distribution_analysis": distribution_analysis,
                "risk_assessment": risk_assessment,
                "request_time": response.get("request_time"),
                "response_time": response.get("response_time")
            }
            
            if not holders_data:
                logger.info(f"No token holders found for {token_address} on chain {chain_id}")
                result["message"] = "No token holders found for the given token and chain."
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching token holders for {token_address} on chain {chain_id}: {str(e)}"
            logger.error(error_msg)
            record_tool_error(self.name, str(e))
            
            return {
                "error": error_msg,
                "chain_id": chain_id,
                "token_address": token_address,
                "holders": [],
                "count": 0
            }

    def _analyze_distribution(self, holders: List[Dict[str, Any]], total_supply: Optional[str]) -> Dict[str, Any]:
        """
        Analyze token distribution patterns.
        """
        if not holders:
            return {
                "gini_coefficient": 0.0,
                "top_1_percent_concentration": 0.0,
                "top_10_percent_concentration": 0.0,
                "whale_addresses": [],
                "summary": "No holders to analyze."
            }

        # Convert balances to float for calculations
        balances = []
        for holder in holders:
            try:
                # Sim API returns balance as string, convert to float
                balances.append(float(holder.get("balance", 0)))
            except ValueError:
                logger.warning(f"Could not convert balance to float: {holder.get('balance')}")
                balances.append(0.0)
        
        balances.sort() # Required for Gini calculation

        gini_coefficient = self._calculate_gini_coefficient(balances)

        total_balance_in_sample = sum(balances)
        
        # Calculate concentration for top 1% and 10% of holders in the sample
        num_holders_in_sample = len(holders)
        top_1_percent_count = max(1, int(num_holders_in_sample * 0.01))
        top_10_percent_count = max(1, int(num_holders_in_sample * 0.10))

        top_1_percent_balance = sum(balances[num_holders_in_sample - top_1_percent_count:])
        top_10_percent_balance = sum(balances[num_holders_in_sample - top_10_percent_count:])

        top_1_percent_concentration = (top_1_percent_balance / total_balance_in_sample) * 100 if total_balance_in_sample > 0 else 0.0
        top_10_percent_concentration = (top_10_percent_balance / total_balance_in_sample) * 100 if total_balance_in_sample > 0 else 0.0

        # Identify whale addresses (e.g., top 5 largest holders in the sample)
        whale_addresses = []
        sorted_holders = sorted(holders, key=lambda x: float(x.get("balance", 0)), reverse=True)
        for i, holder in enumerate(sorted_holders[:5]): # Top 5 whales
            whale_addresses.append({
                "address": holder.get("address"),
                "balance": holder.get("balance"),
                "rank": i + 1
            })

        return {
            "gini_coefficient": round(gini_coefficient, 4),
            "top_1_percent_concentration": round(top_1_percent_concentration, 2),
            "top_10_percent_concentration": round(top_10_percent_concentration, 2),
            "whale_addresses": whale_addresses,
            "summary": (
                f"Gini Coefficient: {gini_coefficient:.4f}. "
                f"Top 1% of holders control {top_1_percent_concentration:.2f}% of the sample supply. "
                f"Top 10% of holders control {top_10_percent_concentration:.2f}% of the sample supply."
            )
        }

    def _calculate_gini_coefficient(self, x: List[float]) -> float:
        """
        Calculate the Gini coefficient of a list of values.
        A measure of statistical dispersion intended to represent the income or wealth
        distribution of a nation's residents, and is the most commonly used measurement
        of inequality.
        A Gini coefficient of 0 expresses perfect equality, while a coefficient of 1
        (or 100%) expresses maximal inequality among values.
        """
        if not x:
            return 0.0
        
        # Sort values and convert to numpy array for efficient calculation
        x = sorted([val for val in x if val >= 0]) # Ensure non-negative values
        n = len(x)
        if n == 0:
            return 0.0
        
        # Gini coefficient formula: sum((2i - n - 1) * xi) / (n * sum(xi))
        # where xi are sorted values
        numerator = sum((i + 1) * val for i, val in enumerate(x))
        denominator = n * sum(x)
        
        if denominator == 0: # Avoid division by zero if all balances are zero
            return 0.0
            
        return (2 * numerator / denominator) - (n + 1) / n

    def _assess_concentration_risk(self, distribution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the risk associated with token concentration.
        """
        risk_factors = []
        risk_score = 0

        gini = distribution_analysis.get("gini_coefficient", 0.0)
        top_1_concentration = distribution_analysis.get("top_1_percent_concentration", 0.0)
        top_10_concentration = distribution_analysis.get("top_10_percent_concentration", 0.0)
        whale_addresses = distribution_analysis.get("whale_addresses", [])

        if gini > 0.7:
            risk_factors.append("Very high Gini coefficient, indicating extreme inequality.")
            risk_score += 5
        elif gini > 0.5:
            risk_factors.append("High Gini coefficient, indicating significant inequality.")
            risk_score += 3
        
        if top_1_concentration > 50:
            risk_factors.append(f"Top 1% of holders control {top_1_concentration:.2f}% of supply (very high concentration).")
            risk_score += 4
        elif top_1_concentration > 20:
            risk_factors.append(f"Top 1% of holders control {top_1_concentration:.2f}% of supply (high concentration).")
            risk_score += 2

        if top_10_concentration > 80:
            risk_factors.append(f"Top 10% of holders control {top_10_concentration:.2f}% of supply (extreme concentration).")
            risk_score += 3
        elif top_10_concentration > 50:
            risk_factors.append(f"Top 10% of holders control {top_10_concentration:.2f}% of supply (high concentration).")
            risk_score += 1

        if len(whale_addresses) > 0:
            risk_factors.append(f"Identified {len(whale_addresses)} whale addresses in the top ranks.")
            # Score already adjusted by concentration, this is just a flag

        return {
            "score": min(risk_score, 10), # Cap score at 10
            "factors": risk_factors if risk_factors else ["No significant concentration risks detected."],
            "is_risky": risk_score > 0
        }
