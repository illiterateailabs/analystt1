"""
Example Gemini 2.5 Agent Configuration

This file demonstrates how to configure agents with different Gemini 2.5 models,
showing various configurations for different use cases and capabilities.

Usage:
    from backend.agents.configs.example_gemini_2_5_agent import create_gemini_agent
    
    # Create a flash agent for quick responses
    flash_agent = create_gemini_flash_agent(...)
    
    # Create a pro agent for complex reasoning
    pro_agent = create_gemini_pro_agent(...)
    
    # Create a multimodal agent for image analysis
    multimodal_agent = create_gemini_multimodal_agent(...)
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64

from crewai import Agent, Task, Crew
from crewai.agent import AgentConfig
from crewai.llm import LLM

from backend.agents.tools import GraphQueryTool, SandboxExecTool, PatternLibraryTool


# Model IDs for Gemini 2.5 models
GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-05-20"
GEMINI_2_5_PRO = "gemini-2.5-pro-preview-05-06"
GEMINI_2_5_FLASH_AUDIO = "gemini-2.5-flash-preview-native-audio-dialog"
GEMINI_2_0_FLASH = "gemini-2.0-flash"  # Fallback option


def create_gemini_flash_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
    allow_delegation: bool = False,
    max_iter: int = 10,
    max_rpm: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Agent:
    """
    Create an agent using Gemini 2.5 Flash model - optimized for speed and cost efficiency.
    
    This agent is best for:
    - Quick responses where latency matters
    - High-volume, simple tasks
    - Initial data processing or triage
    - Real-time interactions with users
    
    Cost considerations:
    - ~50-70% cheaper than Pro models
    - Lower token consumption for similar tasks
    - Better for high-volume operations
    
    Performance notes:
    - 2-3x faster response times than Pro
    - Good for straightforward reasoning
    - Less robust for complex multi-step reasoning
    - Works well with well-structured prompts
    """
    # Configure the LLM with Gemini 2.5 Flash
    llm = LLM(
        model=GEMINI_2_5_FLASH,  # Use the Flash model for speed
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.2,  # Lower temperature for more deterministic outputs
        max_tokens=2048,  # Limit token usage for cost efficiency
    )
    
    # Create and return the agent
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=allow_delegation,
        max_iter=max_iter,
        max_rpm=max_rpm,
        system_prompt=system_prompt,
    )


def create_gemini_pro_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
    allow_delegation: bool = False,
    max_iter: int = 15,  # Higher default for complex reasoning
    max_rpm: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Agent:
    """
    Create an agent using Gemini 2.5 Pro model - optimized for advanced reasoning and quality.
    
    This agent is best for:
    - Complex analytical tasks requiring deep reasoning
    - Tasks involving code generation or analysis
    - Critical decision-making processes
    - Detailed report generation
    
    Cost considerations:
    - Higher cost per token than Flash models
    - May use more tokens for equivalent tasks
    - Consider using for high-value, complex tasks only
    
    Performance notes:
    - Superior multi-step reasoning capabilities
    - Better handling of ambiguous instructions
    - More robust to prompt variations
    - Stronger code generation and analysis
    """
    # Configure the LLM with Gemini 2.5 Pro
    llm = LLM(
        model=GEMINI_2_5_PRO,  # Use the Pro model for complex reasoning
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.4,  # Slightly higher temperature for creative problem-solving
        max_tokens=4096,  # Higher token limit for complex tasks
    )
    
    # Create and return the agent
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=allow_delegation,
        max_iter=max_iter,
        max_rpm=max_rpm,
        system_prompt=system_prompt,
    )


def create_gemini_multimodal_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
    allow_delegation: bool = False,
    max_iter: int = 15,
    max_rpm: Optional[int] = None,
    system_prompt: Optional[str] = None,
    use_pro: bool = True,  # Whether to use Pro (True) or Flash (False)
) -> Agent:
    """
    Create a multimodal agent using Gemini 2.5 models with image analysis capabilities.
    
    This agent is best for:
    - Tasks involving image analysis or understanding
    - Document processing with visual elements
    - Chart/graph interpretation
    - Visual evidence analysis in investigations
    
    Cost considerations:
    - Image processing increases token usage significantly
    - Pro model costs more but provides better analysis
    - Consider image resolution/size for cost optimization
    
    Performance notes:
    - Both models handle images well, but Pro has better understanding
    - Flash is faster for simple image classification
    - Pro is better for detailed image analysis and reasoning
    - Both support multiple images in a single conversation
    """
    # Select model based on reasoning needs
    model = GEMINI_2_5_PRO if use_pro else GEMINI_2_5_FLASH
    
    # Configure the LLM with multimodal capabilities
    llm = LLM(
        model=model,
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.3,
        max_tokens=4096 if use_pro else 2048,
    )
    
    # Create and return the agent with multimodal config enabled
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=allow_delegation,
        max_iter=max_iter,
        max_rpm=max_rpm,
        system_prompt=system_prompt,
        config=AgentConfig(
            multimodal=True  # Enable multimodal capabilities
        )
    )


def test_gemini_models():
    """
    Test function to compare Gemini 2.5 Flash and Pro models on a simple task.
    
    This function creates agents with both models and runs a simple analytical task
    to compare their responses, performance, and token usage.
    
    Returns:
        Dict with test results including responses and timing information
    """
    import time
    
    # Simple analytical question to test reasoning capabilities
    test_question = """
    Analyze the following transaction pattern and identify potential fraud indicators:
    
    - Account A received $50,000 from an overseas source
    - Within 24 hours, Account A sent $9,800 to Account B
    - Account A then sent $9,700 to Account C
    - Account A sent $9,600 to Account D
    - Account A sent $9,500 to Account E
    - Account A withdrew the remaining balance in cash
    
    What type of financial crime pattern might this represent? What regulations might apply?
    """
    
    # Create Flash agent
    flash_agent = create_gemini_flash_agent(
        role="Financial Crime Analyst",
        goal="Identify potential fraud patterns in transaction data",
        backstory="You are an expert in detecting financial crime patterns and regulatory violations.",
    )
    
    # Create Pro agent
    pro_agent = create_gemini_pro_agent(
        role="Financial Crime Analyst",
        goal="Identify potential fraud patterns in transaction data",
        backstory="You are an expert in detecting financial crime patterns and regulatory violations.",
    )
    
    # Test Flash model
    flash_start = time.time()
    flash_response = flash_agent.llm.generate(test_question)
    flash_time = time.time() - flash_start
    
    # Test Pro model
    pro_start = time.time()
    pro_response = pro_agent.llm.generate(test_question)
    pro_time = time.time() - pro_start
    
    # Return comparison results
    return {
        "flash": {
            "model": GEMINI_2_5_FLASH,
            "response": flash_response,
            "time_seconds": flash_time,
        },
        "pro": {
            "model": GEMINI_2_5_PRO,
            "response": pro_response,
            "time_seconds": pro_time,
        },
        "comparison": {
            "speed_difference": f"{pro_time / flash_time:.2f}x (Pro vs Flash)",
            "response_length_difference": f"{len(pro_response) / len(flash_response):.2f}x (Pro vs Flash)",
        }
    }


def test_multimodal_capabilities(image_path: str):
    """
    Test function to demonstrate multimodal capabilities with image analysis.
    
    Args:
        image_path: Path to an image file to analyze
        
    Returns:
        Dict with analysis results from both Flash and Pro models
    """
    import time
    
    # Ensure image exists
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    # Read image and encode as base64
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Image analysis prompt
    image_prompt = """
    Analyze this image in the context of a financial investigation:
    1. Describe what you see in the image
    2. Identify any potential evidence relevant to financial crimes
    3. Suggest next steps for an investigator based on this image
    """
    
    # Create multimodal agents
    flash_agent = create_gemini_multimodal_agent(
        role="Visual Evidence Analyst",
        goal="Analyze visual evidence in financial crime investigations",
        backstory="You are specialized in analyzing visual evidence for financial crime investigations.",
        use_pro=False,  # Use Flash model
    )
    
    pro_agent = create_gemini_multimodal_agent(
        role="Visual Evidence Analyst",
        goal="Analyze visual evidence in financial crime investigations",
        backstory="You are specialized in analyzing visual evidence for financial crime investigations.",
        use_pro=True,  # Use Pro model
    )
    
    # Test Flash model with image
    flash_start = time.time()
    flash_response = flash_agent.llm.generate(
        [{"type": "text", "text": image_prompt}, 
         {"type": "image", "data": img_data}]
    )
    flash_time = time.time() - flash_start
    
    # Test Pro model with image
    pro_start = time.time()
    pro_response = pro_agent.llm.generate(
        [{"type": "text", "text": image_prompt}, 
         {"type": "image", "data": img_data}]
    )
    pro_time = time.time() - pro_start
    
    # Return comparison results
    return {
        "flash": {
            "model": GEMINI_2_5_FLASH,
            "response": flash_response,
            "time_seconds": flash_time,
        },
        "pro": {
            "model": GEMINI_2_5_PRO,
            "response": pro_response,
            "time_seconds": pro_time,
        },
        "comparison": {
            "speed_difference": f"{pro_time / flash_time:.2f}x (Pro vs Flash)",
            "detail_level": "Pro typically provides more detailed visual analysis",
        }
    }


# Example usage in a fraud investigation crew
def create_fraud_investigation_crew_with_gemini_2_5():
    """
    Example of creating a fraud investigation crew using Gemini 2.5 models.
    
    This demonstrates how to select appropriate models for different agent roles
    based on their specific needs and responsibilities.
    
    Returns:
        Crew object configured with Gemini 2.5 agents
    """
    from backend.integrations.neo4j_client import Neo4jClient
    
    # Initialize tools
    neo4j_client = Neo4jClient()
    graph_query_tool = GraphQueryTool(neo4j_client=neo4j_client)
    pattern_library_tool = PatternLibraryTool()
    
    # Create agents with appropriate models for their roles
    
    # NLQ Translator - uses Flash for quick query translation
    nlq_translator = create_gemini_flash_agent(
        role="Natural Language Query Translator",
        goal="Convert natural language questions into precise Neo4j Cypher queries",
        backstory="You are an expert in translating human questions into graph database queries.",
        tools=[graph_query_tool],
        system_prompt="You translate natural language questions about financial transactions into Cypher queries for Neo4j.",
    )
    
    # Graph Analyst - uses Flash for efficient query execution and basic analysis
    graph_analyst = create_gemini_flash_agent(
        role="Graph Data Analyst",
        goal="Execute Cypher queries and analyze graph data",
        backstory="You are specialized in analyzing graph data structures and identifying patterns.",
        tools=[graph_query_tool],
        system_prompt="You analyze graph query results and identify key entities and relationships.",
    )
    
    # Fraud Pattern Hunter - uses Pro for complex reasoning about fraud patterns
    fraud_pattern_hunter = create_gemini_pro_agent(
        role="Fraud Pattern Detection Specialist",
        goal="Identify known and novel fraud patterns in transaction data",
        backstory="You are an expert in detecting sophisticated financial crime patterns.",
        tools=[graph_query_tool, pattern_library_tool],
        system_prompt="You identify both known fraud patterns from the pattern library and detect novel suspicious behaviors.",
    )
    
    # Image Analyst - uses Pro with multimodal for document/image analysis
    image_analyst = create_gemini_multimodal_agent(
        role="Visual Evidence Analyst",
        goal="Analyze visual evidence in financial investigations",
        backstory="You specialize in extracting information from financial documents and images.",
        use_pro=True,  # Use Pro for detailed image analysis
        system_prompt="You analyze images and documents for evidence relevant to financial investigations.",
    )
    
    # Report Writer - uses Pro for high-quality report generation
    report_writer = create_gemini_pro_agent(
        role="Financial Crime Report Writer",
        goal="Create comprehensive investigation reports",
        backstory="You are skilled at synthesizing complex financial crime evidence into clear reports.",
        system_prompt="You create detailed, well-structured reports summarizing financial crime investigations.",
    )
    
    # Create crew with the agents
    crew = Crew(
        agents=[nlq_translator, graph_analyst, fraud_pattern_hunter, image_analyst, report_writer],
        tasks=[
            Task(
                description="Convert the user's question into a Cypher query",
                agent=nlq_translator,
            ),
            Task(
                description="Execute the query and analyze the graph data",
                agent=graph_analyst,
            ),
            Task(
                description="Identify fraud patterns in the transaction data",
                agent=fraud_pattern_hunter,
            ),
            Task(
                description="Analyze any related documents or images",
                agent=image_analyst,
            ),
            Task(
                description="Generate a comprehensive investigation report",
                agent=report_writer,
            ),
        ],
        verbose=True,
    )
    
    return crew


if __name__ == "__main__":
    # Example usage
    print("Testing Gemini 2.5 models...")
    results = test_gemini_models()
    
    print(f"\nGemini 2.5 Flash response time: {results['flash']['time_seconds']:.2f} seconds")
    print(f"Gemini 2.5 Pro response time: {results['pro']['time_seconds']:.2f} seconds")
    print(f"Speed difference: {results['comparison']['speed_difference']}")
    
    print("\nFlash response excerpt:")
    print(results['flash']['response'][:200] + "...")
    
    print("\nPro response excerpt:")
    print(results['pro']['response'][:200] + "...")
    
    # Uncomment to test multimodal capabilities
    # image_results = test_multimodal_capabilities("path/to/test/image.jpg")
    # print("\nMultimodal analysis results:")
    # print(f"Flash image analysis time: {image_results['flash']['time_seconds']:.2f} seconds")
    # print(f"Pro image analysis time: {image_results['pro']['time_seconds']:.2f} seconds")
