"""
Example configuration for a CrewAI agent using Google Gemini LLM.

This file demonstrates how to initialize a Gemini LLM and assign it to a
CrewAI Agent, leveraging CrewAI's native Gemini support (v0.5.0+) with Gemini 2.x models.
"""

import os
from crewai import Agent, LLM
from dotenv import load_dotenv

# Load environment variables from .env file
# This is crucial for picking up MODEL and GEMINI_API_KEY
load_dotenv()

# --- Option 1: Auto-configured LLM from .env ---
# CrewAI's LLM class can automatically pick up configuration
# from environment variables (MODEL, GEMINI_API_KEY).
# This is the simplest way if you have a single LLM for most agents.
auto_configured_gemini_llm = LLM()

# --- Option 2: Explicitly configured LLM ---
# You can also explicitly pass the model name and API key.
# This is useful if you need to use different Gemini models
# for different agents, or if you prefer explicit instantiation.
explicit_gemini_llm = LLM(
    model=os.getenv("MODEL", "gemini/gemini-2.5-pro-preview-05-06"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# --- Define a sample Agent using the Gemini LLM ---
# This agent demonstrates how to assign the configured Gemini LLM
# and enable multimodal capabilities.
example_gemini_agent = Agent(
    role='Multimodal Financial Document Analyst',
    goal='Analyze financial documents (text and images) to extract key data points and identify potential fraud indicators.',
    backstory=(
        "You are an expert financial analyst with a specialization in document review. "
        "You leverage advanced AI capabilities to process both textual and visual information "
        "from financial statements, invoices, and other related documents. "
        "Your insights are crucial for identifying discrepancies and red flags."
    ),
    llm=explicit_gemini_llm,  # Assign the Gemini LLM
    multimodal=True,          # Enable multimodal input for this agent
    verbose=True,             # Set to True to see agent's thought process
    allow_delegation=False,   # Agent does not delegate tasks in this example
    # tools=[your_custom_document_parsing_tool, your_custom_ocr_tool] # Add relevant tools here
)

# You can also define other agents using the auto_configured_gemini_llm
# another_agent = Agent(
#     role='Financial News Summarizer',
#     goal='Summarize daily financial news from various sources.',
#     backstory='An AI assistant specialized in concise news summarization.',
#     llm=auto_configured_gemini_llm,
#     verbose=True
# )

# Note: For this agent to actually process images, you would need
# to provide it with tools that can handle image inputs (e.g., a custom tool
# that takes an image path/bytes and passes it to the LLM for analysis).
# The `multimodal=True` flag primarily tells CrewAI that the LLM assigned
# to this agent supports multimodal inputs.
