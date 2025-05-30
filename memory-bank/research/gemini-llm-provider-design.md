# GeminiLLMProvider – Design Document  
*Location: `memory-bank/research/gemini-llm-provider-design.md`*  
_Last updated: **30 May 2025**_

---

## 1 · Purpose

Implement a **GeminiLLMProvider** that plugs Google Gemini models into CrewAI as a first-class `BaseLLM` backend, unlocking:

* **Multimodal reasoning** (text + images)  
* **Structured function/tool calls** (JSON)  
* **Model Context Protocol (MCP)** native tool discovery  
* **Fine-grained cost & latency tracking**  
* **Robust retries, quota & error handling**

This provider will power all analyst-agent agents (nlq_translator, sandbox_coder, etc.) in Phase 2.

## 1.5 · UPDATE: Native Gemini Support in CrewAI 0.5.0+

**Important Discovery**: CrewAI v0.5.0+ ships with **native Gemini support** via the built-in `LLM` class. A custom `BaseLLM` provider is **no longer required** 🎉.

### How to Use Gemini in CrewAI (Simplified)

1. **Environment Configuration** (`.env`)
```
MODEL=gemini/gemini-2.5-pro-preview-05-06
GEMINI_API_KEY=your_actual_api_key
```

2. **Python Usage**
```python
from crewai import LLM, Agent
import os

# Auto-configured from .env
gemini_llm = LLM()

# Or explicit
gemini_llm = LLM(
    model="gemini/gemini-2.5-pro-preview-05-06",
    api_key=os.getenv("GEMINI_API_KEY")
)

agent = Agent(
    role="Fraud Analyst",
    llm=gemini_llm,
    multimodal=True   # enables image input
)
```

### What This Means
* **Zero custom provider code** – CrewAI now handles message formatting, tool/function calling, etc.  
* **Multimodal** – Set `multimodal=True` on the agent; Gemini model must support images/video.  
* **Function calling** – Works out-of-the-box with CrewAI tools list.  
* **Cost tracking / telemetry** – Likely exposed via CrewAI; verify and extend metrics if needed.

### Modified Implementation Plan
Focus shifts from building a provider to:

1. **Configuration** – Ensure `.env` values propagate in all environments (local, Docker, CI).  
2. **Testing** – Create unit & integration tests confirming Gemini responses, tool calls, and multimodal flows.  
3. **Optimization** – Choose model tier per agent (e.g., Flash for fast tasks, Pro for heavy reasoning).  
4. **Monitoring** – Add Prometheus counters for token usage & USD cost if CrewAI default is insufficient.

---

*The remaining sections still provide deep insight into Gemini capabilities and advanced integration patterns should we need to extend CrewAI’s default implementation.*  

## 2 · CrewAI `BaseLLM` Requirements

| Method / Attr            | Required Behaviour |
|--------------------------|--------------------|
| `call(messages, tools?)` | Synchronous request → returns `LLMResponse` (text OR tool_call list) |
| `acall(...)`             | Async variant (returns `Awaitable`) |
| `supports_function_calling()` | `True` for Gemini models that expose `tool` response schema |
| `token_usage`            | Dict of {prompt_tokens, completion_tokens, total_tokens, cost_$} |
| `model_name` / `mode`    | e.g. `gemini-1.5-pro` / `"multimodal"` |
| Internal: `format_messages()` | Convert CrewAI chat messages → Gemini JSON |
| Internal: `parse_response()`  | Convert Gemini JSON → CrewAI response / ToolCall objects |

> **Tip**: Study `OpenAILLMProvider` in crewai-tools for reference.

...

*(rest of document unchanged)*
