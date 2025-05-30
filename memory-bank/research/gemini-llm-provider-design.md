# GeminiLLMProvider – Design Document  
*Location: `memory-bank/research/gemini-llm-provider-design.md`*  
_Last updated: **30 May 2025**_

---

## 1 · Purpose

Implement a **GeminiLLMProvider** that plugs Google Gemini models into CrewAI as a first-class `BaseLLM` backend, unlocking:

* **Multimodal reasoning** (text + images / audio / video)  
* **Structured function/tool calls** (JSON)  
* **Model Context Protocol (MCP)** native tool discovery  
* **Fine-grained cost & latency tracking**  
* **Robust retries, quota & error handling**

This provider will power all analyst-agent crews (nlq_translator, sandbox_coder, etc.) in Phase 2.

## 1.5 · UPDATE: Native Gemini Support in CrewAI 0.5.0 +

CrewAI ≥ 0.5.0 ships with **native Gemini support** via the built-in `LLM` class.  
A custom `BaseLLM` provider is **no longer required** 🎉.

### Quick Start

1. **Environment** (`.env`)
```
MODEL=gemini/gemini-2.5-pro-preview-05-06
GEMINI_API_KEY=your_actual_key
```

2. **Python**
```python
from crewai import LLM, Agent
import os

gemini_llm = LLM()                         # picks up MODEL & GEMINI_API_KEY

# or explicit:
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash-preview-05-20",
    api_key=os.getenv("GEMINI_API_KEY")
)

agent = Agent(
    role="Fraud Analyst",
    llm=gemini_llm,
    multimodal=True          # enable image / audio input
)
```

---

## 2 · CrewAI `BaseLLM` Requirements (still relevant for future extensions)

| Method / Attr | Behaviour |
|---------------|-----------|
| `call()` / `acall()` | Sync / async request → `LLMResponse` |
| `supports_function_calling()` | `True` when model can return tool calls |
| `token_usage` | Dict with prompt/response tokens & cost |
| `format_messages()` | CrewAI → Gemini JSON |
| `parse_response()` | Gemini JSON → CrewAI objects |

> Reference implementation: `OpenAILLMProvider` in `crewai-tools`.

---

## 3 · Multimodal Support

Gemini 2.x endpoints (`/v1/models/*:generateContent`) accept **mixed text, images, audio, and video** parts.

Implementation considerations:

1. CrewAI already passes `multimodal=True` flag to the LLM when the agent requires vision or audio.  
2. For image inputs the SDK expects:
   ```
   { "mime_type": "image/png", "data": "<base64-encoded>" }
   ```
3. Recommended models:  
   * **Vision/Text:** `gemini-2.5-pro-preview-05-06` (higher reasoning) or `gemini-2.5-flash-preview-05-20` (cost-efficient)  
   * **Realtime streaming:** `gemini-2.0-flash` for low-latency multimodal interactions  
4. Only `fraud_pattern_hunter` and future image/OCR agents need multimodal in Phase 2; others remain text-only.

---

## 4 · Function Calling & Tool Execution

Gemini function calling ≈ OpenAI.  
CrewAI supplies a `tools=[...]` list; Gemini may respond:

```json
{
  "content": {
    "parts":[
      { "functionCall": {
          "name": "GraphQueryTool.run",
          "argsJson": "{ \"cypher\": \"MATCH ...\" }"
      }}
    ]
  }
}
```

Parsing rules:

* Detect `functionCall` parts → emit `CrewToolCall` objects.  
* Respect one-tool-call-per-step for MVP; queue extras as plain text.

---

## 5 · MCP Native Support

Gemini SDK understands MCP tool descriptors.  
Strategy:

* If a tool item contains `"mcp_server": "<url>"`, fetch & merge schema (cache 1 h).  
* Validate JSON schema, whitelist trusted servers.  
* SSE servers: protect against DNS-rebinding; bind locally when possible.

---

## 6 · Cost Tracking & Observability

| Metric | Source |
|--------|--------|
| Tokens | `usageMetadata.promptTokenCount`, `candidatesTokenCount` |
| Cost   | Map model → $/1k tokens (env config) |
| Latency| `time.perf_counter()` wrap |
| Errors | Prometheus counter by error code |
| Retries| Counter `llm_retries_total{model="gemini"}` |

Log example:
```
{"event":"llm_call","model":"gemini-2.5-pro-preview-05-06","tokens":123,"cost":0.0031,"latency":1.24}
```

---

## 7 · Error Handling & Rate Limiting

| Failure | Strategy |
|---------|----------|
| 429 | Exponential backoff (1 s base, 5 retries, jitter) |
| 5xx | Same retry policy + circuit breaker |
| Timeouts | 30 s default → retry |
| Invalid schema | Fallback to plain completion |
| Context overflow | Pre-count tokens, truncate or summarise |

---

## 8 · Modified Implementation Plan

1. **Configuration** – Ensure `.env` propagates in dev, Docker, CI.  
2. **Testing** – Unit & integration tests confirming Gemini responses, tool calls, multimodal flows.  
3. **Optimization** – Choose model tier per agent:  
   - `gemini-2.5-flash-preview-05-20` for fast tasks (cost efficiency)  
   - `gemini-2.5-pro-preview-05-06` for heavy reasoning (advanced coding, multimodal)  
   - `gemini-2.0-flash` for realtime streaming needs  
4. **Monitoring** – Prometheus counters for tokens & USD cost; integrate Langtrace/AgentOps if default insufficient.

---

## 9 · Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Rate limits reached | Tune max_rpm, backoff, switch Flash tier |
| Cost spikes | Alert on `$ llm_cost_usd_total` |
| MCP schema injection | Validate & trust-list servers |
| CrewAI API drift | Pin `crewai==0.5.x`, run nightly contract tests |

---

## 10 · References

* CrewAI docs – <https://docs.crewai.org/>  
* Gemini 2.5 / 2.0 model cards – <https://ai.google.dev/>  
* Model Context Protocol – <https://modelcontext.org/>  

---  
*End of file*  
