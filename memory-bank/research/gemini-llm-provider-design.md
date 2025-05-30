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

---

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

---

## 3 · Message & Prompt Formatting

```
CrewMessage(role="assistant", content="...")  -->  { "role": "model", "parts": [ { "text": "..." } ] }
CrewToolCall(name="GraphQueryTool.run", args={...})  --> function call JSON
Images -> { "inline_data": { "mime_type": "image/png", "data": "<base64>" } }
```

* Maintain **system prompt** slot – prepend to conversation.
* Include **tool schema** in `tools=` param when agent has tools.

---

## 4 · Multimodal Support

Gemini’s `/v1/models/*:generateContent` endpoint accepts **text + image parts**.

Implementation plan:

1. Extend CrewAI message format with helper `ImagePart(path|bytes)` (if not already).
2. `GeminiLLMProvider.format_messages()` detects `ImagePart` and attaches as base64.
3. For Phase 2 only `fraud_pattern_hunter` (image OCR patterns) will use this; others remain text-only.

Limitation: CrewAI executor currently treats prompts as text—add shim that allows agent to pass `{"image_bytes": ...}` in tool args until upstream change.

---

## 5 · Function Calling & Tool Execution

Gemini function-calling ≈ OpenAI:

* Provide `tools=[{"name": "...", "description": "...", "parameters": {...}}]` in request.
* Gemini may return:
  ```json
  {
    "candidates": [
      { "content": { "parts": [
          { "functionCall": { "name": "GraphQueryTool.run", "argsJson": "{...}" } }
      ]}}
    ]
  }
  ```

Parsing rules:

1. Detect `functionCall`; map to `CrewToolCall(name, args_dict)`.
2. For **multiple** calls in a single response, iterate `parts`.
3. Enforce **one tool call** per step for MVP; queue additional parts into assistant text.

---

## 6 · MCP Native Support

Google Gemini SDK is MCP-aware (tool descriptions via [MCP spec §4](https://modelcontext.org)).  
Provider strategy:

* If `tools` list contains an object with `"mcp_server": "url"` key, fetch & merge schema **before** API call (caching for TTL = 1 h).
* Validate schema -> convert to Gemini tool JSON.
* For SSE-based MCP servers ensure origin whitelist (security note in systemPatterns).

---

## 7 · Cost Tracking & Observability

| Metric | How |
|--------|-----|
| **Tokens** | Use `usageMetadata` fields (`promptTokenCount`, `candidatesTokenCount`) |
| **Unit cost** | Map model → $/1k tokens (env config) |
| **Latency** | `time.perf_counter()` around API call |
| **Retries** | Counter `llm_retries_total{model="gemini"}` |
| **Errors** | Counter `llm_errors_total{code=*}` |

Expose Prometheus gauges via `backend/core/metrics.py`.

Log sample:

```
{"event":"llm_call","model":"gemini-1.5-pro","tokens":123,"cost":0.0031,"latency":1.24}
```

---

## 8 · Error Handling & Rate Limiting

| Failure | Strategy |
|---------|----------|
| 429 Rate limit | Exponential back-off (`base=1s`, `max=5 retry`, jitter) |
| 5xx transient | Same retry policy; circuit-breaker after N failures/min |
| Network timeouts | 30 s default → retry |
| Invalid tool schema | Log & fallback to plain completion |
| Context length | Pre-validate token count (tiktoken-style estimator) |

All errors surface to calling agent as `ToolError` or `LLMError` with reason.

---

## 9 · Implementation Roadmap

| Day | Task | Owner |
|-----|------|-------|
| **D0** | Scaffold `GeminiLLMProvider` class, unit-test harness | BE |
| **D1** | Implement `format_messages`, `parse_response` for text-only | BE |
| **D2** | Add function-calling support, tool schema adapter | BE |
| **D3** | Integrate cost & latency metrics, Prometheus exporter | DevOps |
| **D4** | Add retry / error policy, env config (`GEMINI_MAX_RETRIES`) | BE |
| **D5** | Multimodal image part support; mock test with PNG | BE |
| **D6** | MCP tool descriptor fetch + cache layer | BE |
| **D7** | End-to-end tests with nlq_translator and sandbox_coder | QA |
| **D8** | Documentation (`techContext.md`, API examples) | Docs |
| **D9** | Review, security audit (keys, quotas) | Security |
| **D10** | Merge & enable for all agents (toggle via env) | Lead |

Stretch: streaming support via `streamGenerateContent` (Phase 3).

---

## 10 · Testing Strategy

* **Unit Tests** – Mock `google.generativeai.GenerativeModel.generate_content_async`  
* **Contract Tests** – Validate function call JSON → CrewAI ToolCall mapping  
* **Load Tests** – 50 concurrent calls, monitor latency, rate-limit hit  
* **Chaos** – Inject 500/503 responses, ensure graceful retries  
* **Multimodal Demo** – Upload sample SAR PDF image, ensure OCR summarised.

---

## 11 · Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Gemini rate-limit lower than required throughput | Parallelism ≤ model quota; fallback smaller model; request batching |
| Cost spike | Prometheus alert on `$ llm_cost_total` |
| MCP servers return malicious schema | Trust list + JSON schema validation |
| CrewAI API changes | Pin crewai version (=0.5.x) + integration tests |

---

## 12 · References

* CrewAI BaseLLM docs – <https://docs.crewai.org/internals#basellm>
* Google Generative AI Python SDK – <https://github.com/google-gemini/generative-ai-python>
* Model Context Protocol v0.1 – <https://modelcontext.org>
* System patterns – see `memory-bank/systemPatterns.md`
