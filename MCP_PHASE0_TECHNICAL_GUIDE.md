# MCP Phase-0 Technical Guide  
*Hands-on steps to enable Model Context Protocol in analystt1*  
**Document ID:** MCP_PHASE0_TECHNICAL_GUIDE.md · **Applies to:** Branch `droid/gnn-fraud-detection` (or newer)  
**Author:** Marian Stanescu · **Last updated:** 02 Jun 2025

---

## 📍 Phase-0 Objective
> “Connect Gemini 2.5 ➜ MCP client ➜ sample MCP server ➜ CrewAI agent”  
No production refactors yet—just prove the pipeline works end-to-end.

Success criteria  
- `pytest tests/test_mcp_poc.py` passes  
- `make mcp-demo` prints live tool list & executes a tool call through Gemini 2.5 Flash  
- CI job `mcp-poc` green (≈60 s)

---

## 1  Development Environment

| Item | Version |
|------|---------|
| Python | 3.11 (same as backend) |
| Node.js | ≥20 (only for JS sample servers) |
| uv / pip | `uv==0.2.*` recommended |
| Docker | 24.x (compose v2 CLI) |
| OS | Linux/macOS/WSL2 |

Clone & activate venv:

```bash
git checkout droid/gnn-fraud-detection
uv venv .venv && source .venv/bin/activate
```

---

## 2  Dependencies

Add MCP libs **only** for dev extras first.

```bash
uv pip install \
  "crewai-mcp-toolbox>=0.2" \
  "mcpengine>=0.3" \
  "google-generativeai>=1.19" \
  --constraint constraints.txt
```

`constraints.txt` already pins httpx etc.—stay inside file to avoid CI drift.

---

## 3  Proof-of-Concept MCP Server

We’ll build a tiny **Echo Tool** in `backend/mcp_servers/echo_server.py`.

```python
# backend/mcp_servers/echo_server.py
from mcpengine import Server, Tool

server = Server(name="echo-server")

@server.tool(
    name="echo",
    description="Return whatever string the user sends back",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
)
def echo(text: str) -> str:                 # noqa: D401
    return f"ECHO: {text}"

if __name__ == "__main__":
    # stdio transport by default
    server.run()
```

### Local run

```bash
python backend/mcp_servers/echo_server.py
# (process now waits for JSON-RPC on stdin)
```

> Tip  Add an `npx @modelcontextprotocol/server-filesystem ./mcp_demo` if you want a second low-code server.

---

## 4  CrewAI Integration via `crewai-mcp-toolbox`

### 4.1 New registry file  

`config/mcp_servers.yaml`

```yaml
echo:
  command: "python backend/mcp_servers/echo_server.py"
  transport: stdio
```

### 4.2 Update CrewFactory  

```python
# backend/agents/factory.py  (excerpt)
from crewai_mcp_toolbox import MCPToolSet

def init_mcp_tools() -> list:
    """Spin up MCP servers declared in YAML and return dynamic tools."""
    toolset = MCPToolSet(registry_path="config/mcp_servers.yaml")
    return list(toolset)   # Each item is a dynamically generated BaseTool
```

Add to factory init:

```python
TOOLS: list[BaseTool] = [
    *load_builtin_tools(),
    *init_mcp_tools(),          # ← NEW
]
```

On startup `MCPToolSet`:
1. Launches echo-server subprocess
2. Discovers schema
3. Creates `EchoTool` subclass (CrewAI-compatible)
4. Inserts into agent namespace

---

## 5  Connecting Gemini 2.5 with MCP

Modify `backend/integrations/gemini_client.py`:

```python
class GeminiClient:
    def _with_mcp(self) -> dict:
        # Use MCP tools if present in current context
        return {"tools": {"mcp": True}} if os.getenv("ENABLE_MCP") == "1" else {}
    
    async def generate_content(self, prompt: str, **kwargs):
        cfg = genai.types.GenerationConfig(
            # …
            **self._with_mcp()
        )
        # rest unchanged
```

Run with env:

```bash
export ENABLE_MCP=1
export GEMINI_API_KEY=<key>
```

Gemini will now **natively** call the `echo` tool when it decides.

---

## 6  Code Samples & Config Recap

### Makefile targets

```make
.PHONY: mcp-echo mcp-demo
mcp-echo:
	python backend/mcp_servers/echo_server.py

mcp-demo:
	python scripts/mcp_demo.py   # see below
```

### Demo script

```python
# scripts/mcp_demo.py
import asyncio, os
from google import genai
from backend.agents.factory import init_mcp_tools

async def main():
    init_mcp_tools()  # launch echo server
    model = genai.GenerativeModel("gemini-2.5-flash")
    chat = model.start_chat()
    resp = await chat.send_message_async("Please echo 'hello MCP'")
    print(resp.text)

if __name__ == "__main__":
    os.environ["ENABLE_MCP"] = "1"
    asyncio.run(main())
```

---

## 7  Testing Procedures

### 7.1 Unit test – schema discovery

```python
# tests/test_mcp_server.py
from crewai_mcp_toolbox import MCPToolSet

def test_echo_tool_discovery():
    tools = MCPToolSet("config/mcp_servers.yaml")
    names = [t.name for t in tools]
    assert "echo" in names
```

### 7.2 Integration – Gemini loop

```python
# tests/test_mcp_poc.py
import os, asyncio
from google import genai
from backend.agents.factory import init_mcp_tools

async def ask_gemini():
    os.environ["ENABLE_MCP"] = "1"
    init_mcp_tools()
    model = genai.GenerativeModel("gemini-2.5-flash")
    chat = model.start_chat()
    r = await chat.send_message_async("echo 'abc'")
    return r.text

def test_gemini_tool_call():
    text = asyncio.run(ask_gemini())
    assert "ECHO: abc" in text
```

Add to `pytest.ini`:

```
markers =
    mcp: phase0 mcp tests
```

---

## 8  Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `KeyError: 'tools' not allowed` from Gemini | Using old model (2.0) | Use `gemini-2.5-pro` or flash |
| EchoTool never called | Prompt not triggering tool | Prepend “use the echo tool” to prompt; or lower temperature |
| `JSONDecodeError` in MCP server | Sent malformed JSON-RPC | Ensure client uses `crewai-mcp-toolbox`; keep newline framing |
| `BrokenPipeError` when stopping | Stdio subprocess autokilled before pending request | Wrap in `with MCPToolSet(...)` context or call `cleanup()` |

---

## 9  Next Step Checklist ✓

- [ ] Commit PoC server and YAML registry  
- [ ] Add `ENABLE_MCP` env variable to `.env.example`  
- [ ] Merge tests into CI matrix (`pytest -m mcp`)  
- [ ] Demo recorded (loom) for team review

When all green, move to **Phase 1** (wrap GraphQueryTool & GNN server).

---

*End of MCP Phase-0 Technical Guide*  
