# Agent & Crew Configuration Guide

This document explains **how to customise the behaviour of the Analyst Agent** without editing core Python code.  
All runtime-specific tweaks live in simple **YAML files** under `backend/agents/configs/`.  
If a YAML file is missing, the application automatically falls back to the Python-defined defaults located in  
`backend/agents/config.py` (`DEFAULT_AGENT_CONFIGS` and `DEFAULT_CREW_CONFIGS`).

---

## 1. Configuration Resolution Order

```
📦 backend
 └─ agents
    ├─ configs            ← 1️⃣  Your YAML overrides (optional)
    │   ├─ <agent_id>.yaml
    │   └─ crews/
    │       └─ <crew_name>.yaml
    └─ config.py          ← 2️⃣  Library defaults
```

1. **YAML override present** → it is loaded and merged.
2. **No YAML file** → the corresponding entry from `DEFAULT_*_CONFIGS` is used.
3. **Missing from both** → the application raises an error at start-up.

---

## 2. Directory Layout

| Path                                   | Purpose                               |
|----------------------------------------|---------------------------------------|
| `backend/agents/configs/`              | Root folder for all custom configs    |
| `backend/agents/configs/<agent>.yaml`  | Per-agent override                    |
| `backend/agents/configs/crews/`        | Folder that holds crew definitions    |
| `backend/agents/configs/crews/<crew>.yaml` | Per-crew override                   |

> Empty folders are **okay** – defaults will still work.

---

## 3. YAML Schema & Example

### 3.1 Agent YAML

```yaml
id: graph_analyst                       # (string) required – must match agent_id
role: "Graph Data Scientist"            # Short title
goal: "Deep-dive graph analytics"       # High-level purpose
backstory: |                            # Free-form multiline text
  You hold a PhD in network science...
tools:                                  # (list[str]) tool names from CrewFactory
  - graph_query_tool
  - neo4j_schema_tool
verbose: true                           # (bool) include reasoning in output
allow_delegation: false                 # (bool) can spawn sub-agents
max_iter: 5                             # (int) task attempt limit
max_rpm: 15                             # (int|null) requests/min rate-limit
memory: true                            # (bool) agent-level vector memory
llm_model: gemini-1.5-pro               # (string) override default LLM
```

### 3.2 Crew YAML

```yaml
crew_name: fraud_investigation          # (string) required
process_type: sequential                # sequential | hierarchical
manager: orchestrator_manager           # (string|null) agent_id that leads
agents:                                 # (list[str]) ordered execution list
  - nlq_translator
  - graph_analyst
verbose: true                           # (bool) stream intermediate output
max_rpm: 30                             # (int|null) crew-level rate-limit
memory: true                            # (bool) share vector memory
cache: true                             # (bool) enable tool / LLM cache
```

---

## 4. Available Configuration Keys

| Key                | Type     | Agents | Crews | Description                                                                                 |
|--------------------|----------|:------:|:-----:|---------------------------------------------------------------------------------------------|
| `id` / `crew_name` | string   | ✔      | ✔     | Unique identifier (must match file name)                                                    |
| `role`             | string   | ✔      | –     | One-line description                                                                        |
| `goal`             | string   | ✔      | –     | Mission statement                                                                           |
| `backstory`        | string   | ✔      | –     | Long-form context the LLM sees                                                              |
| `tools`            | list[str]| ✔      | –     | Names of tools registered inside `CrewFactory`                                              |
| `process_type`     | string   | –      | ✔     | `sequential` or `hierarchical` execution flow                                               |
| `manager`          | string   | –      | ✔     | Agent that orchestrates others (optional)                                                   |
| `agents`           | list[str]| –      | ✔     | Ordered list of agent ids involved in this crew                                             |
| `verbose`          | bool     | ✔      | ✔     | Stream reasoning steps (may increase token cost)                                            |
| `allow_delegation` | bool     | ✔      | –     | Permit agent to spin up sub-agents                                                          |
| `memory`           | bool     | ✔      | ✔     | Persist vector memory between tasks                                                         |
| `max_iter`         | int      | ✔      | –     | Maximum reasoning loops per task                                                            |
| `max_rpm`          | int/null | ✔      | ✔     | Requests-per-minute safety valve                                                            |
| `cache`            | bool     | –      | ✔     | Enable/disable LRU + result cache                                                           |
| `llm_model`        | string   | ✔      | –     | Override default LLM model name                                                             |

---

## 5. Overriding Examples

### 5.1 Adjust an Existing Agent

Create `backend/agents/configs/nlq_translator.yaml`:

```yaml
max_iter: 8
verbose: true
llm_model: gemini-1.5-flash
```

Only the supplied keys are overridden; all others inherit from
`DEFAULT_AGENT_CONFIGS["nlq_translator"]`.

### 5.2 Spin Up a Custom Crew

`backend/agents/configs/crews/quick_risk_scan.yaml`:

```yaml
crew_name: quick_risk_scan
process_type: sequential
agents:
  - nlq_translator
  - graph_analyst
  - report_writer
verbose: false
memory: false
cache: true
```

After the file is added, the new crew automatically appears in:

```
GET /api/v1/crew/crews
```

---

## 6. Reference Defaults

If you are unsure about the baseline settings, open:

```
backend/agents/config.py
```

and review:

```python
DEFAULT_AGENT_CONFIGS  # dict[str, dict]
DEFAULT_CREW_CONFIGS   # dict[str, dict]
```

Copy the relevant snippet into a YAML file and tweak only what you need.

Happy configuring! 🎛️
