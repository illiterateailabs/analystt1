# Agent Prompt Management – Implementation Guide

_Last updated: **30 May 2025**_

---

## 1 · Feature Overview  
The **Agent Prompt Management** module allows developers and power-users to **view, edit, reset and test system prompts** for every CrewAI agent at runtime—without code changes or container rebuilds. It closes a critical feedback loop: analysts can fine-tune agent behaviour in minutes, observe the impact immediately, and roll back to a known-good default when needed.

Key capabilities  
| Capability | Outcome |
|------------|---------|
| List all agents & prompt status | Quickly spot which agents run **Custom** vs **Default** prompts |
| Fetch single agent prompt | Inspect full system prompt, description & metadata |
| Edit & save prompt | Persist changes to disk, hot-reload CrewFactory cache |
| Reset to default | Remove custom file, revert to repo default |
| Secure access | Endpoints protected by existing JWT auth dependency |

---

## 2 · Backend Implementation

| File | Purpose |
|------|---------|
| `backend/api/v1/prompts.py` | FastAPI router exposing CRUD endpoints |
| `backend/agents/factory.py` (updates) | New methods `update_agent_prompt`, `reset_agent_prompt`, cache invalidation |
| `backend/main.py` | Router mounted at `/api/v1/prompts` |
| `backend/agents/configs/defaults/*.yaml` | Version-controlled default prompts (e.g. `nlq_translator.yaml`) |

### 2.1 Router Highlights
* **In-memory cache** `_prompt_cache` stores merged view of defaults ➜ custom files.  
* YAML persisted to `backend/agents/configs/<agent_id>.yaml` on save.  
* Default prompts live in `backend/agents/configs/defaults/`; custom file deletion triggers reset.  
* CrewFactory cache invalidated so new conversations pick up edits instantly.  
* All routes guarded by `Depends(get_current_user)` → JWT required.

### 2.2 Persistence Rules
```
defaults/         # immutable, version-controlled
└── nlq_translator.yaml
configs/
├── nlq_translator.yaml   # created on first edit → overrides default
└── crews/…               # crew YAML unaffected
```

---

## 3 · Front-End UI

| File | Purpose |
|------|---------|
| `src/components/prompts/PromptsManager.tsx` | React (client) component: list sidebar + editor panel |
| `src/app/prompts/page.tsx` | Next.js page wrapper + copy |
| `src/lib/api.ts` | Axios helpers `listAgents`, `getAgentPrompt`, `updateAgentPrompt`, `resetAgentPrompt` |
| `src/app/page.tsx` | Navigation updated (`PencilSquareIcon`) |

UX features  
• Responsive split-pane (sidebar / editor)  
• **Unsaved-changes** guard & toast notifications  
• Markdown-friendly textarea (`font-mono`) for large prompts  
• Tags showing _Default_ / _Custom_ per agent

---

## 4 · API Reference

| Method & Path | Payload | Response | Notes |
|---------------|---------|----------|-------|
| **GET** `/api/v1/prompts` | – | `{ agents: [ { agent_id, description, has_custom_prompt } ] }` | List agents |
| **GET** `/api/v1/prompts/{agent_id}` | – | `PromptResponse` | Fetch current prompt |
| **PUT** `/api/v1/prompts/{agent_id}` | `PromptUpdate` | `PromptResponse` | Save (creates/overwrites custom file) |
| **POST** `/api/v1/prompts/{agent_id}/reset` | – | `PromptResponse` (is_default=true) | Delete custom → revert |

`PromptResponse` fields  
```
agent_id        string
system_prompt   string
description     string?
metadata        dict?
is_default      bool
```

---

## 5 · Using the Feature

### 5.1 Through the UI  
1. Sign-in → open _Prompts_ tab in sidebar.  
2. Select agent → editor loads default or custom prompt.  
3. Edit text, click **Save Changes**.  
4. Run new crew; agent will use the updated prompt.  
5. Click **Reset to Default** anytime to discard local overrides.

### 5.2 Via API (cURL)
```bash
# List agents
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/prompts

# Update prompt
curl -X PUT http://localhost:8000/api/v1/prompts/nlq_translator \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"system_prompt": "You are a translator… v2"}'

# Reset
curl -X POST http://localhost:8000/api/v1/prompts/nlq_translator/reset \
     -H "Authorization: Bearer $TOKEN"
```

---

## 6 · Example Workflow

1. **Analyst** sees false-positives from `fraud_pattern_hunter`.  
2. Opens Prompts UI, tweaks risk-scoring paragraph, saves.  
3. Runs _fraud_investigation_ crew again → improved results.  
4. Auditor requests revert; analyst hits **Reset to Default** and reruns.  
5. Changes captured in Git diff (`backend/agents/configs/fraud_pattern_hunter.yaml`) for audit trail.

---

## 7 · Technical Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| YAML on disk vs DB | Git-versionable, human-readable, minimal new deps |
| Cache in memory | Avoid disk I/O per request; latency ≤1 ms |
| Router auth | Re-uses existing JWT middleware; no new ACL layer |
| CrewFactory invalidation | Guarantees new prompt on next agent build; running tasks unaffected |
| Separate `defaults/` dir | Protects blessed prompts from accidental overwrite |

---

## 8 · Future Enhancements

* **Version history UI** – diff & rollback among previous custom edits.  
* **Per-environment overrides** – load prompts from S3 or Secrets Manager in prod.  
* **Live-reload running agents** – propagate prompt change to ongoing conversations.  
* **Prompt templates** – parameterised snippets (e.g., jurisdiction, language).  
* **RBAC granularity** – restrict who can edit vs. view prompts.  
* **Validation hooks** – lint for forbidden tokens / maximum length before save.  
* **Metrics** – Prometheus counter `agent_prompt_edits_total{agent_id}`.  

---

### 📂 File created by: `PROMPT_MANAGEMENT_IMPLEMENTATION.md`  
Keep this document in **memory-bank/** or **docs/** for onboarding and audits.  
