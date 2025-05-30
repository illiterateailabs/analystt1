# Crew Configuration Overrides

This folder is for **crew-specific YAML files** that override or extend the defaults shipped with the Analyst Agent.  
If no file is placed here for a crew, the application falls back to the Python defaults in `backend/agents/config.py`.

---

## Built-in Crews

| Crew ID              | Purpose (short)                             |
|----------------------|---------------------------------------------|
| `fraud_investigation`| End-to-end forensic investigation workflow  |
| `alert_enrichment`   | Real-time context & risk scoring for alerts |
| `red_blue_simulation`| Synthetic fraud scenario red/blue exercises |

You may reference these IDs directly in API calls, or create YAML files here to customise their behaviour.

---

## Quick Override Example

Create `fraud_investigation.yaml` in this folder:

```yaml
# backend/agents/configs/crews/fraud_investigation.yaml
crew_name: fraud_investigation
process_type: sequential
verbose: false          # silence intermediate reasoning
max_rpm: 20             # tighten rate-limit
agents:                 # keep same order but swap report writer model
  - nlq_translator
  - graph_analyst
  - fraud_pattern_hunter
  - sandbox_coder
  - compliance_checker
  - report_writer
```

Only the keys provided are overridden; everything else inherits from the original definition.

---

## Need More Details?

See the **parent README** in `backend/agents/configs/` for:

* YAML schema reference  
* Resolution order  
* Full list of agent-level keys

Happy crew-tuning! 🛠️
