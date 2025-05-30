# Pattern Library for Fraud Detection  
`backend/agents/patterns/`

This folder contains **machine-readable templates** that describe money-laundering, fraud, tax-evasion and other financial-crime motifs.  
The `PatternLibraryTool` loads every `*.yaml` or `*.json` file in the directory tree at application start-up, validates the schema, and converts a selected pattern into an executable **Cypher** query (rule-based or LLM-assisted).

---

## 1  Directory layout

```
backend/agents/patterns/
├── README.md          ← this file
├── money_laundering/
│   └── circular_transaction.yaml
├── structuring/
│   └── rapid_succession.yaml
└── … (add more patterns here)
```

* Put each pattern in its own file.  
* Use sub-folders (`money_laundering/`, `tax_evasion/`, …) to keep things tidy.  
* File name **should match** the `id` field inside the document.

---

## 2  Pattern file schema

Every pattern file **MUST** be valid YAML **or** JSON and include the following top-level keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | string | ✓ | Unique snake-case identifier (`circular_transaction`) |
| `name` | string | ✓ | Human-readable title |
| `description` | string | ✓ | One-paragraph explanation of the scheme |
| `type` | string | ✓ | Category (`money_laundering`, `structuring`, …) |
| `severity` | string | ✕ | `low` \| `medium` \| `high` – risk impact |
| `parameters` | map | ✕ | Tunable inputs (see §3) |
| `cypher_template` | string | ✕\* | Ready-to-run Cypher with `{placeholders}` |
| `template` | map | ✕\* | Declarative node/relationship motif (used by LLM) |

\* At least **one** of `cypher_template` **or** `template` is required.

### 2.1  Node/Relationship template (optional)

If you prefer a declarative graph motif (and let the LLM build the final query) supply:

```yaml
template:
  nodes:
    - id: origin
      labels: [Account]
      properties: {}
    - id: intermediary
      labels: [Account]
      properties: {}
      repeat: {min: 1, max: 5}
  relationships:
    - start: origin
      end: intermediary.0
      type: TRANSFER
      properties:
        amount: {param: min_amount, default: 10000}
    - start: intermediary.i
      end: intermediary.i+1
      type: TRANSFER
      repeat: {min: 0, max: 4}
    - start: intermediary.last
      end: origin
      type: TRANSFER
```

Notation rules:

* `repeat` may appear on **nodes** (creates indexed aliases `intermediary.0 … intermediary.n`) or **relationships**.
* `param:` placeholder tells the engine to insert a runtime parameter.

---

## 3  Parameters section

```yaml
parameters:
  max_hops:
    type: int
    default: 6
    description: Maximum length of the loop
  min_amount:
    type: float
    default: 10000
    description: Minimum amount each hop must carry
  limit:
    type: int
    default: 10
    description: Max rows returned
```

Supported `type`: `int`, `float`, `string`, `bool`, `date`, `duration`.

Values supplied at runtime **override** the defaults.

---

## 4  Conversion methods

* **Rule-based** – If `cypher_template` exists, the engine performs a safe, deterministic `str.replace("{param}", value)` plus minor helpers (e.g. `{min_amount_condition}` auto-expands).
* **LLM-based** – If only `template` is present (or `conversion_method: llm` is requested) the `Gemini` model receives the declarative motif + parameters + DB schema and returns optimized Cypher.
* **Hybrid** – Try rule-based first, fall back to LLM if template missing or substitution fails.

---

## 5  Examples

### 5.1  Minimal (rule-based)

```yaml
id: large_single_transfer
name: Single Transfer > $1M
description: Flags any single transaction greater than a configurable threshold
type: high_value
severity: medium

parameters:
  threshold:
    type: float
    default: 1000000
    description: Dollar amount trigger
  limit:
    type: int
    default: 20

cypher_template: |
  MATCH (a:Account)-[t:TRANSFER]->(b:Account)
  WHERE t.amount >= {threshold}
  RETURN a, b, t
  ORDER BY t.amount DESC
  LIMIT {limit}
```

### 5.2  Advanced (LLM template)

See `money_laundering/circular_transaction.yaml` in this folder for a full nodal motif with repeats and dynamic parameters.

---

## 6  Testing your pattern

1. Run the PatternLibraryTool directly:

```bash
curl -X POST http://localhost:8000/api/v1/crew/run \
     -H "Authorization: Bearer <token>" \
     -d '{"crew_name":"fraud_investigation",
          "inputs":{"operation":"convert",
                    "pattern_id":"circular_transaction",
                    "parameters":{"min_amount":15000}}}'
```

2. Copy the generated Cypher and execute in Neo4j Browser or **/api/v1/graph/query** endpoint.
3. Refine parameters & template until results are as expected.
4. Add unit tests in `tests/test_pattern_library.py`.

---

## 7  Best practices

* **Keep Cypher safe** – always add a `LIMIT` placeholder.
* **Be explicit** – document every parameter with `description`.
* **Version control** – change `id` or add `v2` suffix when making breaking changes.
* **One concern per file** – granular patterns combine better than monoliths.
* **Use high-level labels** – let analysts tune `min_amount`, `max_hops`, etc.

Happy hunting!  
— Pattern Library Maintainers
