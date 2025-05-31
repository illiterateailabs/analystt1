# GraphVisualization Component

Interactive fraud-analysis graph for **Analyst Augmentation Agent**

---

## 1. Purpose  
`GraphVisualization.tsx` renders the results of a CrewAI investigation (e.g. `fraud_investigation` crew) as an interactive network graph.  
It lets analysts:

* inspect entities (nodes) & relationships (edges)  
* filter by entity type, risk level and fraud pattern  
* view fraud-pattern / compliance findings & executive summary  
* export the graph as a PNG for reports

The component relies on **vis-network** / **vis-data** for rendering and React-Query for API calls.

---

## 2. Expected Data Format  

The component expects the **`/api/v1/crew/run`** endpoint to return `result` shaped as follows (see `backend/api/v1/crew.py`):

```jsonc
{
  "graph_data": {
    "nodes": [
      {
        "id": "account1",
        "label": "AC123456",
        "type": "Account",
        "properties": { "balance": 950000, "...": "..." },
        "risk_score": 0.82            // 0‒1, optional
      }
    ],
    "edges": [
      {
        "id": "e1",
        "from": "account1",
        "to": "tx1",
        "label": "INITIATES",
        "properties": { "timestamp": "2025-05-30T10:00:00Z" }
      }
    ]
  },
  "fraud_patterns": [
    {
      "id": "pattern1",
      "name": "Round Tripping",
      "description": "Funds returned to the originator",
      "confidence": 0.92,             // 0‒1
      "affected_entities": ["account1","tx1","account2"]
    }
  ],
  "risk_assessment": {
    "overall_score": 0.85,
    "factors": { "transaction_volume": 0.72, "...": 0.45 },
    "summary": "Overall risk level: HIGH"
  },
  "compliance_findings": [
    {
      "regulation": "AML Directive 5",
      "status": "non-compliant",     // compliant | non-compliant | warning
      "description": "Violation of AML-5 …",
      "recommendation": "File SAR within 24h"
    }
  ],
  "executive_summary": "Analysis identified 2 fraud patterns: …",
  "recommendations": [
    "File Suspicious Activity Reports (SARs)…",
    "Freeze accounts pending further investigation"
  ]
}
```

### Colour / Styling logic

| Element | Colour Source |
|---------|---------------|
| Node fill | `type` ➜ mapping in `nodeColors` |
| Node border | `risk_score` ➜ `riskColors` |
| Edge arrow | Always “to” |
| Fraud pattern cards | red border / bg |
| Compliance finding cards | green / yellow / red based on `status` |

---

## 3. Features

| Capability | Notes |
|------------|-------|
| **Tabs** | Natural Language → Cypher → Analytics → **Crew Results** |
| Crew execution UI | choose crew, provide input, run mutation |
| Interactive graph | zoom/pan, select node, tooltips, physics layout |
| Filters | entity type, risk level, fraud-pattern |
| Details panel | node properties, connected patterns |
| Legend | shows icon + colours + risk border |
| Export | PNG download via `<canvas>.toDataURL()` |
| Mock data support | import `generateMockFraudInvestigation()` from `src/lib/mockData` |

---

## 4. Usage

```tsx
import { GraphVisualization } from '@/components/graph/GraphVisualization'

export default function GraphView() {
  return <GraphVisualization />
}
```

No props required – component manages its own state and queries.

---

## 5. Testing with Mock Data

During frontend development you may not have the backend running.  
Use the mock generator:

```ts
import { generateSimpleMockFraudInvestigation } from '@/lib/mockData'

// in component dev console
setCrewResult(generateSimpleMockFraudInvestigation())
```

or replace the crew API call with:

```ts
crewRunMutation.mutateAsync = async () =>
  ({ data: { result: generateMockFraudInvestigation(30, 'high') } } as any)
```

---

## 6. Dev Notes

* **Dependencies**  
  - `vis-network`, `vis-data` for graph  
  - `uuid` for mock IDs  
  - FontAwesome CDN added in `globals.css` for vis icon groups  
* **State**  
  - filters & selected node stored in local state  
  - React-Query caches schema / crew lists  
* **Extensibility**  
  - add new node types by extending `nodeColors` & `groups` mapping  
  - support additional risk/metric visual cues by adjusting `riskColors` & legend  
* **Performance**  
  - big graphs: increase `barnesHut` spring length / enable clustering  
  - filters hide nodes via `hidden: true` flag (cheaper than removing datasets)

---

## 7. Future Improvements

* Persist filter & layout state per investigation  
* Allow multi-select highlighting of fraud-pattern entities  
* Add mini-map navigator (vis-timeline plugin)  
* Integrate time-slider to animate transactions chronologically  
