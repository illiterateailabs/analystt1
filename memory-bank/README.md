# Memory Bank – How We Document & Stay Sane  
*“My memory resets after every session. These files are my only link to the past.”*

---

## 1 · Why the Memory Bank Exists  
After each IDE reset I (the agent) wake up with zero working memory.  
The **Memory Bank** is the *single source of truth* that lets me:

* Re-learn the entire project in minutes.  
* Resume work exactly where the previous session stopped.  
* Keep decisions, rationale and patterns explicit and searchable.

If a fact is *not* in the Memory Bank, it **does not exist** for future-me.

---

## 2 · Folder Layout  

```
memory-bank/
├── projectbrief.md        # Foundation doc – scope & goals
├── productContext.md      # Why & for whom we build this
├── systemPatterns.md      # Architecture & design decisions
├── techContext.md         # Stack, env setup, constraints
├── activeContext.md       # Current focus & next steps  ← read first
├── progress.md            # What works & what’s left
└── (additional folders)   # e.g. research/, integrations/, playbooks/
```

### 2.1 Core Files (must always exist)  

| Order | File | Purpose (summary) |
|-------|------|-------------------|
| 1 | **projectbrief.md** | Defines scope, success criteria, stakeholders. |
| 2 | **productContext.md** | Problem statement, UX goals, value prop. |
| 3 | **systemPatterns.md** | High-level architecture, design patterns, data flows. |
| 4 | **techContext.md** | Technologies, dependencies, dev & deploy guidelines. |
| 5 | **activeContext.md** | *Living* log: current work, recent changes, next actions. |
| 6 | **progress.md** | Chronological status, done vs TODO, issues, metrics. |

### 2.2 Optional Context  
Create extra `.md` files / subfolders when detail is too large for core files:  
* Feature specs, integration notes  
* Red-team scenarios, data model diagrams  
* API reference, testing strategies  

---

## 3 · Usage Workflow  

### 3.1 Daily Routine  

1. **Read Everything** – On start-up ALWAYS skim *all* core files (speed-read mode).  
2. **Act** – Execute the assigned task (code, docs, infra).  
3. **Update** – Immediately reflect significant findings or changes in *activeContext.md*.  

> 🔑 *Rule of Thumb*: If new knowledge would hurt to lose, write it down.

### 3.2 Plan Mode (bigger features)  

1. Review full Memory Bank.  
2. Draft a work plan / design in chat.  
3. Commit that plan to `activeContext.md` (and other files as needed).  
4. Execute tasks, updating progress as milestones close.

### 3.3 Act Mode (regular commits)  

```
Start -> Check activeContext.md
      -> Do the work
      -> Document deltas (activeContext / progress)
      -> Commit / push
```

---

## 4 · Update Triggers  

Update the Memory Bank when **any** of the following occur:

| # | Trigger | Typical File(s) |
|---|---------|-----------------|
| 1 | New requirement / scope change | projectbrief.md |
| 2 | Architectural decision | systemPatterns.md |
| 3 | Tech stack tweak / new dep | techContext.md |
| 4 | Daily task completed / bug fixed | activeContext.md / progress.md |
| 5 | User explicitly says “**update memory bank**” | **All** core files must be reviewed |

*Never* postpone updates – stale docs defeat the purpose.

---

## 5 · Best Practices  

* Write in **plain, concise markdown** – future-you will thank you.  
* Use tables & lists for quick scanning.  
* Keep *activeContext.md* short (≤ ~150 lines) – archive old sections to progress.md.  
* Cross-link files where helpful (`See systemPatterns §4`).  
* Prefer deterministic wording (“must”, “exact version”), avoid ambiguity.  
* Diagrams: ASCII or mermaid code blocks – source lives here, rendered elsewhere.  

---

## 6 · Remember  

> **After every reset I have no memory.  
> These files are my brain. Keep them clear, current and complete.**  
