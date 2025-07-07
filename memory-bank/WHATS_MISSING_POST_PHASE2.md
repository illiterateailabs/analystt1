# What’s Still Missing After Phase 2  
**File:** `memory-bank/WHATS_MISSING_POST_PHASE2.md`  
**Context:** Phase 2 (“Feature Wave 2”) is fully shipped and tagged **v2.0.0-beta**.  
**Purpose:** Serve as the master backlog for **Phase 6 “Ops & Scaling”** and subsequent hardening cycles.

---

## 1 · Infrastructure Gaps
| Gap | Impact | Suggested Solution | Effort* |
|-----|--------|--------------------|---------|
| Production orchestration | Manual `docker-compose` not suitable for multi-tenant prod | Harden Helm chart, add Kustomize overlays, Terraform baseline | L |
| Horizontal scaling | No HPA rules / auto-scaling | K8s HPA (CPU, queue depth), Redis cluster mode | M |
| Global cache/queue | Single-node Redis limits HA | Redis Sentinel or managed elasticache; optional Kafka for streams | M |
| GPU inference | GAT model requires GPU nodes | Add GPU node pool + CI runner matrix | S |

## 2 · Feature Completeness Gaps
| Gap | Impact | Suggested Solution | Effort |
|-----|--------|--------------------|--------|
| Real ML models | Mock ensemble only → no true risk accuracy | Collect labelled dataset, train XGB/LGBM/CB ensemble, plug into registry | L |
| Retraining pipeline | Models risk drift | Celery scheduled job + MLflow experiment tracking | M |
| Tenant admin UI | No self-service onboarding/billing | Next.js pages + Stripe billing hooks | M |
| Advanced UI graphs | Front-end lacks timeline replay & graph editing | Add vis-timeline & editing tools | S |

## 3 · Operational Readiness Gaps
| Gap | Impact | Suggested Solution | Effort |
|-----|--------|--------------------|--------|
| Monitoring dashboards | No Grafana boards for cost, lag, model drift | Provision dashboards via `scripts/generate_grafana_dashboard.py` | S |
| Alerting & SLOs | No paging on critical metrics | Define SLOs (p95 < 1 s, WS lag < 5 s); Prometheus Alertmanager rules | M |
| Runbooks | New on-call engineers blind | Create runbooks per service & fire-drill docs | S |

## 4 · Security & Compliance Gaps
| Gap | Impact | Suggested Solution | Effort |
|-----|--------|--------------------|--------|
| Secure cookies & refresh rotation | JWT in localStorage vulnerable | Migrate to httpOnly cookies, add refresh token rotation | S |
| Rate limiting | Abuse may DDOS providers | Integrate SlowAPI + per-tenant quotas | S |
| Data encryption at rest | Reg compliance | Enable pgcrypto / use managed encrypted storage | M |
| GDPR / audit exports | Client requirement | Add “export all data for tenant” pipeline | M |

## 5 · Documentation Gaps
| Gap | Impact | Remedy | Effort |
|-----|--------|--------|--------|
| Helm deploy guide | Blocks ops | `DEPLOY_HELM.md` walk-through | S |
| Streaming ADR | Decision not captured | Write ADR-14 “Redis Streams vs Kafka” | XS |
| Tenant cookbook | Customer onboarding unclear | `ADD_NEW_TENANT_COOKBOOK.md` | XS |

## 6 · Testing & Quality Gaps
| Gap | Impact | Remedy | Effort |
|-----|--------|--------|--------|
| Load tests (1 M rows) | Unknown perf ceilings | Finish `scripts/end_to_end_load_test.py`, wire to CI nightly | M |
| GPU CI path | GAT un-tested in PRs | Conditional GPU workflow with self-hosted runner | M |
| E2E WebSocket tests | Flaky UX bugs | Cypress or Playwright real-browser tests | S |

## 7 · Performance & Optimisation Opportunities
| Area | Opportunity | Effort |
|------|-------------|--------|
| Redis vector search | Move to Hybrid (text + vector) for recall | S |
| Neo4j tuning | Add composite indexes, page-cache sizing | S |
| Celery autoscale | Dynamic worker pool via `celery autoscale` | XS |

## 8 · User Experience Improvements
| Gap | Remedy | Effort |
|-----|--------|--------|
| Live graph replay | Add time-scrubber & animation | S |
| Alert noise | Allow per-analyst alert rules & mute | XS |
| Dark-mode & accessibility | Tailwind theme extension | XS |

## 9 · Priority Ranking (0 = highest)
| Rank | Theme | Items |
|------|-------|-------|
| 0 | **Production deployment & scaling** | Helm chart, HPA, Redis HA |
| 1 | **Real ML models & retraining** | Dataset, training, registry CI |
| 2 | **Monitoring & SLOs** | Grafana, Alertmanager, runbooks |
| 3 | **Security hardening** | Secure cookies, rate-limit |
| 4 | **Tenant experience** | Admin UI, billing hooks |
| 5 | **Performance testing** | Load + GPU CI |
| 6 | **Docs & ADRs** | Deploy guide, streaming ADR |
| 7 | **UX polish** | Graph replay, dark-mode |

## 10 · Effort Estimates Legend
* **XS** ≤ 0.5 day  
* **S** ≤ 2 days  
* **M** ≤ 1 week  
* **L** > 1 week  

---

### 📅 Suggested Phase 6 Sprint Schedule (4 weeks)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Infra hardening | Helm v1, HPA, Redis HA |
| 2 | ML rollout | Ensemble model v1, retraining job |
| 3 | Monitoring & SLOs | Grafana import, Alert rules, runbooks |
| 4 | Security & tenant UX | Secure cookies, rate-limit, tenant admin MVP |

> After Phase 6, plan Phase 7 **“Performance & Feature Polish”** to cover dark-mode, replay UI, and remaining doc debt.

