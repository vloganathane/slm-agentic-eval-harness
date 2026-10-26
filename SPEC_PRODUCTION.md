# SLM Agentic Evaluation Harness — Production Specification

**Version**: 2.0 (Production)  
**Classification**: Internal — Proprietary  
**Target**: Senior engineering team  
**Scope**: Production-ready, multi-tenant, auditable, compliant

---

## 1. Executive Summary

This document specifies a **production-grade evaluation platform** for SLM fine-tuning that replaces manual eval loops with an autonomous, auditable, multi-agent system. Unlike the prototype spec, this addresses:

- **Multi-tenancy** — Isolated workspaces, RBAC, data residency
- **Reliability** — Exactly-once semantics, checkpointing, disaster recovery
- **Observability** — Structured logging, distributed tracing, SLOs
- **Security** — Encryption at rest/transit, secrets management, audit logs
- **Compliance** — GDPR, SOC2, data lineage, model cards
- **Scalability** — Horizontal scaling, queue-based async, cost control
- **Operability** — Runbooks, alerting, capacity planning, rollback

---

## 2. System Architecture

### 2.1 High-Level Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONTROL PLANE (Managed)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  API Gateway │  │  AuthZ/AuthN │  │  Tenant Mgr  │  │  Billing/    │   │
│  │  (REST+gRPC) │  │  (OIDC/OPA)  │  │  (Workspaces)│  │  Quotas      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │           │
│         └─────────────────┼─────────────────┼─────────────────┘           │
│                           ▼                 ▼                             │
│                  ┌─────────────────────────────────────┐                 │
│                  │         ORCHESTRATION LAYER          │                 │
│                  │  • Workflow Engine (Temporal)       │                 │
│                  │  • State Store (PostgreSQL + Redis) │                 │
│                  │  • Event Bus (Kafka/Redpanda)       │                 │
│                  └─────────────────┬───────────────────┘                 │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  DATA PLANE   │          │  COMPUTE PLANE│          │  AGENT PLANE  │
│  (Per Tenant) │          │  (Fleet)      │          │  (Workers)    │
│               │          │               │          │               │
│ • Object Store│          │ • Local GPU   │          │ • Curator     │
│   (S3/GCS)    │          │   Workers     │          │ • Judge       │
│ • Vector DB   │          │ • Modal Fleet │          │ • Analyst     │
│   (Qdrant)    │          │ • RunPod Pool │          │ • Synthetic   │
│ • Metadata DB │          │ • K8s Batch   │          │ • Reporter    │
│   (Postgres)  │          │ • Spot Fallback│         │ • Custom*     │
└───────────────┘          └───────────────┘          └───────────────┘
```

### 2.2 Component Responsibilities

| Layer | Component | Responsibility | Scaling |
|-------|-----------|----------------|---------|
| **Control** | API Gateway | Rate limit, auth, routing, request validation | Horizontal (stateless) |
| **Control** | AuthZ/AuthN | OIDC + OPA policies, workspace isolation | Horizontal |
| **Control** | Tenant Manager | Workspace CRUD, quotas, data residency config | Single-leader |
| **Orchestration** | Temporal Cluster | Workflow execution, retries, visibility | 3+ nodes |
| **Orchestration** | State Store | PostgreSQL (metadata) + Redis (cache/locks) | Read replicas |
| **Orchestration** | Event Bus | Audit events, async triggers, replay | Partitioned |
| **Data** | Object Store | Model artifacts, holdout images, reports | Multi-AZ |
| **Data** | Vector DB | Embedding search for similar cases | Clustered |
| **Data** | Metadata DB | Case metadata, strata, lineage | Primary + replicas |
| **Compute** | GPU Fleet Manager | Pool management, autoscaling, spot handling | Per-pool |
| **Agents** | Worker Pool | Horizontal scaling per agent type | K8s HPA |

---

## 3. Multi-Tenancy Model

### 3.1 Workspace Isolation

```
Tenant (Organization)
├── Workspace A (Project: invoice-extraction)
│   ├── Dataset: gst-invoices-v3
│   ├── Model: qwen2.5-1.5b-ft-042
│   ├── Holdout: ws-a-holdout (encrypted)
│   ├── Runs: round-1, round-2, ...
│   └── Members: alice@, bob@ (roles: admin, engineer, viewer)
├── Workspace B (Project: code-generation)
│   └── ...
└── Shared Resources
    ├── Base Models (read-only)
    ├── Public Datasets (read-only)
    └── Calibration Templates
```

### 3.2 RBAC Matrix

| Role | Workspaces | Datasets | Models | Runs | Config | Billing | Audit |
|------|------------|----------|--------|------|--------|---------|-------|
| **Org Admin** | CRUD | CRUD | CRUD | Read | CRUD | Read | Read |
| **Workspace Admin** | Read (own) | CRUD | CRUD | CRUD | CRUD | — | Read (own) |
| **Engineer** | Read (own) | Read/Write | Read/Write | CRUD | Read | — | Read (own) |
| **Viewer** | Read (own) | Read | Read | Read | Read | — | Read (own) |
| **CI/CD Bot** | — | Read | Read | Trigger | Read | — | — |

### 3.3 Data Residency

```yaml
# Workspace config
data_residency:
  region: "eu-central-1"          # Stored data never leaves this region
  backup_region: "eu-west-1"      # DR only, encrypted
  encryption:
    at_rest: "AES-256"            # SSE-KMS with tenant key
    in_transit: "TLS 1.3"
    key_management: "customer_managed"  # or "platform_managed"
```

**Egress rules (resolves the §3.3 vs §14.1 conflict):**

1. **Stored data** — datasets, holdouts, artifacts, state, backups — never leaves `region`.
2. **GPU compute** is region-pinned: Modal/RunPod pools declare a `region`; a pool whose region ≠ workspace residency is rejected at workspace save time (local/gpu pools are in-region only).
3. **External LLM calls** must use an in-region endpoint: EU data-residency tier for OpenAI, or Azure OpenAI (`westeurope` / `germanywestcentral`). Every call records `endpoint_region`; a mismatch is a residency violation → quarantine (§11.2).
4. **Redaction gate**: payloads pass through `PIIRedactorAgent` (§10.2) before leaving the cluster. PII is never sent to a third-party LLM unless `compliance.allow_external_pii` is explicitly enabled (Org Admin only, audited; §6.4).
5. The active sub-processor list (LLM + GPU vendors, with DPAs) is served by `GET /workspaces/{id}/subprocessors`.

---

## 4. Reliability & Fault Tolerance

### 4.1 Exactly-Once Semantics

| Operation | Mechanism |
|-----------|-----------|
| **Holdout preparation** | Idempotent: `holdout_hash` = SHA256(case_ids + config_hash) |
| **Inference** | Deduplication key: `model_hash + case_id + prompt_hash` |
| **Judge scoring** | Deterministic prompt + temperature=0 + cached results |
| **Synthetic generation** | Idempotent: `stratum_hash + error_profile_hash + seed` |
| **State advance** | Optimistic locking on `EvalState.version` |

### 4.2 Checkpointing & Recovery

```python
# Every agent implements Checkpointable
@dataclass
class AgentCheckpoint:
    agent_name: str
    round: int
    step: str                    # e.g., "curator.stratify", "judge.scoring"
    input_hash: str
    output_ref: ObjectRef        # S3 path to intermediate output
    started_at: datetime
    completed_at: Optional[datetime]
    error: Optional[str]

# Orchestrator recovers by:
# 1. Load latest EvalState
# 2. Find incomplete checkpoints for current round
# 3. Resume from last completed step (inputs immutable)
# 4. Replay forward
```

### 4.3 Failure Domains & SLAs

| Component | Availability Target | RTO | RPO | Failure Handling |
|-----------|---------------------|-----|-----|------------------|
| API Gateway | 99.99% | 30s | 0 | Multi-AZ, health checks |
| Temporal | 99.95% | 2min | 0 | 3-node cluster, persistence |
| PostgreSQL | 99.95% | 5min | <1s | Primary + sync replica |
| Redis | 99.9% | 1min | <1s | Cluster mode |
| GPU Workers | 99% | 10min | 0 | Preemptible, re-queue |
| Modal/RunPod | 99.5% | 5min | 0 | Retry with backoff |

> **RTO/RPO are verified, not assumed.** Quarterly DR drills restore PostgreSQL + object storage from backup and must meet the targets above; an untested or missed target is surfaced as *at risk* on the executive dashboard.

---

## 5. Observability

### 5.1 Structured Logging (JSON, OpenTelemetry)

```json
{
  "timestamp": "2026-10-25T14:30:00.123Z",
  "level": "INFO",
  "service": "eval-orchestrator",
  "trace_id": "abc123",
  "span_id": "def456",
  "workspace_id": "ws-invoice-001",
  "run_id": "run-042",
  "round": 3,
  "agent": "judge",
  "event": "scoring_complete",
  "metrics": {
    "cases_scored": 187,
    "subjective_fields": 3,
    "exact_match_fields": 7,
    "llm_calls": 561,
    "latency_p50_ms": 45,
    "latency_p99_ms": 230
  },
  "cost_usd": 0.023
}
```

### 5.2 Distributed Tracing

```
Trace: eval-round-3 (ws-invoice-001)
├── Span: orchestrator.run_round (14:30:00.100 - 14:35:22.400)
│   ├── Span: curator.prepare_holdout (14:30:00.100 - 14:30:45.200)
│   │   ├── Span: minhash_dedupe (14:30:00.100 - 14:30:12.000)
│   │   ├── Span: stratify (14:30:12.000 - 14:30:18.000)
│   │   └── Span: human_review_queue (14:30:18.000 - 14:30:45.200)
│   ├── Span: inference.run (14:30:45.200 - 14:32:10.500)
│   │   └── Span: modal.remote_call (14:30:45.200 - 14:32:10.500)
│   ├── Span: judge.score (14:32:10.500 - 14:33:55.000)
│   │   └── Span: llm.batch_judge (14:32:10.500 - 14:33:55.000)
│   ├── Span: analyst.categorize (14:33:55.000 - 14:34:30.000)
│   ├── Span: synthetic.generate (14:34:30.000 - 14:35:05.000)
│   └── Span: reporter.render (14:35:05.000 - 14:35:22.400)
```

### 5.3 Key Metrics & SLOs

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Round latency (p99)** | < 10 min | > 15 min |
| **Inference success rate** | > 99.9% | < 99.5% |
| **Judge calibration κ** | > 0.85 | < 0.80 |
| **Synthetic case validity** | > 95% | < 90% |
| **Gate decision accuracy** *(vs. human-labeled shadow set, §12.3)* | > 98% | < 95% |
| **Cost per round** | < $5 | > $10 |
| **Workspace isolation violations** | 0 | > 0 |

### 5.4 Dashboards (Grafana)

- **Executive**: Cost trends, model promotion rate, workspace activity
- **Engineering**: Agent latency, error rates, queue depths, GPU utilization
- **Science**: F1 trends by stratum, error category drift, calibration stability
- **Security**: Access patterns, data egress, failed auth, quota exhaustion

---

## 6. Security

### 6.1 Threat Model (STRIDE)

| Threat | Mitigation |
|--------|------------|
| **Spoofing** | mTLS between all services, OIDC for API, SPIFFE identities |
| **Tampering** | Immutable inputs (content-addressed), signed artifacts, WAL |
| **Repudiation** | Append-only audit log (Kafka), signed checkpoints |
| **Info Disclosure** | Encryption at rest (tenant KMS), field-level encryption for PII |
| **DoS** | Rate limiting per workspace, priority queues, circuit breakers |
| **Elevation** | Least-privilege service accounts, OPA policies, no root containers |

### 6.2 Secrets Management

```yaml
# No secrets in config/files
secrets:
  llm_api_keys:
    backend: "hashicorp_vault"
    path: "secret/eval/llm/{provider}"
    rotation: "30d"
  cloud_credentials:
    modal: "vault:secret/eval/modal"
    runpod: "vault:secret/eval/runpod"
    aws: "vault:secret/eval/aws"
  database:
    postgres: "vault:database/creds/eval-ro"
    redis: "vault:secret/eval/redis"
  ci_cd_bots:
    backend: "hashicorp_vault"
    path: "secret/eval/ci/{workspace_id}"
    token_ttl: "24h"               # Short-lived, scoped to Trigger + Read only
    rotation: "on_revocation"
```

### 6.3 Audit Logging

```json
{
  "audit_id": "audit-uuid",
  "timestamp": "2026-10-25T14:30:00Z",
  "actor": {
    "type": "user|service|ci",
    "id": "user:alice@company.com",
    "workspace": "ws-invoice-001",
    "roles": ["engineer"]
  },
  "action": "eval.run_round",
  "resource": {
    "type": "model",
    "id": "qwen2.5-1.5b-ft-042",
    "workspace": "ws-invoice-001"
  },
  "decision": "allow",
  "request": { "model": "...", "round": 3 },
  "result": { "promoted": true, "f1": 0.87 }
}
```

### 6.4 External LLM Calls & PII

The judge/analyst/synthetic agents call third-party APIs — the largest info-disclosure surface in the product:

1. **Redact before egress.** Payloads pass through `PIIRedactorAgent` (§10.2); fields in `pii_fields` are replaced before the request leaves the cluster.
2. **Region match is a hard gate.** `models.enforce_region_match` (§14.1) rejects any call whose `endpoint_region` ≠ workspace `data_residency` *before dispatch*; violations quarantine the workspace (§11.2).
3. **No external PII by default.** `compliance.allow_external_pii: false`; enabling it is an Org-Admin-only action recorded in `audit_event`.
4. **Sub-processors**: DPAs with every LLM/GPU provider live in `compliance/evidence/`; the active list is served by `GET /workspaces/{id}/subprocessors`.
5. **Prompt logging** stores `prompt_hash` + template version only — never raw completions containing workspace data, outside the encrypted artifact bucket.

### 6.5 Artifact Signing

- Every artifact (model, holdout, report) is content-addressed (`sha256`) and signed at write time (cosign / SigV4); reads verify before serving.
- `model_artifact.signature` (§15.1) is mandatory at registration — unsigned artifacts are rejected by the registry and by `_run_deployment`.
- Checkpoints are hash-chained like `audit_event`, so a tampered resume is detectable (§4.2).

---

## 7. Compliance & Governance

### 7.1 Data Lineage

```
Raw Dataset (v3.2) 
  → Dedupe (minhash v1.0) 
  → Stratify (keys: layout,lang,quality) 
  → Holdout v3.2-r3 (hash: abc123)
    → Inference (qwen2.5-1.5b-ft-042, modal:A100)
      → Predictions (hash: def456)
        → Judge (gpt-4o-mini, calibrated κ=0.89)
          → Scored (hash: ghi789)
            → Analyst (gpt-4o-mini, taxonomy v2.1)
              → Error Report (worst: layout=rotated_90)
                → Synthetic (gpt-4o-mini, template v1.3)
                  → Adversarial Cases (n=60, hash: jkl012)
                    → Holdout v3.2-r4 (hash: mno345)
```

### 7.2 Model Cards (Auto-Generated)

```markdown
# Model Card: qwen2.5-1.5b-invoice-ft-042

## Intended Use
Structured extraction from Indian GST invoices (EN/HI/TA).

## Training Data
- GST Receipts (Apache 2.0): 12,450 samples
- CORD-v2 (CC BY 4.0): 8,200 samples  
- EvalSafe Invoice (Apache 2.0): 3,100 samples
- Synthetic adversarial (rounds 1-4): 340 cases

## Evaluation (Round 4)
| Stratum | F1 Micro | Cases | Notes |
|---------|----------|-------|-------|
| Overall | 0.87 | 200 | Promoted |
| layout=rotated_90 | 0.72 | 18 | Blocker: needs more data |
| language=hi+ta | 0.81 | 22 | Warning: format_mismatch |
| quality=blurred | 0.85 | 25 | Pass |

## Limitations
- Fails on handwritten invoices (no training data)
- GSTIN validation not enforced in generation
- Multi-page continuing lines: 68% F1

## Ethical Considerations
- No PII in training data (vendor names hashed)
- Bias audit: consistent across vendor categories
```

### 7.3 Regulatory Compliance

| Requirement | Implementation |
|-------------|----------------|
| **GDPR Art. 17 (Right to Erasure)** | `DELETE /workspaces/{id}` cascades to all data; 30-day soft delete |
| **GDPR Art. 25 (Privacy by Design)** | Field-level encryption, minimization, purpose limitation |
| **SOC2 Type II** | Audit logs, access controls, encryption, incident response |
| **AI Act (EU) Annex IV** | Model cards, risk assessment, data governance, human oversight |
| **ISO 42001** | AI management system: policies, risk treatment, monitoring |
| **GDPR Art. 17 (backups)** | Purge job reconciles soft-deleted workspaces against snapshots/backups within 35 days; monthly restore-test verifies deletion, quarterly DR drill verifies restore (§4.3) |
| **Sub-processor register** | `GET /workspaces/{id}/subprocessors` (§3.3); DPA evidence in `compliance/evidence/` |

---

## 8. Scalability & Cost Control

### 8.1 Horizontal Scaling

```yaml
# K8s HPA for agent workers
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: judge-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: judge-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
    - type: External
      external:
        metric:
          name: kafka_consumer_group_lag
          selector:
            matchLabels:
              topic: judge.scoring.requests
        target:
          type: AverageValue
          averageValue: "10"  # 10 messages per replica
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
```

### 8.2 Cost Optimization

| Strategy | Implementation |
|----------|----------------|
| **Spot instances** | 70% of GPU fleet; graceful preemption (checkpoint → re-queue) |
| **Model caching** | Warm pool of loaded models (LRU, TTL 30min) |
| **Batch inference** | Dynamic batching (vLLM continuous batching) |
| **Judge caching** | Redis cache: `prompt_hash → score` (TTL 7d) |
| **Synthetic deduplication** | MinHash on generated cases before adding to holdout |
| **Quota enforcement** | Per-workspace daily $ limit; hard stop at 110% |

### 8.3 Capacity Planning

```
Baseline (per workspace, per eval round):
├── Holdout: 200 cases × 2MB = 400MB
├── Inference: 200 cases ÷ 8 concurrent × 30s ≈ 12.5 min
│   = 0.25 A100-h → $1.00 (at the single rate below)
├── Judge: 200 × 3 subjective × 2 calls = 1,200 LLM calls (~$0.02)
├── Analyst: 200 × 5 errors × 1 call = 1,000 LLM calls (~$0.015)
├── Synthetic: 3 strata × 20 = 60 cases generated
└── Total: ~$1.05/round

Single rate used everywhere: $4.00 / A100-hour (blended Modal+RunPod),
gpt-4o-mini list price. Do not quote per-round cost at a different rate —
the v1 draft mixed $1.47/hr (baseline) with $4.00/hr (projection).

Projected (10 workspaces, 5 eval rounds/day, rates above):
├── Eval GPU: 12.5 A100-h/day → $50/day (Modal pay-per-sec)
├── LLM: 110k calls/day → $3.85/day (judge+analyst+synthetic)
├── Training (pipeline runs): 42 min A100 × 50 runs/day
│   = 35 A100-h/day → $140/day
├── Storage: 50 GB/month → $1.15/month
└── Variable: ~$5,800/month

Fixed control-plane infra (excluded from the variable total above):
├── Temporal 3-node + Kafka + RDS multi-AZ + Redis cluster
├── Vault + OTel/Grafana + K8s (control plane)
└── ~$2,500–4,000/month → All-in ≈ $8,300–9,800/month
```

---

## 9. API Specification

### 9.1 Core Endpoints

```yaml
openapi: 3.1.0
info:
  title: SLM Evaluation API
  version: v2
servers:
  - url: https://api.eval.example.com/v2
paths:
  /workspaces:
    get:
      summary: List workspaces visible to the caller
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/WorkspaceList'
    post:
      summary: Create workspace
      security: [BearerAuth: []]
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WorkspaceCreate'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Workspace'

  /workspaces/{workspace_id}:
    get:
      summary: Get workspace
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Workspace'
    patch:
      summary: Update workspace (data_residency is immutable after creation)
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WorkspaceUpdate'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Workspace'
    delete:
      summary: Soft-delete workspace (GDPR Art. 17; purge after 30 days, §7.3)
      responses:
        '202':
          description: Soft delete initiated; cascade documented in §7.3

  /workspaces/{workspace_id}/members:
    get:
      summary: List members
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MemberList'
    post:
      summary: Add member or change role (Org/Workspace Admin)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/MemberCreate'
      responses:
        '201':
          description: Granted
        '403':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/members/{user_id}:
    delete:
      summary: Remove member (audited)
      responses:
        '204':
          description: Removed

  /workspaces/{workspace_id}/datasets:
    post:
      summary: Register dataset — presigned upload (multipart is never used for artifacts)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DatasetCreate'
      responses:
        '201':
          description: Upload session issued; client PUTs files to the presigned URLs, then finalizes
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatasetCreated'
        '409':
          $ref: '#/components/responses/Problem'
    get:
      summary: List datasets (paginated)
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatasetList'

  /workspaces/{workspace_id}/datasets/{dataset_id}:
    delete:
      summary: Delete dataset version (soft delete; cascade per §7.3)
      responses:
        '202':
          description: Accepted
        '409':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/runs:
    post:
      summary: Start evaluation run
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RunCreate'
      responses:
        '202':
          description: Accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Run'
        '409':
          $ref: '#/components/responses/Problem'
    get:
      summary: List runs (paginated)
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
        - $ref: '#/components/parameters/StatusFilter'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RunList'

  /workspaces/{workspace_id}/runs/{run_id}:
    get:
      summary: Get run status
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RunStatus'
    delete:
      summary: Cancel run (graceful: workers checkpoint, then stop — PO-02)
      responses:
        '202':
          description: Cancellation requested
        '404':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/runs/{run_id}/gate:
    get:
      summary: Get CI gate decision
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/GateDecision'

  /workspaces/{workspace_id}/runs/{run_id}/reviews:
    get:
      summary: Human-review queue for this run (§14.1 human_review)
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
        - name: state
          in: query
          schema:
            type: string
            enum: [pending, escalated, decided, timed_out]
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReviewQueue'

  /workspaces/{workspace_id}/runs/{run_id}/reviews/{case_id}:
    post:
      summary: Submit review decision (blocks gate until decided or 24h escalation)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ReviewDecision'
      responses:
        '201':
          description: Recorded
        '409':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/audit:
    get:
      summary: Query audit log (read-only; RBAC Audit column, §3.2)
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
        - name: since
          in: query
          schema:
            type: string
            format: date-time
        - name: actor
          in: query
          schema:
            type: string
        - name: action
          in: query
          schema:
            type: string
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuditList'

  /workspaces/{workspace_id}/subprocessors:
    get:
      summary: Active LLM/GPU sub-processors with DPA references (§3.3, §6.4)
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SubprocessorList'

  /workspaces/{workspace_id}/events:
    post:
      summary: Subscribe a signed webhook (run completed, gate decided, quota breach)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WebhookCreate'
      responses:
        '201':
          description: Subscribed; payloads signed with X-Eval-Signature (§9.3)

  /workspaces/{workspace_id}/ws-ticket:
    post:
      summary: Issue a one-time 60s WebSocket ticket (§9.3)
      responses:
        '201':
          description: Ticket issued

  /workspaces/{workspace_id}/models:
    post:
      summary: Register model artifact — presigned upload (multipart is never used for artifacts)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ModelRegister'
      responses:
        '201':
          description: Registration recorded; sha256 + signature verified server-side (§6.5)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelRegistered'
        '422':
          $ref: '#/components/responses/Problem'
    get:
      summary: List model artifacts (paginated)
      parameters:
        - $ref: '#/components/parameters/Page'
        - $ref: '#/components/parameters/PerPage'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelList'

  /workspaces/{workspace_id}/models/{model_id}/promote:
    post:
      summary: Promote model to current for this workspace (audited)
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      responses:
        '200':
          description: Promoted
        '409':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/models/{model_id}/rollback:
    post:
      summary: Roll back promotion to the previously promoted model (audited)
      responses:
        '200':
          description: Rolled back

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  parameters:
    IdempotencyKey:
      name: Idempotency-Key
      in: header
      required: false
      schema:
        type: string
      description: >
        Client-generated UUID. Replaying the same key within 24h returns the
        original response instead of creating a duplicate (§4.1, §15.1).
    Page:
      name: page
      in: query
      schema:
        type: integer
        minimum: 1
        default: 1
    PerPage:
      name: per_page
      in: query
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 50
    StatusFilter:
      name: status
      in: query
      schema:
        type: string
        enum: [queued, running, completed, failed, cancelled]

  responses:
    Problem:
      description: RFC 7807 problem details
      content:
        application/problem+json:
          schema:
            $ref: '#/components/schemas/Problem'

  schemas:
    Problem:
      type: object
      required: [type, title, status]
      description: Every non-2xx response uses this shape (§9.3)
      properties:
        type:
          type: string
          format: uri
        title:
          type: string
        status:
          type: integer
        detail:
          type: string
        trace_id:
          type: string
          description: Correlates to logs/traces (§5.1)

    PageMeta:
      type: object
      properties:
        page: {type: integer}
        per_page: {type: integer}
        total: {type: integer}

    WorkspaceCreate:
      type: object
      required: [name, data_residency]
      properties:
        name: {type: string}
        data_residency: {type: string, example: eu-central-1}
        encryption:
          type: object
          properties:
            key_management: {type: string, enum: [customer_managed, platform_managed]}

    WorkspaceUpdate:
      type: object
      properties:
        name: {type: string}
        data_residency:
          type: string
          description: Immutable after creation (§3.3)

    Workspace:
      type: object
      properties:
        id: {type: string}
        org_id: {type: string}
        name: {type: string}
        data_residency: {type: string}
        status: {type: string, enum: [active, soft_deleted, purged]}
        created_at: {type: string, format: date-time}

    WorkspaceList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/Workspace'}
        page: {$ref: '#/components/schemas/PageMeta'}

    MemberCreate:
      type: object
      required: [user_id, role]
      properties:
        user_id: {type: string}
        role:
          type: string
          enum: [org_admin, workspace_admin, engineer, viewer, ci_bot]

    Member:
      type: object
      properties:
        user_id: {type: string}
        role: {type: string}
        granted_by: {type: string}
        granted_at: {type: string, format: date-time}

    MemberList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/Member'}
        page: {$ref: '#/components/schemas/PageMeta'}

    DatasetCreate:
      type: object
      required: [name, manifest_uri, content_hash]
      description: >
        Client uploads files to a presigned session first (§9.3); this call
        registers the result. multipart is never accepted for artifacts.
      properties:
        name: {type: string}
        manifest_uri: {type: string}
        content_hash: {type: string}
        license: {type: string}

    DatasetCreated:
      type: object
      properties:
        id: {type: string}
        version: {type: integer}
        upload_urls:
          type: array
          items: {type: string, format: uri}
          description: Presigned PUT URLs, 15min TTL
        finalize_url: {type: string}

    Dataset:
      type: object
      properties:
        id: {type: string}
        name: {type: string}
        version: {type: integer}
        cases: {type: integer}
        content_hash: {type: string}
        license: {type: string}

    DatasetList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/Dataset'}
        page: {$ref: '#/components/schemas/PageMeta'}

    ModelRegister:
      type: object
      required: [kind, upload_id, sha256, signature, metadata]
      properties:
        kind: {type: string, enum: [base, lora, gguf]}
        upload_id: {type: string, description: Presigned session from the upload API}
        sha256: {type: string}
        signature:
          type: string
          description: Required — cosign/SigV4 signature (§6.5)
        metadata: {type: object}

    ModelRegistered:
      type: object
      properties:
        id: {type: string}
        sha256: {type: string}
        signature_verified: {type: boolean}

    Model:
      type: object
      properties:
        id: {type: string}
        kind: {type: string}
        sha256: {type: string}
        promoted_round: {type: integer, nullable: true}

    ModelList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/Model'}
        page: {$ref: '#/components/schemas/PageMeta'}

    RunCreate:
      type: object
      required: [model_id, dataset_id]
      properties:
        model_id: {type: string}
        dataset_id: {type: string}
        config_override: {type: object}

    Run:
      type: object
      properties:
        run_id: {type: string}
        status: {type: string, enum: [queued, running, completed, failed, cancelled]}
        poll_url: {type: string}
        websocket: {type: string}
        estimated_start: {type: string, format: date-time}

    RunList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/RunStatus'}
        page: {$ref: '#/components/schemas/PageMeta'}

    RunStatus:
      type: object
      properties:
        run_id: {type: string}
        status: {type: string}
        current_round: {type: integer}
        rounds:
          type: array
          items: {$ref: '#/components/schemas/RoundStatus'}
        cost_usd: {type: number}

    RoundStatus:
      type: object
      properties:
        round: {type: integer}
        status: {type: string, enum: [in_progress, completed]}
        f1: {type: number}
        ci_lower: {type: number, description: 95% CI lower bound (§12.3)}
        promoted: {type: boolean}
        started_at: {type: string, format: date-time}

    GateDecision:
      type: object
      properties:
        round: {type: integer}
        current_f1: {type: number}
        ci_lower:
          type: number
          description: 95% CI lower bound — the gate statistic (§12.3)
        target_f1: {type: number}
        promote: {type: boolean}
        blockers:
          type: array
          items: {type: string}
        checked_at: {type: string, format: date-time}

    ReviewQueue:
      type: object
      properties:
        items:
          type: array
          items:
            type: object
            properties:
              case_id: {type: string}
              state: {type: string, enum: [pending, escalated, decided, timed_out]}
              due_at: {type: string, format: date-time}
              assigned_to: {type: string, nullable: true}
              reason: {type: string}
        page: {$ref: '#/components/schemas/PageMeta'}

    ReviewDecision:
      type: object
      required: [decision]
      properties:
        decision: {type: string, enum: [accept, reject, amend]}
        notes: {type: string}
        amended_value: {type: string, nullable: true}

    AuditEvent:
      type: object
      properties:
        audit_id: {type: string}
        timestamp: {type: string, format: date-time}
        actor: {type: object}
        action: {type: string}
        resource: {type: object}
        decision: {type: string, enum: [allow, deny]}
        result: {type: object}

    AuditList:
      type: object
      properties:
        items:
          type: array
          items: {$ref: '#/components/schemas/AuditEvent'}
        page: {$ref: '#/components/schemas/PageMeta'}

    SubprocessorList:
      type: object
      properties:
        items:
          type: array
          items:
            type: object
            properties:
              name: {type: string}
              purpose: {type: string, enum: [llm, gpu, storage, observability]}
              region: {type: string}
              dpa_ref: {type: string}

    WebhookCreate:
      type: object
      required: [url, events]
      properties:
        url: {type: string, format: uri}
        events:
          type: array
          items:
            type: string
            enum: [run.completed, run.failed, gate.decided, quota.breach, review.escalated]
        secret_ref:
          type: string
          description: Vault path for the HMAC signing secret (§9.3)
```

### 9.2 Async Operations (Long-Running)

```python
# Run creation returns immediately
# Header: Idempotency-Key: <uuid>  (§9.3 — replay returns the original run)
POST /workspaces/{ws}/runs
{
  "model_id": "qwen2.5-1.5b-ft-042",
  "dataset_id": "gst-invoices-v3",
  "config_override": { "target_f1": 0.88 }
}

# Response: 202 Accepted
{
  "run_id": "run-042",
  "status": "queued",
  "estimated_start": "2026-10-25T14:30:00Z",
  "poll_url": "/workspaces/ws-001/runs/run-042",
  "websocket": "wss://api.eval.example.com/v2/ws/runs/run-042"
}

# Poll or WebSocket for updates
GET /workspaces/{ws}/runs/run-042
{
  "run_id": "run-042",
  "status": "running",
  "current_round": 3,
  "rounds": [
    {"round": 1, "status": "completed", "f1": 0.72, "promoted": false},
    {"round": 2, "status": "completed", "f1": 0.81, "promoted": false},
    {"round": 3, "status": "in_progress", "started_at": "..."}
  ],
  "cost_usd": 4.23
}

# WebSocket: authenticate with a one-time ticket, never a long-lived JWT
# POST /workspaces/{ws}/ws-ticket → {"ticket": "...", "expires_in": 60}
# wss://.../v2/ws/runs/run-042?ticket=...    (reconnect with Last-Event-ID)
```

### 9.3 API Conventions

| Concern | Rule |
|---------|------|
| **Errors** | Every non-2xx returns RFC 7807 `application/problem+json` (`Problem`) with `trace_id` |
| **Idempotency** | All POSTs accept `Idempotency-Key`; replays within 24h return the original response (§4.1); enforced by `run.idempotency_key UNIQUE` (§15.1) |
| **Pagination** | All list endpoints take `page` / `per_page ≤ 100` and return `PageMeta` |
| **Uploads** | Never multipart for artifacts: presigned S3 URLs (GB-scale GGUFs/datasets); the API records only the resulting `content_hash` |
| **Versioning** | `/v2` in path; breaking changes ship as `/v3`; deprecation announced with a `Sunset` header, minimum 90 days (§13) |
| **Rate limits** | Per-workspace token bucket: 60 req/s API, 10 concurrent uploads; `429` + `Retry-After` |
| **WebSocket** | One-time ticket from `POST .../ws-ticket` (60s TTL, §9.2); reconnect with `Last-Event-ID`; messages carry a monotonic `seq` |
| **Webhooks** | `POST /events`; payloads signed `X-Eval-Signature: HMAC-SHA256(body, secret)`; retries with backoff, 24h expiry |
| **Bot auth** | CI/CD bots use short-lived scoped tokens (§6.2), never user JWTs |
| **Human review** | `requires_human_review` cases appear in `GET .../reviews`; the gate cannot pass until each is decided or the 24h escalation fires (§14.1, PF-09) |

---

## 10. Agent Extensibility (Production)

### 10.1 Custom Agent SDK

```python
# slm_eval/sdk.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel

class AgentContext(BaseModel):
    workspace_id: str
    run_id: str
    round: int
    config: EvalConfig
    secrets: SecretManager
    storage: ObjectStore
    metrics: MetricsClient

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class BaseAgent(ABC, Generic[InputT, OutputT]):
    name: str
    version: str
    requires_human_review: bool = False
    
    @abstractmethod
    async def execute(self, ctx: AgentContext, input: InputT) -> OutputT:
        pass
    
    @abstractmethod
    def validate_input(self, input: InputT) -> ValidationResult:
        pass
    
    @abstractmethod
    def validate_output(self, output: OutputT) -> ValidationResult:
        pass
    
    async def checkpoint(self, ctx: AgentContext, step: str, data: Any) -> None:
        # Auto-checkpointing
        pass
```

### 10.2 Built-in Agent Registry

```yaml
# agents/registry.yaml
agents:
  - name: curator
    class: slm_eval.agents.curator.DataCuratorAgent
    version: "2.1.0"
    input: Dataset
    output: HoldoutSet
    resources:
      cpu: "2"
      memory: "8Gi"
      gpu: "0"
    timeout: 300
    retry:
      max_attempts: 3
      backoff: exponential
  
  - name: judge
    class: slm_eval.agents.judge.JudgeAgent
    version: "2.1.0"
    input: ScoringRequest
    output: ScoredPredictions
    resources:
      cpu: "4"
      memory: "16Gi"
      gpu: "0"  # Uses remote LLM
    timeout: 600
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 60

  - name: custom-pii-redactor
    class: my_company.agents.PIIRedactorAgent
    version: "1.0.0"
    input: HoldoutSet
    output: HoldoutSet
    config:
      pii_fields: ["vendor_name", "address", "phone"]
      replacement: "[REDACTED]"
```

---

## 11. Deployment & Operations

### 11.1 Infrastructure (Terraform)

```hcl
# infra/main.tf
module "control_plane" {
  source = "./modules/control-plane"
  
  cluster_name = "eval-control"
  region       = var.primary_region
  
  temporal = {
    persistence = "postgresql"
    replicas    = 3
  }
  
  postgres = {
    instance_class = "db.r6g.xlarge"
    multi_az       = true
    backup_retention = 30
  }
  
  redis = {
    cluster_mode = true
    node_type    = "cache.r6g.large"
    replicas     = 2
  }
}

module "data_plane" {
  source = "./modules/data-plane"
  for_each = var.tenants
  
  tenant_id     = each.key
  region        = each.value.region
  kms_key_id    = each.value.kms_key
  
  s3 = {
    bucket_name = "eval-${each.key}-${var.env}"
    versioning  = true
    lifecycle_rules = [
      { prefix = "holdouts/", expiration_days = 365 },
      { prefix = "synthetic/", expiration_days = 90 },
      { prefix = "reports/", expiration_days = 30 }
    ]
  }
  
  qdrant = {
    enabled = each.value.vector_search
    shards  = 3
  }
}

module "compute_fleet" {
  source = "./modules/compute-fleet"
  
  pools = {
    "local-gpu" = {
      instance_types = ["g5.2xlarge", "g5.4xlarge"]
      min_size = 2
      max_size = 20
      spot_percentage = 70
    }
    "modal" = {
      type = "modal"
      max_concurrent = 50
      gpu_types = ["A100", "A10G"]
    }
    "runpod" = {
      type = "runpod"
      max_concurrent = 30
      gpu_types = ["A100", "H100"]
    }
  }
}
```

### 11.2 Runbooks

| Incident | Detection | Response |
|----------|-----------|----------|
| **Round latency > 15min** | PagerDuty (SLO breach) | 1. Check queue depth 2. Scale workers 3. Check Modal/RunPod capacity |
| **Judge κ < 0.80** | Grafana alert | 1. Pause auto-promotion 2. Review calibration drift 3. Re-calibrate |
| **Workspace quota exceeded** | API 429 + alert | 1. Notify workspace admin 2. Auto-pause runs at 110% |
| **GPU worker OOM** | Container crash + metric | 1. Reduce batch size 2. Switch to llamacpp 3. Alert capacity team |
| **Data residency violation** | Nightly scan | 1. Immediate quarantine 2. Forensic log review 3. Customer notification |
| **Erasure request received** | API `DELETE /workspaces/{id}` | 1. Soft delete now 2. Purge jobs at T+30d 3. Backup reconciliation ≤35d 4. Certificate of deletion issued |
| **Restore drill fails** | Quarterly DR report | 1. Treat RTO/RPO as unmet 2. File incident 3. Fix before next drill |

### 11.3 Release Process

```yaml
# .github/workflows/release.yml (GitHub Actions uses jobs:, not stages:)
jobs:
  lint:        # ruff, mypy, hadolint
  unit:        # pytest tests/unit -x --cov=80%
  contract:    # pact (providers: our api + orchestrator — not third-party libs)
               # + schemathesis against openapi.yaml
  integration: # pytest tests/integration with testcontainers
               # (Postgres, Redis, Temporal, Redpanda, MinIO)
  e2e:         # golden-path pipeline on staging
  chaos:       # litmuschaos (pod kill, network partition, clock skew) — nightly
  load:        # k6 at 10x peak — pre-release
  security:    # CodeQL, Trivy, OWASP ZAP — every PR
  canary:      # 5% traffic, 30min observation
               # needs: [unit, contract, integration, security]
  promote:     # manual approval → 100%
  rollback:    # automatic on SLO breach (error rate > 1% or latency p99 > 2x baseline)
```

---

## 12. Testing Strategy

### 12.1 Test Pyramid

| Layer | Coverage | Tools | Target |
|-------|----------|-------|--------|
| **Unit** | 90%+ | pytest, hypothesis | < 5 min |
| **Contract** | 100% APIs | Pact, schemathesis | < 10 min |
| **Integration** | Critical paths | testcontainers, localstack | < 30 min |
| **E2E** | Golden paths | Playwright, temporal CLI | < 1 hr |
| **Chaos** | Failure modes | LitmusChaos, Chaos Mesh | Weekly |
| **Load** | 10x peak | k6, Locust | Pre-release |
| **Security** | SAST/DAST/SCA | CodeQL, Trivy, OWASP ZAP | Every PR |

### 12.2 Golden Datasets (Regression)

```python
# tests/golden/datasets.py
GOLDEN_DATASETS = {
    "invoice-extraction-v1": {
        "source": "gst-cord-evalsafe",
        "version": "2026-01-15",
        "cases": 500,
        "expected_f1": {
            "qwen2.5-1.5b-base": 0.62,
            "qwen2.5-1.5b-ft-042": 0.87,
        },
        "strata_distribution": {
            "layout=standard": 0.45,
            "layout=rotated_90": 0.10,
            "language=hi": 0.20,
            "quality=blurred": 0.15,
        }
    }
}

# tests/golden/test_regression.py
def test_golden_dataset_regression(golden_dataset, model_under_test):
    orchestrator = EvalOrchestrator.load_config("config/test.yaml")
    result = orchestrator.run_round(model_under_test, golden_dataset.cases)
    
    expected = golden_dataset.expected_f1[model_under_test.id]
    # Gate on the 95% CI lower bound, not the point estimate (§12.3):
    # a fixed ±2% window is narrower than GPU inference noise at n=500.
    assert result.decision.ci_lower_f1 >= expected - 0.03
    assert result.decision.promote == (expected >= 0.85)
```

### 12.3 Statistical Rigor

| Rule | Definition |
|------|------------|
| **Gate statistic** | Promotion requires the **95% CI lower bound** of micro-F1 ≥ `target_f1`, never the point estimate. `GateDecision` therefore carries `ci_lower` (§9.1), and §16.3 gates on it |
| **Sample size** | `min_per_stratum` cases per stratum; a stratum below it is reported `insufficient_n` and cannot produce a blocker on its own (see `strata_mode`, §14.1) |
| **Gate-accuracy ground truth** | "Gate decision accuracy" (§5.3) is measured against a **human-labeled shadow set**: 200 cases/quarter where two reviewers independently decide promote/block; disagreements resolved by ML Lead. Reported monthly on the Science dashboard |
| **Regression tolerance** | Golden tests assert CI-based bounds (±3pp on the lower bound), never a fixed ±2% on the point estimate |
| **Judge κ** | Cohen's κ on ≥50 calibration pairs each round (§14.1); the κ confidence interval is reported alongside the point estimate |

---

## 13. Migration from Prototype

### 13.1 Compatibility Layer

```python
# slm_eval/compat/v1.py
class V1OrchestratorAdapter:
    """Wraps prototype EvalOrchestrator for zero-downtime migration."""
    
    def __init__(self, v1_orchestrator: "V1EvalOrchestrator"):
        self.v1 = v1_orchestrator
        self.v2_client = V2APIClient()
    
    def run_round(self, model_path: str, dataset: Dataset) -> EvalResult:
        # 1. Persist v1 state to v2 control plane
        workspace_id = self._get_or_create_workspace()
        dataset_id = self._upload_dataset(dataset)
        model_id = self._register_model(model_path)
        
        # 2. Start v2 run
        run = self.v2_client.start_run(workspace_id, model_id, dataset_id)
        
        # 3. Poll for completion (or webhook)
        result = self.v2_client.wait_for_run(run.run_id)
        
        # 4. Convert to v1 format for existing consumers
        return self._convert_result(result)
```

### 13.2 Migration Checklist

- [ ] Deploy control plane (staging)
- [ ] Migrate calibration data → Vault
- [ ] Re-run golden datasets on v2, compare results
- [ ] Shadow mode: run v1 + v2 in parallel, compare gates
- [ ] Cutover: update CI/CD to v2 API
- [ ] Deprecate v1 CLI after 90 days

---

## 14. Appendix: Configuration Reference

### 14.1 Complete `eval_config.yaml` (Production)

```yaml
# Required
task_description: "Structured field extraction from Indian GST invoices"
task_type: "extraction"
workspace_id: "ws-invoice-001"          # Injected at runtime
run_id: "run-042"                       # Injected at runtime

# State & Persistence
state_path: "s3://eval-ws-invoice-001/state/eval_state.json"
checkpoint_interval: 1                  # Every step

# Stratification
strata_keys: [layout_type, language, quality, vendor_category]
strata_mode: "single_key"          # single_key | crossed
# crossed on 4 keys can explode past min_per_stratum × max_holdout_total
# (4 keys × 50 min = 200 = the whole budget), so single_key is default.
strata_merge_rule: "below_min → merge into 'other_<key>', never silently dropped"
strata_definitions: "s3://eval-config/strata_definitions.yaml"

# Fields
subjective_fields: [vendor_name, address, description]
exact_match_fields: [invoice_number, date, total, tax_id, line_items]
field_validators:
  tax_id:
    type: "regex"
    pattern: "^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
  total:
    type: "numeric"
    tolerance: 0.01
  line_items:
    type: "list"
    item_validator: "line_item_schema"

# Thresholds
target_f1: 0.85
failure_threshold: 0.5
synthetic_per_stratum: 20
min_per_stratum: 50
max_holdout_total: 200
worst_strata_count: 3

# Human-in-the-loop
human_review:
  holdout_coverage: 1.0
  train_coverage: 0.1
  timeout_hours: 24
  escalation:
    after_hours: 4
    notify: ["alice@company.com", "bob@company.com"]

# Calibration
calibration:
  path: "s3://eval-ws-invoice-001/calibration/judge_calibration.jsonl"
  min_pairs: 50
  target_kappa: 0.85
  recalibrate_every_rounds: 5

# Models (OpenAI-compatible endpoints)
# Residency guard (§3.3, §6.4): every provider's endpoint_region must equal the
# workspace data_residency or the call is rejected before dispatch. For
# eu-central-1 use an EU data-residency tier, or Azure OpenAI (westeurope).
models:
  enforce_region_match: true        # hard-fail on mismatch (audit: §11.2)
  teacher:
    endpoint: "https://api.openai.com/v1"
    endpoint_region: "eu"
    model: "gpt-4o-mini"
    api_key_ref: "vault:secret/eval/llm/openai"
  judge:
    endpoint: "https://api.openai.com/v1"
    endpoint_region: "eu"
    model: "gpt-4o-mini"
    api_key_ref: "vault:secret/eval/llm/openai"
  analyst:
    endpoint: "https://api.openai.com/v1"
    endpoint_region: "eu"
    model: "gpt-4o-mini"
    api_key_ref: "vault:secret/eval/llm/openai"
  synthetic:
    endpoint: "https://api.openai.com/v1"
    endpoint_region: "eu"
    model: "gpt-4o-mini"
    api_key_ref: "vault:secret/eval/llm/openai"

# Inference
inference:
  backend: "auto"
  args: {}
  fallback_chain: ["modal", "runpod", "vllm", "llamacpp"]
  timeout_seconds: 1800
  retry:
    max_attempts: 3
    backoff_base: 2

# Synthetic generation
synthetic:
  templates: "s3://eval-config/adversarial_templates.yaml"
  augmentation:
    enabled: true
    profiles: "s3://eval-config/augmentation_profiles.yaml"
  validation:
    min_validity_score: 0.8
    human_review_fraction: 0.1

# Error taxonomy
error_taxonomy: "s3://eval-config/error_taxonomy.yaml"

# Reporting
reporting:
  formats: ["markdown", "html", "json", "pdf"]
  dashboard:
    enabled: true
    charts: ["strata_heatmap", "error_trends", "cost_breakdown"]
  notifications:
    on_gate_fail: ["slack:#eval-alerts", "pagerduty"]
    on_promote: ["slack:#model-releases"]
    daily_summary: ["email:team@company.com"]

# Cost control
cost_control:
  max_cost_per_round_usd: 10.0
  max_cost_per_day_usd: 100.0
  alert_at_percent: 80
  hard_stop_at_percent: 110

# Compliance
compliance:
  data_residency: "eu-central-1"
  allow_external_pii: false         # Org-Admin-only, audited enablement (§6.4)
  encryption:
    at_rest: "AES-256"
    kms_key_ref: "vault:secret/eval/kms/ws-invoice-001"
  retention:
    holdouts: "365d"
    synthetic: "90d"
    reports: "30d"
    checkpoints: "7d"
  audit_log: "kafka:audit-logs"
```

---

## 15. Data Model & Local Development

The metadata schema is the contract every other section depends on. All tables live in control-plane PostgreSQL, are workspace-scoped (`workspace_id` on every row), and are versioned via `db/migrations/*`.

### 15.1 Core Tables

| Table | Key columns | Notes |
|-------|-------------|-------|
| `organization` | `id`, `name`, `billing_ref` | Tenant root |
| `workspace` | `id`, `org_id`, `name`, `data_residency`, `kms_key_ref`, `status`, `deleted_at` | Soft delete: 30d (§7.3) |
| `member` | `workspace_id`, `user_id`, `role`, `granted_by`, `granted_at` | RBAC source of truth (§3.2) |
| `dataset` | `id`, `workspace_id`, `name`, `version`, `manifest_uri`, `content_hash`, `license` | Immutable versions |
| `dataset_split` | `dataset_id`, `kind` (train\|holdout), `case_ids_uri`, `content_hash`, `seed` | PF-07 leakage evidence |
| `case` | `id`, `dataset_split_id`, `strata JSONB`, `input_uri`, `pii_flag` | Strata indexed (GIN) |
| `model_artifact` | `id`, `workspace_id`, `kind` (base\|lora\|gguf), `uri`, `sha256`, `signature`, `metadata JSONB` | Signed (§6.5) |
| `run` | `id`, `workspace_id`, `model_id`, `dataset_id`, `config_uri`, `status`, `idempotency_key UNIQUE`, `created_by` | §9 |
| `round` | `id`, `run_id`, `round`, `status`, `holdout_hash`, `metrics JSONB`, `cost_usd` | |
| `score` | `round_id`, `case_id`, `field`, `value_pred`, `value_gold`, `judge_model`, `prompt_hash`, `latency_ms` | Dedup key (§4.1) |
| `checkpoint` | `agent`, `round_id`, `step`, `input_hash`, `output_ref`, `error` | §4.2 |
| `review_task` | `id`, `round_id`, `case_id`, `state`, `assigned_to`, `due_at`, `decided_by`, `decision` | §9.3 human review |
| `audit_event` | `id`, `ts`, `actor`, `action`, `resource`, `decision`, `payload JSONB`, `prev_hash` | Hash-chained, append-only |
| `cost_ledger` | `workspace_id`, `run_id`, `stage`, `usd`, `source`, `ts` | Feeds quotas (§8.2) |

### 15.2 Invariants

- Every query carries `workspace_id` (directly or by join) and is rejected at the repository layer without it — this is what makes §5.3's "isolation violations = 0" testable rather than aspirational.
- `score` uniqueness: `(round_id, case_id, field, prompt_hash)` — the physical implementation of the §4.1 dedup key.
- `audit_event` is append-only (dedicated role with no `UPDATE`/`DELETE` grant) and hash-chained (`prev_hash`) for tamper evidence.
- Enumerations are stored as `text` + `CHECK` constraints; wire/JSON format uses `.value` strings — see the `PipelineState` serialization note in §16.3, which exists because the naive `str(Enum)` round-trip broke resume.
- `run.idempotency_key` is `UNIQUE` per workspace: the physical implementation of the §9.3 idempotency rule.

### 15.3 Local Development

`docker compose up` brings up the full dependency set for single-tenant development:

| Service | Image | Purpose |
|---------|-------|---------|
| PostgreSQL 16 | `postgres:16` | Metadata + audit (§15.1) |
| Redis 7 | `redis:7` | Cache/locks (§4.1) |
| Temporal | `temporalio/auto-setup` | Workflow engine (§2.1) |
| Redpanda | `redpanda/redpanda` | Kafka-compatible audit bus (drop-in for Kafka in dev) |
| MinIO | `minio/minio` | S3-compatible object store |
| Vault | `hashicorp/vault` | Secrets (§6.2), dev mode |
| Qdrant | `qdrant/qdrant` | Vector search (optional) |

Production-only dependencies (OPA, SPIFFE, Grafana stack) sit behind interfaces that are mocked in dev; **no code path may require a cloud service to pass unit tests**.

---

## 16. Pipeline Orchestration: Raw Data → Production Model

The evaluation harness integrates into a **full pipeline orchestrator** that wraps existing fine-tuning tools as callable backends. This transforms the product from "evaluation only" to **"raw data → production model" platform**.

### 16.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR (Agent)                            │
│  • Plans steps  • Calls backends  • Persists state  • Gates                │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────────┬────────────┐
        ▼                  ▼                  ▼                  ▼            ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────┐ ┌─────────┐
│  DATA PREP    │  │  TRAINING     │  │  EXPORT       │  │ EVALUATION│ │ DEPLOY  │
│  BACKENDS     │  │  BACKENDS     │  │  BACKENDS     │  │ (Harness) │ │ BACKENDS│
│               │  │               │  │               │  │           │ │         │
│ • Unsloth     │  │ • Unsloth     │  │ • llama.cpp   │  │ • Curator │ │ • vLLM  │
│ • Axolotl     │  │ • Axolotl     │  │ • Unsloth     │  │ • Judge   │ │ • Ollama│
│ • Custom      │  │ • Modal       │  │ • HF          │  │ • Analyst │ │ • TGI   │
│   (PDF→JSONL) │  │ • RunPod      │  │               │  │ • Synthetic│ │ • Modal │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────┘ └─────────┘
```

**Mandatory split:** `DATA PREP` includes a stratified train/holdout split. The holdout is sealed — separate prefix, write-once, denied to training backends — and is the *only* data the evaluation harness sees. Evaluating on `raw_data_path` would score the model on its own training data; this is enforced by PF-07.

### 16.2 Pipeline Configuration

```yaml
# config/pipeline_config.yaml (extends eval_config.yaml)
pipeline:
  run_id: "auto-generated"              # Injected at runtime
  workspace_id: "ws-invoice-001"        # Injected at runtime
  
  # Data
  raw_data_path: "s3://eval-ws-invoice-001/raw/invoices/"
  dataset_format: "pdf"                 # pdf | jsonl | csv | folder | sharegpt
  task_type: "extraction"               # extraction | generation | classification | reasoning | code

  # Split (MANDATORY — the evaluated holdout must never be trained on, PF-07)
  split:
    seed: 42                            # pinned; recorded in artifact lineage
    strategy: "stratified"              # stratify on eval strata_keys
    holdout_fraction: 0.2
    min_per_stratum: 50                 # merge rule: strata_merge_rule (§14.1)
    seal: true                          # holdout → read-only prefix, denied to
                                        # training backends
  
  # Model
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  model_family: "qwen2.5"
  
  # Training
  training_backend: "modal"             # unsloth | axolotl | modal | runpod
  training_args:
    num_epochs: 3
    learning_rate: 2e-4
    batch_size: 4
    gradient_accumulation: 4
    warmup_ratio: 0.03
    remote: true                        # Run on Modal/RunPod
  lora_config:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  
  # Export
  export_backend: "llamacpp"            # llamacpp | unsloth
  quantization: ["q4_k_m", "q5_k_m", "q8_0"]
  
  # Evaluation (uses existing eval_config.yaml)
  eval_config: "config/eval_config.yaml"
  promotion_threshold: 0.85
  
  # Deployment (optional)
  deploy_backend: "vllm"                # vllm | ollama | tgi | modal
  deploy_args:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 8192
```

### 16.3 Pipeline Orchestrator Implementation

```python
# pipeline_orchestrator.py
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json
import os

class PipelineStage(Enum):
    DATA_PREP = "data_prep"
    TRAINING = "training"
    EXPORT = "export"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"

@dataclass
class PipelineConfig:
    # Injected at runtime
    run_id: str
    workspace_id: str
    
    # Data
    raw_data_path: str
    dataset_format: str
    task_type: str
    
    # Model
    base_model: str
    model_family: str
    
    # Training
    training_backend: str
    training_args: Dict[str, Any]
    lora_config: Dict[str, Any]
    
    # Export
    export_backend: str
    quantization: list[str]
    
    # Evaluation
    eval_config_path: str
    promotion_threshold: float = 0.85
    
    # Deployment
    deploy_backend: Optional[str] = None
    deploy_args: Dict[str, Any] = field(default_factory=dict)
    
    # Split (mandatory — prevents train/eval leakage; see §16.2)
    split_seed: int = 42
    holdout_fraction: float = 0.2
    holdout_min_per_stratum: int = 50
    
    @classmethod
    def from_yaml(cls, path: Path, run_id: str, workspace_id: str) -> "PipelineConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        data["run_id"] = run_id
        data["workspace_id"] = workspace_id
        return cls(**data["pipeline"])

@dataclass
class PipelineState:
    run_id: str
    workspace_id: str
    stage: PipelineStage
    completed_stages: list[PipelineStage] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # key -> artifact path/ref
    metrics: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    version: int = 0  # Optimistic lock (CAS)

    def to_dict(self) -> dict:
        # CRITICAL: enums serialize as .value, never str(enum).
        # str(PipelineStage.DATA_PREP) == "PipelineStage.DATA_PREP", which does
        # not compare equal to the enum on reload → resume never skips → PF-02 fails.
        return {
            **self.__dict__,
            "stage": self.stage.value,
            "completed_stages": [s.value for s in self.completed_stages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        data = dict(data)  # never mutate the caller's dict
        data["stage"] = PipelineStage(data["stage"])
        data["completed_stages"] = [
            PipelineStage(s) for s in data.get("completed_stages", [])
        ]
        return cls(**data)

    def save(self, path: Path, expected_version: Optional[int] = None) -> None:
        if expected_version is not None and self.version != expected_version:
            raise StateConflict(
                f"state version {self.version} != expected {expected_version}; "
                "another writer modified state since our read"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)  # atomic; S3/Postgres backends use conditional writes
                               # (If-Match ETag / UPDATE ... WHERE version = ?)

    @classmethod
    def load(cls, path: Path) -> Optional["PipelineState"]:
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return None

class PipelineGateFailed(Exception):
    pass

class StateConflict(Exception):
    """Optimistic-lock violation: another writer bumped `version` since our read."""


def assert_no_overlap(train_ref: str, holdout_ref: str) -> None:
    """PF-07 guard: MinHash overlap between train and holdout must be zero."""
    if minhash_overlap(train_ref, holdout_ref) > 0:
        raise PipelineGateFailed(
            f"train/holdout leakage detected between {train_ref} and {holdout_ref}"
        )


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig, state_path: Path):
        self.config = config
        self.state_path = state_path
        self.state = PipelineState.load(state_path) or PipelineState(
            run_id=config.run_id,
            workspace_id=config.workspace_id,
            stage=PipelineStage.DATA_PREP
        )
        self.backends = self._init_backends()
    
    def run(self) -> "PipelineResult":
        """Execute full pipeline: data → train → export → eval → deploy"""
        
        stages = [
            (PipelineStage.DATA_PREP, self._run_data_prep),
            (PipelineStage.TRAINING, self._run_training),
            (PipelineStage.EXPORT, self._run_export),
            (PipelineStage.EVALUATION, self._run_evaluation),
        ]
        if self.config.deploy_backend:
            stages.append((PipelineStage.DEPLOYMENT, self._run_deployment))
        
        for stage, fn in stages:
            if stage in self.state.completed_stages:
                continue  # Resume from failure
            
            self._update_state(stage=stage, status="running")
            try:
                artifact = fn()
                if isinstance(artifact, dict):
                    self.state.artifacts.update(artifact)  # data_prep -> {data_prep, holdout}
                else:
                    self.state.artifacts[stage.value] = artifact
                self.state.completed_stages.append(stage)
                self._update_state(status="completed")
            except Exception as e:
                self._update_state(status="failed", error=str(e))
                raise
        
        return PipelineResult(
            model_path=self.state.artifacts[PipelineStage.EXPORT.value],
            eval_result=self.state.artifacts.get(PipelineStage.EVALUATION.value),
            promoted=self.state.metrics.get("promoted", False),
            deploy_url=self.state.artifacts.get(PipelineStage.DEPLOYMENT.value)
        )
    
    def _update_state(self, stage: PipelineStage = None, status: str = None, error: str = None):
        expected = self.state.version  # capture before mutating
        if stage:
            self.state.stage = stage
        if status:
            self.state.metrics[f"{self.state.stage.value}_status"] = status
        if error:
            self.state.error = error
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state.version += 1
        # Real optimistic lock: save fails if another writer (e.g. the cancel
        # API, §16.6) bumped `version` since our read. Advisory for local files;
        # conditional-write equivalent in Postgres/S3 backends.
        self.state.save(self.state_path, expected_version=expected)
    
    # --- Backend Wrappers ---
    
    def _run_data_prep(self) -> Dict[str, str]:
        """Convert raw data → training format, then seal the holdout.

        Returns {"data_prep": train_ref, "holdout": holdout_ref}. The holdout
        goes to a separate read-only prefix that training backends are denied —
        this is what makes PF-07 (no train/eval leakage) enforceable.
        """
        from backends.data_prep import DataPrepBackend
        backend = DataPrepBackend.create(self.config.dataset_format)
        base = f"s3://eval-{self.config.workspace_id}/runs/{self.config.run_id}/data_prep"
        train_ref = backend.prepare(
            input_path=self.config.raw_data_path,
            task_type=self.config.task_type,
            output_dir=f"{base}/train",
        )
        holdout_ref = backend.split(
            input_path=self.config.raw_data_path,
            holdout_fraction=self.config.holdout_fraction,
            min_per_stratum=self.config.holdout_min_per_stratum,
            seed=self.config.split_seed,
            output_dir=f"{base}/holdout",  # sealed: write-once, denied to training
        )
        if not train_ref or not holdout_ref:
            raise PipelineGateFailed("data_prep produced no artifact (never proceed on None)")
        assert_no_overlap(train_ref, holdout_ref)  # MinHash overlap must be 0
        return {"data_prep": train_ref, "holdout": holdout_ref}
    
    def _run_training(self) -> str:
        """Run QLoRA/LoRA training"""
        from backends.training import TrainingBackend
        backend = TrainingBackend.create(self.config.training_backend)
        output_dir = f"s3://eval-{self.config.workspace_id}/runs/{self.config.run_id}/training"
        return backend.train(
            base_model=self.config.base_model,
            train_data=self.state.artifacts[PipelineStage.DATA_PREP.value],
            lora_config=self.config.lora_config,
            args=self.config.training_args,
            output_dir=output_dir
        )
    
    def _run_export(self) -> str:
        """Convert adapter → GGUF quantized models"""
        from backends.export import ExportBackend
        backend = ExportBackend.create(self.config.export_backend)
        output_dir = f"s3://eval-{self.config.workspace_id}/runs/{self.config.run_id}/export"
        return backend.export(
            model_path=self.state.artifacts[PipelineStage.TRAINING.value],
            quantization=self.config.quantization,
            output_dir=output_dir
        )
    
    def _run_evaluation(self) -> str:
        """Run agentic evaluation on the SEALED holdout — never on raw/train data."""
        from eval_orchestrator import EvalOrchestrator
        from backends.export import ExportBackend
        
        # Select best quantization (prefer q4_k_m for eval)
        export_dir = self.state.artifacts[PipelineStage.EXPORT.value]
        best_gguf = ExportBackend.select_best_gguf(export_dir, prefer=["q4_k_m", "q5_k_m"])
        
        orchestrator = EvalOrchestrator.load_config(self.config.eval_config_path)
        holdout_ref = self.state.artifacts["holdout"]  # sealed in _run_data_prep (PF-07)
        result = orchestrator.run_round(best_gguf, holdout_ref)
        
        # Gate decision — on the 95% CI lower bound, not the point estimate (§12.3)
        promoted = (
            result.decision.promote
            and result.decision.ci_lower_f1 >= self.config.promotion_threshold
        )
        
        self.state.metrics.update({
            "eval_f1": result.decision.current_f1,
            "promoted": promoted,
            "blockers": result.decision.blockers,
            "eval_round": result.decision.round
        })
        
        if not promoted:
            raise PipelineGateFailed(f"Evaluation gate failed: {result.decision.blockers}")
        
        return result.report.ci_gate  # Path to CI gate JSON
    
    def _run_deployment(self) -> str:
        """Deploy promoted model"""
        from backends.deployment import DeploymentBackend
        backend = DeploymentBackend.create(self.config.deploy_backend)
        return backend.deploy(
            model_path=self.state.artifacts[PipelineStage.EXPORT.value],
            args=self.config.deploy_args
        )

@dataclass
class PipelineResult:
    model_path: str
    eval_result: Optional[str]
    promoted: bool
    deploy_url: Optional[str] = None
```

### 16.4 Backend Registry Pattern

```python
# backends/__init__.py
from abc import ABC, abstractmethod
from typing import Dict, Type

BACKEND_REGISTRY: Dict[str, Dict[str, Type]] = {
    "data_prep": {},
    "training": {},
    "export": {},
    "deployment": {}
}

def register_backend(category: str, name: str):
    def decorator(cls):
        BACKEND_REGISTRY[category][name] = cls
        return cls
    return decorator

def get_backend(category: str, name: str):
    if name not in BACKEND_REGISTRY[category]:
        raise ValueError(f"Unknown {category} backend: {name}. Available: {list(BACKEND_REGISTRY[category].keys())}")
    return BACKEND_REGISTRY[category][name]
```

```python
# backends/training.py
from abc import ABC, abstractmethod
from backends import register_backend, get_backend

class TrainingBackend(ABC):
    @abstractmethod
    def train(self, base_model: str, train_data: str, lora_config: dict, 
              args: dict, output_dir: str) -> str:
        pass

@register_backend("training", "unsloth")
class UnslothBackend(TrainingBackend):
    def train(self, base_model, train_data, lora_config, args, output_dir):
        # Generate Unsloth training script from template
        script = self._generate_unsloth_script(base_model, train_data, lora_config, args, output_dir)
        
        if args.get("remote") == "modal":
            return self._run_on_modal(script, output_dir)
        elif args.get("remote") == "runpod":
            return self._run_on_runpod(script, output_dir)
        else:
            return self._run_local(script, output_dir)

@register_backend("training", "axolotl")
class AxolotlBackend(TrainingBackend):
    def train(self, base_model, train_data, lora_config, args, output_dir):
        # Generate axolotl YAML config
        config_yaml = self._generate_axolotl_config(base_model, train_data, lora_config, args, output_dir)
        import subprocess
        subprocess.run(["axolotl-cli", "train", config_yaml], check=True)
        return output_dir

@register_backend("training", "modal")
class ModalTrainingBackend(TrainingBackend):
    def train(self, base_model, train_data, lora_config, args, output_dir):
        import modal
        fn = modal.Function.lookup("slm-finetune", "train_lora")
        return fn.remote(
            base_model=base_model,
            train_data_s3=train_data,
            lora_config=lora_config,
            training_args=args
        )

TrainingBackend.create = staticmethod(lambda name: get_backend("training", name)())
```

```python
# backends/export.py
from pathlib import Path
import glob
from backends import register_backend, get_backend

@register_backend("export", "llamacpp")
class LlamaCppExport:
    def export(self, model_path, quantization, output_dir):
        import subprocess
        for q in quantization:
            subprocess.run([
                "llama-export", "--model", model_path,
                "--quantize", q,
                "--output", f"{output_dir}/model-{q}.gguf"
            ], check=True)
        return output_dir
    
    @staticmethod
    def select_best_gguf(export_dir: str, prefer: list[str]) -> str:
        # Logic to pick best quantization for evaluation
        for q in prefer:
            path = f"{export_dir}/model-{q}.gguf"
            if Path(path).exists():
                return path
        # Fallback: first available
        import glob
        ggufs = glob.glob(f"{export_dir}/model-*.gguf")
        return ggufs[0] if ggufs else None

@register_backend("export", "unsloth")
class UnslothExport:
    def export(self, model_path, quantization, output_dir):
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_path)
        for q in quantization:
            model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=q)
        return output_dir

ExportBackend = type("ExportBackend", (), {
    "create": staticmethod(lambda name: get_backend("export", name)()),
    "select_best_gguf": staticmethod(LlamaCppExport.select_best_gguf)
})
```

```python
# backends/data_prep.py
# Every backend must implement BOTH prepare() and split() — split() is the
# mandatory train/holdout seal (§16.2, PF-07). NotImplementedError is used
# instead of `pass`: a None artifact would silently flow into training.
from abc import ABC, abstractmethod

class DataPrepBackend(ABC):
    @abstractmethod
    def prepare(self, input_path, task_type, output_dir) -> str: ...

    @abstractmethod
    def split(self, input_path, holdout_fraction, min_per_stratum,
              seed, output_dir) -> str: ...

@register_backend("data_prep", "pdf")
class PDFDataPrep(DataPrepBackend):
    def prepare(self, input_path, task_type, output_dir):
        # PDF → OCR → JSONL (ShareGPT format)
        # Uses marker-pdf or pymupdf + Tesseract
        raise NotImplementedError("Phase 4 (§17)")

    def split(self, input_path, holdout_fraction, min_per_stratum, seed, output_dir):
        # Stratified split on eval strata_keys; writes train + sealed holdout
        raise NotImplementedError("Phase 4 (§17)")

@register_backend("data_prep", "jsonl")
class JSONLDataPrep(DataPrepBackend):
    def prepare(self, input_path, task_type, output_dir):
        # Validate/convert to ShareGPT format
        raise NotImplementedError("Phase 4 (§17)")

    def split(self, input_path, holdout_fraction, min_per_stratum, seed, output_dir):
        raise NotImplementedError("Phase 4 (§17)")

@register_backend("data_prep", "folder")
class FolderDataPrep(DataPrepBackend):
    def prepare(self, input_path, task_type, output_dir):
        # Recursively process folder of mixed formats
        raise NotImplementedError("Phase 4 (§17)")

    def split(self, input_path, holdout_fraction, min_per_stratum, seed, output_dir):
        raise NotImplementedError("Phase 4 (§17)")

# Registry factory (same pattern as TrainingBackend.create)
DataPrepBackend.create = staticmethod(lambda name: get_backend("data_prep", name)())
```

```python
# backends/deployment.py
# Stubs raise instead of `pass` — a None deploy URL would be reported as success.
@register_backend("deployment", "vllm")
class VLLMDeployment:
    def deploy(self, model_path, args):
        # Deploy to vLLM (K8s, Docker, or Modal); verify /health before returning
        raise NotImplementedError("Phase 4 (§17)")

@register_backend("deployment", "modal")
class ModalDeployment:
    def deploy(self, model_path, args):
        import modal
        fn = modal.Function.lookup("slm-serve", "deploy_vllm")
        return fn.remote(model_path=model_path, args=args)

DeploymentBackend = type("DeploymentBackend", (), {
    "create": staticmethod(lambda name: get_backend("deployment", name)())
})
```

### 16.5 CLI & API

```bash
# CLI
slm-pipeline run \
  --config config/pipeline_config.yaml \
  --workspace ws-invoice-001 \
  --resume  # Resume from last completed stage

# Output:
# 📁 Data Prep (pdf→ShareGPT)... ✅ 2,340 samples
# ✂️ Split (seed=42, stratified)... ✅ train 1,872 / holdout 468 (0 overlap)
# 🚀 Training (modal, A100, 42 min)... ✅ LoRA adapter
# 📦 Export (llamacpp, q4_k_m/q5_k_m/q8_0)... ✅ 3 GGUFs
# 🔍 Evaluation (3 rounds)... ✅ F1=0.87 → PROMOTED
# 🚀 Deploy (vLLM)... ✅ https://eval-model.company.com/v1
```

```python
# Python API
from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig

config = PipelineConfig.from_yaml("config/pipeline_config.yaml", run_id="run-042", workspace_id="ws-001")
result = PipelineOrchestrator(config, Path("state/pipeline_state.json")).run()

print(f"Model: {result.model_path}")
print(f"Promoted: {result.promoted}")
print(f"Endpoint: {result.deploy_url}")
```

### 16.6 Pipeline API Endpoints

```yaml
# Extends OpenAPI spec in Section 9
paths:
  /workspaces/{workspace_id}/pipelines:
    post:
      summary: Start full pipeline run
      parameters:
        - $ref: '#/components/parameters/IdempotencyKey'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PipelineRunCreate'
      responses:
        '202':
          description: Accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PipelineRunStatus'
        '409':
          $ref: '#/components/responses/Problem'

  /workspaces/{workspace_id}/pipelines/{run_id}:
    get:
      summary: Get pipeline run status
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PipelineRunStatus'

  /workspaces/{workspace_id}/pipelines/{run_id}/cancel:
    post:
      summary: Cancel running pipeline
      responses:
        '200':
          description: Cancelled

components:
  schemas:
    PipelineRunCreate:
      type: object
      required: [config_ref]
      description: >
        A pipeline starts from raw data + base model (§16.2), NOT from a
        pre-registered model/dataset — those belong to POST /runs (§9.1).
      properties:
        config_ref:
          type: string
          description: s3:// URI of pipeline_config.yaml uploaded to this workspace
        config_override:
          type: object
        priority:
          type: string
          enum: [low, normal, high]
          default: normal

    PipelineRunStatus:
      type: object
      properties:
        run_id:
          type: string
        status:
          type: string
          enum: [queued, running, completed, failed, cancelled]
        current_stage:
          type: string
          enum: [data_prep, training, export, evaluation, deployment]
        stages:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              status:
                type: string
                enum: [pending, running, completed, failed]
              started_at:
                type: string
                format: date-time
              completed_at:
                type: string
                format: date-time
              artifacts:
                type: object
              metrics:
                type: object
        cost_usd:
          type: number
        promoted:
          type: boolean
        deploy_url:
          type: string
```

### 16.7 What This Enables

| User Type | Before (Eval Only) | After (Full Pipeline) |
|-----------|-------------------|----------------------|
| **Developer** | Fine-tune manually → run eval CLI | One command: `slm-pipeline run --config ...` |
| **Layman (via UI)** | ❌ Impossible | Upload PDFs → get API endpoint (web console per `docs/ui/SPEC.md`, §17) |
| **CI/CD** | Manual model promotion | Auto-gate: eval passes → deploy |
| **ML Engineer** | Stitch 5 tools together | Config-driven, reproducible, auditable |

---

## 17. Deliverables

| Artifact | Location | Owner |
|----------|----------|-------|
| **API Spec** | `docs/api/openapi.yaml` | Platform |
| **Terraform Modules** | `infra/modules/*` | DevOps |
| **Helm Charts** | `deploy/helm/slm-eval/*` | DevOps |
| **Agent SDK** | `slm_eval/sdk.py` | Core Team |
| **Pipeline SDK** | `slm_pipeline/sdk.py` | Core Team |
| **Golden Datasets** | `tests/golden/` | ML Team |
| **Runbooks** | `ops/runbooks/*.md` | SRE |
| **Pipeline Runbooks** | `ops/runbooks/pipeline/*.md` | SRE |
| **Compliance Evidence** | `compliance/evidence/` | Security |
| **Architecture Decision Records** | `docs/adr/*.md` | Architecture |
| **DB Migrations + Data Model** | `db/migrations/*`, §15 | Platform |
| **Web Console Spec** | `docs/ui/SPEC.md` | Product |
| **Sub-processor Register** | `compliance/subprocessors.yaml` | Security |

---

## 18. Acceptance Criteria (Production) — Extended

### 18.1 Functional (Pipeline)

| ID | Criterion | Verification |
|----|-----------|--------------|
| PF-01 | End-to-end pipeline execution | Raw PDFs → deployed endpoint in < 2 hrs |
| PF-02 | Resume from failure | Kill at training → resume completes export+eval |
| PF-03 | Gate enforcement | Failed eval blocks deployment (100% of cases) |
| PF-04 | Multi-backend training | Same config runs on local/Modal/RunPod |
| PF-05 | Artifact lineage | Every artifact traceable to raw data + config |
| PF-06 | Cost tracking | Per-stage cost logged, total matches estimate ±10% |
| PF-07 | Train/holdout isolation | MinHash overlap(train, holdout) = 0; asserted in CI on every pipeline run |
| PF-08 | Residency enforcement | EU workspace: egress audit shows 0 out-of-region LLM/GPU calls over 30 days |
| PF-09 | Human review blocking | `requires_human_review` cases block the gate until decided or 24h escalation (§9.3) |

### 18.2 Non-Functional (Pipeline)

| ID | Criterion | Target | Test |
|----|-----------|--------|------|
| PNF-01 | Pipeline latency (p99) | < 2 hrs | Load test: 10 concurrent pipelines |
| PNF-02 | Training success rate | > 95% | 100 runs across backends |
| PNF-03 | Export success rate | > 99% | All quantizations produced |
| PNF-04 | Deploy success rate | > 99% | Endpoint healthy post-deploy |
| PNF-05 | DR restore drill | Meets §4.3 RTO/RPO | Quarterly restore test (§4.3) |

### 18.3 Operational (Pipeline)

| ID | Criterion | Verification |
|----|-----------|--------------|
| PO-01 | Pipeline runbook coverage | 100% of P1/P2 pipeline alerts |
| PO-02 | Cancel/restart | Running pipeline can be cancelled, restarted cleanly |
| PO-03 | Backend health checks | Each backend has liveness/readiness probes |
| PO-04 | Erasure SLA | Workspace purge (incl. backups) completes ≤35d, certificate issued (§7.3) |

---

**Approval Required**: Platform Lead, ML Lead, Security Lead, SRE Lead  
**Next Review**: 2026-11-25