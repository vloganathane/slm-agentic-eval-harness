# Agentic Evaluation Harness for SLM Fine-Tuning — Build Specification

**Version**: 3.0 (Build Spec)
**Date**: 2026-10-26
**Scope**: Single-tenant, CLI-first, eval-gate-first (milestones M0–M3)
**Supersedes**: `SPEC.md` v1 for build purposes (agent/config detail carried over)
**Evolves into**: `SPEC_PRODUCTION.md` — see Appendix A for trigger conditions
**Working mode**: Autonomous build to M1, then review gate (agreed 2026-10-26)

---

## 1. Golden Path

*You have fine-tuned Qwen2.5-1.5B on invoice data and exported it to GGUF. You have a
held-out invoice dataset with strata (layout, language, quality). One command tells you
whether this model is safe to promote — with a confidence interval, not a vibe.*

```bash
slm-eval run \
  --config eval_config.yaml \
  --model models/qwen-ft-042-q4_k_m.gguf \
  --dataset data/invoices-holdout-v3
```

What happens, end to end:

1. Config validates against the §8.3 schema; inputs are hashed (provenance).
2. Dataset splits into **train remainder + sealed holdout** (stratified, seeded, zero-overlap asserted).
3. Inference runs the model over the holdout via a local OpenAI-compatible server.
4. Scoring: exact-match fields scored deterministically; subjective fields scored by a κ-calibrated judge (temperature 0).
5. Aggregation: micro-F1 with a **95% bootstrap CI**, overall and per stratum.
6. Gate evaluates: CI lower bound vs. target, stratum blockers, judge κ, pending reviews.
7. Outputs written; exit code is the CI signal.

### 1.1 The contract

```
inputs:  model (GGUF) + dataset (dir) + eval_config.yaml + judge endpoint
outputs: runs/<run_id>/gate.json      ← the product
         runs/<run_id>/report.md      ← human summary
         runs/<run_id>/events.jsonl   ← audit/cost/provenance
         exit code: 0 promote · 1 blocked · 2 error
```

`gate.json` (stable schema — CI consumers depend on it):

```json
{
  "run_id": "2026-10-26T14-30-00-qwen-ft-042",
  "schema_version": 1,
  "model": {"path": "models/qwen-ft-042-q4_k_m.gguf", "sha256": "…"},
  "holdout": {"n": 468, "hash": "…", "split_seed": 42},
  "decision": {"promote": false, "reason": "ci_lower_below_target"},
  "metrics": {
    "micro_f1": 0.871, "ci_lower": 0.842, "ci_upper": 0.897,
    "resamples": 10000, "seed": 42
  },
  "strata": [
    {"key": "layout=rotated_90", "n": 47, "micro_f1": 0.72,
     "ci_lower": 0.61, "ci_upper": 0.81, "status": "warning"}
  ],
  "judge": {"kappa": 0.89, "pairs": 60, "model": "gpt-4o-mini", "prompt_version": "v1.3"},
  "blockers": [],
  "warnings": ["stratum layout=rotated_90 below target"],
  "pending_reviews": 0,
  "cost_usd": 1.12,
  "provenance": {"config_hash": "…", "code_version": "git:abc123"}
}
```

---

## 2. Non-Goals (v1)

Explicitly out. Each maps to a trigger in Appendix A — this is the scope firewall.

| Non-goal | Why now | Trigger to revisit |
|---|---|---|
| Multi-tenancy, RBAC, workspaces | Solo/team tool | Second tenant exists |
| HTTP API server, WebSocket, web UI | CLI is the interface | Second consumer (CI is #1) |
| Audit infra (Kafka, hash chains, Vault) | Local events.jsonl suffices | Compliance requirement appears |
| Temporal/Redis/Kafka/Postgres | SQLite + in-process orchestration | Concurrent multi-run pain |
| SOC2 / GDPR / residency machinery | Not selling yet | Customer asks |
| Training / data-prep pipeline (raw → model) | Eval-gate-first was chosen | M1 proven; pipeline = separate milestone |
| Modal/RunPod GPU fleet | Local llama.cpp first | Local inference too slow |
| Non-extraction task types (chat, RAG) | Extraction is the beachhead | Task type demand is real |
| Custom third-party agent SDK | Two lanes, in-repo agents | External teams ship agents |

---

## 3. Architecture

Single process, no services, local state. Layers (dependencies point down only):

```
┌──────────────────────────────────────────────┐
│ CLI (slm-eval)  — §8                         │
├──────────────────────────────────────────────┤
│ Orchestrator    — stage runner, checkpoints  │  §5
├──────────────────────────────────────────────┤
│ Agents (two lanes) — §7                     │
│   deterministic: curator · scorer · judge   │
│   · aggregator · gate · reporter            │
│   advisory:      analyst · synthetic        │
├──────────────────────────────────────────────┤
│ Backends: LLM client · inference client ·   │
│   storage (SQLite + run dir) · minhash      │
└──────────────────────────────────────────────┘
```

### 3.1 Buy vs. build

| Concern | Choice | Verdict |
|---|---|---|
| LLM/inference client | `openai` SDK pointed at any OpenAI-compatible endpoint (llama-server, vLLM, hosted) | **Buy** |
| Structured outputs | `pydantic` v2 validation + JSON mode; retry on parse failure | **Buy** |
| Retries/backoff | `tenacity` | **Buy** |
| MinHash dedup | `datasketch` | **Buy** |
| Bootstrap CI | `numpy` resampling (10k, fixed seed) | **Buy lib, build logic** |
| Cohen's κ | ~60 lines, verified against textbook examples in tests | **Build** |
| Config schema | YAML (`pyyaml`) + pydantic model | **Buy parser, build schema** |
| State store | `sqlite3` (stdlib) with optimistic locking | **Buy (stdlib)** |
| Reporting | Markdown writer | **Build** |
| Workflow engine | In-process stage runner + stage checkpoints | **Build (~150 lines)** |
| Agent framework | None — `BaseAgent` interface (§7) | **Build (~60 lines)** |

### 3.2 Dependency set

Runtime: `pydantic≥2`, `openai≥1`, `tenacity`, `numpy`, `datasketch`, `pyyaml`.
Dev: `pytest`, `ruff`. Python ≥3.11. **No other runtime deps without a decision-log entry.**

---

## 4. Data Model

### 4.1 On-disk layout

```
project/
├── eval_config.yaml
├── slm_eval.db                    # runs, scores, checkpoints, calibration, cost
└── runs/<run_id>/
    ├── config.snapshot.yaml       # exact config used (content-hashed)
    ├── holdout/cases.jsonl        # sealed split (read-only by convention + hash)
    ├── predictions.jsonl          # per-case model outputs
    ├── scores.jsonl               # per-(case, field) correctness + judge detail
    ├── gate.json                  # §1.1
    ├── report.md
    └── events.jsonl               # structured log + cost ledger
```

### 4.2 SQLite schema (v1, single-tenant)

```sql
CREATE TABLE run (
  run_id TEXT PRIMARY KEY, model_path TEXT, model_sha256 TEXT,
  dataset_path TEXT, dataset_hash TEXT, holdout_hash TEXT,
  status TEXT CHECK (status IN ('created','running','completed','failed')),
  config_hash TEXT, code_version TEXT,
  created_at TEXT, updated_at TEXT, version INTEGER DEFAULT 0   -- optimistic lock
);
CREATE TABLE stage_checkpoint (
  run_id TEXT, stage TEXT, input_hash TEXT, status TEXT,
  output_ref TEXT, error TEXT, started_at TEXT, completed_at TEXT,
  PRIMARY KEY (run_id, stage)
);
CREATE TABLE score (
  run_id TEXT, round INTEGER, case_id TEXT, field TEXT,
  correct INTEGER, tp INTEGER, fp INTEGER, fn INTEGER,
  judge_model TEXT, prompt_hash TEXT, latency_ms INTEGER,
  UNIQUE (run_id, round, case_id, field, prompt_hash)
);
CREATE TABLE stratum_case (
  run_id TEXT, case_id TEXT, stratum_key TEXT, stratum_value TEXT
);
CREATE TABLE calibration (
  id INTEGER PRIMARY KEY, pair_hash TEXT UNIQUE, gold TEXT,
  judge TEXT, label TEXT, kappa REAL, pairs INTEGER, created_at TEXT
);
CREATE TABLE review_task (
  run_id TEXT, case_id TEXT, reason TEXT, state TEXT,
  assigned_at TEXT, due_at TEXT, decided_at TEXT, decision TEXT, notes TEXT
);
CREATE TABLE cost_ledger (
  run_id TEXT, stage TEXT, usd REAL, tokens_in INTEGER, tokens_out INTEGER, ts TEXT
);
```

### 4.3 Invariants

- **Content addressing**: model, dataset, holdout, and config each get a SHA-256; every downstream artifact records the upstream hash it consumed (provenance chain, `slm-eval inspect`).
- **Score uniqueness** as defined — the physical implementation of judge-result dedup (§6.3).
- **`run.version`** is bumped on every state write; writes use `UPDATE … WHERE version = ?` and raise `StateConflict` on mismatch (the §16.3 lesson from v2, applied from day one).
- `holdout/cases.jsonl` is written once; a later write attempt with a different hash is a hard error.

---

## 5. Pipeline Semantics

### 5.1 Stages

| # | Stage | Lane | Milestone |
|---|---|---|---|
| 1 | `validate` — config schema, inputs exist, judge reachable | deterministic | M1 |
| 2 | `split` — stratified split, **seal holdout, zero-overlap assert** | deterministic | M1 |
| 3 | `infer` — batch inference over holdout | deterministic | M1 |
| 4 | `score_exact` — validators on exact-match fields | deterministic | M1 |
| 5 | `score_judge` — judge on subjective fields (cached by prompt_hash) | deterministic | M1 |
| 6 | `aggregate` — micro-F1 + bootstrap CI, per stratum | deterministic | M1 |
| 7 | `gate` — decision per §6.5 → `gate.json` | deterministic | M1 |
| 8 | `analyze` — error categorization → **proposals** | advisory | M2 |
| 9 | `synthesize` — adversarial cases from proposals → dedup → next-round holdout | advisory | M3 |
| 10 | `report` — `report.md` | deterministic | M3 (stub in M1) |

### 5.2 State machine & resume

```
PENDING → RUNNING → COMPLETED
                  → FAILED (error recorded; resume allowed)
```

- Each stage checkpointed with `input_hash` (hash of all upstream outputs).
- `slm-eval run --resume` skips stages whose checkpoint is `COMPLETED` **and** whose
  `input_hash` matches current inputs; a mismatch invalidates downstream stages.
- Crash-safety: stage outputs are written to temp files, then atomically renamed.

### 5.3 Split & seal

- `strategy: stratified`, `seed` from config (default 42), `holdout_fraction` (default 0.2),
  `min_per_stratum` floor with the merge rule from §6.2.
- After split: MinHash overlap(train, holdout) must be **0** — non-zero aborts the run
  immediately (exit 2) with both hashes logged. This is leakage proof (AC-02).
- The training remainder is *never* read by any stage after `split`; only the sealed
  holdout path flows forward. (v1 evaluates a pre-trained model; the remainder exists for
  provenance and for the future pipeline milestone.)

### 5.4 Failure modes

| Failure | Behavior |
|---|---|
| Inference server down / timeout | `tenacity` retry (exp backoff, 3 attempts) → stage FAILED, exit 2; resume after fixing backend |
| Judge API 429/5xx | Retry with backoff → FAILED; **partial scores preserved** (UNIQUE dedup makes resume idempotent) |
| Judge returns malformed JSON | Re-request with repair prompt (max 2), then FAILED |
| Split overlap > 0 | Hard abort before any inference — never proceeds |
| Stratum below `min_per_stratum` | `insufficient_n` flag in gate — **never a blocker** (§6.5) |
| Judge κ below target | Promotion blocked (exit 1), run completes — not an error |
| SQLite `StateConflict` | Retry once, then FAILED with clear message (concurrent writer) |
| Cost cap exceeded (`max_cost_per_round_usd`) | Stage pauses at case boundary; gate records `cost_exceeded` warning; `--force` overrides |

---

## 6. Evaluation Science ★ (the product)

### 6.1 Fields & validators

```yaml
subjective_fields: [vendor_name, address, description]
exact_match_fields: [invoice_number, date, total, tax_id, line_items]
field_validators:
  tax_id:   {type: regex, pattern: "^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$"}
  total:    {type: numeric, tolerance: 0.01}
  line_items: {type: list, item_match: exact}
  date:     {type: date, formats: ["%Y-%m-%d", "%d/%m/%Y"]}
  invoice_number: {type: string, normalize: [trim, collapse_ws, casefold]}
```

- Every field defaults to `string` with `normalize: [trim, collapse_ws]` if unspecified.
- Scoring is **element-level** so micro-F1 is genuine precision/recall: scalar fields
  contribute one element; list fields match items (TP per matched item; unmatched pred =
  FP, unmatched gold = FN). Judge-scored fields contribute one element (correct = TP,
  else FP + FN).

### 6.2 Stratification

- `strata_keys` from config; `strata_mode: single_key` default (crossed explodes past
  budget — v2 §14.1 lesson). Each case gets one `(key, value)` per declared key.
- Merge rule: values below `min_per_stratum` merge into `other_<key>` **at split time**,
  never silently dropped; merged counts reported in gate.

### 6.3 Judge design

- Temperature 0, fixed prompt template with **version id** (`prompt_version`), structured
  JSON output validated by pydantic; rubric included in the prompt.
- Judge call = f(gold, pred, rubric); result cached in `score` by
  `prompt_hash = sha256(template_version + gold + pred + model)` — reruns are free and
  byte-identical (AC-03).
- v1 uses **one call per subjective field** (cost); revisit under κ pressure (Decision D-08).

### 6.4 Calibration (mandatory)

```bash
slm-eval calibrate export --run <id> --n 60 --out pairs.jsonl   # sampled across strata
#   …human labels pairs.jsonl…
slm-eval calibrate apply  --run <id> --labels labels.jsonl       # computes κ, stores row
slm-eval calibrate status --run <id>                             # κ, pairs, CI, age
```

- Rules: ≥ `min_pairs` (50) labeled pairs; **κ ≥ 0.85 required for any promotion** when
  `subjective_fields` is non-empty; κ CI (Fleiss/Cohen bootstrap) reported; recalibrate
  every `recalibrate_every_rounds` (5) rounds or when κ point estimate drops below 0.85.
- κ is computed in-house (Decision D-04) and unit-tested against published examples.

### 6.5 Gate statistics

```
micro-F1 = ΣTP / (ΣTP + ΣFP + ΣFN)          # element-level, §6.1
CI       = bootstrap over cases, B=10_000, seed=stats.seed, percentile [2.5, 97.5]
           (cases resampled with replacement; stratum CIs resample within the stratum)

BLOCKER  : stratum with n ≥ min_per_stratum AND ci_upper ≤ failure_threshold
WARNING  : stratum with n ≥ min_per_stratum AND point_f1 < target_f1
           | insufficient_n | κ CI lower < 0.80 | cost > alert | pending reviews
PROMOTE  = ci_lower_micro ≥ target_f1
           AND blockers = ∅
           AND κ ≥ 0.85            (only when subjective_fields ≠ ∅)
           AND pending_reviews = 0
           AND run completed
```

- Point estimates never gate. This is AC-04/AC-05.
- `pending_reviews > 0` → gate status `blocked_pending_review` (not a failure).

### 6.6 Determinism & golden regression

- Same inputs + same seeds + judge cache ⇒ **byte-identical `gate.json` minus
  timestamps** (asserted by hashing canonicalized JSON; AC-03).
- Golden regression (statistical, not ±2%): fixture model + fixture dataset →
  `ci_lower ≥ expected − 0.03` and `promote == (expected ≥ target_f1)` (AC-06).

---

## 7. Agent Contracts

### 7.1 Interface (in-repo, ~60 lines)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class AgentContext(BaseModel):
    run_id: str
    round: int
    config: "EvalConfig"          # §8.3
    holdout_path: str             # sealed — the only data a scoring agent may read
    db: "Store"                   # SQLite wrapper; scores/checkpoints/cost go through it
    llm: "LLMClient"              # OpenAI-compatible; logs prompt_hash + cost
    advisory: bool = False        # lane marker (§7.2)

class BaseAgent(ABC, Generic[InputT, OutputT]):
    name: str
    version: str
    advisory: bool = False        # True → advisory lane rules apply

    @abstractmethod
    def validate_input(self, ctx: AgentContext, input: InputT) -> None:
        """Raise AgentInputError; orchestrator marks stage FAILED (fail loud)."""

    @abstractmethod
    async def execute(self, ctx: AgentContext, input: InputT) -> OutputT: ...

    def validate_output(self, ctx: AgentContext, output: OutputT) -> None:
        """Default: pydantic type check. Agents may tighten, never loosen."""
```

### 7.2 Two lanes — *agents may propose, the pipeline disposes*

| | Deterministic lane | Advisory lane |
|---|---|---|
| Agents | curator/split, scorer, judge, aggregator, gate, reporter | analyst, synthetic |
| Contract | Output **is** the result | Output is a **proposal**; a deterministic rule (or human review) selects |
| Failure | Stage FAILED (exit 2) | Proposal dropped, logged as warning; run continues |
| May write | scores, gate, checkpoints | proposals table + candidate cases (deduped, review-gated) |
| Cache | Reusable as truth | Never reused as truth |

This is the boundary that keeps agency *around* the measurement, never inside it.

### 7.3 Custom agents

Config-declared (`agents:` list → class path + version), must subclass `BaseAgent`,
advisory by default, subject to per-agent token/cost budget. In-repo only for v1
(Appendix A trigger: external teams ship agents).

---

## 8. CLI & Config

### 8.1 Commands

```bash
slm-eval init                       # scaffold eval_config.yaml + dirs + db
slm-eval run -c eval_config.yaml --model M.gguf --dataset D/ [--resume] [--force]
slm-eval gate runs/<id>             # print gate.json summary → exit code (CI entrypoint)
slm-eval calibrate export|apply|status    # §6.4
slm-eval report runs/<id>           # render report.md
slm-eval inspect runs/<id>          # provenance: hashes, strata, stage timeline, cost
```

### 8.2 Exit codes

| Code | Meaning | CI usage |
|---|---|---|
| 0 | Run completed AND promote = true | pass |
| 1 | Run completed AND promote = false (blocked) | fail build |
| 2 | Run error (config, backend, leakage, crash) | fail build, investigate |

### 8.3 `eval_config.yaml` (v1)

```yaml
task_description: "Structured field extraction from Indian GST invoices"
task_type: extraction                    # v1: extraction only (§2)

fields:
  subjective_fields: [vendor_name, address, description]
  exact_match_fields: [invoice_number, date, total, tax_id, line_items]
  validators: {}                         # §6.1

strata:
  keys: [layout_type, language, quality]
  mode: single_key
  definitions: strata_definitions.yaml   # optional; else free-form values

split:        {fraction: 0.2, seed: 42, strategy: stratified}
stats:        {seed: 42, resamples: 10000}

thresholds:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50

judge:
  endpoint: "https://api.openai.com/v1"
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY            # env var name, never inline secrets
  prompt_version: v1.3
  calibration: {path: calibration/judge.jsonl, min_pairs: 50,
                target_kappa: 0.85, recalibrate_every_rounds: 5}

inference:
  backend: llama_server                  # any OpenAI-compatible base_url
  base_url: "http://127.0.0.1:8080/v1"
  timeout_seconds: 1800
  retry: {max_attempts: 3, backoff_base: 2}

human_review: {timeout_hours: 24, notify: []}

cost_control: {max_cost_per_round_usd: 10.0, alert_at_usd: 5.0}
reporting:    {formats: [markdown, json]}
```

---

## 9. Observability & Cost Control (lightweight)

- **`events.jsonl`**: one JSON object per event — `{"ts", "level", "stage", "event",
  "run_id", "round", …metrics}`; mirrors v2 §5.1 shape so a later OTel export is mechanical.
- **Cost ledger**: every LLM/inference call appends `(stage, usd, tokens)` to
  `cost_ledger`; `gate.json.cost_usd` is the sum; caps per §8.3.
- **Console**: human-readable progress on stderr, machine output on stdout.
- No dashboards, no tracing backend in v1 (Appendix A).

---

## 10. Delivery Plan (M0–M3)

| Milestone | Deliverables | Exit criterion | Depends on you |
|---|---|---|---|
| **M0** | Repo scaffold, config schema + pydantic types, SQLite store, synthetic golden fixture (invoices), test harness | `pytest` green; config validates; fixture round-trips | — |
| **M1** ★ | Stages 1–7: split/seal, inference (llama-server), exact + judge scoring, bootstrap CI, gate.json, CLI run/resume/gate, events + cost | Golden regression passes with CI bounds; exit codes correct; resume-after-crash test passes; deterministic hash test passes | **At M1 start**: dataset location, base model GGUF, judge API key (or local-judge fallback decision) |
| **M2** | Stage 8 analyst (advisory), calibration tooling + κ enforcement, human-review queue blocks gate | κ < 0.85 blocks promotion (test); pending review blocks gate (test) | Labeled calibration pairs (50) — tooling exports, you label |
| **M3** | Stage 9 synthetic (advisory, MinHash-deduped, review-gated), report.md, inspect, cost alerting | Full loop: two consecutive rounds run end-to-end; synthetic cases enter next-round holdout with zero train overlap | — |

Working mode: **autonomous through M1**, then review gate (agreed 2026-10-26).

---

## 11. Acceptance Criteria

| ID | Criterion | Verification |
|---|---|---|
| AC-01 | Golden path: model+dataset → `gate.json` + correct exit code | E2E test on fixture (M1) |
| AC-02 | Leakage proof: train∩holdout = 0; non-zero aborts | Unit test + injected-overlap failure test |
| AC-03 | Determinism: rerun with cache ⇒ identical canonical gate hash | Test asserts equal hashes |
| AC-04 | Gate uses CI lower bound, never point estimate | Unit tests at boundary values (0.849 vs 0.851) |
| AC-05 | κ < 0.85 blocks promotion; pending review blocks gate | M2 tests with injected κ/review state |
| AC-06 | Golden regression is statistical (CI-bounded) | Test per §6.6 |
| AC-07 | Resume: kill mid-infer → `--resume` completes without redoing split/score | Integration test |
| AC-08 | Per-stratum metrics + statuses (blocker/warning/insufficient_n) in gate.json | Fixture with known strata |
| AC-09 | Cost ledger sums equal `gate.json.cost_usd`; cap enforcement works | Unit + cap-exceeded test |
| AC-10 | Provenance: `inspect` traces any score back to config/model/dataset hashes | Test walks the chain |
| AC-11 | Advisory agents cannot write scores/gate (lane enforcement) | Test: advisory write attempt raises |
| AC-12 | Synthetic cases (M3) dedupe into holdout with zero train overlap | Test with injected duplicate |

---

## 12. Risks & Open Questions

| # | Risk / question | Impact | Mitigation | Owner |
|---|---|---|---|---|
| R-1 | Subjective-field gold labels missing or noisy in dataset | Judge κ meaningless; gate can't cover those fields | M1 audit of dataset labels; fall back to exact-match-only scoring with gate warning | You |
| R-2 | No judge API key / cost concerns | M1 blocked | Local judge via same llama-server endpoint (config switch); κ still enforced | You at M1 |
| R-3 | Dataset has no strata annotations | Per-stratum science collapses to overall | Derive strata from data (OCR quality heuristics, script detection) — decision needed at M1 | You at M1 |
| R-4 | Local inference too slow for holdout size | Round latency | Downsample holdout, or enable vLLM backend (Appendix A trigger) | Build |
| R-5 | Bootstrap CI too wide at n≈200 → never promotes | Frustration | Widen holdout, or justify target vs. achievable n — surfaced honestly in gate | You |
| R-6 | Scope creep mid-build | Schedule | §2 firewall: anything not in M0–M3 goes to Appendix A | Both |

---

## 13. Decision Log

| ID | Decision | Alternatives | Why | Date |
|---|---|---|---|---|
| D-01 | Spec-first, autonomous to M1 | Code-first, pair style | Quality-first intake; spec is the contract for autonomous build | 2026-10-26 |
| D-02 | SQLite + run-dir artifacts | JSON file, Postgres | ACID/concurrency without services; JSON broke resume in v2 | 2026-10-26 |
| D-03 | Bootstrap CI (numpy) | Wilson/beta analytically | micro-F1 over element-level outcomes isn't binomial; bootstrap is assumption-free and per-stratum ready | 2026-10-26 |
| D-04 | Cohen's κ in-house (~60 lines) | sklearn (heavy), statsmodels | Trivial math, must be textbook-verified in tests; no new deps | 2026-10-26 |
| D-05 | No agent framework; `BaseAgent` in-repo | LangGraph, pydantic-ai, CrewAI | Fixed control flow; frameworks fight determinism/audit (see design discussion 2026-10-26) | 2026-10-26 |
| D-06 | Local llama.cpp first | Modal/RunPod | No services in v1; same OpenAI interface keeps backend swappable | 2026-10-26 |
| D-07 | Synthetic loop at M3, after measurement core proven | Build full loop first | Can't validate an adaptive loop on an unvalidated gate | 2026-10-26 |
| D-08 | One judge call per field in v1 | Two-call (v2 style) | Halves cost; revisit if κ < 0.85 with single call | 2026-10-26 |
| D-09 | `single_key` strata mode default | Crossed keys | Crossed explodes past `min_per_stratum × budget` (v2 lesson) | 2026-10-26 |
| D-10 | Element-level TP/FP/FN for micro-F1 | Field-accuracy | Real precision/recall; list fields (line_items) need item matching | 2026-10-26 |

---

## Appendix A: Evolution Path (from `SPEC_PRODUCTION.md`)

Each production feature is **earned**, not inherited. Attach when the trigger fires.

| Feature | Trigger condition | v2 reference |
|---|---|---|
| HTTP API + OpenAPI + presigned uploads | Second consumer beyond CLI (CI bot API) | §9 |
| Web UI | Non-engineer users | §16.7, `docs/ui/SPEC.md` |
| Multi-tenancy, RBAC, quotas | Second team/tenant exists | §3 |
| Temporal orchestration | Concurrent multi-run or multi-worker execution pain | §2, §4 |
| Kafka/append-only audit, Vault, mTLS | Compliance requirement (SOC2/customer) | §6 |
| Postgres migration | >1 concurrent writer / size limits | §15 |
| Modal/RunPod fleet | Local inference latency unacceptable (R-4) | §11.1 |
| Residency guards, sub-processor register | EU/regulated deployment | §3.3, §6.4 |
| Pipeline: raw data → train → export → deploy | Eval gate proven; demand for one-command models | §16 |
| Custom external agent SDK | Third parties ship agents | §10 |
| OTel/Grafana dashboards | Ops scale beyond events.jsonl + inspect | §5 |
| Model registry + promote/rollback API | >1 deployed model tracked by the tool | §9.1 |

---

**Build order**: M0 → M1 (autonomous) → *review gate* → M2 → M3.
**Definition of done (v1)**: M3 exit criteria + AC-01…AC-12 green.
