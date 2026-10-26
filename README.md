# slm-eval — Agentic Evaluation Harness for SLM Fine-Tuning

Single-tenant, CLI-first evaluation gate for small language model fine-tuning loops:
**model + dataset in → `gate.json` (F1 with 95% CI) + exit code out.**

The gate promotes only when the *confidence interval lower bound* clears the target,
the judge is κ-calibrated (≥ 0.85), strata show no blockers, and human reviews are
decided. Point estimates never gate.

## Specs

| Doc | Role |
|---|---|
| [`SPEC_V3.md`](SPEC_V3.md) | **Build contract** — architecture, data model, evaluation science, M0–M3, acceptance criteria |
| [`SPEC.md`](SPEC.md) | v1 prototype detail (agent/config reference carried into V3) |
| [`SPEC_PRODUCTION.md`](SPEC_PRODUCTION.md) | Evolution target — features attach only when Appendix A triggers fire |

## Status

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Scaffold, config schema, SQLite store, golden fixture | pending |
| M1 | split/seal → infer → score → CI → `gate.json` + CLI | pending |
| M2 | Analyst, calibration tooling + κ enforcement, review queue | pending |
| M3 | Synthetic loop, reports, provenance inspect | pending |

## Quickstart (after M1)

```bash
pip install -e .[dev]
slm-eval init
slm-eval run -c eval_config.yaml --model model.gguf --dataset data/holdout/
slm-eval gate runs/<run_id>   # exit 0 promote · 1 blocked · 2 error
```

## Development

```bash
pytest        # tests (every milestone ships tests)
ruff check .
```
