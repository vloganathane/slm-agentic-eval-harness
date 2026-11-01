# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modal GPU backend for LoRA training and vLLM deployment (`slm-eval modal`)
- Dataset pipeline (`slm-eval data`) — validate, dedup, stratified split, training format
- LoRA training (`slm-eval train`) — PEFT/TRL SFT with checkpointing
- vLLM serving (`slm-eval serve`, `run --auto-serve`)
- AI Data Science analysis report (`runs/<id>/viz/ai_analysis_report.json`)
- Agent contract (`slm_eval.agents`) — advisory lane per AC-11

### Changed
- Removed experimental Rust `slm-eval-infinity` (not used in production)
- Python `Orchestrator` is the canonical pipeline

## [0.1.0] - 2024-10-31

### Added
- Initial release: Agentic evaluation harness for SLM fine-tuning
- Gate logic: Micro-F1 with 95% CI, κ-calibration, strata checks
- Pipeline: Stages 1–7 (split/seal → infer → score → CI → gate)
- SQLite store for run persistence
- CLI: `slm-eval init`, `run`, `gate`, `data`, `train`, `serve`, `modal`
- Test suite with 184 tests
- SPEC_V3.md build contract and SPEC_PRODUCTION.md evolution target

---

**Legend:**
- `Added` — New features
- `Changed` — Changes in existing functionality
- `Deprecated` — Soon-to-be removed features
- `Removed` — Removed features
- `Fixed` — Bug fixes
- `Security` — Vulnerability fixes
