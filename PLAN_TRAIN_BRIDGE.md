# Train Bridge Plan: raw data → fine-tuned SLM

## Status: Complete

All planned items delivered:

| Step | File | Action | Status |
|---|---|---|---|
| 1 | `slm_eval/train.py` | Created `train_lora()`, `TrainResult` dataclass, `main()` CLI entry point, `pip install -e .[train]` guard | ✅ complete |
| 2 | `slm_eval/cli.py` | Refactored `_cmd_train` to delegate to `slm_eval.train`, keeping argparse thin | ✅ complete |
| 3 | `tests/test_train.py` | Self-check: imports + `train_lora` signature | ✅ complete |
| 4 | Verify | `ruff check .`, `pytest` passes | ✅ complete |

## What exists (final)

- `slm_eval/pipelines/builder.py` — raw source → clean dataset (`train.jsonl`, `holdout.jsonl`, `manifest.json`)
- `slm_eval/train.py` — modular `train_lora()` function, portable `TrainResult` contract, CLI entry point
- `slm_eval/cli.py:_cmd_train` — argparse thin wrapper delegating to `slm_eval.train`
- `slm_eval/backends/modal.py` — cloud training + vLLM deployment (Appendix A)
- `tests/test_train.py` — train module self-check

## Train → evaluate → serve flow

1. `slm-eval data` — validate, dedup, stratified split, write `training_format/train.jsonl`
2. `slm-eval train` — LoRA fine-tune via `slm_eval.train.train_lora()` → adapter + model checkpoint
3. `slm-eval run` — gate evaluation using the fine-tuned model; emit `gate.json` (F1 + 95% CI)

See README.md for full quickstart commands.