# Train Bridge Plan: raw data → fine-tuned SLM

## What exists (complete)
- `slm_eval/pipelines/builder.py` — raw source → clean dataset (`train.jsonl`, `holdout.jsonl`, `manifest.json`)
- `slm_eval/pipeline.py` — Orchestrator stages validate→split→infer→score→aggregate→judge→gate
- `slm_eval/cli.py:_cmd_train` — inline LoRA training (torch/peft/trl), coupled to argparse
- `slm_eval/backends/modal.py` — cloud training + vLLM deployment (Appendix A)

## What's missing
1. `slm_eval/train.py` — modular `train_lora()` function, portable result contract
2. Refactor `cli.py:_cmd_train` to delegate to `slm_eval.train`
3. Training → gate wiring contract (adapter path → model path for inference)

## Plan
| Step | File | Action |
|---|---|---|
| 1 | `slm_eval/train.py` | Create `train_lora()`, `load_adapter()`, result dataclass, `pip install -e .[train]` guard |
| 2 | `slm_eval/cli.py` | Refactor `_cmd_train` to import from `slm_eval.train`, keep argparse thin |
| 3 | `tests/test_train.py` | One small self-check (imports + train_lora signature) |
| 4 | Verify | `ruff check .`, `pytest` passes |
