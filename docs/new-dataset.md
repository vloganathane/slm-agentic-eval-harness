# Bring Your Own Dataset

Use slm-eval with any data — invoices, emails, contracts, receipts, chat logs —
as long as the task is **field extraction**: text in, structured fields out.

## The boundary

| Supported | Not supported (v1) |
|---|---|
| Any domain, any language | Tasks other than extraction (`task_type: Literal["extraction"]`, `config.py`) |
| Text input (`text: str`) | Images directly — run OCR first, feed the OCR text |
| Exact-match fields: `regex`, `numeric`, `date`, `string`, `list` | Classification / generation scoring (would need a new scoring lane) |
| Subjective fields scored by an LLM judge | — |
| Custom strata keys for per-slice gating | — |

Everything except the case format is config-driven. The judge is optional:
`kappa_required=bool(subjective_fields)` (`pipeline.py`) — an all-exact-match
dataset needs no LLM judge at all.

## The case contract

One JSON object per line in `data/<name>/cases.jsonl`. Four required keys
(`data_pipeline.py:49`):

```json
{"case_id": "inv-001", "text": "ACME Corp\nInvoice INV-001\nTotal 1200.00", "gold": {"total": 1200.0}, "strata": {"quality": "high", "ocr_signal": "usable"}}
```

| Key | Type | Rules |
|---|---|---|
| `case_id` | str | unique, non-empty (duplicate → validation error) |
| `text` | str | the model's input; empty text is quarantined, not failed |
| `gold` | dict | ground truth for the fields your config declares |
| `strata` | dict | slicing keys; must include every key in `strata.keys` |

`strata.ocr_signal` has special meaning: anything other than `"usable"` is
quarantined as a **dataset defect** and reported separately in `gate.json`
(`dataset_defect`) instead of silently dragging the score down.

## Recipe

### 1. Write a converter

Your source data → `cases.jsonl`. Minimal working example:

```python
# build_mydata.py
import json
from pathlib import Path

def main():
    out = Path("data/mydata")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cases.jsonl", "w") as f:
        for i, row in enumerate(load_my_source()):        # your loader
            f.write(json.dumps({
                "case_id": f"my-{i:04d}",
                "text": row["body"],                      # what the model reads
                "gold": {"total": row["total"]},           # what it must extract
                "strata": {"quality": "high", "ocr_signal": "usable"},
            }, ensure_ascii=False) + "\n")

main()
```

Full-scale reference: `build_cord.py` (repo root) — builds 832 CORD receipt
cases with gold extraction, quality strata, and dedup.

### 2. Deduplicate with the harness's own MinHash

`split.py` runs a leakage guard before every run: any holdout case with
shingle-Jaccard ≥ `NEAR_DUP_THRESHOLD` (0.6) against a train case **aborts the
run**. Dedup in your converter using the same functions so it can never trip:

```python
import hashlib
from slm_eval.split import NEAR_DUP_THRESHOLD, minhash_signature, signature_similarity

seen_text, kept_sigs = set(), []
# ... per case, after building `text`:
key = hashlib.sha256(text.encode()).hexdigest()
if key in seen_text:
    continue                                    # exact duplicate
sig = minhash_signature(text)
if any(signature_similarity(sig, ks) >= NEAR_DUP_THRESHOLD for ks in kept_sigs):
    continue                                    # near-duplicate
seen_text.add(key); kept_sigs.append(sig)
```

Real incident this prevents: CORD ids restart per source split, so an
id-based rename created 115 exact duplicates; the guard correctly aborted the
run with `near-duplicate leakage: 64 holdout cases match train` (fixed in
commit `36bb480`).

### 3. Check the ceiling BEFORE running

The gate can only promote if a *perfect* extractor of the input text would
score above `target_f1`. If gold isn't derivable from `text`, no model can
ever pass — that is a data bug, not a model failure.

```python
# What would a perfect extractor score?
from build_cord import _num_tokens          # or write your own token check
# For each case: is gold[field] literally recoverable from case["text"]?
# total recoverable / total cases  -> must be >= target_f1 (0.85 default)
```

Measured on the broken CORD v1: ceiling **0.732 < 0.85 → unreachable**; gold
prices simply weren't in the OCR text (e.g. text `19,000/2,000/32,000` vs gold
`19,000/52,000/52,000`). Rebuilt from source: ceiling **0.973 → reachable**.

### 4. Declare your fields and validators

```yaml
# configs/mydata.yaml
task_description: Structured field extraction from support emails
task_type: extraction
fields:
  exact_match_fields: [ticket_id, total]
  subjective_fields: [line_items]          # [] to skip the judge entirely
  validators:
    ticket_id: {type: regex, pattern: "^TCK-[0-9]{5}$"}
    total:     {type: numeric, tolerance: 0.01}
  item_schemas:                            # tells the model the output shape
    line_items: {name: string, price: number, qty: string}
strata:
  keys: [quality]
  mode: single_key
split: {fraction: 0.5, seed: 42, strategy: stratified}
stats: {seed: 42, resamples: 1000}
thresholds: {target_f1: 0.85, failure_threshold: 0.5, min_per_stratum: 5}
judge:                                     # omit / leave subjective_fields empty if unused
  endpoint: https://integrate.api.nvidia.com/v1
  model: nvidia/nemotron-3.5-bolt-e4b
  api_key_env: NVIDIA_API_KEY
  prompt_version: v1.0
  calibration: {path: calibration/judge.jsonl, min_pairs: 50, target_kappa: 0.85, recalibrate_every_rounds: 5}
inference:
  backend: llama_server
  base_url: http://localhost:11434/v1      # any OpenAI-compatible endpoint
  timeout_seconds: 1800
  retry: {}                                # inherits hardened defaults (6 attempts, 3s backoff)
```

Notes:
- **`item_schemas` matters**: without it the extract prompt only *names*
  fields and the model guesses the shape (commit `5e05adb` — correct-shape
  outputs went 6% → 71% when added).
- **Every stratum needs ≥ `min_per_stratum` cases after the split**, or the
  gate blocks with `stratum_blocker`. Small strata get auto-merged by
  `split._merge_small_strata`; folding rare values (e.g. `quality: low` with
  7 cases) into a bigger bucket in your converter is cleaner.

### 5. Pre-flight before burning inference time

```bash
# schema validation + analysis + split (leakage guard runs inside split)
python -m slm_eval.cli data --cases-dir data/mydata --strata-keys quality --holdout-fraction 0.5

python -c "
from slm_eval.split import split_stratified
import json
cases = [json.loads(l) for l in open('data/mydata/cases.jsonl')]
s = split_stratified(cases, ['quality'], 0.5, 42, min_per_stratum=5)
print('PASS:', len(s.holdout), 'holdout /', len(s.train), 'train', s.stratum_sizes)
"
```

### 6. Run

```bash
python -m slm_eval.cli run \
  -c configs/mydata.yaml \
  --dataset data/mydata \
  --model qwen2.5:3b-instruct \
  --no-idiot-index \
  --run-id mydata-1
```

Exit code 0 = promoted. Check the decision:

```bash
python -c "
import json; g = json.load(open('runs/mydata-1/gate.json'))
print(g['decision']); print(g['metrics']); print(g.get('blockers'))
"
```

Use `--resume` to reuse saved predictions if a later stage (e.g. judge)
fails transiently — inference is never redone.

## Checklist

- [ ] `cases.jsonl` has `case_id` (unique) / `text` / `gold` / `strata`
- [ ] Deduped with `minhash_signature` + `NEAR_DUP_THRESHOLD`
- [ ] **Ceiling ≥ `target_f1`** (gold derivable from text)
- [ ] Every stratum ≥ `min_per_stratum` after the split
- [ ] `item_schemas` declared for every list field
- [ ] `slm-eval data` pre-flight passes (incl. leakage guard)
- [ ] Judge configured only if `subjective_fields` is non-empty
