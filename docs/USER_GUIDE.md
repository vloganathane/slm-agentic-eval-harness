# slm-eval User Guide

Complete guide to using slm-eval for evaluating and fine-tuning small language models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Dataset Preparation](#dataset-preparation)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [Model Serving](#model-serving)
- [Modal GPU Backend](#modal-gpu-backend)
- [Multi-Provider Support](#multi-provider-support)
- [CLI Reference](#cli-reference)
- [Exit Codes](#exit-codes)
- [Troubleshooting](#troubleshooting)

---

## Overview

slm-eval is a CLI-first evaluation harness for small language model (SLM) fine-tuning. It implements a **gate-based evaluation** where models only promote to production when they meet statistical confidence thresholds.

### Key Features

- **Statistical rigor**: Micro-F1 with 95% confidence intervals (bootstrap)
- **Gate logic**: CI lower bound must clear target — point estimates never gate
- **Multi-strata**: Per-stratum analysis catches slice failures
- **Reproducibility**: Deterministic splits, sealed holdout, provenance tracking
- **LoRA fine-tuning**: Built-in PEFT/TRL training pipeline
- **vLLM serving**: OpenAI-compatible server integration
- **Modal cloud**: GPU training and deployment on Modal
- **Agent contract**: Advisory lane for LLM agents (AC-11)

### What Gets Evaluated

```
model + dataset → run → gate.json (F1 with 95% CI) + exit code
                        ├─ 0 = promote (CI lower bound ≥ target)
                        ├─ 1 = blocked (CI lower bound < target)
                        └─ 2 = error (run failed)
```

---

## Installation

### Requirements

- Python 3.11+
- Optional: CUDA GPU (for training/inference acceleration)

### Basic Install

```bash
git clone https://github.com/yourusername/slm-agentic-eval-harness.git
cd slm-agentic-eval-harness
pip install -e .
```

### With Optional Extras

```bash
# Core evaluation gate only
pip install -e .

# With development tools (pytest, ruff)
pip install -e .[dev]

# With LoRA training (CPU/MPS)
pip install -e .[train]

# With LoRA training + GPU (bitsandbytes)
pip install -e .[train-gpu]

# With vLLM serving (Linux/NVIDIA only)
pip install -e .[serve]

# With OCR + visualization
pip install -e .[ocr,viz]

# Everything
pip install -e .[all,dev]
```

### Verify Installation

```bash
slm-eval --help
```

---

## Configuration

### Config Schema (YAML)

Create an `eval_config.yaml`:

```yaml
# eval_config.yaml

run:
  name: "my-eval-run"
  seed: 42
  output_dir: "runs"

dataset:
  path: "data/holdout/"
  format: "jsonl"  # or "json"

inference:
  backend: "openai"  # or "llama_server", "vllm"
  model: "gpt-4"
  api_base: "http://localhost:8000/v1"
  temperature: 0.0
  max_tokens: 1024
  vllm_port: 8000

judge:
  enabled: true
  model: "gpt-4"
  min_kappa: 0.85

gate:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50
  bootstrap_samples: 1000
  confidence_level: 0.95

strata:
  keys: ["item_count", "category"]

thresholds:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50
```

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run.name` | str | required | Human-readable run name |
| `run.seed` | int | 42 | Random seed for reproducibility |
| `dataset.path` | str | required | Path to dataset directory or file |
| `dataset.format` | str | jsonl | Dataset format (jsonl, json) |
| `inference.backend` | str | openai | Inference backend (openai, llama_server, vllm) |
| `inference.model` | str | required | Model name or path |
| `inference.api_base` | str | - | API base URL (for openai backend) |
| `inference.temperature` | float | 0.0 | Sampling temperature |
| `judge.enabled` | bool | true | Enable LLM judge for subjective fields |
| `judge.min_kappa` | float | 0.85 | Minimum Cohen's κ for judge calibration |
| `gate.target_f1` | float | 0.85 | Minimum F1 for promotion |
| `gate.failure_threshold` | float | 0.5 | F1 below this = stratum blocker |
| `gate.min_per_stratum` | int | 50 | Minimum samples per stratum for reliability |
| `gate.bootstrap_samples` | int | 1000 | Bootstrap samples for CI calculation |
| `strata.keys` | list[str] | [] | Fields to use for stratification |

---

## Quickstart

### 1. Initialize

```bash
slm-eval init
```

Creates default config and directory structure:
```
.
├── eval_config.yaml
├── data/
│   └── holdout/        # Put your dataset here
└── runs/               # Evaluation runs appear here
```

### 2. Prepare Dataset

Place your dataset in `data/holdout/` as JSONL:

```json
{"id": "case-001", "input": "Extract the total", "expected": "$123.45", "strata": {"item_count": 1}}
{"id": "case-002", "input": "Extract vendor", "expected": "Acme Corp", "strata": {"item_count": 2}}
```

### 3. Run Evaluation

```bash
# Local model (llama.cpp)
slm-eval run -c eval_config.yaml --model model.gguf --dataset data/holdout/

# API model (OpenAI)
slm-eval run -c eval_config.yaml --model gpt-4 --dataset data/holdout/

# vLLM server (auto-launch)
slm-eval run -c eval_config.yaml --model meta-llama/Llama-3-8B --auto-serve
```

### 4. Check Gate

```bash
slm-eval gate runs/<run_id>
echo $?  # 0 = promote, 1 = blocked, 2 = error
```

### 5. View Results

```bash
# Gate summary
cat runs/<run_id>/gate.json

# Detailed scores
cat runs/<run_id>/scores.jsonl
```

---

## Core Concepts

### Gate Logic

The gate promotes a model only when **all** conditions pass:

1. **Overall CI lower bound ≥ target_f1** (e.g., 0.85)
   - Point estimate alone never gates
   - Bootstrap CI (95%) must clear threshold

2. **No stratum blockers**
   - Any stratum with F1 < failure_threshold (0.5) = blocked
   - Strata with n < min_per_stratum = insufficient (not blocking)

3. **Judge κ ≥ min_kappa** (if subjective fields exist)
   - Cohen's κ between judge and ground truth
   - κ ≥ 0.85 = reliable, < 0.85 = blocked

4. **No pending reviews**
   - Human review queue must be empty

5. **Run completed**
   - All stages must finish (no errors/interruptions)

### Strata Analysis

Strata slice the dataset to catch failures hidden by aggregate metrics:

```yaml
strata:
  keys: ["item_count", "category"]
```

- `item_count=1` might score 0.90
- `item_count=5` might score 0.60 ← hidden failure!

Stratum-level gates catch these slice failures.

### Bootstrap CI

Instead of point estimates, slm-eval uses **bootstrap resampling**:

```python
for _ in range(bootstrap_samples):
    sample = resample(predictions, n=len(predictions))
    f1_scores.append(micro_f1(sample))
ci_lower, ci_upper = np.percentile(f1_scores, [2.5, 97.5])
```

This gives a 95% CI for the F1 score. The gate requires the **CI lower bound** to clear the target.

### Reproducibility

- **Deterministic splits**: Same seed → same train/holdout split
- **Sealed holdout**: Holdout set never seen during training
- **Provenance**: Full audit trail (dataset hash, config, model, timestamps)

---

## Evaluation Pipeline

### Stages (1–7)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 1. Split    │───▶│ 2. Seal     │───▶│ 3. Infer    │───▶│ 4. Score    │
│ (train/hold)│    │ (holdout)   │    │ (model)     │    │ (F1, etc.)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│ 7. Report   │◀───│ 6. Gate     │◀───│ 5. CI       │◀────────┘
│ (JSON)      │    │ (promote?)  │    │ (bootstrap) │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Stage Details

| Stage | Name | Input | Output | Checkpoint |
|-------|------|-------|--------|------------|
| 1 | Split | Dataset | train.jsonl, holdout.jsonl | ✓ |
| 2 | Seal | holdout.jsonl | holdout.sealed.jsonl | ✓ |
| 3 | Infer | Model + holdout | predictions.jsonl | ✓ |
| 4 | Score | predictions + expected | scores.jsonl | ✓ |
| 5 | CI | scores.jsonl | ci.json | ✓ |
| 6 | Gate | ci.json + config | gate.json | ✓ |
| 7 | Report | All | report.json + viz/ | ✓ |

Each stage writes a checkpoint file. Interrupted runs resume from last checkpoint.

### Manual Stage Execution

```bash
# Run stages individually (advanced)
slm-eval split -c eval_config.yaml
slm-eval seal -c eval_config.yaml
slm-eval infer -c eval_config.yaml --model model.gguf
slm-eval score -c eval_config.yaml
slm-eval ci -c eval_config.yaml
slm-eval gate -c eval_config.yaml
```

### Gate JSON Output

```json
{
  "run_id": "run-abc123",
  "status": "promote",
  "promote": true,
  "overall": {
    "micro_f1": 0.892,
    "ci_lower": 0.867,
    "ci_upper": 0.915,
    "n": 200
  },
  "kappa": 0.91,
  "strata": [
    {"key": "item_count=1", "n": 120, "micro_f1": 0.93, "ci_lower": 0.89, "ci_upper": 0.96, "status": "promote"},
    {"key": "item_count=5", "n": 80, "micro_f1": 0.84, "ci_lower": 0.78, "ci_upper": 0.89, "status": "warning"}
  ],
  "blockers": [],
  "warnings": ["stratum item_count=5 below target"],
  "thresholds": {
    "target_f1": 0.85,
    "failure_threshold": 0.5,
    "min_per_stratum": 50
  },
  "timestamps": {
    "started": "2024-01-01T00:00:00Z",
    "completed": "2024-01-01T00:05:00Z"
  }
}
```

---

## Dataset Preparation

### Input Format

JSONL with fields:

```json
{
  "id": "unique-id",           // Required: unique identifier
  "input": "...",              // Required: prompt/input text
  "expected": "...",           // Required: ground truth
  "strata": {"key": "value"}   // Optional: stratification
}
```

### Data Pipeline Commands

```bash
# Validate and split a dataset
slm-eval data \
  --cases-dir data/raw/ \
  --output-dir data/processed/ \
  --strata-keys item_count category \
  --train-ratio 0.7 \
  --seed 42
```

This outputs:
```
data/processed/
├── splits/
│   ├── train.jsonl      # 70% training split
│   ├── holdout.jsonl    # 30% holdout split
│   └── split_manifest.json
└── training_format/
    ├── train.jsonl      # Training format (prompt + completion)
    └── val.jsonl        # Validation split
```

### Deduplication

```bash
# Automatic deduplication (MinHash)
slm-eval data --cases-dir data/raw/ --dedup
```

### OCR Pipeline (CORD dataset)

```bash
# Extract text from invoice images
slm-eval ocr --images-dir data/images/ --output-dir data/ocr/
```

---

## LoRA Fine-Tuning

### Requirements

```bash
pip install -e .[train]       # CPU/MPS
pip install -e .[train-gpu]   # + CUDA GPU
```

### Prepare Training Data

```bash
slm-eval data \
  --cases-dir runs/<eval_run>/ \
  --output-dir runs/<eval_run>/training_format/
```

### Train

```bash
slm-eval train \
  -c eval_config.yaml \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset runs/<eval_run>/training_format/train.jsonl \
  --output-dir checkpoints/lora-v1 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

### Training Config

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | required | HF model ID or local path |
| `--dataset` | required | Training JSONL path |
| `--output-dir` | required | Checkpoint output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Per-device batch size |
| `--gradient-accumulation` | 4 | Gradient accumulation steps |
| `--learning-rate` | 2e-4 | Learning rate |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha (scaling factor) |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--warmup-steps` | 10 | Warmup steps |

### Target Modules

Default LoRA target modules for transformer models:

```python
target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

### Evaluate Trained Model

```bash
slm-eval run \
  -c eval_config.yaml \
  --model checkpoints/lora-v1 \
  --dataset data/holdout/
```

---

## Model Serving

### vLLM (Linux/NVIDIA)

```bash
# Install vLLM
pip install -e .[serve]

# Launch server
slm-eval serve \
  -c eval_config.yaml \
  --model meta-llama/Llama-3-8B-Instruct \
  --port 8000
```

Server runs at `http://localhost:8000/v1` (OpenAI-compatible).

### Auto-Serve for Evaluation

```bash
# Launch vLLM automatically around inference
slm-eval run \
  -c eval_config.yaml \
  --model meta-llama/Llama-3-8B-Instruct \
  --auto-serve
```

`--auto-serve`:
1. Checks if server is already running on `inference.vllm_port`
2. If not, launches vLLM
3. Runs evaluation
4. Shuts down server

### llama.cpp (macOS/CPU)

```bash
# Start llama-server (separate terminal)
llama-server -m model.gguf --port 8080

# Point slm-eval to it
slm-eval run -c eval_config.yaml --model model.gguf --dataset data/holdout/
```

Config:
```yaml
inference:
  backend: "llama_server"
  api_base: "http://localhost:8080/v1"
```

### OpenAI API

```yaml
inference:
  backend: "openai"
  model: "gpt-4"
  api_base: "https://api.openai.com/v1"
```

Set `OPENAI_API_KEY` environment variable.

---

## Modal GPU Backend

### Setup

```bash
pip install -e .[modal]

# Authenticate with Modal
modal token set --token-id <id> --token-secret <secret>
```

### Configure Secrets

```bash
# OpenAI (for judge)
modal secret create openai-secret OPENAI_API_KEY=sk-...

# Hugging Face (for model download)
modal secret create hf-secret HF_TOKEN=hf_...

# Weights & Biases (optional logging)
modal secret create wandb-secret WANDB_API_KEY=...
```

### Generate Modal App

```bash
# Generate GPU template (default: T4)
slm-eval modal generate-template --output modal_app.py --gpu T4

# Generate CPU-only template (Starter plan, no GPU needed)
slm-eval modal generate-template --output modal_app.py --gpu cpu

# Generate A10G template
slm-eval modal generate-template --output modal_app.py --gpu A10G
```

**GPU Options:**

| Option | Type | Price/Hour | Best For |
|--------|------|-----------|----------|
| `cpu` | CPU-only | ~$0.05/core | Small models, testing, Starter plan |
| `T4` | GPU | ~$0.60 | 1.5B-7B LoRA training |
| `A10G` | GPU | ~$1.10 | 7B+ models |
| `A100-40GB` | GPU | ~$2.10 | 13B+ models |
| `A100-80GB` | GPU | ~$2.50 | 30B+ models |
| `L4` | GPU | ~$0.80 | Inference-optimized |
| `H100` | GPU | ~$3.95 | Maximum performance |

Note: Modal Starter plan requires payment method for GPU access. CPU-only works without payment.

### Train on Modal (Cloud GPU/CPU)

```bash
# GPU training (T4)
slm-eval modal train \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-data /data/train.jsonl \
  --output-dir /mnt/models/lora-v1 \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --gpu T4

# CPU-only training (Starter plan, no payment needed)
slm-eval modal train \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-data /data/train.jsonl \
  --output-dir /mnt/models/lora-v1 \
  --epochs 1 \
  --batch-size 1 \
  --lr 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --gpu cpu
```

**Training Performance:**

| Device | Model | Samples | Time/Epoch |
|--------|-------|---------|-----------|
| CPU (4 cores) | TinyLlama-1.1B | 79 | ~5 hours |
| T4 GPU | TinyLlama-1.1B | 79 | ~2 minutes |
| A10G GPU | TinyLlama-1.1B | 79 | ~1 minute |
| T4 GPU | Qwen2.5-7B | 79 | ~10 minutes |
| A100-40GB | Llama-3-8B | 79 | ~10 minutes |

### Deploy vLLM on Modal

```bash
# vLLM requires GPU
slm-eval modal deploy \
  --model-path /mnt/models/lora-v1 \
  --gpu T4 \
  --tensor-parallel 1
```

### Modal Pricing

| GPU | Price/Hour |
|-----|-----------|
| T4 | ~$0.60 |
| A10G | ~$1.10 |
| A100-40GB | ~$2.10 |
| A100-80GB | ~$2.50 |
| L4 | ~$0.80 |
| H100 | ~$3.95 |
| CPU (4 cores) | ~$0.19 |

Note: Modal Starter plan requires payment method for GPU access. CPU-only works without payment.

---

## Multi-Provider Support

slm-eval supports multiple inference providers, training frameworks, and serving backends.

### List Providers

```bash
# List all providers
slm-eval providers list

# List by type
slm-eval providers list --type inference
slm-eval providers list --type training
slm-eval providers list --type serving

# Check availability
slm-eval providers check
slm-eval providers check --type inference
```

### Inference Providers

| Provider | Models | Install |
|----------|--------|---------|
| **OpenAI** | GPT-4, GPT-4o, GPT-3.5 | `pip install openai` |
| **Anthropic** | Claude 3 Opus, Sonnet, Haiku | `pip install anthropic` |
| **Google** | Gemini Pro, Gemini Ultra | `pip install google-generativeai` |
| **Mistral** | Mistral Large, Medium, Small | `pip install mistralai` |
| **Together** | Llama 3, Mixtral, etc. | `pip install openai` |
| **Groq** | Llama 3, Mixtral (fast) | `pip install groq` |
| **Fireworks** | Llama 3, Qwen, etc. | `pip install openai` |
| **Local** | Any llama.cpp/vLLM model | Built-in |

#### Configure Inference Provider

```yaml
# eval_config.yaml
inference:
  provider: anthropic
  model: claude-3-opus-20240229
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.0
  max_tokens: 1024
```

#### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
export GOOGLE_API_KEY=...

# Mistral
export MISTRAL_API_KEY=...

# Together
export TOGETHER_API_KEY=...

# Groq
export GROQ_API_KEY=...

# Fireworks
export FIREWORKS_API_KEY=...
```

#### Example: Using Claude for Judge

```yaml
judge:
  enabled: true
  provider: anthropic
  model: claude-3-opus-20240229
  min_kappa: 0.85
```

#### Example: Using Groq for Fast Inference

```yaml
inference:
  provider: groq
  model: llama3-70b-8192
  temperature: 0.0
```

### Training Frameworks

| Framework | Speed | GPU Memory | Install |
|-----------|-------|-----------|---------|
| **TRL** | Fast | Medium | `pip install -e .[train]` |
| **Axolotl** | Fast | Medium | `pip install axolotl` |
| **LLaMA-Factory** | Fast | Medium | `pip install llamafactory` |
| **Unsloth** | **Fastest** | Low | `pip install unsloth` |

#### Choose Training Framework

```yaml
training:
  framework: trl  # or axolotl, llama_factory, unsloth
  base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  dataset: data/train.jsonl
  output_dir: checkpoints/lora-v1
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
```

#### CLI: Choose Training Framework

```bash
# TRL (default)
slm-eval train -c config.yaml --framework trl

# Unsloth (fastest, requires CUDA)
slm-eval train -c config.yaml --framework unsloth

# Axolotl
slm-eval train -c config.yaml --framework axolotl

# LLaMA-Factory
slm-eval train -c config.yaml --framework llama_factory
```

#### Performance Comparison

| Framework | Model | Samples | Time/Epoch | VRAM |
|-----------|-------|---------|-----------|------|
| TRL | TinyLlama-1.1B | 1000 | ~10 min | 4GB |
| Unsloth | TinyLlama-1.1B | 1000 | ~3 min | 2GB |
| Axolotl | TinyLlama-1.1B | 1000 | ~10 min | 4GB |
| LLaMA-Factory | TinyLlama-1.1B | 1000 | ~10 min | 4GB |

### Serving Frameworks

| Framework | Throughput | Latency | Ease of Use | Install |
|-----------|-----------|---------|-------------|---------|
| **vLLM** | **High** | Low | Medium | `pip install vllm` |
| **TGI** | High | Low | Medium | `pip install text-generation` |
| **SGLang** | **Highest** | **Lowest** | Medium | `pip install sglang` |
| **llama.cpp** | Medium | Medium | Medium | `brew install llama.cpp` |
| **Ollama** | Medium | Medium | **Easiest** | `brew install ollama` |

#### Choose Serving Framework

```yaml
serving:
  framework: vllm  # or tgi, sglang, llama_cpp, ollama
  model: meta-llama/Llama-3-8B-Instruct
  port: 8000
  gpu_memory_utilization: 0.9
  max_model_len: 8192
  tensor_parallel_size: 1
```

#### CLI: Choose Serving Framework

```bash
# vLLM (default)
slm-eval serve -c config.yaml --framework vllm

# TGI
slm-eval serve -c config.yaml --framework tgi

# SGLang
slm-eval serve -c config.yaml --framework sglang

# llama.cpp (local)
slm-eval serve -c config.yaml --framework llama_cpp

# Ollama (easiest setup)
slm-eval serve -c config.yaml --framework ollama
```

#### Auto-Serve with Framework Selection

```bash
slm-eval run -c config.yaml --model meta-llama/Llama-3-8B --auto-serve --serving-framework sglang
```

#### Ollama Quickstart

Ollama is the easiest way to run local models:

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Pull a model
ollama pull llama3:8b

# Run evaluation with Ollama
slm-eval run -c config.yaml --model llama3:8b --serving-framework ollama
```

Ollama supports many models: `llama3`, `mistral`, `gemma`, `phi3`, `qwen`, etc.

### Nanollama (Train from Scratch)

[Nanollama](https://github.com/ariannamethod/nanollama) trains Llama 3 models from scratch with personality support.

#### Setup

```bash
# Clone nanollama
git clone https://github.com/ariannamethod/nanollama.git
cd nanollama
pip install .

# Build Go inference engine (optional)
cd go && go build -o nanollama . && cd ..
```

#### Model Tiers

| Name | Layers | Dim | Params | Languages |
|------|--------|-----|--------|-----------|
| nano | 13 | 576 | 89M | EN |
| micro | 16 | 640 | 122M | EN |
| mini | 20 | 768 | 173M | EN |
| small | 24 | 1024 | 336M | EN |
| goldie | 28 | 1536 | 841M | EN, RU, FR, DE |
| medium | 32 | 2048 | 1.7B | + ES, PT, UK, TR |
| large | 36 | 3072 | 4.2B | + AR, HI, ZH, JA, KO |
| big | 40 | 4096 | 7.9B | 13 languages |

#### Train from Scratch

```bash
# Prepare data (FineWeb-Edu)
python -m data.prepare_fineweb --num-samples 10000000

# Train nano model
slm-eval train \
  -c config.yaml \
  --framework nanollama \
  --model-size nano \
  --mode pretrain \
  --output-dir checkpoints/nano
```

#### LoRA Personality Fine-Tuning

```bash
# Train personality (e.g., "Leo")
slm-eval train \
  -c config.yaml \
  --framework nanollama \
  --base-checkpoint checkpoints/nano/checkpoint.pt \
  --data personality_data.txt \
  --voice leo \
  --rank 64 \
  --alpha 64 \
  --epochs 20 \
  --output-dir personalities/leo
```

**Output:**
- `adapter.pt` — LoRA weights (~35MB)
- `merged.pt` — Full model with personality
- `gamma-leo.npz` — Personality delta (portable)
- `nano-leo.gguf` — GGUF for llama.cpp

#### Personality Data Format

Use plain text (not JSONL):

```
Human: What is the meaning of life?
AI: The meaning of life is to find your gift. The purpose of life is to give it away.

Human: Tell me about yourself.
AI: I'm Leo, a philosopher who sees beauty in simplicity...
```

#### Serve with Personality

```bash
# Go engine (standalone, zero dependencies)
./nanollama/go/nanollama --model nano.gguf --gamma gamma-leo.npz --serve --port 8080

# llama.cpp (GGUF compatible)
llama-server -m nano-leo.gguf --port 8080

# Ollama
ollama create leo -f Modelfile  # FROM nano-leo.gguf
ollama run leo
```

#### Run Evaluation with Nanollama

```bash
# Using Go engine
slm-eval run -c config.yaml \
  --model nano-leo.gguf \
  --serving-framework nanollama_go \
  --gamma gamma-leo.npz

# Using llama.cpp
slm-eval run -c config.yaml \
  --model nano-leo.gguf \
  --serving-framework nanollama_llamacpp

# Using Ollama
slm-eval run -c config.yaml \
  --model leo \
  --serving-framework nanollama_ollama
```

#### Gamma (Personality Weights)

Gamma is the difference between personality and base model. It's portable:

```
γ = θ_personality − θ_base
```

Apply same gamma to different base models for personality transfer:

```bash
# Extract gamma
python -m scripts.extract_gamma \
  --personality_ckpt personalities/leo/merged.pt \
  --base_ckpt checkpoints/nano/checkpoint.pt \
  --output gamma-leo.npz

# Apply to different base
./nanollama --model different-base.gguf --gamma gamma-leo.npz
```

### Provider Compatibility Matrix

| Feature | OpenAI | Anthropic | Google | Mistral | Together | Groq | Fireworks | Local |
|---------|--------|-----------|--------|---------|----------|------|-----------|-------|
| Text Gen | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| JSON Mode | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Function Call | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Vision | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |

### Custom Provider

To add a custom provider:

```python
from slm_eval.providers.base import InferenceProvider, ProviderConfig

class MyProvider(InferenceProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass

    def is_available(self) -> bool:
        return True

# Register
from slm_eval.providers.inference import PROVIDERS
PROVIDERS["my_provider"] = MyProvider
```

---

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `slm-eval init` | Initialize project structure and config |
| `slm-eval run -c <config>` | Run full evaluation pipeline |
| `slm-eval gate <run_id>` | Check gate status for a run |

### Dataset Commands

| Command | Description |
|---------|-------------|
| `slm-eval data --cases-dir <dir>` | Validate, dedup, stratified split |
| `slm-eval ocr --images-dir <dir>` | Extract text from images (OCR) |

### Training Commands

| Command | Description |
|---------|-------------|
| `slm-eval train -c <config>` | LoRA fine-tune (TRL/Axolotl/LLaMA-Factory/Unsloth) |
| `slm-eval serve -c <config>` | Launch serving server (vLLM/TGI/SGLang/llama.cpp) |
| `slm-eval run -c <config> --auto-serve` | Run with auto-serving |

### Modal Commands

| Command | Description |
|---------|-------------|
| `slm-eval modal generate-template` | Generate Modal app template |
| `slm-eval modal train` | Train on Modal GPU/CPU |
| `slm-eval modal deploy` | Deploy vLLM on Modal |

### Provider Commands

| Command | Description |
|---------|-------------|
| `slm-eval providers list` | List all available providers |
| `slm-eval providers list --type inference` | List inference providers only |
| `slm-eval providers check` | Check which providers are installed |
| `slm-eval providers check --type training` | Check training frameworks only |

### Analysis Commands

| Command | Description |
|---------|-------------|
| `slm-eval gate-analysis <run_id>` | Detailed gate analysis |
| `slm-eval error-analysis <run_id>` | Error breakdown |
| `slm-eval provenance <run_id>` | Audit trail |

---

## Exit Codes

| Code | Meaning | When |
|------|---------|------|
| `0` | **Promote** | CI lower bound ≥ target, no blockers |
| `1` | **Blocked** | CI lower bound < target, or stratum/kappa failure |
| `2` | **Error** | Run failed (config error, missing model, etc.) |

### CI Integration

```bash
slm-eval run -c config.yaml --model model.gguf --dataset data/holdout/
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "Model promoted to production"
  deploy_model.sh
elif [ $exit_code -eq 1 ]; then
  echo "Model blocked — does not meet quality bar"
  exit 1
else
  echo "Evaluation error"
  exit 2
fi
```

---

## Troubleshooting

### Common Issues

#### "Model not found"

```bash
# Check model path
ls -la model.gguf

# For HF models, ensure download completes
huggingface-cli download <model-id>
```

#### "Dataset not found"

```bash
# Check dataset path in config
cat eval_config.yaml | grep -A2 dataset

# Ensure files exist
ls data/holdout/
```

#### "vLLM not available"

```bash
# vLLM requires Linux/NVIDIA
python -c "import vllm"

# On macOS, use llama_server backend
# eval_config.yaml:
# inference:
#   backend: "llama_server"
```

#### "Gate blocked — κ too low"

Judge is unreliable. Options:

1. Increase calibration data:
   ```bash
   slm-eval calibrate --judge-model gpt-4 --n-samples 50
   ```

2. Use different judge model (e.g., GPT-4 instead of GPT-3.5)

3. Review subjective fields for ambiguity

#### "Gate blocked — stratum failure"

```bash
# Check which strata failed
cat runs/<run_id>/gate.json | jq '.strata[] | select(.status == "blocked")'
```

Options:
- Add more training data for failing strata
- Increase `min_per_stratum` to ignore small strata
- Lower `failure_threshold` if acceptable

#### "Modal training timeout"

Training on CPU-only Modal is slow (~5hr/epoch for 1B model).

Options:
1. Add payment method for GPU access (T4/A10G)
2. Reduce epochs: `--epochs 1`
3. Use smaller model (e.g., Qwen2.5-0.5B)

#### "CUDA out of memory"

```bash
# Reduce batch size
slm-eval train --batch-size 1 --gradient-accumulation 16

# Use gradient checkpointing (default: enabled)

# Use smaller LoRA rank
slm-eval train --lora-r 8
```

### Debug Mode

```bash
# Verbose logging
slm-eval run -c config.yaml --verbose

# Run with fixture (no model inference)
slm-eval run -c config.yaml --fixture-gold
```

### Check Run Status

```bash
# List all runs
ls runs/

# Check run checkpoint
ls runs/<run_id>/

# View run manifest
cat runs/<run_id>/run_manifest.json
```

### Contact Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/slm-agentic-eval-harness/issues)
- **Security**: See [SECURITY.md](../SECURITY.md)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/slm-agentic-eval-harness/discussions)

---

## Appendix

### Example Dataset (CORD Invoices)

```json
{"id": "inv-001", "input": "Extract invoice total from: Total: $1,234.56", "expected": "1234.56", "strata": {"item_count": 3}}
{"id": "inv-002", "input": "Extract vendor: Vendor: Acme Corp", "expected": "Acme Corp", "strata": {"item_count": 1}}
```

### Example Config (Full)

```yaml
run:
  name: "invoice-extraction-v1"
  seed: 42
  output_dir: "runs"

dataset:
  path: "data/holdout/"
  format: "jsonl"

inference:
  backend: "vllm"
  model: "meta-llama/Llama-3-8B-Instruct"
  api_base: "http://localhost:8000/v1"
  temperature: 0.0
  max_tokens: 512
  vllm_port: 8000

judge:
  enabled: true
  model: "gpt-4"
  min_kappa: 0.85
  subjective_fields: ["vendor_name", "description"]

gate:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50
  bootstrap_samples: 1000
  confidence_level: 0.95

strata:
  keys: ["item_count", "category"]

thresholds:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50
```

### Performance Benchmarks

| Model | Size | Hardware | Time/Epoch |
|-------|------|----------|-----------|
| TinyLlama-1.1B | 1.1B | T4 GPU | ~2 min (79 samples) |
| TinyLlama-1.1B | 1.1B | M1 MPS | ~3 min (79 samples) |
| Qwen2.5-1.5B | 1.5B | A10G | ~2 min (79 samples) |
| Llama-3-8B | 8B | A100-40GB | ~10 min (79 samples) |

### License

MIT License — see [LICENSE](../LICENSE)
