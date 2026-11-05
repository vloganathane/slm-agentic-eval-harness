"""LoRA fine-tuning for SLM evaluation harness.

Provides a modular `train_lora()` function that takes a model path + dataset
and produces a fine-tuned adapter + model, with portable outputs the gate
pipeline can consume.

Usage (programmatic):
    from slm_eval.train import train_lora
    result = train_lora(
        model="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="runs/run-123/training_format/train.jsonl",
        config_path="eval_config.yaml",
    )
    # result.adapter_path, result.model_path, result.trainable_params, etc.

Usage (CLI):
    python -m slm_eval.train --model Qwen/Qwen2.5-7B-Instruct \
        --dataset runs/run-123/training_format/train.jsonl \
        --config eval_config.yaml
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_eval.config import EvalConfig
from slm_eval.errors import SlmEvalError


@dataclass
class TrainResult:
    """Portable result contract from training.

    All paths are absolute and relative to the project root — the gate pipeline
    can consume any result without knowing the caller's directory structure.
    """
    adapter_path: Path  # Path to PEFT adapter dir (trainer.save_model output)
    model_path: Path  # Path to merged/model + tokenizer saved together
    config_path: Path  # Path to training config JSON
    trainable_params: int  # Count of learnable parameters
    total_params: int  # Count of total parameters
    training_config: dict[str, Any]  # Parsed training args (for provenance)
    run_id: str  # Generated or provided run identifier


def _check_train_dependencies() -> None:
    """Verify required training dependencies are available."""
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SlmEvalError(
            f"Training dependencies not installed: {', '.join(missing)}. "
            f"Run: pip install -e .[train]"
        )


def train_lora(
    *,
    model: str,
    dataset_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_lora: bool = True,
    resume_from_checkpoint: str | bool | None = None,
    fsdp: str | None = None,
    fsdp_config: dict[str, Any] | str | None = None,
    deepspeed: dict[str, Any] | str | None = None,
) -> TrainResult:
    """Run LoRA fine-tuning and return a TrainResult.

    Args:
        model: HuggingFace model ID or local path
        dataset_path: Path to training JSONL (must have "text" field)
        config_path: Path to eval_config.yaml
        output_dir: Where to save adapter/model (defaults to ./models/fine_tuned)
        run_id: Run identifier (auto-generated if None)
        epochs: Override epochs from config/cli
        batch_size: Override batch size from config/cli
        learning_rate: Override lr from config/cli
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_lora: Whether to apply LoRA
        fsdp: FSDP strategy (e.g., "full_shard", "shard_grad_op", "hybrid_shard")
        fsdp_config: FSDP configuration dict or path to config file
        deepspeed: DeepSpeed configuration dict or path to config file

    Returns:
        TrainResult with all output paths and param counts
    """
    _check_train_dependencies()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load config (validates format)
    EvalConfig.from_yaml(config_path)

    # Budget guard — cost tracking done at the orchestrator level.

    # Override params from keyword args if provided
    effective_epochs = epochs or 3
    effective_batch_size = batch_size or 2
    effective_lr = learning_rate or 2e-4

    # Generate run_id if not provided
    if run_id is None:
        run_id = f"run-{int(time.time())}"

    # ---- Prepare dataset ----
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise SlmEvalError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path))
    train_data = dataset["train"]
    print(f"  Loaded {len(train_data)} training cases")

    # ---- Create model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading model {model}...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
    )

    # ---- Apply LoRA if requested ----
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model_obj = get_peft_model(model_obj, lora_config)
        trainable, total = model_obj.get_nb_trainable_parameters()
        print(f"LoRA applied. Trainable params: {trainable:,} / {total:,}")
    else:
        # Count all params when LoRA is not used
        total = sum(p.numel() for p in model_obj.parameters())
        trainable = total  # all params are "trainable" without LoRA

    # ---- Training arguments ----
    from trl import SFTConfig, SFTTrainer

    if output_dir is None:
        output_dir = f"./models/fine_tuned_{run_id}"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=effective_lr,
        logging_steps=10,
        save_steps=500,
        warmup_steps=min(100, max(1, effective_epochs)),
        gradient_checkpointing=has_cuda,
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        report_to="none",
        run_name=f"train-{run_id}",
        max_length=1024,
        dataset_text_field="text",
        packing=False,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        deepspeed=deepspeed,
    )

    # ---- Train ----
    print("Starting training...")
    trainer = SFTTrainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Training completed successfully!")
    except Exception as e:  # noqa: BLE001 — surface any trainer failure
        print(f"Training error: {e}", file=sys.stderr)
        raise

    # ---- Save final model + adapter ----
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    final_model_path = output_dir_path / f"{run_id}_final"
    if trainer.is_world_process_zero():
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Save training config (drop non-serializable args)
        train_config = {k: v for k, v in vars(training_args).items() if not callable(v)}
        config_path = output_dir_path / f"{run_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(train_config, f, indent=2)

        print(f"Model saved to: {final_model_path}")
        print(f"Config saved to: {config_path}")

    # Count params from saved model
    if has_cuda:
        dev_model = AutoModelForCausalLM.from_pretrained(
            str(final_model_path),
            torch_dtype=dtype,
            device_map="auto",
        )
        trainable_params = sum(p.numel() for p in dev_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in dev_model.parameters())
    else:
        # CPU fallback - load the PEFT model
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,
            device_map=None,
        )
        merged = PeftModel.from_pretrained(base_model, str(final_model_path))
        trainable_params = sum(p.numel() for p in merged.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in merged.parameters())

    return TrainResult(
        adapter_path=final_model_path / "adapter",
        model_path=final_model_path,
        config_path=config_path,
        trainable_params=trainable_params,
        total_params=total_params,
        training_config=train_config,
        run_id=run_id,
    )


# CLI entry point for `python -m slm_eval.train`
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA adapter for SLM evaluation")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="Path to training JSONL")
    parser.add_argument("--config", required=True, help="Path to eval_config.yaml")
    parser.add_argument("--output-dir", help="Output directory for adapter/model")
    parser.add_argument("--run-id", help="Run identifier (auto-generated if omitted)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune)")
    parser.add_argument("--fsdp", type=str, help="FSDP strategy (e.g., full_shard, shard_grad_op, hybrid_shard)")
    parser.add_argument("--fsdp-config", type=str, help="FSDP config dict as JSON string or path to config file")
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config dict as JSON string or path to config file")

    args = parser.parse_args()

    # Parse JSON string args for fsdp_config and deepspeed
    fsdp_config = None
    if args.fsdp_config:
        import json
        try:
            fsdp_config = json.loads(args.fsdp_config)
        except json.JSONDecodeError:
            # Treat as file path
            fsdp_config = args.fsdp_config

    deepspeed = None
    if args.deepspeed:
        import json
        try:
            deepspeed = json.loads(args.deepspeed)
        except json.JSONDecodeError:
            # Treat as file path
            deepspeed = args.deepspeed

    try:
        result = train_lora(
            model=args.model,
            dataset_path=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_lora=not args.no_lora,
            fsdp=args.fsdp,
            fsdp_config=fsdp_config,
            deepspeed=deepspeed,
        )
        print(f"Run ID: {result.run_id}")
        print(f"Model: {result.model_path}")
        print(f"Adapter: {result.adapter_path}")
        print(f"Config: {result.config_path}")
        print(f"Trainable params: {result.trainable_params:,}")
        print(f"Total params: {result.total_params:,}")
        return 0
    except SlmEvalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
