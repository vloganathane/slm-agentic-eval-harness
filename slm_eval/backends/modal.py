"""Modal GPU/CPU backend for LoRA fine-tuning.

Supports both GPU (T4, A10G, A100) and CPU-only training.
Trigger: Local inference latency unacceptable (R-4) or 7B+ model training needed.
Per SPEC_PRODUCTION.md §11.1, Appendix A.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


def _modal_available() -> bool:
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        return False


def _modal_app_name() -> str:
    """Get Modal app name from env or default."""
    return os.getenv("MODAL_APP_NAME", "slm-finetune")


def _modal_function_name() -> str:
    """Get Modal function name from env or default."""
    return os.getenv("MODAL_FUNCTION_NAME", "train_lora")


def train_lora_on_modal(
    base_model: str,
    train_data_path: str,
    lora_config: dict[str, Any],
    training_args: dict[str, Any],
    output_dir: str,
    *,
    modal_app: str | None = None,
    modal_function: str | None = None,
    gpu: str = "T4",
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """
    Launch LoRA fine-tuning on Modal.
    
    Args:
        base_model: HF model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        train_data_path: Local path or S3 URI to training JSONL
        lora_config: LoRA config dict (r, alpha, dropout, target_modules)
        training_args: Training args dict (epochs, lr, batch_size, etc.)
        output_dir: Local directory to save adapter
        modal_app: Modal app name (default: slm-finetune)
        modal_function: Modal function name (default: train_lora)
        gpu: GPU type ("T4", "A10G", "A100-40GB", "A100-80GB", "L4", "H100") or "cpu"
        timeout_seconds: Max wait time
    
    Returns:
        Dict with adapter_path, config_path, metrics, device
    """
    if not _modal_available():
        raise RuntimeError("modal not installed. Install with: pip install modal")

    import modal

    app_name = modal_app or _modal_app_name()
    func_name = modal_function or _modal_function_name()

    # Look up the remote function
    fn = modal.Function.from_name(app_name, func_name)

    # Prepare inputs
    # If train_data_path is local, we need to upload it
    train_data_s3 = train_data_path
    if not train_data_path.startswith("s3://"):
        # Path is on the /mnt volume mount
        if train_data_path.startswith("/"):
            train_data_s3 = "/mnt" + train_data_path
        else:
            train_data_s3 = "/mnt/" + train_data_path

    print(f"Launching LoRA training on Modal ({app_name}.{func_name})")
    print(f"  Base model: {base_model}")
    print(f"  Train data: {train_data_s3}")
    print(f"  LoRA config: {lora_config}")
    print(f"  Training args: {training_args}")
    print(f"  Device: {gpu.upper() if gpu != 'cpu' else 'CPU'}")
    print(f"  Output dir: {output_dir}")
    if gpu == "cpu":
        print("  Note: CPU training is slow (~5hr/epoch for 1B model). Use GPU for faster training.")

    # Call remote function
    result = fn.remote(
        base_model=base_model,
        train_data_path=train_data_s3,
        lora_config=lora_config,
        training_args=training_args,
        output_dir=output_dir,
    )

    # Result should contain adapter_path, config_path, metrics
    if isinstance(result, dict):
        return result
    else:
        # Handle unexpected return
        return {"raw_result": str(result)}


def deploy_vllm_on_modal(
    model_path: str,
    *,
    modal_app: str | None = None,
    modal_function: str = "deploy_vllm",
    gpu: str = "A100",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
) -> dict[str, Any]:
    """
    Deploy vLLM server on Modal.
    
    Returns endpoint URL and deployment info.
    """
    if not _modal_available():
        raise RuntimeError("modal not installed. Install with: pip install modal")

    import modal

    app_name = modal_app or _modal_app_name()
    fn = modal.Function.from_name(app_name, modal_function)

    print(f"Deploying vLLM on Modal ({app_name}.{modal_function})")
    print(f"  Model: {model_path}")
    print(f"  GPU: {gpu}")
    print(f"  Tensor parallel: {tensor_parallel_size}")

    result = fn.remote(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    return result if isinstance(result, dict) else {"raw_result": str(result)}


class ModalTrainingBackend:
    """Training backend using Modal (per SPEC_PRODUCTION.md)."""

    def train(
        self,
        base_model: str,
        train_data: str,
        lora_config: dict[str, Any],
        args: dict[str, Any],
        output_dir: str,
    ) -> str:
        """
        Train LoRA on Modal.
        
        Returns output_dir with saved adapter.
        """
        result = train_lora_on_modal(
            base_model=base_model,
            train_data_path=train_data,
            lora_config=lora_config,
            training_args=args,
            output_dir=output_dir,
            modal_app=args.get("modal_app"),
            modal_function=args.get("modal_function", "train_lora"),
            gpu=args.get("gpu", "A100"),
            timeout_seconds=args.get("timeout_seconds", 7200),
        )

        # Check result
        adapter_path = result.get("adapter_path")
        if not adapter_path:
            raise RuntimeError(f"Modal training failed: {result}")

        print(f"Modal training complete. Adapter: {adapter_path}")
        return output_dir


class ModalDeploymentBackend:
    """vLLM deployment backend using Modal."""

    def deploy(
        self,
        model_path: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Deploy model on Modal vLLM."""
        return deploy_vllm_on_modal(
            model_path=model_path,
            modal_app=args.get("modal_app"),
            modal_function=args.get("modal_function", "deploy_vllm"),
            gpu=args.get("gpu", "A100"),
            tensor_parallel_size=args.get("tensor_parallel_size", 1),
            gpu_memory_utilization=args.get("gpu_memory_utilization", 0.9),
            max_model_len=args.get("max_model_len", 8192),
        )


# Example Modal app definition (to be deployed separately)
MODAL_APP_TEMPLATE = '''
import modal
import os
import json

app = modal.App("slm-finetune")

# GPU-optimized image with training deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "wandb",
        "bitsandbytes",
    ])
)

# Volume for model weights and training data
volume = modal.Volume.from_name("slm-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",  # Change to "A10G", "A100", or use cpu=4.0 for CPU
    memory=16384,
    volumes={"/mnt": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def train_lora(
    base_model: str,
    train_data_path: str,
    lora_config: dict,
    training_args: dict,
    output_dir: str,
):
    """Remote LoRA training function."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import wandb

    # Detect device
    has_cuda = torch.cuda.is_available()
    device_type = "cuda" if has_cuda else "cpu"
    print(f"Training on device: {device_type}")

    # Download training data if S3
    if train_data_path.startswith("s3://"):
        import subprocess
        local_path = "/tmp/train.jsonl"
        subprocess.run(["aws", "s3", "cp", train_data_path, local_path], check=True)
        train_data_path = local_path

    # Load dataset - data is in /mnt volume
    if not train_data_path.startswith("/"):
        train_data_path = "/mnt/" + train_data_path
    dataset = load_dataset("json", data_files=train_data_path)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate dtype
    if has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
    )

    # Apply LoRA
    lora_cfg = LoraConfig(**lora_config)
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA applied. Trainable: {trainable:,} / {total:,}")

    # Training args
    train_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_args.get("num_epochs", 3),
        per_device_train_batch_size=training_args.get("batch_size", 4),
        gradient_accumulation_steps=training_args.get("gradient_accumulation", 4),
        learning_rate=training_args.get("learning_rate", 2e-4),
        logging_steps=10,
        save_steps=500,
        warmup_steps=training_args.get("warmup_steps", 10),
        gradient_checkpointing=True,
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=training_args.get("run_name", "modal-lora-train"),
        max_length=1024,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Return path for Modal to copy back
    return {
        "adapter_path": output_dir,
        "trainable_params": trainable,
        "total_params": total,
        "device": device_type,
    }


@app.function(
    image=image,
    cpu=4.0,  # Use GPU (gpu="T4") or CPU (cpu=4.0) for vLLM
    memory=16384,
    volumes={"/mnt": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def deploy_vllm(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
):
    """Deploy vLLM server on Modal."""
    import subprocess
    import time

    # Start vLLM server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    # Run in background
    proc = subprocess.Popen(cmd)
    
    # Wait for server to be ready
    time.sleep(30)
    
    # Get Modal web endpoint
    # This would use modal.web_endpoint or similar
    endpoint = "https://<modal-app>.modal.run"  # placeholder
    
    return {
        "endpoint": endpoint,
        "model": model_path,
        "status": "running",
    }
'''


def generate_modal_app_template(output_path: str | Path, gpu: str = "T4") -> None:
    """Generate Modal app template for user to deploy.

    Args:
        output_path: Path to write template file
        gpu: GPU type ("T4", "A10G", "A100") or "cpu" for CPU-only
    """
    template = MODAL_APP_TEMPLATE
    if gpu.lower() == "cpu":
        # Replace gpu with cpu
        template = template.replace('gpu="T4"', 'cpu=4.0, memory=16384')
        template = template.replace('gpu="T4",  # Change to "A10G", "A100", or use cpu=4.0 for CPU', 'cpu=4.0, memory=16384,  # CPU-only (Starter plan)')
        template = template.replace('cpu=4.0,  # Use GPU (gpu="T4") or CPU (cpu=4.0) for vLLM', 'cpu=4.0, memory=16384,  # CPU-only (vLLM needs GPU)')
    else:
        # Set specific GPU
        template = template.replace('gpu="T4"', f'gpu="{gpu}"')
        template = template.replace('gpu="T4",  # Change to "A10G", "A100", or use cpu=4.0 for CPU', f'gpu="{gpu}",  # GPU-enabled')

    Path(output_path).write_text(template.strip() + "\n")
    print(f"Generated Modal app template: {output_path} (gpu={gpu})")
    print("Deploy with: modal deploy <path>")


# CLI integration point
def _cmd_modal_train(args: argparse.Namespace) -> int:
    """Launch LoRA training on Modal."""
    from slm_eval.backends.modal import train_lora_on_modal

    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    training_args = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "gradient_accumulation": 4,
        "warmup_steps": 10,
        "run_name": f"modal-lora-{int(time.time())}",
    }

    try:
        result = train_lora_on_modal(
            base_model=args.base_model,
            train_data_path=args.train_data,
            lora_config=lora_config,
            training_args=training_args,
            output_dir=args.output_dir,
            modal_app=args.modal_app,
            modal_function=args.modal_function,
            gpu=args.gpu,
        )
        print(f"Training result: {result}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _cmd_modal_deploy(args: argparse.Namespace) -> int:
    """Deploy vLLM on Modal."""
    from slm_eval.backends.modal import deploy_vllm_on_modal

    try:
        result = deploy_vllm_on_modal(
            model_path=args.model_path,
            modal_app=args.modal_app,
            gpu=args.gpu,
            tensor_parallel_size=args.tensor_parallel,
        )
        print(f"Deployment result: {result}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _cmd_modal_generate_template(args: argparse.Namespace) -> int:
    """Generate Modal app template."""
    from slm_eval.backends.modal import generate_modal_app_template

    generate_modal_app_template(args.output, gpu=args.gpu)
    return 0


def add_modal_subparsers(subparsers) -> None:
    """Add Modal commands to CLI."""

    modal_parser = subparsers.add_parser("modal", help="Modal GPU/CPU backend commands")
    modal_sub = modal_parser.add_subparsers(dest="modal_command", required=True)

    # modal train
    train_parser = modal_sub.add_parser("train", help="Launch LoRA training on Modal")
    train_parser.add_argument("--base-model", required=True)
    train_parser.add_argument("--train-data", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--gpu", default="T4", choices=["T4", "A10G", "A100-40GB", "A100-80GB", "L4", "H100", "cpu"], help="GPU type or 'cpu' for CPU-only")
    train_parser.add_argument("--modal-app", default="slm-finetune")
    train_parser.add_argument("--modal-function", default="train_lora")
    train_parser.set_defaults(func=_cmd_modal_train)

    # modal deploy
    deploy_parser = modal_sub.add_parser("deploy", help="Deploy vLLM on Modal")
    deploy_parser.add_argument("--model-path", required=True)
    deploy_parser.add_argument("--gpu", default="T4", choices=["T4", "A10G", "A100-40GB", "A100-80GB", "L4", "H100", "cpu"], help="GPU type (vLLM requires GPU)")
    deploy_parser.add_argument("--tensor-parallel", type=int, default=1)
    deploy_parser.add_argument("--modal-app", default="slm-finetune")
    deploy_parser.set_defaults(func=_cmd_modal_deploy)

    # modal generate-template
    gen_parser = modal_sub.add_parser("generate-template", help="Generate Modal app template")
    gen_parser.add_argument("--output", default="modal_app.py")
    gen_parser.add_argument("--gpu", default="T4", choices=["T4", "A10G", "A100-40GB", "A100-80GB", "L4", "H100", "cpu"], help="GPU type or 'cpu' for CPU-only")
    gen_parser.set_defaults(func=_cmd_modal_generate_template)