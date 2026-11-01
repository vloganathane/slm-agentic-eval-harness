import modal
import os

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
    cpu=4.0,
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

    # Load model
    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
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
    }


@app.function(
    image=image,
    cpu=4.0,
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
