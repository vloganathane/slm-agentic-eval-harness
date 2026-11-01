"""Training frameworks: TRL, Axolotl, LLaMA-Factory, Unsloth."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .base import TrainingConfig, TrainingProvider


class TRLProvider(TrainingProvider):
    """TRL SFTTrainer (HuggingFace)."""

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
        import torch

        # Detect device
        has_cuda = torch.cuda.is_available()
        device_type = "cuda" if has_cuda else "cpu"
        print(f"Training with TRL on {device_type}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        if has_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = None

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            dtype=dtype,
            device_map=device_map,
        )

        # Apply LoRA
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"LoRA applied. Trainable: {trainable:,} / {total:,}")

        # Load dataset
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=config.dataset)

        # Training args
        train_args = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.extra.get("gradient_accumulation", 4),
            learning_rate=config.learning_rate,
            logging_steps=10,
            save_steps=500,
            warmup_steps=config.extra.get("warmup_steps", 10),
            gradient_checkpointing=True,
            bf16=has_cuda and torch.cuda.is_bf16_supported(),
            fp16=has_cuda and not torch.cuda.is_bf16_supported(),
            max_length=1024,
            dataset_text_field="text",
            packing=False,
        )

        # Train
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
        )
        trainer.train()

        # Save
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        return {
            "adapter_path": config.output_dir,
            "trainable_params": trainable,
            "total_params": total,
            "device": device_type,
            "framework": "trl",
        }

    def is_available(self) -> bool:
        try:
            import trl
            return True
        except ImportError:
            return False


class AxolotlProvider(TrainingProvider):
    """Axolotl training framework."""

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        # Generate Axolotl config
        axolotl_config = self._generate_config(config)
        config_path = Path(config.output_dir) / "axolotl_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(axolotl_config)

        # Run Axolotl
        cmd = ["axolotl", "train", str(config_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Axolotl training failed: {result.stderr}")

        return {
            "adapter_path": config.output_dir,
            "framework": "axolotl",
            "config_path": str(config_path),
        }

    def _generate_config(self, config: TrainingConfig) -> str:
        return f"""base_model: {config.base_model}
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: false

datasets:
  - path: {config.dataset}
    ds_type: json
    field: text

dataset_prepared_path: {config.output_dir}/prepared

val_set_size: 0.05

lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

lora_r: {config.lora_r}
lora_alpha: {config.lora_alpha}
lora_dropout: {config.lora_dropout}
lora_target_linear: true

gradient_accumulation_steps: {config.extra.get('gradient_accumulation', 4)}
micro_batch_size: {config.batch_size}
num_epochs: {config.epochs}
optimizer: adamw_bnb_8bit
learning_rate: {config.learning_rate}

output_dir: {config.output_dir}
save_strategy: steps
save_steps: 500

gradient_checkpointing: true
flash_attention: false

wandb:
  project: axolotl
  run_id: null
  entity: null
"""

    def is_available(self) -> bool:
        try:
            import axolotl
            return True
        except ImportError:
            return False


class LLaMAFactoryProvider(TrainingProvider):
    """LLaMA-Factory training framework."""

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        # Generate LLaMA-Factory config
        config_path = self._generate_config(config)

        # Run LLaMA-Factory
        cmd = [
            "llamafactory-cli",
            "train",
            str(config_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LLaMA-Factory training failed: {result.stderr}")

        return {
            "adapter_path": config.output_dir,
            "framework": "llama_factory",
            "config_path": str(config_path),
        }

    def _generate_config(self, config: TrainingConfig) -> Path:
        import yaml

        cfg = {
            "model_name_or_path": config.base_model,
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
            "lora_rank": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "dataset": "data/train.jsonl",
            "template": "default",
            "finetune_type": "lora",
            "output_dir": config.output_dir,
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            "per_device_train_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.extra.get("gradient_accumulation", 4),
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": True,
            "gradient_checkpointing": True,
            "max_grad_norm": 0.5,
            "save_strategy": "steps",
            "save_steps": 500,
            "logging_steps": 10,
        }

        config_path = Path(config.output_dir) / "llama_factory_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        return config_path

    def is_available(self) -> bool:
        try:
            import llamafactory
            return True
        except ImportError:
            return False


class UnslothProvider(TrainingProvider):
    """Unsloth training framework (fastest LoRA)."""

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise RuntimeError("Unsloth not installed. Install with: pip install unsloth")

        import torch

        # Load model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=2048,
            load_in_4bit=True,  # QLoRA
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load dataset
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=config.dataset)

        # Format dataset
        def format_func(examples):
            return {"text": f"### Input:\\n{examples['input']}\\n\\n### Output:\\n{examples['expected']}"}

        dataset = dataset.map(format_func, remove_columns=dataset["train"].column_names)

        # Train with TRL
        from trl import SFTConfig, SFTTrainer

        train_args = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.extra.get("gradient_accumulation", 4),
            learning_rate=config.learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            warmup_steps=config.extra.get("warmup_steps", 10),
            max_length=2048,
            dataset_text_field="text",
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
        )
        trainer.train()

        # Save
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        return {
            "adapter_path": config.output_dir,
            "framework": "unsloth",
            "device": "cuda",
        }

    def is_available(self) -> bool:
        try:
            import unsloth
            return True
        except ImportError:
            return False


# Provider registry
TRAINING_PROVIDERS: dict[str, type[TrainingProvider]] = {
    "trl": TRLProvider,
    "axolotl": AxolotlProvider,
    "llama_factory": LLaMAFactoryProvider,
    "llama-factory": LLaMAFactoryProvider,
    "unsloth": UnslothProvider,
}


def get_training_provider(name: str) -> TrainingProvider:
    """Get training provider by name."""
    provider_cls = TRAINING_PROVIDERS.get(name.lower())
    if provider_cls is None:
        available = ", ".join(TRAINING_PROVIDERS.keys())
        raise ValueError(f"Unknown training framework: {name}. Available: {available}")
    return provider_cls()


def list_training_providers() -> list[str]:
    """List available training frameworks."""
    return list(TRAINING_PROVIDERS.keys())
