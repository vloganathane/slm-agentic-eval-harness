"""Nanollama training provider.

Train Llama 3 models from scratch with personality support.
https://github.com/ariannamethod/nanollama

Supports:
- Pretraining from scratch (FineWeb-Edu)
- LoRA personality fine-tuning
- Gamma extraction (personality weights)
- GGUF export (llama.cpp compatible)
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any

from ..base import TrainingConfig, TrainingProvider

# Model tiers from nanollama
MODEL_TIERS: dict[str, dict[str, Any]] = {
    "nano": {"layers": 13, "dim": 576, "heads": 9, "kv_heads": 9, "ffn": 1536, "params": "89M"},
    "micro": {"layers": 16, "dim": 640, "heads": 10, "kv_heads": 10, "ffn": 1792, "params": "122M"},
    "mini": {"layers": 20, "dim": 768, "heads": 12, "kv_heads": 3, "ffn": 2048, "params": "173M"},
    "small": {"layers": 24, "dim": 1024, "heads": 16, "kv_heads": 4, "ffn": 2816, "params": "336M"},
    "goldie": {"layers": 28, "dim": 1536, "heads": 24, "kv_heads": 6, "ffn": 4096, "params": "841M"},
    "medium": {"layers": 32, "dim": 2048, "heads": 32, "kv_heads": 8, "ffn": 5632, "params": "1.7B"},
    "large": {"layers": 36, "dim": 3072, "heads": 48, "kv_heads": 12, "ffn": 8192, "params": "4.2B"},
    "big": {"layers": 40, "dim": 4096, "heads": 64, "kv_heads": 16, "ffn": 11008, "params": "7.9B"},
}


class NanollamaTrainingProvider(TrainingProvider):
    """Nanollama training framework (from-scratch + personality)."""

    def __init__(self, repo_path: str | Path | None = None):
        """Initialize nanollama provider.

        Args:
            repo_path: Path to nanollama repo. If None, uses 'nanollama' from PATH
                      or expects it to be installed.
        """
        self.repo_path = Path(repo_path) if repo_path else self._find_repo()

    def _find_repo(self) -> Path | None:
        """Find nanollama repository."""
        # Check common locations
        candidates = [
            Path.home() / "nanollama",
            Path.cwd() / "nanollama",
            Path("/opt/nanollama"),
        ]
        for c in candidates:
            if (c / "scripts" / "base_train.py").exists():
                return c
        # Check if installed as package
        if shutil.which("nanollama"):
            return Path(shutil.which("nanollama")).parent
        return None

    def _get_python(self) -> str:
        """Get Python executable for nanollama."""
        if self.repo_path:
            venv_python = self.repo_path / ".venv" / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)
        return "python"

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        """Train with nanollama.

        Supports two modes:
        1. Pretraining from scratch (if no base_model)
        2. LoRA personality fine-tuning (if base_model provided)
        """
        mode = config.extra.get("mode", "lora_personality")
        model_size = config.extra.get("model_size", "nano")

        if mode == "pretrain":
            return self._pretrain(config, model_size)
        elif mode == "lora_personality":
            return self._lora_personality(config, model_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pretrain' or 'lora_personality'")

    def _pretrain(self, config: TrainingConfig, model_size: str) -> dict[str, Any]:
        """Pretrain from scratch."""
        python = self._get_python()

        # Prepare data
        num_samples = config.extra.get("num_samples", 10_000_000)
        print(f"Preparing FineWeb-Edu data ({num_samples:,} samples)...")
        data_cmd = [
            python, "-m", "data.prepare_fineweb",
            "--num-samples", str(num_samples),
        ]
        result = subprocess.run(data_cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Data preparation failed: {result.stderr}")

        # Train
        print(f"Training {model_size} model from scratch...")
        train_cmd = [
            python, "-m", "scripts.base_train",
            "--model-size", model_size,
        ]
        # Add distributed training if configured
        if config.extra.get("distributed"):
            train_cmd = [
                "torchrun", "--nproc_per_node=8",
                "-m", "scripts.base_train",
                "--model-size", model_size,
            ]

        result = subprocess.run(train_cmd, cwd=self.repo_path, text=True)
        if result.returncode != 0:
            raise RuntimeError("Training failed")

        # Export to GGUF
        checkpoint = config.output_dir / "checkpoints" / model_size / "checkpoint.pt"
        gguf_output = config.output_dir / f"{model_size}.gguf"
        export_cmd = [
            python, "-m", "scripts.export_gguf",
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(self.repo_path / "weights" / "tokenizer.model"),
            "--output", str(gguf_output),
            "--dtype", "f16",
        ]
        subprocess.run(export_cmd, cwd=self.repo_path, text=True)

        tier_info = MODEL_TIERS.get(model_size, {})
        return {
            "model_size": model_size,
            "params": tier_info.get("params"),
            "checkpoint": str(checkpoint),
            "gguf": str(gguf_output),
            "framework": "nanollama",
            "mode": "pretrain",
        }

    def _lora_personality(self, config: TrainingConfig, model_size: str) -> dict[str, Any]:
        """LoRA personality fine-tuning."""
        python = self._get_python()
        voice = config.extra.get("voice", "default")
        base_checkpoint = config.extra.get("base_checkpoint")

        if not base_checkpoint:
            # Try to find default checkpoint
            default_ckpt = (
                self.repo_path / "checkpoints" / model_size / "checkpoint.pt"
            )
            if default_ckpt.exists():
                base_checkpoint = str(default_ckpt)
            else:
                raise ValueError(
                    "base_checkpoint required for LoRA personality. "
                    "Train base first or provide --base-checkpoint."
                )

        # Format data: Human/AI plain text (not JSONL!)
        data_file = Path(config.dataset)
        if data_file.suffix == ".jsonl":
            # Convert JSONL to Human/AI format
            humanai_file = config.output_dir / f"{voice}_humanai.txt"
            self._convert_to_humanai(data_file, humanai_file)
            data_file = humanai_file

        # Run LoRA SFT
        print(f"Training LoRA personality: {voice}")
        sft_cmd = [
            python, "-m", "scripts.chat_sft",
            "--base-checkpoint", str(base_checkpoint),
            "--data", str(data_file),
            "--voice", voice,
            "--rank", str(config.lora_r),
            "--alpha", str(config.lora_alpha),
            "--epochs", str(config.epochs),
            "--lr", str(config.learning_rate),
            "--output", str(config.output_dir),
        ]
        result = subprocess.run(sft_cmd, cwd=self.repo_path, text=True)
        if result.returncode != 0:
            raise RuntimeError("LoRA SFT failed")

        # Extract gamma (personality weights)
        gamma_output = config.output_dir / f"gamma-{voice}.npz"
        merged_ckpt = config.output_dir / "merged.pt"
        if merged_ckpt.exists() and base_checkpoint:
            gamma_cmd = [
                python, "-m", "scripts.extract_gamma",
                "--personality_ckpt", str(merged_ckpt),
                "--base_ckpt", str(base_checkpoint),
                "--output", str(gamma_output),
            ]
            subprocess.run(gamma_cmd, cwd=self.repo_path, text=True)

        # Export personality to GGUF
        gguf_output = config.output_dir / f"{model_size}-{voice}.gguf"
        merged_ckpt = config.output_dir / "merged.pt"
        if merged_ckpt.exists():
            export_cmd = [
                python, "-m", "scripts.export_gguf",
                "--checkpoint", str(merged_ckpt),
                "--tokenizer", str(self.repo_path / "weights" / "tokenizer.model"),
                "--output", str(gguf_output),
                "--dtype", "f16",
            ]
            subprocess.run(export_cmd, cwd=self.repo_path, text=True)

        tier_info = MODEL_TIERS.get(model_size, {})
        return {
            "voice": voice,
            "model_size": model_size,
            "params": tier_info.get("params"),
            "adapter": str(config.output_dir / "adapter.pt"),
            "merged": str(merged_ckpt),
            "gamma": str(gamma_output) if gamma_output.exists() else None,
            "gguf": str(gguf_output) if gguf_output.exists() else None,
            "framework": "nanollama",
            "mode": "lora_personality",
        }

    def _convert_to_humanai(self, jsonl_file: Path, output_file: Path):
        """Convert JSONL to Human/AI plain text format."""
        import json

        with open(jsonl_file) as f, open(output_file, "w") as out:
            for line in f:
                record = json.loads(line)
                input_text = record.get("input", "")
                expected = record.get("expected", "")
                out.write(f"Human: {input_text}\nAI: {expected}\n\n")

    def is_available(self) -> bool:
        """Check if nanollama is available."""
        return self.repo_path is not None and (self.repo_path / "scripts").exists()


class NanollamaGoProvider(TrainingProvider):
    """Nanollama Go inference engine (for serving)."""

    def __init__(self, go_path: str | Path | None = None):
        self.go_path = Path(go_path) if go_path else self._find_go_binary()

    def _find_go_binary(self) -> Path | None:
        """Find nanollama Go binary."""
        candidates = [
            Path.home() / "nanollama" / "go" / "nanollama",
            Path.cwd() / "nanollama" / "go" / "nanollama",
            Path("/opt/nanollama/go/nanollama"),
        ]
        for c in candidates:
            if c.exists():
                return c
        # Check PATH
        if shutil.which("nanollama"):
            return Path(shutil.which("nanollama"))
        return None

    def train(self, config: TrainingConfig) -> dict[str, Any]:
        """Go engine doesn't train; it's for inference."""
        raise RuntimeError(
            "NanollamaGoProvider is for inference only. "
            "Use NanollamaTrainingProvider for training."
        )

    def is_available(self) -> bool:
        return self.go_path is not None
