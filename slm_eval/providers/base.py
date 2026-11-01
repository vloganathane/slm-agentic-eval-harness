"""Provider abstraction layer for multi-provider/framework support.

Supports:
- Inference: OpenAI, Anthropic, Google, Mistral, Together, Groq, Fireworks, local
- Training: TRL, Axolotl, LLaMA-Factory, Unsloth
- Serving: vLLM, TGI, SGLang, llama.cpp
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ProviderConfig:
    """Base provider configuration."""
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training framework configuration."""
    framework: str  # trl, axolotl, llama_factory, unsloth
    base_model: str
    dataset: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingConfig:
    """Serving framework configuration."""
    framework: str  # vllm, tgi, sglang, llama_cpp
    model: str
    port: int = 8000
    host: str = "0.0.0.0"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


class InferenceProvider(ABC):
    """Base class for inference providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class TrainingProvider(ABC):
    """Base class for training frameworks."""

    @abstractmethod
    def train(self, config: TrainingConfig) -> dict[str, Any]:
        """Run training."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if framework is available."""
        pass


class ServingProvider(ABC):
    """Base class for serving frameworks."""

    @abstractmethod
    def serve(self, config: ServingConfig) -> str:
        """Start serving, return endpoint URL."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if framework is available."""
        pass
