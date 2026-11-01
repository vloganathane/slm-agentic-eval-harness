"""Multi-provider/framework support for slm-eval.

Inference: OpenAI, Anthropic, Google, Mistral, Together, Groq, Fireworks, local
Training: TRL, Axolotl, LLaMA-Factory, Unsloth
Serving: vLLM, TGI, SGLang, llama.cpp
"""

from .base import (
    InferenceProvider,
    ProviderConfig,
    ServingConfig,
    ServingProvider,
    TrainingConfig,
    TrainingProvider,
)
from .inference import (
    PROVIDERS,
    get_provider,
    list_providers,
)
from .serving import (
    SERVING_PROVIDERS,
    get_serving_provider,
    list_serving_providers,
)
from .training import (
    TRAINING_PROVIDERS,
    get_training_provider,
    list_training_providers,
)

__all__ = [
    # Base
    "InferenceProvider",
    "TrainingProvider",
    "ServingProvider",
    "ProviderConfig",
    "TrainingConfig",
    "ServingConfig",
    # Inference
    "get_provider",
    "list_providers",
    "PROVIDERS",
    # Training
    "get_training_provider",
    "list_training_providers",
    "TRAINING_PROVIDERS",
    # Serving
    "get_serving_provider",
    "list_serving_providers",
    "SERVING_PROVIDERS",
]
