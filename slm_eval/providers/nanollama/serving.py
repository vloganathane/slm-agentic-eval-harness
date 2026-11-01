"""Nanollama serving provider.

Serve models via:
1. Go inference engine (standalone, zero dependencies)
2. GGUF with llama.cpp (llama-server)
3. Ollama (if GGUF is compatible)
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from ..base import ServingConfig, ServingProvider


class NanollamaGoServingProvider(ServingProvider):
    """Serve via nanollama Go inference engine."""

    def __init__(self, go_binary: str | Path | None = None):
        self.go_binary = Path(go_binary) if go_binary else self._find_binary()

    def _find_binary(self) -> Path | None:
        candidates = [
            Path.home() / "nanollama" / "go" / "nanollama",
            Path.cwd() / "nanollama" / "go" / "nanollama",
        ]
        for c in candidates:
            if c.exists():
                return c
        if shutil.which("nanollama"):
            return Path(shutil.which("nanollama"))
        return None

    def serve(self, config: ServingConfig) -> str:
        if not self.go_binary:
            raise RuntimeError(
                "nanollama Go binary not found. Build with:\n"
                "  cd nanollama/go && go build -o nanollama ."
            )

        cmd = [
            str(self.go_binary),
            "--model", config.model,
            "--serve",
            "--port", str(config.port),
        ]

        # Add gamma if provided
        gamma = config.extra.get("gamma")
        if gamma:
            cmd.extend(["--gamma", str(gamma)])

        print(f"Starting nanollama Go server: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        self._wait_for_server(config.port)
        return f"http://localhost:{config.port}"

    def _wait_for_server(self, port: int, timeout: int = 30):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
                print("nanollama Go server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"nanollama Go server did not start within {timeout}s")

    def is_available(self) -> bool:
        return self.go_binary is not None and self.go_binary.exists()


class NanollamaLlamaCppServingProvider(ServingProvider):
    """Serve via llama.cpp (GGUF compatibility)."""

    def serve(self, config: ServingConfig) -> str:
        # Check if model is GGUF
        if not config.model.endswith(".gguf"):
            # Try to export to GGUF first
            print(f"Note: Model {config.model} is not GGUF. Export first:")
            print(f"  python -m scripts.export_gguf --checkpoint {config.model} ...")

        # Use llama-server
        if not shutil.which("llama-server"):
            raise RuntimeError(
                "llama-server not found. Install llama.cpp:\n"
                "  brew install llama.cpp  # macOS\n"
                "  # or build from source: https://github.com/ggml-org/llama.cpp"
            )

        cmd = [
            "llama-server",
            "-m", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "-c", str(config.max_model_len),
        ]

        print(f"Starting llama-server: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        self._wait_for_server(config.port)
        return f"http://localhost:{config.port}/v1"

    def _wait_for_server(self, port: int, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                print("llama-server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"llama-server did not start within {timeout}s")

    def is_available(self) -> bool:
        return shutil.which("llama-server") is not None


class NanollamaOllamaServingProvider(ServingProvider):
    """Serve via Ollama (if GGUF is compatible)."""

    def serve(self, config: ServingConfig) -> str:
        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama not found. Install:\n"
                "  brew install ollama  # macOS\n"
                "  # or download from https://ollama.ai"
            )

        # Create Ollama model from GGUF
        model_name = Path(config.model).stem
        modelfile = f"FROM {config.model}"
        modelfile_path = Path("/tmp/Modelfile")
        modelfile_path.write_text(modelfile)

        # Create model
        subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)

        # Serve
        subprocess.Popen(["ollama", "serve"])

        self._wait_for_server(11434)
        return "http://localhost:11434/v1"

    def _wait_for_server(self, port: int, timeout: int = 30):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=2)
                print("Ollama server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"Ollama server did not start within {timeout}s")

    def is_available(self) -> bool:
        return shutil.which("ollama") is not None
