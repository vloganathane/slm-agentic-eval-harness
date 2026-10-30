"""vLLM serving integration tests (Option C).

No vllm dependency required: we test command construction, config wiring,
readiness probing, adoption of an existing server, and the graceful error when
vllm is absent.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

from slm_eval.config import EvalConfig
from slm_eval.serving import (
    ServeSpec,
    ServingError,
    VllmServer,
    build_spec_from_config,
    endpoint_reachable,
    port_open,
    vllm_available,
)

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cord.yaml"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ListeningServer:
    """Minimal TCP listener standing in for a running vLLM endpoint."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(50)
        self._closed = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.close()

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass


def test_build_command_contains_core_flags():
    spec = ServeSpec(model="Qwen/Qwen2.5-7B-Instruct", port=8123)
    cmd = spec.build_command()
    joined = " ".join(cmd)
    assert "vllm.entrypoints.openai.api_server" in joined
    assert "--model Qwen/Qwen2.5-7B-Instruct" in joined
    assert "--port 8123" in joined
    assert "--host 127.0.0.1" in joined


def test_build_command_includes_optional_flags():
    spec = ServeSpec(
        model="m",
        max_model_len=4096,
        dtype="float16",
        tensor_parallel_size=2,
        extra_args=["--enforce-eager"],
    )
    cmd = spec.build_command()
    assert "--max-model-len 4096" in " ".join(cmd)
    assert "--dtype float16" in " ".join(cmd)
    assert "--tensor-parallel-size 2" in " ".join(cmd)
    assert "--enforce-eager" in cmd


def test_base_url_shape():
    spec = ServeSpec(model="m", port=9001, host="0.0.0.0")
    assert spec.base_url == "http://0.0.0.0:9001/v1"


def test_build_spec_from_config_reads_vllm_fields():
    cfg = EvalConfig.from_yaml(CONFIG)
    cfg.inference.vllm_port = 9321
    cfg.inference.vllm_gpu_memory_utilization = 0.75
    cfg.inference.vllm_max_model_len = 2048
    cfg.inference.vllm_dtype = "bfloat16"
    cfg.inference.vllm_tensor_parallel_size = 4

    spec = build_spec_from_config(cfg, "my-model")
    assert spec.model == "my-model"
    assert spec.port == 9321
    assert spec.gpu_memory_utilization == 0.75
    assert spec.max_model_len == 2048
    assert spec.dtype == "bfloat16"
    assert spec.tensor_parallel_size == 4


def test_port_open_false_for_closed_port():
    assert port_open("127.0.0.1", _free_port(), timeout=0.1) is False


def test_endpoint_reachable_matches_listener():
    port = _free_port()
    srv = _ListeningServer(port)
    try:
        assert endpoint_reachable(f"http://127.0.0.1:{port}/v1") is True
    finally:
        srv.close()


def test_endpoint_reachable_false_after_close():
    port = _free_port()
    srv = _ListeningServer(port)
    srv.close()
    assert endpoint_reachable(f"http://127.0.0.1:{port}/v1", timeout=0.1) is False


def test_server_adopts_existing_endpoint():
    port = _free_port()
    srv = _ListeningServer(port)
    try:
        spec = ServeSpec(model="m", port=port)
        server = VllmServer(spec, adopt_existing=True).start()
        assert server.adopted is True
        server.stop()  # must not kill an adopted server
        assert endpoint_reachable(spec.base_url, timeout=0.1) is True
    finally:
        srv.close()


def test_server_refuses_double_launch_when_adoption_disabled():
    port = _free_port()
    srv = _ListeningServer(port)
    try:
        spec = ServeSpec(model="m", port=port)
        with pytest.raises(ServingError, match="already in use"):
            VllmServer(spec, adopt_existing=False).start()
    finally:
        srv.close()


def test_server_errors_gracefully_when_vllm_missing():
    if vllm_available():
        pytest.skip("vllm installed; missing-dependency path not exercised")
    spec = ServeSpec(model="m", port=_free_port())
    with pytest.raises(ServingError, match="vllm is not installed"):
        VllmServer(spec).start()
