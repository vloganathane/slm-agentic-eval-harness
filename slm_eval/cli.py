"""CLI (SPEC_V3 §8): init, run, gate, calibrate, report, inspect.

Exit codes are the CI contract (§8.2): 0 promote · 1 blocked · 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from slm_eval import __version__, fixture
from slm_eval.config import EvalConfig
from slm_eval.errors import SlmEvalError
from slm_eval.judge import CalibrationGate
from slm_eval.llm import FixtureJudge, LLMClient, OpenAICompatClient
from slm_eval.pipeline import Orchestrator
from slm_eval.store import Store

EXIT_PROMOTE, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2


def _endpoint_reachable(endpoint: str) -> bool:
    """True if the endpoint host answers a TCP connect within a short timeout.

    Keeps the judge client choice honest: a real endpoint gets the real client,
    an unreachable one falls back to the fixture judge rather than hanging the
    run behind retries.
    """
    import socket

    parsed = urlparse(endpoint if "//" in endpoint else f"//{endpoint}")
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False

DEFAULT_CONFIG = """\
task_description: "Structured field extraction from Indian GST invoices"
task_type: extraction

fields:
  subjective_fields: [vendor_name, address, description]
  exact_match_fields: [invoice_number, date, total, tax_id, line_items]
  validators:
    tax_id:   {type: regex, pattern: "^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z].*Z[0-9A-Z]$"}
    total:    {type: numeric, tolerance: 0.01}
    line_items: {type: list, item_match: exact}
    date:     {type: date, formats: ["%Y-%m-%d", "%d/%m/%Y"]}
    invoice_number: {type: string, normalize: [trim, collapse_ws, casefold]}

strata:
  keys: [layout_type, language, quality]
  mode: single_key

split:        {fraction: 0.2, seed: 42, strategy: stratified}
stats:        {seed: 42, resamples: 10000}

thresholds:
  target_f1: 0.85
  failure_threshold: 0.5
  min_per_stratum: 50

judge:
  endpoint: "https://api.openai.com/v1"
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  prompt_version: v1.0
  calibration: {path: calibration/judge.jsonl, min_pairs: 50,
                target_kappa: 0.85, recalibrate_every_rounds: 5}

inference:
  backend: llama_server
  base_url: "http://127.0.0.1:8080/v1"
  timeout_seconds: 1800
  retry: {max_attempts: 3, backoff_base: 2}

human_review: {timeout_hours: 24, notify: []}
cost_control: {max_cost_per_round_usd: 10.0, alert_at_usd: 5.0}
reporting:    {formats: [markdown, json]}
"""


def _cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "eval_config.yaml"
    if cfg_path.exists() and not args.force:
        print(f"refusing to overwrite {cfg_path} (use --force)", file=sys.stderr)
        return EXIT_ERROR
    cfg_path.write_text(DEFAULT_CONFIG)
    EvalConfig.from_yaml(cfg_path)  # validate what we just wrote
    (root / "data").mkdir(exist_ok=True)
    (root / "calibration").mkdir(exist_ok=True)
    with Store(root / "slm_eval.db"):
        pass  # creates schema
    print(f"initialized harness in {root.resolve()}")
    return EXIT_PROMOTE


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")


def _make_judge_client(cfg: EvalConfig, judge_endpoint: str | None) -> LLMClient:
    """Build the judge's LLM client.

    With an endpoint (or cfg.judge.endpoint reachable), use the real
    OpenAICompatClient. Otherwise fall back to a fixture judge that echoes a
    valid match verdict so the M1 dev loop runs without a live judge (§10, M1).
    """
    endpoint = judge_endpoint or cfg.judge.endpoint
    api_key = os.environ.get(cfg.judge.api_key_env) if cfg.judge.api_key_env else None
    if judge_endpoint or (api_key and _endpoint_reachable(endpoint)):
        return OpenAICompatClient(
            base_url=endpoint, model=cfg.judge.model, api_key=api_key,
            max_attempts=cfg.inference.retry.max_attempts,
            backoff_base=cfg.inference.retry.backoff_base,
        )
    return FixtureJudge()


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = EvalConfig.from_yaml(args.config)
    run_dir = Path(args.runs_dir) / (args.run_id or _run_id())
    model_path = args.model or "fixture-model.gguf"
    store = Store(Path(args.config).parent / "slm_eval.db")
    try:
        orch = Orchestrator(
            cfg, store, run_dir, _make_judge_client(cfg, args.judge_endpoint),
            code_version=f"git:{__version__}", resume=args.resume,
        )
        if args.dataset:
            dataset = args.dataset
        else:
            # Generate the golden fixture as a stand-in dataset (M1 dev loop).
            dataset = str(run_dir / "data")
            fixture.write_dataset(dataset, count=args.fixture_count, seed=cfg.split.seed)

        gate = orch.run(
            model_path=model_path,
            model_sha256="fixture" * 11,  # replaced by real hash in M2 inspect
            dataset_path=dataset,
        )
        decision = gate["decision"]
        print(json.dumps(gate, indent=2, sort_keys=True))
        if decision["status"] == "error":
            return EXIT_ERROR
        return EXIT_PROMOTE if decision["promote"] else EXIT_BLOCKED
    except SlmEvalError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        store.close()


def _cmd_gate(args: argparse.Namespace) -> int:
    gate_path = Path(args.run_dir) / "gate.json"
    if not gate_path.exists():
        print(f"no gate.json in {args.run_dir}", file=sys.stderr)
        return EXIT_ERROR
    gate = json.loads(gate_path.read_text())
    decision = gate["decision"]
    m = gate["metrics"]
    print(
        f"{gate['run_id']}: {decision['status'].upper()} — "
        f"micro-F1 {m['micro_f1']:.3f} CI [{m['ci_lower']:.3f}, {m['ci_upper']:.3f}] "
        f"(target {gate.get('thresholds', {}).get('target_f1', '—')})"
    )
    for b in gate.get("blockers", []):
        print(f"  blocker: {b}")
    for w in gate.get("warnings", []):
        print(f"  warning: {w}")
    if decision["status"] == "error":
        return EXIT_ERROR
    return EXIT_PROMOTE if decision["promote"] else EXIT_BLOCKED


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    for name in ("gate.json",):
        p = run_dir / name
        if p.exists():
            print(f"== {name} ==")
            print(json.dumps(json.loads(p.read_text()), indent=2, sort_keys=True))
    events = run_dir / "events.jsonl"
    if events.exists():
        print(f"== events ({sum(1 for _ in events.open())} lines) ==")
        for line in events.open():
            e = json.loads(line)
            print(f"  {e['ts']} [{e['stage']}] {e['event']}")
    return EXIT_PROMOTE


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Apply human labels to compute κ (§6.4)."""
    labels = [json.loads(l) for l in Path(args.labels).read_text().splitlines() if l.strip()]
    human = [l["human"] for l in labels]
    judge = [l["judge"] for l in labels]
    cfg = EvalConfig.from_yaml(args.config) if args.config else None
    gate_obj = CalibrationGate(cfg.judge) if cfg else None
    if gate_obj is None:
        # Minimal default gate for label-only computation.
        from slm_eval.config import JudgeConfig

        gate_obj = CalibrationGate(JudgeConfig())
    status = gate_obj.evaluate(human, judge, resamples=args.resamples)
    blocked, reason = gate_obj.blocks_promotion(status)
    out = {
        "kappa": round(status.kappa, 4), "pairs": status.pairs,
        "ci": [round(status.ci_lower, 4), round(status.ci_upper, 4)],
        "model": status.model, "blocks_promotion": blocked, "reason": reason,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return EXIT_BLOCKED if blocked else EXIT_PROMOTE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slm-eval", description=__doc__)
    p.add_argument("--version", action="version", version=f"slm-eval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("init", help="scaffold config + dirs + db")
    c.add_argument("--dir", default=".")
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=_cmd_init)

    c = sub.add_parser("run", help="run the eval gate")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("--model")
    c.add_argument("--dataset")
    c.add_argument("--runs-dir", default="runs")
    c.add_argument("--run-id")
    c.add_argument("--fixture-count", type=int, default=200)
    c.add_argument("--resume", action="store_true")
    c.add_argument(
        "--judge-endpoint",
        help="Override judge endpoint. Falls back to a fixture judge when no "
        "reachable endpoint/API key is available (M1 dev loop).",
    )
    c.set_defaults(func=_cmd_run)

    c = sub.add_parser("gate", help="print gate summary + exit code")
    c.add_argument("run_dir")
    c.set_defaults(func=_cmd_gate)

    c = sub.add_parser("inspect", help="provenance + timeline")
    c.add_argument("run_dir")
    c.set_defaults(func=_cmd_inspect)

    c = sub.add_parser("calibrate", help="compute κ from human labels")
    c.add_argument("action", choices=["apply"])
    c.add_argument("--labels", required=True)
    c.add_argument("-c", "--config")
    c.add_argument("--resamples", type=int, default=2000)
    c.set_defaults(func=_cmd_calibrate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SlmEvalError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
