"""CLI (SPEC_V3 §8): init, run, gate, calibrate, report, inspect.

Exit codes are the CI contract (§8.2): 0 promote · 1 blocked · 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from slm_eval import __version__, fixture
from slm_eval.backends.modal import add_modal_subparsers
from slm_eval.config import EvalConfig
from slm_eval.errors import SlmEvalError
from slm_eval.idiot_index import run_idiot_index
from slm_eval.judge import CalibrationGate, CalibrationStatus
from slm_eval.llm import FixtureJudge, LLMClient, OpenAICompatClient
from slm_eval.pipeline import Orchestrator
from slm_eval.store import Store, _now
from slm_eval.synthetic import generate_synthetic_cases, inject_synthetic_cases

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


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch a vLLM OpenAI-compatible server for the model under test."""
    from slm_eval.serving import ServingError, serve_sync

    cfg = EvalConfig.from_yaml(args.config)
    try:
        return serve_sync(cfg, args.model)
    except ServingError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


def _cmd_data(args: argparse.Namespace) -> int:
    """Data pipeline: load, validate, analyze, split."""
    from slm_eval.data_pipeline import DataPipeline

    pipeline = DataPipeline(data_dir=Path(args.cases_dir))

    # Determine the data source
    cases_path = Path(args.cases_dir) / "holdout" / "cases.jsonl"
    alt_cases_path = Path(args.cases_dir) / "cases.jsonl"

    if cases_path.exists():
        print(f"Loading cases from {cases_path}")
        cases = pipeline.load_cases(cases_path)
    elif alt_cases_path.exists():
        print(f"Loading cases from {alt_cases_path}")
        cases = pipeline.load_cases(alt_cases_path)
    else:
        print(f"ERROR: cases.jsonl not found in {args.cases_dir}/holdout/ or {args.cases_dir}/")
        return EXIT_ERROR

    print(f"Loaded {len(cases)} cases")

    # Validate cases
    print("\nRunning data validation...")
    report = pipeline.validate(cases)
    print(f"Valid: {report.valid_cases}/{report.total_cases}")
    print(f"Invalid: {report.invalid_cases}")

    if report.schema_errors:
        print("Schema errors (first 5):")
        for err in report.schema_errors[:5]:
            print(f"  - {err}")

    # Print recommendations
    if report.recommendations:
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Analyze with MinHash dedup
    src_path = cases_path if cases_path.exists() else alt_cases_path
    print(f"\nRunning dataset analysis (minhash threshold: {args.minhash_threshold})...")
    analysis = pipeline.analyze(
        src_path,
        minhash_threshold=args.minhash_threshold,
    )
    print(f"Total cases: {analysis.get('n_cases', 'N/A')}")
    print(f"Fields seen: {analysis.get('n_fields', 'N/A')}")
    print(f"Duplicate pairs: {len(analysis.get('duplicate_pairs', []))}")

    # Stratified split
    print(f"\nPreparing stratified split (holdout fraction: {args.holdout_fraction})")
    if args.strata_keys is None:
        split_result = pipeline.prepare_split(
            cases=cases, strata_keys=None, holdout_fraction=args.holdout_fraction, seed=args.seed
        )
    else:
        split_result = pipeline.prepare_split(
            cases=cases,
            strata_keys=args.strata_keys,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
    print(f"Train cases: {len(split_result['train'])}")
    print(f"Holdout cases: {len(split_result['holdout'])}")

    # Save splits
    runs_dir = Path(args.cases_dir)
    output_dir = runs_dir / "splits"
    counts = pipeline.save_splits(split_result, output_dir)
    print(f"Saved {counts['train']} training cases to {output_dir / 'train.jsonl'}")
    print(f"Saved {counts['holdout']} holdout cases to {output_dir / 'holdout.jsonl'}")

    # Also save in training format for LoRA fine-tuning
    training_dir = runs_dir / "training_format"
    pipeline.to_training_format(split_result["train"], training_dir / "train.jsonl")
    pipeline.to_training_format(split_result["holdout"], training_dir / "holdout.jsonl")
    print(f"Saved training-format cases to {training_dir}/")

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


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars.

    Keeps secrets out of the config file: eval_config.yaml names the env var
    (api_key_env), .env supplies its value, and .env stays gitignored.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _file_sha256(path: str | Path) -> str:
    """Provenance hash of a model file (§4.3). 'unknown' when not on disk."""
    import hashlib

    p = Path(path)
    if not p.is_file():
        return "unknown"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_calibration(cfg: EvalConfig) -> CalibrationStatus | None:
    """Load human/judge label pairs and evaluate κ (§6.4).

    Returns None when no calibration file exists — the gate then reports
    ``kappa_missing`` and blocks any promotion with subjective fields, which is
    the correct conservative behavior (AC-05).
    """
    path = Path(cfg.judge.calibration.path)
    if not path.exists():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    human = [str(r["human"]) for r in rows]
    judge = [str(r["judge"]) for r in rows]
    if len(human) < cfg.judge.calibration.min_pairs:
        return CalibrationStatus(
            kappa=float("nan"), pairs=len(human), ci_lower=0.0, ci_upper=0.0,
            model=cfg.judge.model,
        )
    gate = CalibrationGate(cfg.judge)
    return gate.evaluate(human, judge, resamples=2000)


def _make_predictor(cfg: EvalConfig, model_label: str):
    """Build the model-under-test client for live inference (--infer)."""
    from slm_eval.infer import infer_case

    client = OpenAICompatClient(
        base_url=cfg.inference.base_url,
        model=model_label,
        api_key="local",
        timeout=cfg.inference.timeout_seconds,
        max_attempts=cfg.inference.retry.max_attempts,
        backoff_base=cfg.inference.retry.backoff_base,
    )
    return lambda case: infer_case(client, cfg, case["text"])


def _cmd_run(args: argparse.Namespace) -> int:
    # Idiot Index Gate (default on) — audit config before running
    if args.idiot_index:
        report = run_idiot_index(args.config)
        if report.cuttable > 0:
            print(report.summary(), file=sys.stderr)
            print(f"\nIdiot Index: {report.cuttable} cuttable component(s). Fix or run with --no-idiot-index.", file=sys.stderr)
            return EXIT_ERROR
        elif report.questionable > 2:
            print(report.summary(), file=sys.stderr)
            print(f"\nIdiot Index: {report.questionable} questionable components — proceeding but review recommended.", file=sys.stderr)
        else:
            print(f"Idiot Index: ✓ lean ({report.justified} justified, {report.questionable} questionable)", file=sys.stderr)

    cfg = EvalConfig.from_yaml(args.config)
    run_dir = Path(args.runs_dir) / (args.run_id or _run_id())
    model_path = args.model or "fixture-model.gguf"
    store = Store(Path(args.config).parent / "slm_eval.db")
    try:
        orch = Orchestrator(
            cfg, store, run_dir, _make_judge_client(cfg, args.judge_endpoint),
            code_version=f"git:{__version__}", resume=args.resume,
            config_path=args.config, force_cost=args.force,
        )
        if args.dataset:
            dataset = args.dataset
        else:
            # Generate the golden fixture as a stand-in dataset (M1 dev loop).
            dataset = str(run_dir / "data")
            fixture.write_dataset(dataset, count=args.fixture_count, seed=cfg.split.seed)

        # Model-under-test wiring. Explicit modes only — gold-as-prediction must
        # be requested (--fixture-gold) so a dev run can never masquerade as a
        # measured one, and live inference requires a reachable endpoint.
        predictor, predictions = None, None
        server = None
        if args.auto_serve and not args.fixture_gold:
            from slm_eval.serving import ServingError, VllmServer, build_spec_from_config

            spec = build_spec_from_config(cfg, args.model or model_path)
            cfg.inference.base_url = spec.base_url  # route inference at vLLM
            try:
                server = VllmServer(spec).start()
            except ServingError as e:
                print(f"error: {e}", file=sys.stderr)
                return EXIT_ERROR
        try:
            if args.fixture_gold:
                print(
                    "warning: --fixture-gold scores the dataset against its OWN gold — "
                    "metrics are meaningless (dev loop only)",
                    file=sys.stderr,
                )
                predictions = {
                    c["case_id"]: c["gold"] for c in fixture.read_dataset(dataset)
                }
            else:
                if not args.auto_serve and not _endpoint_reachable(cfg.inference.base_url):
                    print(
                        f"error: inference endpoint {cfg.inference.base_url} is not "
                        "reachable.\n  Start your model server (e.g. llama-server) and "
                        "re-run, or pass\n  --fixture-gold for a dev-loop run that scores "
                        "against the dataset's own gold,\n  or pass --auto-serve to launch "
                        "a vLLM server for --model.",
                        file=sys.stderr,
                    )
                    return EXIT_ERROR
                if args.infer and not _endpoint_reachable(cfg.inference.base_url):
                    print(
                        f"error: --infer requested but {cfg.inference.base_url} "
                        "is not reachable",
                        file=sys.stderr,
                    )
                    return EXIT_ERROR
                predictor = _make_predictor(cfg, Path(model_path).name)

            gate = orch.run(
                model_path=model_path,
                model_sha256=_file_sha256(model_path),
                dataset_path=dataset,
                predictions=predictions,
                predictor=predictor,
                calibration=_load_calibration(cfg),
            )
        finally:
            if server is not None:
                server.stop()
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
    dd = gate.get("dataset_defect")
    if dd and dd.get("defect_n"):
        print(
            f"  dataset_defect: {dd['defect_n']}/{dd['total_n']} cases "
            f"({dd['rate']:.0%}) unextractable — excluded from micro-F1 "
            f"(scored {dd['scored_n']})"
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
    for name in ("gate.json", "predictions.jsonl", "scores.jsonl", "synthetic_proposals.jsonl"):
        p = run_dir / name
        if p.exists():
            print(f"== {name} ==")
            if name.endswith(".jsonl"):
                for line in p.open():
                    if line.strip():
                        print(json.dumps(json.loads(line), indent=2, sort_keys=True))
            else:
                print(json.dumps(json.loads(p.read_text()), indent=2, sort_keys=True))
    events = run_dir / "events.jsonl"
    if events.exists():
        print(f"== events ({sum(1 for _ in events.open())} lines) ==")
        for line in events.open():
            e = json.loads(line)
            print(f"  {e['ts']} [{e['stage']}] {e['event']}")
    return EXIT_PROMOTE


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate markdown report from gate.json + review queue."""
    from datetime import UTC, datetime

    from jinja2 import Environment, FileSystemLoader
    
    run_dir = Path(args.run_dir)
    gate_file = run_dir / "gate.json"
    if not gate_file.exists():
        print(f"Gate file not found: {gate_file}", file=sys.stderr)
        return EXIT_ERROR
    
    gate = json.loads(gate_file.read_text())
    
    # Load review tasks if config provided
    reviews = []
    error_categories = {}
    if args.config:
        store = Store(Path(args.config).parent / "slm_eval.db")
        try:
            run_id = gate.get("run_id", run_dir.name)
            reviews = store.get_review_tasks(run_id)
            error_categories = store.get_error_summary(run_id)
        finally:
            store.close()
    
    # Prepare template context
    ctx = {
        "run_id": gate.get("run_id", run_dir.name),
        "model_path": gate.get("model", {}).get("path", "unknown"),
        "model_sha256": gate.get("model", {}).get("sha256", "unknown"),
        "dataset_path": "unknown",
        "holdout_hash": gate.get("holdout", {}).get("hash", "unknown"),
        "config_hash": gate.get("provenance", {}).get("config_hash", "unknown"),
        "generated_at": datetime.now(UTC).isoformat(),
        "decision_status": gate.get("decision", {}).get("status", "unknown"),
        "decision_promote": gate.get("decision", {}).get("promote", False),
        "decision_reason": gate.get("decision", {}).get("reason", "unknown"),
        "micro_f1": gate.get("metrics", {}).get("micro_f1", 0.0),
        "ci_lower": gate.get("metrics", {}).get("ci_lower", 0.0),
        "ci_upper": gate.get("metrics", {}).get("ci_upper", 0.0),
        "target": gate.get("thresholds", {}).get("micro_f1", 0.85),
        "resamples": gate.get("metrics", {}).get("resamples", 0),
        "seed": gate.get("metrics", {}).get("seed", 0),
        "overall_n": gate.get("metrics", {}).get("n", gate.get("holdout", {}).get("n", 0)),
        "strata": gate.get("strata", []),
        "dataset_defect": gate.get("dataset_defect", {}),
        "kappa": gate.get("judge", {}).get("kappa"),
        "kappa_pairs": gate.get("judge", {}).get("pairs"),
        "kappa_required": bool(gate.get("judge", {}).get("kappa_required", False)),
        "judge_model": gate.get("judge", {}).get("model", "unknown"),
        "cost_usd": gate.get("cost_usd", 0.0),
        "blockers": gate.get("blockers", []),
        "warnings": gate.get("warnings", []),
        "reviews": reviews,
        "error_categories": error_categories,
        "code_version": gate.get("provenance", {}).get("code_version", "unknown"),
        "run_dir": str(run_dir),
        "schema_version": gate.get("schema_version", 1),
    }
    
    template_path = Path(__file__).parent / "templates" / "report.md.j2"
    if not template_path.exists():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return EXIT_ERROR
    
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    output = template.render(**ctx)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
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

    c = sub.add_parser("data", help="data pipeline: load, validate, analyze, split")
    c.add_argument("--cases-dir", default="runs", help="Directory containing cases.jsonl")
    c.add_argument("--minhash-threshold", type=float, default=0.8, help="Minimum similarity for near-duplicate detection")
    c.add_argument("--holdout-fraction", type=float, default=0.2, help="Fraction of data to use as holdout")
    c.add_argument("--strata-keys", nargs="+", default=None, help="Keys to use for stratified splitting")
    c.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    c.set_defaults(func=_cmd_data)

    c = sub.add_parser("serve", help="launch vLLM OpenAI-compatible server")
    c.add_argument("-c", "--config", required=True, help="Path to eval_config.yaml")
    c.add_argument("--model", required=True, help="Model repo/path to serve")
    c.set_defaults(func=_cmd_serve)

    c = sub.add_parser("run", help="run the eval gate")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("--model")
    c.add_argument("--dataset")
    c.add_argument("--runs-dir", default="runs")
    c.add_argument("--run-id")
    c.add_argument("--fixture-count", type=int, default=200)
    c.add_argument("--resume", action="store_true")
    c.add_argument(
        "--force",
        action="store_true",
        help="Continue when the round cost cap is exceeded (records a warning).",
    )
    c.add_argument(
        "--infer",
        action="store_true",
        help="Run the model under test at inference.base_url (default when the "
        "endpoint is reachable). Requires a live model server.",
    )
    c.add_argument(
        "--auto-serve",
        action="store_true",
        help="Launch a vLLM server for --model (backend=vllm) around this run. "
        "Adopts an existing server on the port if one is already answering.",
    )
    c.add_argument(
        "--fixture-gold",
        action="store_true",
        help="DEV ONLY: score the dataset against its own gold. Metrics are "
        "meaningless; use to exercise pipeline mechanics without a model.",
    )
    c.add_argument(
        "--judge-endpoint",
        help="Override judge endpoint. Falls back to a fixture judge when no "
        "reachable endpoint/API key is available (M1 dev loop).",
    )
    c.add_argument(
        "--idiot-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Idiot Index audit before execution (default: on). Use --no-idiot-index to skip.",
    )
    c.set_defaults(func=_cmd_run)

    c = sub.add_parser("gate", help="print gate summary + exit code")
    c.add_argument("run_dir")
    c.set_defaults(func=_cmd_gate)

    c = sub.add_parser("inspect", help="provenance + timeline")
    c.add_argument("run_dir")
    c.set_defaults(func=_cmd_inspect)

    c = sub.add_parser("report", help="generate markdown report from gate.json")
    c.add_argument("run_dir")
    c.add_argument("-c", "--config", help="Config file (for review queue + error categories)")
    c.add_argument("-o", "--output", help="Output file path")
    c.set_defaults(func=_cmd_report)

    c = sub.add_parser("calibrate", help="compute κ from human labels")
    c.add_argument("action", choices=["apply"])
    c.add_argument("--labels", required=True)
    c.add_argument("-c", "--config")
    c.add_argument("--resamples", type=int, default=2000)
    c.set_defaults(func=_cmd_calibrate)

    c = sub.add_parser("review", help="human review queue (advisory lane)")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("run_id")
    c.add_argument("action", choices=["list", "assign", "resolve", "reopen", "blocked", "categorize", "categories", "comment"])
    c.add_argument("--case-id")
    c.add_argument("--field", help="field name for categorize")
    c.add_argument("--reason", help="review reason (e.g. 'judge_disagreement', 'low_confidence', 'analyst_flag')")
    c.add_argument("--decision", choices=["approve", "reject", "escalate", "relabel"])
    c.add_argument("--notes", default="")
    c.add_argument("--due-at", help="ISO8601 due date")
    c.add_argument("--state", choices=["pending", "decided"], help="filter for list")
    c.add_argument("--category", help="error category for categorize")
    c.add_argument("--subcategory", help="error subcategory for categorize")
    c.add_argument("--confidence", type=float, default=1.0, help="confidence 0-1 for categorize")
    c.add_argument("--reviewer", help="reviewer name for categorize/comment")
    c.add_argument("--comment", help="comment text for comment action")
    c.set_defaults(func=_cmd_review)

    c = sub.add_parser("idiot-index", help="audit config for over-engineering (Idiot Index Gate)")
    c.add_argument("-c", "--config", required=True)
    c.set_defaults(func=_cmd_idiot_index)

    c = sub.add_parser("train", help="LoRA fine-tuning with evaluation integration")
    c.add_argument("-c", "--config", required=True, help="Path to eval_config.yaml")
    c.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model to fine-tune")
    c.add_argument("--dataset", default=None, help="Path to training dataset (JSONL)")
    c.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    c.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    c.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    c.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True, help="Use LoRA for PEFT")
    c.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    c.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    c.add_argument("--output-dir", default="./models/fine_tuned", help="Output directory for checkpoints")
    c.add_argument("--run-id", default=None, help="Run ID for tracking")
    c.add_argument("--resume", default=None, help="Resume from checkpoint path")
    c.add_argument("--max-cost", type=float, default=50.0, help="Max cost per round in USD")
    c.set_defaults(func=_cmd_train)

    c = sub.add_parser("synthetic", help="generate synthetic cases (M3)")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("--run-dir", required=True, help="Run directory (for holdout + output)")
    c.add_argument("--n", type=int, default=20, help="Number of cases to propose")
    c.add_argument("--threshold", type=float, default=0.6, help="MinHash similarity threshold")
    c.add_argument("--model", help="Override model for generation")
    c.add_argument("--inject", action="store_true", help="Inject as review tasks (requires --run-id)")
    c.add_argument("--run-id", help="Run ID for injection")
    c.add_argument("--use-web", action="store_true", help="Use web search for real merchant data")
    c.add_argument("--web-provider", default="tavily", help="Web search provider (tavily, exa, etc.)")
    c.add_argument("--web-api-key", help="Web search API key (or set PROVIDER_API_KEY env var)")
    c.set_defaults(func=_cmd_synthetic)

    c = sub.add_parser("synthetic-loop", help="two-round synthetic loop (M3)")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("--dataset", required=True, help="Dataset path")
    c.add_argument("--model", required=True, help="Model path")
    c.add_argument("--runs-dir", default="runs", help="Runs base directory")
    c.add_argument("--rounds", type=int, default=2, help="Consecutive passing rounds required")
    c.add_argument("--n-per-round", type=int, default=10, help="Synthetic cases per round")
    c.add_argument("--threshold", type=float, default=0.6, help="MinHash threshold")
    c.add_argument("--no-idiot-index", action="store_true", help="Skip Idiot Index Gate")
    c.add_argument("--use-web", action="store_true", help="Use web search for real merchant data")
    c.add_argument("--web-provider", default="tavily", help="Web search provider (tavily, exa, etc.)")
    c.add_argument("--web-api-key", help="Web search API key (or set PROVIDER_API_KEY env var)")
    c.set_defaults(func=_cmd_synthetic_loop)

    # Modal GPU backend (Appendix A trigger: local inference too slow)
    add_modal_subparsers(sub)

    # Multi-provider support
    providers_parser = sub.add_parser("providers", help="List available providers/frameworks")
    providers_sub = providers_parser.add_subparsers(dest="providers_command", required=True)

    providers_list_parser = providers_sub.add_parser("list", help="List all providers")
    providers_list_parser.add_argument("--type", choices=["inference", "training", "serving", "all"], default="all", help="Provider type to list")
    providers_list_parser.set_defaults(func=_cmd_providers_list)

    providers_check_parser = providers_sub.add_parser("check", help="Check which providers are available")
    providers_check_parser.add_argument("--type", choices=["inference", "training", "serving", "all"], default="all", help="Provider type to check")
    providers_check_parser.set_defaults(func=_cmd_providers_check)

    return p


def _cmd_providers_list(args: argparse.Namespace) -> int:
    """List available providers/frameworks."""
    from slm_eval.providers import (
        list_providers,
        list_serving_providers,
        list_training_providers,
    )

    if args.type in ("inference", "all"):
        print("\n📦 Inference Providers:")
        print("-" * 40)
        for p in sorted(list_providers()):
            print(f"  • {p}")

    if args.type in ("training", "all"):
        print("\n🏋️ Training Frameworks:")
        print("-" * 40)
        for p in sorted(list_training_providers()):
            print(f"  • {p}")

    if args.type in ("serving", "all"):
        print("\n🚀 Serving Frameworks:")
        print("-" * 40)
        for p in sorted(list_serving_providers()):
            print(f"  • {p}")

    return EXIT_PROMOTE


def _cmd_providers_check(args: argparse.Namespace) -> int:
    """Check which providers are available."""
    from slm_eval.providers import (
        get_provider,
        get_serving_provider,
        get_training_provider,
        list_providers,
        list_serving_providers,
        list_training_providers,
    )

    results: dict[str, list[tuple[str, bool, str]]] = {
        "inference": [],
        "training": [],
        "serving": [],
    }

    if args.type in ("inference", "all"):
        for name in sorted(list_providers()):
            try:
                provider = get_provider(name, config=None)  # type: ignore
                available = provider.is_available()
                results["inference"].append((name, available, ""))
            except Exception as e:
                results["inference"].append((name, False, str(e)[:50]))

    if args.type in ("training", "all"):
        for name in sorted(list_training_providers()):
            try:
                provider = get_training_provider(name)
                available = provider.is_available()
                results["training"].append((name, available, ""))
            except Exception as e:
                results["training"].append((name, False, str(e)[:50]))

    if args.type in ("serving", "all"):
        for name in sorted(list_serving_providers()):
            try:
                provider = get_serving_provider(name)
                available = provider.is_available()
                results["serving"].append((name, available, ""))
            except Exception as e:
                results["serving"].append((name, False, str(e)[:50]))

    # Print results
    for provider_type, items in results.items():
        if not items:
            continue
        print(f"\n{provider_type.upper()} PROVIDERS:")
        print("-" * 60)
        for name, available, error in items:
            status = "✅" if available else "❌"
            error_msg = f" ({error})" if error else ""
            print(f"  {status} {name}{error_msg}")

    # Summary
    total = sum(len(v) for v in results.values())
    available = sum(1 for v in results.values() for _, a, _ in v if a)
    print(f"\nSummary: {available}/{total} providers available")

    return EXIT_PROMOTE if available > 0 else EXIT_BLOCKED


def _cmd_synthetic_loop(args: argparse.Namespace) -> int:
    """Run two-round synthetic loop: generate → review → inject → eval → repeat until N consecutive passes."""
    cfg = EvalConfig.from_yaml(args.config)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    consecutive_passes = 0
    round_num = 0
    
    while consecutive_passes < args.rounds:
        round_num += 1
        run_id = f"synth-loop-r{round_num}"
        run_dir = runs_dir / run_id
        
        print(f"\n{'='*60}")
        print(f"SYNTHETIC LOOP ROUND {round_num} (consecutive passes: {consecutive_passes}/{args.rounds})")
        print(f"{'='*60}")
        
        # 1. Run evaluation
        cmd = [
            sys.executable, "-m", "slm_eval.cli", "run",
            "-c", args.config,
            "--dataset", args.dataset,
            "--model", args.model,
            "--runs-dir", args.runs_dir,
            "--run-id", run_id,
        ]
        if args.no_idiot_index:
            cmd.append("--no-idiot-index")
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, check=False)
        if result.returncode != 0 and result.returncode != 1:  # 1 = blocked is OK
            print(f"Run failed with exit code {result.returncode}")
            return EXIT_ERROR
        
        # 2. Check gate
        gate_file = run_dir / "gate.json"
        if not gate_file.exists():
            print(f"Gate file not found: {gate_file}")
            return EXIT_ERROR
        
        gate = json.loads(gate_file.read_text())
        promote = gate.get("decision", {}).get("promote", False)
        
        if promote:
            consecutive_passes += 1
            print(f"✅ Round {round_num} PASSED (consecutive: {consecutive_passes}/{args.rounds})")
            if consecutive_passes >= args.rounds:
                print(f"\n🎉 Synthetic loop completed: {args.rounds} consecutive passes!")
                return EXIT_PROMOTE
        else:
            consecutive_passes = 0
            print(f"❌ Round {round_num} BLOCKED: {gate.get('decision', {}).get('reason', 'unknown')}")
            
            # 3. Generate synthetic cases
            print(f"Generating {args.n_per_round} synthetic cases...")
            result = generate_synthetic_cases(
                cfg, run_dir / "holdout", args.n_per_round, args.threshold,
                use_web=args.use_web,
                web_provider=args.web_provider,
                web_api_key=args.web_api_key,
            )
            print(f"Proposed: {result.proposed}, Novel: {result.novel}, Duplicate: {result.duplicate}")
            
            if result.novel == 0:
                print("No novel cases generated. Stopping.")
                return EXIT_ERROR
            
            # 4. Inject as review tasks
            store = Store(Path(args.config).parent / "slm_eval.db")
            try:
                inserted = inject_synthetic_cases(store, run_id, result.cases)
                print(f"Injected {inserted} cases as review tasks")
            finally:
                store.close()
            
            print(f"\n⏳ Awaiting human review for run {run_id}...")
            print(f"   Run: slm-eval review -c {args.config} {run_id} list")
            print(f"   Resolve with: slm-eval review -c {args.config} {run_id} resolve --case-id <id> --reason synthetic_injection --decision approve")
            
            # 5. Wait for reviews to be resolved
            while True:
                store = Store(Path(args.config).parent / "slm_eval.db")
                try:
                    synthetic_pending = [
                        t for t in store.get_review_tasks(run_id, "pending")
                        if t["reason"] == "synthetic_injection"
                    ]
                    if not synthetic_pending:
                        break
                    print(f"   {len(synthetic_pending)} synthetic reviews pending... waiting 30s")
                finally:
                    store.close()
                time.sleep(30)
            
            print("✅ All synthetic reviews resolved. Continuing to next round.")
    
    return EXIT_PROMOTE


def _cmd_synthetic(args: argparse.Namespace) -> int:
    """Generate synthetic cases with MinHash deduplication."""
    cfg = EvalConfig.from_yaml(args.config)
    run_dir = Path(args.run_dir)
    holdout_dir = run_dir / "holdout"

    if not holdout_dir.exists():
        print(f"Holdout dir not found: {holdout_dir}", file=sys.stderr)
        return EXIT_ERROR

    result = generate_synthetic_cases(
        cfg, holdout_dir, args.n, args.threshold, 
        model=args.model,
        use_web=args.use_web,
        web_provider=args.web_provider,
        web_api_key=args.web_api_key,
    )

    print(f"Proposed: {result.proposed}, Novel: {result.novel}, Duplicate: {result.duplicate}")

    out_path = run_dir / "synthetic_proposals.jsonl"
    with out_path.open("w") as f:
        for c in result.cases:
            f.write(json.dumps({
                "case_id": c.case_id,
                "text": c.text,
                "gold": c.gold,
                "strata": c.strata,
                "minhash": c.minhash,
            }, sort_keys=True) + "\n")

    print(f"Written to {out_path}")

    if args.inject:
        if not args.run_id:
            print("error: --inject requires --run-id", file=sys.stderr)
            return EXIT_ERROR
        store = Store(Path(args.config).parent / "slm_eval.db")
        try:
            inserted = inject_synthetic_cases(store, args.run_id, result.cases)
            print(f"Injected {inserted} cases as review tasks for run {args.run_id}")
        finally:
            store.close()

    return EXIT_PROMOTE


def _cmd_review(args: argparse.Namespace) -> int:
    """Human review queue operations."""
    store = Store(Path(args.config).parent / "slm_eval.db")
    try:
        if args.action == "list":
            tasks = store.get_review_tasks(args.run_id, state=args.state)
            if not tasks:
                print(f"No review tasks for run {args.run_id}" + (f" (state={args.state})" if args.state else ""))
                return EXIT_PROMOTE
            for t in tasks:
                print(f"  {t['case_id']} | {t['reason']} | {t['state']} | assigned={t['assigned_at']} | due={t.get('due_at', '—')} | decision={t.get('decision', '—')}")
            return EXIT_PROMOTE

        elif args.action == "assign":
            if not args.case_id or not args.reason:
                print("error: --case-id and --reason required for assign", file=sys.stderr)
                return EXIT_ERROR
            store.insert_review_tasks(args.run_id, [{
                "run_id": args.run_id, "case_id": args.case_id, "reason": args.reason,
                "assigned_at": _now(), "due_at": args.due_at,
            }])
            print(f"Assigned review: {args.case_id} ({args.reason})")
            return EXIT_PROMOTE

        elif args.action == "resolve":
            if not args.case_id or not args.reason or not args.decision:
                print("error: --case-id, --reason, --decision required for resolve", file=sys.stderr)
                return EXIT_ERROR
            store.update_review_task(args.run_id, args.case_id, args.reason, args.decision, args.notes or "")
            print(f"Resolved review: {args.case_id} ({args.reason}) -> {args.decision}")
            return EXIT_PROMOTE

        elif args.action == "reopen":
            if not args.case_id or not args.reason:
                print("error: --case-id and --reason required for reopen", file=sys.stderr)
                return EXIT_ERROR
            store.reopen_review_task(args.run_id, args.case_id, args.reason)
            print(f"Reopened review: {args.case_id} ({args.reason})")
            return EXIT_PROMOTE

        elif args.action == "blocked":
            # Check if any pending reviews block the gate
            n = store.pending_review_count(args.run_id)
            if n > 0:
                print(f"REVIEW_BLOCKED: {n} pending review(s) for run {args.run_id}")
                return EXIT_BLOCKED
            print(f"REVIEW_CLEAR: 0 pending reviews for run {args.run_id}")
            return EXIT_PROMOTE

        elif args.action == "categorize":
            if not args.case_id or not args.field or not args.category:
                print("error: --case-id, --field, --category required for categorize", file=sys.stderr)
                return EXIT_ERROR
            new = store.insert_error_category(
                args.run_id, args.case_id, args.field, args.category,
                args.subcategory or "", args.confidence, args.reviewer or ""
            )
            print(f"Error categorized: {args.case_id}.{args.field} -> {args.category}" + ("" if new else " (updated)"))
            return EXIT_PROMOTE

        elif args.action == "categories":
            summary = store.get_error_summary(args.run_id)
            if not summary:
                print(f"No error categories for run {args.run_id}")
            else:
                for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
                    print(f"  {cat}: {count}")
            return EXIT_PROMOTE

        elif args.action == "comment":
            if not args.case_id or not args.reason or not args.reviewer or not args.comment:
                print("error: --case-id, --reason, --reviewer, --comment required", file=sys.stderr)
                return EXIT_ERROR
            store.add_review_comment(args.run_id, args.case_id, args.reason, args.reviewer, args.comment)
            print(f"Comment added: {args.case_id} ({args.reason}) by {args.reviewer}")
            return EXIT_PROMOTE

    finally:
        store.close()
    return EXIT_ERROR


def _cmd_idiot_index(args) -> int:
    report = run_idiot_index(args.config)
    print(report.summary())
    return report.exit_code()


def _cmd_train(args) -> int:
    """LoRA fine-tuning with evaluation integration."""
    import json
    import os
    import sys
    from pathlib import Path

    # Budget guard: the eval round that validates this adapter must fit --max-cost.
    try:
        eval_cfg = EvalConfig.from_yaml(args.config)
    except SlmEvalError as e:
        print(f"error: cannot read eval config {args.config}: {e}", file=sys.stderr)
        return EXIT_ERROR
    round_cap = eval_cfg.cost_control.max_cost_per_round_usd
    if round_cap > args.max_cost:
        print(
            f"error: eval round cap ${round_cap:.2f} exceeds --max-cost "
            f"${args.max_cost:.2f}; raise --max-cost or lower cost_control.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    import torch
    import wandb
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Initialize wandb
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="slm-training",
            name=args.run_id or f"train-{int(time.time())}",
            config=vars(args),
            save_code=True,
        )

    print("Starting LoRA fine-tuning...")
    print(f"  Model: {args.model}")
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA: {args.use_lora} (r={args.lora_r}, alpha={args.lora_alpha})")

    # Prepare dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}", file=sys.stderr)
            return 2
        dataset = load_dataset("json", data_files=str(dataset_path))
    else:
        # Generate synthetic data for testing
        print("No dataset provided, generating synthetic data...")
        data = [{"text": "Extract receipt: total amount 100.00, items: coffee, sandwich, drink."} for _ in range(100)]
        dataset = load_dataset("json", data=data)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"LoRA applied. Trainable params: {trainable:,} / {total:,}")

    # Training arguments — TRL >= 1.0 expects the SFT-specific knobs on
    # SFTConfig (dataset_text_field / max_length / packing), not on the trainer.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        warmup_steps=min(100, max(1, args.epochs)),
        gradient_checkpointing=has_cuda,
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=args.run_id or f"train-{int(time.time())}",
        max_length=1024,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        print("Training completed successfully!")
    except Exception as e:  # noqa: BLE001 — surface any trainer failure as exit 2
        print(f"Training error: {e}", file=sys.stderr)
        return 2

    # Save final model
    final_model_path = Path(args.output_dir) / f"{args.run_id or 'model'}_final"
    if trainer.is_world_process_zero():
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Save training config (drop the argparse handler, which isn't JSON)
        config_path = Path(args.output_dir) / f"{args.run_id or 'model'}_config.json"
        train_config = {k: v for k, v in vars(args).items() if not callable(v)}
        with open(config_path, "w") as f:
            json.dump(train_config, f, indent=2)

        print(f"Model saved to: {final_model_path}")
        print(f"Config saved to: {config_path}")

    if wandb.run:
        wandb.finish()

    print("\n=== Training Complete ===")
    print(f"Run ID: {args.run_id or 'model'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size: {trainer.model.num_parameters():,} parameters")
    print(f"Learnable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")

    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SlmEvalError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
