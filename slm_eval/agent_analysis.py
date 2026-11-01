"""agent_analysis.py — Agent-facing analysis tools.

Parallel to agent_idiot.py: provides JSON-callable functions that the LLM
agent can invoke to automatically analyze the evaluation harness state
before/during/after a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_eval.analysis.dataset import DatasetAnalyzer
from slm_eval.analysis.error import ErrorAnalyzer
from slm_eval.analysis.gate import GateAnalyzer, load_gate_for_analyzer


def dataset_analysis_tool(config_path: str, cases_path: str, minhash_threshold: float = 0.6) -> dict[str, Any]:
    """Tool for agents: analyze dataset readiness before running pipeline.

    Returns structured JSON with defect rate estimate, duplicate pairs,
    difficult fields, and OCR signal quality.

    Args:
        config_path: Path to eval_config.yaml
        cases_path: Path to cases.jsonl
        minhash_threshold: MinHash Jaccard threshold for duplicate detection

    Returns:
        Dict with exit_code, analysis summary, and structured data.
    """
    try:
        analyzer = DatasetAnalyzer(cases_path, minhash_threshold=minhash_threshold)
        analysis = analyzer.analyze()
        return {
            "exit_code": 0,
            "summary": analyzer.summary(),
            "data": {
                "n_cases": analysis.n_cases,
                "n_fields": analysis.n_fields,
                "fields_seen": sorted(analysis.fields_seen),
                "strata_distribution": analysis.strata_distribution,
                "defect_rate_estimate": analysis.defect_rate_estimate,
                "top_difficult_fields": analysis.top_difficult_fields,
                "duplicate_pairs": analysis.duplicate_pairs,
                "ocr_signal_distribution": analysis.ocr_signal_dist,
            },
        }
    except Exception as e:  # noqa: BLE001 - tool wrapper: return structured error, not raise
        return {"exit_code": 1, "error": str(e), "summary": "", "data": {}}


def error_analysis_tool(run_id: str, score_path: str, config_path: str) -> dict[str, Any]:
    """Tool for agents: analyze scoring results after pipeline stage.

    Returns per-field breakdown, error clusters, stratum performance,
    and actionable recommendations.

    Args:
        run_id: Run identifier
        score_path: Path to scores JSONL (case_id, field, tp, fp, fn, error_category)
        config_path: Path to eval_config.yaml (for strata keys)

    Returns:
        Dict with exit_code and structured error analysis.
    """
    try:
        analyzer = ErrorAnalyzer(score_path, config_path, run_id)
        analysis = analyzer.analyze()
        return {
            "exit_code": 0,
            "summary": analyzer.summary(),
            "data": {
                "n_cases": analysis.n_cases,
                "n_fields": analysis.n_fields,
                "total_tp": analysis.total_tp,
                "total_fp": analysis.total_fp,
                "total_fn": analysis.total_fn,
                "micro_f1": analysis.f1_point,
                "ci_lower": analysis.f1_lower,
                "ci_upper": analysis.f1_upper,
                "kappa": analysis.kappa,
                "defect_rate": analysis.defect_rate,
                "defect_n": analysis.defect_n,
                "error_clusters": analysis.error_clusters,
                "stratum_perf": analysis.stratum_perf,
                "recommendations": analysis.recommendations,
            },
        }
    except Exception as e:  # noqa: BLE001 - tool wrapper: return structured error, not raise
        return {"exit_code": 1, "error": str(e), "summary": "", "data": {}}


def gate_analysis_tool(gate_path: str, config_path: str, score_path: str | None = None) -> dict[str, Any]:
    """Tool for agents: explain gate.json decision and provide root-cause analysis.

    Returns why gate blocked/passed, specific blockers, and remediation
    recommendations to flip BLOCKED → PROMOTE.

    Args:
        gate_path: Path to gate.json
        config_path: Path to eval_config.yaml
        score_path: Path to scores JSONL (optional, for field-level drill-down)

    Returns:
        Dict with exit_code and gate analysis.
    """
    try:
        analyzer = GateAnalyzer(gate_path, config_path, score_path)
        analysis = analyzer.analyze()
        return {
            "exit_code": 0,
            "summary": analyzer.summary(),
            "data": {
                "run_id": analysis.run_id,
                "passed": analysis.passed,
                "status": "promote" if analysis.passed else "blocked",
                "micro_f1": analysis.micro_f1,
                "ci_lower": analysis.ci_lower,
                "ci_upper": analysis.ci_upper,
                "kappa": analysis.kappa,
                "n": analysis.n,
                "root_cause": analysis.root_cause,
                "blockers": analysis.blockers,
                "stratum_blockers": analysis.stratum_blockers,
                "remediation": analysis.remediation,
                "warnings": analysis.warnings,
            },
        }
    except Exception as e:  # noqa: BLE001 - tool wrapper: return structured error, not raise
        return {"exit_code": 1, "error": str(e), "summary": "", "data": {}}


def load_gate_for_analyzer_tool(run_dir: str, config_path: str) -> dict[str, Any]:
    """Convenience: load GateAnalyzer from a run directory.

    scan dir for gate.json + scores.jsonl and return structured analysis.
    """
    try:
        analyzer = load_gate_for_analyzer(Path(run_dir), Path(config_path))
        analysis = analyzer.analyze()
        return {
            "exit_code": 0,
            "summary": analyzer.summary(),
            "data": {
                "run_id": analysis.run_id,
                "status": "PROMOTED" if analysis.passed else "BLOCKED",
                "micro_f1": analysis.micro_f1,
                "ci_lower": analysis.ci_lower,
                "ci_upper": analysis.ci_upper,
                "kappa": analysis.kappa,
                "n": analysis.n,
                "root_cause": analysis.root_cause,
                "blockers": analysis.blockers,
                "stratum_blockers": [
                    {"stratum": sb["stratum"], "ci_upper": sb["ci_upper"], "n": sb["n"]}
                    for sb in analysis.stratum_blockers
                ],
                "remediation": analysis.remediation,
            },
        }
    except Exception as e:  # noqa: BLE001 - tool wrapper: return structured error, not raise
        return {"exit_code": 1, "error": str(e), "summary": "", "data": {}}


# Agent-use example:
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SLM Evaluation Harness Analysis Tools")
    sub = p.add_subparsers(dest="command")

    # dataset
    d = sub.add_parser("dataset")
    d.add_argument("--config", required=True)
    d.add_argument("--cases", required=True)
    d.add_argument("--threshold", type=float, default=0.6)

    # error
    e = sub.add_parser("error")
    e.add_argument("--run-id", required=True)
    e.add_argument("--score", required=True)
    e.add_argument("--config", required=True)

    # gate
    g = sub.add_parser("gate")
    g.add_argument("--gate", required=True)
    g.add_argument("--config", required=True)
    g.add_argument("--score", default=None)

    # load from run dir
    l = sub.add_parser("load")
    l.add_argument("--rundir", required=True)
    l.add_argument("--config", required=True)

    args = p.parse_args()

    if args.command == "dataset":
        result = dataset_analysis_tool(args.config, args.cases, args.threshold)
    elif args.command == "error":
        result = error_analysis_tool(args.run_id, args.score, args.config)
    elif args.command == "gate":
        result = gate_analysis_tool(args.gate, args.config, args.score)
    elif args.command == "load":
        result = load_gate_for_analyzer_tool(args.rundir, args.config)
    else:
        result = {"exit_code": 1, "error": "unknown command"}

    print(json.dumps(result, indent=2))