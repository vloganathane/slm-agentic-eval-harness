"""Agent-facing Idiot Index: LLM can call this to self-audit its own plans/configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from slm_eval.idiot_index import run_idiot_index


def idiot_index_tool(config_path: str) -> dict[str, Any]:
    """Tool for agents: run Idiot Index on a config, return structured JSON.

    Args:
        config_path: Path to eval_config.yaml

    Returns:
        Dict with findings, summary, and exit_code.
    """
    report = run_idiot_index(config_path)
    return {
        "exit_code": report.exit_code(),
        "summary": report.summary(),
        "components": {
            "total": report.total_components,
            "cuttable": report.cuttable,
            "questionable": report.questionable,
            "justified": report.justified,
        },
        "findings": [
            {
                "severity": f.severity,
                "component": f.component,
                "message": f.message,
                "savings": f.savings,
            }
            for f in report.findings
        ],
    }


def idiot_index_config_patch(config_path: str) -> dict[str, Any]:
    """Generate a minimal patch to fix cuttable findings (auto-fix where safe).

    Returns a JSON patch the agent can apply.
    """
    report = run_idiot_index(config_path)
    cfg = yaml.safe_load(Path(config_path).read_text())
    patches = []

    for f in report.findings:
        if f.severity == "cut":
            if f.component == "reporting":
                # Force single format: json
                patches.append({"op": "replace", "path": "/reporting/formats", "value": ["json"]})
            elif f.component == "kappa":
                # Drop subjective fields if no calibration exists
                if "subjective_fields" in cfg.get("fields", {}):
                    patches.append({"op": "remove", "path": "/fields/subjective_fields"})
            elif f.component == "judge_scope":
                # Move structured fields to exact_match
                subj = cfg.get("fields", {}).get("subjective_fields", [])
                exact = cfg.get("fields", {}).get("exact_match_fields", [])
                # Heuristic: total, date, invoice_number are exact
                for field in ["total", "date", "invoice_number", "tax_id"]:
                    if field in subj and field not in exact:
                        patches.append({"op": "add", "path": "/fields/exact_match_fields/-", "value": field})
                        patches.append({"op": "remove", "path": f"/fields/subjective_fields/{subj.index(field)}"})

        # Also auto-fix questionable findings that have clear remediation
        elif f.severity == "question":
            if f.component == "strata":
                # Single-key strata with high min_per_stratum → lower min_per_stratum
                # Note: schema only allows single_key mode, so we can't switch to random
                if cfg.get("thresholds", {}).get("min_per_stratum", 5) > 5:
                    patches.append({"op": "replace", "path": "/thresholds/min_per_stratum", "value": 5})
            elif f.component == "bootstrap.resamples":
                # Reduce resamples to 1000 (CI width within ~0.005 of B=10000; min valid is 1000)
                if cfg.get("stats", {}).get("resamples", 1000) > 1000:
                    patches.append({"op": "replace", "path": "/stats/resamples", "value": 1000})
            elif f.component == "synthetic_loop":
                # Reduce or zero out synthetic budget; alert_at must be <= max
                if cfg.get("cost_control", {}).get("max_cost_per_round_usd", 0) > 0:
                    patches.append({"op": "replace", "path": "/cost_control/max_cost_per_round_usd", "value": 0.0})
                    patches.append({"op": "replace", "path": "/cost_control/alert_at_usd", "value": 0.0})
            elif f.component == "human_review" and cfg.get("human_review", {}).get("timeout_hours", 0) > 0:
                # Disable advisory lane
                patches.append({"op": "replace", "path": "/human_review/timeout_hours", "value": 0})

    return {"patches": patches, "config_path": config_path}


# Agent-use example:
if __name__ == "__main__":
    import sys
    result = idiot_index_tool(sys.argv[1] if len(sys.argv) > 1 else "configs/cord.yaml")
    print(json.dumps(result, indent=2))