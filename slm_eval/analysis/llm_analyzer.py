"""LLM-enhanced analyzers — semantic analysis on top of heuristic results.

Uses the existing LLM client infrastructure (llama-server, OpenAI-compatible)
to provide:
- Natural language root cause analysis
- Semantic error pattern recognition
- Context-aware remediation recommendations
- Field-level deep-dive with reasoning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm_eval.analysis.dataset import DatasetAnalysis, DatasetAnalyzer
from slm_eval.analysis.error import ErrorAnalysis, ErrorAnalyzer
from slm_eval.analysis.gate import GateAnalysis, GateAnalyzer
from slm_eval.llm import LLMClient, LLMResponse, OpenAICompatClient


# --- LLM Analysis Prompts ----------------------------------------------------

DATASET_ANALYSIS_PROMPT = """You are a data quality expert analyzing a structured extraction dataset.

Dataset Summary:
- Total cases: {n_cases}
- Fields: {fields_seen}
- Strata distribution: {strata_distribution}
- Defect rate estimate: {defect_rate_estimate}
- OCR signal distribution: {ocr_signal_distribution}
- Duplicate pairs found: {duplicate_count}
- Top difficult fields (low coverage): {top_difficult_fields}

Provide a JSON response with:
{{
    "summary": "2-3 sentence natural language summary of dataset readiness",
    "risks": [
        {{"type": "risk_type", "severity": "high|medium|low", "description": "...", "affected_strata": [...]}}
    ],
    "recommendations": [
        {{"action": "...", "priority": 1, "rationale": "..."}}
    ],
    "data_quality_score": 0.0-1.0,
    "ready_for_eval": true/false
}}"""

ERROR_ANALYSIS_PROMPT = """You are an ML evaluation expert analyzing model errors on a structured extraction task.

Error Summary:
- Run ID: {run_id}
- Total cases: {n_cases}
- Fields evaluated: {n_fields}
- Micro-F1: {micro_f1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]
- Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}
- Kappa: {kappa}

Top Error Clusters:
{error_clusters}

Stratum Performance:
{stratum_perf}

Field-level F1 (worst first):
{field_f1}

Provide a JSON response with:
{{
    "summary": "2-3 sentence natural language summary of model failure modes",
    "root_causes": [
        {{"cause": "...", "evidence": "...", "affected_fields": [...], "severity": "high|medium|low"}}
    ],
    "field_deep_dives": [
        {{"field": "...", "issue": "...", "pattern": "...", "suggestion": "..."}}
    ],
    "stratum_issues": [
        {{"stratum": "...", "issue": "...", "suggestion": "..."}}
    ],
    "recommendations": [
        {{"action": "...", "priority": 1, "expected_impact": "..."}}
    ]
}}"""

GATE_ANALYSIS_PROMPT = """You are an evaluation gate expert analyzing why a promotion gate passed or failed.

Gate Decision:
- Run ID: {run_id}
- Status: "{status}"
- Micro-F1: {micro_f1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (target: 0.85)
- Kappa: {kappa}
- Cases: {n}
- Blockers: {blockers}
- Warnings: {warnings}
- Stratum Blockers: {stratum_blockers}
- Root Cause (heuristic): {root_cause}
- Remediation (heuristic): {remediation}

Provide a JSON response with:
{{
    "summary": "2-3 sentence natural language explanation of gate decision",
    "blocking_factors": [
        {{"factor": "...", "evidence": "...", "quantitative_gap": "...", "fixable": true/false}}
    ],
    "promotion_path": [
        {{"step": "...", "estimated_f1_gain": 0.0, "effort": "low|medium|high", "prerequisites": [...]}}
    ],
    "risk_factors": [
        {{"risk": "...", "likelihood": "high|medium|low", "mitigation": "..."}}
    ],
    "confidence": 0.0-1.0
}}"""


class LLMAnalyzer:
    """LLM-backed semantic analysis for evaluation harness."""

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_dataset(self, analysis: DatasetAnalysis) -> Dict[str, Any]:
        """Run LLM semantic analysis on dataset analysis results."""
        prompt = DATASET_ANALYSIS_PROMPT.format(
            n_cases=analysis.n_cases,
            fields_seen=sorted(analysis.fields_seen),
            strata_distribution=analysis.strata_distribution,
            defect_rate_estimate=analysis.defect_rate_estimate,
            ocr_signal_distribution=analysis.ocr_signal_dist,
            duplicate_count=len(analysis.duplicate_pairs),
            top_difficult_fields=analysis.top_difficult_fields,
        )
        resp = self.client.complete(prompt, json_mode=True, temperature=self.temperature)
        return self._parse_json_response(resp.text)

    def analyze_errors(self, analysis: ErrorAnalysis) -> Dict[str, Any]:
        """Run LLM semantic analysis on error analysis results."""
        # Summarize error clusters by category
        cluster_summary = {}
        for ec in analysis.error_clusters[:20]:
            cat = ec.get("error_category", "unknown")
            if cat not in cluster_summary:
                cluster_summary[cat] = {"count": 0, "fields": set()}
            cluster_summary[cat]["count"] += 1
            if ec.get("field"):
                cluster_summary[cat]["fields"].add(ec["field"])

        error_clusters_str = "\n".join(
            f"  {cat}: {v['count']} cases, fields={sorted(v['fields'])}"
            for cat, v in sorted(cluster_summary.items(), key=lambda x: -x[1]["count"])
        )

        # Format stratum performance
        stratum_str = "\n".join(
            f"  {k}: tp={v['tp_pct']:.1%}, fp={v['fp_pct']:.1%}, fn={v['fn_pct']:.1%}"
            for k, v in analysis.stratum_perf.items()
        )

        prompt = ERROR_ANALYSIS_PROMPT.format(
            run_id=analysis.run_id,
            n_cases=analysis.n_cases,
            n_fields=analysis.n_fields,
            micro_f1=analysis.f1_point,
            ci_lower=analysis.f1_lower,
            ci_upper=analysis.f1_upper,
            total_tp=analysis.total_tp,
            total_fp=analysis.total_fp,
            total_fn=analysis.total_fn,
            kappa=analysis.kappa if analysis.kappa else "N/A",
            error_clusters=error_clusters_str,
            stratum_perf=stratum_str,
            field_f1="See error clusters above",
        )
        resp = self.client.complete(prompt, json_mode=True, temperature=self.temperature)
        return self._parse_json_response(resp.text)

    def analyze_gate(self, analysis: GateAnalysis) -> Dict[str, Any]:
        """Run LLM semantic analysis on gate decision."""
        prompt = GATE_ANALYSIS_PROMPT.format(
            run_id=analysis.run_id,
            status="PROMOTED" if analysis.passed else "BLOCKED",
            micro_f1=analysis.micro_f1,
            ci_lower=analysis.ci_lower,
            ci_upper=analysis.ci_upper,
            kappa=analysis.kappa if analysis.kappa else "N/A",
            n=analysis.n,
            blockers=analysis.blockers,
            warnings=analysis.warnings,
            stratum_blockers=analysis.stratum_blockers,
            root_cause=analysis.root_cause or "None identified",
            remediation=analysis.remediation,
        )
        resp = self.client.complete(prompt, json_mode=True, temperature=self.temperature)
        return self._parse_json_response(resp.text)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response with code fence tolerance."""
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else s
            if s.endswith("```"):
                s = s[: -len("```")]
            s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            start, end = s.find("{"), s.rfind("}")
            if start == -1 or end <= start:
                return {"error": "Failed to parse LLM response", "raw": text}
            return json.loads(s[start : end + 1])


def create_llm_analyzer(
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "qwen2.5-1.5b-instruct",
    api_key: str | None = None,
    **kwargs,
) -> LLMAnalyzer:
    """Create LLMAnalyzer with OpenAI-compatible client (llama-server by default)."""
    client = OpenAICompatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        **kwargs,
    )
    return LLMAnalyzer(client)


# --- Convenience: run full LLM-enhanced analysis pipeline ---

def run_llm_analysis_pipeline(
    config_path: str,
    cases_path: str | None = None,
    score_path: str | None = None,
    gate_path: str | None = None,
    run_dir: str | None = None,
    llm_base_url: str = "http://127.0.0.1:8080/v1",
    llm_model: str = "qwen2.5-1.5b-instruct",
) -> Dict[str, Any]:
    """
    Run complete LLM-enhanced analysis pipeline.
    
    Args:
        config_path: Path to eval_config.yaml
        cases_path: Path to cases.jsonl (optional, defaults to data/cord/cases.jsonl)
        score_path: Path to scores.jsonl (optional)
        gate_path: Path to gate.json (optional)
        run_dir: Run directory (if provided, auto-detects score/gate paths)
        llm_base_url: LLM endpoint (default: llama-server)
        llm_model: Model name
    
    Returns:
        Dict with heuristic + LLM analysis for each stage
    """
    results = {}
    llm = create_llm_analyzer(base_url=llm_base_url, model=llm_model)

    # Auto-detect from run_dir if provided
    if run_dir:
        rd = Path(run_dir)
        if score_path is None and (rd / "scores.jsonl").exists():
            score_path = str(rd / "scores.jsonl")
        if gate_path is None and (rd / "gate.json").exists():
            gate_path = str(rd / "gate.json")

    # Default cases path
    if cases_path is None:
        cases_path = "data/cord/cases.jsonl"

    # 1. Dataset Analysis
    if Path(cases_path).exists():
        dataset_analyzer = DatasetAnalyzer(cases_path)
        dataset_analysis = dataset_analyzer.analyze()
        results["dataset"] = {
            "heuristic": {
                "summary": dataset_analyzer.summary(),
                "data": {
                    "n_cases": dataset_analysis.n_cases,
                    "fields_seen": sorted(dataset_analysis.fields_seen),
                    "strata_distribution": dataset_analysis.strata_distribution,
                    "defect_rate_estimate": dataset_analysis.defect_rate_estimate,
                    "duplicate_pairs": dataset_analysis.duplicate_pairs,
                    "ocr_signal_distribution": dataset_analysis.ocr_signal_dist,
                },
            },
            "llm": llm.analyze_dataset(dataset_analysis),
        }

    # 2. Error Analysis
    if score_path and Path(score_path).exists():
        error_analyzer = ErrorAnalyzer(score_path, config_path, "auto")
        error_analysis = error_analyzer.analyze()
        results["error"] = {
            "heuristic": {
                "summary": error_analyzer.summary(),
                "data": {
                    "n_cases": error_analysis.n_cases,
                    "n_fields": error_analysis.n_fields,
                    "total_tp": error_analysis.total_tp,
                    "total_fp": error_analysis.total_fp,
                    "total_fn": error_analysis.total_fn,
                    "micro_f1": error_analysis.f1_point,
                    "ci_lower": error_analysis.f1_lower,
                    "ci_upper": error_analysis.f1_upper,
                    "kappa": error_analysis.kappa,
                    "defect_rate": error_analysis.defect_rate,
                    "defect_n": error_analysis.defect_n,
                    "error_clusters": error_analysis.error_clusters,
                    "stratum_perf": error_analysis.stratum_perf,
                    "recommendations": error_analysis.recommendations,
                },
            },
            "llm": llm.analyze_errors(error_analysis),
        }

    # 3. Gate Analysis
    if gate_path and Path(gate_path).exists():
        gate_analyzer = GateAnalyzer(gate_path, config_path, score_path)
        gate_analysis = gate_analyzer.analyze()
        results["gate"] = {
            "heuristic": {
                "summary": gate_analyzer.summary(),
                "data": {
                    "run_id": gate_analysis.run_id,
                    "passed": gate_analysis.passed,
                    "micro_f1": gate_analysis.micro_f1,
                    "ci_lower": gate_analysis.ci_lower,
                    "ci_upper": gate_analysis.ci_upper,
                    "kappa": gate_analysis.kappa,
                    "n": gate_analysis.n,
                    "blockers": gate_analysis.blockers,
                    "stratum_blockers": gate_analysis.stratum_blockers,
                    "root_cause": gate_analysis.root_cause,
                    "remediation": gate_analysis.remediation,
                },
            },
            "llm": llm.analyze_gate(gate_analysis),
        }

    return results


# --- CLI ---
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM-Enhanced Analysis Pipeline")
    p.add_argument("--config", required=True, help="Path to eval_config.yaml")
    p.add_argument("--cases", default=None, help="Path to cases.jsonl")
    p.add_argument("--score", default=None, help="Path to scores.jsonl")
    p.add_argument("--gate", default=None, help="Path to gate.json")
    p.add_argument("--run-dir", default=None, help="Run directory (auto-detects score/gate)")
    p.add_argument("--llm-url", default="http://127.0.0.1:8080/v1", help="LLM base URL")
    p.add_argument("--llm-model", default="qwen2.5-1.5b-instruct", help="LLM model")
    args = p.parse_args()

    results = run_llm_analysis_pipeline(
        config_path=args.config,
        cases_path=args.cases,
        score_path=args.score,
        gate_path=args.gate,
        run_dir=args.run_dir,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
    )

    print(json.dumps(results, indent=2))