"""AI Data Science Team integration for SLM evaluation harness.

Provides post-evaluation analysis, visualization, and integration with
the Rust-based evaluation pipeline (slm-eval-infinity).

Usage:
    python -m slm_eval.ai_data_integration gate cord-baseline
    python -m slm_eval.ai_data_integration dataset data/cord/cases.jsonl
    python -m slm_eval.ai_data_integration error cord-baseline scores.jsonl
    python -m slm_eval.ai_data_integration ai data/cord/cases.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Gate analysis – consume the Rust gate.json output
# ---------------------------------------------------------------------------

def analyze_gate(run_dir: Path, config_path: Path) -> Dict[str, Any]:
    """Load a gate.json file and return a structured analysis.

    This function is the entry point for the AI Data Science Team integration.
    It wraps the Rust gate.json output and provides:
    - Overall promotion decision (promote / block / blocked)
    - Micro-F1, CI bounds, kappa
    - Blockers and warnings
    - Stratum-level breakdown
    - Root cause analysis
    """
    from slm_eval.analysis.gate import GateAnalysis, load_gate_for_analyzer

    # Load gate analysis (reuses existing GateAnalyzer logic)
    analyzer = load_gate_for_analyzer(run_dir, config_path)
    analysis = analyzer.analyze()

    return {
        "run_id": analysis.run_id,
        "status": "promote" if analysis.passed else "blocked",
        "promote": analysis.passed,
        "micro_f1": analysis.micro_f1,
        "ci_lower": analysis.ci_lower,
        "ci_upper": analysis.ci_upper,
        "kappa": analysis.kappa,
        "n": analysis.n,
        "root_cause": analysis.root_cause,
        "blockers": analysis.blockers,
        "warnings": analysis.warnings,
        "stratum_blockers": analysis.stratum_blockers,
        "remediation": analysis.remediation,
    }


# ---------------------------------------------------------------------------
# Dataset analysis – quality, defect rate, OCR signal
# ---------------------------------------------------------------------------

def analyze_dataset(cases_path: Path, minhash_threshold: float = 0.6) -> Dict[str, Any]:
    """Analyze the dataset used for evaluation.

    Returns a structured analysis including:
    - Total cases, fields, strata distribution
    - Defect rate (unextractable cases)
    - OCR signal distribution
    - Top difficult fields
    - Duplicate pairs
    """
    from slm_eval.analysis.dataset import DatasetAnalyzer

    analyzer = DatasetAnalyzer(cases_path, minhash_threshold=minhash_threshold)
    analysis = analyzer.analyze()

    return {
        "n_cases": analysis.n_cases,
        "n_fields": analysis.n_fields,
        "fields_seen": sorted(analysis.fields_seen),
        "strata_distribution": analysis.strata_distribution,
        "defect_rate_estimate": analysis.defect_rate_estimate,
        "top_difficult_fields": analysis.top_difficult_fields,
        "duplicate_pairs": analysis.duplicate_pairs,
        "ocr_signal_distribution": analysis.ocr_signal_dist,
    }


# ---------------------------------------------------------------------------
# Error analysis – scoring results, error clusters, stratum performance
# ---------------------------------------------------------------------------

def analyze_errors(run_id: str, score_path: Path, config_path: Path) -> Dict[str, Any]:
    """Analyze scoring results after the evaluation pipeline.

    Returns:
    - Micro-F1, CI bounds, kappa
    - Total TP/FP/FN counts
    - Error clusters (by error category)
    - Stratum performance (TP/FP/FN per stratum)
    - Recommendations for improvement
    """
    from slm_eval.analysis.error import ErrorAnalyzer

    analyzer = ErrorAnalyzer(str(score_path), str(config_path), str(run_id))
    analysis = analyzer.analyze()

    return {
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
    }


# ---------------------------------------------------------------------------
# AI Data Science Team agents – post-evaluation analysis
# ---------------------------------------------------------------------------

def ai_data_integration_agent(
    run_dir: Path,
    config_path: Path,
    score_path: Optional[Path] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper that combines AI Data Science Team agents with the existing
    analysis pipeline. This function can be called by the main pipeline
    to add semantic analysis on top of the heuristic results.

    It provides:
    - Dataset quality summary (defect rate, OCR signal, difficult fields)
    - Micro-F1 trend (over multiple runs)
    - Stratified breakdown (per quality/item_count strata)
    - LLM-powered error analysis and root-cause identification
    - Visualization-ready data structures
    """
    # 1. Gate analysis
    gate_result = analyze_gate(run_dir, config_path)

    # 2. Dataset analysis (if cases.jsonl is available)
    dataset_result = {}
    cases_path = config_path.parent / "data" / "cases.jsonl"
    if cases_path.exists():
        dataset_result = analyze_dataset(cases_path)

    # 3. Error analysis (if scores.jsonl is available)
    error_result = {}
    if score_path and score_path.exists():
        error_result = analyze_errors(str(run_dir), score_path, config_path)

    # 4. LLM-powered semantic analysis (optional)
    llm_result = {}
    if llm_base_url:
        try:
            from slm_eval.analysis.llm_analyzer import LLMAnalyzer, OpenAICompatClient
            from slm_eval.analysis.dataset import DatasetAnalyzer
            from slm_eval.analysis.error import ErrorAnalyzer

            client = OpenAICompatClient(
                base_url=llm_base_url,
                model=llm_model or "qwen2.5-1.5b-instruct",
            )
            analyzer = LLMAnalyzer(client)

            # LLM dataset analysis
            if dataset_result:
                da = DatasetAnalyzer(cases_path)
                dataset_analysis = da.analyze()
                llm_result["dataset"] = analyzer.analyze_dataset(dataset_analysis)

            # LLM error analysis
            if error_result:
                ea = ErrorAnalyzer(str(score_path), str(config_path), str(run_dir))
                error_analysis = ea.analyze()
                llm_result["errors"] = analyzer.analyze_errors(error_analysis)
        except Exception as e:
            llm_result["error"] = str(e)

    # 5. Build a comprehensive report
    report = {
        "gate": gate_result,
        "dataset": dataset_result,
        "errors": error_result,
        "llm_analysis": llm_result,
        "summary": {
            "overall_status": gate_result["status"],
            "promote": gate_result["promote"],
            "micro_f1": gate_result["micro_f1"],
            "ci_lower": gate_result["ci_lower"],
            "ci_upper": gate_result["ci_upper"],
            "kappa": gate_result["kappa"],
            "defect_rate": gate_result.get("dataset_defect", {}).get("rate", 0.0),
            "defect_n": gate_result.get("dataset_defect", {}).get("defect_n", 0),
            "stratum_blockers": gate_result["stratum_blockers"],
        },
    }

    return report


# ---------------------------------------------------------------------------
# Visualization helpers – matplotlib / seaborn
# ---------------------------------------------------------------------------

def plot_micro_f1_trend(gate_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Create a line plot of micro-F1 over runs.

    Args:
        gate_results: List of gate analysis dicts (one per run).
        output_path: Path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping visualization.")
        return

    # Extract micro-F1 values
    micro_f1s = [r["micro_f1"] for r in gate_results if r.get("micro_f1") is not None]
    n = len(micro_f1s)

    if n < 2:
        print("Not enough runs to plot micro-F1 trend.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(n), micro_f1s, marker="o", linewidth=2, color="steelblue")
    plt.title("Micro-F1 Trend Over Runs")
    plt.xlabel("Run index")
    plt.ylabel("Micro-F1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Micro-F1 trend plot saved to {output_path}")


def plot_stratum_performance(gate_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Create a bar chart of micro-F1 per stratum.

    Args:
        gate_results: List of gate analysis dicts (one per run).
        output_path: Path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping visualization.")
        return

    # Aggregate micro-F1 per stratum across runs
    stratum_f1 = {}
    for r in gate_results:
        for stratum in r.get("stratum_blockers", []):
            stratum_name = stratum.get("stratum", "unknown")
            if stratum_name not in stratum_f1:
                stratum_f1[stratum_name] = []
            # Use the micro-F1 from the gate result (same for all strata in a run)
            stratum_f1[stratum_name].append(r["micro_f1"])

    if not stratum_f1:
        print("No stratum performance data to plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(stratum_f1.keys()),
        y=list(stratum_f1.values()),
        palette="viridis",
    )
    plt.title("Micro-F1 by Stratum")
    plt.xlabel("Stratum")
    plt.ylabel("Micro-F1")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Stratum performance plot saved to {output_path}")


def plot_defect_rate_over_time(gate_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot defect rate over runs.

    Args:
        gate_results: List of gate analysis dicts (one per run).
        output_path: Path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping visualization.")
        return

    defect_rates = [r.get("defect_rate", 0.0) for r in gate_results]
    n = len(defect_rates)

    if n < 2:
        print("Not enough runs to plot defect rate.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(n), defect_rates, marker="o", linewidth=2, color="coral")
    plt.title("Defect Rate Over Runs")
    plt.xlabel("Run index")
    plt.ylabel("Defect Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Defect rate plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main entry point for CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the AI Data Science Team integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Data Science Team integration for SLM evaluation harness"
    )
    sub = parser.add_subparsers(dest="command")

    # gate
    gate_p = sub.add_parser("gate", help="Analyze gate.json decision")
    gate_p.add_argument("run_id", help="Run ID from gate.json")
    gate_p.add_argument("--config", required=True, help="Path to eval_config.yaml")

    # dataset
    dataset_p = sub.add_parser("dataset", help="Analyze dataset quality")
    dataset_p.add_argument("cases_path", help="Path to cases.jsonl")
    dataset_p.add_argument("--minhash_threshold", type=float, default=0.6, help="MinHash threshold for duplicate detection")

    # error
    error_p = sub.add_parser("error", help="Analyze scoring results")
    error_p.add_argument("run_id", help="Run ID from scores.jsonl")
    error_p.add_argument("--score", required=True, help="Path to scores.jsonl")
    error_p.add_argument("--config", required=True, help="Path to eval_config.yaml")

    # ai
    ai_p = sub.add_parser("ai", help="Run full AI Data Science Team integration")
    ai_p.add_argument("run_id", help="Run ID from gate.json")
    ai_p.add_argument("--config", required=True, help="Path to eval_config.yaml")
    ai_p.add_argument("--score", default=None, help="Optional scores.jsonl for error analysis")
    ai_p.add_argument("--llm_url", default=None, help="LLM endpoint for semantic analysis (optional)")
    ai_p.add_argument("--llm_model", default=None, help="LLM model name (optional)")
    ai_p.add_argument("--viz_dir", default=None, help="Directory to save visualization plots (optional)")

    args = parser.parse_args()

    if args.command == "gate":
        # run_dir = runs/run_id (gate.json is inside)
        result = analyze_gate(Path("runs") / args.run_id, Path(args.config))
        print(json.dumps(result, indent=2))
    elif args.command == "dataset":
        result = analyze_dataset(Path(args.cases_path), args.minhash_threshold)
        print(json.dumps(result, indent=2))
    elif args.command == "error":
        result = analyze_errors(args.run_id, Path(args.score), Path(args.config))
        print(json.dumps(result, indent=2))
    elif args.command == "ai":
        # run_dir = runs/run_id (gate.json is inside)
        result = ai_data_integration_agent(
            Path("runs") / args.run_id, Path(args.config),
            Path(args.score) if args.score else None,
            llm_base_url=args.llm_url,
            llm_model=args.llm_model,
        )
        # Optionally generate visualizations
        if args.viz_dir:
            viz_dir = Path(args.viz_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)
            gate_results = [result["gate"]]  # single run for now
            plot_micro_f1_trend(gate_results, viz_dir / "micro_f1_trend.png")
            report_path = viz_dir / "ai_analysis_report.json"
            report_path.write_text(json.dumps(result, indent=2))
            print(f"\nVisualizations saved to {viz_dir}")
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
