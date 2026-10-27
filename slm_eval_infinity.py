#!/usr/bin/env python3
"""
slm-eval-infinity wrapper - delegates to the Infinity Agent Runtime binary.

This provides the same CLI interface as the Python slm-eval but runs
the evaluation pipeline using Infinity's RAP tool servers and orchestrator agent.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "endpoint": "http://localhost:8080/v1",
        "model": "qwen2.5-1.5b-instruct",
        "api_key_env": None
    },
    "gate": {
        "micro_f1_target": 0.85,
        "kappa_target": 0.85,
        "min_strata_n": 5
    },
    "synthetic": {
        "enabled": True,
        "max_iterations": 5,
        "minhash_threshold": 0.6,
        "web_provider": "crw"
    },
    "calibration": {
        "judge_model": "nemotron-3.5-bolt-e4b",
        "judge_endpoint": "https://integrate.api.nvidia.com/v1",
        "bootstrap_resamples": 1000,
        "ci_level": 0.95
    },
    "holdout": {
        "frac": 0.2,
        "seed": 42,
        "strata": ["category", "group"]
    }
}

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = json.load(f)
        # Deep merge
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    return config

def run_infinity_daemon(daemon_config: str = "infinity-daemon.toml") -> subprocess.Popen:
    """Start the Infinity daemon with tool servers."""
    # Check if daemon is already running
    try:
        import requests
        resp = requests.get("http://localhost:9999/health", timeout=2)
        if resp.status_code == 200:
            print("Infinity daemon already running")
            return None
    except:
        pass
    
    # Start the daemon
    print(f"Starting Infinity daemon with config: {daemon_config}")
    proc = subprocess.Popen(
        ["infinity", "daemon", "start", "--config", daemon_config],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for daemon to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            import requests
            resp = requests.get("http://localhost:9999/health", timeout=2)
            if resp.status_code == 200:
                print("Infinity daemon started")
                return proc
        except:
            continue
    
    # If we get here, daemon failed to start
    stdout, stderr = proc.communicate(timeout=5)
    print(f"Daemon failed to start:")
    print(f"stdout: {stdout.decode()}")
    print(f"stderr: {stderr.decode()}")
    return None

def stop_infinity_daemon(proc: Optional[subprocess.Popen]):
    """Stop the Infinity daemon."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

async def run_evaluation(run_id: str, dataset_path: str, config: Dict[str, Any], database: str) -> int:
    """Run the evaluation pipeline using Infinity agent."""
    # This would use the Infinity agent to run the full pipeline
    # For now, we'll delegate to the Rust binary if available
    
    # Check if slm-eval-infinity binary exists
    binary_path = Path("slm-eval-infinity/target/release/slm-eval-infinity")
    if not binary_path.exists():
        binary_path = Path("slm-eval-infinity/target/debug/slm-eval-infinity")
    
    if binary_path.exists():
        # Run the Rust binary
        cmd = [
            str(binary_path),
            "--database", database,
            "run",
            "--run-id", run_id,
            "--dataset", dataset_path,
        ]
        if config.get("model", {}).get("model"):
            cmd.extend(["--model", config["model"]["model"]])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    else:
        print("slm-eval-infinity binary not found. Please build it first:")
        print("  cd slm-eval-infinity && cargo build --release")
        return 2

def cmd_run(args: argparse.Namespace) -> int:
    """Run the evaluation pipeline."""
    config = load_config(args.config)
    
    # Override with CLI args
    if args.model:
        config["model"]["model"] = args.model
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = "data/cord/cases.jsonl"
    
    run_id = args.run_id or f"run-{int(time.time())}"
    database = args.database or "sqlite:runs/slm_eval.db"
    
    # Run evaluation
    return asyncio.run(run_evaluation(run_id, dataset_path, config, database))

def cmd_review(args: argparse.Namespace) -> int:
    """Manage the review queue."""
    binary_path = Path("slm-eval-infinity/target/release/slm-eval-infinity")
    if not binary_path.exists():
        binary_path = Path("slm-eval-infinity/target/debug/slm-eval-infinity")
    
    if not binary_path.exists():
        print("slm-eval-infinity binary not found. Please build it first.")
        return 2
    
    cmd = [str(binary_path), "--database", args.database, "review"]
    if args.action == "list":
        cmd.extend(["list", "--run-id", args.run_id])
    elif args.action == "approve":
        cmd.extend(["approve", "--task-id", args.task_id])
    elif args.action == "reject":
        cmd.extend(["reject", "--task-id", args.task_id])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

def cmd_adapt_cord(args: argparse.Namespace) -> int:
    """Convert CORD dataset to eval format."""
    binary_path = Path("slm-eval-infinity/target/release/slm-eval-infinity")
    if not binary_path.exists():
        binary_path = Path("slm-eval-infinity/target/debug/slm-eval-infinity")
    
    if not binary_path.exists():
        print("slm-eval-infinity binary not found. Please build it first.")
        return 2
    
    cmd = [
        str(binary_path),
        "adapt-cord",
        "--input", args.input,
        "--output", args.output,
        "--holdout-dir", args.holdout_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

def cmd_daemon(args: argparse.Namespace) -> int:
    """Manage the Infinity daemon."""
    if args.action == "start":
        proc = run_infinity_daemon(args.config)
        if proc:
            print("Daemon started in background. Use 'slm-eval-infinity daemon stop' to stop.")
            return 0
        return 1
    elif args.action == "stop":
        subprocess.run(["infinity", "daemon", "stop"])
        return 0
    elif args.action == "status":
        try:
            import requests
            resp = requests.get("http://localhost:9999/health", timeout=2)
            if resp.status_code == 200:
                print("Daemon is running")
                return 0
            else:
                print("Daemon not responding")
                return 1
        except:
            print("Daemon not running")
            return 1

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SLM Evaluation Harness on Infinity Agent Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database", default="sqlite:runs/slm_eval.db", help="Database URL")
    parser.add_argument("--config", help="Path to config JSON file")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run full evaluation pipeline")
    run_parser.add_argument("--run-id", help="Unique run identifier")
    run_parser.add_argument("--dataset", help="Path to dataset JSONL")
    run_parser.add_argument("--model", help="Model name to evaluate")
    
    # Review command
    review_parser = subparsers.add_parser("review", help="Manage review queue")
    review_sub = review_parser.add_subparsers(dest="action", required=True)
    
    list_parser = review_sub.add_parser("list", help="List review tasks")
    list_parser.add_argument("--run-id", required=True, help="Run ID to list tasks for")
    
    approve_parser = review_sub.add_parser("approve", help="Approve a review task")
    approve_parser.add_argument("--task-id", required=True, help="Task ID to approve")
    
    reject_parser = review_sub.add_parser("reject", help="Reject a review task")
    reject_parser.add_argument("--task-id", required=True, help="Task ID to reject")
    
    # Adapt CORD command
    adapt_parser = subparsers.add_parser("adapt-cord", help="Convert CORD dataset")
    adapt_parser.add_argument("--input", required=True, help="Input CORD JSONL")
    adapt_parser.add_argument("--output", required=True, help="Output eval JSONL")
    adapt_parser.add_argument("--holdout-dir", required=True, help="Holdout directory")
    
    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Manage Infinity daemon")
    daemon_sub = daemon_parser.add_subparsers(dest="action", required=True)
    daemon_sub.add_parser("start", help="Start daemon").add_argument("--config", default="infinity-daemon.toml")
    daemon_sub.add_parser("stop", help="Stop daemon")
    daemon_sub.add_parser("status", help="Check daemon status")
    
    args = parser.parse_args()
    
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "review":
        return cmd_review(args)
    elif args.command == "adapt-cord":
        return cmd_adapt_cord(args)
    elif args.command == "daemon":
        return cmd_daemon(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())