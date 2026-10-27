"""
Infinity Agent Runtime Integration for slm_eval

This module shows how to wrap the slm_eval pipeline as Infinity RAP tools
and an orchestrating agent. This is a DESIGN DOCUMENT - the actual
implementation would be in Rust.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Infinity Daemon                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent: EvalOrchestrator                 │   │
│  │  Tools: WebSearch, Inference, MinHash, Review,      │   │
│  │         Scoring, Gate, SyntheticGen                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│         RAP/HTTP callbacks (async, hibernating)             │
│                          ▼                                   │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐          │
│  │WebSearch│ │Inference │ │MinHash │ │ Review   │ ...       │
│  │ Server  │ │ Server   │ │ Server │ │ Server   │          │
│  └─────────┘ └──────────┘ └────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────┘

Key Benefits for Low Memory:
- Each tool call = agent yields (hibernates), ~bytes memory
- 100s of concurrent eval agents on laptop
- Tools run on separate processes/servers (can be on different machines)
- Scale-to-zero when idle
"""

# ============================================================
# RAP Tool Definitions (what the agent sees)
# ============================================================

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for real-world merchant/item data for synthetic generation",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
            "provider": {"type": "string", "enum": ["crw", "tavily", "searxng"], "default": "crw"}
        },
        "required": ["query"]
    }
}

INFERENCE_TOOL = {
    "name": "run_inference",
    "description": "Run model inference on a prompt (llama-server / OpenAI-compatible)",
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string", "default": "qwen2.5-1.5b"},
            "json_mode": {"type": "boolean", "default": False},
            "temperature": {"type": "number", "default": 0.0},
            "max_tokens": {"type": "integer", "default": 2048}
        },
        "required": ["prompt"]
    }
}

MINHASH_DEDUP_TOOL = {
    "name": "minhash_dedup",
    "description": "Deduplicate cases against holdout using MinHash",
    "inputSchema": {
        "type": "object",
        "properties": {
            "cases": {"type": "array", "items": {"type": "object"}},
            "holdout_dir": {"type": "string"},
            "threshold": {"type": "number", "default": 0.6}
        },
        "required": ["cases", "holdout_dir"]
    }
}

REVIEW_QUEUE_TOOL = {
    "name": "human_review",
    "description": "Submit cases to human review queue, wait for resolution",
    "inputSchema": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "cases": {"type": "array", "items": {"type": "object"}},
            "reason": {"type": "string", "enum": ["synthetic_injection", "judge_disagreement", "low_confidence"]}
        },
        "required": ["run_id", "cases", "reason"]
    }
}

SCORING_TOOL = {
    "name": "score_cases",
    "description": "Score predictions against gold using element-level micro-F1 + bootstrap CI",
    "inputSchema": {
        "type": "object",
        "properties": {
            "predictions_file": {"type": "string"},
            "gold_file": {"type": "string"},
            "strata_config": {"type": "object"},
            "bootstrap_resamples": {"type": "integer", "default": 1000}
        },
        "required": ["predictions_file", "gold_file"]
    }
}

GATE_TOOL = {
    "name": "evaluate_gate",
    "description": "Evaluate promotion gate using CI lower bounds",
    "inputSchema": {
        "type": "object",
        "properties": {
            "scores_file": {"type": "string"},
            "calibration_file": {"type": "string"},
            "thresholds": {"type": "object"},
            "pending_reviews": {"type": "integer", "default": 0}
        },
        "required": ["scores_file", "calibration_file"]
    }
}

SYNTHETIC_GEN_TOOL = {
    "name": "generate_synthetic",
    "description": "Generate synthetic cases using web search + templates",
    "inputSchema": {
        "type": "object",
        "properties": {
            "n_propose": {"type": "integer", "default": 20},
            "merchants": {"type": "array", "items": {"type": "object"}},
            "minhash_threshold": {"type": "number", "default": 0.6},
            "use_web": {"type": "boolean", "default": True}
        },
        "required": ["n_propose"]
    }
}

# ============================================================
# Agent Prompt (system prompt for the orchestrator)
# ============================================================

ORCHESTRATOR_PROMPT = """
You are the EvalOrchestrator agent for the SLM Evaluation Harness.

Your job: Run the complete evaluation pipeline for fine-tuning SLMs on receipt/invoice extraction.

PIPELINE STAGES:
1. SPLIT - Stratified split dataset into train/holdout, seal holdout
2. INFERENCE - Run model on holdout cases, produce predictions
3. SCORING - Score predictions (element-level micro-F1, bootstrap CI)
4. JUDGE - Calibrate judge (Cohen's κ), judge fuzzy fields
5. GATE - CI-lower-bound decision (promote/blocked/error)
6. SYNTHETIC (if blocked) - Generate synthetic cases, human review, inject
7. REPEAT - Loop until N consecutive passes

TOOLS AVAILABLE:
- web_search: Find real merchant/item data
- run_inference: Call model on cases
- minhash_dedup: Deduplicate against holdout
- human_review: Submit to review queue, wait for human
- score_cases: Compute micro-F1 + bootstrap CI
- evaluate_gate: Promotion decision
- generate_synthetic: Create synthetic cases

RULES:
- ALWAYS use minhash_dedup before injecting synthetic cases
- ALWAYS wait for human_review to resolve before continuing
- Gate decision uses CI LOWER BOUND, never point estimate
- κ ≥ 0.85 required for subjective fields
- Exit codes: 0=promote, 1=blocked, 2=error

OUTPUT: Final gate.json with decision, CI, strata breakdown
"""

# ============================================================
# Python Prototype: How the agent logic would work
# (This runs synchronously; Infinity makes it async/hibernating)
# ============================================================

import json
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Awaitable
from pathlib import Path


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    call_id: str
    result: dict[str, Any] | None = None
    error: str | None = None


class MockInfinityRuntime:
    """
    Prototype showing how the agent would work.
    In real Infinity, this is handled by the runtime with hibernation.
    """
    
    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.history: list = []
    
    def register_tool(self, name: str, func: Callable[[dict], Awaitable[dict]]):
        self.tools[name] = func
    
    async def call_tool(self, name: str, args: dict, call_id: str) -> ToolResult:
        """Simulate RAP tool call with callback."""
        if name not in self.tools:
            return ToolResult(call_id=call_id, error=f"Unknown tool: {name}")
        
        try:
            # In real Infinity: agent YIELDS here, hibernates
            # Tool server processes, calls callback URL
            # Agent WAKES, continues
            result = await self.tools[name](args)
            return ToolResult(call_id=call_id, result=result)
        except Exception as e:
            return ToolResult(call_id=call_id, error=str(e))
    
    async def run_orchestrator(self, config: dict) -> dict:
        """
        Main orchestration logic - this is what the Infinity agent would execute.
        """
        run_id = config.get("run_id", "eval-1")
        dataset_path = config["dataset"]
        model_path = config["model"]
        holdout_dir = f"runs/{run_id}/holdout"
        
        print(f"🚀 Starting eval run: {run_id}")
        
        # Stage 1: Split (assumed done, holdout exists)
        print("📂 Stage 1: Holdout ready")
        
        # Stage 2: Inference
        print("🔮 Stage 2: Running inference...")
        inference_result = await self.call_tool("run_inference", {
            "prompt": f"Process holdout at {holdout_dir}",
            "model": model_path,
            "json_mode": True
        }, "call-1")
        
        if inference_result.error:
            return {"error": inference_result.error, "exit_code": 2}
        
        predictions_file = inference_result.result.get("predictions_file")
        
        # Stage 3: Scoring
        print("📊 Stage 3: Scoring...")
        scoring_result = await self.call_tool("score_cases", {
            "predictions_file": predictions_file,
            "gold_file": f"{holdout_dir}/cases.jsonl",
            "bootstrap_resamples": 1000
        }, "call-2")
        
        if scoring_result.error:
            return {"error": scoring_result.error, "exit_code": 2}
        
        scores_file = scoring_result.result.get("scores_file")
        
        # Stage 4: Judge calibration (simplified)
        print("⚖️ Stage 4: Judge calibration...")
        # ... calibration logic ...
        
        # Stage 5: Gate
        print("🚪 Stage 5: Gate evaluation...")
        gate_result = await self.call_tool("evaluate_gate", {
            "scores_file": scores_file,
            "calibration_file": "calibration/judge.jsonl",
            "pending_reviews": 0
        }, "call-3")
        
        if gate_result.error:
            return {"error": gate_result.error, "exit_code": 2}
        
        decision = gate_result.result.get("decision", {})
        promote = decision.get("promote", False)
        
        if promote:
            print("✅ PROMOTED!")
            return {"decision": "promote", "exit_code": 0}
        
        # Stage 6: Synthetic loop (if blocked)
        print("🔄 Stage 6: Synthetic generation...")
        
        # Get web search data
        web_result = await self.call_tool("web_search", {
            "query": "popular retail stores receipt items 2024",
            "max_results": 10
        }, "call-4")
        
        merchants = web_result.result.get("merchants", [])
        
        # Generate synthetic
        synth_result = await self.call_tool("generate_synthetic", {
            "n_propose": 20,
            "merchants": merchants,
            "minhash_threshold": 0.6,
            "use_web": True
        }, "call-5")
        
        # Dedup
        dedup_result = await self.call_tool("minhash_dedup", {
            "cases": synth_result.result.get("cases", []),
            "holdout_dir": holdout_dir,
            "threshold": 0.6
        }, "call-6")
        
        # Human review
        review_result = await self.call_tool("human_review", {
            "run_id": run_id,
            "cases": dedup_result.result.get("novel_cases", []),
            "reason": "synthetic_injection"
        }, "call-7")
        
        # Loop back to inference with new cases
        # ... (would repeat from Stage 2)
        
        return {"decision": "blocked", "exit_code": 1, "reason": "synthetic loop needed"}


# ============================================================
# Tool Server Implementations (run as separate processes)
# ============================================================

# These would be RAP tool servers in Rust (using rap-test-servers pattern)
# Each runs on its own port, exposes /.well-known/rap-toolset and /invoke

TOOL_SERVER_SPECS = """
# WebSearch Server (port 8001)
# Wraps: crw CLI, Tavily API, or SearXNG
# Tool: web_search(query, max_results, provider)

# Inference Server (port 8002) 
# Wraps: llama-server / OpenAI-compatible endpoint
# Tool: run_inference(prompt, model, json_mode, temperature)

# MinHash Server (port 8003)
# Wraps: slm_eval.hashing.minhash_signature/similarity
# Tool: minhash_dedup(cases, holdout_dir, threshold)

# Review Server (port 8004)
# Wraps: slm_eval.store review_task tables
# Tool: human_review(run_id, cases, reason)

# Scoring Server (port 8005)
# Wraps: slm_eval.stats.micro_f1, bootstrap_ci, cohens_kappa
# Tool: score_cases(predictions_file, gold_file, strata_config)

# Gate Server (port 8006)
# Wraps: slm_eval.gate.decide
# Tool: evaluate_gate(scores_file, calibration_file, pending_reviews)

# SyntheticGen Server (port 8007)
# Wraps: slm_eval.synthetic.generate_synthetic_cases
# Tool: generate_synthetic(n_propose, merchants, minhash_threshold)
"""


# ============================================================
# Deployment Config (for Infinity daemon)
# ============================================================

DAEMON_CONFIG = {
    "conversation_store": "sqlite",
    "state_store": "sqlite",
    "model_provider": {
        "type": "openai_compat",
        "base_url": "http://localhost:8080/v1",  # llama-server
        "model": "qwen2.5-1.5b-instruct"
    },
    "rap_toolsets": [
        {"type": "toolset_server", "server_url": "http://localhost:8001"},  # WebSearch
        {"type": "toolset_server", "server_url": "http://localhost:8002"},  # Inference
        {"type": "toolset_server", "server_url": "http://localhost:8003"},  # MinHash
        {"type": "toolset_server", "server_url": "http://localhost:8004"},  # Review
        {"type": "toolset_server", "server_url": "http://localhost:8005"},  # Scoring
        {"type": "toolset_server", "server_url": "http://localhost:8006"},  # Gate
        {"type": "toolset_server", "server_url": "http://localhost:8007"},  # SyntheticGen
    ],
    "builtin_tools": True,  # spawn_thread, sleep, etc.
}

# ============================================================
# Running Instructions
# ============================================================

RUN_INSTRUCTIONS = """
# 1. Build tool servers (Rust)
cd infinity-integration/tool-servers
cargo build --release --workspace

# 2. Start tool servers (each on own port)
./web-search-server --port 8001 &
./inference-server --port 8002 --model-endpoint http://localhost:8080/v1 &
./minhash-server --port 8003 &
./review-server --port 8004 --db slm_eval.db &
./scoring-server --port 8005 &
./gate-server --port 8006 &
./synthetic-server --port 8007 &

# 3. Write RAP config for session
mkdir -p runs/eval-1/.infinity
cat > runs/eval-1/.infinity/rap.json <<EOF
{"tool_sets":[{"type":"toolset_server","server_url":"http://127.0.0.1:8001"},...]}
EOF

# 4. Start Infinity daemon
infinity-daemon --config daemon.toml

# 5. Launch agent session
infinity thread launch --config daemon.toml --prompt "$ORCHESTRATOR_PROMPT" \\
  --input '{"run_id":"eval-1","dataset":"data/cord","model":"qwen2.5-1.5b"}'

# 6. Agent runs, yields on each tool call, hibernates (bytes memory)
#    Tool servers do work, callback wakes agent
#    100s of agents can run concurrently on laptop
"""


if __name__ == "__main__":
    print("Infinity Integration Design for slm_eval")
    print("=" * 50)
    print("\nTool Definitions:")
    for name, tool in [
        ("web_search", WEB_SEARCH_TOOL),
        ("run_inference", INFERENCE_TOOL),
        ("minhash_dedup", MINHASH_DEDUP_TOOL),
        ("human_review", REVIEW_QUEUE_TOOL),
        ("score_cases", SCORING_TOOL),
        ("evaluate_gate", GATE_TOOL),
        ("generate_synthetic", SYNTHETIC_GEN_TOOL),
    ]:
        print(f"  - {name}: {tool['description']}")
    
    print("\n" + TOOL_SERVER_SPECS)
    print("\nDaemon Config:")
    print(json.dumps(DAEMON_CONFIG, indent=2))
    print("\nRun Instructions:")
    print(RUN_INSTRUCTIONS)