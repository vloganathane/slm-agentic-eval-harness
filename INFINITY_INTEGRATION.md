# Quick Start Guide for Infinity + slm_eval Integration

## Overview
This integrates the slm_eval pipeline with the Infinity agent runtime framework, enabling:
- **Concurrent evaluation agents** (100s on laptop) via hibernation/yield
- **Distributed tool execution** (web search, inference, scoring on separate servers)
- **Long-running synthetic loops** (agents hibernate while waiting for human review)
- **Scale-to-zero** when idle

## Prerequisites

### 1. Install Infinity
```bash
# From the infinity repo
cd /tmp/infinity
cargo install --path crates/infinity-agent-cli --path crates/infinity-daemon
```

### 2. Start llama-server (for inference)
```bash
# In a separate terminal
llama-server -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 --port 8080 -ngl 99 -t 8
```

### 3. (Optional) Install web search providers
```bash
# crw (Rust CLI)
cargo install --git https://github.com/fastcrw/crw crw-cli

# Or set Tavily API key
export TAVILY_API_KEY=your_key_here
```

## Running the Full Stack

### Option 1: All-in-one launch script
```bash
cd /Users/loganathanev/workspace/daily-apps/2026/10/slm-agentic-eval-harness
chmod +x launch_infinity.sh
./launch_infinity.sh
```

### Option 2: Manual step-by-step

#### Step 1: Build tool servers
```bash
cd infinity-tool-servers
cargo build --release --workspace
```

#### Step 2: Start tool servers (each in own terminal, or background)
```bash
# Terminal 1: Web Search
./target/release/web-search-server --port 8001

# Terminal 2: Inference
./target/release/inference-server --port 8002 \
    --model-endpoint http://localhost:8080/v1 \
    --default-model qwen2.5-1.5b-instruct

# Terminal 3: MinHash
./target/release/minhash-server --port 8003

# Terminal 4: Review Queue
./target/release/review-server --port 8004 \
    --database-url sqlite:runs/slm_eval.db

# Terminal 5: Scoring
./target/release/scoring-server --port 8005

# Terminal 6: Gate
./target/release/gate-server --port 8006

# Terminal 7: Synthetic Generation
./target/release/synthetic-server --port 8007
```

#### Step 3: Start Infinity daemon
```bash
cd ..
infinity-daemon --config infinity-daemon.toml
```

#### Step 4: Launch evaluation agent
```bash
infinity thread launch --config infinity-daemon.toml \
  --input '{"run_id":"eval-1","dataset":"data/cord","model":"qwen2.5-1.5b-instruct"}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Infinity Daemon                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Agent: EvalOrchestrator (hibernates on each tool call) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│         RAP/HTTP callbacks (async, ~bytes memory)              │
│                          ▼                                       │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────┐  │
│  │WebSearch│ │Inference │ │MinHash │ │ Review   │ │Scoring │  │
│  │ :8001   │ │ :8002    │ │ :8003  │ │ :8004    │ │ :8005  │  │
│  └─────────┘ └──────────┘ └────────┘ └──────────┘ └────────┘  │
│  ┌────────┐ ┌──────────┐                                       │
│  │ Gate   │ │ Synthetic│                                       │
│  │ :8006  │ │ :8007    │                                       │
│  └────────┘ └──────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Benefits for Low Memory

| Traditional | Infinity |
|-------------|----------|
| 1 eval process = full memory | 100 agents = ~bytes each (hibernated) |
| Sequential pipeline | Concurrent agents, parallel tools |
| Blocking on human review | Agent hibernates, wakes on callback |
| All tools in one process | Tools on separate servers (can be remote) |

## Testing Individual Tools

```bash
# Test web search
curl http://localhost:8001/.well-known/rap-toolset | jq

# Test inference
curl -X POST http://localhost:8002/invoke \
  -H "Content-Type: application/json" \
  -d '{"group_id":"test","id":"1","call_id":"1","arguments":{"prompt":"Hello","json_mode":true},"callback_url":"http://localhost:9999/callback"}'

# Test scoring
curl -X POST http://localhost:8005/invoke \
  -H "Content-Type: application/json" \
  -d '{"group_id":"test","id":"1","call_id":"1","arguments":{"predictions_file":"runs/eval-1/predictions.jsonl","gold_file":"data/cord/cases.jsonl"},"callback_url":"http://localhost:9999/callback"}'
```

## Human Review Workflow

When the agent calls `human_review`:
1. Cases are inserted into `review_task` table
2. Agent **hibernates** (near-zero memory)
3. Human reviews via UI or API:
   ```bash
   curl -X POST http://localhost:8004/resolve \
     -H "Content-Type: application/json" \
     -d '{"task_id":"uuid","status":"approved","resolution":{"modified_case":{...}}}'
   ```
4. Callback wakes agent with resolution

## Stopping Everything

```bash
# If using launch script: Ctrl+C
# Manual:
pkill -f "web-search-server|inference-server|minhash-server|review-server|scoring-server|gate-server|synthetic-server|infinity-daemon"
```

## Next Steps

1. **Build actual Infinity agent** (Rust) using `infinity-agent-core` with the orchestrator prompt
2. **Add persistent conversation store** for multi-session continuity
3. **Implement distributed deployment** (tool servers on different machines)
4. **Add metrics/observability** (tracing, Prometheus)