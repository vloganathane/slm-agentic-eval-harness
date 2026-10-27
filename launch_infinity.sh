#!/usr/bin/env bash
# Launch script for Infinity + slm_eval integration
# 
# This starts all RAP tool servers and the Infinity daemon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Infinity slm_eval Integration Launcher ==="
echo "Project: $PROJECT_ROOT"

# Check prerequisites
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 not found. Please install it first."
        exit 1
    fi
}

check_cmd cargo
check_cmd infinity-daemon
check_cmd infinity

# Build tool servers
echo ""
echo "=== Building tool servers ==="
cd "$PROJECT_ROOT/infinity-tool-servers"
cargo build --release --workspace

# Start tool servers
echo ""
echo "=== Starting tool servers ==="

# Web Search Server (port 8001)
echo "Starting web-search-server on port 8001..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/web-search-server" --port 8001 &
WEB_PID=$!
sleep 1

# Inference Server (port 8002) - requires llama-server running on 8080
echo "Starting inference-server on port 8002..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/inference-server" --port 8002 \
    --model-endpoint http://localhost:8080/v1 \
    --default-model qwen2.5-1.5b-instruct &
INF_PID=$!
sleep 1

# MinHash Server (port 8003)
echo "Starting minhash-server on port 8003..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/minhash-server" --port 8003 &
MINHASH_PID=$!
sleep 1

# Review Server (port 8004)
echo "Starting review-server on port 8004..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/review-server" --port 8004 \
    --database-url sqlite:runs/slm_eval.db &
REVIEW_PID=$!
sleep 1

# Scoring Server (port 8005)
echo "Starting scoring-server on port 8005..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/scoring-server" --port 8005 &
SCORING_PID=$!
sleep 1

# Gate Server (port 8006)
echo "Starting gate-server on port 8006..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/gate-server" --port 8006 &
GATE_PID=$!
sleep 1

# Synthetic Server (port 8007)
echo "Starting synthetic-server on port 8007..."
"$PROJECT_ROOT/infinity-tool-servers/target/release/synthetic-server" --port 8007 &
SYNTH_PID=$!
sleep 1

echo ""
echo "=== All tool servers started ==="
echo "PIDs: Web=$WEB_PID Inf=$INF_PID MinHash=$MINHASH_PID Review=$REVIEW_PID Scoring=$SCORING_PID Gate=$GATE_PID Synth=$SYNTH_PID"

# Verify toolsets are discoverable
echo ""
echo "=== Verifying toolset manifests ==="
for port in 8001 8002 8003 8004 8005 8006 8007; do
    echo "Checking port $port..."
    curl -s "http://localhost:$port/.well-known/rap-toolset" | jq '.tools[].name' 2>/dev/null || echo "  (curl/jq not available or server not ready)"
done

# Start Infinity daemon
echo ""
echo "=== Starting Infinity daemon ==="
cd "$PROJECT_ROOT"
infinity-daemon --config "$PROJECT_ROOT/infinity-daemon.toml" &
DAEMON_PID=$!
sleep 2

echo ""
echo "=== Infinity daemon started (PID: $DAEMON_PID) ==="
echo ""
echo "=== Ready to launch eval agents ==="
echo ""
echo "Example launch:"
echo "  infinity thread launch --config infinity-daemon.toml \\"
echo "    --input '{\"run_id\":\"eval-1\",\"dataset\":\"data/cord\",\"model\":\"qwen2.5-1.5b-instruct\"}'"
echo ""
echo "To stop all servers:"
echo "  kill $WEB_PID $INF_PID $MINHASH_PID $REVIEW_PID $SCORING_PID $GATE_PID $SYNTH_PID $DAEMON_PID"
echo ""
echo "Or press Ctrl+C to stop this script and all children."

# Trap cleanup
cleanup() {
    echo ""
    echo "=== Shutting down ==="
    kill $WEB_PID $INF_PID $MINHASH_PID $REVIEW_PID $SCORING_PID $GATE_PID $SYNTH_PID $DAEMON_PID 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Keep running
wait