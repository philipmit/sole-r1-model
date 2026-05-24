#!/usr/bin/env bash
# Launch script for the Qwen reward server on a single H100 node.

set -euo pipefail

# ---- Defaults ----
CHECKPOINT_PATH=""
PORT=8001

usage() {
    echo "Usage: $0 --checkpoint-path <path> [--port <port>]"
    echo ""
    echo "  --checkpoint-path   Path to the model checkpoint directory (required)"
    echo "  --port              ZMQ server port (default: 8001)"
    exit 1
}

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint-path is required."
    echo ""
    usage
fi

# ---- Check for NVIDIA GPU ----
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. An NVIDIA GPU with drivers installed is required."
    exit 1
fi
echo "Detected GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Install uv if not present ----
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- Install Python dependencies ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing Python dependencies via uv..."
uv sync

# ---- Launch server ----
echo ""
echo "========================================"
echo "Starting Qwen reward server"
echo "  Checkpoint : $CHECKPOINT_PATH"
echo "  Port       : $PORT"
echo "========================================"

uv run python src/reward_server/main.py \
    "checkpoint_path=$CHECKPOINT_PATH" \
    "server_port=$PORT"
