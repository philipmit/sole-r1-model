#!/usr/bin/env bash

set -euo pipefail

HOST=${1:-10.151.1.36}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Install uv if not present ----
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- Install Python dependencies ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv sync

# ---- Run client ----
uv run python main.py \
  --host "$HOST" \
  --front ../example/open_drawer/front.webm \
  --wrist ../example/open_drawer/wrist.webm \
  --task "open drawer" \
  --video-output drawer.webm
