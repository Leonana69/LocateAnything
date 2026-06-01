#!/usr/bin/env bash
# Launch the LocateAnything Flask test server.
#
# Usage:
#   bash scripts/run-server.sh                                  # defaults
#   MODEL_PATH=path/to/ckpt PORT=8080 bash scripts/run-server.sh
set -euo pipefail

# Ignore ~/.local (user-site) packages so a self-contained conda env is used.
# Avoids stale user-site builds (e.g. a numpy-1-compiled scikit-learn) shadowing
# the env and breaking imports under numpy 2.x. Override with PYTHONNOUSERSITE=0.
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-"cuda"}
DTYPE=${DTYPE:-"bfloat16"}

# Optional weight quantization (needs bitsandbytes). Set LOAD_IN_8BIT=1 to roughly
# halve GPU memory, or LOAD_IN_4BIT=1 for the smallest footprint. (The server also
# honors the LA_LOAD_IN_8BIT / LA_LOAD_IN_4BIT env vars directly.)
LOAD_IN_8BIT=${LOAD_IN_8BIT:-${LA_LOAD_IN_8BIT:-0}}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-${LA_LOAD_IN_4BIT:-0}}
EXTRA_ARGS=()
[ "$LOAD_IN_8BIT" = "1" ] && EXTRA_ARGS+=(--load-in-8bit)
[ "$LOAD_IN_4BIT" = "1" ] && EXTRA_ARGS+=(--load-in-4bit)

HERE="$(dirname "$0")"

python "$HERE/locateanything_server.py" \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${EXTRA_ARGS[@]}" \
    "$@"
