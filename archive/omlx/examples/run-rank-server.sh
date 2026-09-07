#!/bin/zsh
set -euo pipefail

: "${DSV4_WORK:?Set DSV4_WORK to the editable oMLX/MLX environment}"
: "${DSV4_MODEL:?Set DSV4_MODEL to this rank's official checkpoint path}"
: "${DSV4_SERVER_EP:?Set DSV4_SERVER_EP to runtime/server_ep2.py}"

RUNTIME_DIR=${DSV4_RUNTIME_DIR:-/tmp/deepseek-v4}
mkdir -p "$RUNTIME_DIR"

export OMLX_MTP_ENABLED=1
export OMLX_MTP_DRAFT_TOKENS=5
export OMLX_MTP_ROWWISE_BATCH=${OMLX_MTP_ROWWISE_BATCH:-0}
: "${DSV4_CONTROL_HOST:?Set DSV4_CONTROL_HOST to rank 0's direct-link address}"
export DSV4_CONTROL_PORT=${DSV4_CONTROL_PORT:-29650}
export DSV4_READY_FILE=${DSV4_READY_FILE:-$RUNTIME_DIR/ready}

exec /usr/bin/caffeinate -is \
  "$DSV4_WORK/venv/bin/python" "$DSV4_SERVER_EP" \
  --model "$DSV4_MODEL" \
  --host 127.0.0.1 --port 8888 \
  --decode-concurrency 6 --prompt-concurrency 6 \
  --prefill-step-size 512 \
  --prompt-cache-size 64 --prompt-cache-bytes 12GB \
  --max-tokens 1048576 --log-level INFO
