#!/usr/bin/env bash
# Download and verify data / 下载、校验并解包数据；无需 huggingface-cli。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/download_data.py" "$@"
