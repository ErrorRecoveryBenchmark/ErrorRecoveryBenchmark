#!/usr/bin/env bash
# Export the public tree with pinned dependencies / 导出公开代码并保留依赖版本。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/prepare_github_release.py" "${1:-./github_release}"
