#!/usr/bin/env bash
# Alias for the recommended full-training entrypoint (2x A800).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/train_2gpu.sh" "$@"
