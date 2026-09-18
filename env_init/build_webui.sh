#!/usr/bin/env bash
# Build Science Control webui (Vite) into webui/dist for ./run.sh ui
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEBUI="${ROOT}/webui"

if [[ ! -f "${WEBUI}/package.json" ]]; then
  echo "Skip webui build: ${WEBUI}/package.json not found"
  exit 0
fi

bash "${SCRIPT_DIR}/install_nodejs.sh"

cd "${WEBUI}"
if [[ ! -d node_modules ]]; then
  echo "npm install (webui)..."
  npm install
fi

if [[ "${WEBUI_REBUILD:-0}" == "1" ]] || [[ ! -f dist/index.html ]]; then
  echo "npm run build (webui)..."
  npm run build
else
  echo "webui/dist already present (set WEBUI_REBUILD=1 to force rebuild)"
fi

echo "webui ready: ${WEBUI}/dist/index.html"
