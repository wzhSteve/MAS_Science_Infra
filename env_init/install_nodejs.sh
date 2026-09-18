#!/usr/bin/env bash
# Install Node.js 20 LTS + npm (required to build Science Control webui).
# Ubuntu apt nodejs is too old for Vite 5; use NodeSource.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

need_node=0
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  need_node=1
else
  # Vite 5 needs Node >= 18
  major="$(node -v | sed -E 's/^v([0-9]+).*/\1/')"
  if [[ "${major}" -lt 18 ]]; then
    echo "Node $(node -v) is too old for webui (need >= 18); upgrading via NodeSource..."
    need_node=1
  fi
fi

if [[ "${need_node}" != "1" ]]; then
  echo "Node.js OK: $(node -v), npm $(npm -v)"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get not found; install Node.js 20+ manually, then re-run." >&2
  exit 1
fi

echo "Installing Node.js 20 (NodeSource)..."
# curl may already be present on AutoDL; ensure ca-certificates for HTTPS
apt_get() {
  if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
    sudo apt-get "$@"
  else
    apt-get "$@"
  fi
}

export DEBIAN_FRONTEND=noninteractive
apt_get update -qq
apt_get install -y --no-install-recommends ca-certificates curl gnupg
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt_get install -y --no-install-recommends nodejs

echo "Node.js installed: $(node -v), npm $(npm -v)"
