#!/usr/bin/env bash
set -euo pipefail

main() {
  if [[ $# -gt 0 ]]; then
    echo "用法: bash deploy_webui.sh"
    echo "自动拉取、停止网站、确认清理训练、按需构建并启动。无需参数。"
    return 0
  fi

  local root script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
  cd "${root}"
  if [[ "$(git branch --show-current)" != "integration/new-webui" ]]; then
    echo "请先切换到 integration/new-webui；不会自动切换分支。" >&2
    return 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "服务器有未提交修改，请先处理；不会自动覆盖。" >&2
    git status --short
    return 1
  fi

  git pull --ff-only origin integration/new-webui
  git --no-pager log -1 --format='部署版本: %h %s'
  export PYTHONPATH="${root}:${root}/mas${PYTHONPATH:+:${PYTHONPATH}}"
  .venv/bin/python -c "import science_infra.ui.cli, psutil"

  local stamp="artifacts/control/webui-build-trees" sources previous="" build_flag="--no-build"
  sources="$(git rev-parse HEAD:webui HEAD:agent-lightning/dashboard)"
  if [[ -f "${stamp}" ]]; then previous="$(cat "${stamp}")"; fi
  if [[ "${sources}" != "${previous}" || ! -f webui/dist/index.html || ! -f agent-lightning/agentlightning/dashboard/index.html ]]; then
    if [[ ! -d webui/node_modules || ! -d agent-lightning/dashboard/node_modules ]]; then
      echo "前端需要构建，但依赖缺失。请先按文档安装依赖。" >&2
      return 1
    fi
    build_flag="--rebuild"
    echo "首次部署、前端有变化或构建产物缺失，本次自动 rebuild。"
  else
    echo "前端未变化，跳过 rebuild。"
  fi

  echo "停止网站并清理本项目训练；发现进程时只需确认一次 CLEAN。"
  bash ./run.sh ui --stop
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  if ! .venv/bin/python scripts/cleanup_training.py; then
    echo "清理取消或失败，网站保持停止；处理后重新运行本脚本。" >&2
    return 1
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

  bash ./run.sh ui "${build_flag}" --port 8787 --daemon
  curl --noproxy '*' --fail --silent --show-error --max-time 10 http://127.0.0.1:8787/api/health
  mkdir -p "$(dirname "${stamp}")"
  printf '%s\n' "${sources}" > "${stamp}"
  printf '\n网站已启动: http://127.0.0.1:18787/\n'
  echo "网站日志: artifacts/run_smoke/ui.log"
}

main "$@"
