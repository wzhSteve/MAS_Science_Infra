#!/usr/bin/env bash
set -euo pipefail

ask() {
  local answer
  while true; do
    read -r -p "$1 [y/N]: " answer
    case "${answer}" in
      y|Y|yes|YES) return 0 ;;
      ""|n|N|no|NO) return 1 ;;
      *) echo "请输入 y 或 n。" ;;
    esac
  done
}

main() {
  local root script_dir branch pull="" rebuild=""
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  branch="integration/new-webui"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pull) pull=1 ;;
      --no-pull) pull=0 ;;
      --rebuild) rebuild=1 ;;
      --no-rebuild) rebuild=0 ;;
      -h|--help)
        echo "用法: bash deploy_webui.sh [--pull|--no-pull] [--rebuild|--no-rebuild]"
        echo "未指定的选项逐项询问；启动前确认训练已停止。不安装依赖、不启动训练。"
        return 0
        ;;
      *) echo "未知选项: $1" >&2; return 1 ;;
    esac
    shift
  done
  if ! root="$(git -C "${script_dir}" rev-parse --show-toplevel)"; then
    echo "脚本不在 Git 仓库内，请放到项目根目录或 scripts 目录。" >&2
    return 1
  fi
  cd "${root}"
  if [[ "$(git branch --show-current)" != "${branch}" ]]; then
    echo "请先确认并切换到 ${branch}；脚本不自动切换分支。" >&2
    return 1
  fi

  echo "项目: ${root}"
  git --no-pager log -1 --format='当前版本: %h %s'
  if [[ -z "${pull}" ]]; then
    if ask "1. 拉取 origin/${branch} 最新代码？"; then pull=1; else pull=0; fi
  fi
  if [[ -z "${rebuild}" ]]; then
    if ask "2. 重新构建前端（前端源码有更新时选 y）？"; then rebuild=1; else rebuild=0; fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
  echo "重启 Control 不等于停止训练。请先结束活动训练并确认旧 Ray/vLLM 进程已清理。"
  if ! ask "3. 已确认可以更新并重启网站？"; then
    echo "已取消，未拉取或重启。"
    return 0
  fi

  if [[ "${pull}" == 1 ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "服务器工作目录有修改，请先处理；不会自动 stash、覆盖或删除。" >&2
      git status --short
      return 1
    fi
    git pull --ff-only origin "${branch}"
  fi
  git --no-pager log -1 --format='部署版本: %h %s'

  if [[ ! -x .venv/bin/python ]]; then
    echo "缺少服务器 .venv/bin/python，请先准备原训练环境。" >&2
    return 1
  fi
  export PYTHONPATH="${root}:${root}/mas${PYTHONPATH:+:${PYTHONPATH}}"
  .venv/bin/python -c "import science_infra.ui.cli"
  local build_flag="--no-build"
  if [[ "${rebuild}" == 1 ]]; then
    if [[ ! -d webui/node_modules || ! -d agent-lightning/dashboard/node_modules ]]; then
      echo "构建依赖缺失，请先按文档使用公网镜像安装；脚本不自动安装。" >&2
      return 1
    fi
    build_flag="--rebuild"
  elif [[ ! -f webui/dist/index.html || ! -f agent-lightning/agentlightning/dashboard/index.html ]]; then
    echo "缺少构建产物，请选择 --rebuild。尚未停止网站。" >&2
    return 1
  fi

  bash ./run.sh ui --stop
  bash ./run.sh ui "${build_flag}" --port 8787 --daemon
  curl --noproxy '*' --fail --silent --show-error --max-time 10 http://127.0.0.1:8787/api/health
  printf '\n网站已启动（后端 + 前端静态页面）。\n'
  echo "SSH 转发后的浏览器地址: http://127.0.0.1:18787/"
  echo "网站日志: artifacts/run_smoke/ui.log"
}

main "$@"
