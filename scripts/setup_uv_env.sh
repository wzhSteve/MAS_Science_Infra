#!/usr/bin/env bash
# 将 AgentFlow/.venv 的全部包同步到本仓库 .venv（uv 环境，禁止 miniconda）
#
# 策略：完整 rsync 复制（与 AgentFlow 包集合一致；freeze 全量 resolve 会因版本冲突失败）
# 之后用 uv 把 science-infra / 仓内 agent-lightning 装成 editable。
#
# 用法:
#   bash scripts/setup_uv_env.sh
#   bash scripts/setup_uv_env.sh --from-agentflow   # 默认行为
#   bash scripts/setup_uv_env.sh --minimal          # 仅 live 依赖（旧行为）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AF_VENV="${AGENTFLOW_VENV:-/root/autodl-tmp/AgentFlow/.venv}"
DST="${ROOT}/.venv"
MODE="${1:---from-agentflow}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 未找到 uv。安装: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${ROOT}"

if [[ "${MODE}" == "--minimal" ]]; then
  echo "+ uv venv --python 3.11 --prompt science-infra"
  uv venv --python 3.11 --prompt science-infra
  echo "+ uv pip install -e '.[live,dev]' --python .venv/bin/python"
  uv pip install -e '.[live,dev]' --python .venv/bin/python
else
  [[ -d "${AF_VENV}" ]] || {
    echo "ERROR: 找不到 AgentFlow venv: ${AF_VENV}" >&2
    echo "可设置 AGENTFLOW_VENV=/path/to/.venv 或改用: $0 --minimal" >&2
    exit 1
  }
  echo "+ rsync -a ${AF_VENV}/ -> ${DST}/  (完整复制全部包)"
  rm -rf "${DST}"
  mkdir -p "${DST}"
  rsync -a "${AF_VENV}/" "${DST}/"

  echo "+ rewrite VIRTUAL_ENV / shebang paths"
  if [[ -d "${DST}/bin" ]]; then
    # text files only
    find "${DST}/bin" -type f -print0 | while IFS= read -r -d '' f; do
      if grep -q '/root/autodl-tmp/AgentFlow/.venv\|'"${AF_VENV}" "$f" 2>/dev/null; then
        if file "$f" | grep -q 'text\|empty\|script'; then
          sed -i "s|/root/autodl-tmp/AgentFlow/.venv|${DST}|g; s|${AF_VENV}|${DST}|g" "$f"
        fi
      fi
    done
  fi
  if [[ -f "${DST}/pyvenv.cfg" ]]; then
    sed -i 's|^prompt = .*|prompt = science-infra|' "${DST}/pyvenv.cfg"
  fi

  echo "+ uv pip install -e . --no-deps (science-infra；避免改动 AgentFlow 已钉版本)"
  uv pip install -e "${ROOT}" --python "${DST}/bin/python" --no-deps

  if [[ -d "${ROOT}/agent-lightning" ]]; then
    echo "+ uv pip install -e ./agent-lightning --no-deps"
    uv pip install -e "${ROOT}/agent-lightning" --python "${DST}/bin/python" --no-deps
  fi
fi

echo
echo "OK: ${DST}"
echo "  python: $(${DST}/bin/python -V)"
${DST}/bin/python -c "import langchain, langgraph; print('langchain+langgraph OK')"
${DST}/bin/python -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "torch: (not in --minimal)"
echo "包数量: $(uv pip freeze --python "${DST}/bin/python" | wc -l)"
echo "用法: source .venv/bin/activate   或直接 ./run.sh live"
