#!/usr/bin/env python3
"""功能分组测试入口（feature-test）。

域列表和用例数以 --list 为准。exit 0=全绿。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIR = REPO / "mas"
TESTS = TIR / "tests"

# 功能域 → (测试模块前缀列表, 中文说明)
FEATURES: dict[str, tuple[list[str], str]] = {
    "mas-core": (
        ["test_stage1_mas", "test_stage5_mas_sockets", "test_spec", "test_compiler"],
        "MAS 基础层: spec/compiler/依赖红线/奖励/mock 采集/memory sockets/runner 注入",
    ),
    "rl": (["test_stage2_rl"], "RL overlay: TrainSignal/Archive resume 边界"),
    "harness": (["test_stage3_harness", "test_stage6_harness_thin"], "Harness: log/loss 波动/认知收敛/reward hacking"),
    "branch": (
        ["test_gates_and_rae", "test_branch_policy_activeset", "test_phase_abcd_rae_activeset"],
        "分支采样: gates/RAE/active set/plan_forks 轨迹站点/k-hop credit",
    ),
    "rollout-tree": (["test_rollout_tree"], "RolloutTree 契约: 建树/leaves/path/JSON round-trip"),
    "agent-framework": (["test_agent_framework"], "agent 化: AgentRegistry 双后端/RouterSpec/两层 memory/PEV per-agent events"),
    "schema03": (
        [
            "test_schema03_tool_agents",
            "test_protocol",
            "test_router",
            "test_runtime_dispatch",
            "test_tool_agents",
            "test_centralized_integration",
            "test_palette_contract",
            "test_template_combinations",
        ],
        "schema 0.3: sugar/协议/router fan-out/HIVE tool-agent/中心化循环/palette 契约/模板组合 episode",
    ),
    "daemon": (["test_daemon_expand"], "Daemon: expansion enqueue/_rollout_trees 存储"),
    "realtime": (["test_realtime_harness"], "实时 Harness: stdout JSONL 帧/SSE 过滤/Diagnoser.consume"),
    "cli": (["test_stage4_cli"], "CLI: status HTML/子进程闭环"),
    "control-ui": (["test_stage9_ui", "test_stage10_control_ui"], "Control: dashboard/API 端点/rl.yaml CLI"),
    "gpu-compiler": (["test_stage11_gpu_compiler"], "GPU 编译: compiler/compiled collect/api"),
    "verifier": (["test_stage8_verifier"], "Verifier: registry/hop 反馈"),
    "e2e": (["test_stage7_e2e"], "端到端 C1-C8: 采集→reward→诊断→fork→CLI"),
    "functional": (
        [
            "test_functional_workflow",
            "test_functional_tools",
            "test_functional_harness",
            "test_functional_rl",
            "test_functional_runtime_ui",
            "test_functional_studio",
        ],
        "功能合同: workflow/tool/harness/RL/SSE/Studio 产品面",
    ),
    "user-space": (
        ["test_user_space", "test_hive_wrap", "test_epc_aw_wrap", "test_assistant_chat"],
        "用户区隔离: 路径监狱/HIVE/EPC-AW 封装/辅助对话",
    ),
    "studio": (
        [
            "test_functional_studio",
            "test_eval_sources",
            "test_task_files",
            "test_training_preflight",
            "test_training_lifecycle",
            "test_experiment_config",
        ],
        "Studio 控制面: 算法/palette/单题调试/eval/训练门禁/实验 bundle",
    ),
    "sampling-runtime": (
        [
            "test_arpo_sampling_adapter",
            "test_sampling_core",
            "test_sampling_policy_contract",
            "test_rl_config_contract",
            "test_training_agent_tracing",
            "test_execution_tree",
            "test_rollout_tree_results",
            "test_rollout_tree_archive",
        ],
        "pytest: SamplingCore/adapter/execution-tree/rollout-tree 归档",
    ),
    "frontend": ([], "前端 vitest（webui npm run test:traj）"),
    "smoke": ([], "无 GPU 冒烟: doctor→层依赖→user_zone→mock 采集→diagnose→dashboard"),
}

# Python 单测域（--all 默认包含）
UNIT_FEATURES = [
    "mas-core",
    "rl",
    "harness",
    "branch",
    "rollout-tree",
    "agent-framework",
    "schema03",
    "daemon",
    "realtime",
    "cli",
    "control-ui",
    "gpu-compiler",
    "verifier",
    "e2e",
    "functional",
    "user-space",
    "studio",
    "sampling-runtime",
]


def run_pytest_suite(modules: list[str]) -> tuple[bool, int, int, float]:
    """Run pytest-only modules that unittest discover does not load."""
    paths = [str(TESTS / f"{m}.py") for m in modules]
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    n_pass = n_fail = n_skip = 0
    for line in out.splitlines():
        if "passed" in line or "failed" in line or "skipped" in line:
            import re

            m_pass = re.search(r"(\d+) passed", line)
            m_fail = re.search(r"(\d+) failed", line)
            m_err = re.search(r"(\d+) error", line)
            m_skip = re.search(r"(\d+) skipped", line)
            if m_pass:
                n_pass = int(m_pass.group(1))
            if m_fail:
                n_fail += int(m_fail.group(1))
            if m_err:
                n_fail += int(m_err.group(1))
            if m_skip:
                n_skip = int(m_skip.group(1))
    n_total = n_pass + n_fail + n_skip
    if r.returncode != 0:
        print(out[-2000:])
    else:
        print("  " + "\n  ".join(l for l in out.splitlines() if "passed" in l or "failed" in l)[-3:])
    return (r.returncode == 0, max(n_total, n_pass), n_fail, dt)


def run_python_suite(modules: list[str]) -> tuple[bool, int, int, float]:
    """跑一组测试模块（unittest discover 模式过滤模块名）。

    cwd 保持仓库根（与 `.venv/bin/python -m unittest discover -s mas/tests -t .`
    一致）：部分测试用相对路径读 experiments/…/workflow.yaml。
    """
    sys.path.insert(0, str(TIR))
    sys.path.insert(0, str(REPO))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for m in modules:
        suite.addTests(loader.discover(str(TESTS), pattern=f"{m}.py"))
    t0 = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    dt = time.time() - t0
    n_run = result.testsRun
    n_fail = len(result.failures) + len(result.errors)
    return (n_fail == 0, n_run, n_fail, dt)


def run_frontend() -> tuple[bool, int, int, float]:
    """webui vitest（test:traj）。"""
    webui = REPO / "webui"
    if not (webui / "package.json").is_file():
        return (False, 0, 1, 0.0)
    if not (webui / "node_modules" / "vitest").is_dir():
        print("  npm install（首次，补 vitest）…")
        subprocess.run(["npm", "install"], cwd=webui, check=False)
    t0 = time.time()
    r = subprocess.run(["npm", "run", "test:traj"], cwd=webui, capture_output=True, text=True)
    dt = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    # vitest 输出 "Tests  9 passed (9)"
    n_pass = 0
    n_total = 0
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Tests"):
            parts = line.split()
            # "Tests  9 passed (9)"
            if "passed" in parts:
                idx = parts.index("passed") - 1
                try:
                    n_pass = int(parts[idx])
                except (ValueError, IndexError):
                    pass
            n_total = n_pass
    print(out[-1500:] if not r.returncode == 0 else "\n".join(l for l in out.splitlines() if "Tests" in l or "Test Files" in l))
    return (r.returncode == 0 and n_pass > 0, n_total, n_total - n_pass, dt)


def run_smoke() -> tuple[bool, int, int, float]:
    """等价 ./run.sh smoke 的核心路径（doctor/deps/collect mock/diagnose/dashboard）。"""
    py = sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{TIR}"
    out_dir = REPO / "artifacts" / "run_feature_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = out_dir / "traj.json"
    cmds = [
        [py, "-m", "science_infra.ui.cli", "doctor"],
        [py, str(TIR / "scripts" / "check_workflow_deps.py")],
        [py, str(TIR / "scripts" / "check_user_zone.py")],
        [py, "-m", "science_infra.ui.cli", "collect", "--mock", "--n", "2", "--out", str(traj)],
        [py, "-m", "science_infra.ui.cli", "diagnose", str(traj)],
        [py, "-m", "science_infra.ui.cli", "dashboard", str(traj), "--html", str(out_dir / "dashboard.html")],
    ]
    t0 = time.time()
    n_ok = 0
    for c in cmds:
        r = subprocess.run(c, cwd=REPO, env=env, capture_output=True, text=True)
        mark = "PASS" if r.returncode == 0 else "FAIL"
        print(f"    [{mark}] {' '.join(c[1:4])}…")
        if r.returncode == 0:
            n_ok += 1
        else:
            tail = "\n".join((r.stderr or r.stdout or "").splitlines()[-8:])
            print("      " + "\n      ".join(tail.splitlines()))
    dt = time.time() - t0
    return (n_ok == len(cmds), len(cmds), len(cmds) - n_ok, dt)


def main() -> int:
    p = argparse.ArgumentParser(description="功能分组测试（feature-test）")
    p.add_argument("features", nargs="*", help="功能域名（--list 查看）")
    p.add_argument("--list", action="store_true", help="列出所有功能域")
    p.add_argument("--all", action="store_true", help="全部单测域 + frontend（smoke 除外）")
    args = p.parse_args()

    if args.list:
        print("功能域（feature-test）:")
        for k, (mods, desc) in FEATURES.items():
            n = sum(_count_tests(TESTS / f"{m}.py") for m in mods)
            extra = f"  [{n} tests]" if n else ""
            print(f"  {k:<18} {desc}{extra}")
        print("\n用法: python scripts/feature_test.py <域> [<域>…] | --all | --list")
        return 0

    selected: list[str] = []
    if args.all:
        selected = list(UNIT_FEATURES) + ["frontend"]
    elif args.features:
        for f in args.features:
            if f not in FEATURES:
                print(f"未知功能域: {f}（--list 查看）")
                return 1
            selected.append(f)
    else:
        print("未指定功能域。--list 查看 / --all 全量")
        return 1

    results: list[tuple[str, bool, int, int, float]] = []
    print(f"\n功能测试 → {len(selected)} 个域: {', '.join(selected)}\n")
    for feat in selected:
        mods, desc = FEATURES[feat]
        print("=" * 60)
        print(f"[{feat}] {desc}")
        print("=" * 60)
        if feat == "frontend":
            okflag, n, nf, dt = run_frontend()
        elif feat == "smoke":
            okflag, n, nf, dt = run_smoke()
        elif feat == "sampling-runtime":
            okflag, n, nf, dt = run_pytest_suite(mods)
        else:
            okflag, n, nf, dt = run_python_suite(mods)
        mark = "PASS" if okflag else "FAIL"
        print(f"\n  → [{mark}] {feat}: {n - nf}/{n} 通过, {dt:.1f}s\n")
        results.append((feat, okflag, n, nf, dt))

    print("\n" + "=" * 60)
    print("汇总矩阵")
    print("=" * 60)
    n_pass = n_total = 0
    n_feat_ok = 0
    for feat, okflag, n, nf, dt in results:
        mark = "✅" if okflag else "❌"
        print(f"  {mark} {feat:<18} {n - nf:>3}/{n:<3}  {dt:6.1f}s")
        n_pass += n - nf
        n_total += n
        if okflag:
            n_feat_ok += 1
    print("=" * 60)
    print(f"  功能域 {n_feat_ok}/{len(results)} 绿 · 测试 {n_pass}/{n_total} 通过")
    if n_feat_ok != len(results):
        return 1
    return 0


def _count_tests(path: Path) -> int:
    """静态统计一个测试文件的 test 方法数（粗略，用于 --list 展示）。"""
    if not path.is_file():
        return 0
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"def (test_\w+)\(", text))


if __name__ == "__main__":
    raise SystemExit(main())
