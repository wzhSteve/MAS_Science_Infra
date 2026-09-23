"""Confirm and stop project-owned training processes, including orphan workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "mas" / "train_tir_agent.py"
WORKER_NAMES = ("ray::WorkerDict", "ray::TaskRunner", "ray::PatchedvLLMServer", "VLLM::")


def discover() -> dict[int, psutil.Process]:
    targets: dict[int, psutil.Process] = {}
    for process in psutil.process_iter():
        try:
            if process.pid == os.getpid() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            args = process.cmdline()
            cwd = Path(process.cwd()).resolve()
            training = any(
                arg.endswith("train_tir_agent.py")
                and (cwd / arg).resolve() == TRAIN_SCRIPT
                for arg in args
            )
            project_cwd = cwd == ROOT or ROOT in cwd.parents
            worker = bool(args) and (
                args[0].startswith(WORKER_NAMES)
                or any(arg.endswith(("/raylet", "/ray/_private/workers/default_worker.py"))
                       for arg in args)
            )
            if training or (project_cwd and worker):
                targets[process.pid] = process
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            print(f"无法检查 PID {process.pid}，不会自动清理该进程。", file=sys.stderr)
    capture_children(targets)
    return targets


def capture_children(targets: dict[int, psutil.Process]) -> None:
    for process in list(targets.values()):
        try:
            if not process.is_running():
                continue
            if process.ppid() in targets:
                continue
            for child in process.children(recursive=True):
                targets[child.pid] = child
        except psutil.NoSuchProcess:
            continue


def alive(targets: dict[int, psutil.Process]) -> list[psutil.Process]:
    result = []
    for process in targets.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                result.append(process)
        except psutil.NoSuchProcess:
            continue
    return result


def describe(process: psutil.Process) -> str:
    try:
        args = process.cmdline()
        # Do not dump arbitrary command arguments or environment credentials.
        role = args[0] if args and args[0].startswith(WORKER_NAMES) else process.name()
        return f"PID {process.pid:<8} {role[:80]}  cwd={process.cwd()}"
    except psutil.NoSuchProcess:
        return f"PID {process.pid} 已退出"


def stop(targets: dict[int, psutil.Process]) -> None:
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        deadline = time.monotonic() + timeout
        signalled: set[tuple[int, float]] = set()
        while True:
            capture_children(targets)
            remaining = alive(targets)
            if not remaining:
                return
            for process in reversed(remaining):
                try:
                    identity = (process.pid, process.create_time())
                    if identity not in signalled:
                        process.send_signal(sig)
                        signalled.add(identity)
                except psutil.NoSuchProcess:
                    continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    remaining = alive(targets)
    if remaining:
        raise RuntimeError(f"清理未完成，残留 PID：{[process.pid for process in remaining]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只列出，不发送信号")
    parser.add_argument("--yes", action="store_true", help="跳过 CLEAN 确认")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("此清理工具仅用于 Linux 训练服务器。")
    targets = discover()
    if not targets:
        print("没有发现可确认属于本项目的训练进程。请通过 nvidia-smi 确认剩余占用。")
        return 0
    print(f"项目：{ROOT}\n以下训练进程及其后代将被停止：")
    for process in targets.values():
        print(describe(process))
    if args.dry_run:
        return 0
    print("仅清理上列 PID 及停止期间发现的后代；不会清理其他项目、删除日志或模型。")
    if not args.yes and input("确认清理请输入 CLEAN，其余输入取消：").strip() != "CLEAN":
        print("已取消清理。")
        return 1
    stop(targets)
    remaining = discover()
    if remaining:
        print("仍发现本项目训练进程，请勿启动新训练：", file=sys.stderr)
        for process in remaining.values():
            print(describe(process), file=sys.stderr)
        return 1
    print("所列训练进程已退出。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (psutil.Error, OSError, RuntimeError) as error:
        print(f"清理失败：{error}。网站如已停止，不会自动继续部署。", file=sys.stderr)
        raise SystemExit(1) from error
