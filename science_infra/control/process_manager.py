"""Process-group lifecycle and durable run status for Control jobs."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psutil

from science_infra.control.events import BUS
from science_infra.control.experiments import artifacts_dir
from science_infra.control.paths import tir_agent_root

ACTIVE_STATES = {"preparing", "starting", "running", "stopping"}
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        f"{path.suffix}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class ManagedProcess:
    run_id: str
    kind: str
    experiment_id: str
    popen: Optional[subprocess.Popen]
    pid: int
    pid_create_time: float
    pgid: Optional[int]
    command_hash: str
    cwd: str
    log_path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    state: str = "running"
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    returncode: Optional[int] = None
    stop_requested: bool = False
    stop_reason: Optional[str] = None
    failure_stage: Optional[str] = None
    message: Optional[str] = None
    descendants: Dict[int, psutil.Process] = field(default_factory=dict)
    recovered: bool = False


class ProcessManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._procs: Dict[str, ManagedProcess] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._by_kind: Dict[str, str] = {}
        self._disk_cache: Dict[Path, tuple[int, int, Dict[str, Any]]] = {}

    def get(self, run_id: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self._procs.get(run_id)

    def active(self, kind: str) -> Optional[ManagedProcess]:
        with self._lock:
            run_id = self._by_kind.get(kind)
            process = self._procs.get(run_id) if run_id else None
            if process is not None and self._running(process):
                return process
        candidates = [
            row for row in self._disk_runs(None)
            if row.get("kind") == kind and row.get("running")
        ]
        if not candidates:
            return None
        row = max(candidates, key=lambda item: float(item.get("started_at") or 0))
        return self._adopt(row)

    def list_runs(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._procs.values())
            pending = list(self._pending.values())
        output: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in pending:
            if experiment_id and row["experiment_id"] != experiment_id:
                continue
            output.append(dict(row))
            seen.add(row["run_id"])
        for process in items:
            if experiment_id and process.experiment_id != experiment_id:
                continue
            row = self._status_dict(process)
            output.append(row)
            seen.add(process.run_id)
        output.extend(
            row
            for row in self._disk_runs(experiment_id)
            if row["run_id"] not in seen
        )
        output.sort(key=lambda row: float(row.get("started_at") or 0), reverse=True)
        return output

    def find_request(
        self, *, kind: str, experiment_id: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        if not request_id:
            return None
        return next(
            (
                row
                for row in self.list_runs(experiment_id)
                if row.get("kind") == kind
                and (row.get("meta") or {}).get("request_id") == request_id
            ),
            None,
        )

    def status(self, run_id: str, experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            process = self._procs.get(run_id)
            if process:
                return self._status_dict(process)
            pending = self._pending.get(run_id)
            if pending:
                return dict(pending)
        return self.disk_run(run_id, experiment_id)

    def write_preparing(
        self,
        *,
        run_id: str,
        kind: str,
        experiment_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        row = {
            "run_id": run_id,
            "kind": kind,
            "experiment_id": experiment_id,
            "state": "preparing",
            "running": False,
            "pid": None,
            "process_identity": None,
            "returncode": None,
            "log_path": str(self.run_dir(experiment_id, run_id) / "stdout.log"),
            "started_at": now,
            "ended_at": None,
            "stop_reason": None,
            "failure_stage": None,
            "message": None,
            "meta": dict(meta or {}),
        }
        with self._lock:
            self._pending[run_id] = row
        self._write_status(experiment_id, run_id, row)
        return row

    def mark_failed(
        self,
        *,
        run_id: str,
        experiment_id: str,
        kind: str,
        stage: str,
        message: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        previous = self.status(run_id) or self.write_preparing(
            run_id=run_id,
            kind=kind,
            experiment_id=experiment_id,
            meta=meta,
        )
        row = {
            **previous,
            "state": "failed",
            "running": False,
            "ended_at": time.time(),
            "failure_stage": stage,
            "message": message,
            "meta": dict(meta or previous.get("meta") or {}),
        }
        with self._lock:
            self._pending.pop(run_id, None)
        self._write_status(experiment_id, run_id, row)
        return row

    def start(
        self,
        *,
        kind: str,
        experiment_id: str,
        argv: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        replace: bool = True,
        run_id: Optional[str] = None,
    ) -> ManagedProcess:
        run_id = run_id or uuid4().hex[:12]
        with self._lock:
            existing = self.active(kind)
            if existing is not None:
                if not replace:
                    raise RuntimeError(f"{kind} already running: {existing.run_id}")
                self._stop_process(existing, timeout=15.0, reason="replaced")

            run_dir = self.run_dir(experiment_id, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "stdout.log"
            started_at = time.time()
            starting = {
                "run_id": run_id,
                "kind": kind,
                "experiment_id": experiment_id,
                "state": "starting",
                "running": False,
                "pid": None,
                "returncode": None,
                "log_path": str(log_path),
                "started_at": started_at,
                "ended_at": None,
                "stop_reason": None,
                "failure_stage": None,
                "message": None,
                "meta": dict(meta or {}),
            }
            self._write_status(experiment_id, run_id, starting)

            full_env = os.environ.copy()
            if env:
                full_env.update(env)
            full_env["PYTHONUNBUFFERED"] = "1"
            log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            options: Dict[str, Any] = {}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            try:
                popen = subprocess.Popen(
                    argv,
                    cwd=str(cwd or tir_agent_root()),
                    env=full_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    **options,
                )
            except Exception as error:
                log_file.close()
                self._pending.pop(run_id, None)
                failed = {
                    **starting,
                    "state": "failed",
                    "ended_at": time.time(),
                    "failure_stage": "process_start",
                    "message": str(error),
                }
                self._write_status(experiment_id, run_id, failed)
                raise

            try:
                pid_create_time = psutil.Process(popen.pid).create_time()
            except psutil.NoSuchProcess:
                pid_create_time = started_at
            process = ManagedProcess(
                run_id=run_id,
                kind=kind,
                experiment_id=experiment_id,
                popen=popen,
                pid=popen.pid,
                pid_create_time=pid_create_time,
                pgid=os.getpgid(popen.pid) if os.name != "nt" else None,
                command_hash=self._command_hash(argv),
                cwd=str((cwd or tir_agent_root()).resolve()),
                log_path=log_path,
                meta=dict(meta or {}),
                state="running",
                started_at=started_at,
            )
            self._procs[run_id] = process
            self._pending.pop(run_id, None)
            self._by_kind[kind] = run_id
            self._write_status(experiment_id, run_id, self._status_dict(process))

        def watch() -> None:
            code = popen.wait()
            with self._lock:
                process.returncode = code
                process.ended_at = time.time()
                if not process.stop_requested:
                    if code == 0:
                        process.state = "succeeded"
                    else:
                        process.state = "failed"
                        process.failure_stage = process.failure_stage or "process"
                        process.message = process.message or f"训练进程退出码 {code}"
                if not self._running(process) and self._by_kind.get(kind) == run_id:
                    self._by_kind.pop(kind, None)
                if self._running(process):
                    process.ended_at = None
                self._write_status(
                    experiment_id, run_id, self._status_dict(process)
                )
            log_file.close()
            BUS.publish(
                experiment_id,
                f"{kind}_done",
                {
                    "run_id": run_id,
                    "returncode": code,
                    "state": process.state,
                },
            )

        threading.Thread(target=watch, daemon=True).start()
        BUS.publish(
            experiment_id,
            f"{kind}_started",
            {"run_id": run_id, "argv": argv[:6], "state": "running"},
        )
        return process

    def stop(
        self,
        kind: str,
        *,
        timeout: float = 15.0,
        reason: str = "user_requested",
    ) -> Optional[Dict[str, Any]]:
        process = self.active(kind)
        if process is None:
            return None
        return self.stop_run(
            process.run_id,
            experiment_id=process.experiment_id,
            timeout=timeout,
            reason=reason,
        )

    def stop_run(
        self,
        run_id: str,
        *,
        experiment_id: Optional[str] = None,
        timeout: float = 15.0,
        reason: str = "user_requested",
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            process = self._procs.get(run_id)
        if process is None:
            row = self.disk_run(run_id, experiment_id)
            if not row or not row.get("running"):
                return None
            process = self._adopt(row)
            if process is None:
                return None
        with self._lock:
            if experiment_id and process.experiment_id != experiment_id:
                raise ValueError("run does not belong to experiment")
            if not self._running(process):
                return self._status_dict(process)
            self._stop_process(process, timeout=timeout, reason=reason)
            return self._status_dict(process)

    def _stop_process(
        self, process: ManagedProcess, *, timeout: float, reason: str
    ) -> None:
        if not self._running(process):
            return
        process.stop_requested = True
        process.stop_reason = reason
        process.state = "stopping"
        process.failure_stage = None
        process.message = None
        self._write_status(
            process.experiment_id,
            process.run_id,
            self._status_dict(process),
        )
        BUS.publish(
            process.experiment_id,
            f"{process.kind}_stopping",
            {"run_id": process.run_id, "reason": reason},
        )
        try:
            self._capture_descendants(process)
            if self._root_alive(process):
                self._signal_process(process, signal.SIGINT)
            if not self._wait_stopped(process, timeout):
                self._terminate_tree(process, kill=False)
                if not self._wait_stopped(process, 5):
                    self._terminate_tree(process, kill=True)
                    if not self._wait_stopped(process, 5):
                        remaining = [child.pid for child in self._live_descendants(process)]
                        if self._root_alive(process):
                            remaining.insert(0, process.pid)
                        raise RuntimeError(f"停止超时，仍有运行进程：{remaining}")
        except (OSError, psutil.Error, RuntimeError) as error:
            process.failure_stage = "stop"
            process.message = f"训练进程清理未完成：{error}"
            self._write_status(process.experiment_id, process.run_id, self._status_dict(process))
            logging.getLogger(__name__).error("Stop failed for run %s: %s", process.run_id, error)
            raise RuntimeError(process.message) from error
        process.returncode = process.popen.returncode if process.popen else None
        process.ended_at = time.time()
        process.state = "cancelled"
        self._write_status(
            process.experiment_id,
            process.run_id,
            self._status_dict(process),
        )
        BUS.publish(
            process.experiment_id,
            f"{process.kind}_stopped",
            {"run_id": process.run_id, "reason": reason},
        )

    @staticmethod
    def _signal_process(process: ManagedProcess, sig: int) -> None:
        try:
            if os.name == "nt":
                if process.popen is not None and sig == signal.SIGINT:
                    process.popen.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    psutil.Process(process.pid).send_signal(sig)
            elif process.pgid is not None:
                os.killpg(process.pgid, sig)
            else:
                os.kill(process.pid, sig)
        except ProcessLookupError:
            pass
        except psutil.NoSuchProcess:
            pass

    @staticmethod
    def _live_descendants(process: ManagedProcess) -> List[psutil.Process]:
        live = []
        for child in process.descendants.values():
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    live.append(child)
            except psutil.NoSuchProcess:
                continue
        return live

    def _running(self, process: ManagedProcess) -> bool:
        return self._root_alive(process) or bool(self._live_descendants(process))

    def _capture_descendants(self, process: ManagedProcess) -> None:
        parents = self._live_descendants(process)
        if self._root_alive(process):
            try:
                parents.append(psutil.Process(process.pid))
            except psutil.NoSuchProcess:
                pass
        parent_ids = {parent.pid for parent in parents}
        for parent in parents:
            try:
                if parent.ppid() in parent_ids:
                    continue
                for child in parent.children(recursive=True):
                    process.descendants[child.pid] = child
            except psutil.NoSuchProcess:
                continue

    def _wait_stopped(self, process: ManagedProcess, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self._running(process):
            self._capture_descendants(process)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
        return True

    def _terminate_tree(self, process: ManagedProcess, *, kill: bool) -> None:
        self._capture_descendants(process)
        # psutil retains process identity, including children that detached from the group.
        for child in reversed(self._live_descendants(process)):
            try:
                child.kill() if kill else child.terminate()
            except psutil.NoSuchProcess:
                continue
        if self._root_alive(process):
            try:
                self._signal_process(
                    process,
                    signal.SIGKILL if kill else signal.SIGTERM,
                )
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass

    def _status_dict(self, process: ManagedProcess) -> Dict[str, Any]:
        running = self._running(process)
        if process.recovered and not running and process.state in ACTIVE_STATES:
            process.state = "interrupted"
            process.ended_at = process.ended_at or time.time()
            process.message = process.message or "Control 重启后恢复的进程已结束，退出码不可用。"
        return {
            "run_id": process.run_id,
            "kind": process.kind,
            "experiment_id": process.experiment_id,
            "state": process.state,
            "running": running,
            "pid": process.pid,
            "process_identity": {
                "pid_create_time": process.pid_create_time,
                "pgid": process.pgid,
                "command_hash": process.command_hash,
                "cwd": process.cwd,
            },
            "recovered": process.recovered,
            "returncode": (
                None if running
                else process.popen.returncode if process.popen else process.returncode
            ),
            "log_path": str(process.log_path),
            "started_at": process.started_at,
            "ended_at": process.ended_at,
            "stop_reason": process.stop_reason,
            "failure_stage": process.failure_stage,
            "message": process.message,
            "meta": process.meta,
        }

    @staticmethod
    def run_dir(experiment_id: str, run_id: str) -> Path:
        return artifacts_dir(experiment_id) / "runs" / run_id

    def _write_status(
        self, experiment_id: str, run_id: str, row: Dict[str, Any]
    ) -> None:
        _write_json(self.run_dir(experiment_id, run_id) / "status.json", row)

    def tail_log(self, run_id: str, n: int = 160) -> str:
        path = self.resolve_log_path(run_id)
        if not path or not path.is_file():
            return ""
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-max(1, int(n)) :])

    def resolve_log_path(self, run_id: str) -> Optional[Path]:
        process = self.get(run_id)
        if process:
            return process.log_path
        row = self.disk_run(run_id)
        if not row:
            return None
        path = Path(str(row.get("log_path") or ""))
        return path if str(path) else None

    def disk_run(self, run_id: str, experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if experiment_id:
            return self._read_disk_run(self.run_dir(experiment_id, run_id), experiment_id)
        return next(
            (row for row in self._disk_runs(None) if row.get("run_id") == run_id),
            None,
        )

    def _read_disk_run(self, run_dir: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
        path = run_dir / "status.json"
        try:
            stat = path.stat()
            with self._lock:
                cached = self._disk_cache.get(path)
            if (
                cached
                and cached[:2] == (stat.st_mtime_ns, stat.st_size)
                and cached[2].get("state") in TERMINAL_STATES
            ):
                return dict(cached[2])
            row = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            with self._lock:
                self._disk_cache.pop(path, None)
            return None
        if not isinstance(row, dict):
            raise ValueError(f"Invalid run status: {run_dir}")
        row.setdefault("run_id", run_dir.name)
        row.setdefault("experiment_id", experiment_id)
        row.setdefault("log_path", str(run_dir / "stdout.log"))
        state = row.get("state")
        if not state:
            code = row.get("returncode")
            state = "succeeded" if code == 0 else "failed" if code is not None else "interrupted"
        running = state in ACTIVE_STATES and self._identity_matches(row)
        if state in ACTIVE_STATES and not running:
            state = "interrupted"
        result = {
            **row,
            "state": state,
            "running": running,
            "recovered": running,
        }
        if not running:
            with self._lock:
                self._disk_cache[path] = (stat.st_mtime_ns, stat.st_size, result)
        return dict(result)

    @staticmethod
    def _command_hash(argv: List[str]) -> str:
        payload = json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _identity_matches(cls, row: Dict[str, Any]) -> bool:
        pid = row.get("pid")
        identity = row.get("process_identity")
        if not isinstance(pid, int) or not isinstance(identity, dict):
            return False
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - float(identity["pid_create_time"])) > 0.01:
                return False
            if cls._command_hash(process.cmdline()) != identity.get("command_hash"):
                return False
            if str(Path(process.cwd()).resolve()) != str(Path(identity["cwd"]).resolve()):
                return False
            if os.name != "nt" and os.getpgid(pid) != identity.get("pgid"):
                return False
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (KeyError, OSError, TypeError, ValueError, psutil.Error):
            return False

    @staticmethod
    def _root_alive(process: ManagedProcess) -> bool:
        try:
            root = psutil.Process(process.pid)
            return (
                abs(root.create_time() - process.pid_create_time) <= 0.01
                and root.is_running()
                and root.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.Error:
            return False

    def _adopt(self, row: Dict[str, Any]) -> Optional[ManagedProcess]:
        if not row.get("running") or not self._identity_matches(row):
            return None
        identity = row["process_identity"]
        process = ManagedProcess(
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            experiment_id=str(row["experiment_id"]),
            popen=None,
            pid=int(row["pid"]),
            pid_create_time=float(identity["pid_create_time"]),
            pgid=int(identity["pgid"]) if identity.get("pgid") is not None else None,
            command_hash=str(identity["command_hash"]),
            cwd=str(identity["cwd"]),
            log_path=Path(str(row["log_path"])),
            meta=dict(row.get("meta") or {}),
            state=str(row.get("state") or "running"),
            started_at=float(row.get("started_at") or time.time()),
            ended_at=row.get("ended_at"),
            returncode=row.get("returncode"),
            stop_reason=row.get("stop_reason"),
            failure_stage=row.get("failure_stage"),
            message=row.get("message"),
            recovered=True,
        )
        with self._lock:
            existing = self._procs.get(process.run_id)
            if existing is not None:
                return existing if self._running(existing) else None
            self._procs[process.run_id] = process
            self._by_kind[process.kind] = process.run_id
        return process

    def _disk_runs(self, experiment_id: Optional[str]) -> List[Dict[str, Any]]:
        from science_infra.control.experiments import list_experiments

        experiment_ids = [experiment_id] if experiment_id else list_experiments()
        output: List[Dict[str, Any]] = []
        for current_id in experiment_ids:
            if not current_id:
                continue
            runs_dir = artifacts_dir(current_id) / "runs"
            if not runs_dir.is_dir():
                continue
            for run_dir in runs_dir.iterdir():
                status_path = run_dir / "status.json"
                if not status_path.is_file():
                    continue
                try:
                    row = self._read_disk_run(run_dir, current_id)
                except (OSError, ValueError) as error:
                    logging.getLogger(__name__).warning("Cannot read run status %s: %s", status_path, error)
                    continue
                if row is not None:
                    output.append(row)
        return output


PROCS = ProcessManager()
