"""Process-group lifecycle and durable run status for Control jobs."""

from __future__ import annotations

import json
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
    popen: subprocess.Popen
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
            if process is None or process.popen.poll() is not None:
                return None
            return process

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

            process = ManagedProcess(
                run_id=run_id,
                kind=kind,
                experiment_id=experiment_id,
                popen=popen,
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
                if process.stop_requested:
                    process.state = "cancelled"
                elif code == 0:
                    process.state = "succeeded"
                else:
                    process.state = "failed"
                    process.failure_stage = process.failure_stage or "process"
                    process.message = process.message or f"训练进程退出码 {code}"
                if self._by_kind.get(kind) == run_id:
                    self._by_kind.pop(kind, None)
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
                return None
            if experiment_id and process.experiment_id != experiment_id:
                raise ValueError("run does not belong to experiment")
            if process.popen.poll() is not None:
                return self._status_dict(process)
            self._stop_process(process, timeout=timeout, reason=reason)
            return self._status_dict(process)

    def _stop_process(
        self, process: ManagedProcess, *, timeout: float, reason: str
    ) -> None:
        if process.popen.poll() is not None:
            return
        process.stop_requested = True
        process.stop_reason = reason
        process.state = "stopping"
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
        self._interrupt(process.popen)
        try:
            process.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.popen.terminate()
            try:
                process.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.popen.kill()
                process.popen.wait(timeout=5)
        process.returncode = process.popen.returncode
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
    def _interrupt(popen: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                popen.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(popen.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            popen.terminate()

    def _status_dict(self, process: ManagedProcess) -> Dict[str, Any]:
        running = process.popen.poll() is None
        return {
            "run_id": process.run_id,
            "kind": process.kind,
            "experiment_id": process.experiment_id,
            "state": process.state,
            "running": running,
            "pid": process.popen.pid,
            "returncode": None if running else process.popen.returncode,
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
            if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
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
        elif state in ACTIVE_STATES:
            state = "interrupted"
        result = {**row, "state": state, "running": False}
        with self._lock:
            self._disk_cache[path] = (stat.st_mtime_ns, stat.st_size, result)
        return dict(result)

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
