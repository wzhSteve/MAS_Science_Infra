"""Process group manager for LLM serve / collect / train jobs."""

from __future__ import annotations

import json
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


@dataclass
class ManagedProcess:
    run_id: str
    kind: str  # llm | collect | train
    experiment_id: str
    popen: subprocess.Popen
    log_path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    returncode: Optional[int] = None


class ProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[str, ManagedProcess] = {}
        self._by_kind: Dict[str, str] = {}  # kind -> run_id (single active)

    def get(self, run_id: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self._procs.get(run_id)

    def active(self, kind: str) -> Optional[ManagedProcess]:
        with self._lock:
            rid = self._by_kind.get(kind)
            if not rid:
                return None
            return self._procs.get(rid)

    def list_runs(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._procs.values())
        out = []
        seen = set()
        for mp in items:
            if experiment_id and mp.experiment_id != experiment_id:
                continue
            row = self._status_dict(mp)
            out.append(row)
            seen.add(row["run_id"])
        for row in self._disk_runs(experiment_id):
            if row["run_id"] not in seen:
                out.append(row)
        out.sort(key=lambda r: float(r.get("started_at") or 0), reverse=True)
        return out

    def _status_dict(self, mp: ManagedProcess) -> Dict[str, Any]:
        running = mp.popen.poll() is None
        if not running and mp.ended_at is None:
            mp.ended_at = time.time()
            mp.returncode = mp.popen.returncode
        return {
            "run_id": mp.run_id,
            "kind": mp.kind,
            "experiment_id": mp.experiment_id,
            "running": running,
            "pid": mp.popen.pid,
            "returncode": mp.popen.returncode if not running else None,
            "log_path": str(mp.log_path),
            "started_at": mp.started_at,
            "ended_at": mp.ended_at,
            "meta": mp.meta,
        }

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
    ) -> ManagedProcess:
        with self._lock:
            existing_id = self._by_kind.get(kind)
            if existing_id and replace:
                old = self._procs.get(existing_id)
                if old and old.popen.poll() is None:
                    self._kill_unlocked(old, sig=signal.SIGTERM)
            elif existing_id:
                old = self._procs.get(existing_id)
                if old and old.popen.poll() is None:
                    raise RuntimeError(f"{kind} already running: {existing_id}")

        run_id = uuid4().hex[:12]
        run_dir = artifacts_dir(experiment_id) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "stdout.log"
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        full_env["PYTHONUNBUFFERED"] = "1"
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        popen = subprocess.Popen(
            argv,
            cwd=str(cwd or tir_agent_root()),
            env=full_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        mp = ManagedProcess(
            run_id=run_id,
            kind=kind,
            experiment_id=experiment_id,
            popen=popen,
            log_path=log_path,
            meta=dict(meta or {}),
        )
        (run_dir / "status.json").write_text(
            __import__("json").dumps(self._status_dict(mp), indent=2),
            encoding="utf-8",
        )
        with self._lock:
            self._procs[run_id] = mp
            self._by_kind[kind] = run_id

        def _watch() -> None:
            code = popen.wait()
            mp.returncode = code
            mp.ended_at = time.time()
            try:
                log_f.close()
            except Exception:
                pass
            BUS.publish(
                experiment_id,
                f"{kind}_done",
                {"run_id": run_id, "returncode": code},
            )
            try:
                (run_dir / "status.json").write_text(
                    __import__("json").dumps(self._status_dict(mp), indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        threading.Thread(target=_watch, daemon=True).start()
        BUS.publish(experiment_id, f"{kind}_started", {"run_id": run_id, "argv": argv[:6]})
        return mp

    def stop(self, kind: str, *, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        with self._lock:
            rid = self._by_kind.get(kind)
            mp = self._procs.get(rid) if rid else None
            if mp:
                self._kill_unlocked(mp, sig=signal.SIGINT, timeout=timeout)
                return self._status_dict(mp)
        for row in self._disk_runs(None):
            if row.get("kind") != kind or not row.get("running"):
                continue
            pid = row.get("pid")
            if isinstance(pid, int):
                self.kill_pid(pid, timeout=timeout)
                row["running"] = False
                return row
        return None

    def stop_run(self, run_id: str, *, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        with self._lock:
            mp = self._procs.get(run_id)
            if not mp:
                return None
            self._kill_unlocked(mp, sig=signal.SIGINT, timeout=timeout)
            return self._status_dict(mp)

    def _kill_unlocked(self, mp: ManagedProcess, *, sig: int = signal.SIGTERM, timeout: float = 15.0) -> None:
        if mp.popen.poll() is not None:
            return
        try:
            os.killpg(mp.popen.pid, sig)
        except ProcessLookupError:
            return
        try:
            mp.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(mp.popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            mp.popen.wait(timeout=5)
        mp.ended_at = time.time()
        mp.returncode = mp.popen.returncode
        BUS.publish(mp.experiment_id, f"{mp.kind}_stopped", {"run_id": mp.run_id})

    def tail_log(self, run_id: str, n: int = 160) -> str:
        path = self.resolve_log_path(run_id)
        if not path or not path.is_file():
            return ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        return "\n".join(lines[-max(1, int(n)) :])

    def resolve_log_path(self, run_id: str) -> Optional[Path]:
        mp = self.get(run_id)
        if mp:
            return mp.log_path
        row = self.disk_run(run_id)
        if not row:
            return None
        p = Path(str(row.get("log_path") or ""))
        return p if str(p) else None

    def disk_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        for row in self._disk_runs(None):
            if row.get("run_id") == run_id:
                return row
        return None

    def _pid_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False

    def _disk_runs(self, experiment_id: Optional[str]) -> List[Dict[str, Any]]:
        from science_infra.control.experiments import list_experiments

        ids = [experiment_id] if experiment_id else list_experiments()
        out: List[Dict[str, Any]] = []
        for eid in ids:
            if not eid:
                continue
            runs_dir = artifacts_dir(eid) / "runs"
            if not runs_dir.is_dir():
                continue
            for rd in runs_dir.iterdir():
                st = rd / "status.json"
                if not st.is_file():
                    continue
                try:
                    row = json.loads(st.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                row.setdefault("run_id", rd.name)
                row.setdefault("experiment_id", eid)
                row.setdefault("log_path", str(rd / "stdout.log"))
                pid = row.get("pid")
                row["running"] = self._pid_alive(pid if isinstance(pid, int) else None)
                out.append(row)
        return out

    def kill_pid(self, pid: int, *, timeout: float = 15.0) -> None:
        if not self._pid_alive(pid):
            return
        try:
            os.killpg(int(pid), signal.SIGINT)
        except ProcessLookupError:
            return
        except PermissionError:
            try:
                os.kill(int(pid), signal.SIGINT)
            except ProcessLookupError:
                return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_alive(pid):
                return
            time.sleep(0.2)
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass


PROCS = ProcessManager()
