"""Archive snapshots for TIR resume (compatible with resume_messages / ARPO).

Writes only under TIR_ARCHIVE_DIR (default ``.tir_archives``).
Legacy ``.resume_cache`` is still *read* as fallback, never written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .contracts import ArchiveRef, BranchPoint, ExecutionEvent, EventKind, Snapshot

DEFAULT_ARCHIVE_ROOT = os.getenv(
    "TIR_ARCHIVE_DIR",
    os.path.join(os.path.dirname(__file__), "..", ".tir_archives"),
)

_GLOBAL: Dict[str, "Archive"] = {}
# rollout_id -> (archive_id, snapshot_id, root_dir)
_ROLLOUT_INDEX: Dict[str, tuple[str, str, Optional[str]]] = {}


class Archive:
    def __init__(self, archive_id: Optional[str] = None, root_dir: Optional[str] = None) -> None:
        self.archive_id = archive_id or uuid4().hex
        self.root_dir = root_dir if root_dir is not None else DEFAULT_ARCHIVE_ROOT
        self._events: List[ExecutionEvent] = []
        self._snaps: Dict[str, Snapshot] = {}
        self._snap_order: List[str] = []
        self._rollout_snapshots: Dict[str, str] = {}
        if self.root_dir:
            Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    def append(self, event: ExecutionEvent) -> ExecutionEvent:
        self._events.append(event)
        if self.root_dir:
            path = Path(self.root_dir) / self.archive_id / "events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        return event

    def snapshot(
        self,
        *,
        messages: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
    ) -> Snapshot:
        if event_id is None:
            ev = self.append(
                ExecutionEvent(
                    kind=EventKind.SNAPSHOT,
                    payload={"reason": (meta or {}).get("reason", "branch")},
                )
            )
            event_id = ev.event_id
        snap = Snapshot(
            archive_id=self.archive_id,
            event_id=event_id,
            messages=list(messages),
            token_ids=list(token_ids) if token_ids is not None else None,
            meta=dict(meta or {}),
        )
        self._snaps[snap.snapshot_id] = snap
        self._snap_order.append(snap.snapshot_id)
        if rollout_id:
            self._rollout_snapshots[rollout_id] = snap.snapshot_id
            _ROLLOUT_INDEX[rollout_id] = (self.archive_id, snap.snapshot_id, self.root_dir)
        if self.root_dir:
            base = Path(self.root_dir) / self.archive_id
            (base / "snapshots").mkdir(parents=True, exist_ok=True)
            (base / "snapshots" / f"{snap.snapshot_id}.json").write_text(
                snap.model_dump_json(indent=2), encoding="utf-8"
            )
            if rollout_id:
                idx_path = base / "rollout_index.json"
                idx = {}
                if idx_path.is_file():
                    try:
                        idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    except Exception:
                        idx = {}
                idx[rollout_id] = snap.snapshot_id
                idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        return snap

    def restore(self, snapshot_id: str) -> Snapshot:
        if snapshot_id in self._snaps:
            return self._snaps[snapshot_id]
        if self.root_dir:
            path = Path(self.root_dir) / self.archive_id / "snapshots" / f"{snapshot_id}.json"
            if path.is_file():
                snap = Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
                self._snaps[snap.snapshot_id] = snap
                return snap
        raise KeyError(f"snapshot_id not found: {snapshot_id}")

    def ref(self) -> ArchiveRef:
        return ArchiveRef(
            archive_id=self.archive_id,
            root_dir=self.root_dir,
            n_events=len(self._events),
            n_snapshots=len(self._snap_order),
            rollout_snapshots=dict(self._rollout_snapshots),
        )

    def to_branch_point(
        self,
        snapshot_id: str,
        *,
        parent_rollout_id: Optional[str] = None,
        reason: str = "post_tool",
        beam_size: int = 2,
    ) -> BranchPoint:
        snap = self.restore(snapshot_id)
        return BranchPoint(
            archive_id=self.archive_id,
            snapshot_id=snapshot_id,
            beam_size=beam_size,
            reason=reason,
            parent_rollout_id=parent_rollout_id,
            meta=dict(snap.meta),
        )

    @classmethod
    def load(cls, archive_id: str, root_dir: Optional[str] = None) -> "Archive":
        root = root_dir or DEFAULT_ARCHIVE_ROOT
        arch = cls(archive_id=archive_id, root_dir=root)
        base = Path(root) / archive_id
        events_path = base / "events.jsonl"
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    arch._events.append(ExecutionEvent.model_validate_json(line))
        snap_dir = base / "snapshots"
        if snap_dir.is_dir():
            for p in sorted(snap_dir.glob("*.json")):
                snap = Snapshot.model_validate_json(p.read_text(encoding="utf-8"))
                arch._snaps[snap.snapshot_id] = snap
                arch._snap_order.append(snap.snapshot_id)
        idx_path = base / "rollout_index.json"
        if idx_path.is_file():
            try:
                arch._rollout_snapshots = {
                    str(k): str(v) for k, v in json.loads(idx_path.read_text(encoding="utf-8")).items()
                }
                for rid, sid in arch._rollout_snapshots.items():
                    _ROLLOUT_INDEX[rid] = (archive_id, sid, root)
            except Exception:
                pass
        return arch


def register_archive(archive: Archive) -> Archive:
    _GLOBAL[archive.archive_id] = archive
    return archive


def get_archive(archive_id: str, root_dir: Optional[str] = None) -> Archive:
    if archive_id in _GLOBAL:
        return _GLOBAL[archive_id]
    arch = Archive.load(archive_id, root_dir=root_dir)
    return register_archive(arch)


_DEFAULT_ARCHIVE: Optional[Archive] = None


def get_default_archive(root_dir: Optional[str] = None) -> Archive:
    global _DEFAULT_ARCHIVE
    if _DEFAULT_ARCHIVE is None:
        _DEFAULT_ARCHIVE = register_archive(Archive(root_dir=root_dir))
    return _DEFAULT_ARCHIVE


def dump_resume_with_archive(
    rollout_id: str,
    payload: Dict[str, Any],
    *,
    archive: Optional[Archive] = None,
) -> Dict[str, str]:
    """Write Archive snapshot only (legacy .resume_cache is read-compat, not written)."""
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    arch = archive or get_default_archive()
    meta = {
        "h_root": payload.get("h_root", 0.0),
        "h_tool": payload.get("h_tool", 0.0),
        "consecutive_high": payload.get("consecutive_high", 0),
        "reason": "post_tool",
        "rollout_id": rollout_id,
    }
    register_archive(arch)
    snap = arch.snapshot(messages=messages, meta=meta, rollout_id=rollout_id)
    return {
        "archive_id": arch.archive_id,
        "snapshot_id": snap.snapshot_id,
    }


def load_resume_messages(rollout_id: str) -> Optional[Dict[str, Any]]:
    """Load resume payload: prefer Archive index, fallback to legacy resume_cache."""
    if rollout_id in _ROLLOUT_INDEX:
        archive_id, snapshot_id = _ROLLOUT_INDEX[rollout_id][0], _ROLLOUT_INDEX[rollout_id][1]
        root_dir = _ROLLOUT_INDEX[rollout_id][2] if len(_ROLLOUT_INDEX[rollout_id]) > 2 else None
        try:
            arch = get_archive(archive_id, root_dir=root_dir)
            snap = arch.restore(snapshot_id)
            return {
                "messages": snap.messages,
                "h_root": snap.meta.get("h_root", 0.0),
                "h_tool": snap.meta.get("h_tool", 0.0),
                "consecutive_high": snap.meta.get("consecutive_high", 0),
                "archive_id": archive_id,
                "snapshot_id": snapshot_id,
            }
        except Exception:
            pass
    # Disk: scan archives for rollout_index
    root = Path(DEFAULT_ARCHIVE_ROOT)
    if root.is_dir():
        for arch_dir in root.iterdir():
            if not arch_dir.is_dir():
                continue
            idx_path = arch_dir / "rollout_index.json"
            if not idx_path.is_file():
                continue
            try:
                idx = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rollout_id in idx:
                arch = get_archive(arch_dir.name, root_dir=str(root))
                snap = arch.restore(str(idx[rollout_id]))
                return {
                    "messages": snap.messages,
                    "h_root": snap.meta.get("h_root", 0.0),
                    "h_tool": snap.meta.get("h_tool", 0.0),
                    "consecutive_high": snap.meta.get("consecutive_high", 0),
                    "archive_id": arch.archive_id,
                    "snapshot_id": snap.snapshot_id,
                }
    return _load_legacy_resume_cache(rollout_id)


def _load_legacy_resume_cache(rollout_id: str) -> Optional[Dict[str, Any]]:
    resume_dir = os.getenv(
        "TIR_RESUME_DIR",
        os.path.join(os.path.dirname(__file__), "..", ".resume_cache"),
    )
    dest = Path(resume_dir) / f"{rollout_id}.json"
    if not dest.is_file():
        return None
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return None


def branch_point_to_resume_task_fields(bp: BranchPoint) -> Dict[str, Any]:
    """Convert BranchPoint → task fields understood by LitTirAgent / TirAgent."""
    arch = get_archive(bp.archive_id)
    snap = arch.restore(bp.snapshot_id)
    return {
        "resume_messages": snap.messages,
        "resume_parent_id": bp.parent_rollout_id or "",
        "resume_from": bp.model_dump(mode="json"),
    }
