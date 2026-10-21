"""In-process SSE event bus for Control UI."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional

# Ray prefixes worker stdout, for example "\x1b[36m(TaskRunner pid=304860)\x1b[0m {json}".
_RAY_PREFIX = re.compile(r"^(?:\x1b\[[0-9;]*m)*\([^)\n]*\)(?:\x1b\[[0-9;]*m)*\s*")
_TREE_MARK = '{"__rollout_tree_event__":'


def complete_log_lines(pending: str, chunk: str) -> tuple[str, List[str]]:
    """Return finished log lines and keep a trailing partial line for the next read."""
    text = f"{pending}{chunk}"
    if not text:
        return "", []
    if text.endswith(("\n", "\r")):
        return "", text.splitlines()
    head, separator, tail = text.rpartition("\n")
    if not separator:
        return text, []
    return tail, head.splitlines()


def rollout_tree_event_from_line(line: str) -> Optional[Dict[str, Any]]:
    """Return the tree payload from a stdout line, including a Ray-prefixed one."""
    text = _RAY_PREFIX.sub("", str(line or "").strip())
    idx = text.find(_TREE_MARK)
    if idx < 0:
        return None
    try:
        frame = json.loads(text[idx:])
    except Exception:
        return None
    payload = frame.get("__rollout_tree_event__") if isinstance(frame, dict) else None
    return payload if isinstance(payload, dict) else None


class EventBus:
    def __init__(self, maxlen: int = 200) -> None:
        self._lock = threading.Lock()
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._waiters: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, experiment_id: str, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "type": event_type,
            "experiment_id": experiment_id,
            "ts": time.time(),
            "data": data or {},
        }
        with self._lock:
            self._buffers[experiment_id].append(payload)
            waiters = list(self._waiters.get(experiment_id, []))
            loop = self._loop

        def _put(q: asyncio.Queue) -> None:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        # Same-thread publishers (the SSE drain) must deliver on the loop that
        # is actually running. Watcher threads schedule onto the bound loop.
        target = running or loop
        for q in waiters:
            if target is not None and running is not target:
                try:
                    target.call_soon_threadsafe(_put, q)
                    continue
                except RuntimeError:
                    pass
            _put(q)

    def history(self, experiment_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            buf = list(self._buffers.get(experiment_id, []))
        return buf[-limit:]

    async def subscribe(self, experiment_id: str) -> AsyncIterator[Dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._waiters[experiment_id].append(q)
        try:
            for item in self.history(experiment_id):
                yield item
            while True:
                item = await q.get()
                yield item
        finally:
            with self._lock:
                lst = self._waiters.get(experiment_id, [])
                if q in lst:
                    lst.remove(q)


def format_sse(payload: Dict[str, Any]) -> str:
    # Always use default "message" so EventSource.onmessage receives all events.
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: message\ndata: {data}\n\n"


BUS = EventBus()
