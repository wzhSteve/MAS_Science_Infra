"""In-process SSE event bus for Control UI."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional


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
        for q in waiters:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

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
