"""Bounded, resumable reads of registered training output."""

from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from science_infra.control.process_manager import ACTIVE_STATES, PROCS
from science_infra.control.experiments import require_experiment

router = APIRouter(prefix="/api/rl/runs")
CHUNK_SIZE = 64 * 1024


def training_run(experiment_id: str, run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment_id) or not re.fullmatch(r"[a-f0-9]{12}", run_id):
        raise HTTPException(404, "训练运行不存在")
    try:
        require_experiment(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "实验不存在") from error
    row = PROCS.status(run_id, experiment_id)
    if not row or row.get("kind") != "train" or row.get("experiment_id") != experiment_id:
        raise HTTPException(404, "当前实验中没有此训练运行")
    return row


def read_log(experiment_id: str, run_id: str, offset: Optional[int], limit: int, generation: Optional[str]) -> dict:
    row = training_run(experiment_id, run_id)
    path = PROCS.run_dir(experiment_id, run_id) / "stdout.log"
    terminal = row.get("state") not in ACTIVE_STATES
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return {"offset": 0, "next_offset": 0, "text": "", "size": 0,
                "generation": "", "reset": bool(offset), "terminal": terminal, "state": row}
    with stream:
        stat = os.fstat(stream.fileno())
        identity = f"{stat.st_dev}:{stat.st_ino}:{stat.st_ctime_ns if not stat.st_ino else 0}"
        reset = offset is not None and (offset > stat.st_size or bool(generation and generation != identity))
        start = max(0, stat.st_size - limit) if offset is None or reset else offset
        stream.seek(start)
        raw = stream.read(limit)
        if offset is None or reset:
            # A tail window can begin inside a UTF-8 character.
            while raw and raw[0] & 0xC0 == 0x80:
                start += 1
                raw = raw[1:]
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(raw, final=terminal and start + len(raw) >= stat.st_size)
        pending, _ = decoder.getstate()
        return {"offset": start, "next_offset": start + len(raw) - len(pending),
                "text": text, "size": stat.st_size, "generation": identity,
                "reset": reset, "terminal": terminal, "state": row}


@router.get("/{run_id}/log")
def log_chunk(run_id: str, experiment_id: str, offset: Optional[int] = Query(None, ge=0),
              limit: int = Query(CHUNK_SIZE, ge=1, le=CHUNK_SIZE), generation: Optional[str] = None):
    return read_log(experiment_id, run_id, offset, limit, generation)


@router.get("/{run_id}/log/download")
def download_log(run_id: str, experiment_id: str):
    training_run(experiment_id, run_id)
    path = PROCS.run_dir(experiment_id, run_id) / "stdout.log"
    if not path.is_file():
        raise HTTPException(404, "日志文件尚未生成")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{run_id}.log")


@router.get("/{run_id}/events")
async def events(request: Request, run_id: str, experiment_id: str,
                 offset: Optional[int] = Query(None, ge=0), generation: Optional[str] = None):
    training_run(experiment_id, run_id)

    async def stream():
        cursor, identity = offset, generation
        previous_state = None
        while not await request.is_disconnected():
            block = await asyncio.to_thread(read_log, experiment_id, run_id, cursor, CHUNK_SIZE, identity)
            state = block.pop("state")
            if state != previous_state:
                yield f"event: state\ndata: {json.dumps(state, ensure_ascii=False)}\n\n"
                previous_state = state
            yield f"event: stdout\ndata: {json.dumps(block, ensure_ascii=False)}\n\n"
            cursor, identity = block["next_offset"], block["generation"]
            if block["terminal"] and cursor >= block["size"]:
                yield "event: complete\ndata: {}\n\n"
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.05 if cursor < block["size"] else 1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_SECRET = re.compile(r"api[_-]?key|password|secret|credential|access[_-]?token|authorization|^token$", re.I)


def _redact(value):
    if isinstance(value, dict):
        return {key: "[redacted]" if _SECRET.search(str(key)) else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", value)
        return re.sub(r"(?i)((?:api[_-]?key|password|secret|access[_-]?token)=)[^&\s]+", r"\1[redacted]", value)
    return value


@router.get("/{run_id}/snapshot")
def snapshot(run_id: str, experiment_id: str):
    training_run(experiment_id, run_id)
    directory = PROCS.run_dir(experiment_id, run_id)
    result = {}
    for name in ("effective-rl.yaml", "effective-workflow.yaml", "launch.json"):
        path: Path = directory / name
        if path.is_file():
            result[name] = _redact(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not result:
        raise HTTPException(404, "此运行没有配置快照（可能创建于旧版本或在准备阶段失败）")
    return result
