"""HTTP routes for the floating assistant (moved verbatim from control.assistant)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from ai_assistant.context import ApplyBody, ChatBody
from ai_assistant.engine import _redact, _sse, iter_chat_events
from ai_assistant.tools.repo_tools import ensure_mas_path


def _user_projects():
    """Imported lazily: science_infra.control.__init__ eagerly imports app.py,
    which imports ai_assistant back — importing at module top-level would cycle."""
    from science_infra.control import user_projects

    return user_projects


router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.get("/projects")
def api_list_projects(experiment_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return {"projects": _user_projects().list_user_projects(experiment_id)}


@router.get("/projects/{project_id}")
def api_get_project(project_id: str) -> Dict[str, Any]:
    try:
        return _user_projects().get_user_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/projects/{project_id}/files")
def api_list_files(project_id: str) -> Dict[str, Any]:
    try:
        return {"files": _user_projects().list_project_files(project_id)}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/projects/{project_id}/files/{rel_path:path}")
def api_read_file(project_id: str, rel_path: str) -> Dict[str, Any]:
    try:
        return {"path": rel_path, "content": _user_projects().read_project_file(project_id, rel_path)}
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/projects")
async def api_upload_project(
    title: str = Form(...),
    mode: str = Form("io_module"),
    experiment_id: str = Form(""),
    project_id: str = Form(""),
    file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    try:
        return _user_projects().ingest_upload(
            title=title,
            mode=mode,
            experiment_id=experiment_id,
            project_id=project_id or None,
            upload=file,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/projects/{project_id}/apply")
def api_apply(project_id: str, body: ApplyBody) -> Dict[str, Any]:
    try:
        return _user_projects().apply_project(project_id, body.current_workflow, replace=body.replace)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/projects/{project_id}/scaffold")
def api_scaffold(project_id: str, mode: str = Query("io_module")) -> Dict[str, Any]:
    ensure_mas_path()
    from workflow.user_gateway.scaffold import scaffold_project

    try:
        return scaffold_project(project_id, mode)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/chat")
def api_chat(body: ChatBody) -> StreamingResponse:
    def gen() -> Iterable[str]:
        try:
            for event, data in iter_chat_events(body):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": _redact(str(exc), 400)})
            yield _sse("done", {"project_id": body.project_id, "experiment_id": body.experiment_id, "wrote_files": False})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
