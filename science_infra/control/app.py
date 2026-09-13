"""FastAPI Control Plane for Science UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from science_infra.control.events import BUS, format_sse
from science_infra.control.experiments import (
    HARNESS_PLUGINS,
    STUB_HARNESS,
    VALID_ALGOS,
    ensure_experiment,
    list_experiments,
    load_bundle,
    recommend_rl,
    save_section,
)
from science_infra.control.paths import webui_dist
from science_infra.control.process_manager import PROCS
from science_infra.control import services
from science_infra.control.readiness import model_readiness
from science_infra.control import rollout_runs
from science_infra.control.model_resource_api import router as model_resource_router
from science_infra.control.model_resources import ResourceError


class CreateExperimentBody(BaseModel):
    id: str
    seed: int = 42
    name: str = ""


class SectionBody(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


class CollectBody(BaseModel):
    mock: bool = True
    n: int = 1
    algo: str = "grpo"
    tasks: Optional[List[Dict[str, Any]]] = None
    # live-api-data style: sample from parquet (relative to tir_agent/ or absolute)
    parquet: Optional[str] = None
    data_n: Optional[int] = None
    source: str = "gsm8k"
    sequential: bool = False


class SampleDataBody(BaseModel):
    parquet: str = "data/val.parquet"
    data_n: int = 5
    source: str = "gsm8k"


class DiagnoseBody(BaseModel):
    metrics_path: Optional[str] = None


class TrainBody(BaseModel):
    stop_llm: bool = True
    confirm_gpu: bool = False


class LlmHealthBody(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(default=None, repr=False)
    model: Optional[str] = None
    kind: Optional[str] = None


class RolloutTaskBody(BaseModel):
    id: Optional[str] = None
    question: str = Field(min_length=1)

    model_config = {"extra": "forbid"}

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入希望 Workflow 完成的任务。")
        return value


class RolloutRunBody(BaseModel):
    workflow: Dict[str, Any]
    task: RolloutTaskBody
    execution: Literal["mock", "live"]

    model_config = {"extra": "forbid"}


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import asyncio

        BUS.bind_loop(asyncio.get_event_loop())
        ensure_experiment("demo")
        yield

    app = FastAPI(title="Science Control Plane", version="0.1.0", lifespan=lifespan)
    # Must precede the legacy /{experiment_id}/{section} PUT route.
    app.include_router(model_resource_router)

    @app.exception_handler(RequestValidationError)
    async def safe_resource_validation(request: Request, error: RequestValidationError):
        if request.url.path.startswith("/api/model-resources") or "/model-bindings" in request.url.path:
            return JSONResponse(status_code=400, content={"detail": "请求参数格式无效，请检查字段和修订号。"})
        return await request_validation_exception_handler(request, error)

    @app.exception_handler(ResourceError)
    async def resource_error(_request: Request, error: ResourceError):
        return JSONResponse(status_code=error.status, content={"detail": str(error)})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "service": "science-control"}

    @app.get("/api/meta")
    def meta() -> Dict[str, Any]:
        return {
            "algos": list(VALID_ALGOS),
            "harness_plugins": list(HARNESS_PLUGINS),
            "stub_harness": list(STUB_HARNESS),
            "profiles": ["fast", "a800", "a800_2gpu"],
            "llm_kinds": ["api", "local", "rl_endpoint"],
        }

    @app.get("/api/gpus")
    def gpus() -> Dict[str, Any]:
        info = services.list_gpus()
        rec = recommend_rl(int(info.get("count") or 0))
        info["recommend"] = rec
        return info

    @app.get("/api/agl/health")
    def agl_health() -> Dict[str, Any]:
        return services.agl_health()

    async def _proxy_agl(request: Request, origin_path: str) -> Response:
        import httpx

        origin = services.AGL_ORIGIN
        dest = f"{origin}/{origin_path.lstrip('/')}" if origin_path else f"{origin}/"
        if request.url.query:
            dest = f"{dest}?{request.url.query}"
        hop = {"host", "content-length", "connection", "transfer-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in hop}
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await client.request(request.method, dest, headers=headers, content=body or None)
        except Exception as e:
            raise HTTPException(502, f"AGL dashboard unavailable ({origin}): {e}") from e
        ct = r.headers.get("content-type") or ""
        payload = services.rewrite_agl_payload(r.content, ct)
        out_headers = {
            k: v
            for k, v in r.headers.items()
            if k.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
        }
        return Response(content=payload, status_code=r.status_code, headers=out_headers, media_type=ct.split(";")[0] or None)

    @app.api_route("/agl", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    @app.api_route("/agl/{full_path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def agl_proxy(request: Request, full_path: str = "") -> Response:
        return await _proxy_agl(request, full_path)

    @app.api_route("/v1/agl", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    @app.api_route("/v1/agl/{full_path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def agl_api_alias(request: Request, full_path: str = "") -> Response:
        dest_path = f"v1/agl/{full_path}" if full_path else "v1/agl"
        return await _proxy_agl(request, dest_path)

    @app.get("/api/experiments")
    def experiments() -> Dict[str, Any]:
        return {"experiments": list_experiments()}

    @app.post("/api/experiments")
    def create_experiment(body: CreateExperimentBody) -> Dict[str, Any]:
        try:
            ensure_experiment(body.id, seed=body.seed, name=body.name)
            return load_bundle(body.id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str) -> Dict[str, Any]:
        try:
            return load_bundle(exp_id)
        except Exception as e:
            raise HTTPException(404, str(e)) from e

    @app.put("/api/experiments/{exp_id}")
    def put_experiment(exp_id: str, body: SectionBody) -> Dict[str, Any]:
        try:
            return save_section(exp_id, "meta", body.data)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/experiments/{exp_id}/workflow")
    def get_workflow(exp_id: str) -> Dict[str, Any]:
        b = load_bundle(exp_id)
        return {"workflow": b["workflow"], "executable": b["executable"]}

    @app.put("/api/experiments/{exp_id}/workflow")
    def put_workflow(exp_id: str, body: SectionBody) -> Dict[str, Any]:
        try:
            return save_section(exp_id, "workflow", body.data)
        except ResourceError:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.put("/api/experiments/{exp_id}/{section}")
    def put_section(exp_id: str, section: str, body: SectionBody) -> Dict[str, Any]:
        if section not in ("llm", "workflow", "rl", "harness", "meta", "experiment"):
            raise HTTPException(404, f"unknown section {section}")
        try:
            return save_section(exp_id, section, body.data)
        except ResourceError:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/mas/palette")
    def palette() -> Dict[str, Any]:
        return services.mas_palette()

    @app.post("/api/llm/health")
    async def llm_health(body: LlmHealthBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return await services.llm_health(
                body.base_url, body.api_key, experiment_id=experiment_id, model=body.model, kind=body.kind,
            )
        except ResourceError:
            raise
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/mas/readiness")
    def mas_readiness(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return model_readiness(experiment_id)
        except ResourceError:
            raise
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/llm/start")
    def llm_start(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return services.start_local_llm(experiment_id)
        except ResourceError:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/llm/stop")
    def llm_stop(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        return services.stop_local_llm(experiment_id)

    @app.post("/api/mas/sample-data")
    def mas_sample_data(body: SampleDataBody) -> Dict[str, Any]:
        try:
            tasks = services.sample_parquet_tasks(
                body.parquet, n=body.data_n, source=body.source
            )
            return {
                "n": len(tasks),
                "tasks": tasks,
                "parquet": body.parquet,
                "source": body.source,
            }
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/mas/collect")
    def mas_collect(body: CollectBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return services.run_collect(
                experiment_id,
                mock=body.mock,
                n=body.n,
                tasks=body.tasks,
                algo=body.algo,
                parquet=body.parquet,
                data_n=body.data_n,
                source=body.source,
                sequential=body.sequential,
            )
        except ResourceError:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/mas/rollout-runs", response_model=rollout_runs.RolloutRunSummary)
    def mas_rollout(body: RolloutRunBody, experiment_id: str = Query("demo")) -> rollout_runs.RolloutRunSummary:
        try:
            return rollout_runs.run_rollout(
                experiment_id, body.workflow, body.task.model_dump(exclude_none=True), body.execution,
            )
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs")
    def mas_rollout_list(experiment_id: str = Query("demo"), cursor: Optional[str] = None,
                         limit: int = Query(15, ge=1, le=50)) -> Dict[str, Any]:
        try:
            return rollout_runs.list_rollouts(experiment_id, cursor, limit)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}", response_model=rollout_runs.RolloutRunSummary)
    def mas_rollout_summary(run_id: str, experiment_id: str = Query("demo")) -> rollout_runs.RolloutRunSummary:
        try:
            return rollout_runs.get_rollout(experiment_id, run_id).summary
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}/trajectory")
    def mas_rollout_trajectory(
        run_id: str, experiment_id: str = Query("demo"), view: Literal["full", "preview"] = "full",
        offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200),
    ) -> Dict[str, Any]:
        try:
            return rollout_runs.get_trajectory(experiment_id, run_id, preview=view == "preview", offset=offset, limit=limit)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}/context")
    def mas_rollout_context(run_id: str, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return rollout_runs.get_rollout_context(experiment_id, run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}/trajectory/events/{event_id}")
    def mas_rollout_event(run_id: str, event_id: str, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return rollout_runs.get_trajectory_event(experiment_id, run_id, event_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}/trajectory/messages")
    def mas_rollout_messages(run_id: str, experiment_id: str = Query("demo"),
                             offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50)) -> Dict[str, Any]:
        try:
            return rollout_runs.get_trajectory_messages(experiment_id, run_id, offset, limit)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.get("/api/mas/rollout-runs/{run_id}/export")
    def mas_rollout_export(run_id: str, experiment_id: str = Query("demo")) -> Response:
        import json

        try:
            payload = rollout_runs.export_rollout(experiment_id, run_id)
            return Response(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="rollout-{run_id}.json"'},
            )
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except rollout_runs.RolloutStorageError as error:
            raise HTTPException(500, str(error)) from error

    @app.post("/api/harness/diagnose")
    def harness_diagnose(body: DiagnoseBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return services.run_diagnose(experiment_id, metrics_path=body.metrics_path)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/rl/train")
    def rl_train(body: TrainBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return services.start_train(
                experiment_id,
                stop_llm=body.stop_llm,
                confirm_gpu=body.confirm_gpu,
            )
        except ResourceError:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/rl/stop")
    def rl_stop(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        return services.stop_train(experiment_id)

    @app.get("/api/runs")
    def list_runs(experiment_id: Optional[str] = None) -> Dict[str, Any]:
        return {"runs": PROCS.list_runs(experiment_id)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, tail: int = 160) -> Dict[str, Any]:
        mp = PROCS.get(run_id)
        st = PROCS.list_runs()
        row = next((r for r in st if r["run_id"] == run_id), None)
        if not mp and not row:
            row = PROCS.disk_run(run_id)
        if not mp and not row:
            raise HTTPException(404, "run not found")
        return {**(row or {}), "log_tail": PROCS.tail_log(run_id, tail)}

    @app.get("/api/monitor/{exp_id}")
    def monitor(exp_id: str) -> Dict[str, Any]:
        try:
            return services.monitor_model(exp_id)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/events")
    async def events(experiment_id: str = Query("demo")) -> StreamingResponse:
        async def gen():
            async for payload in BUS.subscribe(experiment_id):
                yield format_sse(payload)

        return StreamingResponse(gen(), media_type="text/event-stream")

    dist = webui_dist()
    if dist.is_dir() and (dist / "index.html").is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            if (
                full_path.startswith("api/")
                or full_path.startswith("agl")
                or full_path.startswith("v1/")
                or full_path in ("docs", "openapi.json", "redoc")
            ):
                raise HTTPException(404)
            candidate = dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/")
        def index_fallback() -> Dict[str, Any]:
            return {
                "message": "Science Control API. Build webui (cd webui && npm run build) or open /docs",
                "docs": "/docs",
                "health": "/api/health",
            }

    return app


app = create_app()
