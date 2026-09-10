"""FastAPI Control Plane for Science UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    api_key: Optional[str] = None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import asyncio

        BUS.bind_loop(asyncio.get_event_loop())
        ensure_experiment("demo")
        yield

    app = FastAPI(title="Science Control Plane", version="0.1.0", lifespan=lifespan)
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
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.put("/api/experiments/{exp_id}/{section}")
    def put_section(exp_id: str, section: str, body: SectionBody) -> Dict[str, Any]:
        if section not in ("llm", "workflow", "rl", "harness", "meta", "experiment"):
            raise HTTPException(404, f"unknown section {section}")
        try:
            return save_section(exp_id, section, body.data)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/mas/palette")
    def palette() -> Dict[str, Any]:
        return services.mas_palette()

    @app.post("/api/llm/health")
    async def llm_health(body: LlmHealthBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        base = body.base_url
        if not base:
            base = str(load_bundle(experiment_id)["llm"].get("base_url") or "")
        return await services.llm_health(base, body.api_key)

    @app.post("/api/llm/start")
    def llm_start(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        try:
            return services.start_local_llm(experiment_id)
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
        except Exception as e:
            raise HTTPException(400, str(e)) from e

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
