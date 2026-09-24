"""FastAPI Control Plane for Science UI."""

from __future__ import annotations

import json
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
    create_experiment,
    ensure_experiment,
    list_experiments,
    load_bundle,
    recommend_rl,
    save_section,
)
from science_infra.control.paths import webui_dist
from science_infra.control.process_manager import PROCS
from science_infra.control import services
from science_infra.control.model_resource_api import router as model_resource_router
from science_infra.control.dataset_resources import router as dataset_resource_router
from science_infra.control.training_logs import router as training_logs_router, training_run
from science_infra.control.model_resources import ResourceError
from science_infra.control.readiness import model_readiness
from science_infra.control.training import training_preflight
from science_infra.control.training import (
    TrainingError,
    launch_training,
    stop_training_run,
)


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


class TrainRunBody(BaseModel):
    request_id: str
    preflight_revision: str
    stop_local_llm: bool = True


class LlmHealthBody(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    kind: Optional[str] = None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import asyncio

        BUS.bind_loop(asyncio.get_event_loop())
        ensure_experiment("demo")
        yield

    app = FastAPI(title="Science Control Plane", version="0.1.0", lifespan=lifespan)
    app.include_router(model_resource_router)
    app.include_router(dataset_resource_router)
    app.include_router(training_logs_router)

    @app.exception_handler(ResourceError)
    async def resource_error(_request: Request, error: ResourceError) -> Response:
        return Response(
            content=json.dumps({"detail": str(error)}, ensure_ascii=False),
            status_code=error.status,
            media_type="application/json",
        )

    @app.exception_handler(TrainingError)
    async def training_error(_request: Request, error: TrainingError) -> Response:
        return Response(
            content=json.dumps(
                {
                    "detail": str(error),
                    "code": error.code,
                    **error.data,
                },
                ensure_ascii=False,
            ),
            status_code=error.status,
            media_type="application/json",
        )
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
        # SPA routes need built dashboard assets; otherwise Store returns bare FastAPI 404 JSON.
        if (
            request.method in ("GET", "HEAD")
            and services.is_agl_spa_path(origin_path)
            and not services.agl_dashboard_built()
        ):
            raise HTTPException(502, services.agl_spa_missing_message())
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
        if (
            r.status_code == 404
            and request.method in ("GET", "HEAD")
            and services.is_agl_spa_path(origin_path)
            and not services.agl_dashboard_built()
        ):
            raise HTTPException(502, services.agl_spa_missing_message())
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
    def create_experiment_endpoint(body: CreateExperimentBody) -> Dict[str, Any]:
        try:
            create_experiment(body.id, seed=body.seed, name=body.name)
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

    @app.post("/api/mas/sampling/preview")
    def preview_sampling(body: SectionBody) -> Dict[str, Any]:
        try:
            from workflow.site_policy import sampling_preview

            return sampling_preview(body.data)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/mas/rollout-trees")
    def rollout_trees(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        """Read-only scan of mas/.local_expansion/*.json → list of RolloutTree payloads."""
        import json
        from science_infra.control.paths import tir_agent_root

        exp_dir = tir_agent_root() / ".local_expansion"
        # Daemon-persisted trees (tree_*.json) carry REAL store rollout ids on
        # child nodes; the same query also exists inside runner expansions
        # (ro-*.json) with synthetic "{parent}:0" ids. Serve daemon trees and
        # skip the duplicated runner expansion for the same tree_id.
        daemon_trees: Dict[str, Dict[str, Any]] = {}
        expansion_trees: List[Dict[str, Any]] = []
        if exp_dir.is_dir():
            for f in sorted(exp_dir.glob("tree_*.json")):
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload.get("nodes"), list) and payload.get("tree_id"):
                    daemon_trees[str(payload["tree_id"])] = {"file": f.name, "tree": payload}
            for f in sorted(exp_dir.glob("*.json")):
                if f.name.startswith("tree_"):
                    continue
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                tree = payload.get("tree")
                if isinstance(tree, dict):
                    # Skip degenerate trees (no nodes / blank root id) written
                    # by runner expansions of samples that produced no plans.
                    nodes = tree.get("nodes")
                    if not isinstance(nodes, list) or not nodes or not any(
                        str(n.get("node_id") or "") for n in nodes if isinstance(n, dict)
                    ):
                        continue
                    expansion_trees.append({"file": f.name, "tree": tree})
        seen_ids = set(daemon_trees)
        trees = list(daemon_trees.values()) + [
            t for t in expansion_trees if str((t["tree"] or {}).get("tree_id") or "") not in seen_ids
        ]
        return {"experiment_id": experiment_id, "n": len(trees), "trees": trees}

    @app.post("/api/llm/health")
    async def llm_health(body: LlmHealthBody, experiment_id: str = Query("demo")) -> Dict[str, Any]:
        return await services.llm_health(
            body.base_url,
            body.api_key,
            experiment_id=experiment_id,
            model=body.model,
            kind=body.kind,
        )

    @app.get("/api/mas/readiness")
    def mas_readiness(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        return model_readiness(experiment_id)

    @app.get("/api/rl/preflight")
    def rl_preflight(experiment_id: str = Query("demo")) -> Dict[str, Any]:
        return training_preflight(experiment_id)

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

    @app.post("/api/rl/runs")
    def create_train_run(
        body: TrainRunBody, experiment_id: str = Query("demo")
    ) -> Dict[str, Any]:
        return launch_training(
            experiment_id,
            request_id=body.request_id,
            preflight_revision=body.preflight_revision,
            stop_local_llm=body.stop_local_llm,
        )

    @app.get("/api/rl/runs")
    def list_train_runs(
        experiment_id: Optional[str] = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
    ) -> Dict[str, Any]:
        rows = [row for row in PROCS.list_runs(experiment_id) if row.get("kind") == "train"]
        return {"runs": rows[offset:offset + limit], "total": len(rows)}

    @app.get("/api/rl/activity")
    def train_activity(experiment_id: str = Query(...)) -> Dict[str, Any]:
        process = PROCS.active("train")
        return {
            "run": (
                PROCS.status(process.run_id, experiment_id)
                if process and process.experiment_id == experiment_id else None
            )
        }

    @app.get("/api/rl/runs/{run_id}")
    def get_train_run(
        run_id: str, experiment_id: str = Query(...), tail: int = Query(160, ge=0, le=160)
    ) -> Dict[str, Any]:
        row = training_run(experiment_id, run_id)
        return {**row, "log_tail": PROCS.tail_log(run_id, tail) if tail else ""}

    @app.post("/api/rl/runs/{run_id}/stop")
    def stop_train_run(
        run_id: str, experiment_id: str = Query(...)
    ) -> Dict[str, Any]:
        return stop_training_run(experiment_id, run_id)

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
            # P3: tail active train/collect subprocess stdout for marked
            # rollout-tree frames and republish them as SSE `rollout_tree` events.
            tail_state = {"offset": 0, "run_id": ""}

            def _drain_tree_frames() -> None:
                mp = PROCS.active("train") or PROCS.active("collect")
                if mp is None or mp.experiment_id != experiment_id:
                    return
                if str(mp.run_id) != tail_state["run_id"]:
                    tail_state["run_id"] = str(mp.run_id)
                    tail_state["offset"] = 0
                try:
                    with open(mp.log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(tail_state["offset"])
                        chunk = f.read()
                        tail_state["offset"] = f.tell()
                except Exception:
                    return
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line.startswith('{"__rollout_tree_event__"'):
                        continue
                    try:
                        frame = json.loads(line)
                        payload = frame.get("__rollout_tree_event__") or {}
                    except Exception:
                        continue
                    BUS.publish(experiment_id, "rollout_tree", payload)

            async for payload in BUS.subscribe(experiment_id):
                _drain_tree_frames()
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
