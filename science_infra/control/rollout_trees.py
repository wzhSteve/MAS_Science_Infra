"""Read-only, bounded access to persisted trees of a registered training run."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from .process_manager import PROCS
from .readiness import ensure_workflow_path
from .training_logs import _redact, training_run

router = APIRouter(prefix="/api/rl/runs")
logger = logging.getLogger(__name__)
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
TREE_ID = re.compile(r"[a-f0-9]{32}")


def _directory(experiment_id: str, run_id: str) -> tuple[Path, dict]:
    run = training_run(experiment_id, run_id)
    run_dir = PROCS.run_dir(experiment_id, run_id).resolve()
    directory = (run_dir / "rollout-trees").resolve()
    if not directory.is_relative_to(run_dir):
        raise HTTPException(404, "运行记录路径无效")
    return directory, run


def _read_json(directory: Path, name: str, limit: int = MAX_FILE_BYTES) -> dict:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise HTTPException(404, "树记录路径无效")
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise HTTPException(413, "此记录超过读取上限，不能作为完整结果展示")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        return value
    except FileNotFoundError as error:
        raise HTTPException(404, "尚无对应树记录") from error
    except (OSError, ValueError) as error:
        logger.error("Cannot read rollout tree record %s: %s", path, error)
        raise HTTPException(500, "树记录读取失败或内容损坏") from error


def _tree(directory: Path, tree_id: str, experiment_id: str, run_id: str):
    if not TREE_ID.fullmatch(tree_id):
        raise HTTPException(404, "树记录不存在")
    raw = _read_json(directory, f"{tree_id}.json")
    ensure_workflow_path()
    from workflow.contracts import RolloutTree
    from workflow.rollout_tree import stable_id

    try:
        tree = RolloutTree.model_validate(raw)
        if tree.schema_version not in (2, 3) or (tree.experiment_id, tree.run_id, tree.tree_id) != (
            experiment_id, run_id, tree_id
        ) or tree_id != stable_id(experiment_id, run_id, tree.mode, tree.group_id):
            raise ValueError("Tree identity does not match run")
        return tree
    except ValueError as error:
        logger.error("Invalid rollout tree %s: %s", tree_id, error)
        raise HTTPException(500, "树结构或运行归属不一致") from error


def _respond(request: Request, value: dict) -> Response:
    body = json.dumps(_redact(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
    etag = f'"{hashlib.sha256(body).hexdigest()}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type="application/json", headers=headers)


@router.get("/{run_id}/rollout-trees")
def list_trees(
    request: Request, run_id: str, experiment_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
    mode: Literal["train", "val"] | None = None, task_id: str | None = None,
):
    directory, run = _directory(experiment_id, run_id)
    common = {"schema_version": 2, "experiment_id": experiment_id, "run_id": run_id,
              "run_state": run.get("state"), "offset": offset, "limit": limit}
    if not (directory / "manifest.json").exists():
        return _respond(request, {
            **common, "availability": "missing", "recording_status": "unavailable",
            "expected": bool((run.get("meta") or {}).get("rollout_tree")),
            "items": [], "total": 0, "next_offset": None,
        })
    manifest = _read_json(directory, "manifest.json", MAX_MANIFEST_BYTES)
    if (manifest.get("schema_version"), manifest.get("experiment_id"), manifest.get("run_id")) != (
        2, experiment_id, run_id
    ) or not isinstance(manifest.get("trees"), list):
        raise HTTPException(500, "树目录的运行归属或格式无效")
    summaries = manifest["trees"]
    if any(not isinstance(row, dict) or not TREE_ID.fullmatch(str(row.get("tree_id", ""))) for row in summaries):
        raise HTTPException(500, "树目录条目损坏")
    selected = [row for row in summaries if (mode is None or row.get("mode") == mode)
                and (task_id is None or row.get("task_id") == task_id)]
    def _still_running(row: dict) -> bool:
        return int(row.get("rollout_count") or 0) > int(row.get("completed_count") or 0)

    active = [row for row in selected if _still_running(row)]
    finished = [row for row in selected if not _still_running(row)]
    active.sort(key=lambda row: (str(row.get("updated_at") or ""), row["tree_id"]), reverse=True)
    finished.sort(key=lambda row: (row.get("created_at", ""), row["tree_id"]))
    selected = active + finished
    items = []
    for summary in selected[offset:offset + limit]:
        try:
            tree = _tree(directory, summary["tree_id"], experiment_id, run_id)
            from workflow.rollout_tree import RolloutTreeArchive

            items.append({**RolloutTreeArchive._summary(tree), "query": tree.query[:240],
                          "query_truncated": len(tree.query) > 240, "availability": "available"})
        except HTTPException as error:
            items.append({**summary, "availability": "unreadable", "error": error.detail})
    return _respond(request, {
        **common, "availability": "available" if summaries else "empty",
        "recording_status": manifest.get("status", "unknown"),
        "recording_errors": manifest.get("errors", []),
        "items": items, "total": len(selected),
        "next_offset": offset + limit if offset + limit < len(selected) else None,
    })


@router.get("/{run_id}/rollout-trees/{tree_id}")
def tree_detail(
    request: Request, run_id: str, tree_id: str, experiment_id: str,
    cursor: str | None = None, limit: int = Query(100, ge=1, le=500),
    parent_id: str | None = None, plan_offset: int = Query(0, ge=0),
    diagnose: bool = False, reward_level: Literal["outcome", "credit"] = "outcome",
    node_id: str | None = None,
):
    directory, run = _directory(experiment_id, run_id)
    tree = _tree(directory, tree_id, experiment_id, run_id)
    offset = 0
    if cursor:
        if not re.fullmatch(r"\d+:\d+", cursor):
            raise HTTPException(400, "节点分页游标无效")
        revision, offset = map(int, cursor.split(":"))
        if revision != tree.revision:
            raise HTTPException(409, "树已更新，请从第一页重新读取")
    by_id = {node.node_id: node for node in [*tree.nodes, *tree.pending_nodes]}
    if parent_id is not None and parent_id not in by_id:
        raise HTTPException(404, "父节点不存在")
    candidates = [node for node in by_id.values() if node.kind != "query"
                  and (parent_id is None or node.parent_id == parent_id or any(
                      edge.source_node_id == parent_id and edge.target_node_id == node.node_id
                      for edge in tree.edges))]
    if node_id is not None:
        positions = {node.node_id: index for index, node in enumerate(candidates)}
        if node_id not in positions:
            raise HTTPException(404, "执行节点不存在")
        offset = positions[node_id] // limit * limit
    page = candidates[offset:offset + limit]
    visible = {node.node_id for node in tree.nodes if node.kind == "query"}
    for node in page:
        for ancestor in tree.path_to_root(node.node_id):
            visible.add(ancestor)
            if len(visible) > 2000:
                raise HTTPException(413, "节点祖先范围过大，请按子树读取")
    public = tree.model_dump(mode="json")
    public["nodes"] = [node for node in public["nodes"] if node["node_id"] in visible]
    public["pending_nodes"] = [node for node in public["pending_nodes"] if node["node_id"] in visible]
    public["edges"] = [edge for edge in public["edges"]
                       if edge["source_node_id"] in visible and edge["target_node_id"] in visible]
    involved = {by_id[key].rollout_id for key in visible if key in by_id}
    public["rollouts"] = [record for record in public["rollouts"] if record["node_id"] in involved
                          or record["store_status"] not in ("succeeded", "failed", "cancelled")][:500]
    result_ids = visible | {record["result_node_id"] for record in public["rollouts"]}
    public["outcomes"] = {key: result for key, result in public["outcomes"].items() if key in result_ids}
    for node in [*public["nodes"], *public["pending_nodes"]]:
        if node["kind"] == "query":
            continue
        outcome = public["outcomes"].get(node["node_id"], {})
        node["missing_result_fields"] = [
            key for key in ("answer", "reward") if outcome.get(key) is None
        ] if node["kind"] in ("rollout", "outcome") else []
        node["terminal_unconfirmed"] = (
            run.get("state") in ("succeeded", "failed", "cancelled", "interrupted")
            and (node["status"] if node["kind"] == "execution" else node["store_status"])
            not in ("succeeded", "failed", "cancelled")
        )
    public["plans"] = public["plans"][plan_offset:plan_offset + limit]
    public["query_truncated"] = len(tree.query) > 4000
    public["query"] = tree.query[:4000]
    diagnostics = {"status": "not_requested", "items": []}
    if diagnose and tree.schema_version == 3:
        diagnostics = {"status": "unsupported", "items": [],
                       "reason": "执行节点的 credit 诊断尚未实现，不能套用旧 rollout 粒度"}
    elif diagnose:
        from workflow.harness import RewardHackingMonitor

        findings = RewardHackingMonitor(reward_level=reward_level).check_tree(tree)
        diagnostics = {
            "status": "computed", "tree_revision": tree.revision, "reward_level": reward_level,
            "total": len(findings), "truncated": len(findings) > 100,
            "items": [finding.model_dump(mode="json") for finding in findings[:100]],
        }
    return _respond(request, {
        "schema_version": tree.schema_version, "experiment_id": experiment_id, "run_id": run_id,
        "run_state": run.get("state"), "tree": public, "diagnostics": diagnostics,
        "page": {"offset": offset, "limit": limit, "total": len(candidates),
                 "next_cursor": f"{tree.revision}:{offset + limit}" if offset + limit < len(candidates) else None,
                 "truncated": offset > 0 or offset + limit < len(candidates),
                 "parent_id": parent_id,
                 "plan_total": len(tree.plans),
                 "next_plan_offset": plan_offset + limit if plan_offset + limit < len(tree.plans) else None},
    })


@router.get("/{run_id}/rollout-trees/{tree_id}/nodes/{node_id}")
def node_detail(request: Request, run_id: str, tree_id: str, node_id: str, experiment_id: str):
    directory, run = _directory(experiment_id, run_id)
    tree = _tree(directory, tree_id, experiment_id, run_id)
    node = next((node for node in [*tree.nodes, *tree.pending_nodes] if node.node_id == node_id), None)
    if node is None:
        raise HTTPException(404, "节点不存在")
    detail = None
    if node.detail_ref:
        from workflow.execution_recording import attempt_directory

        expected = attempt_directory(directory, node.rollout_id, node.attempt_id) / f"{node.node_id}.json"
        if not re.fullmatch(r"[a-f0-9]{32}", node.node_id) or (
            directory / node.detail_ref
        ).resolve() != expected.resolve():
            raise HTTPException(500, "节点内容引用无效")
        detail = _read_json(directory, node.detail_ref, 2 * 1024 * 1024)
        if (detail.get("node_id"), detail.get("rollout_id"), detail.get("attempt_id")) != (
            node.node_id, node.rollout_id, node.attempt_id
        ):
            raise HTTPException(500, "节点内容归属不一致")
    return _respond(request, {
        "experiment_id": experiment_id, "run_id": run_id, "tree_id": tree_id,
        "run_state": run.get("state"), "node": node.model_dump(mode="json"),
        "detail": detail, "outcome": tree.outcomes.get(node_id),
    })
