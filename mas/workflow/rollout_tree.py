"""Run-scoped rollout lineage aggregation. No training-framework dependencies."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .contracts import ExecutionTrace, RolloutTree, RolloutTreeNode, RolloutTreePlan, _utcnow
from .rollout_results import TERMINAL, merge_judgments, merge_observation

logger = logging.getLogger(__name__)


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]


def identify_plans(parent_id: str, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign legacy plans IDs before filtering; never include message contents."""
    return [
        {**plan, "plan_id": plan.get("plan_id") or "plan:" + stable_id(
            parent_id, index, plan.get("parent_id"), plan.get("site_id"),
            (plan.get("meta") or {}).get("window_id"),
            (plan.get("meta") or {}).get("snapshot_ref"),
        )}
        for index, plan in enumerate(plans)
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RolloutTreeArchive:
    """Single-writer archive; keep only the current batch's trees in memory."""

    def __init__(self, root: Path, experiment_id: str, run_id: str, *, schema_version: int = 2):
        self.root = root
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.schema_version = schema_version
        self.trees: dict[str, RolloutTree] = {}
        self.dirty: set[str] = set()
        self.errors: list[str] = []
        self._manifest_cache: dict[str, Any] | None = None
        self._execution_cache: dict[Path, tuple[int, int, ExecutionTrace]] = {}
        root.mkdir(parents=True, exist_ok=True)
        self.summaries: dict[str, dict[str, Any]] = {}
        manifest = root / "manifest.json"
        if manifest.exists():
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                if (previous.get("experiment_id"), previous.get("run_id")) != (experiment_id, run_id):
                    raise ValueError("Archive directory belongs to another run")
                self.errors = list(previous.get("errors") or [])[-20:]
            except (OSError, ValueError) as error:
                logger.error("Cannot restore RolloutTree manifest: %s", error)
                raise
        # Reconcile snapshots rather than trusting a possibly stale manifest.
        for path in root.glob("*.json"):
            if path.name == "manifest.json":
                continue
            try:
                tree = RolloutTree.model_validate_json(path.read_text(encoding="utf-8"))
                self._check_identity(tree)
                if path.stem != tree.tree_id:
                    raise ValueError("Tree filename does not match identity")
                self.summaries[tree.tree_id] = self._summary(tree)
            except (OSError, ValueError) as error:
                self.report_error(f"restore {path.name}: {error}")
        self._write_manifest()

    def _check_identity(self, tree: RolloutTree) -> None:
        if tree.schema_version not in (2, 3) or (tree.experiment_id, tree.run_id) != (
            self.experiment_id, self.run_id
        ) or tree.tree_id != stable_id(self.experiment_id, self.run_id, tree.mode, tree.group_id):
            raise ValueError("Tree belongs to a different run or group")

    def group(self, sample: Mapping[str, Any], mode: str) -> RolloutTree:
        group_id = str(sample.get("data_id") or "")
        if not group_id:
            raise ValueError("Cannot archive rollout without Store data_id")
        tree_id = stable_id(self.experiment_id, self.run_id, mode, group_id)
        if tree_id not in self.trees:
            path = self.root / f"{tree_id}.json"
            if path.exists():
                tree = RolloutTree.model_validate_json(path.read_text(encoding="utf-8"))
                self._check_identity(tree)
            else:
                task_id = sample.get("id")
                tree = RolloutTree(
                    schema_version=self.schema_version, tree_id=tree_id, experiment_id=self.experiment_id,
                    run_id=self.run_id, group_id=group_id,
                    task_id=str(task_id) if task_id is not None else None,
                    mode=mode, query=str(sample.get("question") or sample.get("query") or ""),
                    nodes=[RolloutTreeNode(node_id=f"query:{tree_id}", kind="query", role="root")],
                )
                self.dirty.add(tree_id)
            self.trees[tree_id] = tree
        return self.trees[tree_id]

    def plans(self, sample: Mapping[str, Any], parent_id: str, raw_plans: list[dict[str, Any]],
              remaining: int) -> None:
        tree = self.group(sample, "train")
        existing = {plan.plan_id: plan for plan in tree.plans}
        for raw in raw_plans:
            if raw["plan_id"] in existing:
                if existing[raw["plan_id"]].parent_id != str(raw.get("parent_id") or parent_id):
                    raise ValueError(f"Conflicting parent for plan {raw['plan_id']}")
                continue
            meta = raw.get("meta") or {}
            reason = "missing_resume_messages" if not raw.get("resume_messages") else (
                "budget_exhausted" if remaining <= 0 else None
            )
            if reason is None:
                remaining -= 1
            plan = RolloutTreePlan(
                plan_id=raw["plan_id"], parent_id=str(raw.get("parent_id") or parent_id),
                site_id=raw.get("site_id") or meta.get("site_id"),
                window_id=meta.get("window_id"), event_id=meta.get("event_id"),
                boundary_snapshot_ref=meta.get("snapshot_ref"),
                branch_depth=raw.get("depth"), role=raw.get("role") or "child",
                decision={key: meta["decision"][key] for key in ("passed", "score", "reason")
                          if isinstance(meta.get("decision"), dict) and key in meta["decision"]},
                metrics={key: meta[key] for key in ("h_root", "h_tool", "event_kind", "reward_scheme")
                         if key in meta},
                source_attempt_id=meta.get("source_attempt_id"),
                fork_node_ids=list(meta.get("fork_node_ids") or []),
                status="skipped" if reason else "planned", reason=reason,
            )
            tree.plans.append(plan)
            existing[plan.plan_id] = plan
            self.dirty.add(tree.tree_id)
        self._attach_pending(tree)

    def record(self, rollout_id: str, sample: Mapping[str, Any], mode: str,
               origin: str = "initial") -> None:
        tree = self.group(sample, mode)
        parent_id = str(sample.get("resume_parent_id") or "") or None
        origin = "branch" if parent_id else origin
        parent_id = parent_id or f"query:{tree.tree_id}"
        known = {node.node_id: node for node in [tree.nodes[0], *tree.rollout_records()]}
        previous = known.get(rollout_id)
        if previous:
            if previous.parent_id != parent_id or previous.plan_id != sample.get("plan_id"):
                raise ValueError(f"Conflicting lineage for rollout {rollout_id}")
            return
        ancestor = parent_id
        visited = {rollout_id}
        while ancestor in known:
            if ancestor in visited:
                raise ValueError(f"Cyclic lineage for rollout {rollout_id}")
            visited.add(ancestor)
            ancestor = known[ancestor].parent_id
        if ancestor == rollout_id:
            raise ValueError(f"Cyclic lineage for rollout {rollout_id}")
        for other in self.trees.values():
            if other is tree:
                continue
            if any(node.node_id in (rollout_id, parent_id) for node in other.rollout_records()):
                raise ValueError(f"Cross-group lineage for rollout {rollout_id}")
        plan_id = sample.get("plan_id")
        plan = next((item for item in tree.plans if item.plan_id == plan_id), None) if plan_id else None
        if plan_id and (plan is None or plan.parent_id != parent_id):
            raise ValueError(f"Missing or mismatched plan for rollout {rollout_id}")
        if plan and plan.child_rollout_id not in (None, rollout_id):
            raise ValueError(f"Plan {plan_id} already belongs to {plan.child_rollout_id}")
        parent = known.get(parent_id)
        node = RolloutTreeNode(
            node_id=rollout_id, parent_id=parent_id, origin=origin,
            role=plan.role if plan else ("child" if origin == "branch" else "parent"),
            depth=parent.depth + 1 if parent else 0, status="enqueued", created_at=_utcnow(),
            plan_id=plan_id, site_id=plan.site_id if plan else sample.get("site_id"),
            window_id=plan.window_id if plan else sample.get("window_id"),
            event_id=plan.event_id if plan else sample.get("event_id"),
            boundary_snapshot_ref=plan.boundary_snapshot_ref if plan else sample.get("snapshot_ref"),
            branch_depth=plan.branch_depth if plan else None,
            metrics=dict(plan.metrics) if plan else {},
            decision=dict(plan.decision) if plan else {},
        )
        if plan:
            plan.child_rollout_id = rollout_id
            plan.status, plan.reason = "enqueued", None
        if tree.schema_version == 3:
            tree.rollouts.append(node)
        else:
            tree.pending_nodes.append(node)
        self._attach_pending(tree)
        self.dirty.add(tree.tree_id)

    @staticmethod
    def _attach_pending(tree: RolloutTree) -> None:
        if tree.schema_version == 3:
            return
        by_id = {node.node_id: node for node in tree.nodes}
        while True:
            attached = []
            for node in tree.pending_nodes:
                parent = by_id.get(node.parent_id)
                if parent:
                    node.depth = parent.depth + 1
                    tree.nodes.append(node)
                    by_id[node.node_id] = node
                    attached.append(node.node_id)
            if not attached:
                break
            attached_ids = set(attached)
            tree.pending_nodes = [node for node in tree.pending_nodes if node.node_id not in attached_ids]
        tree.issues = [f"Unresolved parent {node.parent_id} for {node.node_id}" for node in tree.pending_nodes]
        tree.issues.extend(
            f"Unresolved plan parent {plan.parent_id} for {plan.plan_id}"
            for plan in tree.plans if plan.parent_id not in by_id
        )

    @staticmethod
    def _summary(tree: RolloutTree) -> dict[str, Any]:
        nodes = tree.rollout_records()
        rewards = [outcome["reward"] for node in nodes
                   if (outcome := tree.outcomes.get(node.result_node_id or node.node_id, {})).get("reward") is not None]
        return {
            "tree_id": tree.tree_id, "group_id": tree.group_id, "mode": tree.mode,
            "schema_version": tree.schema_version,
            "task_id": tree.task_id, "created_at": tree.created_at.isoformat(),
            "revision": tree.revision, "updated_at": tree.updated_at.isoformat() if tree.updated_at else None,
            "rollout_count": len(nodes),
            "execution_count": sum(node.kind == "execution" for node in tree.nodes),
            "branch_count": sum(node.origin == "branch" for node in nodes),
            "plan_count": len(tree.plans), "pending_count": len(tree.pending_nodes),
            "issue_count": len(tree.issues) + sum(len(node.record_issues) for node in nodes),
            "gap_count": len(tree.issues),
            "completed_count": sum(node.store_status in TERMINAL for node in nodes),
            "failed_count": sum(node.status == "failed" for node in nodes),
            "unconfirmed_count": sum(node.store_status not in TERMINAL for node in nodes),
            "missing_result_count": sum(
                node.store_status in TERMINAL and (
                    tree.outcomes.get(node.result_node_id or node.node_id, {}).get("answer") is None or
                    tree.outcomes.get(node.result_node_id or node.node_id, {}).get("reward") is None
                ) for node in nodes
            ),
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
        }

    def observe(self, rollout_id: str, sample: Mapping[str, Any], mode: str, **observation: Any) -> None:
        tree = self.group(sample, mode)
        node = next((item for item in tree.rollout_records() if item.node_id == rollout_id), None)
        if node is None:
            raise ValueError(f"Observed rollout {rollout_id} has no archived enqueue record")
        if merge_observation(tree, node, **observation):
            self.dirty.add(tree.tree_id)
        if tree.schema_version == 3:
            from .execution_tree import assemble_executions

            if assemble_executions(tree, self.root, cache=self._execution_cache):
                self.dirty.add(tree.tree_id)

    def judgments(self, rollout_id: str, sample: Mapping[str, Any], mode: str,
                  judgments: list[dict[str, Any]], step: int, attempt_id: str | None) -> None:
        tree = self.group(sample, mode)
        node = next((item for item in tree.rollout_records() if item.node_id == rollout_id), None)
        if node is None or node.attempt_id != attempt_id:
            raise ValueError(f"Judgment attempt mismatch for {rollout_id}")
        if merge_judgments(node, judgments, step):
            self.dirty.add(tree.tree_id)
        if tree.schema_version == 3:
            from .execution_tree import assemble_executions

            assemble_executions(tree, self.root, cache=self._execution_cache)

    def report_error(self, message: str) -> None:
        logger.error("RolloutTree run=%s: %s", self.run_id, message)
        if message not in self.errors:
            self.errors = [*self.errors[-19:], message]

    def clear_errors(self, prefix: str) -> None:
        """Drop a recovered transient error so one blip does not stick for the run."""
        kept = [item for item in self.errors if not str(item).startswith(prefix)]
        if len(kept) == len(self.errors):
            return
        self.errors = kept
        try:
            self._write_manifest()
        except (OSError, ValueError) as error:
            self.report_error(f"write manifest: {error}")

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": 2, "experiment_id": self.experiment_id, "run_id": self.run_id,
            # Node notes such as a conflicting store status are not missing records.
            "status": "degraded" if self.errors or any(
                row.get("gap_count") for row in self.summaries.values()
            ) else "ready",
            "errors": list(self.errors), "trees": sorted(
                self.summaries.values(), key=lambda row: (row.get("created_at", ""), row["tree_id"])
            ),
        }
        if manifest != self._manifest_cache:
            atomic_json(self.root / "manifest.json", manifest)
            self._manifest_cache = manifest

    def flush(self) -> list[dict[str, Any]]:
        changes = []
        for tree_id in list(self.dirty):
            tree = self.trees[tree_id]
            try:
                snapshot = tree.model_copy(update={"revision": tree.revision + 1, "updated_at": _utcnow()})
                snapshot = RolloutTree.model_validate(snapshot.model_dump())
                atomic_json(self.root / f"{tree_id}.json", snapshot.model_dump(mode="json"))
                self.trees[tree_id] = snapshot
                self.summaries[tree_id] = self._summary(snapshot)
                self.dirty.remove(tree_id)
                changes.append({
                    "schema_version": snapshot.schema_version, "event": "tree_updated", "experiment_id": self.experiment_id,
                    "run_id": self.run_id, "tree_id": tree_id, "revision": snapshot.revision,
                })
            except (OSError, ValueError) as error:
                self.report_error(f"write {tree_id}: {error}")
        try:
            self._write_manifest()
        except (OSError, ValueError) as error:
            self.report_error(f"write manifest: {error}")
        return changes

    def release_batch(self) -> None:
        self.flush()
        if self.dirty:
            self.report_error(f"Releasing batch with unpersisted trees: {sorted(self.dirty)}")
            try:
                self._write_manifest()
            except (OSError, ValueError) as error:
                self.report_error(f"write manifest: {error}")
        self.trees.clear()
        self.dirty.clear()
        self._execution_cache.clear()
