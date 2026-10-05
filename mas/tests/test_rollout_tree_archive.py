"""Fast lineage/archive checks, including actual Daemon methods without GPU imports."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workflow.active_set import ActiveSetResult, ForkPlan, expansion_payload_from_result
from workflow.contracts import RolloutTree
from workflow.rollout_tree import RolloutTreeArchive, identify_plans
from rl.hooks.rollout_tree import RolloutTreeRecorder


def sample(group="g1", **extra):
    return {"data_id": group, "id": "same-question", "question": "1+1", **extra}


def plans(parent, count=2):
    return identify_plans(parent, [
        {"parent_id": parent, "site_id": "site", "depth": 1,
         "resume_messages": [{"role": "user", "content": "private-prefix"}],
         "meta": {"window_id": "w2", "snapshot_ref": "archive:window",
                  "decision": {"passed": True, "reason": "gate_passed"}}}
        for _ in range(count)
    ])


def test_group_identity_initials_and_restore(tmp_path):
    archive = RolloutTreeArchive(tmp_path / "run1", "exp", "run1")
    archive.record("a", sample(), "train")
    archive.record("b", sample(), "train")
    archive.record("c", sample("g2"), "train")
    archive.record("v", sample(), "val")
    archive.flush()
    first = archive.group(sample(), "train")
    assert len(archive.summaries) == 3
    assert first.nodes[0].kind == "query"
    assert [node.parent_id for node in first.nodes[1:]] == [first.nodes[0].node_id] * 2
    revision = first.revision
    archive.record("a", sample(), "train")
    archive.flush()
    assert archive.group(sample(), "train").revision == revision
    archive.release_batch()
    assert not archive.trees
    # Repair stale/missing index from the authoritative snapshots.
    (tmp_path / "run1" / "manifest.json").unlink()
    restored = RolloutTreeArchive(tmp_path / "run1", "exp", "run1")
    assert len(restored.summaries) == 3
    assert restored.group(sample(), "train").revision == revision
    other = RolloutTreeArchive(tmp_path / "run2", "exp", "run2")
    assert other.group(sample(), "train").tree_id != first.tree_id
    with pytest.raises(ValueError, match="another run"):
        RolloutTreeArchive(tmp_path / "run1", "other", "run1")
    manifest = (tmp_path / "run1" / "manifest.json").read_text()
    assert "1+1" not in manifest


def test_plans_budget_mapping_nested_and_idempotence(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "run1")
    archive.record("a", sample(), "train")
    archive.record("b", sample(), "train")
    a_plans = plans("a", 3)
    a_plans[0]["resume_messages"] = []
    archive.plans(sample(), "a", a_plans, remaining=1)
    archive.plans(sample(), "b", plans("b", 1), remaining=1)
    selected = sample(resume_parent_id="a", plan_id=a_plans[1]["plan_id"])
    archive.record("a1", selected, "train")
    archive.record("a1", selected, "train")
    nested = plans("a1", 1)
    archive.plans(sample(), "a1", nested, remaining=1)
    archive.record("a11", sample(resume_parent_id="a1", plan_id=nested[0]["plan_id"]), "train")
    archive.record("fill", sample(), "train", "independent_fill")
    archive.flush()
    tree = archive.group(sample(), "train")
    assert [plan.status for plan in tree.plans[:3]] == ["skipped", "enqueued", "skipped"]
    assert tree.plans[2].reason == "budget_exhausted"
    assert tree.path_to_root("a11")[:3] == ["a11", "a1", "a"]
    assert next(node for node in tree.nodes if node.node_id == "a11").depth == 3
    assert next(node for node in tree.nodes if node.node_id == "a11").branch_depth == 1
    assert len(tree.nodes) == 6
    raw = (tmp_path / f"{tree.tree_id}.json").read_text()
    assert "private-prefix" not in raw and '"resume_messages":' not in raw
    assert tree.outcomes == {}


def test_pending_parent_cycles_and_cross_group(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "run")
    archive.record("child", sample(resume_parent_id="later"), "train")
    archive.flush()
    tree = archive.group(sample(), "train")
    assert len(tree.pending_nodes) == 1 and tree.issues
    with pytest.raises(ValueError, match="Cyclic"):
        archive.record("later", sample(resume_parent_id="child"), "train")
    archive.record("later", sample(), "train")
    archive.flush()
    tree = archive.group(sample(), "train")
    assert not tree.pending_nodes and not tree.issues
    assert tree.path_to_root("child")[:2] == ["child", "later"]
    with pytest.raises(ValueError, match="Cross-group"):
        archive.record("foreign", sample("other", resume_parent_id="later"), "train")
    invalid = tree.model_dump()
    invalid["nodes"][1]["depth"] = 99
    with pytest.raises(ValueError, match="parent or depth"):
        RolloutTree.model_validate(invalid)


def test_write_failure_preserves_snapshot_and_retries(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "run")
    archive.record("a", sample(), "train")
    archive.flush()
    tree = archive.group(sample(), "train")
    before = (tmp_path / f"{tree.tree_id}.json").read_bytes()
    archive.record("b", sample(), "train")
    with patch("workflow.rollout_tree.atomic_json", side_effect=OSError("disk full")):
        archive.flush()
    assert archive.dirty and archive.errors
    assert (tmp_path / f"{tree.tree_id}.json").read_bytes() == before
    archive.flush()
    assert not archive.dirty
    restored = RolloutTreeArchive(tmp_path, "exp", "run")
    assert len(restored.group(sample(), "train").nodes) == 3
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "degraded"


def test_legacy_expansion_keeps_shape_and_plan_identity():
    plan = ForkPlan(parent_id="p", depth=1, resume_messages=[])
    result = ActiveSetResult(plans=[plan])
    one = expansion_payload_from_result(result, task={"question": "query"})
    two = expansion_payload_from_result(result)
    assert one["plans"][0]["plan_id"] == two["plans"][0]["plan_id"]
    assert one["tree"]["query"] == "query"
    assert set(one["tree"]) == {"tree_id", "query", "nodes", "outcomes"}
    assert "kind" not in one["tree"]["nodes"][0]
    assert identify_plans("p", [{}]) == identify_plans("p", [{}])


def daemon_class(expansions):
    """Load the real class body with a fake transport, not torch/VERL/Ray."""
    source = ast.parse((ROOT.parent / "rl" / "hooks" / "daemon.py").read_text(encoding="utf-8"))
    body = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "TirAgentModeDaemon")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), body], type_ignores=[])
    from rl.hooks.overlay import BRANCHING_ALGOS

    namespace = {
        "AgentModeDaemon": object, "RolloutTreeRecorder": RolloutTreeRecorder,
        "identify_plans": identify_plans, "load_local_expansion": expansions.get,
        "EnqueueRolloutRequest": SimpleNamespace, "RolloutConfig": SimpleNamespace,
        "_to_native": lambda value: value,
        "resume_boundary_proxy": len,
        "BRANCHING_ALGOS": BRANCHING_ALGOS,
    }
    exec(compile(ast.fix_missing_locations(module), "daemon.py", "exec"), namespace)
    return namespace["TirAgentModeDaemon"]


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("branching", [False, True])
def test_actual_daemon_submission_paths(tmp_path, incremental, branching):
    a, b = plans("a", 3), plans("b", 2)
    a[0]["resume_messages"] = []
    expansions = {"a": {"plans": a}, "b": {"plans": b}} if branching else {}
    cls = daemon_class(expansions)
    daemon = cls.__new__(cls)
    daemon.mode = "v1"
    daemon._full_group_n = 5
    daemon.train_rollout_n = 5
    daemon.llm_timeout_seconds = 30
    daemon._store_enqueue_branch_count = daemon._branch_local_count_total = 0
    daemon._total_tasks_queued = 2
    daemon._enqueued_expansion_parents = set()
    daemon._pending_original_by_data_id = {"g1": sample()}
    daemon._task_id_to_original_sample = {"a": sample(), "b": sample()}
    daemon._completed_rollouts_v0 = {}
    daemon._rollout_meta = {}
    daemon._tree_recorder = RolloutTreeRecorder({
        "TIR_ROLLOUT_TREE_DIR": str(tmp_path), "TIR_EXPERIMENT_ID": "exp", "TIR_RUN_ID": "run",
    })
    daemon._tree_recorder.initial(daemon._task_id_to_original_sample, "train")
    submitted = []

    async def enqueue(requests):
        result = []
        for request in requests:
            submitted.append(request)
            result.append(SimpleNamespace(rollout_id=f"new{len(submitted)}", metadata=request.metadata))
        return result

    daemon.store = SimpleNamespace(enqueue_many_rollouts=AsyncMock(side_effect=enqueue))
    if incremental:
        asyncio.run(daemon._enqueue_expansion_for_parent("a"))
        asyncio.run(daemon._enqueue_expansion_for_parent("b"))
        asyncio.run(daemon._enqueue_from_runner_expansions())
    else:
        asyncio.run(daemon._enqueue_from_runner_expansions())
    assert len(submitted) == 3
    tree = daemon._tree_recorder.archive.group(sample(), "train")
    children = [node for node in tree.nodes if node.origin == "branch"]
    if branching:
        assert [node.parent_id for node in children] == ["a", "a", "b"]
        assert [node.plan_id for node in children] == [a[1]["plan_id"], a[2]["plan_id"], b[0]["plan_id"]]
        assert next(plan for plan in tree.plans if plan.plan_id == b[1]["plan_id"]).status == "skipped"
    else:
        assert not children and not tree.plans
        assert len([node for node in tree.nodes if node.origin == "independent_fill"]) == 3
    assert not tree.pending_nodes
    assert daemon._total_tasks_queued == 5
    count = len(tree.nodes)
    if incremental:
        asyncio.run(daemon._enqueue_expansion_for_parent("a"))
    else:
        asyncio.run(daemon._enqueue_from_runner_expansions())
    assert len(daemon._tree_recorder.archive.group(sample(), "train").nodes) == count


def test_recorder_rejects_mismatched_store_response_without_failing_training(tmp_path):
    recorder = RolloutTreeRecorder({
        "TIR_ROLLOUT_TREE_DIR": str(tmp_path), "TIR_EXPERIMENT_ID": "exp", "TIR_RUN_ID": "run",
    })
    recorder.initial({"a": sample()}, "train")
    requested = plans("a", 1)
    recorder.plans(sample(), "a", requested, 1)
    recorder.submitted(
        [SimpleNamespace(rollout_id="wrong", metadata={"data_id": "other"})],
        [sample(resume_parent_id="a", plan_id=requested[0]["plan_id"])],
    )
    tree = recorder.archive.group(sample(), "train")
    assert len(tree.nodes) == 2
    assert recorder.archive.errors
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "degraded"


def test_bound_daemon_keeps_launch_identity_across_environment_changes():
    import os
    from rl.hooks.rollout_tree import TREE_ENV_KEYS

    source = ast.parse((ROOT.parent / "rl" / "hooks" / "trainer.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "bound_daemon_cls")

    class Daemon:
        def __init__(self, **kwargs):
            self.config = kwargs

    namespace = {"os": os, "TREE_ENV_KEYS": TREE_ENV_KEYS, "TirAgentModeDaemon": Daemon, "Type": type}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "trainer.py", "exec"), namespace)
    context = dict(zip(TREE_ENV_KEYS, ["output", "experiment", "run"]))
    with patch.dict(os.environ, context):
        bound = namespace["bound_daemon_cls"]("grpo", {})
    with patch.dict(os.environ, {}, clear=True):
        assert bound().config["tree_context"] == context
