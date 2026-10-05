"""Fast phase-two checks; real result merge, fake Store and in-process HTTP."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workflow.rollout_tree import RolloutTreeArchive
from workflow.rollout_results import RunnerResult, result_annotation, RESULT_ATTRIBUTE
from rl.hooks.rollout_tree import RolloutTreeRecorder
from science_infra.control import rollout_trees as api


def sample():
    return {"data_id": "group", "question": "query api_key=private", "id": "task"}


def result(rid="a", attempt="try1", answer="answer", reward=0.0, failed=False):
    return RunnerResult.model_validate_json(result_annotation(
        rid, attempt, answer, reward, True, failed, "archive"
    ))


def observation(attempt="try1", sequence=1, status="succeeded", **extra):
    return dict(attempt_id=attempt, sequence=sequence, store_status=status,
                attempt_status=status, **extra)


def test_parent_outcomes_retry_late_results_and_zero(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "run")
    archive.record("a", sample(), "train")
    archive.record("child", {**sample(), "resume_parent_id": "a"}, "train")
    archive.observe("a", sample(), "train", **observation(results=[result()], reward=0.0))
    archive.observe("child", sample(), "train", **observation(results=[result("child", reward=1)]))
    archive.flush()
    tree = archive.group(sample(), "train")
    assert tree.outcomes["a"]["reward"] == 0
    assert len(tree.outcomes) == 2 and "a" not in tree.leaves()
    revision = tree.revision
    archive.observe("a", sample(), "train", **observation(results=[result()], reward=0.0))
    archive.flush()
    assert archive.group(sample(), "train").revision == revision
    archive.observe("a", sample(), "train", **observation("try2", 2, "running"))
    archive.observe("a", sample(), "train", **observation(results=[result(answer="late", reward=9)]))
    tree = archive.group(sample(), "train")
    node = next(n for n in tree.nodes if n.node_id == "a")
    assert node.attempt_id == "try2" and node.status == "running"
    assert "a" not in tree.outcomes and node.previous_attempts[0]["outcome"]["reward"] == 0
    archive.observe("a", sample(), "train", **observation("try2", 2, results=[result(attempt="try2", failed=True)]))
    assert node.status == "failed" and node.store_status == "succeeded"
    assert node.reward is None
    archive.flush()
    restored = RolloutTreeArchive(tmp_path, "exp", "run").group(sample(), "train")
    assert restored.outcomes["a"]["execution_error"] == "execution_error"


def test_conflicts_missing_values_and_multiple_verdicts(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "run")
    archive.record("a", sample(), "train")
    archive.observe("a", sample(), "train", **observation())
    tree = archive.group(sample(), "train")
    assert "a" not in tree.outcomes
    archive.observe("a", sample(), "train", **observation(results=[result()]))
    archive.observe("a", sample(), "train", **observation(results=[result(answer="different")]))
    assert tree.outcomes["a"]["answer"] == "answer"
    node = tree.nodes[1]
    assert node.record_issues
    archive.judgments("a", sample(), "train", [
        {"verdict": "validate", "action_key": "w1"},
        {"verdict": "invalidate", "action_key": "w2"},
    ], 1, "try1")
    assert node.verdict is None and len(node.judgments) == 2
    archive.judgments("a", sample(), "train", [{"verdict": "validate", "action_key": "w1"}], 1, "try1")
    assert len(node.judgments) == 2
    with pytest.raises(ValueError, match="attempt"):
        archive.judgments("a", sample(), "train", [], 1, "old")
    assert node.reward is None
    assert result(answer="x" * 5000).answer_truncated
    assert len(result(answer="x" * 5000).answer) == 4000


def rollout(status="succeeded"):
    return SimpleNamespace(
        rollout_id="a", mode="train", metadata={"data_id": "group"}, status=status,
        start_time=1.0, end_time=2.0,
        attempt=SimpleNamespace(attempt_id="try1", sequence_id=1, status=status,
                                start_time=1.0, end_time=2.0),
    )


def test_store_reconcile_event_and_cancelled_poll_task(tmp_path, capsys):
    recorder = RolloutTreeRecorder({
        "TIR_ROLLOUT_TREE_DIR": str(tmp_path), "TIR_EXPERIMENT_ID": "exp", "TIR_RUN_ID": "run",
    })
    recorder.initial({"a": sample()}, "train")
    store = SimpleNamespace(
        query_rollouts=AsyncMock(return_value=[rollout()]),
        query_spans=AsyncMock(return_value=[SimpleNamespace(attributes={
            RESULT_ATTRIBUTE: result().model_dump_json()
        })]),
    )

    async def run():
        async with recorder.watching(store, {"a": sample()}, "train"):
            await recorder.reconcile(store, {"a": sample()}, "train")
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(run())
    tree = recorder.archive.group(sample(), "train")
    assert tree.outcomes["a"]["answer"] == "answer"
    assert store.query_spans.call_args.kwargs["attempt_id"] == "try1"
    assert '"event": "tree_updated"' in capsys.readouterr().out
    store.query_rollouts.side_effect = RuntimeError("disconnected")
    asyncio.run(recorder._reconcile_safely(store, {"a": sample()}, "train"))
    assert recorder.archive.errors


@pytest.fixture
def client(tmp_path, monkeypatch):
    run = tmp_path / "run"
    archive = RolloutTreeArchive(run / "rollout-trees", "exp", "abcdef123456")
    for rid in ("a", "b", "c"):
        archive.record(rid, sample(), "train")
        archive.observe(rid, sample(), "train", **observation(results=[result(rid)]))
    archive.flush()
    tree = archive.group(sample(), "train")

    def registered(exp, rid):
        if (exp, rid) != ("exp", "abcdef123456"):
            raise HTTPException(404, "wrong run")
        return {"state": "succeeded", "meta": {"rollout_tree": {}}}

    monkeypatch.setattr(api, "training_run", registered)
    monkeypatch.setattr(api.PROCS, "run_dir", lambda exp, rid: run)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as http:
        yield http, archive, tree.tree_id, run


def test_api_pagination_etag_identity_and_restart(client):
    http, archive, tree_id, run = client
    base = "/api/rl/runs/abcdef123456/rollout-trees"
    listed = http.get(base, params={"experiment_id": "exp"})
    assert listed.status_code == 200
    assert "private" not in listed.text
    assert listed.json()["items"][0]["completed_count"] == 3
    assert http.get(base, params={"experiment_id": "other"}).status_code == 404
    url = f"{base}/{tree_id}?experiment_id=exp&limit=1"
    first = http.get(url)
    assert first.status_code == 200
    assert len(first.json()["tree"]["nodes"]) == 2
    assert first.json()["page"]["truncated"]
    assert http.get(url, headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    diagnostic = http.get(url + "&diagnose=true")
    assert diagnostic.status_code == 200
    assert diagnostic.json()["diagnostics"]["tree_revision"] == first.json()["tree"]["revision"]
    second = http.get(url + "&cursor=" + first.json()["page"]["next_cursor"])
    assert second.json()["tree"]["nodes"][1]["node_id"] == "b"
    archive.record("d", sample(), "train")
    archive.flush()
    assert http.get(url + "&cursor=" + first.json()["page"]["next_cursor"]).status_code == 409
    raw = json.loads((run / "rollout-trees" / f"{tree_id}.json").read_text())
    raw["run_id"] = "foreign"
    (run / "rollout-trees" / f"{tree_id}.json").write_text(json.dumps(raw))
    assert http.get(url).status_code == 500
    assert http.get(base + "/invalid?experiment_id=exp").status_code == 404


def test_api_missing_corrupt_and_size_limit(client, monkeypatch):
    http, archive, tree_id, run = client
    base = "/api/rl/runs/abcdef123456/rollout-trees"
    manifest = run / "rollout-trees" / "manifest.json"
    manifest.unlink()
    response = http.get(base + "?experiment_id=exp")
    assert response.json()["availability"] == "missing"
    RolloutTreeArchive(run / "rollout-trees", "exp", "abcdef123456")
    assert http.get(base + "?experiment_id=exp").json()["total"] == 1
    monkeypatch.setattr(api, "MAX_MANIFEST_BYTES", 10)
    assert http.get(base + "?experiment_id=exp").status_code == 413


def test_harness_compares_only_same_prefix_and_reward_level(tmp_path):
    from workflow.harness import RewardHackingMonitor

    archive = RolloutTreeArchive(tmp_path, "exp", "run")
    archive.record("a", sample(), "train")
    for i, reward in enumerate((0.1, 0.1, 0.99)):
        rid = f"child{i}"
        archive.record(rid, {**sample(), "resume_parent_id": "a",
                             "window_id": "w", "snapshot_ref": "snap"}, "train")
        archive.observe(rid, sample(), "train", **observation(results=[result(rid, reward=reward)]))
    tree = archive.group(sample(), "train")
    for node in tree.nodes[2:]:
        node.metrics["reward_scheme"] = "same"
    hits = RewardHackingMonitor().check_tree(tree)
    assert len(hits) == 1 and hits[0].meta["node_id"] == "child2"
    json.dumps([hit.model_dump() for hit in hits], allow_nan=False)
    assert not RewardHackingMonitor(reward_level="credit").check_tree(tree)
    tree.nodes[-1].window_id = "other"
    assert not RewardHackingMonitor().check_tree(tree)


@pytest.mark.parametrize("rejected", [False, True])
def test_actual_daemon_completion_uses_exact_attempt(tmp_path, rejected):
    import ast
    from workflow.rollout_results import RESULT_ATTRIBUTE

    recorder = RolloutTreeRecorder({
        "TIR_ROLLOUT_TREE_DIR": str(tmp_path), "TIR_EXPERIMENT_ID": "exp", "TIR_RUN_ID": "run",
    })
    recorder.initial({"a": sample()}, "train")
    source = ast.parse((ROOT.parent / "rl" / "hooks" / "daemon.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "TirAgentModeDaemon")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_validate_data_v1")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method
    ], type_ignores=[])
    from rl.hooks.overlay import BRANCHING_ALGOS

    namespace = {"extract_tir_meta_from_spans": lambda spans: {},
                 "Task": SimpleNamespace, "RolloutLegacy": SimpleNamespace,
                 "BRANCHING_ALGOS": BRANCHING_ALGOS}
    exec(compile(ast.fix_missing_locations(module), "daemon.py", "exec"), namespace)
    spans = [SimpleNamespace(attributes={RESULT_ATTRIBUTE: result().model_dump_json()})]

    def adapt(_):
        if rejected:
            raise ValueError("adapter rejected")
        return [SimpleNamespace(reward=0.0)]

    store = SimpleNamespace(query_spans=AsyncMock(return_value=spans))
    daemon = SimpleNamespace(
        store=store, adapter=SimpleNamespace(adapt=adapt), _tree_recorder=recorder,
        _task_id_to_original_sample={"a": sample()}, _rollout_meta={},
        _branch_local_count_total=0, _validate_data=lambda result: None,
        is_train=True, tir_algo="grpo",
    )
    actual = rollout()
    actual.input, actual.resources_id = sample(), "resource"
    if rejected:
        with pytest.raises(ValueError, match="adapter rejected"):
            asyncio.run(namespace["_validate_data_v1"](daemon, actual))
    else:
        validated = asyncio.run(namespace["_validate_data_v1"](daemon, actual))
        assert validated.final_reward == 0.0
    assert store.query_spans.call_args.kwargs["attempt_id"] == "try1"
    tree = recorder.archive.group(sample(), "train")
    assert tree.outcomes["a"]["answer"] == "answer"
    assert tree.nodes[1].training_status == ("rejected" if rejected else "adapted")
