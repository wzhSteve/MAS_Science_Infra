"""Centralized (run_compiled_episode) execution recording for rollout trees.

Covers task-4: the centralized topology (AgentFlow / planning-led walks) must
feed ``ExecutionRecorder`` the same branch bookkeeping the legacy ReAct path
provides — ``resume_applied`` on branch rollouts, ``recorder.window()`` with a
snapshot_ref shared with the child's resume dict, and window metrics carrying
``fork_node_ids`` / ``source_attempt_id`` — so
``execution_tree.assemble_executions`` can draw real branch edges. The
negative cases pin the "never fabricate a confirmed prefix" contract: a child
with a missing resume confirmation or a broken anchor stays an
``Unconfirmed execution prefix`` issue and gets no branch edge.
"""

import json
from pathlib import Path

import pytest

from tests.test_execution_tree import SAMPLE  # noqa: F401  (marker import)


def observe(archive, rid, *, attempt="at1", reward=0.0, status="succeeded", sequence=1):
    from workflow.rollout_results import RunnerResult

    result = RunnerResult(rollout_id=rid, attempt_id=attempt, answer="answer",
                          reward=reward, format_ok=True)
    archive.observe(rid, SAMPLE, "train", attempt_id=attempt, sequence=sequence,
                    store_status=status, attempt_status=status,
                    results=[result] if status == "succeeded" else [])


def _centralized_spec():
    """Tiny planner→executor topology with an always-branch planner site."""
    from workflow.contracts import BranchSite
    from workflow.spec import MASSpec

    spec = MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "tools": ["python_coder"],
        "agents": [
            {"id": "planner", "kind": "planner", "tools": ["python_coder"]},
            {"id": "python_coder", "kind": "tool", "trainable": False},
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
        ],
        "routers": [{"id": "route_exec", "candidates": ["python_coder"], "strategy": "llm_choice"}],
        "edges": [
            {"from": "planner", "to": "route_exec", "kind": "route"},
            {"from": "route_exec", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "sampling": {
            "mode": "arpo",
            "group_n": 2,
            "beam_size": 2,
            "sites": [
                {"id": "after_planner", "enabled": True,
                 "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
                 "gate": {"type": "always"}, "fork": {"beam_size": 2, "resume_mode": "messages"}},
            ],
        },
    })
    site = BranchSite(id="after_planner", anchor={"kind": "after_agent_turn", "agent_id": "planner"},
                      gate={"type": "always"}, fork={"beam_size": 2, "resume_mode": "messages"})
    return spec, site


def _mock_llm():
    from workflow.centralized_runtime import MockWindowLLM

    return MockWindowLLM({
        "planner": [json.dumps({"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True})],
        "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
    })


def _run_parent(task, spec, *, recorder):
    """Run one centralized parent episode (run_compiled_episode dispatch)."""
    from workflow.archive import Archive, register_archive
    from workflow.compiler import compile_spec
    from workflow.memory import MemoryStore
    from workflow.runtime import run_compiled_episode

    arch = register_archive(Archive())
    compiled = compile_spec(spec)
    return run_compiled_episode(
        task, None, arch, spec, MemoryStore(), compiled,
        window_llm=_mock_llm(), execution_recorder=recorder,
    )


def test_centralized_parent_records_windows_and_fork_metrics(tmp_path, monkeypatch):
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    from workflow.execution_recording import ExecutionRecorder

    task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1"}
    recorder = ExecutionRecorder.from_task(task)
    spec, _site = _centralized_spec()
    raw = _run_parent(task, spec, recorder=recorder)

    planner_events = [e for e in raw.window_events if e.get("agent_id") == "planner"]
    assert planner_events, "planner window event missing"
    planner_nodes = [n for n in recorder.trace.nodes if n.agent_id == "planner"]
    assert planner_nodes, "planner execution node missing"
    assert recorder.trace.windows, "recorder windows must not be empty"

    event = planner_events[0]
    window = recorder.trace.windows.get(event["event_id"])
    assert window is not None, "window event not mirrored into recorder"
    assert window["snapshot_ref"] == event["snapshot_ref"]
    assert window["fork_node_ids"] == [planner_nodes[0].node_id]
    # Branch bookkeeping present for ActiveSetSession plan_forks_from_raw.
    assert event["metrics"]["fork_node_ids"] == [planner_nodes[0].node_id]
    assert event["metrics"]["source_attempt_id"] == "at1"
    assert event["metrics"]["source_attempt_id"] == recorder.trace.attempt_id
    # prefix_hash must match a child that resumed from this exact message list.
    from workflow.rollout_tree import stable_id

    assert window["prefix_hash"] == stable_id(event["messages"])


def test_centralized_episode_branch_edges_confirmed(tmp_path, monkeypatch):
    """Positive: parent + resume child on the centralized path draw a branch edge."""
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    from workflow.active_set import ActiveSetConfig, ActiveSetSession
    from workflow.execution_recording import ExecutionRecorder
    from workflow.rollout_tree import RolloutTreeArchive
    from workflow.runtime import run_compiled_episode
    from workflow.archive import Archive, register_archive
    from workflow.compiler import compile_spec
    from workflow.memory import MemoryStore

    spec, site = _centralized_spec()
    compiled = compile_spec(spec)
    parent_task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1",
                  "expand_in_runner": True, "sampling_budget": 2, "ready_batch": False,
                  "force_local_expand": False}
    recorder = ExecutionRecorder.from_task(parent_task)
    arch = register_archive(Archive())

    def run_episode(t):
        return run_compiled_episode(
            t, None, arch, spec, MemoryStore(), compiled,
            window_llm=_mock_llm(),
            execution_recorder=recorder if t.get("_rollout_id") == "a" else None,
        )

    session = ActiveSetSession(
        ActiveSetConfig(group_budget=2, sites=[site], run_probes=False),
        on_decision=recorder.sampling,
    )
    generated = session.run(parent_task, run_episode, parent_rollout_id="a")
    assert generated.plans, "centralized episode must plan branch rollouts"
    plan = generated.plans[0]
    planner_node = next(n for n in recorder.trace.nodes if n.agent_id == "planner")
    assert plan.meta["fork_node_ids"] == [planner_node.node_id]
    assert plan.meta["source_attempt_id"] == "at1"
    assert plan.meta["window_id"] and plan.meta["snapshot_ref"] == plan.meta["window_id"]

    # Child rollout: daemon enqueue would stamp these task fields via
    # daemon._enqueue_expansion_for_parent; emulate the full pipeline:
    # tree plans registration (daemon RolloutTreeRecorder.plans) + enqueue.
    recorded = RolloutTreeArchive(root, "exp", "123456abcdef", schema_version=3)
    from workflow.rollout_tree import identify_plans

    recorded.plans(SAMPLE, "a", identify_plans("a", [{
        "plan_id": plan.plan_id, "parent_id": "a", "depth": plan.depth, "role": plan.role,
        "site_id": plan.site_id, "resume_messages": plan.resume_messages, "meta": plan.meta,
    }]), 1)
    child_task = {
        **SAMPLE,
        "_rollout_id": "b", "_attempt_id": "at1",
        "resume_messages": plan.resume_messages,
        "resume_parent_id": "a", "is_branch": True, "plan_id": plan.plan_id,
        "site_id": plan.site_id, "window_id": plan.meta["window_id"],
        "snapshot_ref": plan.meta["snapshot_ref"],
        "fork_node_ids": plan.meta["fork_node_ids"],
        "source_attempt_id": plan.meta["source_attempt_id"],
    }
    child_recorder = ExecutionRecorder.from_task(child_task)
    assert child_recorder.trace.resume["source_rollout_id"] == "a"
    assert child_recorder.trace.resume["source_attempt_id"] == "at1"
    assert child_recorder.trace.resume["fork_node_ids"] == [planner_node.node_id]
    child_raw = run_compiled_episode(
        child_task, None, arch, spec, MemoryStore(), compiled,
        window_llm=_mock_llm(), execution_recorder=child_recorder,
    )
    assert child_raw.final_answer and not child_raw.error
    assert child_recorder.trace.resume_applied is True
    child_recorder.complete()
    recorder.complete()

    recorded.record("a", SAMPLE, "train")
    recorded.record("b", child_task, "train")
    observe(recorded, "a")
    observe(recorded, "b", reward=1.0)
    recorded.flush()

    tree = recorded.group(SAMPLE, "train")
    assert not [issue for issue in tree.issues if "Unconfirmed execution prefix" in issue], tree.issues
    branches = [edge for edge in tree.edges if edge.kind == "branch"]
    assert len(branches) == 1
    assert branches[0].source_node_id == planner_node.node_id
    assert branches[0].target_node_id == child_recorder.trace.entry_ids[0]
    assert branches[0].window_id == plan.meta["window_id"]
    assert branches[0].snapshot_ref == plan.meta["snapshot_ref"]
    assert branches[0].site_id == "after_planner"


def test_centralized_child_without_resume_confirmation_stays_unconfirmed(tmp_path, monkeypatch):
    """Negative: no resume_applied → issue + no fabricated branch edge."""
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    from workflow.execution_recording import ExecutionRecorder
    from workflow.rollout_tree import RolloutTreeArchive

    spec, _site = _centralized_spec()
    parent_task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1"}
    recorder = ExecutionRecorder.from_task(parent_task)
    raw = _run_parent(parent_task, spec, recorder=recorder)
    planner_event = next(e for e in raw.window_events if e.get("agent_id") == "planner")
    planner_node = next(n for n in recorder.trace.nodes if n.agent_id == "planner")
    recorder.complete()

    # Child never applies its resume (old centralized behaviour).
    child_task = {
        **SAMPLE, "_rollout_id": "b", "_attempt_id": "at1",
        "resume_messages": planner_event["messages"], "resume_parent_id": "a",
        "is_branch": True, "window_id": planner_event["event_id"],
        "snapshot_ref": planner_event["snapshot_ref"],
        "fork_node_ids": [planner_node.node_id], "source_attempt_id": "at1",
    }
    child_recorder = ExecutionRecorder.from_task(child_task)
    # Deliberately skip child run; simulate only a completed trace prefix-less.
    child_recorder.trace.resume_applied = False
    child_recorder.complete()

    recorded = RolloutTreeArchive(root, "exp", "123456abcdef", schema_version=3)
    recorded.record("a", SAMPLE, "train")
    recorded.record("b", child_task, "train")
    observe(recorded, "a")
    observe(recorded, "b", reward=1.0)
    recorded.flush()

    tree = recorded.group(SAMPLE, "train")
    assert any("Unconfirmed execution prefix" in issue for issue in tree.issues)
    assert not [edge for edge in tree.edges if edge.kind == "branch"]


def test_centralized_child_without_window_records_stays_unconfirmed(tmp_path, monkeypatch):
    """Negative: parent without recorder.window rows can still not confirm."""
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    from workflow.execution_recording import ExecutionRecorder
    from workflow.rollout_tree import RolloutTreeArchive

    spec, _site = _centralized_spec()
    parent_task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1"}
    recorder = ExecutionRecorder.from_task(parent_task)
    _raw = _run_parent(parent_task, spec, recorder=recorder)
    # Emulate the pre-fix state: recorder produced nodes but never windows.
    recorder.trace.windows = {}
    recorder.complete()

    child_task = {
        **SAMPLE, "_rollout_id": "b", "_attempt_id": "at1",
        "resume_messages": [{"role": "user", "content": "question"}],
        "resume_parent_id": "a", "is_branch": True,
        "window_id": "planner:1:after_agent_turn",
        "snapshot_ref": "planner:1:after_agent_turn",
        "fork_node_ids": [], "source_attempt_id": "at1",
    }
    child_recorder = ExecutionRecorder.from_task(child_task)
    child_recorder.resume_applied(child_task["resume_messages"])
    child_recorder.complete()

    recorded = RolloutTreeArchive(root, "exp", "123456abcdef", schema_version=3)
    recorded.record("a", SAMPLE, "train")
    recorded.record("b", child_task, "train")
    observe(recorded, "a")
    observe(recorded, "b", reward=1.0)
    recorded.flush()

    tree = recorded.group(SAMPLE, "train")
    assert any("Unconfirmed execution prefix" in issue for issue in tree.issues)
    assert not [edge for edge in tree.edges if edge.kind == "branch"]


def test_centralized_prefix_hash_matches_between_parent_window_and_child(tmp_path, monkeypatch):
    """The parent window prefix_hash and child resume hash are same-source."""
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    from workflow.execution_recording import ExecutionRecorder
    from workflow.rollout_tree import stable_id

    spec, _site = _centralized_spec()
    parent_task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1"}
    recorder = ExecutionRecorder.from_task(parent_task)
    raw = _run_parent(parent_task, spec, recorder=recorder)
    planner_event = next(e for e in raw.window_events if e.get("agent_id") == "planner")
    window = recorder.trace.windows[planner_event["event_id"]]

    # Child resume messages come straight from the event (same list).
    resume_messages = planner_event["messages"]
    child_recorder = ExecutionRecorder(root, "b", "at1", resume={
        "source_rollout_id": "a", "source_attempt_id": "at1",
        "fork_node_ids": window["fork_node_ids"],
        "window_id": planner_event["event_id"],
        "snapshot_ref": planner_event["snapshot_ref"],
    })
    child_recorder.resume_applied(resume_messages)
    assert child_recorder.trace.resume["prefix_hash"] == window["prefix_hash"]
    assert window["prefix_hash"] == stable_id(resume_messages)
