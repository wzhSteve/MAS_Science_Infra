"""Fast execution-tree checks: real recording, shared prefixes and bounded HTTP."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from workflow.contracts import RolloutTree
from workflow.execution_recording import ExecutionRecorder, attempt_directory
from workflow.rollout_results import RunnerResult
from workflow.rollout_tree import RolloutTreeArchive
from science_infra.control import rollout_trees as api


SAMPLE = {"data_id": "g", "question": "question", "id": "q"}


def observe(archive, rid, *, attempt="at1", reward=0.0, status="succeeded", sequence=1):
    result = RunnerResult(rollout_id=rid, attempt_id=attempt, answer="answer",
                          reward=reward, format_ok=True)
    archive.observe(rid, SAMPLE, "train", attempt_id=attempt, sequence=sequence,
                    store_status=status, attempt_status=status,
                    results=[result] if status == "succeeded" else [])


def make_tree(root):
    archive = RolloutTreeArchive(root, "exp", "123456abcdef", schema_version=3)
    archive.record("a", SAMPLE, "train")
    recorder = ExecutionRecorder(root, "a", "at1")
    first = recorder.begin("hub", "agent", 1, [{"role": "user", "content": "question"}])
    recorder.finish(first, "use python", reasoning="visible reasoning")
    tool = recorder.begin("execute_python", "tool", 1, {"code": "1+1"})
    recorder.finish(tool, "2", observation="2")
    recorder.window("w", snapshot_ref="arch:snap", messages=["2"])
    last = recorder.begin("hub", "agent", 2, ["2"])
    recorder.finish(last, "answer")
    recorder.complete()
    observe(archive, "a")
    child_sample = {**SAMPLE, "resume_parent_id": "a", "window_id": "w", "snapshot_ref": "arch:snap"}
    archive.record("b", child_sample, "train")
    child = ExecutionRecorder(root, "b", "at1", resume={
        "source_rollout_id": "a", "source_attempt_id": "at1", "fork_node_ids": [tool],
        "window_id": "w", "snapshot_ref": "arch:snap",
    })
    child.resume_applied(["2"])
    child_node = child.begin("hub", "agent", 2, ["2"])
    child.finish(child_node, "other answer")
    child.complete()
    observe(archive, "b", reward=1.0)
    archive.flush()
    return archive, (first, tool, last, child_node)


def test_shared_prefix_is_not_parent_suffix_and_results_survive(tmp_path):
    archive, (first, tool, last, child) = make_tree(tmp_path)
    tree = archive.group(SAMPLE, "train")
    assert tree.schema_version == 3 and len(tree.rollouts) == 2
    assert len([node for node in tree.nodes if node.kind == "execution"]) == 4
    assert len(tree.leaves()) == 2
    assert [(edge.source_node_id, edge.target_node_id) for edge in tree.edges if edge.kind == "branch"] == [(tool, child)]
    child_result = tree.rollouts[1].result_node_id
    assert set(tree.path_to_root(child_result)) == {tree.nodes[0].node_id, first, tool, child, child_result}
    assert last not in tree.path_to_root(child_result)
    assert tree.outcomes[tree.rollouts[0].result_node_id]["reward"] == 0.0
    revision = tree.revision
    observe(archive, "b", reward=1.0)
    archive.flush()
    assert archive.group(SAMPLE, "train").revision == revision
    restored = RolloutTreeArchive(tmp_path, "exp", "123456abcdef").group(SAMPLE, "train")
    assert restored.model_dump() == archive.group(SAMPLE, "train").model_dump()
    assert RolloutTreeArchive._summary(restored)["rollout_count"] == 2


def test_parallel_tools_join_without_becoming_sampling_branches(tmp_path):
    archive = RolloutTreeArchive(tmp_path, "exp", "123456abcdef", schema_version=3)
    archive.record("a", SAMPLE, "train")
    recorder = ExecutionRecorder(tmp_path, "a", "at1")
    first = recorder.begin("hub", "agent", 1, "input")
    recorder.finish(first, "tools")
    tools = [recorder.begin(name, "tool", 1, {}, parents=[first]) for name in ("python", "search")]
    for tool in tools:
        recorder.finish(tool, "result")
    recorder.join(tools)
    last = recorder.begin("hub", "agent", 2, "results")
    recorder.finish(last, "answer")
    recorder.complete()
    observe(archive, "a")
    tree = archive.group(SAMPLE, "train")
    assert {e.source_node_id for e in tree.edges if e.target_node_id == last} == set(tools)
    assert not any(e.kind == "branch" for e in tree.edges)
    RolloutTree.model_validate(tree.model_dump())


def test_recorded_input_reasoning_redaction_and_retry(tmp_path):
    recorder = ExecutionRecorder(tmp_path, "a", "at1")
    node = recorder.begin("hub", "agent", 1, "original input")
    recorder.request_error(node, ValueError("api_key=private"), retry_input="short input")
    recorder.finish(node, "x" * 140_000, reasoning=None)
    data = json.loads((recorder.directory / f"{node}.json").read_text(encoding="utf-8"))
    assert data["input"] == "short input" and data["previous_inputs"] == ["original input"]
    assert data["input_changed"] and data["content_truncated"]
    assert data["reasoning"] is None
    assert "private" not in json.dumps(data)
    assert recorder.trace.nodes[0].status == "succeeded"


def test_source_attempt_survives_retry_and_unconfirmed_prefix_is_not_faked(tmp_path):
    archive, ids = make_tree(tmp_path)
    observe(archive, "a", attempt="at2", sequence=2, status="running")
    archive.flush()
    tree = archive.group(SAMPLE, "train")
    assert any(e.kind == "branch" and e.source_node_id == ids[1] for e in tree.edges)
    trace_path = attempt_directory(tmp_path, "b", "at1") / "index.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["resume_applied"] = False
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    observe(archive, "b", reward=1.0)
    tree = archive.group(SAMPLE, "train")
    assert any("Unconfirmed execution prefix" in issue for issue in tree.issues)
    assert not any(e.target_node_id == ids[3] and e.kind == "branch" for e in tree.edges)


def test_v3_api_pagination_node_lookup_and_details(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    archive, ids = make_tree(run_dir / "rollout-trees")
    tree = archive.group(SAMPLE, "train")

    def registered(exp, run):
        if (exp, run) != ("exp", "123456abcdef"):
            raise HTTPException(404, "wrong run")
        return {"state": "succeeded"}

    monkeypatch.setattr(api, "training_run", registered)
    monkeypatch.setattr(api.PROCS, "run_dir", lambda *args: run_dir)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        base = f"/api/rl/runs/123456abcdef/rollout-trees/{tree.tree_id}"
        response = client.get(base, params={"experiment_id": "exp", "limit": 1, "node_id": ids[3]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["schema_version"] == 3
        assert ids[1] in {n["node_id"] for n in body["tree"]["nodes"]}
        assert ids[2] not in {n["node_id"] for n in body["tree"]["nodes"]}
        detail = client.get(f"{base}/nodes/{ids[0]}?experiment_id=exp")
        assert detail.json()["detail"]["reasoning"] == "visible reasoning"
        assert client.get(f"{base}/nodes/{ids[0]}?experiment_id=exp",
                          headers={"If-None-Match": detail.headers["etag"]}).status_code == 304
        assert client.get(f"{base}/nodes/{ids[0]}?experiment_id=other").status_code == 404
        diagnostic = client.get(base, params={"experiment_id": "exp", "diagnose": True})
        assert diagnostic.json()["diagnostics"]["status"] == "unsupported"


def test_actual_graph_records_model_tool_and_model_without_network(tmp_path):
    from langchain_core.messages import AIMessage, HumanMessage
    from tir_agent import TirAgent

    responses = iter([
        AIMessage(content="calculate", tool_calls=[{"name": "execute_python", "args": {"code": "1+1"}, "id": "c"}]),
        AIMessage(content="<answer>2</answer>", additional_kwargs={"reasoning_content": "visible"}),
    ])
    class Model:
        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            return next(responses)

    agent = TirAgent.__new__(TirAgent)
    agent.execution_recorder = ExecutionRecorder(tmp_path, "a", "at1")
    agent.agent_id, agent.model_name = "hub", "test"
    agent.llm = agent.llm_finalize = Model()
    agent.max_tokens, agent.max_turns, agent.max_model_len = 128, 8, None
    agent.entropy_tokens, agent.entropy_threshold = 8, 0.15
    agent._router_by_tool, agent.routers = {}, {}
    agent._blank_adapters = {}
    agent.tool_agent_invoker = SimpleNamespace(invoke=lambda name, args: "2")
    result = agent.graph().invoke({"messages": [HumanMessage(content="1+1")], "question": "1+1"})
    assert result["messages"][-1].content == "<answer>2</answer>"
    trace = agent.execution_recorder.trace
    assert [n.agent_kind for n in trace.nodes] == ["agent", "tool", "agent"]
    kinds = [e.get("kind") for e in result["window_events"]]
    assert "after_agent_turn" in kinds
    after_tool = next(e for e in result["window_events"] if e.get("kind") == "after_tool")
    assert after_tool["metrics"]["source_attempt_id"] == "at1"
    assert after_tool["metrics"]["fork_node_ids"] == [trace.nodes[1].node_id]


def test_real_episode_window_plan_and_child_share_only_recorded_prefix(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from tir_agent import TirAgent
    from workflow.active_set import ActiveSetConfig, ActiveSetSession
    from workflow.archive import Archive
    from workflow.contracts import BranchSite
    from workflow.runtime import LLMConfig, run_episode
    from workflow.spec import MASSpec

    responses = iter([
        AIMessage(content="tool", tool_calls=[{"name": "execute_python", "args": {"code": "1+1"}, "id": "c"}]),
        AIMessage(content="<answer>2</answer>"),
        AIMessage(content="<answer>two</answer>"),
    ])
    model = SimpleNamespace()
    model.bind = lambda **kwargs: model
    model.invoke = lambda messages: next(responses)

    def from_spec(cls, spec, **kwargs):
        agent = cls.__new__(cls)
        agent.agent_id, agent.model_name = kwargs.get("agent_id") or "planner", "test"
        agent.llm = agent.llm_finalize = model
        agent.max_tokens, agent.max_turns, agent.max_model_len = 128, 8, None
        agent.entropy_tokens, agent.entropy_threshold = 8, 0.15
        agent._router_by_tool, agent.routers = {}, {}
        agent._blank_adapters = {}
        agent.tool_agent_invoker = SimpleNamespace(invoke=lambda name, args: "2")
        return agent

    monkeypatch.setattr(TirAgent, "from_spec", classmethod(from_spec))
    root = tmp_path / "rollout-trees"
    monkeypatch.setenv("TIR_ROLLOUT_TREE_DIR", str(root))
    spec = MASSpec(hub={"skills": []}, tools=["execute_python"])
    llm = LLMConfig(endpoint="http://unused", model="test")
    task = {**SAMPLE, "_rollout_id": "a", "_attempt_id": "at1", "expand_in_runner": True}
    recorder = ExecutionRecorder.from_task(task)
    archive = Archive(root_dir=str(tmp_path / "snapshots"))
    session = ActiveSetSession(ActiveSetConfig(group_budget=2, sites=[
        BranchSite(id="python", anchor={"kind": "after_tool", "agent_id": "planner", "tool_id": "execute_python"},
                   gate={"type": "always"}, fork={"beam_size": 2})
    ]), on_decision=recorder.sampling)
    generated = session.run(task, lambda t: run_episode(t, llm, archive, spec=spec, execution_recorder=recorder))
    assert len(generated.plans) == 1
    plan = generated.plans[0]
    assert plan.meta["fork_node_ids"] == [recorder.trace.nodes[1].node_id]
    assert plan.meta["source_attempt_id"] == "at1"
    assert recorder.trace.nodes[1].decision["passed"] is True
    child_task = {**SAMPLE, **plan.meta, "_rollout_id": "b", "_attempt_id": "at1",
                  "resume_parent_id": "a", "resume_messages": plan.resume_messages}
    child_recorder = ExecutionRecorder.from_task(child_task)
    raw = run_episode(child_task, llm, Archive(root_dir=str(tmp_path / "child")),
                      spec=spec, execution_recorder=child_recorder)
    assert raw.final_answer == "two"
    assert len(child_recorder.trace.nodes) == 1
    recorded = RolloutTreeArchive(root, "exp", "123456abcdef", schema_version=3)
    recorded.record("a", SAMPLE, "train")
    recorded.record("b", child_task, "train")
    observe(recorded, "a")
    observe(recorded, "b")
    tree = recorded.group(SAMPLE, "train")
    assert not tree.issues
    branches = [edge for edge in tree.edges if edge.kind == "branch"]
    assert len(branches) == 1
    assert branches[0].source_node_id == recorder.trace.nodes[1].node_id
    assert branches[0].target_node_id == child_recorder.trace.nodes[0].node_id


def test_recording_gate_does_not_draw_rng_twice():
    from workflow.sampling.core import SamplingCore, ResumableWindow
    from workflow.sampling.adapters.arpo import ArpoSamplingAdapter
    from workflow.sampling.contracts import SamplingWindow, WindowKind
    from workflow.contracts import BranchSite

    class Rng:
        calls = 0

        def random(self):
            self.calls += 1
            return 0.9

    rng = Rng()
    decisions = []
    core = SamplingCore(ArpoSamplingAdapter(), on_decision=lambda window, data: decisions.append(data))
    plans = core.plan(
        parent_rollout_id="a", sites=[BranchSite(id="s", gate={"type": "arpo"})],
        windows=[ResumableWindow(SamplingWindow(
            window_id="w", owner_agent_id="hub", kind=WindowKind.TOOL_RESULT, sequence=1,
            snapshot_ref="archive:snap", metrics={"h_root": 0.3, "h_tool": 0.3},
        ), [{"role": "tool", "content": "2"}])],
        remaining=1, depth=0, max_depth=2, default_beam=2,
        context_factory=lambda site, window: {"rng": rng},
    )
    assert not plans and rng.calls == 1
    assert len(decisions) == 1 and decisions[0]["passed"] is False


def test_live_store_reconcile_reads_runner_files_and_preserves_training_identity(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    from rl.hooks.rollout_tree import RolloutTreeRecorder

    recorder = RolloutTreeRecorder({
        "TIR_ROLLOUT_TREE_DIR": str(tmp_path), "TIR_EXPERIMENT_ID": "exp",
        "TIR_RUN_ID": "123456abcdef", "TIR_EXECUTION_TREE_VERSION": "3",
    })
    recorder.initial({"a": SAMPLE}, "train")
    execution = ExecutionRecorder(tmp_path, "a", "at1")
    node = execution.begin("hub", "agent", 1, "input")
    rollout = SimpleNamespace(
        rollout_id="a", mode="train", metadata={"data_id": "g"}, status="running",
        attempt=SimpleNamespace(attempt_id="at1", sequence_id=1, status="running", start_time=1, end_time=None),
        start_time=1, end_time=None,
    )
    store = SimpleNamespace(query_rollouts=AsyncMock(return_value=[rollout]), query_spans=AsyncMock())
    asyncio.run(recorder.reconcile(store, {"a": SAMPLE}, "train"))
    tree = recorder.archive.group(SAMPLE, "train")
    assert tree.schema_version == 3
    assert tree.nodes[1].node_id == node and tree.nodes[1].status == "running"
    assert not store.query_spans.called
    execution.finish(node, "answer")
    asyncio.run(recorder.reconcile(store, {"a": SAMPLE}, "train"))
    assert recorder.archive.group(SAMPLE, "train").nodes[1].status == "succeeded"
