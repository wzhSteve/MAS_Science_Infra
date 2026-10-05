"""CPU-only regression for real LangGraph nodes, failure records and empty batches."""

import ast
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workflow.agent_tracing import agent_node_name, agent_span_pattern
from workflow.archive import Archive
from workflow.contracts import RolloutTree, RolloutTreeNode
from workflow.rollout_results import RunnerResult, merge_observation, result_annotation
from workflow.runtime import LLMConfig, run_episode
from workflow.spec import MASSpec
from rl.hooks.training_batch import require_training_samples


@pytest.mark.parametrize("agent_id", ["hub", "planner", "a:b", "a|b", "a%3Ab", "agent.+[1]", "验证"])
def test_real_tir_graph_uses_the_same_legal_name_as_adapter(agent_id):
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage
    from tir_agent import TirAgent

    names = []

    class TraceNames(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            names.append(kwargs.get("name"))

    agent = TirAgent.__new__(TirAgent)
    agent.agent_id = agent_id
    agent.call_model = lambda state: {**state, "messages": [AIMessage(content="<answer>2</answer>")]}
    agent.should_continue = lambda state: "end"
    agent.call_tools = agent.request_finalize = agent.resume_react = lambda state: state
    # Use the actual graph() method and installed LangGraph, without initializing an LLM.
    graph = agent.graph()
    result = graph.invoke({"messages": []}, {"callbacks": [TraceNames()]})
    name = agent_node_name(agent_id)
    assert ":" not in name and "|" not in name
    assert name in names
    assert result["messages"][0].content == "<answer>2</answer>"
    pattern = agent_span_pattern([agent_id])
    assert re.search(pattern, name)
    assert not re.search(pattern, agent_node_name("frozen_" + agent_id))
    assert not re.search(pattern, "tools")


def test_agent_window_span_emits_mas_agent_name(monkeypatch):
    """user_space windows must tag spans with agent.name == agent_node_name(id)."""
    import sys
    import types

    from rl_agent_span import agent_window_span

    captured: list = []

    class _CM:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _operation(**attrs):
        captured.append(attrs)
        return _CM()

    agl_mod = types.ModuleType("agentlightning")
    agl_mod.operation = _operation  # type: ignore[attr-defined]
    tracer_pkg = types.ModuleType("agentlightning.tracer")
    tracer_base = types.ModuleType("agentlightning.tracer.base")
    tracer_base.get_active_tracer = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentlightning", agl_mod)
    monkeypatch.setitem(sys.modules, "agentlightning.tracer", tracer_pkg)
    monkeypatch.setitem(sys.modules, "agentlightning.tracer.base", tracer_base)

    with agent_window_span("planner"):
        pass
    assert captured == [{"agent.name": "agent__planner"}]
    pattern = agent_span_pattern(["planner"])
    assert re.search(pattern, captured[0]["agent.name"])
    assert not re.search(pattern, agent_node_name("u_agentflow__executor"))


def test_agent_window_span_is_noop_without_tracer(monkeypatch):
    import sys
    import types

    from rl_agent_span import agent_window_span

    called = []

    agl_mod = types.ModuleType("agentlightning")
    agl_mod.operation = lambda **kw: called.append(kw)  # type: ignore[attr-defined]
    tracer_pkg = types.ModuleType("agentlightning.tracer")
    tracer_base = types.ModuleType("agentlightning.tracer.base")
    tracer_base.get_active_tracer = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentlightning", agl_mod)
    monkeypatch.setitem(sys.modules, "agentlightning.tracer", tracer_pkg)
    monkeypatch.setitem(sys.modules, "agentlightning.tracer.base", tracer_base)

    with agent_window_span("planner"):
        pass
    assert called == []


def test_node_encoding_is_injective_and_scope_stays_exact():
    assert agent_node_name("hub") == "agent__hub"
    assert agent_node_name("a:b") != agent_node_name("a%3Ab")
    pattern = agent_span_pattern(["hub", "planner", "hub"])
    assert re.search(pattern, agent_node_name("hub"))
    assert re.search(pattern, agent_node_name("planner"))
    assert not re.search(pattern, agent_node_name("verifier"))
    assert not re.search(pattern, "agent__hub_extra")
    assert agent_span_pattern([]) is None


@pytest.mark.parametrize("stage", ["graph_build", "graph_execution"])
def test_episode_failure_is_logged_redacted_and_round_trips_to_tree(monkeypatch, caplog, stage):
    from tir_agent import TirAgent

    secret = "test-private-key"
    failure = ValueError(f"bad node ':' api_key={secret} https://user:pass@example.test/v1?token=private")

    def fail(*args, **kwargs):
        raise failure

    fake = SimpleNamespace(
        system_prompt="system", graph=fail if stage == "graph_build"
        else lambda: SimpleNamespace(invoke=fail),
    )
    monkeypatch.setattr(TirAgent, "from_spec", classmethod(lambda cls, *args, **kwargs: fake))
    with caplog.at_level(logging.ERROR):
        raw = run_episode(
            {"question": "q", "_rollout_id": "ro-test", "_attempt_id": "at-test"},
            LLMConfig(endpoint="http://unused", model="unused", api_key=secret),
            Archive(), spec=MASSpec(hub={"skills": []}, tools=[]),
        )
    assert raw.error_details.stage == stage
    assert raw.error_details.error_type == "ValueError"
    assert secret not in raw.error and "user:pass" not in raw.error and "token=private" not in raw.error
    assert "ro-test" in caplog.text and "at-test" in caplog.text and stage in caplog.text
    assert secret not in caplog.text
    encoded = result_annotation("ro-test", "at-test", None, -1, False, True, error_details=raw.error_details)
    result = RunnerResult.model_validate_json(encoded)
    node = RolloutTreeNode(node_id="ro-test")
    tree = RolloutTree(tree_id="test", nodes=[node])
    merge_observation(tree, node, attempt_id="at-test", sequence=1, store_status="succeeded", results=[result])
    assert tree.outcomes["ro-test"]["error_details"]["stage"] == stage
    assert tree.outcomes["ro-test"]["reward"] == -1
    assert node.status == "failed" and node.store_status == "succeeded"


@pytest.mark.parametrize("triplets", [[], [SimpleNamespace(prompt={"token_ids": []}, response={"token_ids": []})]])
def test_empty_batch_stops_at_our_daemon_before_upstream_tensor_construction(triplets):
    source = ast.parse((ROOT.parent / "rl" / "hooks" / "daemon.py").read_text(encoding="utf-8"))
    cls = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == "TirAgentModeDaemon")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "get_train_data_batch")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"require_training_samples": require_training_samples}
    exec(compile(ast.fix_missing_locations(module), "daemon.py", "exec"), namespace)
    daemon = SimpleNamespace(
        _completed_rollouts_v0={"ro-test": SimpleNamespace(triplets=triplets)},
        adapter=SimpleNamespace(agent_match=agent_span_pattern(["hub"])),
    )
    # Extracted super() is intentionally unavailable: the guard must fail before reaching it.
    with pytest.raises(RuntimeError, match="No usable training samples.*1 rollouts"):
        namespace["get_train_data_batch"](daemon, 2048, 512, "cpu", 0)


def test_valid_batch_is_not_filtered_reweighted_or_modified():
    triplet = SimpleNamespace(prompt={"token_ids": [1, 2]}, response={"token_ids": [3]}, reward=-1)
    rollouts = {"bad": SimpleNamespace(triplets=[]), "good": SimpleNamespace(triplets=[triplet])}
    require_training_samples(rollouts, agent_span_pattern(["hub"]))
    assert list(rollouts) == ["bad", "good"]
    assert rollouts["good"].triplets == [triplet]
    assert triplet.reward == -1
