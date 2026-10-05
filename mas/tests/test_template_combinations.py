"""Template-as-test: every GraphPalette YAML runs a MockWindowLLM episode.

Templates live in ``mas/specs/templates/`` and are the same objects the UI
palette loads. Search / think kernels are mocked so tests never hit the
network or a local vLLM.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import patch

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workflow.archive import Archive, register_archive  # noqa: E402
from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode  # noqa: E402
from workflow.compiler import compile_spec, trainable_agents  # noqa: E402
from workflow.memory import MemoryStore  # noqa: E402
from workflow.router import reset_round_robin  # noqa: E402
from workflow.spec import MASSpec  # noqa: E402
from workflow.templates import TEMPLATE_ORDER, load_template_workflow  # noqa: E402

SEARCH_IDS = ("wikipedia_search", "bing_search", "web_fetch")
NETWORK_TOOL_IDS = SEARCH_IDS + ("think",)

VERIFY_OK = json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})
VERIFY_FAIL = json.dumps({"ok": False, "reason": "wrong answer", "step_conclusion": "INCOMPLETE", "slot_updates": []})


def _spec(template_id: str) -> MASSpec:
    return MASSpec.model_validate(load_template_workflow(template_id))


def _plan(next_val: Any, args: Any, *, done: bool, sub_goal: str = "step") -> str:
    return json.dumps({"next": next_val, "args": args, "sub_goal": sub_goal, "done": done})


@contextmanager
def _mock_tool_invokes(ids: Iterable[str], tracker: Optional[List] = None):
    """Replace ``ToolAgent.invoke`` so search/think never touch the network/LLM."""
    from tools.tool_agents import TOOL_AGENTS

    records: List = tracker if tracker is not None else []

    def _make(name: str):
        def _invoke(args, **_kwargs):
            records.append((name, dict(args or {})))
            return f"mocked:{name}"

        return _invoke

    with ExitStack() as stack:
        for aid in ids:
            ta = TOOL_AGENTS.get(aid)
            if ta is None:
                continue
            stack.enter_context(patch.object(ta, "invoke", side_effect=_make(aid)))
        yield records


def _run(spec: MASSpec, win: MockWindowLLM, *, task_id: str = "tmpl"):
    compiled = compile_spec(spec)
    if not compiled.ok:
        raise AssertionError(f"compile failed: {compiled.reason}")
    arch = register_archive(Archive())
    raw = run_centralized_episode(
        {"id": task_id, "question": "template episode?"},
        None,
        arch,
        spec,
        MemoryStore(),
        compiled,
        window_llm=win,
    )
    kinds = [m.get("kind") for m in raw.messages]
    if "tool_calls" in kinds:
        raise AssertionError("episode used TirAgent tool_calls instead of AgentMessage")
    return raw, compiled, kinds


def _window_ids(raw) -> List[str]:
    return [str(e.get("agent_id") or "") for e in raw.window_events]


def _msgs(raw, kind: str) -> List[Dict[str, Any]]:
    return [m for m in raw.messages if m.get("kind") == kind]


class TestTemplatesCompile(unittest.TestCase):
    def test_every_palette_template_compiles(self):
        from workflow.templates import list_palette_templates

        listed = [t["id"] for t in list_palette_templates()]
        self.assertEqual(listed, list(TEMPLATE_ORDER))
        for tid in TEMPLATE_ORDER:
            spec = _spec(tid)
            self.assertEqual(spec.topology, "centralized")
            self.assertEqual(spec.entry_agent, "planner")
            kinds = {a.id: a.kind for a in spec.agents}
            self.assertEqual(kinds.get("planner"), "planner")
            if tid == "tir_five_tools":
                self.assertNotIn("verifier", kinds)
            else:
                self.assertEqual(kinds.get("verifier"), "verifier")
            self.assertNotIn("hub", kinds)
            edge_kinds = {e.kind for e in spec.edges}
            allowed = {"route", "message", "feedback"}
            if tid == "tir_five_tools":
                allowed.add("tool_call")
            self.assertTrue(edge_kinds <= allowed)
            compiled = compile_spec(spec)
            self.assertTrue(compiled.ok, f"{tid}: {compiled.reason}")
            self.assertTrue(compiled.routers)
            self.assertEqual(compiled.entry_agent, "planner")
            names = set(trainable_agents(spec))
            agent_ids = {a.id for a in spec.agents}
            for agent in spec.agents:
                self.assertNotIn(agent.id, ("hub", "executor"), tid)
                self.assertNotIn(agent.kind, ("hub", "executor"), tid)
                if agent.kind == "tool":
                    self.assertFalse(agent.trainable, tid)
                    self.assertNotIn(agent.id, names, tid)
                elif agent.trainable:
                    self.assertIn(agent.id, names, tid)
            route_targets = {e.target for e in spec.edges if e.source == "planner" and e.kind == "route"}
            self.assertEqual(set(compiled.fan_out.get("planner") or []), route_targets, tid)
            if tid == "score_robin":
                self.assertEqual(route_targets, {"route_score", "route_robin"})
            elif tid == "fanout_parallel":
                self.assertEqual(route_targets, {"route_a", "route_b"})
            else:
                self.assertEqual(len(route_targets), 1, tid)
            for router in compiled.routers.values():
                self.assertTrue(router.candidates, router.id)
                for candidate in router.candidates:
                    self.assertIn(candidate, agent_ids, f"{tid}:{candidate}")
                if tid != "tir_five_tools":
                    self.assertEqual(compiled.message_out.get(router.id), "verifier", tid)


class TestEveryTemplateEpisode(unittest.TestCase):
    def setUp(self) -> None:
        reset_round_robin()

    def tearDown(self) -> None:
        reset_round_robin()

    def test_every_template_completes_one_turn(self):
        plans = {
            "centralized": _plan("wikipedia_search", {"query": "q"}, done=True, sub_goal="search"),
            "pev_python": _plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute"),
            "pev_search": _plan("wikipedia_search", {"query": "q"}, done=True, sub_goal="search"),
            "fanout_parallel": _plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute"),
            "blank_set": _plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute"),
            "score_robin": _plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute"),
            "tir_five_tools": _plan("wikipedia_search", {"query": "q"}, done=True, sub_goal="search"),
        }
        selected = {
            "centralized": "wikipedia_search",
            "pev_python": "python_coder",
            "pev_search": "wikipedia_search",
            "fanout_parallel": "python_coder",
            "blank_set": "python_coder",
            "score_robin": "python_coder",
            "tir_five_tools": "wikipedia_search",
        }
        self.assertEqual(set(plans), set(TEMPLATE_ORDER))
        for tid in TEMPLATE_ORDER:
            spec = _spec(tid)
            win = MockWindowLLM({"planner": [plans[tid]], "verifier": [VERIFY_OK]})
            with _mock_tool_invokes(NETWORK_TOOL_IDS):
                raw, _compiled, kinds = _run(spec, win, task_id=f"one-turn-{tid}")
            self.assertFalse(raw.error, f"{tid}: {raw.error}")
            for kind in ("plan_step", "tool_invoke", "tool_result", "verify"):
                self.assertIn(kind, kinds, tid)
            self.assertNotIn("tool_calls", kinds, tid)
            ids = _window_ids(raw)
            self.assertIn("planner", ids, tid)
            self.assertIn("verifier", ids, tid)
            self.assertIn(selected[tid], ids, tid)
            self.assertNotIn(selected[tid], trainable_agents(spec), tid)
            for router in spec.routers:
                self.assertIn(router.id, ids, tid)


class TestPevPython(unittest.TestCase):
    def test_episode_python_kernel(self):
        spec = _spec("pev_python")
        win = MockWindowLLM({
            "planner": [_plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute")],
            "verifier": [VERIFY_OK],
        })
        raw, _compiled, kinds = _run(spec, win, task_id="pev_python")
        self.assertIn("plan_step", kinds)
        self.assertIn("tool_invoke", kinds)
        self.assertIn("tool_result", kinds)
        self.assertIn("verify", kinds)
        ids = _window_ids(raw)
        self.assertIn("planner", ids)
        self.assertIn("route_exec", ids)
        self.assertIn("python_coder", ids)
        self.assertIn("verifier", ids)
        tool = _msgs(raw, "tool_result")[0]
        self.assertEqual(tool["src"], "python_coder")
        self.assertTrue(tool["payload"]["ok"])
        self.assertIn("42", str(tool["payload"].get("output") or ""))

    def test_illegal_next_feedback_no_invoke(self):
        spec = _spec("pev_python")
        win = MockWindowLLM({
            "planner": [
                _plan("ghost", {}, done=False, sub_goal="x"),
                json.dumps({"next": "", "args": {}, "sub_goal": "done", "done": True, "answer": "stop"}),
            ],
            "verifier": [VERIFY_OK],
        })
        raw, _c, _k = _run(spec, win, task_id="pev_python_illegal")
        first: List[str] = []
        for m in raw.messages:
            first.append(m.get("kind"))
            if m.get("kind") == "feedback":
                break
        self.assertEqual(first, ["plan_step", "feedback"])
        self.assertNotIn("tool_invoke", first)
        self.assertNotIn("tool_result", first)

    def test_verifier_fail_replans(self):
        spec = _spec("pev_python")
        win = MockWindowLLM({
            "planner": [
                _plan("python_coder", {"code": "result=1"}, done=False, sub_goal="compute"),
                _plan("python_coder", {"code": "result=42"}, done=True, sub_goal="compute"),
            ],
            "verifier": [VERIFY_FAIL, VERIFY_OK],
        })
        raw, _c, _k = _run(spec, win, task_id="pev_python_retry")
        verifies = _msgs(raw, "verify")
        self.assertEqual(len(verifies), 2)
        self.assertFalse(verifies[0]["payload"]["ok"])
        self.assertTrue(verifies[1]["payload"]["ok"])
        self.assertGreaterEqual(len(_msgs(raw, "feedback")), 1)
        self.assertEqual(len(_msgs(raw, "plan_step")), 2)
        self.assertEqual(len(_msgs(raw, "tool_invoke")), 2)


class TestPevSearch(unittest.TestCase):
    def test_three_search_agents_mocked(self):
        spec = _spec("pev_search")
        win = MockWindowLLM({
            "planner": [_plan(
                ["wikipedia_search", "bing_search", "web_fetch"],
                [
                    {"query": "wiki-q"},
                    {"query": "bing-q"},
                    {"query": "fetch-q", "url": "https://example.com"},
                ],
                done=True,
                sub_goal="search",
            )],
            "verifier": [VERIFY_OK],
        })
        tracker: List = []
        with _mock_tool_invokes(SEARCH_IDS, tracker):
            raw, _c, kinds = _run(spec, win, task_id="pev_search")
        invoked = {name for name, _args in tracker}
        self.assertEqual(invoked, set(SEARCH_IDS))
        self.assertNotIn("tool_calls", kinds)
        results = _msgs(raw, "tool_result")
        srcs = {m["src"] for m in results}
        self.assertEqual(srcs, set(SEARCH_IDS))
        self.assertTrue(all(m["payload"]["ok"] for m in results))
        self.assertEqual(raw.n_search, 3)
        by_name = {name: args for name, args in tracker}
        self.assertEqual(by_name["wikipedia_search"]["query"], "wiki-q")
        self.assertEqual(by_name["bing_search"]["query"], "bing-q")
        self.assertEqual(by_name["web_fetch"]["url"], "https://example.com")

    def test_web_fetch_missing_url_errors_without_kernel(self):
        spec = _spec("pev_search")
        win = MockWindowLLM({
            "planner": [_plan("web_fetch", {"query": "no-url"}, done=True, sub_goal="web")],
            "verifier": [VERIFY_OK],
        })
        tracker: List = []
        with _mock_tool_invokes(("web_fetch",), tracker):
            raw, _c, _k = _run(spec, win, task_id="pev_search_nourl")
        self.assertEqual(tracker, [])
        tool = _msgs(raw, "tool_result")[0]
        self.assertEqual(tool["src"], "web_fetch")
        self.assertFalse(tool["payload"]["ok"])
        self.assertEqual(tool["payload"]["evidence_type"], "ERROR")


class TestCentralizedFiveTools(unittest.TestCase):
    def test_router_can_select_every_tool_id(self):
        spec = _spec("centralized")
        sequence = [
            ("wikipedia_search", {"query": "q"}),
            ("bing_search", {"query": "q"}),
            ("web_fetch", {"query": "q", "url": "https://example.com"}),
            ("python_coder", {"code": "result=42"}),
            ("think", {"text": "reason"}),
        ]
        planner = [
            _plan(tid, args, done=(i == len(sequence) - 1), sub_goal=tid)
            for i, (tid, args) in enumerate(sequence)
        ]
        win = MockWindowLLM({
            "planner": planner,
            "verifier": [VERIFY_OK] * len(sequence),
        })
        tracker: List = []
        with _mock_tool_invokes(NETWORK_TOOL_IDS, tracker):
            raw, compiled, kinds = _run(spec, win, task_id="centralized")
        self.assertIn("python_coder", compiled.routers["route_exec"].candidates)
        for tid, _args in sequence:
            self.assertIn(tid, compiled.routers["route_exec"].candidates)
        invokers = _msgs(raw, "tool_invoke")
        self.assertEqual([m["dst"] for m in invokers], [tid for tid, _ in sequence])
        results = _msgs(raw, "tool_result")
        self.assertEqual([m["src"] for m in results], [tid for tid, _ in sequence])
        self.assertEqual(len(_msgs(raw, "plan_step")), 5)
        self.assertEqual(len(_msgs(raw, "verify")), 5)
        self.assertNotIn("tool_calls", kinds)
        mocked = {name for name, _ in tracker}
        self.assertEqual(mocked, set(SEARCH_IDS) | {"think"})


class TestFanoutParallel(unittest.TestCase):
    def test_two_routers_parallel_hops(self):
        spec = _spec("fanout_parallel")
        win = MockWindowLLM({
            "planner": [_plan(
                ["python_coder", "think"],
                [{"code": "result=1"}, {"text": "reason"}],
                done=True,
                sub_goal="parallel",
            )],
            "verifier": [VERIFY_OK],
        })
        with _mock_tool_invokes(("think",)):
            raw, compiled, _k = _run(spec, win, task_id="fanout")
        self.assertEqual(set(compiled.fan_out.get("planner") or []), {"route_a", "route_b"})
        invokes = _msgs(raw, "tool_invoke")
        self.assertEqual(len(invokes), 2)
        dsts = sorted(m["dst"] for m in invokes)
        self.assertEqual(dsts, ["python_coder", "think"])
        results = _msgs(raw, "tool_result")
        self.assertEqual(sorted(m["src"] for m in results), ["python_coder", "think"])
        ids = _window_ids(raw)
        self.assertIn("route_a", ids)
        self.assertIn("route_b", ids)
        self.assertIn("python_coder", ids)
        self.assertIn("think", ids)


class TestBlankSet(unittest.TestCase):
    def test_blank_is_router_candidate_not_pipeline(self):
        spec = _spec("blank_set")
        compiled_pre = compile_spec(spec)
        self.assertEqual(
            list(compiled_pre.routers["route_exec"].candidates),
            ["python_coder", "summarizer"],
        )
        win = MockWindowLLM({
            "planner": [
                _plan("python_coder", {"code": "result=42"}, done=False, sub_goal="code"),
                _plan("summarizer", {"input": "42"}, done=True, sub_goal="summarize"),
            ],
            "summarizer": [json.dumps({"output": "the answer is 42"})],
            "verifier": [VERIFY_OK, VERIFY_OK],
        })
        raw, _c, _k = _run(spec, win, task_id="blank_set")
        invokes = _msgs(raw, "tool_invoke")
        self.assertEqual([m["dst"] for m in invokes], ["python_coder", "summarizer"])
        results = _msgs(raw, "tool_result")
        srcs = [m["src"] for m in results]
        self.assertEqual(srcs, ["python_coder", "summarizer"])
        ids = _window_ids(raw)
        self.assertIn("summarizer", ids)
        self.assertFalse(any(i.startswith("blank:") for i in ids))
        by_src = {m["src"]: m for m in results}
        self.assertTrue(by_src["summarizer"]["payload"]["ok"])
        self.assertEqual(by_src["summarizer"]["src"], "summarizer")
        # blank is a router hop, not a post-tool pipeline chained off python_coder
        self.assertNotEqual(by_src["summarizer"]["trace_ref"], by_src["python_coder"]["msg_id"])


class TestScoreRobin(unittest.TestCase):
    def setUp(self) -> None:
        reset_round_robin()

    def tearDown(self) -> None:
        reset_round_robin()

    def test_score_picks_highest_blank(self):
        spec = _spec("score_robin")
        win = MockWindowLLM({
            "planner": [_plan(
                "python_coder",
                {"code": "result=1", "text": "note"},
                done=True,
                sub_goal="score",
            )],
            "judge": [json.dumps({"score": 0.2}), json.dumps({"score": 0.9})],
            "expert_a": [json.dumps({"output": "A"})],
            "expert_b": [json.dumps({"output": "B-cand"}), json.dumps({"output": "B-win"})],
            "verifier": [VERIFY_OK],
        })
        with _mock_tool_invokes(("think",)):
            raw, compiled, _k = _run(spec, win, task_id="score")
        self.assertEqual(compiled.routers["route_score"].strategy, "score")
        self.assertEqual(compiled.routers["route_score"].scorer, "judge")
        score_ev = next(e for e in raw.window_events if e.get("agent_id") == "route_score")
        self.assertEqual(score_ev["metrics"]["selected"], ["expert_b"])
        self.assertEqual(score_ev["metrics"]["strategy"], "score")
        scores = score_ev["metrics"].get("scores") or {}
        self.assertGreater(float(scores.get("expert_b", 0)), float(scores.get("expert_a", 0)))
        result_srcs = {m["src"] for m in _msgs(raw, "tool_result")}
        self.assertIn("expert_b", result_srcs)
        self.assertNotIn("expert_a", result_srcs)
        ids = _window_ids(raw)
        self.assertIn("expert_b", ids)
        self.assertFalse(any(i.startswith("blank:") for i in ids))
        robin_ev = next(e for e in raw.window_events if e.get("agent_id") == "route_robin")
        self.assertEqual(robin_ev["metrics"]["selected"], ["python_coder"])

    def test_round_robin_rotates_python_then_think(self):
        spec = _spec("score_robin")
        win = MockWindowLLM({
            "planner": [
                _plan("python_coder", {"code": "result=1", "text": "t1"}, done=False, sub_goal="r1"),
                _plan("think", {"code": "result=2", "text": "t2"}, done=True, sub_goal="r2"),
            ],
            "judge": [
                json.dumps({"score": 0.1}), json.dumps({"score": 0.9}),
                json.dumps({"score": 0.1}), json.dumps({"score": 0.9}),
            ],
            "verifier": [VERIFY_OK, VERIFY_OK],
        })
        with _mock_tool_invokes(("think",)):
            raw, compiled, _k = _run(spec, win, task_id="robin")
        self.assertEqual(compiled.routers["route_robin"].strategy, "round_robin")
        robin = [e for e in raw.window_events if e.get("agent_id") == "route_robin"]
        self.assertEqual(len(robin), 2)
        self.assertEqual(robin[0]["metrics"]["selected"], ["python_coder"])
        self.assertEqual(robin[1]["metrics"]["selected"], ["think"])
        invoke_dsts = [m["dst"] for m in _msgs(raw, "tool_invoke")]
        self.assertIn("python_coder", invoke_dsts)
        self.assertIn("think", invoke_dsts)


if __name__ == "__main__":
    unittest.main()
