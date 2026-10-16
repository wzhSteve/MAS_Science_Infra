"""Mode B-window planner-only span selection (Flow-GRPO: frozen roles never enter the batch)."""

from __future__ import annotations

import itertools
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

AF_WORKFLOW = ROOT / "experiments" / "agentflow_flow_grpo_4b" / "workflow.yaml"

try:
    from agentlightning.adapter.triplet import TracerTraceToTriplet
    from agentlightning.types import Span
except Exception:  # pragma: no cover
    TracerTraceToTriplet = None  # type: ignore

_SEQ = itertools.count()


def _span(span_id, name, parent, start, end, attrs=None):
    return Span.from_attributes(
        rollout_id="r", attempt_id="a", sequence_id=next(_SEQ), trace_id="t",
        span_id=span_id, parent_id=parent, name=name, attributes=attrs or {},
        start_time=start, end_time=end,
    )


def _agent(span_id, agent_name, parent, start, end):
    return _span(span_id, "agentlightning.operation", parent, start, end, {"agent.name": agent_name})


def _llm(span_id, parent, start, end, marker):
    return _span(span_id, "openai.chat.completion", parent, start, end, {
        "prompt_token_ids": [marker, marker], "response_token_ids": [marker + 1],
        "gen_ai.response.id": span_id,
    })


@unittest.skipIf(TracerTraceToTriplet is None, "agentlightning missing")
class TestWindowAgentMatch(unittest.TestCase):
    def test_pattern_enabled_for_window_graph(self) -> None:
        from train_tir_agent import window_agent_match

        pat = window_agent_match(str(AF_WORKFLOW), ["planner"])
        self.assertEqual(pat, r"^(?:agent__planner)$")

    def test_disabled_without_window_graph_or_by_env(self) -> None:
        import train_tir_agent as t

        self.assertIsNone(t.window_agent_match(None, ["planner"]))
        self.assertIsNone(t.window_agent_match(str(AF_WORKFLOW), []))
        with mock.patch("workflow.runtime.agent_window_rollout", return_value=False):
            self.assertIsNone(t.window_agent_match(str(AF_WORKFLOW), ["planner"]))
        with mock.patch.dict(os.environ, {"TIR_AGENT_MATCH": "0"}):
            self.assertIsNone(t.window_agent_match(str(AF_WORKFLOW), ["planner"]))

    def test_only_trainable_planner_calls_survive(self) -> None:
        spans = [
            _span("root", "session", None, 0.0, 20.0),
            _agent("pw", "agent__planner", "root", 1.0, 5.0),
            _llm("p_train", "pw", 1.1, 1.5, 10),
            _agent("pf", "agent__frozen__planner_fixed", "pw", 2.0, 3.0),
            _llm("p_fixed", "pf", 2.1, 2.9, 20),
            # proxy-side span with no parent link: repair_hierarchy reparents by time
            _llm("p_orphan", "root", 3.5, 4.5, 30),
            _agent("ew", "agent__u_agentflow__executor", "root", 5.0, 6.0),
            _llm("e_call", "ew", 5.1, 5.9, 40),
            _agent("vw", "agent__verifier", "root", 6.0, 9.0),
            _llm("v_call", "vw", 6.1, 6.9, 50),
        ]
        adapter = TracerTraceToTriplet(agent_match=r"^(?:agent__planner)$", _skip_empty_token_spans=True)
        kept = sorted(t.prompt["token_ids"][0] for t in adapter.adapt(spans))
        self.assertEqual(kept, [10, 30])

        unfiltered = TracerTraceToTriplet(_skip_empty_token_spans=True).adapt(spans)
        self.assertEqual(len(unfiltered), 5)


if __name__ == "__main__":
    unittest.main()
