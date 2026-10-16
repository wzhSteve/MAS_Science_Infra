"""Phase 4 prompts module tests (docs/MAS_design.md §7).

Covers the frozen prompt contracts:
  - empty-string rule: None / whitespace-only fall through, non-empty adopted
  - priority chain: node.system_prompt > hub.system_prompt > DEFAULT_PLANNER_PROMPT
  - candidate_menu: multi-router dedup + single-router passthrough
  - planner_user_text: the three-part composition (feedback / menu / direct)
  - trajectory prompt meta: prompt_hash / prompt_source land in Trajectory.meta
  - tier override log: SCIENCE_INFRA_TOOL_KERNEL=1 forces lite with reason
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MAS_ROOT = Path(__file__).resolve().parents[1]
REPO = MAS_ROOT.parent
if str(MAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAS_ROOT))

from workflow.prompts import (  # noqa: E402
    DEFAULT_PLANNER_PROMPT,
    DEFAULT_VERIFIER_PROMPT,
    _usable,
    candidate_menu,
    planner_prompt,
    planner_user_text,
    prompt_provenance,
    verifier_prompt,
)


def _spec(hub_prompt: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(hub=SimpleNamespace(system_prompt=hub_prompt))


def _compiled(
    planner_prompt_text: str | None = None,
    verifier_prompt_text: str | None = None,
    agents: dict | None = None,
) -> SimpleNamespace:
    agents = agents or {}
    if planner_prompt_text is not None:
        agents["planner"] = SimpleNamespace(system_prompt=planner_prompt_text, kind="planner")
    else:
        agents["planner"] = SimpleNamespace(system_prompt=None, kind="planner")
    if verifier_prompt_text is not None:
        agents["verifier"] = SimpleNamespace(system_prompt=verifier_prompt_text, kind="verifier")
    return SimpleNamespace(agents=agents)


class TestEmptyStringRule(unittest.TestCase):
    """MAS_design §7.1: None / missing / whitespace-only → next tier."""

    def test_none_falls_through(self) -> None:
        self.assertFalse(_usable(None))
        c = _compiled(planner_prompt_text=None)
        s = _spec(hub_prompt=None)
        # all tiers empty → default
        self.assertEqual(planner_prompt(s, c), DEFAULT_PLANNER_PROMPT)

    def test_whitespace_only_falls_through(self) -> None:
        self.assertFalse(_usable("   \n\t  "))
        c = _compiled(planner_prompt_text="   \n\t  ")
        s = _spec(hub_prompt="hub prompt")
        # whitespace node prompt must NOT mask a usable hub prompt
        self.assertEqual(planner_prompt(s, c), "hub prompt")

    def test_non_empty_adopted(self) -> None:
        self.assertTrue(_usable(" real prompt "))
        c = _compiled(planner_prompt_text=" node prompt ")
        s = _spec(hub_prompt="hub prompt")
        self.assertEqual(planner_prompt(s, c), " node prompt ")


class TestPriorityChain(unittest.TestCase):
    def test_node_beats_hub(self) -> None:
        c = _compiled(planner_prompt_text="node wins")
        s = _spec(hub_prompt="hub loses")
        self.assertEqual(planner_prompt(s, c), "node wins")

    def test_hub_beats_default(self) -> None:
        c = _compiled(planner_prompt_text=None)
        s = _spec(hub_prompt="hub wins")
        self.assertEqual(planner_prompt(s, c), "hub wins")

    def test_default_when_all_empty(self) -> None:
        c = _compiled(planner_prompt_text=None)
        s = _spec(hub_prompt=None)
        self.assertEqual(planner_prompt(s, c), DEFAULT_PLANNER_PROMPT)


class TestCandidateMenu(unittest.TestCase):
    def test_multi_router_dedup(self) -> None:
        routers = {
            "r1": SimpleNamespace(candidates=["web_fetch", "wikipedia_search"]),
            "r2": SimpleNamespace(candidates=["wikipedia_search", "python_coder"]),
        }
        menu = candidate_menu(["r1", "r2"], routers)
        self.assertEqual(menu, ["web_fetch", "wikipedia_search", "python_coder"])

    def test_single_router_passthrough(self) -> None:
        routers = {"only": SimpleNamespace(candidates=["web_fetch"])}
        self.assertEqual(candidate_menu(["only"], routers), ["web_fetch"])

    def test_user_text_uses_menu(self) -> None:
        text = planner_user_text(question="q?", candidates=["a", "b"])
        self.assertIn("['a', 'b']", text)
        self.assertTrue(text.startswith("q?"))


class TestPlannerUserText(unittest.TestCase):
    """Three-part composition (MAS_design §7.3), byte-frozen strings."""

    def test_feedback_variant(self) -> None:
        text = planner_user_text(question="q?", feedback="fix it")
        self.assertEqual(
            text,
            "q?\n\n[Verifier feedback from previous turn]\nfix it\n"
            "Decide the next step. If the feedback already answers the question, "
            "set done to true and next to empty. "
            "Output JSON with keys: next, args, sub_goal, done.",
        )

    def test_feedback_keeps_candidate_ids(self) -> None:
        text = planner_user_text(
            question="q?",
            feedback="plan_step.next ['agent1'] not in candidates",
            candidates=["base_generator", "bing_search"],
        )
        self.assertIn("['base_generator', 'bing_search']", text)
        self.assertIn("next must be one of", text)
        self.assertIn("set done to true and next to empty", text)

    def test_menu_variant(self) -> None:
        text = planner_user_text(question="q?", candidates=["x"])
        self.assertEqual(
            text,
            "q?\n\nDecide the next step and pick one tool from "
            "['x']. Output JSON with keys: next, args, sub_goal, done.",
        )

    def test_direct_executor_variant(self) -> None:
        text = planner_user_text(question="q?", direct_executor="exec_1")
        self.assertEqual(
            text,
            "q?\n\nDecide the next step. next must be exec_1. "
            "Output JSON with keys: next, args, sub_goal, done.",
        )


class TestPromptProvenance(unittest.TestCase):
    def test_provenance_tiers(self) -> None:
        c = _compiled(planner_prompt_text="node wins", verifier_prompt_text="v prompt")
        s = _spec(hub_prompt="hub loses")
        prov = prompt_provenance(s, c)
        self.assertEqual(prov["prompt_source"]["planner"], "node")
        self.assertEqual(prov["prompt_source"]["verifier"], "node")
        self.assertEqual(len(prov["prompt_hash"]["planner"]), 16)

    def test_provenance_default_and_none(self) -> None:
        c = _compiled(planner_prompt_text=None, verifier_prompt_text=None)
        s = _spec(hub_prompt=None)
        prov = prompt_provenance(s, c)
        self.assertEqual(prov["prompt_source"]["planner"], "default")
        self.assertEqual(prov["prompt_source"]["verifier"], "none")

    def test_empty_verifier_prompt_uses_default(self) -> None:
        c = _compiled(verifier_prompt_text="  ")
        self.assertEqual(verifier_prompt(c), DEFAULT_VERIFIER_PROMPT)
        prov = prompt_provenance(_spec(hub_prompt=None), c)
        self.assertEqual(prov["prompt_source"]["verifier"], "default")

    def test_canvas_copy_matches_runtime_defaults(self) -> None:
        text = (REPO / "webui/src/features/mas/model/defaultPrompts.ts").read_text(encoding="utf-8")
        for sentence in (*DEFAULT_PLANNER_PROMPT.split("。"), *DEFAULT_VERIFIER_PROMPT.split("。")):
            piece = sentence.strip()
            if piece:
                self.assertIn(piece, text)

    def test_trajectory_meta_receives_prompt_fields(self) -> None:
        from workflow.runtime import EpisodeRaw, episode_to_trajectory
        from workflow.memory.archive import Archive

        raw = EpisodeRaw(
            messages=[{"role": "user", "content": "q"}],
            final_answer="42",
            prompt_hash={"planner": "abc123def456abc1"},
            prompt_source={"planner": "node"},
        )
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            arch = Archive(root_dir=td)
            traj = episode_to_trajectory(
                {"id": "t1", "question": "q", "answer": "42", "source": "gsm8k"},
                raw,
                arch,
                collector="mock",
            )
        self.assertEqual(traj.meta.get("prompt_hash"), {"planner": "abc123def456abc1"})
        self.assertEqual(traj.meta.get("prompt_source"), {"planner": "node"})


class TestTierOverrideLog(unittest.TestCase):
    def test_kernel_switch_forces_lite_with_reason(self) -> None:
        import tools.tool_agents as ta

        old = os.environ.get("SCIENCE_INFRA_TOOL_KERNEL")
        try:
            os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = "1"
            ta._logged_tier_mismatches.clear()
            with self.assertLogs("tools.tool_agents", level="WARNING") as cm:
                resolved = ta.log_effective_tier("web_fetch", "pro")
            self.assertEqual(resolved, "lite")
            self.assertTrue(any("SCIENCE_INFRA_TOOL_KERNEL" in line for line in cm.output))
            self.assertTrue(any('"tool": "web_fetch"' in line or "'tool': 'web_fetch'" in line for line in cm.output))
            # deduped: second call logs nothing
            ta.log_effective_tier("web_fetch", "pro")
        finally:
            if old is None:
                os.environ.pop("SCIENCE_INFRA_TOOL_KERNEL", None)
            else:
                os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = old

    def test_default_clean_resolution_is_quiet(self) -> None:
        import tools.tool_agents as ta

        old = os.environ.get("SCIENCE_INFRA_TOOL_KERNEL")
        try:
            os.environ.pop("SCIENCE_INFRA_TOOL_KERNEL", None)
            ta._logged_tier_mismatches.clear()
            self.assertEqual(ta.log_effective_tier("web_fetch", None), "lite")  # clean, quiet
            self.assertEqual(ta.log_effective_tier("web_fetch", "pro"), "pro")  # honoured, quiet
            self.assertEqual(ta.effective_tier("pro"), "pro")
            self.assertEqual(ta.effective_tier("lite"), "lite")
        finally:
            if old is not None:
                os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = old


if __name__ == "__main__":
    unittest.main()
