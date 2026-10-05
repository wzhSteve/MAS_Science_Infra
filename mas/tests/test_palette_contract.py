"""Palette / template contract: Agent / Tool / Router product objects."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestPaletteContract(unittest.TestCase):
    def test_agent_templates_are_planner_verifier_blank(self):
        from science_infra.control.services import mas_palette

        pal = mas_palette()
        kinds = [t["kind"] for t in pal["agent_templates"]]
        self.assertEqual(kinds, ["planner", "verifier", "blank"])
        self.assertNotIn("hub", kinds)
        self.assertNotIn("tool", kinds)
        ids = {t["id"] for t in pal["agent_templates"]}
        self.assertNotIn("hub", ids)
        self.assertNotIn("tool_agent", ids)

    def test_tools_are_five_encapsulated_agents(self):
        from science_infra.control.services import mas_palette

        pal = mas_palette()
        expected = ["wikipedia_search", "bing_search", "web_fetch", "python_coder", "think"]
        self.assertEqual(pal["tools"], expected)
        tool_ids = [t["id"] for t in pal["tool_agents"]]
        self.assertEqual(sorted(tool_ids), sorted(expected))
        self.assertTrue(all(t.get("llm_required") for t in pal["tool_agents"]))

    def test_centralized_template_pev_router(self):
        from science_infra.control.services import mas_palette
        from workflow.compiler import compile_spec
        from workflow.spec import MASSpec
        from workflow.templates import TEMPLATE_ORDER

        pal = mas_palette()
        ids = {t["id"] for t in pal["templates"]}
        self.assertEqual(ids, set(TEMPLATE_ORDER))
        self.assertEqual([t["id"] for t in pal["templates"]], list(TEMPLATE_ORDER))
        self.assertNotIn("hub_react", ids)
        tmpl = next(t for t in pal["templates"] if t["id"] == "centralized")
        wf = tmpl["workflow"]
        self.assertEqual(wf["topology"], "centralized")
        self.assertEqual(wf["entry_agent"], "planner")
        kinds = {a["id"]: a["kind"] for a in wf["agents"]}
        self.assertEqual(kinds["planner"], "planner")
        self.assertEqual(kinds["verifier"], "verifier")
        self.assertNotIn("hub", kinds)
        for tid in pal["tools"]:
            self.assertEqual(kinds[tid], "tool")
        self.assertEqual(wf["routers"][0]["strategy"], "from_plan")
        edge_kinds = {e["kind"] for e in wf["edges"]}
        self.assertTrue(edge_kinds <= {"route", "message", "feedback"})
        spec = MASSpec.model_validate(wf)
        c = compile_spec(spec)
        self.assertTrue(c.ok, c.reason)
        self.assertIn("python_coder", c.routers["route_exec"].candidates)

    def test_every_template_compiles_and_stays_centralized(self):
        from science_infra.control.services import mas_palette
        from workflow.compiler import compile_spec
        from workflow.spec import MASSpec

        pal = mas_palette()
        strategies = set()
        for tmpl in pal["templates"]:
            wf = tmpl["workflow"]
            self.assertNotIn("label", wf)
            self.assertEqual(wf["topology"], "centralized", tmpl["id"])
            self.assertEqual(wf["entry_agent"], "planner", tmpl["id"])
            kinds = {a["id"]: a["kind"] for a in wf["agents"]}
            self.assertEqual(kinds["planner"], "planner", tmpl["id"])
            if tmpl["id"] == "tir_five_tools":
                self.assertNotIn("verifier", kinds)
            else:
                self.assertEqual(kinds["verifier"], "verifier", tmpl["id"])
            self.assertNotIn("hub", kinds)
            edge_kinds = {e["kind"] for e in wf["edges"]}
            allowed = {"route", "message", "feedback"}
            if tmpl["id"] == "tir_five_tools":
                allowed.add("tool_call")
            self.assertTrue(edge_kinds <= allowed, tmpl["id"])
            spec = MASSpec.model_validate(wf)
            c = compile_spec(spec)
            self.assertTrue(c.ok, f"{tmpl['id']}: {c.reason}")
            for r in wf.get("routers") or []:
                strategies.add(r.get("strategy"))
        self.assertTrue({"from_plan", "score", "round_robin"} <= strategies)


if __name__ == "__main__":
    unittest.main()
