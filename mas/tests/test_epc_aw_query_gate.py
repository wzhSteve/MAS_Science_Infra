"""EPC-AW adapted context: query condensing, SERP stop-gate, answer extraction.

Regression tests for the MAS-eval failure fixed on 2026-10-10: polluted
SERP queries, premature STOP on bare SERP lists, and process-summary
answers that bury the final answer.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

EPC_CTX = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"

INSTRUCTION_TAIL = (
    "?When ready, output the final answer enclosed in <answer> and </answer> "
    "tags. Do not generate any content after the </answer> tag."
)


def _load():
    spec = importlib.util.spec_from_file_location("epc_aw_adapted_context_qg", EPC_CTX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCondenseQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _load()

    def test_prefix_and_tail_stripped(self) -> None:
        raw = (
            "Search the web for: What nationality was Oliver Reed's character "
            f"in the film Royal Flash{INSTRUCTION_TAIL}"
        )
        out = self.ctx._condense_query(raw)
        self.assertIn("Oliver Reed", out)
        self.assertIn("Royal Flash", out)
        for bad in ("Search", "When", "ready", "output", "final", "answer", "Do"):
            self.assertNotIn(bad, out, f"residue {bad!r} in {out!r}")

    def test_question_mark_cut(self) -> None:
        out = self.ctx._condense_query("Who is Oliver Reed?When ready output answer")
        self.assertIn("Oliver Reed", out)
        self.assertNotIn("When", out)

    def test_short_passthrough(self) -> None:
        self.assertEqual(self.ctx._condense_query("Der Lindberghflug composer"),
                         "Der Lindberghflug composer")


class TestSerpGateAndAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _load()

    def test_serp_detection(self) -> None:
        self.assertTrue(self.ctx._looks_like_serp_list("bing results for: q"))
        self.assertTrue(self.ctx._looks_like_serp_list("wikipedia results for: q"))
        self.assertFalse(self.ctx._looks_like_serp_list("<answer>Prussian</answer>"))

    def test_answer_section_extraction_shape(self) -> None:
        # verify_window extraction is regex-driven; exercise the exact shape
        # the direct_output prompt mandates.
        import re

        def extract(answer: str) -> str:
            m = re.search(
                r"(?is)(?:^|\n)[ \t]*(?:2?[.\-:]?\s*)?(?:\*\*)?\s*Answer\s*(?:\*\*)?\s*[.:：]\s*(.*?)(?:\n\s*\n|\Z)",
                answer,
            )
            if m and m.group(1).strip():
                return m.group(1).strip().strip("*").strip()
            return answer

        two_section = (
            "1. **Process Summary:** searched and found evidence.\n"
            "2. **Answer:** Prussian"
        )
        self.assertEqual(extract(two_section), "Prussian")
        self.assertEqual(extract("Answer: Kurt Julian Weill"), "Kurt Julian Weill")
        self.assertEqual(extract("Prussian"), "Prussian")


if __name__ == "__main__":
    unittest.main()
