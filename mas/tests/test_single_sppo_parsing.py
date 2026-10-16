"""single-ppo parsing/loop tests (stub LLM text; no real LLM, no network).

Covers the task sheet contract:
  - validate_format: <search></search> pairing, search/result nesting order,
    <answer>恰好一个 (accept / reject cases);
  - parsing 三板斧: strip code fence -> first balanced JSON object -> tail
    <answer> regex (including tolerant unclosed-answer extraction);
  - the window ReAct loop: search-><result>(800-char truncation)->answer
    close-out; max_steps exhaustion with forced-answer self-correction;
    double-<answer> violation -> one self-correct -> tolerant extraction;
    error payload path (never raises).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

SINGLE_PPO_CTX = ROOT / "user_space" / "projects" / "single-ppo" / "adapted" / "context.py"


def load_context_module():
    spec = importlib.util.spec_from_file_location("single_ppo_adapted_context", SINGLE_PPO_CTX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubLLM:
    """Pops canned chunks in order; records every call's message tail."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self.chunks:
            return ""
        return self.chunks.pop(0)


class SystemPromptTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_context_module()

    def test_prompt_under_1200_chars(self):
        self.assertLessEqual(len(self.mod.SYSTEM_PROMPT), 1200)
        for tag in ("<search>", "<result>", "<python>", "<answer>"):
            self.assertIn(tag, self.mod.SYSTEM_PROMPT)

    def test_runtime_defaults(self):
        cfg = self.mod._load_runtime()
        self.assertEqual(cfg["max_steps"], 6)
        self.assertEqual(cfg["temperature"], 0.7)
        self.assertEqual(cfg["reply_budget"], 800)
        self.assertEqual(
            cfg["enabled_tools"],
            ["Wikipedia_Search_Tool", "Bing_Search_Tool", "Web_Fetch_Tool", "Python_Coder_Tool"],
        )


class ValidateFormatTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_context_module()
        self.validate = self.mod.validate_format

    def test_valid_full_trajectory(self):
        text = (
            "I need facts. <search>Paracetamol</search> <result> wiki says X </result>"
            "Now compute. <python> print(2*21) </python> <result> 42 </result>"
            "<answer>42</answer>"
        )
        ok, reason = self.validate(text)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "format is correct")

    def test_valid_answer_only(self):
        ok, _ = self.validate("直接作答 <answer>Paris</answer>")
        self.assertTrue(ok)

    def test_missing_answer(self):
        ok, reason = self.validate("<search>q</search> <result>r</result>")
        self.assertFalse(ok)
        self.assertIn("answer", reason)

    def test_two_answers_rejected(self):
        ok, reason = self.validate("<answer>a</answer> more <answer>b</answer>")
        self.assertFalse(ok)
        self.assertIn("exactly once", reason)

    def test_unclosed_search_rejected(self):
        ok, reason = self.validate("<search>q</search> <answer>a</answer>")
        self.assertFalse(ok)
        self.assertIn("incomplete", reason)

    def test_wrong_nesting_rejected(self):
        # result BEFORE </search> (model emitted its own result inside)
        ok, reason = self.validate("<search>q <result>r</result></search> <answer>a</answer>")
        self.assertFalse(ok)
        self.assertIn("nested in the wrong order", reason)

    def test_answer_inside_result_rejected(self):
        ok, reason = self.validate(
            "<search>q</search> <result> <answer>a</answer> </result>"
        )
        self.assertFalse(ok)

    def test_post_answer_noise_tolerated_after_cut(self):
        # the window validates _cut_after_answer(full_text): trailing junk after
        # </answer> is trimmed, so the cut text stays valid.
        text = "<search>q</search> <result>r</result> <answer>a</answer> junk"
        cut = self.mod._cut_after_answer(text)
        ok, _ = self.validate(cut)
        self.assertTrue(ok)


class ParsingTriadTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_context_module()

    def test_strip_code_fence(self):
        self.assertEqual(
            self.mod.strip_code_fence('```json\n{"a": 1}\n```'),
            '{"a": 1}',
        )
        self.assertEqual(
            self.mod.strip_code_fence('```\nplain\n```'),
            'plain',
        )
        self.assertEqual(self.mod.strip_code_fence("no fence"), "no fence")

    def test_extract_first_json_object(self):
        blob = 'noise before {"next": "a", "args": {"q": "}"}} trailing'
        parsed = self.mod.extract_first_json_object(blob)
        self.assertEqual(parsed, {"next": "a", "args": {"q": "}"}})

    def test_extract_first_json_object_none(self):
        self.assertIsNone(self.mod.extract_first_json_object("no braces here"))
        self.assertIsNone(self.mod.extract_first_json_object("{not json"))

    def test_extract_tail_answer_paired(self):
        self.assertEqual(
            self.mod.extract_tail_answer("think <answer>The Eiffel Tower</answer>"),
            "The Eiffel Tower",
        )

    def test_extract_tail_answer_last_pair_wins(self):
        self.assertEqual(
            self.mod.extract_tail_answer("<answer>first</answer> tail <answer>second</answer>"),
            "second",
        )

    def test_extract_tail_answer_unclosed(self):
        self.assertEqual(
            self.mod.extract_tail_answer("reasoning <answer>boxed answer cut mid"),
            "boxed answer cut mid",
        )

    def test_unwrap_boxed(self):
        self.assertEqual(self.mod._unwrap_boxed(r"\boxed{42}"), "42")
        self.assertEqual(self.mod._unwrap_boxed(r"The value is \(\boxed{12}\)."), "12")
        self.assertEqual(self.mod._unwrap_boxed("plain"), "plain")
        self.assertEqual(self.mod._unwrap_boxed(r"\boxed {\thirteen}"), r"\thirteen")


class ToolDispatchTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_context_module()

    def test_default_routes_to_wikipedia(self):
        with patch.object(self.mod, "_wiki", return_value="wiki ok") as wiki:
            tool, out = self.mod.run_tool("search", "Paracetamol overdose")
        self.assertEqual(tool, "Wikipedia_Search_Tool")
        self.assertEqual(out, "wiki ok")
        wiki.assert_called_once_with("Paracetamol overdose")

    def test_bing_prefix_routes_to_bing(self):
        with patch.object(self.mod, "_bing", return_value="bing ok") as bing:
            tool, out = self.mod.run_tool("search", "bing: Lady Gaga birth year")
        self.assertEqual(tool, "Bing_Search_Tool")
        self.assertEqual(out, "bing ok")
        bing.assert_called_once_with("Lady Gaga birth year")

    def test_url_routes_to_fetch(self):
        with patch.object(self.mod, "_fetch", return_value="page text") as fetch:
            tool, out = self.mod.run_tool("search", "https://en.wikipedia.org/wiki/Paris")
        self.assertEqual(tool, "Web_Fetch_Tool")
        self.assertEqual(out, "page text")
        fetch.assert_called_once_with("https://en.wikipedia.org/wiki/Paris")

    def test_python_routes_to_python_tool(self):
        with patch.object(self.mod, "_py_execute", return_value="tier=lite\n42") as py:
            tool, out = self.mod.run_tool("python", "print(6*7)")
        self.assertEqual(tool, "Python_Coder_Tool")
        self.assertEqual(out, "tier=lite\n42")
        py.assert_called_once_with("print(6*7)")

    def test_instruction_tail_stripped_from_query(self):
        with patch.object(self.mod, "_wiki", return_value="wiki ok") as wiki:
            self.mod.run_tool(
                "search",
                "Who wrote Hamlet?When ready, output the final answer enclosed in <answer> </answer> tags.",
            )
        q = wiki.call_args[0][0]
        self.assertNotIn("When ready", q)
        self.assertIn("Hamlet", q)

    def test_sanitize_strips_protocol_tags_from_tool_output(self):
        dirty = "tier=lite\nresult says <answer>not an answer</answer> <result>x</result> done"
        clean = self.mod._sanitize_for_injection(dirty)
        self.assertNotIn("tier=", clean)
        self.assertNotIn("<answer>", clean)
        self.assertNotIn("<result>", clean)
        self.assertIn("done", clean)


class ReActLoopTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_context_module()

    def _ctx(self):
        return self.mod.EpisodeContext()

    def _runtime(self, **over):
        cfg = self.mod._load_runtime()
        cfg.update(over)
        return cfg

    def test_search_then_answer_closes_episode(self):
        stub = _StubLLM(
            [
                "Let me check. <search>tallest tower Paris</search>",
                "Based on the result, the answer is clear. <answer>The Eiffel Tower</answer>",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub), patch.object(
            self.mod, "_wiki", return_value="wikipedia results for: tallest tower Paris"
        ) as wiki:
            ctx = self._ctx()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "Tallest tower in Paris?"}})
        payload = out["payload"]
        self.assertEqual(out["kind"], "plan_step")
        self.assertTrue(payload["done"])
        self.assertEqual(payload["next"], "")
        self.assertTrue(payload["ready_to_stop"])
        self.assertEqual(payload["answer"], "<answer>The Eiffel Tower</answer>")
        trace = payload["trace"]
        self.assertEqual(trace["stop"], "answer")
        self.assertEqual(trace["n_search"], 1)
        self.assertEqual(trace["n_python"], 0)
        self.assertFalse(trace["self_corrected"])
        self.assertFalse(trace.get("error"))
        wiki.assert_called_once()
        # message flow: system, user(question), assistant(search), user(<result>), assistant(answer)
        self.assertEqual(len(stub.calls), 2)
        second_call_tail = stub.calls[1]
        self.assertIn("wikipedia results for: tallest tower Paris", second_call_tail[3]["content"])
        self.assertIn("<result>", second_call_tail[3]["content"])

    def test_result_injection_truncated_to_800_chars(self):
        stub = _StubLLM(
            [
                "Search time. <search>big</search>",
                "I have enough. <answer>done</answer>",
            ]
        )
        big = "Z" * 5000
        with patch.object(self.mod, "_llm_chat", side_effect=stub), patch.object(
            self.mod, "_wiki", return_value=big
        ):
            ctx = self._ctx()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        injected = stub.calls[1][3]["content"]
        zs = injected.count("Z")
        # exactly reply_budget(800) chars of tool text survive the injection
        self.assertEqual(zs, 800)
        self.assertTrue(injected.startswith("<result>"))
        self.assertEqual(out["payload"]["trace"]["n_search"], 1)

    def test_python_loop_and_tool_count(self):
        stub = _StubLLM(
            [
                "Math needed. <python> print(7*6) </python>",
                "So it is 42. <answer>\\boxed{42}</answer>",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub), patch.object(
            self.mod, "_py_execute", return_value="tier=lite\n39? no: 42"
        ):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "7*6?"}})
        payload = out["payload"]
        self.assertEqual(payload["answer"], "<answer>42</answer>")
        trace = payload["trace"]
        self.assertEqual(trace["n_python"], 1)
        self.assertEqual(trace["n_search"], 0)
        self.assertEqual(trace["steps"][0]["tool"], "Python_Coder_Tool")

    def test_max_steps_exhaustion_forces_answer(self):
        stub = _StubLLM(
            [
                "<search>a</search>",
                "<search>b</search>",
                "<search>c</search>",
                "Fine. <answer>best-effort answer</answer>",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub), patch.object(
            self.mod, "_wiki", return_value="wikipedia results"
        ), patch.object(self.mod, "_load_runtime", return_value=self._runtime(max_steps=3)):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        payload = out["payload"]
        trace = payload["trace"]
        self.assertEqual(trace["stop"], "max_steps")
        self.assertEqual(trace["n_search"], 3)
        self.assertTrue(trace["self_corrected"])
        self.assertEqual(payload["answer"], "<answer>best-effort answer</answer>")
        self.assertEqual(len(stub.calls), 4)  # 3 loop + 1 forced-answer round

    def test_double_answer_triggers_self_correction(self):
        stub = _StubLLM(
            [
                "First attempt. <answer>a</answer> oops <answer>b</answer>",
                "Sorry. <answer>b</answer>",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        payload = out["payload"]
        trace = payload["trace"]
        self.assertTrue(trace["self_corrected"])
        self.assertEqual(payload["answer"], "<answer>b</answer>")
        # violation note injected before the correction call (last message of
        # call #2's prompt is the correction user turn)
        correction_prompt = stub.calls[1][-1]["content"]
        self.assertIn("violates the required format", correction_prompt)
        self.assertIn("exactly one", correction_prompt)

    def test_correction_fails_then_tolerant_extraction(self):
        stub = _StubLLM(
            [
                "Bad. <answer>a</answer> <answer>b</answer>",
                "Still bad, no tags at all",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        payload = out["payload"]
        trace = payload["trace"]
        self.assertTrue(trace["self_corrected"])
        self.assertEqual(payload["answer"], "<answer>b</answer>")  # tolerant last-pair
        self.assertIn("tag count", trace.get("error", ""))

    def test_never_answered_returns_error_payload_without_raise(self):
        stub = _StubLLM(
            [
                "<search>a</search>",
                "<search>b</search>",
                "<search>c</search>",
                "I cannot decide.",
            ]
        )
        with patch.object(self.mod, "_llm_chat", side_effect=stub), patch.object(
            self.mod, "_wiki", return_value="wikipedia results"
        ):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        payload = out["payload"]
        self.assertEqual(out["kind"], "plan_step")
        self.assertTrue(payload["done"])
        self.assertEqual(payload["answer"], "")
        self.assertTrue(payload["ready_to_stop"])
        self.assertIn("no <answer>", payload["trace"].get("error", ""))

    def test_llm_transport_error_is_contained(self):
        def boom(*args, **kwargs):
            raise RuntimeError("connection refused")

        with patch.object(self.mod, "_llm_chat", side_effect=boom):
            ctx = self.mod.EpisodeContext()
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        payload = out["payload"]
        self.assertEqual(payload["answer"], "")
        self.assertTrue(payload["done"])
        self.assertIn("llm transport error", payload["trace"].get("error", ""))

    def test_same_question_uses_cache(self):
        stub = _StubLLM(["<answer>cached</answer>"])
        with patch.object(self.mod, "_llm_chat", side_effect=stub):
            ctx = self.mod.EpisodeContext()
            first = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
            second = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        self.assertEqual(first["payload"]["answer"], "<answer>cached</answer>")
        self.assertEqual(second["payload"]["answer"], "<answer>cached</answer>")
        self.assertEqual(len(stub.calls), 1)


def trace_key(payload):
    return "trace"


class WindowPayloadSchemaTest(unittest.TestCase):
    """Returned dicts must validate against mas protocol payload schemas."""

    def setUp(self):
        self.mod = load_context_module()

    def test_plan_step_payload_validates(self):
        from workflow.protocol import validate_payload

        ctx = self.mod.EpisodeContext()
        with patch.object(
            self.mod, "_llm_chat", return_value="<answer>ok</answer>"
        ):
            out = ctx.plan_window({"kind": "plan_step", "payload": {"question": "q"}})
        ok, reason = validate_payload("plan_step", out["payload"])
        self.assertTrue(ok, reason)

    def test_exec_and_verify_payloads_validate(self):
        from workflow.protocol import validate_payload

        ctx = self.mod.EpisodeContext()
        ctx.answer_cache["q"] = ("<answer>ok</answer>", {})
        exec_out = ctx.exec_window(
            {"kind": "tool_invoke", "payload": {"question": "q"}}
        )
        ok, reason = validate_payload("tool_result", exec_out["payload"])
        self.assertTrue(ok, reason)
        ctx._last_answer = "<answer>ok</answer>"
        verify_out = ctx.verify_window({"kind": "verify", "payload": {}})
        ok, reason = validate_payload("verify", verify_out["payload"])
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
