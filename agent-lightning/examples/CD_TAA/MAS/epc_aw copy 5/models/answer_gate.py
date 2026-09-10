"""Final-answer format and sanity gates for EPC_AW Solver."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from MAS.epc_aw.models.task_profile import SlotGate, _extract_character_name, _is_character_name_answer


class AnswerGateMixin:
    """Mixin: answer format / normalization / sanity pass."""

    def _candidate_final_answer(self) -> Optional[str]:
        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        if profile and SlotGate.can_stop(profile, records):
            return SlotGate.extract_final_answer(profile, records)
        return None

    @staticmethod
    def _answer_format_mismatch(question: str, answer: str) -> Optional[str]:
        """Fast rule-based check for obvious answer-vs-question type mismatches.

        Returns a human-readable reason when a mismatch is detected, else None.
        """
        q = str(question or "").lower()
        ans = str(answer or "").strip()
        if not ans:
            return "empty answer"
        # "how many" → answer must contain a number
        if re.search(r"\bhow many\b", q) and not re.search(r"\b\d+\b", ans):
            return "question asks 'how many' but answer contains no number"
        # "name of the character"/"what character" → answer should be a short token
        if re.search(
            r"(name of the char(?:a)?cter|what char(?:a)?cter|exact char(?:a)?cter)",
            q,
        ):
            stripped = ans.strip("`'\" \t\r\n")
            tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", stripped)
            if (
                len(stripped) > 30
                or not 1 <= len(tokens) <= 2
                or re.search(r"[^A-Za-z\-\s]", stripped)
            ):
                return "question asks for a character name but answer is not a short character name"
        # "thousand hours"/"hours" → answer should be a bare integer (not "121 minutes")
        if re.search(r"\bthousand hours\b|\bhours\b", q):
            if not re.search(r"^\s*\d+\s*$", ans):
                return "question asks for hours but answer is not a bare integer"
            # Reject the rounding quantum itself (e.g. "nearest 1000 hours" → 1000).
            quantum = AnswerGateMixin._rounding_quantum_from_question(q)
            if quantum is not None and ans.strip() == str(quantum):
                return (
                    f"answer equals question rounding quantum ({quantum}) "
                    "rather than a computed count"
                )
            # Sensible band for "thousand hours" distance÷pace puzzles.
            if re.search(r"\bthousand hours\b", q):
                try:
                    n = int(ans.strip())
                except ValueError:
                    n = None
                if n is not None and not (5 <= n <= 50):
                    return f"thousand-hours answer {n} outside sensible band 5–50"
        return None

    @staticmethod
    def _rounding_quantum_from_question(question: str) -> Optional[int]:
        """Extract N from 'round/nearest … N hours/…' style instructions."""
        q = str(question or "").lower()
        m = re.search(
            r"(?:round(?:ed|ing)?\s+to\s+(?:the\s+)?nearest|nearest)\s+(\d[\d,]*)",
            q,
        )
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    def _normalize_final_answer(self, question: str, raw_answer: str) -> str:
        """Extract the shortest answer to the question from a verbose slot value.

        Used when the final_answer slot got filled with a long explanatory text
        instead of the minimal answer the question expects (e.g. pid=3).
        """
        raw = str(raw_answer or "").strip()
        if not raw:
            return raw
        # Schema-wrapped character answers (JSON) must unwrap even when short.
        if "{" in raw or ('"' in raw and "character" in raw.lower()):
            name = _extract_character_name(raw)
            if name:
                return name
        # Short, single-token answers need no normalization
        if len(raw) <= 30 and len(raw.split()) <= 4 and "{" not in raw:
            return raw
        # Deterministic unwrap for character questions before LLM normalize.
        if re.search(
            r"(name of the char(?:a)?cter|what char(?:a)?cter|exact char(?:a)?cter)",
            str(question or "").lower(),
        ):
            name = _extract_character_name(raw)
            if name:
                return name
        prompt = (
            "You are an answer extractor. Given the original question and a verbose "
            "candidate answer, output ONLY the minimal exact answer the question asks "
            "for (a number, a single character name, or a short term). No explanation.\n\n"
            f"Question: {question}\n"
            f"Candidate answer: {raw}\n\n"
            "Minimal answer:"
        )
        try:
            extracted = self.executor.llm_generate_tool_command([prompt])
            extracted = str(extracted or "").strip().strip('"').strip("'")
            # Strip common leading prefixes like "Answer:" / "The answer is"
            extracted = re.sub(
                r"^(answer|the answer is|final answer)\s*[:\-]?\s*", "", extracted, flags=re.I,
            ).strip()
            if extracted and len(extracted) < len(raw):
                return extracted
        except Exception as e:
            print(f"\n==> ⚠️ Answer normalization failed: {e}\n")
        return raw

    def _answer_consistency_check(self, question: str, answer: str) -> Tuple[bool, str]:
        """Validate that the candidate answer actually answers the question.

        Returns (ok, reason). Uses a fast rule-based pre-filter plus an optional
        LLM consistency pass for verbose free-form answers.
        """
        reason = self._answer_format_mismatch(question, answer)
        if reason:
            return False, reason
        # Free-form final_answer slots that are still verbose after normalization
        # get an LLM consistency pass.
        profile = self.system_memory.get_task_profile()
        is_freeform = bool(profile and any(
            s.name == "final_answer" for s in profile.slots
        ))
        ans = str(answer or "").strip()
        if is_freeform and len(ans) > 60:
            prompt = (
                "Does the candidate answer directly and minimally answer the question? "
                "Reply with exactly 'YES' or 'NO' on the first line. If 'NO', on the "
                "second line give a short reason.\n\n"
                f"Question: {question}\n"
                f"Candidate answer: {ans}\n"
            )
            try:
                resp = str(self.executor.llm_generate_tool_command([prompt]) or "").strip()
                first = resp.splitlines()[0].strip().upper() if resp else ""
                if first.startswith("NO"):
                    return False, f"LLM consistency: {resp.splitlines()[-1].strip() if len(resp.splitlines()) > 1 else 'verbose/inconsistent'}"
            except Exception:
                pass
        return True, ""

    def _answer_sanity_pass(self, question: str) -> bool:
        """Unified STOP sanity gate (v3). Returns True if it is safe to STOP now.

        Rule-first (no new LLM): runs the existing `_answer_consistency_check`,
        which applies `_answer_format_mismatch` (field matching) before any LLM
        fallback. If the candidate fails, inject a single revise outline step so
        the main loop re-derives the answer on the next iteration. If no candidate
        is extractable yet, allow STOP (SlotGate already confirmed slots filled).
        """
        candidate = self._candidate_final_answer()
        if candidate is None:
            return True
        ok, reason = self._answer_consistency_check(question, candidate)
        if not ok:
            print(
                f"\n==> 🔄 Answer-sanity gate blocked STOP — {reason}; "
                f"injecting revise step\n"
            )
            self.system_memory.set_outline({
                "1": (
                    "Target Information: Re-derive the minimal answer that "
                    "directly matches the question's requested type/unit. "
                    "Operation Details: Base_Generator_Tool or Python_Coder_Tool "
                    "using only verified obtained information. "
                    f"Expected Output: A short answer to: {question[:200]}. "
                    f"Rejected candidate: {str(candidate)[:120]}"
                ),
            })
            return False
        return True

    def _detect_question_intent_conflict(self, question: str) -> bool:
        """Scheme A: detect a question-interpretation conflict.

        True when a how-many / quantity question has produced evidence records
        but NONE came from an external retrieval tool — i.e. the agent tried to
        answer a retrieval-grounded question from pure reasoning.
        """
        q = str(question or "").lower()
        if not re.search(r"\bhow many\b|\bhow much\b", q):
            return False
        records = self.system_memory.evidence_records or []
        if not records:
            return False
        retrieval_tools = {
            "google_search_tool", "wikipedia_search_tool", "web_search_tool",
        }
        has_retrieval = any(
            str(r.get("tool", "")).lower() in retrieval_tools
            for r in records
            if r.get("status") != "disputed"
        )
        return not has_retrieval

