"""Single outline / STOP controller for EPC_AW Solver."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from MAS.epc_aw.models.task_profile import SlotGate
from MAS.epc_aw.models.tool_router import infer_subgoal_kind
from MAS.epc_aw.models.solver_types import VerificationResult


class PlanControllerMixin:
    """Mixin: next_outline / can_stop as the only plan exits."""

    def can_stop(self, question: str, analysis: str = "") -> bool:
        return self._can_stop_execution(question, analysis)

    def next_outline(
        self,
        question: str,
        outline: Dict[str, Any],
        verification: Optional[VerificationResult] = None,
    ) -> Dict[str, Any]:
        """Ensure a non-empty next outline (recovery / compute / synthesis)."""
        return self._ensure_outline_nonempty(question, outline, verification)

    def _fallback_advance_outline(
        self,
        prev_outline: Dict[str, Any],
        completed_step_num: int,
    ) -> Dict[str, Any]:
        """Drop completed step and renumber remaining steps (deterministic fallback)."""
        if not prev_outline:
            return {}

        keys = sorted(prev_outline.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        remaining = [prev_outline[k] for k in keys if str(k) != str(completed_step_num)]
        if not remaining and len(keys) > 1:
            remaining = [prev_outline[k] for k in keys[1:]]

        return {str(i + 1): step for i, step in enumerate(remaining)}

    def _minimal_continuation_outline(
        self,
        question: str,
        verification: Optional["VerificationResult"] = None,
    ) -> Dict[str, str]:
        ctx = ((verification.analysis if verification else "") or "")[:300]
        return {
            "1": (
                "Target Information: Obtain the missing evidence required to answer the "
                "original question. "
                "Operation Details: Try alternative search tools, queries, or parameters. "
                f"Expected Output: Information that directly answers: {question[:180]}. "
                f"Context: {ctx}"
            ),
        }

    def _outline_synthesis_step(self, question: str) -> Dict[str, str]:
        return {
            "1": (
                "Target Information: Produce the final verified answer to the original question "
                "using all collected evidence. "
                "Operation Details: Base_Generator_Tool with query summarizing obtained information. "
                f"Expected Output: Direct answer to: {question[:200]}"
            ),
        }

    def _outline_compute_step(self, question: str) -> Dict[str, str]:
        """Inject a Python_Coder_Tool step when a computed slot is still unfilled.

        Ensures a calculation step exists in the outline even if the LLM
        outline-update dropped it, so _can_stop_execution won't trap us.
        """
        records = self.system_memory.evidence_records
        metrics_summary = "; ".join(
            str(r.get("content", ""))[:200]
            for r in records
            if r.get("status") != "disputed"
        )[:600]
        return {
            "1": (
                "Target Information: Compute the final numerical answer from the "
                "already-retrieved input metrics using an explicit calculation. "
                "Operation Details: Python_Coder_Tool with a script that reads the "
                "obtained input values, applies the requested formula/rounding, and "
                "prints the integer result. "
                f"Expected Output: A single integer answering: {question[:200]}. "
                f"Retrieved inputs: {metrics_summary}"
            ),
        }

    def _ensure_outline_nonempty(
        self,
        question: str,
        outline: Dict[str, Any],
        verification: Optional["VerificationResult"] = None,
    ) -> Dict[str, Any]:
        """Keep a non-empty outline only when the answer is not yet ready."""
        if outline and isinstance(outline, dict) and len(outline) > 0:
            return outline
        analysis = verification.analysis if verification else ""
        if self._can_stop_execution(question, analysis):
            return {}
        recovery = self._generate_recovery_outline(
            question,
            verification or VerificationResult(
                analysis="", step_conclusion="SUBGOAL_INCOMPLETE",
                info_flag=False, obtained_info="", task_conclusion="CONTINUE",
                diagnostic_signal=None, subgoal_complete=False,
            ),
        )
        if recovery:
            return recovery
        return self._minimal_continuation_outline(question, verification)

    def _outline_has_remaining_steps(self) -> bool:
        outline = self.system_memory.get_outline() or {}
        return bool(outline)

    @staticmethod
    def _outline_step_is_verify_only(target: str) -> bool:
        """True when remaining outline step is verify/synthesis, not acquisition."""
        t = str(target or "").lower()
        if not t.strip():
            return True
        verify_markers = (
            "final verification",
            "final answer",
            "cross-reference",
            "cross reference",
            "verify the",
            "verification and synthesis",
            "synthesize",
            "synthesis",
            "base_generator",
            "confirm the answer",
            "confirm that",
        )
        acquire_markers = (
            "identify the",
            "retrieve",
            "find the",
            "search for",
            "look up",
            "obtain",
            "extract",
            "compute",
            "calculate",
            "who was",
            "who is",
            "founding year",
            "president",
        )
        # Synthesis/final-answer markers win over incidental acquire words
        # (e.g. "president" inside a synthesis Expected Output sentence).
        if any(m in t for m in verify_markers):
            return True
        if any(m in t for m in acquire_markers):
            return False
        return False

    def _outline_blocks_stop(self) -> bool:
        """Refuse STOP while outline still has non-verify acquisition work."""
        getter = getattr(self.system_memory, "get_outline", None)
        outline = (getter() if callable(getter) else None) or {}
        if not outline:
            return False
        numeric_keys = sorted(
            [k for k in outline.keys() if str(k).isdigit()],
            key=lambda x: int(x),
        )
        if not numeric_keys:
            # Non-numeric outline dict still counts as remaining work.
            first = next(iter(outline.values()), "")
            return not self._outline_step_is_verify_only(str(first))
        first_target = outline.get(numeric_keys[0], "")
        return not self._outline_step_is_verify_only(str(first_target))

    def _can_stop_execution(self, question: str, analysis: str = "") -> bool:
        """Hard gate: STOP when SlotGate confirms all required slots are filled."""
        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        if profile:
            if not SlotGate.can_stop(profile, records):
                return False
            # Never STOP while the plan still has acquisition steps (e.g. president
            # lookup after founding-year evidence incorrectly filled final_answer).
            if self._outline_blocks_stop():
                return False
            # Computed-source slots (e.g. computed_count) must be filled by a
            # computed record (Python_Coder_Tool) — a retrieved value must not
            # satisfy them. This blocks premature STOP on tasks needing a calc.
            if SlotGate.has_compute_slot(profile):
                if not SlotGate.compute_slot_filled_by_computed(profile, records):
                    return False
                return True
            if SlotGate.needs_compute_step(profile, records):
                outline = self.system_memory.get_outline() or {}
                if any(
                    "Python_Coder_Tool" in str(v)
                    for v in outline.values()
                ):
                    return False
            return True
        if self._outline_has_remaining_steps():
            return False
        cumulative = self._cumulative_evidence_text()
        return self.diagnoser._answer_verified_ready(question, cumulative, analysis)

    def _generate_recovery_outline(
        self,
        question: str,
        verification: "VerificationResult",
    ) -> Dict[str, Any]:
        """Inject steps when the plan is exhausted but the answer is not verified."""
        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        if profile and SlotGate.can_stop(profile, records):
            return {}
        if profile and SlotGate.needs_compute_step(profile, records):
            return {
                "1": (
                    "Target Information: Compute the final numeric answer from verified "
                    "input metrics already in obtained information. "
                    "Operation Details: Python_Coder_Tool with explicit numeric inputs and "
                    "rounding rules from the question. "
                    "Expected Output: computed_count slot filled with the final number."
                ),
            }

        cumulative = self.system_memory.get_obtained_information_for_prompt()
        combined = f"{question} {verification.analysis} {cumulative}".lower()
        steps: Dict[str, str] = {}

        if self.diagnoser._is_multi_hop_question(question) and "arxiv" in combined:
            profile = self.system_memory.get_task_profile()
            records = self.system_memory.evidence_records
            has_axis_labels = bool(
                profile and "axis_labels" in SlotGate.filled_slots(profile, records)
            )
            if not has_axis_labels:
                has_axis_labels = SlotGate._axis_labels_satisfied(combined)
            if not has_axis_labels:
                # Generic arxiv ID extraction from evidence — no hardcoded default ID.
                arxiv_id_match = re.search(
                    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{5})", combined, re.I,
                )
                arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else ""
                id_clause = (
                    f"arXiv paper {arxiv_id} " if arxiv_id else "the identified arXiv paper "
                )
                steps["1"] = (
                    "Target Information: Extract all axis end-label words from the figure in "
                    f"{id_clause}. "
                    "Operation Details: Google_Search_Tool with a query built from the paper "
                    "identifier and the phrase 'figure 1 axis labels'. "
                    "Expected Output: The axis end-label words shown in the paper figure."
                )

            q_years = sorted(set(re.findall(r"\b(20\d{2})\b", question)))
            for year in q_years:
                if year not in cumulative:
                    step_num = len(steps) + 1
                    steps[str(step_num)] = (
                        f"Target Information: Locate the arXiv article submitted in {year} "
                        "and extract evidence needed to answer the cross-reference question. "
                        "Operation Details: Google_Search_Tool with a query combining 'arXiv', "
                        f"the year {year}, and the topic keywords in the question, then "
                        "Web_Search_Tool on the article URL. "
                        "Expected Output: Evidence that answers the cross-reference."
                    )
                    break

        if ("usgs" in combined or "zip" in combined) and not steps:
            if not self.diagnoser._extract_zip_codes(cumulative):
                steps["1"] = (
                    "Target Information: Resolve the five-digit U.S. zip code for the "
                    "location referenced in the question. "
                    "Operation Details: Google_Search_Tool with a query combining the "
                    "location name from the question and 'zip code'. "
                    "Expected Output: A five-digit U.S. zip code."
                )
        elif not steps:
            # Significant lowercase terms from the question (not Capitalized verbs
            # like "Round") + inline code fragments as anchors.
            stop = {
                "the", "and", "for", "from", "with", "that", "this", "into", "using",
                "use", "retrieve", "find", "identify", "calculate", "compute", "what",
                "which", "how", "many", "much", "exact", "value", "information",
                "round", "please", "assume", "they", "their", "would", "could",
                "when", "where", "why", "does", "did", "are", "was", "were",
                "search", "get", "list", "who",
            }
            terms: List[str] = []
            seen = set()
            for t in re.findall(r"[a-z0-9]+", str(question or "").lower()):
                if len(t) < 4 or t in stop or t in seen:
                    continue
                seen.add(t)
                terms.append(t)
                if len(terms) >= 8:
                    break
            code_bits = re.findall(r"`([^`]+)`", str(question or ""))
            code_bits = [c.strip() for c in code_bits if len(c.strip()) >= 2]
            for run in re.findall(r"`{2,}", str(question or "")):
                if run not in code_bits:
                    code_bits.append(run)
            code_bits = code_bits[:3]
            anchor_parts = []
            if terms:
                anchor_parts.append(" ".join(terms))
            if code_bits:
                anchor_parts.append(" ".join(f"`{c}`" for c in code_bits))
            anchors = " ".join(anchor_parts).strip()
            target = (
                f"Retrieve missing evidence for: {anchors}."
                if anchors
                else "Retrieve missing evidence to answer the original question."
            )
            ops = (
                f"Google_Search_Tool with a query that includes these anchors: {anchors}."
                if anchors
                else "Use an appropriate search tool grounded in the question text."
            )
            steps["1"] = (
                f"Target Information: {target} "
                f"Operation Details: {ops} "
                f"Expected Output: Verified facts that answer the question. "
                f"Context: {(verification.analysis or '')[:200]}"
            )
        return steps

    def _is_answer_ready(self, question: str) -> bool:
        # v3: early-stop bypass disabled. STOP decisions must flow through the
        # unified answer-sanity gate (`_answer_sanity_pass`) inside
        # `_handle_subgoal_complete` or the outline-empty branch. Returning False
        # here ensures the revise step injected by a blocked gate actually runs
        # on the next loop iteration instead of being short-circuited.
        return False

    def _sanitize_outline_update(
        self,
        prev_outline: Dict[str, Any],
        new_outline: Dict[str, Any],
        current_step_num: int,
        task_conclusion: str,
        question: str,
        verification: "VerificationResult",
    ) -> Dict[str, Any]:
        """Outline updates prefer LLM output; empty is allowed when answer is ready."""
        prev_outline = prev_outline or {}
        new_outline = new_outline if isinstance(new_outline, dict) else {}

        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        exportable = bool(
            profile and SlotGate.extract_final_answer(profile, records)
        )
        answered = self._can_stop_execution(question, verification.analysis)
        # Plan exhausted + COMPLETE + exportable answer: do not bag-of-words
        # recovery ("born ready") even if SlotGate.can_stop is still false.
        skip_recovery = answered or (
            bool(getattr(verification, "subgoal_complete", False)) and exportable
        )

        if new_outline:
            result = new_outline
        elif answered:
            result = {}
        else:
            result = self._fallback_advance_outline(prev_outline, current_step_num)
            if not result:
                print(
                    "\n==> ⚠️ LLM returned empty outline — using deterministic advance/recovery\n"
                )

        if not result and not skip_recovery:
            result = self._generate_recovery_outline(question, verification)

        if not result:
            if skip_recovery or self._can_stop_execution(question, verification.analysis):
                result = {}
            else:
                result = self._minimal_continuation_outline(question, verification)

        profile = self.system_memory.get_task_profile()
        if profile and infer_subgoal_kind(
            verification.analysis or "", "",
        ) == "synthesis" and not SlotGate.can_stop(profile, self.system_memory.evidence_records):
            synth_markers = ("base_generator", "final answer", "synthesize")
            text = " ".join(str(v).lower() for v in result.values())
            if any(m in text for m in synth_markers):
                missing = profile.missing_slot_names(self.system_memory.evidence_records)
                if missing:
                    print(
                        f"\n==> 🔄 Synthesis blocked — missing slots {missing}; "
                        f"injecting acquisition recovery step\n"
                    )
                    result = {
                        "1": (
                            f"Target Information: Retrieve primary evidence for missing slots: "
                            f"{', '.join(missing)}. "
                            "Operation Details: Google_Search_Tool or Web_Search_Tool. "
                            "Expected Output: Verified facts to fill required answer slots."
                        ),
                    }

        # Compute-slot enforcement: if the profile requires a computed answer
        # but no compute step remains in the outline and the computed slot is
        # not yet filled by a computed source, inject a Python_Coder_Tool step.
        # This prevents premature STOP on calculation tasks (e.g. pid=4).
        # Guard: only inject when the required input slots (base_count /
        # input_metrics) are already filled — otherwise we would overwrite a
        # legitimate retrieval outline (e.g. produced by decompose_goal) with
        # an empty compute step that has nothing to compute.
        profile = self.system_memory.get_task_profile()
        if profile and SlotGate.has_compute_slot(profile):
            records = self.system_memory.evidence_records
            if (
                not SlotGate.compute_slot_filled_by_computed(profile, records)
                and SlotGate.needs_compute_step(profile, records)
            ):
                outline_text = " ".join(str(v) for v in result.values()).lower()
                if "python_coder_tool" not in outline_text:
                    print(
                        "\n==> 🔄 Compute step injected — computed slot unfilled; "
                        "adding Python_Coder_Tool calculation step\n"
                    )
                    result = self._outline_compute_step(question)

        result = self._ensure_outline_nonempty(question, result, verification)
        return result

    def _inject_partial_cross_ref_outline(self, question: str) -> bool:
        """Inject targeted retrieval when one slot is filled but a cross-ref slot is not."""
        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        if not profile or not SlotGate.has_partial_cross_ref(profile, records):
            return False

        answer = SlotGate.extract_final_answer(profile, records)
        cumulative = self.system_memory.get_obtained_information_for_prompt()
        missing = profile.missing_slot_names(records)
        missing_desc = ", ".join(missing) if missing else "the missing cross-reference slot"

        # Pull a source identifier from evidence when present (generic paper/host id).
        src_m = re.search(
            r"(?:arxiv\.org/(?:abs|pdf)/|doi\.org/)([\w./-]+)", cumulative, re.I,
        )
        src_id = src_m.group(0) if src_m else ""
        source_clause = (
            f"from source {src_id} " if src_id else "from the identified source "
        )

        outline = {
            "1": (
                f"Target Information: Retrieve evidence for {missing_desc} "
                f"{source_clause}to complete the cross-reference. "
                "Operation Details: Google_Search_Tool with a query built from the "
                "already-filled slot value and the source identifier. "
                "Expected Output: Evidence that fills the missing slot."
            ),
            "2": (
                "Target Information: Confirm the final answer by cross-matching the "
                "newly retrieved slot with the value already in memory. "
                "Operation Details: Base_Generator_Tool — evidence only. "
                "Expected Output: The single matching value"
                + (f" ({answer})" if answer else "")
            ),
        }
        self.system_memory.set_outline(outline)
        print("\n==> 🔄 Partial cross-ref — injected slot retrieval + match steps\n")
        return True

    def _maybe_escalate_partial_cross_ref(
        self,
        question: str,
        step_id: str,
        verification: VerificationResult,
    ) -> bool:
        """Track no-delta streaks and inject recovery when cross-ref is partially filled."""
        no_delta = self.system_memory.record_step_attempt(
            step_id, verification.had_slot_delta,
        )
        if no_delta < 2:
            return False
        return self._inject_partial_cross_ref_outline(question)


