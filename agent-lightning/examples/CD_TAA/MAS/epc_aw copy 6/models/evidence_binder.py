"""Single post-verify evidence / slot write path for EPC_AW Solver."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from MAS.epc_aw.models.task_profile import (
    SlotGate,
    FINAL_TERMS,
    _extract_zip_codes,
    _has_usgs_context,
    rejects_family_journal_base_count,
)
from MAS.epc_aw.models.solver_types import VerificationResult, StepContext


class EvidenceBinderMixin:
    """Mixin: after_verify is the only evidence write exit after Diagnoser.verify."""

    def after_verify(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
        step_count: int,
        *,
        record_obtained: bool = True,
    ) -> VerificationResult:
        """Persist slot deltas + harvest executor slots + optional obtained record."""
        had = verification.had_slot_delta
        if verification.slot_updates or verification.analysis:
            if self._persist_slot_deltas(verification, step_ctx, step_count):
                had = True
        if self._harvest_slots_from_executor(step_ctx, verification, step_count):
            had = True
        if record_obtained:
            self._record_obtained_information(step_count, verification, step_ctx)
        verification.had_slot_delta = had
        return verification

    @staticmethod
    def _slot_value_in_source(value: str, *sources: str) -> bool:
        val = str(value).lower().strip()
        if not val:
            return False
        blob = " ".join(str(s) for s in sources if s).lower()
        return val in blob

    def _harvest_slots_from_executor(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
        step_count: int,
    ) -> bool:
        """Write slot bindings when executor output contains structured answers."""
        profile = self.system_memory.get_task_profile()
        if not profile:
            return False

        raw = str(step_ctx.result_executor or "")
        if not raw or raw.strip().lower().startswith("error"):
            return False
        if verification.evidence_type in ("ABSENCE", "EMPTY") and (
            "do not provide information" in raw.lower()
            or "search results do not" in raw.lower()
        ):
            return False

        records = self.system_memory.evidence_records
        missing = profile.missing_slot_names(records)
        bindings: Dict[str, str] = {}
        parts: List[str] = []

        if "axis_labels" in missing:
            axis_val = SlotGate.extract_axis_labels_snippet(raw)
            if axis_val:
                bindings["axis_labels"] = axis_val
                parts.append(f"Axis labels: {axis_val}")

        if "final_term" in missing:
            lower = raw.lower()
            for term in FINAL_TERMS:
                if term in lower:
                    bindings["final_term"] = term
                    parts.append(f"Societal descriptor: {term}")
                    break

        if "zip_codes" in missing:
            zips = _extract_zip_codes(raw)
            cumulative = " ".join(str(r.get("content", "")) for r in records)
            if zips and (_has_usgs_context(raw) or _has_usgs_context(cumulative)):
                bindings["zip_codes"] = zips[0]
                parts.append(f"Zip code: {zips[0]}")

        compute_real = False
        if (
            "computed_count" in missing
            and step_ctx.tool_name == "Python_Coder_Tool"
            and self._python_execution_is_real_computation(step_ctx.result_executor)
        ):
            payload = step_ctx.result_executor
            if isinstance(payload, list) and payload:
                payload = payload[0]
            printed = payload.get("printed_output") if isinstance(payload, dict) else None
            if printed is None:
                match = re.search(r"printed_output['\"]?\s*:\s*['\"]([^'\"]+)", raw)
                printed = match.group(1) if match else None
            numeric = re.search(r"-?\d+(?:\.\d+)?", str(printed or ""))
            if numeric:
                value = numeric.group(0)
                question = ""
                getter = getattr(self.system_memory, "get_query", None)
                if callable(getter):
                    question = str(getter() or "")
                if not question and profile:
                    question = str(getattr(profile, "question", "") or "")
                # Reject non-positive answers for count / thousand-hours questions
                # (e.g. distance=2/pace=50 → 0 must not fill computed_count).
                q_lower = question.lower()
                try:
                    numeric_val = float(value)
                except ValueError:
                    numeric_val = None
                if numeric_val is not None and numeric_val <= 0 and (
                    "how many" in q_lower
                    or "thousand hours" in q_lower
                    or "round" in q_lower
                ):
                    pass
                else:
                    bindings["computed_count"] = value
                    parts.append(f"Computed result: {value}")
                    compute_real = True

        if not bindings:
            return False

        content = " | ".join(parts) if parts else raw[:500]
        quality = "secondary" if step_ctx.tool_name == "Google_Search_Tool" else "primary"
        if compute_real:
            quality = "computed"
        if step_ctx.tool_name == "Base_Generator_Tool":
            quality = "inferred"
        return self.system_memory.add_evidence_record(
            content,
            outline_step=step_ctx.step_key,
            exec_step=step_count,
            subgoal=step_ctx.sub_goal,
            tool=step_ctx.tool_name,
            source_quality=quality,
            subgoal_complete=verification.subgoal_complete,
            confidence=0.85,
            claim_type="fact",
            slot_bindings=bindings,
            compute_real=compute_real if step_ctx.tool_name == "Python_Coder_Tool" else None,
        )

    def _persist_slot_deltas(
        self,
        verification: VerificationResult,
        step_ctx: StepContext,
        step_count: int,
    ) -> bool:
        """Ensure verifier slot_updates and inferred deltas land in evidence_records."""
        profile = self.system_memory.get_task_profile()
        slot_updates = list(verification.slot_updates or [])
        analysis = verification.analysis or ""
        executor_blob = str(step_ctx.result_executor or "")
        obtained = verification.obtained_info or ""

        inferred: Dict[str, str] = {}
        if verification.evidence_type not in ("ABSENCE", "EMPTY"):
            term_m = re.search(
                r"societal descriptor(?: term)?\s*['\"]?(\w+)['\"]?",
                analysis,
                re.I,
            )
            if term_m and self._slot_value_in_source(term_m.group(1), executor_blob, obtained):
                inferred["final_term"] = term_m.group(1).lower()
            axis_m = re.search(
                r"axis labels?[:\s]+(.+?)(?:\.|$)",
                analysis,
                re.I,
            )
            if axis_m and SlotGate._axis_labels_satisfied(axis_m.group(1)):
                if self._slot_value_in_source(" vs", axis_m.group(1), executor_blob, obtained):
                    inferred["axis_labels"] = axis_m.group(1).strip()[:300]

        if verification.evidence_type in ("ABSENCE", "EMPTY"):
            slot_updates = [
                u for u in slot_updates
                if isinstance(u, dict)
                and u.get("filled")
                and u.get("value")
                and self._slot_value_in_source(str(u.get("value")), executor_blob, obtained)
            ]

        existing_slots = {u.get("slot") for u in slot_updates if isinstance(u, dict)}
        for slot_name, value in inferred.items():
            if slot_name not in existing_slots and value:
                slot_updates.append({"slot": slot_name, "value": value, "filled": True})

        if not slot_updates:
            return False

        question = str((profile.question if profile else "") or "")
        if not question:
            getter = getattr(self.system_memory, "get_query", None)
            if callable(getter):
                question = str(getter() or "")
        bindings: Dict[str, str] = {}
        for upd in slot_updates:
            if isinstance(upd, dict) and upd.get("filled") and upd.get("value"):
                slot_name = str(upd.get("slot", ""))
                value = str(upd.get("value"))
                if slot_name == "base_count" and rejects_family_journal_base_count(
                    question, f"{executor_blob} {obtained} {value}"
                ):
                    continue
                # computed_count is derivation-only — never bind from acquisition
                # tools (e.g. mistaking marathon time 1:59:40 → 59).
                if slot_name == "computed_count" and step_ctx.tool_name != "Python_Coder_Tool":
                    continue
                bindings[slot_name] = value

        if not bindings:
            return False

        records = self.system_memory.evidence_records
        if records and records[-1].get("exec_step") == step_count:
            SlotGate.apply_slot_updates(profile, records, slot_updates)
            return True

        summary = obtained[:400] if obtained else executor_blob[:400]
        if not summary:
            summary = f"Slot bindings from step {step_count}"
        added = self.system_memory.add_evidence_record(
            summary,
            outline_step=step_ctx.step_key,
            exec_step=step_count,
            subgoal=step_ctx.sub_goal,
            tool=step_ctx.tool_name,
            source_quality="secondary",
            subgoal_complete=verification.subgoal_complete,
            confidence=0.75,
            claim_type="fact",
            slot_bindings=bindings,
        )
        return added

    def _python_execution_is_real_computation(self, result_executor: Any) -> bool:
        """v3 rule-based check (no LLM): did a Python_Coder_Tool step actually
        perform a derivation that references already-retrieved inputs?

        Returns True only when the executed code (comments stripped) contains a
        real arithmetic/aggregate operation AND embeds at least one numeric
        token previously recorded in evidence. Catches:
          - "printer" steps that just echo a retrieved/hallucinated constant
            (no operation), and
          - "hallucinated computation" steps that run an op over fabricated
            constants not present in any retrieved evidence (e.g. pid=6 beads
            `12000 / 1000`).
        Parse failure → True (conservative, avoid false negatives).
        """
        try:
            blob = result_executor
            if isinstance(blob, list) and blob:
                blob = blob[0]
            if not isinstance(blob, dict):
                return True
            code = str(blob.get("execution_code") or "")
            if not code.strip():
                return True
            # Strip comments (full-line and inline) so numbers in prose do not
            # masquerade as referenced inputs.
            stripped_lines = []
            for ln in code.splitlines():
                if "#" in ln:
                    ln = ln.split("#", 1)[0]
                stripped_lines.append(ln)
            code_no_comments = "\n".join(stripped_lines)
            if not code_no_comments.strip():
                return True
            op_pattern = (
                r"[-+*/%]\s*[^=]", r"\*\*", r"//", r"\bsum\(", r"\blen\(",
                r"\bmax\(", r"\bmin\(", r"\bround\(", r"\bmath\.ceil",
                r"\bmath\.floor", r"\bstatistics\.", r"\bnumpy\.", r"\bnp\.",
            )
            has_op = any(re.search(p, code_no_comments) for p in op_pattern)
            if not has_op:
                return False
            code_nums = set(re.findall(r"\d+(?:\.\d+)?", code_no_comments))
            if not code_nums:
                return False
            retrieved_nums = set()
            for rec in self.system_memory.evidence_records:
                for m in re.findall(r"\d+(?:\.\d+)?", str(rec.get("content") or "")):
                    retrieved_nums.add(m)
            if not retrieved_nums:
                return False
            return bool(code_nums & retrieved_nums)
        except Exception:
            return True

    def _record_obtained_information(
        self,
        step_count: int,
        verification: VerificationResult,
        step_ctx: StepContext,
    ) -> None:
        should_record, info = self.diagnoser.should_record_obtained_information(
            step_conclusion=verification.step_conclusion,
            subgoal_actually_complete=verification.subgoal_complete,
            obtained_information_flag=verification.info_flag,
            obtained_information=verification.obtained_info,
            result_executor=step_ctx.result_executor,
            analysis=verification.analysis,
            tool_name=step_ctx.tool_name,
            evidence_type=verification.evidence_type,
        )
        if should_record:
            provenance = self.diagnoser._assess_evidence_provenance(
                step_ctx.result_executor,
                self.system_memory.get_query() or "",
            )
            source_quality = provenance.get("level", "unknown")
            compute_real: Optional[bool] = None
            if step_ctx.tool_name == "Python_Coder_Tool":
                compute_real = self._python_execution_is_real_computation(step_ctx.result_executor)
                source_quality = "computed" if compute_real else "secondary"
            elif step_ctx.tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
                source_quality = "primary"
            claim_type = "fact"
            if verification.evidence_type == "ABSENCE":
                claim_type = "absence"
            elif step_ctx.tool_name == "Base_Generator_Tool":
                claim_type = "hypothesis"
                source_quality = "inferred"

            slot_bindings = {}
            for upd in verification.slot_updates or []:
                if isinstance(upd, dict) and upd.get("filled") and upd.get("value"):
                    slot_bindings[str(upd.get("slot", ""))] = str(upd.get("value"))

            added = self.system_memory.add_evidence_record(
                info,
                outline_step=step_ctx.step_key,
                exec_step=step_count,
                subgoal=step_ctx.sub_goal,
                tool=step_ctx.tool_name,
                source_quality=source_quality,
                subgoal_complete=verification.subgoal_complete,
                confidence=0.8 if verification.subgoal_complete else 0.4,
                claim_type=claim_type,
                slot_bindings=slot_bindings,
                compute_real=compute_real,
            )
            if verification.slot_updates:
                profile = self.system_memory.get_task_profile()
                if profile:
                    SlotGate.apply_slot_updates(
                        profile,
                        self.system_memory.evidence_records,
                        verification.slot_updates,
                    )
            if added:
                print(f"\n==> ℹ️ Step {step_count}: New relevant information added to system memory.\n")
                print(f"\n[New Obtained Information]:\n{info}\n")
            else:
                print(f"\n==> ℹ️ Step {step_count}: Duplicate or low-quality information skipped.\n")
        else:
            print(f"\n==> ℹ️ Step {step_count}: No new relevant information obtained.\n")

    def _get_slot_values_from_evidence(self) -> Dict[str, str]:
        """Collect the latest bound value for each slot from evidence records."""
        values: Dict[str, str] = {}
        for rec in getattr(self.system_memory, "evidence_records", []) or []:
            if rec.get("status") == "disputed":
                continue
            for slot_name, val in (rec.get("slot_bindings") or {}).items():
                if slot_name and val:
                    values[str(slot_name)] = str(val).strip()
        return values

    def _cumulative_evidence_text(self) -> str:
        return self.system_memory.get_obtained_information_for_prompt()

