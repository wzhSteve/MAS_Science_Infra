from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import (
    FailureAttributionOutput,
    FinalAuditDecision,
    VerificationDecision,
    parse_structured_response,
)
from MAS.epc_aw.models.memory import DiagnoserMemory
from MAS.epc_aw.models.state import (
    AgentState,
    EventKind,
    EvidenceRecord,
    ExecutionTrace,
    FailureRecord,
    Milestone,
    MilestoneStatus,
    VerifierEvent,
)


_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "diagnoser"
_CALCULATOR_TOOLS = {"Python_Coder_Tool"}
_ERROR_MARKERS = (
    "error:",
    "[fetch_failed]",
    "[processing_failed]",
    "[pdf_failed]",
    "no results found",
    "failed after",
)


def _load_prompt(name: str) -> str:
    return (_PROMPT_ROOT / name).read_text(encoding="utf-8")


class Diagnoser:
    """Evidence verifier, failure attributor, and final DONE auditor."""

    def __init__(
        self,
        llm_engine_name: str,
        toolbox_metadata: Optional[dict] = None,
        available_tools: Optional[List[str]] = None,
        verbose: bool = False,
        is_multimodal: bool = False,
        check_model: bool = True,
        temperature: float = 0.0,
        n: int = 1,
        llm_engine: Any = None,
        auditor_model: Optional[str] = None,
        auditor_llm_engine: Any = None,
    ):
        del check_model, n
        self.llm_engine_name = llm_engine_name
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.verbose = verbose
        self.temperature = temperature
        self.memory = DiagnoserMemory()
        self.llm_engine = llm_engine or create_llm_engine(
            model_string=llm_engine_name,
            is_multimodal=is_multimodal,
            temperature=temperature,
        )
        self.auditor_llm_engine = auditor_llm_engine or (
            create_llm_engine(
                model_string=auditor_model,
                is_multimodal=is_multimodal,
                temperature=temperature,
            )
            if auditor_model and auditor_model != llm_engine_name
            else self.llm_engine
        )

    @staticmethod
    def deterministic_checks(
        trace: ExecutionTrace, milestone: Milestone
    ) -> Dict[str, Any]:
        result_text = json.dumps(trace.result, ensure_ascii=False, default=str).strip()
        structured_error = (
            isinstance(trace.result, dict) and bool(trace.result.get("error"))
        )
        marker = next(
            (item for item in _ERROR_MARKERS if item in result_text.lower()), None
        )
        has_error = bool(trace.error or structured_error or marker)
        nonempty = trace.result not in (None, "", [], {})
        source_required = milestone.verify.require_source
        source_present = bool(trace.source_urls) or trace.tool_name in _CALCULATOR_TOOLS
        return {
            "tool_completed": trace.error is None,
            "nonempty_result": nonempty,
            "no_error_marker": not has_error,
            "source_required": source_required,
            "source_present": source_present,
            "passed": (
                not has_error
                and nonempty
                and (not source_required or source_present)
            ),
        }

    def verify(
        self,
        state: AgentState,
        milestone: Milestone,
        trace: ExecutionTrace,
    ) -> Tuple[VerifierEvent, VerificationDecision]:
        checks = self.deterministic_checks(trace, milestone)
        if not checks["passed"]:
            reason = self._deterministic_rejection_reason(checks, trace)
            decision = VerificationDecision(
                analysis="Deterministic evidence pre-check rejected the trace.",
                verdict=EventKind.REJECTED.value,
                reason=reason,
                evidence_summary="",
                confidence="high",
                facts=[],
            )
        else:
            prompt = _load_prompt("verify_evidence.txt").format(
                Question=state.question,
                Milestone=json.dumps(
                    {"id": milestone.id, "description": milestone.description},
                    ensure_ascii=False,
                ),
                Verification_Spec=json.dumps(
                    asdict(milestone.verify), ensure_ascii=False, default=str
                ),
                Facts=json.dumps(
                    state.compact_view()["facts"], ensure_ascii=False, default=str
                ),
                Trace=json.dumps(asdict(trace), ensure_ascii=False, default=str),
                Deterministic_Checks=json.dumps(checks, ensure_ascii=False),
            )
            response = self.llm_engine(
                prompt, response_format=VerificationDecision, temperature=self.temperature, n=1
            )
            decision = parse_structured_response(response, VerificationDecision)
            decision = self._enforce_verification_contract(
                decision, milestone, trace
            )

        evidence = EvidenceRecord(
            id=f"evidence_{trace.step}_{len(state.evidence) + 1}",
            milestone_id=milestone.id,
            step=trace.step,
            tool_name=trace.tool_name,
            summary=decision.evidence_summary or decision.reason,
            raw_result=trace.result,
            source_urls=list(trace.source_urls),
            deterministic_checks=checks,
            verdict=decision.verdict,
        )
        event = VerifierEvent(
            kind=decision.verdict,
            milestone_id=milestone.id,
            step=trace.step,
            reason=decision.reason,
            evidence=evidence,
            facts={item.name: item.value for item in decision.facts},
            confidence=decision.confidence,
        )
        return event, decision

    @staticmethod
    def _deterministic_rejection_reason(
        checks: Dict[str, Any], trace: ExecutionTrace
    ) -> str:
        if trace.error:
            return f"Tool execution failed: {trace.error}"
        if not checks["nonempty_result"]:
            return "Tool returned no evidence."
        if not checks["no_error_marker"]:
            return "Tool result contains an explicit error or no-results marker."
        if checks["source_required"] and not checks["source_present"]:
            return "Verification contract requires a source URL or citation."
        return "Evidence failed deterministic validation."

    @staticmethod
    def _enforce_verification_contract(
        decision: VerificationDecision,
        milestone: Milestone,
        trace: ExecutionTrace,
    ) -> VerificationDecision:
        if decision.verdict == EventKind.INVALIDATED.value:
            if milestone.status != MilestoneStatus.VERIFIED:
                return VerificationDecision(
                    analysis=decision.analysis,
                    verdict=EventKind.REJECTED.value,
                    reason="A never-verified milestone cannot be invalidated.",
                    evidence_summary=decision.evidence_summary,
                    confidence=decision.confidence,
                    facts=[],
                )
        if decision.verdict == EventKind.SATISFIED.value:
            fact_names = {item.name for item in decision.facts}
            missing = set(milestone.verify.required_fact_names) - fact_names
            if missing:
                return VerificationDecision(
                    analysis=decision.analysis,
                    verdict=EventKind.REJECTED.value,
                    reason=f"Verifier omitted required facts: {sorted(missing)}",
                    evidence_summary=decision.evidence_summary,
                    confidence=decision.confidence,
                    facts=[],
                )
            if (
                milestone.verify.require_source
                and not trace.source_urls
                and trace.tool_name not in _CALCULATOR_TOOLS
            ):
                return VerificationDecision(
                    analysis=decision.analysis,
                    verdict=EventKind.REJECTED.value,
                    reason="Satisfied claim lacks required source provenance.",
                    evidence_summary=decision.evidence_summary,
                    confidence=decision.confidence,
                    facts=[],
                )
        return decision

    def attribute_failure(
        self,
        state: AgentState,
        milestone: Milestone,
        trace: ExecutionTrace,
        verification: VerificationDecision,
    ) -> FailureRecord:
        if trace.error:
            parsed = FailureAttributionOutput(
                attributed_to="tool_environment",
                reason=trace.error,
                recovery_hint="Retry with a healthy tool or choose another evidence source.",
            )
        else:
            prompt = _load_prompt("failure_attribution.txt").format(
                Question=state.question,
                Milestone=json.dumps(asdict(milestone), ensure_ascii=False, default=str),
                Trace=json.dumps(asdict(trace), ensure_ascii=False, default=str),
                Verification=json.dumps(
                    (
                        verification.model_dump()
                        if hasattr(verification, "model_dump")
                        else verification.dict()
                    ),
                    ensure_ascii=False,
                    default=str,
                ),
                Failures=json.dumps(
                    [asdict(item) for item in state.failures[-5:]],
                    ensure_ascii=False,
                    default=str,
                ),
            )
            response = self.llm_engine(
                prompt,
                response_format=FailureAttributionOutput,
                temperature=self.temperature,
                n=1,
            )
            parsed = parse_structured_response(response, FailureAttributionOutput)
        return FailureRecord(
            step=trace.step,
            milestone_id=milestone.id,
            attributed_to=parsed.attributed_to,
            reason=parsed.reason,
            recovery_hint=parsed.recovery_hint,
            tool_name=trace.tool_name,
        )

    def audit_final_answer(
        self, state: AgentState, answer: str
    ) -> FinalAuditDecision:
        if not state.can_attempt_done():
            return FinalAuditDecision(
                verdict="FAIL",
                reason="Not all leaf milestones are verified.",
                unsupported_claims=[],
            )
        prompt = _load_prompt("final_audit.txt").format(
            Question=state.question,
            Answer=answer,
            State=json.dumps(state.to_dict(), ensure_ascii=False, default=str),
        )
        response = self.auditor_llm_engine(
            prompt, response_format=FinalAuditDecision, temperature=self.temperature, n=1
        )
        return parse_structured_response(response, FinalAuditDecision)
