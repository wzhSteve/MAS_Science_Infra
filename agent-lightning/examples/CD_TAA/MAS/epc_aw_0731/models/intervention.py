"""Causal intervention ladder (L1 observe → L2a/L2b → L3 counterfactual)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.task_profile import SlotGate
from MAS.epc_aw.models.tool_router import ToolRouter, infer_subgoal_kind
from MAS.epc_aw.models.utils import make_json_serializable_truncated
from MAS.epc_aw.models.solver_types import VerificationResult, StepContext

MAX_PARAMETER_RETRIES = 3       # L2a 候选集轮数上限（默认跑 1 轮；预算允许最多 3 轮）
MAX_SWITCH_TOOL_ATTEMPTS = 2    # L2b 候选集轮数上限（每轮 ≤2 个替代工具）
MAX_DECOMPOSE_GOAL_ATTEMPTS = 1
MAX_MODIFY_STATE_ATTEMPTS = 1
MAX_REVISE_BELIEF_ATTEMPTS = 1
MAX_RECOVERY_OUTLINE_INJECTIONS = 1
INTERVENTION_LADDER: List[str] = [
    "retry_with_different_parameters",  # L2a: 参数候选集（Executor 侧）
    "switch_tool",                       # L2b: 替代工具候选集（Executor 侧）
    "revise_belief",                     # L3:  撤回冲突 claim + 重新分析题意（Planner 侧）
    "decompose_goal",                    # L3:  分解子目标（Planner 侧）
    "modify_state",                      # L3:  环境/前置条件修复
]
INTERVENTION_LIMITS: Dict[str, int] = {
    "retry_with_different_parameters": MAX_PARAMETER_RETRIES,
    "switch_tool": MAX_SWITCH_TOOL_ATTEMPTS,
    "revise_belief": MAX_REVISE_BELIEF_ATTEMPTS,
    "decompose_goal": MAX_DECOMPOSE_GOAL_ATTEMPTS,
    "modify_state": MAX_MODIFY_STATE_ATTEMPTS,
}
INTERVENTION_TARGET_VARIABLE: Dict[str, str] = {
    "retry_with_different_parameters": "executor",   # do(Command params) — also covers do(query)
    "switch_tool": "tool",                           # do(Tool)
    "revise_belief": "planner_belief",               # do(Belief)
    "decompose_goal": "task_graph",                  # do(Task decomposition)
    "modify_state": "environment",                   # do(Environment)
}
CAUSAL_HYPOTHESIS_TARGETS: List[str] = [
    "tool", "executor", "planner_belief", "task_graph", "environment",
    "query", "coverage", "external", "command",
]
PDF_ACCESS_ERROR = re.compile(
    r"Download is starting|Failed to load image from",
    re.I,
)
ARXIV_PDF_URL = re.compile(r"arxiv\.org/pdf/", re.I)
CANDIDATE_SET_SIZE = 3          # N candidates from one executor LLM call
MAX_VERIFY_PER_ROUND = 3        # hard cap on diagnoser LLM verifies per round


@dataclass
class StepInterventionState:
    """Per outline-step tracker for causal intervention attempts."""
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    intervention_attempts: int = 0
    failed_tools: List[str] = field(default_factory=list)
    # Change 3: record the (root_cause, target_variable) pair produced by
    # each fired intervention so the Counterfactual stage can detect a
    # no-op loop (same cause, same target → terminate). A small list of
    # prior diagnoses is enough; we cap at 8 to bound memory.
    root_cause_history: List[Tuple[str, str]] = field(default_factory=list)

    # ── Causal Hypothesis drift tracking (P0-③) ──────────────────────
    # Per outline-step last-observation state. When the current
    # observation differs from the stored value along ANY axis
    # (evidence_type / slots / hypothesis), the prior Causal Hypothesis
    # is falsified and must be invalidated before re-diagnosis. Keyed by
    # outline step key so replans (which mint new step keys) start clean.
    last_evidence_type: Dict[str, str] = field(default_factory=dict)
    last_slots: Dict[str, Set[str]] = field(default_factory=dict)
    last_hypothesis: Dict[str, str] = field(default_factory=dict)
    # last_cause is the (target_variable, reason) pair recorded alongside
    # root_cause_history; kept as a parallel per-step snapshot so the
    # Information Gain gate (P0-⑤) can compute ΔCause without scanning
    # the full history list.
    last_cause: Dict[str, str] = field(default_factory=dict)

    # ── Information Gain tracking (P0-⑤) ────────────────────────────
    # Per step: last action signature (command + tool) for Exploration
    # Gain, and a consecutive-no-knowledge-gain counter for the
    # Knowledge Gain Gate. The gate escalates when 2 consecutive
    # interventions produce zero knowledge gain (no ΔEvidence / ΔSlot /
    # ΔCause) — this is what stops the agent from burning budget on
    # near-identical retries (test_gaia_20260703_194121.log step 2-4).
    last_command: Dict[str, str] = field(default_factory=dict)
    last_tool: Dict[str, str] = field(default_factory=dict)
    consecutive_no_knowledge_gain: Dict[str, int] = field(default_factory=dict)

    # NOTE: Action Variable Gate state (P1-⑥) — ``last_planner_action``
    # and ``no_op_penalty`` — lives on the Solver instance, not here,
    # because decompose_goal mints a new step_key and a fresh tracker.
    # Per-step_key state would lose continuity across the replan
    # boundary; solver-level state survives it.

    def remaining(self, intervention: str) -> int:
        limit = INTERVENTION_LIMITS.get(intervention, 1)
        return max(0, limit - self.counts[intervention])

    def is_exhausted(self, intervention: str) -> bool:
        return self.remaining(intervention) <= 0

    def exhausted_set(self) -> Set[str]:
        return {name for name in INTERVENTION_LADDER if self.is_exhausted(name)}

    def record(self, intervention: str, amount: int = 1, tool: Optional[str] = None) -> None:
        self.counts[intervention] += amount
        if tool and intervention == "switch_tool":
            self.record_failure_tool(tool)

    def record_failure_tool(self, tool: Optional[str]) -> None:
        if tool and tool not in self.failed_tools:
            self.failed_tools.append(tool)

    def record_root_cause(self, root_cause: str, target_variable: str) -> None:
        """Append a diagnosis to the cause history (Change 3)."""
        if len(self.root_cause_history) >= 8:
            self.root_cause_history.pop(0)
        self.root_cause_history.append((str(root_cause), str(target_variable)))

    def cause_repeated(
        self,
        root_cause: str,
        target_variable: str,
        *,
        min_repeats: int = 2,
    ) -> bool:
        """True when the same (root_cause, target) pair has already been
        observed at least ``min_repeats`` times (counting the current
        occurrence, which is recorded before this check runs).

        This is the Cause-Changed gate: a Counterfactual that reproduces
        the same root_cause on the same target variable has not actually
        intervened on the cause and must terminate to avoid a cycle.

        Default ``min_repeats=2``: the first occurrence is the original
        diagnosis; the second occurrence means the intervening action
        (L2a/L2b) failed to change the cause → the upcoming
        Counterfactual must not repeat the same no-op.
        """
        if not root_cause or not target_variable:
            return False
        prior = sum(
            1 for rc, tv in self.root_cause_history
            if rc == root_cause and tv == target_variable
        )
        return prior >= min_repeats

    # ── Causal Hypothesis drift API (P0-③) ───────────────────────────

    def hypothesis_changed(
        self,
        step_key: str,
        cur_hypothesis: str,
        cur_evidence_type: str,
        cur_slots: Set[str],
    ) -> bool:
        """True when the current observation falsifies the prior hypothesis.

        Any of Evidence Drift / Slot Drift / Hypothesis (Reason) Drift
        counts as a falsification — the prior Causal Hypothesis was
        formed under a different observation regime and must be retired
        before a fresh diagnosis runs. This unifies the per-axis drift
        checks into a single ``hypothesis_changed`` gate so the caller
        does not accumulate ``if evidence_changed or slot_changed or
        ...`` branches as new axes are added.
        """
        if step_key not in self.last_hypothesis:
            # First observation on this step — nothing to falsify yet.
            return False
        if self.last_evidence_type.get(step_key) != cur_evidence_type:
            return True
        if self.last_slots.get(step_key) != cur_slots:
            return True
        if self.last_hypothesis.get(step_key) != cur_hypothesis:
            return True
        return False

    def invalidate_hypothesis(self, step_key: str) -> None:
        """Drop prior cause records for this step so re-diagnosis starts clean.

        ``root_cause_history`` is a flat list of (root_cause, target)
        pairs without a step binding, so we cannot selectively clear it
        by step. Instead we drop the whole history — this is safe because
        invalidation only fires mid-step (the very next diagnosis will
        repopulate it), and the Cause-Changed gate is meaningful only
        within a single outline step's intervention sequence.
        """
        self.root_cause_history.clear()
        self.last_hypothesis.pop(step_key, None)
        self.last_cause.pop(step_key, None)

    def snapshot_hypothesis(
        self,
        step_key: str,
        cur_hypothesis: str,
        cur_evidence_type: str,
        cur_slots: Set[str],
        cur_cause: str,
    ) -> None:
        """Record the current observation as the new baseline for drift."""
        self.last_evidence_type[step_key] = cur_evidence_type
        self.last_slots[step_key] = set(cur_slots)
        self.last_hypothesis[step_key] = cur_hypothesis
        self.last_cause[step_key] = cur_cause

    def reset_parameter_budget(self) -> None:
        self.counts["retry_with_different_parameters"] = 0

    def force_escalate(self, current_intervention: str) -> str:
        """Mark ``current_intervention`` as exhausted and return the next
        non-exhausted ladder step (P0-⑤ Knowledge Gain Gate).

        This bypasses the natural budget-counting path: when 2 consecutive
        interventions yield zero knowledge gain, continuing the same
        intervention is provably wasteful, so we artificially exhaust it
        and let ``resolve()`` pick the next ladder rung. Returns the
        resolved next intervention (or the last rung if all are exhausted).
        """
        if current_intervention in INTERVENTION_LIMITS:
            limit = INTERVENTION_LIMITS[current_intervention]
            if self.counts[current_intervention] < limit:
                self.counts[current_intervention] = limit
        return self.resolve(current_intervention)

    def record_exploration(self, step_key: str, command: str, tool_name: str) -> None:
        """Snapshot the last action signature for Exploration Gain."""
        self.last_command[step_key] = command
        self.last_tool[step_key] = tool_name

    def bump_no_knowledge_gain(self, step_key: str) -> int:
        self.consecutive_no_knowledge_gain[step_key] = (
            self.consecutive_no_knowledge_gain.get(step_key, 0) + 1
        )
        return self.consecutive_no_knowledge_gain[step_key]

    def reset_no_knowledge_gain(self, step_key: str) -> None:
        self.consecutive_no_knowledge_gain[step_key] = 0

    def resolve(self, raw_recommendation: str) -> str:
        """Return raw recommendation if budget remains, else next ladder step."""
        if raw_recommendation not in INTERVENTION_LADDER:
            return raw_recommendation
        if not self.is_exhausted(raw_recommendation):
            return raw_recommendation
        start = INTERVENTION_LADDER.index(raw_recommendation)
        for candidate in INTERVENTION_LADDER[start + 1:]:
            if not self.is_exhausted(candidate):
                return candidate
        return INTERVENTION_LADDER[-1]

    def to_context(self) -> Dict[str, Any]:
        return {
            "exhausted": self.exhausted_set(),
            "failed_tools": list(self.failed_tools),
            "counts": dict(self.counts),
            "intervention_attempts": self.intervention_attempts,
        }

@dataclass
class FailureContext:
    """Level 1 Observation (Pearl Level 1): a structured snapshot of the failed
    execution step, assembled purely from existing diagnostic outputs.

    Observation makes NO decision and triggers NO LLM call. It only collects:
    current subgoal, tool, parameters, tool output, verifier analysis, failure
    patterns, and the exhausted/failed-tool state from the per-step intervention
    tracker. Downstream Level 2 (Intervention) and Level 3 (Counterfactual)
    read this context instead of re-deriving it.
    """
    step_key: str
    target_information: str
    subgoal: str
    tool: str
    parameters: str
    tool_output: Any
    verification_analysis: str
    failure_patterns: Dict[str, Any]
    evidence_type: str
    tool_appropriate: bool
    exhausted: Set[str]
    failed_tools: List[str]


class InterventionMixin:
    """Mixin: L1 observation + L2a/L2b intervention + L3 counterfactual."""

    CANDIDATE_SET_SIZE = CANDIDATE_SET_SIZE
    MAX_PROGRAMMATIC_PERTURBS = 2
    MAX_VERIFY_PER_ROUND = MAX_VERIFY_PER_ROUND
    CANDIDATE_PARALLEL_WORKERS = 3
    # Soft coverage misses: same reason may reoccur while query rewrites continue.
    SOFT_REPEATED_CAUSES = frozenset({
        "low_recall",
        "no_match",
        "over_constrained",
        "source_not_indexed",
    })

    def _emit_layer_telemetry(
        self,
        json_data: Dict[str, Any],
        event: str,
        **fields: Any,
    ) -> None:
        """Default telemetry sink; Solver may override with ablation gating."""
        ablation = getattr(self, "ablation", None)
        if ablation is not None and not ablation.emit_layer_telemetry:
            return
        json_data.setdefault("layer_telemetry", []).append({"event": event, **fields})

    @staticmethod
    def _normalize_cause_reason_key(reason: str, description: str = "") -> str:
        """Prefer structured reason enum; fall back to known tokens in prose."""
        r = str(reason or "").strip()
        if r in InterventionMixin.SOFT_REPEATED_CAUSES or (
            r and " " not in r and len(r) <= 40
        ):
            # Enum-like reason (snake_case token) — use as Cause-Changed key.
            if r:
                return r
        blob = f"{r} {description}".lower()
        for token in (
            "low_recall",
            "no_match",
            "over_constrained",
            "source_not_indexed",
            "paywalled",
            "empty_compute_result",
            "wrong_tool_class",
            "generic_keyword",
            "entity_missing",
        ):
            if token in blob or token.replace("_", " ") in blob:
                return token
        # Last resort: short description snippet (legacy hard-cause compare).
        desc = str(description or reason or "").strip()
        return desc[:120] if desc else ""

    def _get_intervention_state(
        self,
        trackers: Dict[str, StepInterventionState],
        step_key: str,
    ) -> StepInterventionState:
        if step_key not in trackers:
            trackers[step_key] = StepInterventionState()
        return trackers[step_key]

    def _apply_escalation_to_signal(
        self,
        signal: Dict[str, Any],
        tracker: StepInterventionState,
    ) -> Dict[str, Any]:
        """Escalate recommendation when current intervention budget is exhausted."""
        raw = signal.get("recommendation", "")
        resolved = tracker.resolve(raw)
        if resolved == raw:
            return signal
        escalated = dict(signal)
        escalated["recommendation"] = resolved
        escalated["escalated_from"] = raw
        escalated["escalation_reason"] = (
            f"Intervention '{raw}' exhausted ({tracker.counts.get(raw, 0)}/"
            f"{INTERVENTION_LIMITS.get(raw, '?')} attempts on this outline step)"
        )
        print(f"\n==> ⬆️ Intervention Escalation: {raw} → {resolved}")
        print(f"    {escalated['escalation_reason']}\n")
        return escalated

    def _build_diagnostic_signal_for_replan(
        self,
        base_signal: Optional[Dict[str, Any]],
        recommendation: str,
        step_ctx: StepContext,
        tracker: StepInterventionState,
        analysis: str = "",
        question: str = "",
    ) -> Dict[str, Any]:
        signal = dict(base_signal or {})
        signal.update({
            "triggered": True,
            "recommendation": recommendation,
            "tool": step_ctx.tool_name,
            "sub_goal": step_ctx.sub_goal,
            "analysis": analysis or signal.get("analysis", ""),
            "failed_tools": list(tracker.failed_tools),
        })
        if recommendation == "switch_tool":
            signal = self._enrich_diagnostic_signal(signal, step_ctx.target_information, question, step_ctx=step_ctx)
        return signal

    def _apply_decompose_goal(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
        tracker: Optional[StepInterventionState] = None,
    ) -> bool:
        """L3: break down the stuck sub-goal by updating the outline."""
        print(f"\n==> 🔀 L3 Intervention: Decompose Goal (outline step {step_ctx.step_key})\n")
        current_step_num = int(step_ctx.step_key) if step_ctx.step_key.isdigit() else 1
        prev_outline = dict(self.system_memory.get_outline() or {})
        failure_digest = ""
        if tracker:
            failure_digest = self._build_failure_digest(json_data, tracker, exec_step)
            tracker.reset_parameter_budget()
        updated = self.diagnoser.update_outline(
            question=question,
            context_verification=(
                f"{verification.analysis}\n\n"
                "[INTERVENTION: decompose_goal] The current sub-goal failed after exhausting "
                "parameter retries and tool switches. Decompose it into smaller achievable steps. "
                "NEVER return an empty ExecutionOutline.\n"
                "FORBIDDEN PATHS (from execution history — do NOT repeat):\n"
                f"{failure_digest}\n"
                "Do NOT suggest Screenshot_Tool or Vision_OCR_Tool on arxiv.org/pdf/* URLs."
            ),
            target_information=step_ctx.target_information,
            outline=prev_outline,
            result_executor=step_ctx.result_executor,
            current_step=current_step_num,
            obtained_information=self.system_memory.get_obtained_information_for_prompt(),
            toolbox_metadata=self.system_memory.get_toolbox_metadata(),
            execution_memory=self._build_execution_memory_summary(json_data, exec_step),
        )
        updated = self._sanitize_outline_update(
            prev_outline, updated, current_step_num, "CONTINUE", question, verification,
        )
        normalize = lambda outline: re.sub(
            r"\s+", " ", json.dumps(outline or {}, sort_keys=True, ensure_ascii=False).lower(),
        ).strip()
        changed = normalize(updated) != normalize(prev_outline)
        if not changed:
            print(
                "\n==> 🛑 Decompose Goal produced a semantically unchanged outline; "
                "rejecting no-op replan\n"
            )
            return False
        self.system_memory.set_outline(updated)
        # v3: grant one bonus step so the corrected outline can execute even if
        # max_steps is about to be reached (e.g. pid=2 fixed formula never ran).
        self._decompose_bonus = 1
        if self.verbose:
            print(f"[Decomposed Outline]:\n{json.dumps(updated, indent=4)}")
        return True

    def _apply_modify_state(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
    ) -> Dict[str, Any]:
        """L3: signal environment/prerequisite change before retrying."""
        print(f"\n==> 🔧 L3 Intervention: Modify State (outline step {step_ctx.step_key})\n")
        signal: Dict[str, Any] = {
            "triggered": True,
            "recommendation": "modify_state",
            "tool": step_ctx.tool_name,
            "sub_goal": step_ctx.sub_goal,
            "analysis": verification.analysis,
            "state_action": (
                "Address environment blocker (API quota, auth, missing prerequisite). "
                "Try an alternative data source or simplify the sub-goal."
            ),
        }

        result_str = str(step_ctx.result_executor)
        cmd = step_ctx.command
        if self._is_pdf_access_error(result_str):
            url = self._extract_url_param(cmd)
            image_match = re.search(r'image_input=["\']([^"\']+)["\']', cmd)
            if url == "N/A" and image_match:
                url = image_match.group(1)
            if url != "N/A" and (ARXIV_PDF_URL.search(url) or url.rstrip("/").endswith(".pdf")):
                abs_url = self._pdf_url_to_abs_url(url)
                signal["state_mutations"] = {
                    "url_rewrite": {"from": url, "to": abs_url},
                    "preferred_url": abs_url,
                    "preferred_tools": ["Google_Search_Tool", "Web_Search_Tool"],
                    "forbidden_tools": ["Screenshot_Tool", "Vision_OCR_Tool"],
                }
                signal["state_action"] = (
                    f"PDF direct access failed for {url}. "
                    f"Use text search on {abs_url} or Google_Search_Tool instead of "
                    "Screenshot_Tool/Vision_OCR_Tool on PDF URLs."
                )
                print(f"\n==> 🔄 State mutation: PDF URL → abs page ({abs_url})\n")

        return signal

    def _enrich_diagnostic_signal(
        self,
        signal: Optional[Dict[str, Any]],
        target_information: str,
        question: str = "",
        step_ctx: Optional[StepContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attach alternative tool to switch_tool signals for Planner."""
        if not signal or signal.get("recommendation") != "switch_tool":
            return signal
        enriched = dict(signal)
        failed_tool = enriched.get("tool")
        outline_tool = Planner._extract_tool_from_target(target_information)

        patterns = enriched.get("failure_patterns") or {}
        if patterns.get("exhaustiveness_unmet") and failed_tool == "Wikipedia_Search_Tool":
            # Snippet-only Wikipedia result on a count/enumerate task: deep-fetch
            # the full page via Web_Search_Tool with the Wikipedia URL.
            enriched["suggested_tool"] = "Web_Search_Tool"
            if step_ctx is not None:
                url = self._extract_wikipedia_url(step_ctx.result_executor)
                if url:
                    enriched["web_search_url"] = url
            return enriched

        alt = self._suggest_alternative_tool(failed_tool, target_information, question)
        if alt and alt != failed_tool:
            enriched["suggested_tool"] = alt
        elif outline_tool and outline_tool != failed_tool:
            enriched["suggested_tool"] = outline_tool

        if "usgs" in f"{target_information} {question}".lower():
            enriched["web_search_url"] = "https://nas.er.usgs.gov/"
        return enriched

    def _observe_failure(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
        tracker: StepInterventionState,
    ) -> FailureContext:
        """从既有诊断输出装配结构化 FailureContext。

        纯数据收集 —— 不做决策、不调用 LLM。复用：
          * step_ctx（subgoal / tool / parameters / tool 输出）
          * verification（analysis / failure_patterns（来自 diagnostic_signal）/ evidence_type / tool_appropriate）
          * tracker（已耗尽的介入 / 已失败的 tool 列表）
        """
        signal = verification.diagnostic_signal or {}
        failure_patterns = signal.get("failure_patterns") or {}
        return FailureContext(
            step_key=step_ctx.step_key,
            target_information=step_ctx.target_information,
            subgoal=step_ctx.sub_goal,
            tool=step_ctx.tool_name,
            parameters=step_ctx.command,
            tool_output=step_ctx.result_executor,
            verification_analysis=verification.analysis or "",
            failure_patterns=failure_patterns,
            evidence_type=verification.evidence_type,
            tool_appropriate=verification.tool_appropriate,
            exhausted=tracker.exhausted_set(),
            failed_tools=list(tracker.failed_tools),
        )

    def _build_causal_effect(
        self,
        step_ctx: "StepContext",
        success: bool,
        attempt_seq: int,
        diagnostic_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symptom = None
        l1 = "成功" if success else "参数问题"
        if diagnostic_signal:
            patterns = diagnostic_signal.get("failure_patterns") or {}
            active = [k for k, v in patterns.items() if v]
            if active:
                symptom = ", ".join(active)
                l1 = diagnostic_signal.get("recommendation", l1)

        return {
            "attempt_seq": attempt_seq,
            "tool": step_ctx.tool_name,
            "parameter": str(step_ctx.command)[:200],
            "result": {
                "success": success,
                "preview": str(step_ctx.result_executor)[:500],
            },
            "symptom": symptom,
            "L1_diagnosis": l1,
        }

    def _record_causal_effect(
        self,
        step_ctx: "StepContext",
        exec_step: int,
        success: bool,
        verification: "VerificationResult",
        diagnostic_signal: Optional[Dict[str, Any]] = None,
        attempt_seq: int = 1,
        finalize: bool = False,
        contrast_failed_parameter: Optional[str] = None,
    ) -> None:
        """Append one effect to the causal trace for this outline step + subgoal."""
        state = self.system_memory.infer_state_type()
        L2: Dict[str, Any] = {}
        if success:
            L2 = {
                "root_cause": "success",
                "recommendation": verification.task_conclusion or "CONTINUE",
            }
        elif diagnostic_signal:
            L2 = {
                "root_cause": str(diagnostic_signal.get("analysis", ""))[:200],
                "recommendation": diagnostic_signal.get("recommendation", ""),
            }

        self.system_memory.append_causal_effect(
            outline_step=step_ctx.step_key,
            exec_step=exec_step,
            state=state,
            subgoal=step_ctx.sub_goal,
            effect=self._build_causal_effect(
                step_ctx, success, attempt_seq, diagnostic_signal,
            ),
            L2_diagnosis=L2 if (success or diagnostic_signal) else None,
            finalize=finalize,
            contrast_failed_parameter=contrast_failed_parameter,
        )

    def _parse_slots_from_obtained(self, verification: VerificationResult) -> Set[str]:
        """Extract a comparable slot signature from a VerificationResult.

        Returns a set of ``"slot=value"`` strings (sorted-stable via set
        semantics). Used by the Hypothesis Drift gate (P0-③) and the
        Knowledge Gain gate (P0-⑤) to detect Slot Drift — e.g.
        ``base_count=8000`` superseded by ``base_count=12000`` counts as
        a falsifying change even when evidence_type stays DIRECT.

        Empty set when no slots are filled (typical for ABSENCE / ERROR).
        """
        sig: Set[str] = set()
        try:
            for upd in verification.slot_updates or []:
                if isinstance(upd, dict) and upd.get("filled") and upd.get("value"):
                    sig.add(f"{upd.get('slot', '')}={upd.get('value', '')}")
        except Exception:
            return set()
        return sig

    def _action_variable_signature(self, step_ctx: StepContext) -> Tuple[str, str, str]:
        """Extract the (tool, query, url) action signature from a StepContext.

        Used by the Action Variable Gate (P1-⑥) to detect Planner-side
        interventions that did NOT change any action variable — i.e. the
        LLM merely rephrased the outline but emitted the same tool/query.
        Comparison is exact-string (no embedding / Jaccard) per the
        plan's "keep it lightweight" requirement.
        """
        tool = step_ctx.tool_name or ""
        query = self._extract_query_param(step_ctx.command) or ""
        url = self._extract_url_param(step_ctx.command) or ""
        return (tool, query, url)

    def _compute_intervention_gains(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
        diagnostic_signal: Dict[str, Any],
        tracker: StepInterventionState,
        step_key: str,
    ) -> Tuple[int, int]:
        """Return (exploration_gain, knowledge_gain) for the current attempt.

        Exploration Gain (ΔAction): did the agent actually try something
        different? Compares the current command/tool against the last
        recorded action signature. Recorded but NOT used for escalation —
        exploration without learning is allowed (the agent may need
        several tries to find a productive query).

        Knowledge Gain (ΔEvidence ∪ ΔSlot ∪ ΔCause): did the agent learn
        something new? Any of evidence_type change, slot set change, or
        Causal Hypothesis (target_variable.reason) change counts as 1.
        This is the escalation trigger: 2 consecutive zero-knowledge-gain
        interventions mean the current intervention class is provably
        uninformative and must escalate.

        First observation on a step (no prior snapshot) returns
        knowledge_gain=1 so the gate does not fire prematurely.
        """
        if not step_key or step_key not in tracker.last_hypothesis:
            # First observation on this step — everything is new.
            return 1, 1

        exploration = 0
        if step_ctx.command != tracker.last_command.get(step_key, ""):
            exploration += 1
        if step_ctx.tool_name != tracker.last_tool.get(step_key, ""):
            exploration += 1

        knowledge = 0
        cur_et = (
            diagnostic_signal.get("evidence_type")
            or verification.evidence_type
            or ""
        )
        if cur_et != tracker.last_evidence_type.get(step_key):
            knowledge += 1
        cur_slots = self._parse_slots_from_obtained(verification)
        if cur_slots != tracker.last_slots.get(step_key):
            knowledge += 1
        _hypo = diagnostic_signal.get("causal_hypothesis") or {}
        cur_hypo = f"{_hypo.get('target_variable', '')}.{_hypo.get('reason', '')}"
        if cur_hypo != tracker.last_hypothesis.get(step_key):
            knowledge += 1

        return exploration, knowledge

    def _record_failure_trace(
        self,
        step_count: int,
        step_ctx: StepContext,
        diagnostic_signal: Optional[Dict[str, Any]],
        attempt_seq: int = 1,
    ) -> None:
        self._record_causal_effect(
            step_ctx, step_count, success=False,
            verification=VerificationResult(
                analysis="", step_conclusion="SUBGOAL_INCOMPLETE",
                info_flag=False, obtained_info="", task_conclusion=None,
                diagnostic_signal=diagnostic_signal, subgoal_complete=False,
            ),
            diagnostic_signal=diagnostic_signal,
            attempt_seq=attempt_seq,
            finalize=False,
        )

    def _record_success_trace(
        self,
        step_count: int,
        step_ctx: StepContext,
        verification: VerificationResult,
        failed_parameter: Optional[str] = None,
        attempt_seq: int = 1,
    ) -> None:
        self._record_causal_effect(
            step_ctx, step_count, success=True,
            verification=verification,
            attempt_seq=attempt_seq,
            finalize=True,
            contrast_failed_parameter=failed_parameter,
        )

    def _apply_parameter_guidance(
        self,
        diagnostic_signal: Dict[str, Any],
        failed_command: str,
        tried_params: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        signal = dict(diagnostic_signal)
        failed_param = self._extract_query_param(failed_command)
        guidance = signal.setdefault("parameter_guidance", {})

        tried_queries: Set[str] = set()
        for fp in (tried_params or set()):
            tried_queries.add(fp.split("|")[0])
        if failed_param != "N/A":
            tried_queries.add(self._normalize_query_param(failed_param))

        if tried_queries:
            guidance["failed_parameter"] = (
                failed_param if failed_param != "N/A" else sorted(tried_queries)[0]
            )
            guidance["avoid_exact_match"] = (
                "Do NOT reuse any of these exact queries: "
                + ", ".join(f'"{q}"' for q in sorted(tried_queries))
            )
            guidance["tried_queries"] = sorted(tried_queries)
        elif failed_param != "N/A":
            guidance["failed_parameter"] = failed_param
            guidance["avoid_exact_match"] = f'Do NOT use query="{failed_param}"'

        return signal

    def _resolve_unique_retry_command(
        self,
        question: str,
        image_path: Optional[str],
        current_ctx: "StepContext",
        step_count: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Dict[str, Any],
        tried_params: Set[str],
        attempt: int,
    ) -> Tuple[str, str, str, str]:
        """
        Generate a command whose parameters differ from all tried fingerprints.
        Re-generates or programmatically perturbs — never returns a duplicate.
        """
        max_regen = 3
        last_command = current_ctx.command
        last_query = self._extract_query_param(last_command)
        tool_name = current_ctx.tool_name

        for regen in range(max_regen):
            signal = self._apply_parameter_guidance(
                diagnostic_signal, current_ctx.first_attempt_command, tried_params,
            )
            command, analysis, explanation = self._generate_executor_command(
                question, image_path,
                current_ctx.context, current_ctx.sub_goal, tool_name,
                step_count, json_data, signal,
            )

            if self.verbose:
                print(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
                print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")

            curr_query = self._extract_query_param(command)
            fp = self._param_fingerprint(command, tool_name)

            print(f"\n[Parameter Comparison]: '{last_query}' → '{curr_query}'")

            if fp not in tried_params:
                return command, analysis, explanation, curr_query

            print("  ⚠️ WARNING: Parameters already tried — perturbing without duplicate execution")
            if tool_name == "Web_Search_Tool":
                command, curr_query = self._perturb_web_search_command(
                    command, curr_query if curr_query != "N/A" else last_query,
                    attempt * 10 + regen, tried_params,
                )
            elif tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
                url = self._extract_url_param(command)
                tried_urls = {t for t in tried_params if "/" in t or t.startswith("http")}
                perturbed_url = self._perturb_url_param(url, attempt * 10 + regen, tried_urls)
                command = self._sanitize_command_for_tool(
                    tool_name, self._inject_url_into_command(command, perturbed_url),
                )
                curr_query = perturbed_url
            else:
                perturbed = self._perturb_query_param(
                    curr_query if curr_query != "N/A" else last_query,
                    attempt * 10 + regen,
                    {t.split("|")[0] for t in tried_params},
                    sub_goal=current_ctx.sub_goal,
                    target_information=current_ctx.target_information,
                )
                command = self._inject_query_into_command(command, perturbed)
                curr_query = perturbed
            print(f"  🔀 Perturbed: query='{curr_query}'")

            fp = self._param_fingerprint(command, tool_name)
            if fp not in tried_params:
                return command, analysis, explanation, curr_query

            last_query = curr_query

        if tool_name == "Web_Search_Tool":
            command, curr_query = self._perturb_web_search_command(
                last_command, last_query, attempt * 100, tried_params,
            )
        elif tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
            url = self._extract_url_param(last_command)
            tried_urls = {t for t in tried_params if "/" in t or t.startswith("http")}
            perturbed_url = self._perturb_url_param(url, attempt * 100, tried_urls)
            command = self._sanitize_command_for_tool(
                tool_name, self._inject_url_into_command(last_command, perturbed_url),
            )
            curr_query = perturbed_url
        else:
            perturbed = self._perturb_query_param(
                last_query, attempt * 100, set(),
                sub_goal=current_ctx.sub_goal,
                target_information=current_ctx.target_information,
            )
            command = self._inject_query_into_command(last_command, perturbed)
            curr_query = perturbed
        print(f"  🔀 Forced perturb after max regen: '{curr_query}'")
        return command, "", "", curr_query

    def _retry_with_parameter_variation(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        exec_step: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Dict[str, Any],
        tracker: StepInterventionState,
    ) -> Tuple[StepContext, VerificationResult, int, Dict[str, Any]]:
        """[DEPRECATED] Superseded by `_run_intervention_level2a` (candidate-set).

        Retained as a thin backward-compatible shim that delegates to the new
        Level 2a candidate-set runner. The legacy sequential single-candidate
        retry loop has been removed; Level 2a generates N candidates in ONE
        executor LLM call, executes them in parallel, and verifies them with a
        rule pre-filter + capped diagnoser verifies (first COMPLETE wins).
        Returns (ctx, verification, exec_step, replan_signal) shaped like the
        old contract so any external caller keeps working.
        """
        import warnings
        warnings.warn(
            "_retry_with_parameter_variation is deprecated; use _run_intervention_level2a.",
            DeprecationWarning,
            stacklevel=2,
        )
        action, new_ctx, verification, new_exec_step, replan_signal = self._run_intervention_level2a(
            question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
        )
        if action == "complete":
            return new_ctx, verification, new_exec_step, verification.diagnostic_signal or {}
        return new_ctx, verification, new_exec_step, replan_signal or diagnostic_signal

    def _generate_parameter_candidate_set(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        signal: Dict[str, Any],
        tried_params: Set[str],
        n: int = CANDIDATE_SET_SIZE,
    ) -> List[str]:
        """【L2a 核心成本控制】用「1 次 executor LLM」生成一组候选命令。

        成本中性策略（关键）：
        1. 单次 executor LLM 调用（n_candidates=N）一次返回至多 N 条不同命令 ——
           把「多次重试」压成「一次生成多条」，避免 LLM 调用随重试线性增长。
        2. 再叠加至多 MAX_PROGRAMMATIC_PERTURBS 条「0-LLM」程序化扰动
           （复用既有 `_perturb_query_param` / `_perturb_web_search_command`）。
        3. 用既有 `_param_fingerprint` 去重，保证不重复已试参数。
        """
        candidates: List[str] = []
        tool_name = step_ctx.tool_name
        last_command = step_ctx.command
        last_query = self._extract_query_param(last_command)

        # Resolve a real seed query when the first command was missing/unparsed.
        seed = last_query if last_query not in ("", "N/A") else ""
        if self._is_garbage_retry_query(seed):
            seed = self._seed_query_from_step(
                step_ctx.sub_goal, step_ctx.target_information, question,
            )
        # Inject template when last_command has no tool.execute(...) skeleton.
        inject_base = last_command
        if (
            not inject_base
            or "tool.execute" not in str(inject_base)
            or re.search(r"no command found", str(inject_base), re.I)
        ):
            inject_base = (
                f"execution = tool.execute(query={repr(seed or 'placeholder')})"
            )

        def _query_usable(cmd: str) -> bool:
            q = self._extract_query_param(cmd)
            if self._is_garbage_retry_query(q) or len(str(q).strip()) < 5:
                return False
            sg_terms = self._significant_query_terms(step_ctx.sub_goal)
            q_terms = self._significant_query_terms(q)
            if sg_terms and q_terms and not (sg_terms & q_terms):
                # Also allow overlap with seed / target quoted phrase.
                seed_terms = self._significant_query_terms(seed)
                if seed_terms and (seed_terms & q_terms):
                    return True
                return False
            return True

        # 1. 单次 LLM 调用 → N 条候选命令。
        try:
            raw = self.executor.generate_tool_command(
                question, image_path, step_ctx.context, step_ctx.sub_goal,
                tool_name, self.system_memory.toolbox_metadata[tool_name],
                0, None, signal, n_candidates=n,
            )
            llm_commands = self.executor._extract_multiple_commands(raw)
        except Exception as e:
            print(f"\n==> ⚠️ Multi-candidate generation failed: {e}; falling back to perturbations\n")
            llm_commands = []

        for cmd in llm_commands:
            if tool_name == "Web_Search_Tool":
                cmd = self._ensure_web_search_command(cmd, step_ctx.sub_goal, step_ctx.context)
            cmd = self._align_command_with_subgoal(
                tool_name, cmd, question, step_ctx.context, step_ctx.sub_goal,
            )
            cmd = self._sanitize_command_for_tool(tool_name, cmd)
            if not _query_usable(cmd):
                continue
            fp = self._param_fingerprint(cmd, tool_name)
            if fp not in tried_params and fp not in {
                self._param_fingerprint(c, tool_name) for c in candidates
            }:
                candidates.append(cmd)

        # 2. 0-LLM 程序化扰动补齐候选数。
        perturb_attempts = 0
        attempt_seed = 1
        while len(candidates) < n + self.MAX_PROGRAMMATIC_PERTURBS and perturb_attempts < self.MAX_PROGRAMMATIC_PERTURBS:
            perturbed_cmd = None
            try:
                if tool_name == "Web_Search_Tool":
                    perturbed_cmd, _ = self._perturb_web_search_command(
                        inject_base, seed,
                        attempt_seed, tried_params,
                    )
                elif tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
                    url = self._extract_url_param(last_command)
                    tried_urls = {t for t in tried_params if "/" in t or t.startswith("http")}
                    perturbed_url = self._perturb_url_param(url, attempt_seed, tried_urls)
                    perturbed_cmd = self._sanitize_command_for_tool(
                        tool_name, self._inject_url_into_command(inject_base, perturbed_url),
                    )
                else:
                    perturbed = self._perturb_query_param(
                        seed,
                        attempt_seed,
                        {t.split("|")[0] for t in tried_params},
                        sub_goal=step_ctx.sub_goal,
                        target_information=step_ctx.target_information,
                        question=question,
                    )
                    if not perturbed:
                        break
                    perturbed_cmd = self._inject_query_into_command(inject_base, perturbed)
            except Exception:
                perturbed_cmd = None
            attempt_seed += 10
            if not perturbed_cmd:
                break
            if not _query_usable(perturbed_cmd):
                perturb_attempts += 1
                continue
            fp = self._param_fingerprint(perturbed_cmd, tool_name)
            if fp in tried_params or fp in {
                self._param_fingerprint(c, tool_name) for c in candidates
            }:
                perturb_attempts += 1
                continue
            candidates.append(perturbed_cmd)
            perturb_attempts += 1

        if not candidates and seed:
            # Last-resort: seed itself as a single executable command.
            forced = self._perturb_query_param(
                seed, 100, set(),
                sub_goal=step_ctx.sub_goal,
                target_information=step_ctx.target_information,
                question=question,
            ) or seed
            candidates.append(self._inject_query_into_command(inject_base, forced))
        return candidates

    def _execute_candidate_set(
        self,
        tool_name: str,
        commands: List[str],
        step_count: int,
        json_data: Dict[str, Any],
    ) -> List[Tuple[str, Any]]:
        """在主线程串行执行候选命令。

        Executor 使用 SIGALRM 做超时保护，而 Python signal 只能在主线程注册。
        因此不能用 ThreadPoolExecutor 扇出候选；候选数量本身有严格上限。
        每个候选复用 `_validate_command` + `executor.execute_tool_command`。
        返回 [(command, result), ...]，保持输入顺序。
        """
        results: List[Tuple[str, Any]] = []
        if not commands:
            return results

        def _run_one(idx: int, cmd: str) -> Tuple[int, str, Any]:
            validation_error = self._validate_command(tool_name, cmd)
            if validation_error:
                print(f"\n==> ⚠️ Candidate {idx + 1} validation failed: {validation_error}\n")
                res = f"Error in execute_tool_command: {validation_error}"
                json_data[f"tool_result_{step_count}_cand{idx}"] = res
                return idx, cmd, res
            # 注意：此处绕过 _execute_generated_command（它写共享键 `tool_result_{step_count}`），
            # 否则并行候选会竞争同一个键互相覆盖。改用候选专属键 `_cand{idx}`。
            try:
                res = self.executor.execute_tool_command(tool_name, cmd)
                res = make_json_serializable_truncated(res)
            except Exception as e:
                res = f"Error in execute_tool_command: {e}"
            json_data[f"tool_result_{step_count}_cand{idx}"] = res
            return idx, cmd, res

        for i, cmd in enumerate(commands):
            _, candidate_command, result = _run_one(i, cmd)
            results.append((candidate_command, result))
        return results

    def _candidate_passes_prefilter(self, tool_name: str, result: Any) -> bool:
        """规则预过滤（0 次 LLM）：对结果明显损坏的候选跳过 LLM 验证。
        复用 Diagnoser._has_usable_result + Solver._is_pdf_access_error + 错误前缀检查。
        """
        if self._is_pdf_access_error(result):
            return False
        text = str(result or "").strip().lower()
        if text.startswith("error") or text.startswith("error in execute_tool_command"):
            return False
        if not self.diagnoser._has_usable_result(result):
            return False
        return True

    def _verify_candidate_set(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        candidates_with_results: List[Tuple[str, Any]],
        tracker: StepInterventionState,
        exec_step: int,
        max_verify: int = MAX_VERIFY_PER_ROUND,
    ) -> Tuple[Optional[Tuple[str, Any, VerificationResult, int]], int]:
        """用规则预过滤 + 复用 `_run_verification` 验证候选。

        流程：规则预过滤（0 LLM）→ 按质量排序 → 串行复用 diagnoser 验证
        → 首个 SUBGOAL_COMPLETE 即胜出并停止。`max_verify` 上限封顶每轮 diagnoser LLM 成本。
        返回 (winner_or_None, exec_step)。候选验证只增加 tracker.intervention_attempts，
        不消耗主循环的 exec_step 预算。
        """
        passing = [
            (cmd, res) for (cmd, res) in candidates_with_results
            if self._candidate_passes_prefilter(step_ctx.tool_name, res)
        ]
        failing = [
            (cmd, res) for (cmd, res) in candidates_with_results
            if not self._candidate_passes_prefilter(step_ctx.tool_name, res)
        ]
        ordered = passing + failing  # 优先验证有希望的候选

        verified = 0
        current_exec_step = exec_step
        for cmd, res in ordered:
            if verified >= max_verify:
                break
            tracker.intervention_attempts += 1
            verified += 1
            cand_ctx = StepContext(
                step_key=step_ctx.step_key,
                target_information=step_ctx.target_information,
                context=step_ctx.context,
                sub_goal=step_ctx.sub_goal,
                tool_name=step_ctx.tool_name,
                command=cmd,
                result_executor=res,
                first_attempt_command=step_ctx.first_attempt_command,
            )
            # LLM 调用：diagnoser 验证该候选结果（受 max_verify 上限封顶）。
            verification = self._run_verification(
                question, image_path, cand_ctx, current_exec_step,
                label=(
                    f"Candidate Verification ({verified}/{max_verify}; "
                    f"intervention_attempt={tracker.intervention_attempts})"
                ),
                intervention_context=tracker.to_context(),
            )
            if verification.step_conclusion == "SUBGOAL_COMPLETE":
                print(f"\n==> ✅ Candidate verified COMPLETE on attempt {verified}\n")
                return (cmd, res, verification, current_exec_step), current_exec_step
        return None, current_exec_step

    def _run_intervention_level2a(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        exec_step: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Dict[str, Any],
        tracker: StepInterventionState,
    ) -> Tuple[str, StepContext, VerificationResult, int, Optional[Dict[str, Any]]]:
        """Level 2a：在「当前工具」上做参数候选集干预。

        1 轮 = 1 次 executor LLM（生成 N 候选）+ 并行执行 + 受 max_verify 封顶的验证。
        counts 语义 = 候选集轮数，而非单次尝试。任一候选 COMPLETE 即记录成功路径并返回 'complete'；
        全部失败则升级到 Level 2b。
        """
        print(
            f"\n==> 🧪 Level 2a: Parameter candidate set "
            f"(tool={step_ctx.tool_name}, N={self.CANDIDATE_SET_SIZE})\n"
        )
        tried_params = self._collect_tried_params(step_ctx)
        # LLM 调用（1 次）：生成 N 条候选命令 + 程序化扰动补齐。
        candidates = self._generate_parameter_candidate_set(
            question, image_path, step_ctx, diagnostic_signal, tried_params,
        )
        # 记录已试指纹，避免后续轮次重复。
        for c in candidates:
            tried_params.add(self._param_fingerprint(c, step_ctx.tool_name))

        if self.verbose:
            print(f"[Candidate Set]: {len(candidates)} command(s) generated")
            for i, c in enumerate(candidates):
                print(f"  cand{i + 1}: {c[:120]}")

        # 主线程串行执行候选，保证 executor 的 SIGALRM 超时机制可用。
        candidates_with_results = self._execute_candidate_set(
            step_ctx.tool_name, candidates, exec_step, json_data,
        )

        # 验证候选集（受 max_verify 封顶的 diagnoser LLM 调用，首个 COMPLETE 即胜出）。
        winner, final_exec_step = self._verify_candidate_set(
            question, image_path, step_ctx, candidates_with_results, tracker, exec_step,
        )
        tracker.record("retry_with_different_parameters")

        if winner is not None:
            cmd, res, verification, new_exec_step = winner
            new_ctx = StepContext(
                step_key=step_ctx.step_key,
                target_information=step_ctx.target_information,
                context=step_ctx.context,
                sub_goal=step_ctx.sub_goal,
                tool_name=step_ctx.tool_name,
                command=cmd,
                result_executor=res,
                first_attempt_command=step_ctx.first_attempt_command,
            )
            # Success recording (_record_success_trace + _record_obtained_information)
            # is delegated to the main loop's _handle_subgoal_complete on action="complete".
            print(f"\n==> ✅ Level 2a succeeded — recording successful intervention path\n")
            return "complete", new_ctx, verification, new_exec_step, None

        print(f"\n==> ⬆️ Level 2a exhausted — escalating to Level 2b (tool switch)\n")
        next_rec = tracker.resolve("retry_with_different_parameters")
        replan_signal = self._build_diagnostic_signal_for_replan(
            diagnostic_signal, next_rec, step_ctx, tracker, question=question,
        )
        return "escalate", step_ctx, VerificationResult(
            analysis=diagnostic_signal.get("analysis", ""),
            step_conclusion="SUBGOAL_INCOMPLETE",
            info_flag=False, obtained_info="",
            task_conclusion=None,
            diagnostic_signal=diagnostic_signal,
            subgoal_complete=False,
        ), final_exec_step, replan_signal

    def _run_intervention_level2b(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        exec_step: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Dict[str, Any],
        tracker: StepInterventionState,
    ) -> Tuple[str, StepContext, VerificationResult, int, Optional[Dict[str, Any]]]:
        """Level 2b：替代工具候选集（Executor 侧修复，子目标保持不变）。

        复用 `_suggest_alternative_tool` + diagnostic_signal.suggested_tool 选至多 2 个替代工具。
        每个工具：1 次 executor LLM 生成命令（n_candidates=1）→ 执行 → 预过滤 → 验证；
        首个 COMPLETE 即胜出。全部失败则升级到 Level 3 Counterfactual。
        """
        failed_tool = step_ctx.tool_name
        kind = infer_subgoal_kind(step_ctx.target_information, step_ctx.sub_goal)
        suggested = diagnostic_signal.get("suggested_tool")
        candidates_tools: List[str] = []
        if kind == "compute":
            if (
                "Python_Coder_Tool" in self.planner.available_tools
                and failed_tool != "Python_Coder_Tool"
            ):
                candidates_tools.append("Python_Coder_Tool")
        elif (
            suggested
            and suggested != failed_tool
            and suggested in self.planner.available_tools
        ):
            candidates_tools.append(suggested)
        if kind != "compute":
            alt = self._suggest_alternative_tool(failed_tool, step_ctx.target_information, question)
            if alt and alt not in candidates_tools and alt in self.planner.available_tools:
                candidates_tools.append(alt)
        # Synthesis must not escalate into retrieval tools.
        if kind == "synthesis":
            _search = {
                "Google_Search_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool",
            }
            candidates_tools = [t for t in candidates_tools if t not in _search]
        # 封顶 2 个替代工具以控制成本（每个工具 ≤1 次 exec + ≤2 次 diag）。
        candidates_tools = candidates_tools[:2]

        if not candidates_tools:
            print(f"\n==> ⬆️ Level 2b: no alternative tool available → escalate to Counterfactual\n")
            tracker.record("switch_tool", tool=failed_tool)
            next_rec = tracker.resolve("switch_tool")
            replan_signal = self._build_diagnostic_signal_for_replan(
                diagnostic_signal, next_rec, step_ctx, tracker, question=question,
            )
            return "escalate", step_ctx, VerificationResult(
                analysis=diagnostic_signal.get("analysis", ""),
                step_conclusion="SUBGOAL_INCOMPLETE",
                info_flag=False, obtained_info="",
                task_conclusion=None,
                diagnostic_signal=diagnostic_signal,
                subgoal_complete=False,
            ), exec_step, replan_signal

        print(f"\n==> 🔀 Level 2b: tool-switch candidate set {failed_tool} → {candidates_tools}\n")
        current_exec_step = exec_step
        verified = 0
        for new_tool in candidates_tools:
            if verified >= self.MAX_VERIFY_PER_ROUND:
                break
            # LLM 调用：为替代工具生成 1 条命令（n_candidates=1）。
            try:
                raw = self.executor.generate_tool_command(
                    question, image_path, step_ctx.context, step_ctx.sub_goal,
                    new_tool, self.system_memory.toolbox_metadata[new_tool],
                    current_exec_step + 1, json_data, diagnostic_signal, n_candidates=1,
                )
                analysis, explanation, command = self.executor.extract_explanation_and_command(raw)
            except Exception as e:
                print(f"\n==> ⚠️ Level 2b command-gen failed for {new_tool}: {e}\n")
                continue
            if new_tool == "Web_Search_Tool":
                command = self._ensure_web_search_command(command, step_ctx.sub_goal, step_ctx.context)
            command = self._sanitize_command_for_tool(new_tool, command)
            command = self._align_command_with_subgoal(
                new_tool, command, question, step_ctx.context, step_ctx.sub_goal,
            )
            validation_error = self._validate_command(new_tool, command)
            if validation_error:
                print(f"\n==> ⚠️ Level 2b validation failed for {new_tool}: {validation_error}\n")
                # Missing url on Web_Search: try Google next rather than inventing Wikipedia.
                if new_tool == "Web_Search_Tool" and "requires a url" in str(validation_error).lower():
                    google = "Google_Search_Tool"
                    if (
                        google in self.planner.available_tools
                        and google != failed_tool
                        and google not in candidates_tools
                    ):
                        candidates_tools.append(google)
                continue
            tracker.intervention_attempts += 1
            result_key = f"{exec_step}_l2b_{tracker.intervention_attempts}"
            result = self._execute_generated_command(new_tool, command, result_key, json_data)
            if not self._candidate_passes_prefilter(new_tool, result):
                verified += 1
                continue
            verified += 1
            cand_ctx = StepContext(
                step_key=step_ctx.step_key,
                target_information=step_ctx.target_information,
                context=step_ctx.context,
                sub_goal=step_ctx.sub_goal,
                tool_name=new_tool,
                command=command,
                result_executor=result,
                first_attempt_command=step_ctx.first_attempt_command,
            )
            # LLM 调用：diagnoser 验证替代工具结果，首个 COMPLETE 即胜出。
            verification = self._run_verification(
                question, image_path, cand_ctx, current_exec_step,
                label=(
                    f"Level 2b Verification ({new_tool}; "
                    f"intervention_attempt={tracker.intervention_attempts})"
                ),
                intervention_context=tracker.to_context(),
            )
            if verification.step_conclusion == "SUBGOAL_COMPLETE":
                tracker.record("switch_tool", tool=failed_tool)
                # 成功记录交给主循环的 _handle_subgoal_complete（action="complete" 时统一处理）。
                print(f"\n==> ✅ Level 2b succeeded with {new_tool}\n")
                return "complete", cand_ctx, verification, current_exec_step, None

        tracker.record("switch_tool", tool=failed_tool)
        print(f"\n==> ⬆️ Level 2b exhausted — escalating to Level 3 Counterfactual\n")
        next_rec = tracker.resolve("switch_tool")
        replan_signal = self._build_diagnostic_signal_for_replan(
            diagnostic_signal, next_rec, step_ctx, tracker, question=question,
        )
        return "escalate", step_ctx, VerificationResult(
            analysis=diagnostic_signal.get("analysis", ""),
            step_conclusion="SUBGOAL_INCOMPLETE",
            info_flag=False, obtained_info="",
            task_conclusion=None,
            diagnostic_signal=diagnostic_signal,
            subgoal_complete=False,
        ), current_exec_step, replan_signal

    def _handle_subgoal_incomplete(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
        tracker: StepInterventionState,
    ) -> Tuple[str, StepContext, VerificationResult, int, Optional[Dict[str, Any]]]:
        """层次化因果介入控制器（核心调度）。

        范式：Observation (L1 观察) → Intervention (L2 干预) → Counterfactual (L3 反事实)。
        层次化归因：默认信任 Planner；先质疑 Executor（L2a 参数、L2b 替代工具）；
        只有 Executor 侧介入全部耗尽，责任才升级到 Planner（L3：保留 vs 修改子目标，
        复用既有 replanning 模块）。

        返回动作：'complete' | 'replan'
        """
        print(f"\n==> 🕵️‍♂️ Execution #{exec_step}: Causal Diagnosis (Subgoal Incomplete)\n")
        if self.verbose:
            print(f"[Intervention Budget] {json.dumps(dict(tracker.counts), ensure_ascii=False)}")
            exhausted = tracker.exhausted_set()
            if exhausted:
                print(f"[Exhausted Interventions]: {sorted(exhausted)}")

        ablation = getattr(self, "ablation", None)
        self._emit_layer_telemetry(
            json_data,
            "intervention_trigger",
            exec_step=exec_step,
            outline_step=step_ctx.step_key,
            tool=step_ctx.tool_name,
            enable_intervention=bool(ablation.enable_intervention) if ablation else True,
        )

        # Paper track: full intervention OFF — record failure then naive replan
        # (main loop retries planner/executor on the same outline until budget).
        if ablation is not None and not ablation.enable_intervention:
            self._record_obtained_information(exec_step, verification, step_ctx)
            self._record_failure_trace(exec_step, step_ctx, verification.diagnostic_signal)
            self._emit_layer_telemetry(
                json_data,
                "intervention_disabled",
                exec_step=exec_step,
                outline_step=step_ctx.step_key,
                action="replan",
            )
            print(
                "\n==> ⏭️ Ablation no_intervention: skipping L1→L2a→L2b→L3; "
                "returning replan (naive retry)\n"
            )
            return "replan", step_ctx, verification, exec_step, verification.diagnostic_signal

        self._record_obtained_information(exec_step, verification, step_ctx)

        already_failed = step_ctx.tool_name in tracker.failed_tools
        diagnostic_signal = verification.diagnostic_signal

        # ── P0-⑤ Information Gain Gate (Exploration + Knowledge split) ──
        # Runs before any L2a/L2b dispatch so a provably-uninformative
        # intervention class is skipped BEFORE burning budget on it. The
        # gate compares the current attempt against the last recorded
        # observation for this outline step; 2 consecutive zero-knowledge-
        # gain attempts force-escalate to the next ladder rung.
        if diagnostic_signal and step_ctx.step_key:
            _exploration, _knowledge = self._compute_intervention_gains(
                step_ctx, verification, diagnostic_signal, tracker, step_ctx.step_key,
            )
            if _knowledge == 0:
                _n = tracker.bump_no_knowledge_gain(step_ctx.step_key)
                if _n >= 2:
                    _cur_rec = diagnostic_signal.get("recommendation", "retry_with_different_parameters")
                    _next_rec = tracker.force_escalate(_cur_rec)
                    print(
                        f"\n==> 🚪 Knowledge Gain Gate: {_n} consecutive zero-knowledge-gain "
                        f"attempts on step {step_ctx.step_key} — force-escalating "
                        f"'{_cur_rec}' → '{_next_rec}'"
                    )
                    # Re-tag the signal so the L2a/L2b dispatch below
                    # honours the escalation (skip_param_retry will fire
                    # because the recommendation no longer matches the
                    # current intervention class, OR the budget check
                    # will fail because force_escalate exhausted it).
                    diagnostic_signal = dict(diagnostic_signal)
                    diagnostic_signal["recommendation"] = _next_rec
                    verification = VerificationResult(
                        analysis=verification.analysis,
                        step_conclusion=verification.step_conclusion,
                        info_flag=verification.info_flag,
                        obtained_info=verification.obtained_info,
                        task_conclusion=verification.task_conclusion,
                        diagnostic_signal=diagnostic_signal,
                        subgoal_complete=verification.subgoal_complete,
                        slot_updates=verification.slot_updates,
                        evidence_type=verification.evidence_type,
                        tool_appropriate=verification.tool_appropriate,
                    )
            else:
                tracker.reset_no_knowledge_gain(step_ctx.step_key)
            # Record the current action signature so the next iteration
            # can compute Exploration Gain against it.
            tracker.record_exploration(step_ctx.step_key, step_ctx.command, step_ctx.tool_name)

        # diagnostic_signal 来自上一轮 diagnoser LLM（在 _run_verification 中产出）。
        # 把硬错误工具记入 failed_tools，并按 tracker 预算情况升级信号里的 recommendation。
        if diagnostic_signal:
            patterns = diagnostic_signal.get("failure_patterns") or {}
            if patterns.get("error_occurred") or self._is_pdf_access_error(step_ctx.result_executor):
                tracker.record_failure_tool(step_ctx.tool_name)

            diagnostic_signal = self._apply_escalation_to_signal(diagnostic_signal, tracker)
            diagnostic_signal = self._enrich_diagnostic_signal(
                diagnostic_signal, step_ctx.target_information, question, step_ctx=step_ctx,
            )
            self.system_memory.set_diagnostic_signal(diagnostic_signal)
            if self.verbose:
                print(f"[Diagnostic Signal]: {json.dumps(diagnostic_signal, indent=2, ensure_ascii=False)}\n")

            # Change 2 + 3: record (reason_key, target_variable) so the
            # Counterfactual stage can detect a no-op loop. Prefer the
            # structured enum ``reason`` over free-text description — soft
            # continue sets match reason tokens (low_recall, …), not prose.
            _hypo = diagnostic_signal.get("causal_hypothesis") or {}
            _rec = diagnostic_signal.get("recommendation", "")
            _tv = (
                _hypo.get("target_variable")
                or diagnostic_signal.get("target_variable")
                or INTERVENTION_TARGET_VARIABLE.get(_rec, "unknown")
            )
            _reason = _hypo.get("reason") or diagnostic_signal.get("reason") or ""
            _desc = (
                _hypo.get("description")
                or diagnostic_signal.get("root_cause")
                or (diagnostic_signal.get("failure_patterns") or {}).get("root_cause")
                or ""
            )
            _rc = self._normalize_cause_reason_key(_reason, _desc)
            _conf = _hypo.get("confidence") or diagnostic_signal.get("root_cause_confidence") or "MEDIUM"

            # P0-③ Hypothesis Drift detection — runs BEFORE record_root_cause
            # so a falsified prior hypothesis is invalidated before the new
            # one is appended to root_cause_history. This unifies Evidence
            # Drift, Slot Drift and Reason Drift into a single gate: any
            # of them falsifies the current Causal Hypothesis and forces a
            # fresh diagnosis (the new diagnostic_signal already reflects
            # the new observation; we just clear the stale cause history
            # so the Cause-Changed gate in _run_counterfactual does not
            # mistake the superseded hypothesis for a recurring cause).
            _step_key = step_ctx.step_key
            _cur_hypo = f"{_tv}.{_reason}"
            _cur_et = diagnostic_signal.get("evidence_type") or verification.evidence_type or ""
            _cur_slots = self._parse_slots_from_obtained(verification)
            _cur_cause = _cur_hypo
            if _step_key and tracker.hypothesis_changed(_step_key, _cur_hypo, _cur_et, _cur_slots):
                tracker.invalidate_hypothesis(_step_key)
                print(
                    f"\n==> 🔄 Hypothesis Drift on step {_step_key} — "
                    f"prior Causal Hypothesis falsified (evidence/slot/reason changed); "
                    f"re-diagnosing from clean slate"
                )
            if _step_key:
                tracker.snapshot_hypothesis(_step_key, _cur_hypo, _cur_et, _cur_slots, _cur_cause)

            if _rc:
                tracker.record_root_cause(_rc, _tv)
                if self.verbose:
                    print(
                        f"[Causal Hypothesis] failure_type="
                        f"{diagnostic_signal.get('failure_type')} "
                        f"target_variable={_tv} reason={_reason} "
                        f"confidence={_conf} → do({_tv})"
                    )

        self._record_failure_trace(exec_step, step_ctx, diagnostic_signal)
        self._append_trace_event(
            json_data,
            "intervention_failure",
            exec_step=exec_step,
            outline_step=step_ctx.step_key,
            sub_goal=step_ctx.sub_goal,
            generated_query=self._extract_query_param(step_ctx.command),
            tool=step_ctx.tool_name,
            root_cause=str(
                (diagnostic_signal or {}).get("root_cause")
                or ((diagnostic_signal or {}).get("causal_hypothesis") or {}).get("description")
                or ""
            ),
            slot_delta=verification.had_slot_delta,
            intervention_attempt=tracker.intervention_attempts,
        )

        # 无诊断信号 → 无法定向介入，直接 replan。
        if not diagnostic_signal:
            return "replan", step_ctx, verification, exec_step, None

        recommendation = diagnostic_signal.get("recommendation")

        # 诊断认为子目标其实已达成（或建议进入下一子目标）→ 直接判 complete。
        if recommendation == "continue_to_next_subgoal" or verification.subgoal_complete:
            print(f"\n==> ✅ Execution #{exec_step}: Subgoal actually complete — proceeding\n")
            if verification.step_conclusion != "SUBGOAL_COMPLETE":
                verification = VerificationResult(
                    analysis=verification.analysis,
                    step_conclusion="SUBGOAL_COMPLETE",
                    info_flag=verification.info_flag,
                    obtained_info=verification.obtained_info,
                    task_conclusion=verification.task_conclusion or "CONTINUE",
                    diagnostic_signal=diagnostic_signal,
                    subgoal_complete=True,
                )
            return "complete", step_ctx, verification, exec_step, None

        # ───────── Level 1：观察（0 次 LLM）─────────
        # 纯数据归集失败现场（_observe_failure 不调 LLM），供 L2/L3 决策复用。
        fc = self._observe_failure(step_ctx, verification, tracker)
        self._emit_layer_telemetry(
            json_data,
            "l1_observe",
            exec_step=exec_step,
            outline_step=step_ctx.step_key,
            tool=fc.tool,
            evidence_type=fc.evidence_type,
        )
        if self.verbose:
            print(f"[FailureContext] tool={fc.tool} evidence={fc.evidence_type} "
                  f"patterns={[k for k, v in fc.failure_patterns.items() if v]}")

        enable_l2a = ablation is None or ablation.enable_l2a
        enable_l2b = ablation is None or ablation.enable_l2b
        enable_l3 = ablation is None or ablation.enable_l3

        # ───────── Level 2a：参数候选集（Executor 侧）─────────
        # 默认信任 Planner，首个干预对象是 Executor。仅在「参数扰动无法修复的硬错误」时跳过 L2a。
        patterns = diagnostic_signal.get("failure_patterns") or {}
        # Network / infrastructure errors cannot be fixed by parameter
        # perturbation — detect them both via diagnoser classification and via
        # direct text matching on the executor result (the diagnoser has
        # historically under-classified urlopen / retrieval-incomplete errors).
        _res_str = str(step_ctx.result_executor)
        _res_lower = _res_str.lower()
        _network_error_text = (
            "urlopen error" in _res_lower
            or "retrieval incomplete" in _res_lower
            or "error in execute_tool_command" in _res_lower
            or "connection reset" in _res_lower
            or "connection timed out" in _res_lower
            or "ssl: certificate" in _res_lower
            or "no such host" in _res_lower
        )
        # Distinguish infrastructure errors (parameter changes cannot
        # help) from environment-constraint errors (a command rewrite CAN
        # help — e.g. scipy ImportError is fixed by regenerating the code
        # with math/numpy). The new ``failure_type`` from the structural
        # diagnosis makes this distinction precise.
        # Skip rules read only structured CausalHypothesis fields
        # {failure_type, target_variable, reason} — not free-text root_cause.
        _hypo = diagnostic_signal.get("causal_hypothesis") or {}
        _failure_type = (
            _hypo.get("failure_type")
            or diagnostic_signal.get("failure_type")
            or patterns.get("failure_type")
            or ""
        )
        _target_variable = (
            _hypo.get("target_variable")
            or diagnostic_signal.get("target_variable")
            or patterns.get("target_variable")
            or ""
        )
        _reason = (
            _hypo.get("reason")
            or diagnostic_signal.get("reason")
            or patterns.get("reason")
            or ""
        )
        # New failure_type enum (Causal Hypothesis): Tool / Execution /
        # Environment / External / Command / Retrieval / Unclear. The
        # thread_context reason (Execution under external target) is
        # parameter-unfixable — it must skip L2a even though failure_type
        # is "External", because the LLM previously mis-routed it to L2a
        # and burned 3 candidates (test_gaia_20260703_194121.log step 2-4).
        _is_infrastructure_error = (
            _failure_type in ("External", "Tool", "Command", "CapabilityMismatch", "ExternalError", "CapabilityMismatch", "CommandError")
            or (_failure_type == "Execution" and _reason == "thread_context")
            or patterns.get("url_client_error")
            or patterns.get("timeout")
            or _network_error_text
        )
        _is_command_schema_error = (
            "missing 1 required positional argument" in _res_str
            or "unexpected keyword argument" in _res_str
            or _failure_type in ("Command", "CommandError")
        )
        # Soft Retrieval/coverage ABSENCE must NOT skip L2a — Pearl ladder
        # retries parameters on the same tool before switching tools.
        skip_param_retry = (
            not enable_l2a
            or not verification.tool_appropriate
            or patterns.get("wrong_tool_class")
            or _is_infrastructure_error
            or _is_command_schema_error
            or self._is_pdf_access_error(_res_str)
            or (
                patterns.get("error_occurred")
                and already_failed
                and _failure_type not in ("Execution", "ExecutionFailure", "Retrieval")
            )
        )

        if not skip_param_retry and tracker.remaining("retry_with_different_parameters") > 0:
            self._emit_layer_telemetry(
                json_data, "l2a_enter",
                exec_step=exec_step, outline_step=step_ctx.step_key, tool=step_ctx.tool_name,
            )
            action, step_ctx, verification, exec_step, replan_signal = self._run_intervention_level2a(
                question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
            )
            self._emit_layer_telemetry(
                json_data,
                "l2a_complete" if action == "complete" else "l2a_fail",
                exec_step=exec_step,
                outline_step=step_ctx.step_key,
                tool=step_ctx.tool_name,
                rescued=action == "complete",
            )
            if action == "complete":
                return "complete", step_ctx, verification, exec_step, replan_signal
            diagnostic_signal = replan_signal or diagnostic_signal
        else:
            if skip_param_retry:
                if not enable_l2a:
                    reason = "product kill-switch enable_l2a=False"
                elif not verification.tool_appropriate or patterns.get("wrong_tool_class"):
                    reason = "wrong tool class"
                elif _failure_type in ("Tool", "CapabilityMismatch"):
                    reason = "capability mismatch"
                elif _failure_type in ("External", "ExternalError") or patterns.get("url_client_error") or _network_error_text:
                    reason = "network/external error"
                elif _reason == "thread_context":
                    reason = "thread-context error (not parameter-fixable)"
                elif _failure_type in ("Command", "CommandError") or _is_command_schema_error:
                    reason = "command schema error"
                elif patterns.get("timeout"):
                    reason = "timeout"
                elif self._is_pdf_access_error(_res_str):
                    reason = "pdf access error"
                else:
                    reason = "hard/infrastructure error"
                self._emit_layer_telemetry(
                    json_data, "l2a_skip",
                    exec_step=exec_step, outline_step=step_ctx.step_key,
                    tool=step_ctx.tool_name, reason=reason,
                )
                print(
                    f"\n==> ⏭️ Skipping Level 2a (parameter retry) — {step_ctx.tool_name} "
                    f"({reason}); escalating to Level 2b\n"
                )
            # Only force switch_tool when L2a was skipped or its budget is gone.
            diagnostic_signal = dict(diagnostic_signal)
            diagnostic_signal["recommendation"] = "switch_tool"

        # ───────── Level 2b：替代工具候选集（Executor 侧）─────────
        if enable_l2b and tracker.remaining("switch_tool") > 0:
            self._emit_layer_telemetry(
                json_data, "l2b_enter",
                exec_step=exec_step, outline_step=step_ctx.step_key, tool=step_ctx.tool_name,
            )
            action, step_ctx, verification, exec_step, replan_signal = self._run_intervention_level2b(
                question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
            )
            self._emit_layer_telemetry(
                json_data,
                "l2b_complete" if action == "complete" else "l2b_fail",
                exec_step=exec_step,
                outline_step=step_ctx.step_key,
                tool=step_ctx.tool_name,
                rescued=action == "complete",
            )
            if action == "complete":
                return "complete", step_ctx, verification, exec_step, replan_signal
            diagnostic_signal = replan_signal or diagnostic_signal
        elif not enable_l2b:
            self._emit_layer_telemetry(
                json_data, "l2b_skip",
                exec_step=exec_step, outline_step=step_ctx.step_key,
                reason="product kill-switch enable_l2b=False",
            )
            print("\n==> ⏭️ Skipping Level 2b — product kill-switch enable_l2b=False\n")

        # ───────── Level 3：反事实（Planner 侧 replanning）─────────
        if not enable_l3:
            self._emit_layer_telemetry(
                json_data, "l3_skip",
                exec_step=exec_step, outline_step=step_ctx.step_key,
                reason="product kill-switch enable_l3=False",
            )
            print("\n==> ⏭️ Skipping Level 3 — product kill-switch enable_l3=False; replan\n")
            return "replan", step_ctx, verification, exec_step, diagnostic_signal

        print(f"\n==> 🧠 Level 3: Counterfactual — escalating to Planner-side replanning\n")
        self._emit_layer_telemetry(
            json_data, "l3_enter",
            exec_step=exec_step, outline_step=step_ctx.step_key,
            recommendation=(diagnostic_signal or {}).get("recommendation"),
        )
        return self._run_counterfactual(
            question, step_ctx, verification, exec_step, json_data, tracker, diagnostic_signal,
        )

    def _run_counterfactual(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
        tracker: StepInterventionState,
        signal: Dict[str, Any],
    ) -> Tuple[str, StepContext, VerificationResult, int, Optional[Dict[str, Any]]]:
        """Level 3 反事实：决定「保留 vs 修改子目标」并调用既有 planner replanning 机制。

        成本中性关键：不引入新的 LLM 推理阶段。决策本身是规则驱动
        （`_suggest_intervention` 阶梯映射），replanning 复用 `revise_belief` /
        `decompose_goal` / `modify_state` 既有模块。

        本方法即原 `_route_post_intervention` 重命名为 Counterfactual 阶段；
        逻辑保持不变，仅重构了框架与 docstring。

        Change 3 (Cause-Changed gate): before firing a Planner-side
        intervention we check whether the *same root cause on the same
        target variable* has already been observed on this outline step.
        If it has, the counterfactual has not actually intervened on the
        cause (it merely re-sampled the plan) and we terminate the
        intervention loop to avoid the cyclic failure observed in
        test_gaia_20260703_185037.log (3× decompose_goal → same compute
        outline → same scipy ImportError).
        """
        rec = signal.get("recommendation", "switch_tool")
        _hypo = signal.get("causal_hypothesis") or {}
        _desc = (
            _hypo.get("description")
            or signal.get("root_cause")
            or (signal.get("failure_patterns") or {}).get("root_cause")
            or ""
        )
        _reason = _hypo.get("reason") or signal.get("reason") or ""
        # Cause-Changed / soft-continue keys MUST be the structured reason enum,
        # not free-text description (otherwise soft set never matches).
        root_cause = self._normalize_cause_reason_key(_reason, _desc)
        # Authoritative target_variable comes from the Causal Hypothesis;
        # the INTERVENTION_TARGET_VARIABLE fallback is only used when the
        # hypothesis is missing (e.g. legacy diagnostic signals).
        target_variable = (
            _hypo.get("target_variable")
            or signal.get("target_variable")
            or INTERVENTION_TARGET_VARIABLE.get(rec, "unknown")
        )
        confidence = (
            _hypo.get("confidence")
            or signal.get("root_cause_confidence")
            or "MEDIUM"
        )

        # P0-④: Confidence ONLY affects Diagnosis priority, never Terminate.
        # A LOW-confidence hypothesis is not trustworthy enough to drive a
        # Counterfactual — force a re-diagnosis by returning "replan" with
        # the current signal; the next loop iteration will run the Verifier
        # + Diagnoser again and produce a fresh (hopefully higher-confidence)
        # hypothesis. This is *not* a terminate: the intervention loop is
        # still alive, it just refuses to act on an untrusted hypothesis.
        # MEDIUM / HIGH proceed to the Cause-Changed gate below.
        if confidence == "LOW":
            print(
                f"\n==> 🔁 LOW-confidence Causal Hypothesis "
                f"(target={target_variable}, root_cause='{root_cause}') — "
                f"forcing re-diagnosis before any Planner-side intervention"
            )
            return "replan", step_ctx, verification, exec_step, signal

        # Cause-Changed gate: a real counterfactual must change the causal
        # variable that produced the failure. If the same (root_cause,
        # target) pair has already been recorded for this outline step,
        # the proposed intervention is a no-op on the cause → terminate.
        # NOTE: terminate here is driven by Cause Repeated (an observation
        # about whether the intervention changed the cause), NOT by
        # confidence. The Knowledge Gain gate (P0-⑤) and Action Variable
        # gate (P1-⑥) provide the other two independent terminate signals.
        # Soft coverage misses should keep trying query rewrites, not kill the
        # task after two identical no-match diagnoses.
        if root_cause and tracker.cause_repeated(root_cause, target_variable):
            if root_cause in self.SOFT_REPEATED_CAUSES:
                print(
                    f"\n==> 🔁 Cause-Changed soft-continue: root_cause='{root_cause}' "
                    f"reoccurred on do({target_variable}) — allowing further "
                    f"query/tool rewrite (not terminating)\n"
                )
            else:
                print(
                    f"\n==> 🛑 Cause-Changed gate: root_cause='{root_cause}' "
                    f"reoccurred on do({target_variable}) — counterfactual did "
                    f"NOT change the cause; terminating intervention loop\n"
                )
                return "terminate", step_ctx, verification, exec_step, signal

        # ── P1-⑥ Action Variable Gate (Warning → Penalty → Escalate) ──
        # Detects Planner-side interventions that did NOT change any
        # action variable (tool / query / url). The check is cross-replan
        # because decompose_goal mints a new step_key: ``self._last_planner_action``
        # is a solver-level slot that survives across replans within the
        # same task (reset alongside intervention_trackers at run entry).
        # 1st no-op → warn + allow (LLM self-correction chance).
        # 2nd no-op → force decompose_goal (NOT terminate; terminate only
        # fires when decompose_goal is also exhausted). This avoids the
        # false-positive termination where the LLM just needed one more
        # sampling to find a genuinely different plan.
        _PLANNER_SIDE = {"revise_belief", "decompose_goal", "modify_state"}
        if rec in _PLANNER_SIDE:
            _cur_action = self._action_variable_signature(step_ctx)
            _prev_action = self._last_planner_action
            if _prev_action is not None and _cur_action == _prev_action:
                self._no_op_penalty += 1
                _penalty = self._no_op_penalty
                if _penalty == 1:
                    print(
                        f"\n==> ⚠️ Action Variable unchanged (1st no-op) — "
                        f"tool={_cur_action[0]} query='{_cur_action[1]}' "
                        f"url='{_cur_action[2]}'; warning, allowing 1 more "
                        f"Planner-side attempt"
                    )
                    # Fall through to the normal intervention; record the
                    # action so the next iteration can compare.
                    self._last_planner_action = _cur_action
                else:
                    print(
                        f"\n==> 🔁 Action Variable unchanged ({_penalty}x no-op) — "
                        f"Planner-side intervention is LLM rephrasing only; "
                        f"forcing decompose_goal escalation"
                    )
                    if not tracker.is_exhausted("decompose_goal"):
                        rec = "decompose_goal"
                        signal = self._build_diagnostic_signal_for_replan(
                            signal, rec, step_ctx, tracker, question=question,
                        )
                        self._last_planner_action = _cur_action
                    else:
                        print(
                            f"\n==> 🛑 decompose_goal also exhausted — "
                            f"terminating intervention loop (Action Variable "
                            f"gate: no Planner-side intervention can change "
                            f"the action)"
                        )
                        return "terminate", step_ctx, verification, exec_step, signal
            else:
                # Action actually changed (or first Planner-side intervention
                # on this task) → clear the no-op penalty and record the new
                # baseline.
                self._no_op_penalty = 0
                self._last_planner_action = _cur_action

        if rec == "switch_tool":
            tracker.record("switch_tool", tool=step_ctx.tool_name)
            if not tracker.is_exhausted("switch_tool"):
                return "replan", step_ctx, verification, exec_step, signal
            # switch_tool 预算耗尽 → 沿阶梯 resolve() 升级到下一个 L3 推荐。
            rec = tracker.resolve("switch_tool")
            signal = self._build_diagnostic_signal_for_replan(signal, rec, step_ctx, tracker, question=question)

        # revise_belief：撤回冲突 claim + 注入外部验证步骤（replanning 内部走 planner LLM）。
        if rec == "revise_belief" and not tracker.is_exhausted("revise_belief"):
            tracker.record("revise_belief")
            signal = self._apply_revise_belief(question, step_ctx, verification, exec_step, tracker)
            return "replan", step_ctx, verification, exec_step, signal

        # decompose_goal：分解子目标（replanning 内部走 planner LLM）。
        if rec == "decompose_goal" and not tracker.is_exhausted("decompose_goal"):
            tracker.record("decompose_goal")
            changed = self._apply_decompose_goal(
                question, step_ctx, verification, exec_step, json_data, tracker,
            )
            if not changed:
                return "terminate", step_ctx, verification, exec_step, signal
            return "replan", step_ctx, verification, exec_step, signal

        # modify_state：环境/前置条件修复。
        if rec == "modify_state" and not tracker.is_exhausted("modify_state"):
            tracker.record("modify_state")
            return "replan", step_ctx, verification, exec_step, self._apply_modify_state(step_ctx, verification)

        # 兜底：所有 L3 推荐都耗尽 → 强制再分解一次子目标，给 Planner 最后一次机会。
        tracker.record("decompose_goal")
        changed = self._apply_decompose_goal(
            question, step_ctx, verification, exec_step, json_data, tracker,
        )
        if not changed:
            return "terminate", step_ctx, verification, exec_step, signal
        return "replan", step_ctx, verification, exec_step, signal

    def _apply_revise_belief(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        tracker: StepInterventionState,
    ) -> Dict[str, Any]:
        """Retract conflicting inferred claims and inject external verification step."""
        conflicts = self.system_memory.detect_conflicts()
        signal = dict(verification.diagnostic_signal or {})

        if not conflicts:
            # Scheme A: question-interpretation conflict — a quantity question
            # answered without any external retrieval. Re-analyze the query and
            # reset the outline so the base quantity gets retrieved.
            if self._detect_question_intent_conflict(question):
                print(
                    "\n==> 🔄 revise_belief: question-interpretation conflict — "
                    "quantity question answered without external retrieval; "
                    "re-analyzing query\n"
                )
                try:
                    new_analysis, new_outline = self.planner.analyze_query(
                        question, None,
                    )
                except Exception:
                    new_outline = None
                if new_outline and isinstance(new_outline, dict) and new_outline:
                    self.system_memory.set_outline(new_outline)
                    signal["recommendation"] = "revise_belief"
                    signal["suggested_tool"] = "Google_Search_Tool"
                    signal["reason"] = "Question-interpretation conflict; re-analyzed outline"
                    tracker.record_failure_tool(step_ctx.tool_name)
                    return signal
            print("\n==> ⏭️ revise_belief skipped — no claim conflicts detected\n")
            signal["recommendation"] = "switch_tool"
            signal["suggested_tool"] = "Google_Search_Tool"
            signal["reason"] = "No conflicts to revise; switch retrieval approach"
            return signal

        entity_key = ""
        for conflict in conflicts:
            entity_key = conflict.get("entity_key", "")
            for rid in conflict.get("record_ids", []):
                rec = next(
                    (r for r in self.system_memory.evidence_records if r.get("id") == rid),
                    None,
                )
                if rec and (
                    rec.get("source_quality") == "inferred"
                    or rec.get("tool") == "Base_Generator_Tool"
                ):
                    self.system_memory.retract_evidence(rid, conflict.get("reason", "conflict"))

        entity_disp = entity_key.replace("arxiv:", "") if entity_key else "target entity"
        is_arxiv_entity = entity_key.startswith("arxiv:")
        search_clause = (
            f"\"arxiv {entity_disp}\" OR Web_Search_Tool on the article abs page URL"
            if is_arxiv_entity else
            f"\"{entity_disp}\""
        )
        outline = {
            "1": (
                f"Target Information: Verify {entity_disp} with primary external sources "
                f"and resolve conflicting claims. "
                f"Operation Details: Google_Search_Tool query {search_clause}. "
                "Expected Output: Authoritative facts resolving the conflict."
            ),
        }
        self.system_memory.set_outline(outline)
        print(f"\n==> 🔄 revise_belief: retracted conflicts, injected verification for {entity_disp}\n")

        signal["recommendation"] = "revise_belief"
        signal["suggested_tool"] = "Google_Search_Tool"
        signal["reason"] = f"Conflicting claims on {entity_disp}; verify with retrieval"
        tracker.record_failure_tool(step_ctx.tool_name)
        return signal

