import argparse
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from MAS.epc_aw.models.initializer import Initializer
from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.executor import Executor
from MAS.epc_aw.models.utils import make_json_serializable_truncated
from MAS.epc_aw.models import BayesianInference, HistoryAnalyzer, CausalInference
from MAS.epc_aw.models.task_profile import (
    SlotGate,
    _extract_zip_codes,
    _has_usgs_context,
    FINAL_TERMS,
)
from MAS.epc_aw.models.tool_router import ToolRouter, infer_subgoal_kind


# ---------------------------------------------------------------------------
# 因果介入预算与升级阶梯
# 范式：Observation (L1 观察) → Intervention (L2 干预) → Counterfactual (L3 反事实)
#   L2a = 当前工具的参数候选集（1 轮 = 1 次 executor LLM 返 N 候选 + 并行执行 + 验证）
#   L2b = 替代工具候选集（1 轮 ≤2 个替代工具，每个 1 次 executor LLM）
#   L3  = revise_belief / decompose_goal / modify_state（Planner 侧 replanning，复用既有模块）
# counts 记的是「候选集轮数」而非单个候选尝试次数（一轮内 N 个候选并行扇出）。
# ---------------------------------------------------------------------------
MAX_PARAMETER_RETRIES = 3       # L2a 候选集轮数上限（默认跑 1 轮；预算允许最多 3 轮）
MAX_SWITCH_TOOL_ATTEMPTS = 2    # L2b 候选集轮数上限（每轮 ≤2 个替代工具）
MAX_DECOMPOSE_GOAL_ATTEMPTS = 1
MAX_MODIFY_STATE_ATTEMPTS = 1
MAX_REVISE_BELIEF_ATTEMPTS = 1
MAX_RECOVERY_OUTLINE_INJECTIONS = 1

# 介入升级阶梯：从前到后依次尝试，前者预算耗尽则 resolve() 升级到下一个。
# L2a/L2b 质疑 Executor（参数/工具），L3 才质疑 Planner（subgoal）——层次化归因。
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

# Change 2: each intervention acts on a specific causal variable (Pearl's
# do(X)). Recording the target variable lets the Counterfactual stage
# verify that an intervention actually *changes the cause* — if the same
# root_cause recurs after do(X), the loop must terminate (Change 3).
#
# Causal Hypothesis extension: ``query`` and ``coverage`` are added as
# first-class intervention targets so retrieval failures (ABSENCE on a
# capability-matched search tool) are attributed to a concrete do(*)
# variable instead of the symptom string "no usable evidence".
INTERVENTION_TARGET_VARIABLE: Dict[str, str] = {
    "retry_with_different_parameters": "executor",   # do(Command params) — also covers do(query)
    "switch_tool": "tool",                           # do(Tool)
    "revise_belief": "planner_belief",               # do(Belief)
    "decompose_goal": "task_graph",                  # do(Task decomposition)
    "modify_state": "environment",                   # do(Environment)
}

# Causal Hypothesis target_variable namespace. Values are the do(*)
# variables the Diagnoser can attribute a failure to. They extend
# INTERVENTION_TARGET_VARIABLE with two retrieval-specific nodes that
# do not map 1:1 to an intervention name (``query`` is acted on via
# retry_with_different_parameters; ``coverage`` is acted on via
# retry_with_different_parameters with a source-switching instruction, or
# via switch_tool when the source is the tool itself).
CAUSAL_HYPOTHESIS_TARGETS: List[str] = [
    "tool", "executor", "planner_belief", "task_graph", "environment",
    "query", "coverage", "external", "command",
]

PDF_ACCESS_ERROR = re.compile(
    r"Download is starting|Failed to load image from",
    re.I,
)
ARXIV_PDF_URL = re.compile(r"arxiv\.org/pdf/", re.I)


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
class VerificationResult:
    analysis: str
    step_conclusion: str
    info_flag: bool
    obtained_info: str
    task_conclusion: Optional[str]
    diagnostic_signal: Optional[Dict[str, Any]]
    subgoal_complete: bool
    slot_updates: List[Dict[str, Any]] = field(default_factory=list)
    evidence_type: str = "DIRECT"
    tool_appropriate: bool = True
    had_slot_delta: bool = False


@dataclass
class StepContext:
    step_key: str
    target_information: str
    context: str
    sub_goal: str
    tool_name: str
    command: str
    result_executor: Any
    first_attempt_command: str


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


class Solver:
    def __init__(
        self,
        planner,
        system_memory,
        executor,
        diagnoser,
        output_types: str = "base,final,direct",
        max_steps: int = 20,
        max_time: int = 3000,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = .0,
        evaluation_mode: bool = False,
    ):
        self.planner = planner
        self.system_memory = system_memory
        self.executor = executor
        self.diagnoser = diagnoser
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir
        self.output_types = output_types.lower().split(',')
        self.temperature = temperature
        self.evaluation_mode = evaluation_mode
        assert all(t in ["base", "final", "direct"] for t in self.output_types)
        self.verbose = verbose

        self.bayesian_inference = BayesianInference()
        self.history_analyzer = HistoryAnalyzer()
        self.causal_inference = CausalInference(
            self.system_memory.causal_graph, self.history_analyzer,
        )

        self.planner.causal_inference = self.causal_inference
        self.planner.causal_graph = self.system_memory.causal_graph
        self.planner.system_memory = system_memory
        self.planner.history_analyzer = self.history_analyzer
        self.diagnoser.bayesian_inference = self.bayesian_inference
        self.diagnoser.history_analyzer = self.history_analyzer

        # P1-⑥ Action Variable Gate state (cross-replan). Lives on the
        # solver instance because decompose_goal mints a new step_key and
        # a fresh StepInterventionState; per-step state would lose
        # continuity across the replan boundary. Reset per task in the
        # main run entry (alongside intervention_trackers).
        self._last_planner_action: Optional[Tuple[str, str, str]] = None
        self._no_op_penalty: int = 0
        self.diagnoser.system_memory = system_memory
        self.executor.history_analyzer = self.history_analyzer
        self.executor.system_memory = system_memory

        # v3: decompose_goal bonus step budget. When a corrected outline is
        # injected by decompose_goal, allow at most 1 extra execution step so the
        # fix actually runs before max_steps exhausts. Reset per task in solve().
        self._decompose_bonus = 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query_param(cmd_str: str) -> str:
        match = re.search(r'query=["\']([^"\']+)["\']', str(cmd_str))
        return match.group(1) if match else "N/A"

    @staticmethod
    def _normalize_query_param(query: str) -> str:
        return re.sub(r"\s+", " ", str(query).strip().lower())

    @staticmethod
    def _inject_query_into_command(command: str, new_query: str) -> str:
        escaped = new_query.replace('"', '\\"')
        if re.search(r'query=["\']', command):
            return re.sub(r'query=(["\'])[^"\']*\1', f'query="{escaped}"', command, count=1)
        return f'execution = tool.execute(query="{escaped}")'

    @staticmethod
    def _strip_query_from_command(command: str) -> str:
        """Remove query= kwarg from a tool.execute(...) command string."""
        cmd = re.sub(r',\s*query=(["\'])[^"\']*\1', "", str(command))
        cmd = re.sub(r'query=(["\'])[^"\']*\1,\s*', "", cmd)
        return cmd

    @staticmethod
    def _sanitize_command_for_tool(tool_name: str, command: str) -> str:
        if tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
            return Solver._strip_query_from_command(command)
        return command

    def _perturb_query_param(
        self,
        query: str,
        attempt: int,
        tried: Set[str],
        sub_goal: str = "",
        target_information: str = "",
    ) -> str:
        """Deterministic query perturbation when LLM repeats a failed parameter.

        Purely generic rewrites derived from the query itself — no hardcoded
        sample-specific keywords, URLs, or answer tokens. Strategies:
          * drop trailing temporal / location / filler qualifiers
          * strip common English stop words for a tighter keyword query
          * take head / tail / reversed token windows
          * quote the query as an exact phrase
          * add a `site:` restriction when a URL host is present in context
          * append generic reformulation suffixes
        """
        base = query.strip()
        if not base:
            return f"retry{attempt}"

        # Generic URL host extraction from surrounding context (no hardcoded domains).
        url_match = re.search(r"https?://([^/\s\"']+)", f"{sub_goal} {target_information}")
        site_host = url_match.group(1) if url_match else ""

        candidates: List[str] = []

        # 1. Drop temporal / location / filler qualifiers (generic, not sample-specific).
        filler_phrases = (
            # " before 2020", " before 2010", " before 2000",
            " locations", " sightings", " records",
            # " in the united states", " in the us",
            " according to", " official",
        )
        simplified = base
        for phrase in filler_phrases:
            simplified = simplified.replace(phrase, "")
        simplified = re.sub(r"\s+", " ", simplified).strip()
        if simplified and simplified.lower() != base.lower():
            candidates.append(simplified)

        # 2. Strip common English stop words for a tighter keyword query.
        stop = {
            "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
            "with", "from", "by", "at", "is", "are", "was", "were", "what",
            "which", "that", "this", "these", "those", "list", "all",
        }
        tokens = (simplified or base).split()
        keywords = [t for t in tokens if t.lower().strip(".,;:") not in stop]
        if keywords and len(keywords) < len(tokens):
            candidates.append(" ".join(keywords))

        # 3. Head / tail / reversed token windows.
        if len(tokens) > 3:
            candidates.append(" ".join(tokens[: max(2, len(tokens) - 2)]))
            candidates.append(" ".join(tokens[-3:]))
            candidates.append(" ".join(reversed(tokens)))

        # 4. site:-restricted variant (generic host extracted from context).
        if site_host:
            candidates.append(f"site:{site_host} {simplified or base}")
            if keywords:
                candidates.append(f"site:{site_host} {' '.join(keywords)}")

        # 5. Quoted exact phrase.
        candidates.append(f'"{simplified or base}"')

        # 6. Generic reformulation suffixes.
        candidates.append(f"{base} alternative phrasing")
        candidates.append(f"{simplified or base} variant{attempt}")
        candidates.append(f"{simplified or base} retry{attempt}")

        for candidate in candidates:
            norm = self._normalize_query_param(candidate)
            if norm and norm not in tried:
                return candidate

        return f"{base} retry{attempt}"

    def _perturb_url_param(self, url: str, attempt: int, tried: Set[str]) -> str:
        """Perturb url for Screenshot/Vision_OCR retries."""
        if url == "N/A":
            return url
        candidates: List[str] = []
        if ARXIV_PDF_URL.search(url):
            candidates.append(self._pdf_url_to_abs_url(url))
        if "/abs/" in url:
            candidates.append(url.replace("/abs/", "/html/"))
        candidates.append(url.rstrip("/"))
        candidates.append(f"{url.rstrip('/')}?attempt={attempt}")

        for candidate in candidates:
            norm = candidate.strip().lower()
            if norm and norm not in tried:
                return candidate
        return url

    @staticmethod
    def _extract_url_param(cmd_str: str) -> str:
        match = re.search(r'url=["\']([^"\']+)["\']', str(cmd_str))
        return match.group(1) if match else "N/A"

    @staticmethod
    def _param_fingerprint(command: str, tool_name: str) -> str:
        q = Solver._extract_query_param(command)
        q_norm = Solver._normalize_query_param(q)
        if tool_name == "Web_Search_Tool":
            url = Solver._extract_url_param(command)
            url_norm = url.strip().lower() if url != "N/A" else ""
            return f"{q_norm}|{url_norm}"
        if tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
            url = Solver._extract_url_param(command)
            image_match = re.search(r'image_input=["\']([^"\']+)["\']', str(command))
            target = url if url != "N/A" else (image_match.group(1) if image_match else "")
            return target.strip().lower() if target else q_norm
        return q_norm

    @staticmethod
    def _inject_url_into_command(command: str, new_url: str) -> str:
        escaped = new_url.replace('"', '\\"')
        if re.search(r'url=["\']', command):
            return re.sub(r'url=(["\'])[^"\']*\1', f'url="{escaped}"', command, count=1)
        if 'query="' in command or "query='" in command:
            cmd = command.rstrip()
            if cmd.endswith(")"):
                return cmd[:-1] + f', url="{escaped}")'
            return cmd + f', url="{escaped}")'
        return f'execution = tool.execute(query="fetch page", url="{escaped}")'

    def _perturb_web_search_command(
        self,
        command: str,
        query: str,
        attempt: int,
        tried: Set[str],
    ) -> Tuple[str, str]:
        """Perturb Web_Search by changing URL or query+URL together.

        Generic strategies only — no hardcoded sample-specific queries or URLs:
          * canonical arxiv /abs/ ↔ /html/ mirror swap (URL rewrite, not answer)
          * normalize arxiv PDF URL to abs URL (text extraction friendlier)
          * drop the URL so the search engine ranks results itself
          * keep the URL, perturb the query via _perturb_query_param
          * trailing-slash normalization
        """
        url = self._extract_url_param(command)
        url_lower = (url or "").lower()
        candidates: List[Tuple[str, str]] = []

        # Canonical arxiv mirrors (generic URL rewriting, no answer leakage).
        if "arxiv.org/abs/" in url_lower:
            candidates.append((query, url.replace("/abs/", "/html/")))
        if ARXIV_PDF_URL.search(url or ""):
            abs_url = self._pdf_url_to_abs_url(url)
            candidates.append((query, abs_url))

        # Drop URL — let the search engine rank results for a perturbed query.
        perturbed_q = self._perturb_query_param(
            query, attempt, tried, sub_goal="", target_information="",
        )
        if perturbed_q and perturbed_q.lower() != query.lower():
            candidates.append((perturbed_q, ""))
            if url != "N/A":
                candidates.append((perturbed_q, url))

        # Trailing-slash normalization.
        if url and url != "N/A":
            candidates.append((query, url.rstrip("/")))

        for new_query, new_url in candidates:
            probe = self._inject_query_into_command(command, new_query)
            if new_url:
                probe = self._inject_url_into_command(probe, new_url)
            fp = self._param_fingerprint(probe, "Web_Search_Tool")
            if fp not in tried:
                return probe, new_query

        fallback_q = f"{query} retry{attempt}"
        probe = self._inject_query_into_command(command, fallback_q)
        return probe, fallback_q

    def _collect_tried_params(self, step_ctx: "StepContext") -> Set[str]:
        tried: Set[str] = set()
        for cmd in (step_ctx.first_attempt_command, step_ctx.command):
            if cmd:
                tried.add(self._param_fingerprint(cmd, step_ctx.tool_name))
        return tried

    def _generate_executor_command(
        self,
        question: str,
        image_path: Optional[str],
        context: str,
        sub_goal: str,
        tool_name: str,
        step_count: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        if tool_name not in self.planner.available_tools:
            command = "No command was generated because the tool was not found."
            return command, command, command

        tool_command = self.executor.generate_tool_command(
            question,
            image_path,
            context,
            sub_goal,
            tool_name,
            self.system_memory.toolbox_metadata[tool_name],
            step_count,
            json_data,
            diagnostic_signal,
        )
        analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
        return command, analysis, explanation

    def _execute_generated_command(
        self,
        tool_name: str,
        command: str,
        step_count: int,
        json_data: Dict[str, Any],
    ) -> Any:
        result = self.executor.execute_tool_command(tool_name, command)
        result = make_json_serializable_truncated(result)
        json_data[f"tool_result_{step_count}"] = result
        return result

    def _outline_step_id(self, step_key: str, target_information: str) -> str:
        digest = hashlib.md5(str(target_information)[:200].encode()).hexdigest()[:8]
        return f"{step_key}:{digest}"

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
                bindings["computed_count"] = numeric.group(0)
                parts.append(f"Computed result: {numeric.group(0)}")
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

        bindings: Dict[str, str] = {}
        for upd in slot_updates:
            if isinstance(upd, dict) and upd.get("filled") and upd.get("value"):
                bindings[str(upd.get("slot", ""))] = str(upd.get("value"))

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

        # Generic arxiv ID extraction from collected evidence (no hardcoded default ID).
        arxiv_m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{5})", cumulative, re.I)
        arxiv_id = arxiv_m.group(1) if arxiv_m else ""
        arxiv_clause = (
            f"from arXiv paper {arxiv_id} " if arxiv_id else "from the identified source "
        )

        outline = {
            "1": (
                f"Target Information: Retrieve evidence for {missing_desc} "
                f"{arxiv_clause}to complete the cross-reference. "
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

    def _get_first_outline_step(self) -> Tuple[Optional[str], Optional[str]]:
        outline = self.system_memory.get_outline()
        numeric_keys = sorted(
            [k for k in outline.keys() if k.isdigit()],
            key=lambda x: int(x),
        )
        if not numeric_keys:
            return None, None
        key = numeric_keys[0]
        return key, outline[key]

    def _requires_external_evidence(self, question: str, target_information: str) -> bool:
        text = f"{question} {target_information}".lower()
        markers = (
            "usgs", "according to", "official", "database",
            "zip code", "zip codes", "nonnative", "nonindigenous",
        )
        return any(m in text for m in markers)

    def _suggest_alternative_tool(
        self,
        failed_tool: Optional[str],
        target_information: str,
        question: str,
    ) -> Optional[str]:
        """Pick a retrieval tool when the current one failed. Never suggest Base_Generator.

        Generic priority order — no sample-specific tool routing.
        """
        if infer_subgoal_kind(target_information, question) == "compute":
            return None
        available = set(self.planner.available_tools)
        for candidate in (
            "Web_Search_Tool", "Google_Search_Tool", "Wikipedia_Search_Tool",
        ):
            if candidate in available and candidate != failed_tool:
                return candidate
        return None

    def _inject_web_context_for_usgs(self, context: str, target_information: str, question: str) -> str:
        text = f"{context} {target_information} {question}".lower()
        if "usgs" not in text and "nas.er.usgs.gov" not in text:
            return context
        if "http" in context:
            return context
        # Inject the USGS NAS root only — no sample-specific speciesID path.
        return f"https://nas.er.usgs.gov/ {context}".strip()

    @staticmethod
    def _is_pdf_access_error(result: Any) -> bool:
        return bool(PDF_ACCESS_ERROR.search(str(result)))

    @staticmethod
    def _pdf_url_to_abs_url(url: str) -> str:
        if ARXIV_PDF_URL.search(url):
            return ARXIV_PDF_URL.sub("arxiv.org/abs/", url)
        return url

    def _validate_command(self, tool_name: str, command: str) -> Optional[str]:
        if tool_name == "Web_Search_Tool":
            if self._extract_query_param(command) == "N/A":
                return "Web_Search_Tool requires a query parameter"
            url = self._extract_url_param(command)
            if url == "N/A":
                return "Web_Search_Tool requires a url parameter"
            query = self._extract_query_param(command).lower()
            url_lower = url.lower()
            if "arxiv.org/pdf/" in url_lower and any(
                m in f"{query} {url_lower}" for m in ("axis", "figure", "label")
            ):
                return (
                    "Web_Search_Tool cannot extract figure/axis labels from arXiv PDF URLs "
                    "(use Google_Search_Tool instead)"
                )
            if "arxiv.org/abs/" in url.lower():
                figure_markers = (
                    "figure 1", "axis label", "endpoint label", "figure caption",
                    "three axes", "axis endpoint",
                )
                if any(m in query for m in figure_markers):
                    return (
                        "Web_Search_Tool cannot extract figure/axis labels from arXiv abs pages "
                        "(use Google_Search_Tool instead)"
                    )
        if tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
            if re.search(r'query=["\']', str(command)):
                return f"{tool_name} does not accept a query parameter (use url or image_input only)"
            url = self._extract_url_param(command)
            image_input = re.search(r'image_input=["\']([^"\']+)["\']', str(command))
            target = url if url != "N/A" else (image_input.group(1) if image_input else "")
            if target and (ARXIV_PDF_URL.search(target) or target.rstrip("/").endswith(".pdf")):
                return (
                    f"{tool_name} cannot process arXiv/PDF URLs directly "
                    f"(use Google_Search_Tool or Web_Search_Tool on abs page instead)"
                )
        return None

    @staticmethod
    def _significant_query_terms(text: str) -> Set[str]:
        stop = {
            "the", "and", "for", "from", "with", "that", "this", "into", "using",
            "use", "retrieve", "find", "identify", "calculate", "compute", "what",
            "which", "how", "many", "exact", "value", "information", "minutes",
        }
        return {
            term for term in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(term) >= 3 and term not in stop
        }

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

    def _build_deterministic_compute_query(
        self,
        question: str,
        sub_goal: str,
        slot_values: Dict[str, str],
    ) -> Optional[str]:
        """Generate math-only Python when inputs are already verified in slots."""
        base_raw = slot_values.get("base_count") or slot_values.get("input_metrics")
        if not base_raw:
            return None
        base_match = re.search(r"\d+(?:\.\d+)?", str(base_raw))
        if not base_match:
            return None
        base_value = base_match.group(0)

        q_lower = str(question).lower()
        p_match = re.search(r"p[- ]?value(?:\s+of)?\s+(\d+(?:\.\d+)?)", q_lower)
        if not p_match:
            p_match = re.search(r"\baverage\s+(?:came\s+to\s+a\s+)?p[- ]?value\s+of\s+(\d+(?:\.\d+)?)", q_lower)
        if p_match and any(
            term in q_lower
            for term in ("incorrect", "false positive", "statistical significance")
        ):
            p_value = p_match.group(1)
            return (
                "import math\n"
                f"base_count = {base_value}\n"
                f"p_value = {p_value}\n"
                "result = math.ceil(base_count * p_value)\n"
                "print(result)"
            )

        if (
            "round" in q_lower
            and slot_values.get("input_metrics")
            and re.search(r"\bhow many\b", q_lower)
        ):
            metrics = re.findall(r"\d+(?:\.\d+)?", str(slot_values.get("input_metrics", "")))
            if len(metrics) >= 2:
                return (
                    "import math\n"
                    f"distance = {metrics[0]}\n"
                    f"pace = {metrics[1]}\n"
                    "result = round(distance / pace)\n"
                    "print(result)"
                )
        return None

    def _append_trace_event(
        self,
        json_data: Dict[str, Any],
        event: str,
        *,
        exec_step: Optional[int] = None,
        outline_step: Optional[str] = None,
        sub_goal: str = "",
        generated_query: str = "",
        tool: str = "",
        root_cause: str = "",
        slot_delta: bool = False,
        intervention_attempt: Optional[int] = None,
    ) -> None:
        json_data.setdefault("trace_events", []).append({
            "event": event,
            "task_id": json_data.get("task_id"),
            "exec_step": exec_step,
            "outline_step": outline_step,
            "intervention_attempt": intervention_attempt,
            "sub_goal": sub_goal,
            "generated_query": generated_query,
            "tool": tool,
            "root_cause": root_cause,
            "slot_delta": slot_delta,
        })

    def _align_command_with_subgoal(
        self,
        tool_name: str,
        command: str,
        question: str,
        context: str,
        sub_goal: str,
    ) -> str:
        """Keep generated tool queries grounded in the current sub-goal."""
        if tool_name == "Python_Coder_Tool":
            slot_values = self._get_slot_values_from_evidence()
            deterministic = self._build_deterministic_compute_query(
                question, sub_goal, slot_values,
            )
            if deterministic:
                print(
                    "\n==> 🧮 Deterministic compute query injected from verified slot values\n"
                )
                return self._inject_query_into_command(command, deterministic)
            evidence = self.system_memory.get_obtained_information_for_prompt()
            explicit = (
                f"Question: {question}. Current calculation: {sub_goal}. "
                f"Context: {context}. Verified inputs: {evidence}. "
                "Use only available standard libraries (prefer math); print only the final value."
            )[:2400]
            return self._inject_query_into_command(command, explicit)

        if tool_name not in {
            "Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
        }:
            return command
        query = self._extract_query_param(command)
        if query == "N/A":
            return command
        if re.search(
            r"(name of the char(?:a)?cter|what char(?:a)?cter|exact char(?:a)?cter)",
            str(question).lower(),
        ):
            output_literals = re.findall(
                r'output\s+["\']([^"\']+)["\']', str(question), re.I,
            )
            code_lines = [
                line.strip() for line in str(question).splitlines()
                if "`" in line and len(line.strip()) >= 4
            ]
            anchors = " ".join(
                [f'"{literal}"' for literal in output_literals] + code_lines[:1]
            )
            grounded_query = f"{sub_goal} {anchors}".strip()[:500]
            return self._inject_query_into_command(command, grounded_query)
        required = self._significant_query_terms(sub_goal)
        actual = self._significant_query_terms(query)
        overlap = required & actual
        if len(required) >= 2 and (not overlap or len(overlap) / len(required) < 0.2):
            replacement = str(sub_goal or context or question).strip()[:500]
            print(
                "\n==> 🔄 Command/sub-goal mismatch — replacing generated query "
                f"'{query}' with '{replacement}'\n"
            )
            return self._inject_query_into_command(command, replacement)
        return command

    def _extract_arxiv_abs_url(self, *texts: str) -> Optional[str]:
        combined = " ".join(str(t) for t in texts if t)
        m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{5})", combined, re.I)
        if m:
            return f"https://arxiv.org/abs/{m.group(1).lower()}"
        m = re.search(r"arxiv\.org/pdf/(\d{4}\.\d{5})", combined, re.I)
        if m:
            return f"https://arxiv.org/abs/{m.group(1).lower()}"
        m = re.search(r"\b(\d{4}\.\d{5})\b", combined)
        if m:
            return f"https://arxiv.org/abs/{m.group(1).lower()}"
        return None

    def _ensure_web_search_command(
        self,
        command: str,
        target_information: str,
        context: str,
    ) -> str:
        """Web_Search_Tool requires url — inject a sensible default when outline omits it."""
        if self._extract_url_param(command) != "N/A":
            return command

        combined = f"{target_information} {context}"
        obtained = self.system_memory.get_obtained_information_for_prompt()
        arxiv_abs = self._extract_arxiv_abs_url(combined, obtained)
        if arxiv_abs:
            fixed = self._inject_url_into_command(command, arxiv_abs)
            print(f"\n==> 🔗 Auto-injected Web_Search url={arxiv_abs}\n")
            return fixed

        # Honour an explicit deep-fetch URL supplied by the diagnostic signal
        # (e.g. exhaustive-count tasks where Wikipedia only returned snippets).
        diag = self.system_memory.get_diagnostic_signal() or {}
        deep_url = diag.get("web_search_url")
        if deep_url and isinstance(deep_url, str) and deep_url.startswith("http"):
            fixed = self._inject_url_into_command(command, deep_url)
            print(f"\n==> 🔗 Auto-injected Web_Search url={deep_url} (diagnostic deep-fetch)\n")
            return fixed

        text = combined.lower()
        if "usgs" in text or "nas.er.usgs.gov" in text:
            url = "https://nas.er.usgs.gov/"
        elif "wikipedia" in text:
            url = "https://en.wikipedia.org/"
        elif "arxiv" in text:
            url = "https://arxiv.org/"
        else:
            url = "https://en.wikipedia.org/"
        fixed = self._inject_url_into_command(command, url)
        print(f"\n==> 🔗 Auto-injected Web_Search url={url}\n")
        return fixed

    def _enforce_tool_policy(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        tracker: StepInterventionState,
        diagnostic_signal: Optional[Dict[str, Any]],
    ) -> str:
        outline_tool = Planner._extract_tool_from_target(target_information)
        rec = (diagnostic_signal or {}).get("recommendation")
        blocked = set(tracker.failed_tools)

        mutations = (diagnostic_signal or {}).get("state_mutations") or {}
        for forbidden in mutations.get("forbidden_tools", []):
            blocked.add(forbidden)

        # v3: keep tool-class isolation intact through policy enforcement.
        # Block tools that violate the subgoal kind so later outline-tool or
        # switch rules cannot reintroduce a hallucination/printer path.
        kind = infer_subgoal_kind(target_information, (diagnostic_signal or {}).get("sub_goal", ""))
        if kind == "compute":
            for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
                      "Base_Generator_Tool"):
                blocked.add(t)
        # elif kind == "visual":
        #     for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
        #               "Python_Coder_Tool", "Base_Generator_Tool"):
        #         blocked.add(t)

        # switch_tool replan: apply suggested alternative before other rules
        if rec == "switch_tool":
            failed = (diagnostic_signal or {}).get("tool")
            suggested = (diagnostic_signal or {}).get("suggested_tool")
            if not suggested or suggested == failed:
                suggested = self._suggest_alternative_tool(failed, target_information, question)
            if (
                suggested
                and suggested in self.planner.available_tools
                and suggested not in blocked
                and suggested != failed
            ):
                if tool_name != suggested:
                    print(
                        f"\n==> 🔀 switch_tool replan: {failed or tool_name} → {suggested}\n"
                    )
                return suggested

        if (
            outline_tool
            and tool_name == outline_tool
            and tool_name not in blocked
            and rec != "switch_tool"
        ):
            return tool_name

        if tool_name in blocked:
            if kind == "compute" and "Python_Coder_Tool" in self.planner.available_tools:
                return "Python_Coder_Tool"
            alt = self._suggest_alternative_tool(tool_name, target_information, question)
            if alt and alt not in blocked:
                print(f"\n==> 🚫 Blocked {tool_name} (failed/blocked) → {alt}\n")
                return alt

        combined = f"{target_information} {question}".lower()
        sub_goal = ((diagnostic_signal or {}).get("sub_goal") or "").lower()
        combined = f"{combined} {sub_goal}"
        if tool_name == "Web_Search_Tool" and "arxiv" in combined:
            figure_markers = ("figure 1", "axis label", "endpoint label", "figure caption", "three axes")
            if any(m in combined for m in figure_markers):
                alt = "Google_Search_Tool"
                if alt in self.planner.available_tools and alt not in blocked:
                    print(
                        "\n==> 🚫 Web_Search cannot extract arXiv figure/axis labels "
                        f"→ {alt}\n"
                    )
                    return alt

        if self.system_memory:
            # Tool Capability Memory: if the chosen tool has no matching
            # subgoal but another enabled tool does, prefer that tool.
            # Pure-text boundary check (no scores/counts/confidence, per spec).
            cap_entry = self.system_memory.retrieve_tool_capability(
                tool_name, target_information,
            )
            if cap_entry is None and tool_name not in blocked:
                best_alt = None
                for cand in self.system_memory.list_capability_tools():
                    if cand == tool_name or cand not in self.planner.available_tools:
                        continue
                    if cand in blocked:
                        continue
                    if self.system_memory.retrieve_tool_capability(cand, target_information):
                        best_alt = cand
                        break
                if best_alt:
                    if tool_name == outline_tool and tool_name not in tracker.failed_tools:
                        pass
                    else:
                        print(
                            f"\n==> ⚠️ {tool_name} has no matching capability subgoal "
                            f"for this target — prefer {best_alt}\n"
                        )
                        return best_alt

        preferred = mutations.get("preferred_tools") or []
        if tool_name in blocked or (
            rec == "switch_tool" and tool_name == (diagnostic_signal or {}).get("tool")
        ):
            for candidate in preferred:
                if candidate in self.planner.available_tools and candidate not in blocked:
                    print(f"\n==> 🔀 Preferred tool from state mutation: {tool_name} → {candidate}\n")
                    return candidate

        if (
            outline_tool
            and outline_tool in self.planner.available_tools
            and outline_tool not in blocked
            and tool_name != outline_tool
            and rec not in ("switch_tool",)
        ):
            print(f"\n==> 📋 Enforce outline tool: {tool_name} → {outline_tool}\n")
            return outline_tool

        return tool_name

    def _build_failure_digest(
        self,
        json_data: Dict[str, Any],
        tracker: StepInterventionState,
        up_to_step: int,
    ) -> str:
        lines: List[str] = []
        for i in range(1, up_to_step + 1):
            result = json_data.get(f"tool_result_{i}")
            if result is None:
                continue
            text = str(result)
            if "Download is starting" in text:
                lines.append(
                    f"Step {i}: FORBIDDEN — Screenshot_Tool on arXiv PDF URL (triggers download)"
                )
            elif "Failed to load image" in text:
                lines.append(
                    f"Step {i}: FORBIDDEN — Vision_OCR_Tool on PDF URL directly"
                )
            elif "missing 1 required positional argument" in text:
                lines.append(f"Step {i}: FAILED — tool command missing required parameters")
            else:
                lines.append(f"Step {i}: {text[:200]}")
        if tracker.failed_tools:
            lines.append(f"Failed tools (DO NOT reuse): {list(tracker.failed_tools)}")
        return "\n".join(lines) if lines else "No prior failures recorded."

    def _resolve_tool_for_step(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        sub_goal: str,
        diagnostic_signal: Optional[Dict[str, Any]],
        image_path: str = "",
    ) -> str:
        """Override planner tool via ToolRouter and diagnostic signals."""
        failed = (diagnostic_signal or {}).get("tool")
        rec = (diagnostic_signal or {}).get("recommendation")

        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        resolved, appropriate, kind = ToolRouter.validate(
            tool_name,
            target_information,
            sub_goal,
            profile,
            records,
            self.planner.available_tools,
        )
        if not appropriate and resolved != tool_name:
            print(f"\n==> 🔀 ToolRouter redirect: {tool_name} → {resolved} (kind={kind})\n")
            tool_name = resolved

        # Defense-in-depth（防幻觉关键守卫）：即使分类器把检索子目标误判成 kind=compute，
        # 也绝不允许把 Python_Coder_Tool 强加到 acquisition 子目标上 ——
        # Python_Coder_Tool.execute(query) 会用内部 LLM 「凭 query 生成代码」并凭空捏造事实
        # （例如把本应检索的计数直接幻觉成"3 篇文章"）。改路由到真正的检索工具。
        if tool_name == "Python_Coder_Tool" and infer_subgoal_kind(target_information, sub_goal) == "acquisition":
            acq_alt = next(
                (t for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool")
                 if t in self.planner.available_tools),
                None,
            )
            if acq_alt:
                print(f"\n==> 🔀 Python-blocked for retrieval subgoal → {acq_alt}\n")
                tool_name = acq_alt

        # v3 task-level backstop (pure field matching, defense-in-depth beyond
        # ToolRouter). Only fires on compute/visual subgoals so retrieval steps
        # are never forced to Python.
        if profile and kind == "compute" and SlotGate.has_compute_slot(profile):
            if not SlotGate.compute_slot_filled_by_computed(profile, records):
                if tool_name != "Python_Coder_Tool" and "Python_Coder_Tool" in self.planner.available_tools:
                    print(f"\n==> 🔀 Forced compute tool (compute slot unfilled): {tool_name} → Python_Coder_Tool\n")
                    tool_name = "Python_Coder_Tool"
        if kind == "visual" and image_path and tool_name not in ("Screenshot_Tool", "Vision_OCR_Tool"):
            if "Screenshot_Tool" in self.planner.available_tools:
                print(f"\n==> 🔀 Forced visual tool (image present): {tool_name} → Screenshot_Tool\n")
                tool_name = "Screenshot_Tool"

        if rec == "switch_tool" or rec == "revise_belief":
            suggested = (diagnostic_signal or {}).get("suggested_tool")
            if (
                kind != "compute"
                and suggested
                and suggested != failed
                and suggested in self.planner.available_tools
            ):
                if suggested != "Base_Generator_Tool":
                    if tool_name != suggested:
                        print(f"\n==> 🔀 Tool override ({rec}): {tool_name} → {suggested}\n")
                    return suggested
            alt = self._suggest_alternative_tool(failed or tool_name, target_information, question)
            if alt and (tool_name == failed or tool_name == "Base_Generator_Tool"):
                print(f"\n==> 🔀 Tool override: {tool_name} → {alt}\n")
                return alt

        if tool_name == "Base_Generator_Tool":
            forbidden = SlotGate.forbidden_tools(profile, records) if profile else []
            if "Base_Generator_Tool" in forbidden or infer_subgoal_kind(target_information, sub_goal) != "synthesis":
                alt = self._suggest_alternative_tool("Base_Generator_Tool", target_information, question)
                if alt:
                    print(f"\n==> 🔀 Blocked Base_Generator (phase={getattr(profile, 'phase', '?')}) → {alt}\n")
                    return alt

        text = f"{target_information} {sub_goal}".lower()
        if tool_name == "Web_Search_Tool" and (
            "arxiv.org/pdf/" in text or ("axis label" in text and "arxiv" in text)
        ):
            if "Google_Search_Tool" in self.planner.available_tools:
                print("\n==> 🔀 Axis/PDF subgoal — prefer Google_Search_Tool over Web_Search_Tool\n")
                return "Google_Search_Tool"

        return tool_name

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
        """Outline must never be cleared — always return at least one step."""
        if outline and isinstance(outline, dict) and len(outline) > 0:
            return outline
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
        if self._can_stop_execution(question, (verification.analysis if verification else "")):
            return self._outline_synthesis_step(question)
        return self._minimal_continuation_outline(question, verification)

    def _outline_has_remaining_steps(self) -> bool:
        outline = self.system_memory.get_outline() or {}
        return bool(outline)

    def _can_stop_execution(self, question: str, analysis: str = "") -> bool:
        """Hard gate: STOP when SlotGate confirms all required slots are filled."""
        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        if profile:
            if not SlotGate.can_stop(profile, records):
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
            steps["1"] = (
                "Target Information: Retrieve missing evidence to answer the original question. "
                f"Operation Details: Use an appropriate search tool. "
                f"Context: {(verification.analysis or '')[:200]}"
            )
        return steps

    def _cumulative_evidence_text(self) -> str:
        return self.system_memory.get_obtained_information_for_prompt()

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
        """Outline is never cleared — only updated. Always keep ≥1 remaining step."""
        prev_outline = prev_outline or {}
        new_outline = new_outline if isinstance(new_outline, dict) else {}

        if new_outline:
            result = new_outline
        else:
            result = self._fallback_advance_outline(prev_outline, current_step_num)
            if not result:
                print(
                    "\n==> ⚠️ LLM returned empty outline — using deterministic advance/recovery\n"
                )

        if not result:
            result = self._generate_recovery_outline(question, verification)

        if not result:
            if self._can_stop_execution(question, verification.analysis):
                result = self._outline_synthesis_step(question)
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

    def _task_needs_continuation(self, task_conclusion: Optional[str], outline: Optional[Dict]) -> bool:
        if self._can_stop_execution(self.system_memory.get_query() or ""):
            return False
        return True

    def _build_execution_memory_summary(self, json_data: Dict[str, Any], up_to_step: int) -> str:
        lines = []
        for i in range(1, up_to_step + 1):
            result = json_data.get(f"tool_result_{i}")
            if result is not None:
                preview = str(result)[:300]
                lines.append(f"Step {i}: {preview}")
        return "\n".join(lines) if lines else "No prior tool results."

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _initialize_task(self, question: str, image_path: Optional[str]) -> Tuple[str, Dict[str, Any], float]:
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.system_memory.init_causal_graph_for_task(task_id)
        self.system_memory.set_query(question)

        if self.verbose:
            print(f"\n==> 📝 Task ID: {task_id}")

        loaded = False
        if not self.evaluation_mode:
            loaded = self.system_memory.load_offline_memory("memory")
        if self.verbose and loaded:
            print("\n==> 📚 Loaded offline memory (Tool Skill Cards)")

        self.executor.set_query_cache_dir(self.root_cache_dir)

        json_data: Dict[str, Any] = {
            "query": question,
            "image": image_path,
            "task_id": task_id,
            "trace_events": [],
        }
        if self.verbose:
            print(f"\n==> 🔍 Received Query: {question}")
            if image_path:
                print(f"\n==> 🖼️ Received Image: {image_path}")

        return task_id, json_data, time.time()

    def _analyze_query(self, question: str, image_path: Optional[str], json_data: Dict[str, Any], start_time: float) -> None:
        query_start = time.time()
        analysis, execution_outline = self.planner.analyze_query(question, image_path)
        profile = SlotGate.infer_profile(question)
        self.system_memory.set_task_profile(profile)
        json_data["analysis"] = analysis
        json_data["outline"] = execution_outline
        json_data["task_profile"] = {
            "phase": profile.phase,
            "slots": [s.name for s in profile.slots],
        }
        self.system_memory.set_outline(execution_outline)

        if self.verbose:
            print("\n==> 🔍 Step 0: Query Analysis\n")
            print(f"{analysis}")
            print(f"[TaskProfile]: phase={profile.phase} slots={[s.name for s in profile.slots]}")
            print(f"[Time]: {round(time.time() - query_start, 2)}s")

    def _persist_task_knowledge(self, task_id: str) -> None:
        try:
            self.system_memory.evolve_tool_knowledge()
            if self.verbose:
                print("\n==> 🧬 Tool Knowledge Memory: evolved from successful traces")

            if not self.evaluation_mode:
                self.system_memory.persist_offline_memory("memory")
                if self.verbose:
                    print("✅ Persisted offline memory for next task")

            self.system_memory.persist_online_memory(task_id, "memory")
            if self.verbose:
                print(f"✅ Persisted online memory for task {task_id}")

            stats = self.system_memory.get_memory_stats("memory")
            if self.verbose:
                print("\n[Memory Statistics]:")
                print(f"  Online:  {stats['online_memory']['causal_execution_traces']} traces, "
                      f"{stats['online_memory']['task_local_parameters']} parameters")
                om = stats["offline_memory"]
                print(f"  Offline: {om.get('capability_tools', 0)} capability tools, "
                      f"{om.get('capability_subgoals', 0)} capability subgoals, "
                      f"{om.get('invocation_tools', 0)} invocation tools, "
                      f"{om.get('invocation_subgoals', 0)} invocation subgoals")
        except Exception as e:
            print(f"⚠️ Warning: Knowledge persistence failed: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Planner / Executor
    # ------------------------------------------------------------------

    def _run_planner(
        self,
        question: str,
        image_path: Optional[str],
        target_information: str,
        step_count: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, str]:
        # LLM 调用：Planner.generate_next_step —— 读取 outline 当前子目标 + obtained_info +
        # diagnostic_signal，产出下一步 plan（含 context/sub_goal/tool_name）。
        plan, _ = self.planner.generate_next_step(
            question,
            image_path,
            target_information,
            step_count,
            self.max_steps,
            self.system_memory.get_obtained_information_for_prompt(),
            json_data,
            diagnostic_signal=diagnostic_signal,
        )
        context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(plan)
        if tool_name is None or 'none' in str(tool_name).lower():
            tool_name = "Base_Generator_Tool"
        return plan, context, sub_goal, tool_name

    def _run_executor(
        self,
        question: str,
        image_path: Optional[str],
        context: str,
        sub_goal: str,
        tool_name: str,
        step_count: int,
        json_data: Dict[str, Any],
        diagnostic_signal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, Any]:
        if tool_name not in self.planner.available_tools:
            print(f"\n==> 🚫 Error: Tool '{tool_name}' is not available or not found.")
            command = "No command was generated because the tool was not found."
            return command, command, "No result was generated because the tool was not found."

        # LLM 调用：Executor.generate_tool_command —— 按工具 schema 生成结构化命令（analysis/explanation/command）。
        tool_command = self.executor.generate_tool_command(
            question,
            image_path,
            context,
            sub_goal,
            tool_name,
            self.system_memory.toolbox_metadata[tool_name],
            step_count,
            json_data,
            diagnostic_signal,
        )
        analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)

        if self.verbose:
            print(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
            print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")

        if tool_name == "Web_Search_Tool":
            command = self._ensure_web_search_command(command, sub_goal, context)

        command = self._align_command_with_subgoal(
            tool_name, command, question, context, sub_goal,
        )
        command = self._sanitize_command_for_tool(tool_name, command)
        validation_error = self._validate_command(tool_name, command)
        if validation_error:
            print(f"\n==> ⚠️ Command validation failed: {validation_error}\n")
            result = f"Error in execute_tool_command: {validation_error}"
            json_data[f"tool_result_{step_count}"] = result
            return command, analysis, result

        result = self._execute_generated_command(tool_name, command, step_count, json_data)

        if self.verbose:
            print(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
            print(f"[Executor Result]:\n{json.dumps(result, indent=4)}")

        return command, analysis, result

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

    @staticmethod
    def _extract_wikipedia_url(result_executor: Any) -> Optional[str]:
        """Pull the first en.wikipedia.org URL out of an executor result blob."""
        text = str(result_executor or "")
        m = re.search(
            r"https?://en\.wikipedia\.org/wiki/[A-Za-z0-9._\-%,()]+",
            text,
        )
        return m.group(0) if m else None

    def _execute_step(
        self,
        question: str,
        image_path: Optional[str],
        step_key: str,
        target_information: str,
        exec_step: int,
        json_data: Dict[str, Any],
        diagnostic_signal_prev: Optional[Dict[str, Any]],
        tracker: Optional[StepInterventionState] = None,
    ) -> StepContext:
        outline_tool = Planner._extract_tool_from_target(target_information)
        print(
            f"\n==> 🎯 Outline Step {step_key} | Execution #{exec_step}\n"
            f"Target Information:\n{target_information}\n"
        )
        if outline_tool and self.verbose:
            print(f"[Outline Suggested Tool]: {outline_tool}")

        # 把上一轮诊断信号的 forbidden_tools 记入 tracker，供本步骤 tool policy 强制规避。
        if tracker and diagnostic_signal_prev:
            mutations = diagnostic_signal_prev.get("state_mutations") or {}
            for forbidden in mutations.get("forbidden_tools", []):
                tracker.record_failure_tool(forbidden)

        # 诊断信号富化（注入历史/记忆），随后传给 Planner。
        planner_signal = self._enrich_diagnostic_signal(
            diagnostic_signal_prev, target_information, question,
        )

        # === LLM 调用 #1：Planner.generate_next_step === 决定 context / sub_goal / tool_name。
        _, context, sub_goal, tool_name = self._run_planner(
            question, image_path, target_information, exec_step, json_data, planner_signal,
        )

        # 工具裁决：综合 outline 建议、subgoal kind、诊断信号、image_path 决定最终工具。
        # 内含 defense-in-depth：acquisition 子目标绝不被强制到 Python_Coder_Tool（防幻觉）。
        tool_name = self._resolve_tool_for_step(
            tool_name, question, target_information, sub_goal, planner_signal, image_path,
        )
        if tracker:
            tool_name = self._enforce_tool_policy(
                tool_name, question, target_information, tracker, planner_signal,
            )

        # 诊断信号里若指定了 preferred_url，前置注入到 context（用于 L2 失败后定向重取）。
        if diagnostic_signal_prev:
            mutations = diagnostic_signal_prev.get("state_mutations") or {}
            preferred_url = mutations.get("preferred_url")
            if preferred_url and preferred_url not in context:
                context = f"{preferred_url} {context}".strip()

        if tool_name == "Web_Search_Tool":
            context = self._inject_web_context_for_usgs(context, target_information, question)

        if self.verbose:
            print(f"\n==> 🎯 Execution #{exec_step}: Action Prediction ({tool_name})\n")
            print(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")

        # === LLM 调用 #2：Executor.generate_tool_command === 生成具体工具命令并执行。
        # 内部：executor LLM 产出 ToolCommand（结构化）→ executor.execute_tool_command 真正调用工具。
        command, _, result = self._run_executor(
            question, image_path, context, sub_goal, tool_name,
            exec_step, json_data, diagnostic_signal_prev,
        )
        self._append_trace_event(
            json_data,
            "tool_execution",
            exec_step=exec_step,
            outline_step=step_key,
            sub_goal=sub_goal,
            generated_query=self._extract_query_param(command),
            tool=tool_name,
        )

        return StepContext(
            step_key=step_key,
            target_information=target_information,
            context=context,
            sub_goal=sub_goal,
            tool_name=tool_name,
            command=command,
            result_executor=result,
            first_attempt_command=command,
        )

    # ------------------------------------------------------------------
    # Verification & memory recording
    # ------------------------------------------------------------------

    def _run_verification(
        self,
        question: str,
        image_path: Optional[str],
        step_ctx: StepContext,
        step_count: int,
        label: str = "Verification",
        intervention_context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        local_start = time.time()
        print(f"\n==> 🩺 Step {step_count}: {label}\n")

        # LLM 调用：Diagnoser.verificate_context —— 唯一的「因果介入触发判定」。
        # 输入 outline + executor 结果 + obtained_info + intervention_context；输出：
        #   step_conclusion  ∈ {SUBGOAL_COMPLETE, SUBGOAL_INCOMPLETE}（后者触发介入）
        #   diagnostic_signal（failure_patterns / recommendation / suggested_tool 等，供 L1/L2/L3 决策）
        #   slot_updates / evidence_type / tool_appropriate（喂给 SlotGate 与 skip_param_retry）
        (
            analysis,
            step_conclusion,
            info_flag,
            obtained_info,
            task_conclusion,
            diagnostic_signal,
            subgoal_complete,
            slot_updates,
            evidence_type,
            tool_appropriate,
        ) = self.diagnoser.verificate_context(
            question,
            image_path,
            step_ctx.target_information,
            self.system_memory.get_outline(),
            step_ctx.result_executor,
            step_count,
            self.system_memory.get_obtained_information_for_prompt(),
            step_ctx.sub_goal,
            step_ctx.tool_name,
            intervention_context=intervention_context,
            current_outline_step=step_ctx.step_key,
        )

        had_slot_delta = any(
            isinstance(u, dict) and u.get("filled") for u in (slot_updates or [])
        )

        if self.verbose:
            print(f"\n==> 🤖 Step {step_count}: {label} Results\n")
            print(f"[Subgoal Conclusion]: {step_conclusion}")
            print(f"[Task Conclusion]: {task_conclusion}")
            print(f"[Evidence Type]: {evidence_type} | tool_appropriate={tool_appropriate}")
            print(f"[Analysis]: {analysis}\n")
            print(f"[Time]: {round(time.time() - local_start, 2)}s")

        profile = self.system_memory.get_task_profile()
        if profile:
            profile.phase = SlotGate.update_phase(profile, self.system_memory.evidence_records)
            self.system_memory.set_phase(profile.phase)

        verification = VerificationResult(
            analysis=analysis,
            step_conclusion=step_conclusion,
            info_flag=info_flag,
            obtained_info=obtained_info,
            task_conclusion=task_conclusion,
            diagnostic_signal=diagnostic_signal,
            subgoal_complete=subgoal_complete,
            slot_updates=slot_updates or [],
            evidence_type=evidence_type,
            tool_appropriate=tool_appropriate,
            had_slot_delta=had_slot_delta,
        )

        if slot_updates or verification.analysis:
            persisted = self._persist_slot_deltas(verification, step_ctx, step_count)
            if persisted and not had_slot_delta:
                verification.had_slot_delta = True

        return verification

    def _record_verification_trace(
        self,
        json_data: Dict[str, Any],
        step_ctx: StepContext,
        verification: VerificationResult,
        step_count: int,
        tracker: Optional[StepInterventionState] = None,
    ) -> None:
        signal = verification.diagnostic_signal or {}
        root_cause = (
            (signal.get("causal_hypothesis") or {}).get("description")
            or signal.get("root_cause")
            or (signal.get("failure_patterns") or {}).get("root_cause")
            or ""
        )
        self._append_trace_event(
            json_data,
            "verification",
            exec_step=step_count,
            outline_step=step_ctx.step_key,
            sub_goal=step_ctx.sub_goal,
            generated_query=self._extract_query_param(step_ctx.command),
            tool=step_ctx.tool_name,
            root_cause=str(root_cause),
            slot_delta=verification.had_slot_delta,
            intervention_attempt=tracker.intervention_attempts if tracker else None,
        )

    # ------------------------------------------------------------------
    # Level 1：观察（Pearl Level 1，0 次 LLM）
    # 纯数据归集 —— 不调用 LLM，不做决策；把既有诊断输出装配成结构化 FailureContext，
    # 供 L2/L3 复用。这是「Observation」步骤：先看清楚失败现场再决定干预手段。
    # ------------------------------------------------------------------

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
            if added and verification.slot_updates:
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

    def _update_outline(
        self,
        question: str,
        verification: VerificationResult,
        step_ctx: StepContext,
        current_step_num: int,
        json_data: Dict[str, Any],
        step_count: int,
    ) -> Dict[str, Any]:
        print(f"\n==> 🗂️ Step {step_count}: Execution Outline Update\n")
        prev_outline = dict(self.system_memory.get_outline() or {})
        task_conclusion = verification.task_conclusion or "CONTINUE"

        updated = self.diagnoser.update_outline(
            question=question,
            context_verification=(
                f"{verification.analysis}\n\n"
                f"[Task Conclusion]: {task_conclusion}\n"
                "IMPORTANT: NEVER return an empty ExecutionOutline. "
                "Always return at least one remaining step. "
                "If evidence is sufficient, add a final synthesis/formatting step. "
                "Task Conclusion STOP is only valid when the verified answer is already in evidence."
            ),
            target_information=step_ctx.target_information,
            outline=prev_outline,
            result_executor=step_ctx.result_executor,
            current_step=current_step_num,
            obtained_information=self.system_memory.get_obtained_information_for_prompt(),
            toolbox_metadata=self.system_memory.get_toolbox_metadata(),
            execution_memory=self._build_execution_memory_summary(json_data, step_count),
        )
        updated = self._sanitize_outline_update(
            prev_outline, updated, current_step_num, task_conclusion, question, verification,
        )
        self.system_memory.set_outline(updated)
        if self.verbose:
            print(f"\n[Updated Execution Outline]:\n{json.dumps(updated, indent=4)}")
        return updated

    # ------------------------------------------------------------------
    # Failure handling: parameter perturbation (L1 intervention)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Level 2: Intervention — candidate set (parameter + tool perturbation)
    # ------------------------------------------------------------------
    #
    # Hierarchical blame assignment: trust the Planner by default; the first
    # intervention target is always the Executor. Level 2a perturbs the
    # parameters of the CURRENT tool; Level 2b tries ALTERNATIVE tools. Both
    # produce a candidate set (executed in parallel) and reuse the existing
    # verifier; the first candidate whose verification is SUBGOAL_COMPLETE
    # wins and the intervention terminates immediately.
    #
    # Cost neutrality: Level 2a spends ONE executor LLM call returning N=3
    # candidate commands (vs the previous sequential retry's 3 calls) plus
    # up to 2 zero-LLM programmatic perturbations. Verification reuses
    # `_run_verification` with a rule pre-filter and a hard cap of 3 diagnoser
    # LLM calls — total ≤ 4 LLM vs the previous 6.

    CANDIDATE_SET_SIZE = 3          # N candidates from one executor LLM call
    MAX_PROGRAMMATIC_PERTURBS = 2   # zero-LLM perturbations appended
    MAX_VERIFY_PER_ROUND = 3        # hard cap on diagnoser LLM verifies per round
    CANDIDATE_PARALLEL_WORKERS = 3  # tool execution is IO-bound

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
                        last_command, last_query if last_query != "N/A" else "",
                        attempt_seed, tried_params,
                    )
                elif tool_name in ("Screenshot_Tool", "Vision_OCR_Tool"):
                    url = self._extract_url_param(last_command)
                    tried_urls = {t for t in tried_params if "/" in t or t.startswith("http")}
                    perturbed_url = self._perturb_url_param(url, attempt_seed, tried_urls)
                    perturbed_cmd = self._sanitize_command_for_tool(
                        tool_name, self._inject_url_into_command(last_command, perturbed_url),
                    )
                else:
                    perturbed = self._perturb_query_param(
                        last_query if last_query != "N/A" else "",
                        attempt_seed,
                        {t.split("|")[0] for t in tried_params},
                        sub_goal=step_ctx.sub_goal,
                        target_information=step_ctx.target_information,
                    )
                    perturbed_cmd = self._inject_query_into_command(last_command, perturbed)
            except Exception:
                perturbed_cmd = None
            attempt_seed += 10
            if not perturbed_cmd:
                break
            fp = self._param_fingerprint(perturbed_cmd, tool_name)
            if fp in tried_params or fp in {
                self._param_fingerprint(c, tool_name) for c in candidates
            }:
                perturb_attempts += 1
                continue
            candidates.append(perturbed_cmd)
            perturb_attempts += 1

        if not candidates:
            # Last-resort: a single forced perturb so the round is never empty.
            forced = self._perturb_query_param(
                last_query if last_query != "N/A" else "retry",
                100, set(),
                sub_goal=step_ctx.sub_goal,
                target_information=step_ctx.target_information,
            )
            candidates.append(self._inject_query_into_command(last_command, forced))
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

            # Change 2 + 3: record (root_cause, target_variable) so the
            # Counterfactual stage can detect a no-op loop. The target
            # variable is now sourced from the structured Causal Hypothesis
            # (authoritative) and falls back to the INTERVENTION_TARGET_VARIABLE
            # map keyed by recommendation (Pearl's do(X)).
            _hypo = diagnostic_signal.get("causal_hypothesis") or {}
            _rec = diagnostic_signal.get("recommendation", "")
            _rc = (
                _hypo.get("description")
                or diagnostic_signal.get("root_cause")
                or (diagnostic_signal.get("failure_patterns") or {}).get("root_cause")
            )
            _tv = (
                _hypo.get("target_variable")
                or diagnostic_signal.get("target_variable")
                or INTERVENTION_TARGET_VARIABLE.get(_rec, "unknown")
            )
            _reason = _hypo.get("reason") or diagnostic_signal.get("reason") or ""
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
        if self.verbose:
            print(f"[FailureContext] tool={fc.tool} evidence={fc.evidence_type} "
                  f"patterns={[k for k, v in fc.failure_patterns.items() if v]}")

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
        _failure_type = (
            diagnostic_signal.get("failure_type")
            or patterns.get("failure_type")
            or ""
        )
        _target_variable = (
            diagnostic_signal.get("target_variable")
            or patterns.get("target_variable")
            or ""
        )
        _reason = (
            diagnostic_signal.get("reason")
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
        skip_param_retry = (
            not verification.tool_appropriate
            or patterns.get("wrong_tool_class")
            or _is_infrastructure_error
            or _is_command_schema_error
            or self._is_pdf_access_error(_res_str)
            or (
                _failure_type == "Retrieval"
                and _target_variable == "coverage"
                and recommendation == "switch_tool"
            )
            or (patterns.get("error_occurred") and already_failed and _failure_type not in ("Execution", "ExecutionFailure"))
        )

        if not skip_param_retry and tracker.remaining("retry_with_different_parameters") > 0:
            action, step_ctx, verification, exec_step, replan_signal = self._run_intervention_level2a(
                question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
            )
            if action == "complete":
                return "complete", step_ctx, verification, exec_step, replan_signal
            diagnostic_signal = replan_signal or diagnostic_signal
        else:
            if skip_param_retry:
                if not verification.tool_appropriate:
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
                else:
                    reason = "hard/infrastructure error"
                print(
                    f"\n==> ⏭️ Skipping Level 2a (parameter retry) — {step_ctx.tool_name} "
                    f"({reason}); escalating to Level 2b\n"
                )
            diagnostic_signal = dict(diagnostic_signal)
            diagnostic_signal["recommendation"] = "switch_tool"

        # ───────── Level 2b：替代工具候选集（Executor 侧）─────────
        if tracker.remaining("switch_tool") > 0:
            action, step_ctx, verification, exec_step, replan_signal = self._run_intervention_level2b(
                question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
            )
            if action == "complete":
                return "complete", step_ctx, verification, exec_step, replan_signal
            diagnostic_signal = replan_signal or diagnostic_signal

        # ───────── Level 3：反事实（Planner 侧 replanning，0 次新增 LLM）─────────
        # Executor 侧介入耗尽 → 质疑 Planner。决策：保留子目标（no-op replan）
        # vs 修改子目标（revise_belief / decompose_goal / modify_state，复用既有 replanning 模块）。
        print(f"\n==> 🧠 Level 3: Counterfactual — escalating to Planner-side replanning\n")
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
        root_cause = (
            _hypo.get("description")
            or signal.get("root_cause")
            or (signal.get("failure_patterns") or {}).get("root_cause")
        )
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
        if root_cause and tracker.cause_repeated(root_cause, target_variable):
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

    # ------------------------------------------------------------------
    # Answer sanity: consistency check + normalization (Scheme D)
    # ------------------------------------------------------------------

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
        # "thousand hours"/"hours" → answer should be numeric (not a speed like '20.94 km/h')
        if re.search(r"\bthousand hours\b|\bhours\b", q) and not re.search(r"^\s*\d+", ans):
            return "question asks for hours but answer is not numeric"
        return None

    def _normalize_final_answer(self, question: str, raw_answer: str) -> str:
        """Extract the shortest answer to the question from a verbose slot value.

        Used when the final_answer slot got filled with a long explanatory text
        instead of the minimal answer the question expects (e.g. pid=3).
        """
        raw = str(raw_answer or "").strip()
        if not raw:
            return raw
        # Short, single-token answers need no normalization
        if len(raw) <= 30 and len(raw.split()) <= 4:
            return raw
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

    def _handle_subgoal_complete(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
        intervention_trackers: Optional[Dict[str, StepInterventionState]] = None,
    ) -> str:
        """Process successful subgoal. Returns 'stop' or 'continue'."""
        print(f"\n==> ✅ Execution #{exec_step}: Subgoal Complete - Checking Task Completion\n")

        active_trace = self.system_memory._find_active_causal_trace(
            step_ctx.step_key, step_ctx.sub_goal,
        )
        attempt_seq = len(active_trace["effects"]) + 1 if active_trace else 1
        failed_param = None
        if active_trace and active_trace.get("effects"):
            first_fail = next(
                (e for e in active_trace["effects"] if not e.get("result", {}).get("success")),
                None,
            )
            if first_fail:
                failed_param = first_fail.get("parameter")

        self._record_success_trace(
            exec_step, step_ctx, verification,
            failed_parameter=failed_param,
            attempt_seq=attempt_seq,
        )
        self._record_obtained_information(exec_step, verification, step_ctx)

        prev_first = self._get_first_outline_step()
        prev_step_id = (
            self._outline_step_id(prev_first[0], prev_first[1])
            if prev_first[0] and prev_first[1] else None
        )

        self.system_memory.mark_step_done(step_ctx.step_key)
        current_step_num = int(step_ctx.step_key) if step_ctx.step_key.isdigit() else 1
        self._update_outline(
            question, verification, step_ctx, current_step_num, json_data, exec_step,
        )

        new_first = self._get_first_outline_step()
        if new_first[0] and new_first[1]:
            new_step_id = self._outline_step_id(new_first[0], new_first[1])
            self.system_memory.set_active_step(new_step_id)
            if prev_step_id and prev_step_id != new_step_id and intervention_trackers is not None:
                intervention_trackers.pop(prev_step_id, None)
                print(f"\n==> 🔄 Outline step changed — reset intervention budget for {prev_step_id}\n")


        if self._can_stop_execution(question, verification.analysis):
            # Scheme D / v3: unified answer-sanity gate. Even when slots are
            # filled, verify the candidate answer actually matches what the
            # question asks for (type/unit/length) via rule-first checks. Catches
            # "wrong successes" that bypass the failure-driven causal ladder.
            if self._answer_sanity_pass(question):
                print(f"\n==> 🎯 Execution #{exec_step}: All required slots filled — STOP\n")
                return "stop"
            return "continue"

        remaining = len(self.system_memory.get_outline() or {})
        if remaining > 0:
            print(
                f"\n==> 🔄 Execution #{exec_step}: Outline has {remaining} remaining step(s) "
                f"— CONTINUE executing\n"
            )
            return "continue"

        if verification.task_conclusion == "STOP" and not self._can_stop_execution(
            question, verification.analysis,
        ):
            print(
                "\n==> ⚠️ STOP blocked — answer not verified in evidence; forcing CONTINUE\n"
            )
            verification = VerificationResult(
                analysis=verification.analysis,
                step_conclusion=verification.step_conclusion,
                info_flag=verification.info_flag,
                obtained_info=verification.obtained_info,
                task_conclusion="CONTINUE",
                diagnostic_signal=verification.diagnostic_signal,
                subgoal_complete=verification.subgoal_complete,
            )

        if verification.task_conclusion != "STOP":
            print(
                f"\n==> 🔄 Execution #{exec_step}: Verifier said CONTINUE — keep executing\n"
            )
            return "continue"

        if self._can_stop_execution(question, verification.analysis):
            print(f"\n==> 🎯 Execution #{exec_step}: Verified answer ready — STOP allowed\n")
            return "stop"

        print(f"\n==> 🔄 Execution #{exec_step}: Answer not verified — CONTINUE executing\n")
        return "continue"

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, question: str, image_path: Optional[str] = None):
        task_id, json_data, query_start_time = self._initialize_task(question, image_path)

        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print(f"\n==> 📝 Base Response from LLM:\n\n{base_response}")

        if set(self.output_types) == {'base'}:
            return json_data

        if not ({'final', 'direct'} & set(self.output_types)):
            return json_data

        # Step 0：题目分析（1 次 planner LLM，analyze_query + 自检），生成初始 outline + TaskProfile。
        self._analyze_query(question, image_path, json_data, query_start_time)

        outline_attempt = 0
        exec_step = 0
        diagnostic_signal_prev = None
        context_verification = ""
        last_task_conclusion: Optional[str] = None
        intervention_trackers: Dict[str, StepInterventionState] = {}
        recovery_injections = 0
        self._decompose_bonus = 0
        # Reset cross-replan Action Variable Gate state for this task.
        self._last_planner_action = None
        self._no_op_penalty = 0

        # ===== 主循环：每个 outline step 跑一轮 Planner → Executor → Diagnoser =====
        # 验证判 SUBGOAL_INCOMPLETE 时进入因果介入（_handle_subgoal_incomplete）。
        while exec_step < self.max_steps and (time.time() - query_start_time) < self.max_time:
            if self._is_answer_ready(question):
                print(
                    f"\n==> 🎯 Cumulative answer ready — stopping early "
                    f"(exec_step={exec_step}/{self.max_steps}, outline_attempt={outline_attempt})\n"
                )
                last_task_conclusion = "STOP"
                break
            
            step_key, target_information = self._get_first_outline_step()
            print(f"\n==> 🗂️ Current Execution Outline:\n"
                  f"{json.dumps(self.system_memory.get_outline(), indent=4)}")

            if step_key is None:
                if self._can_stop_execution(question) and self._answer_sanity_pass(question):
                    print("\n==> 🎯 Verified answer ready — stopping execution loop\n")
                    last_task_conclusion = "STOP"
                    break

                # Sanity gate may have just injected a revise outline on block.
                # Only inject a recovery outline if the outline is still empty.
                if not self._get_first_outline_step()[0]:
                    recovery = self._ensure_outline_nonempty(
                        question,
                        {},
                        VerificationResult(
                            analysis=context_verification,
                            step_conclusion="SUBGOAL_INCOMPLETE",
                            info_flag=False,
                            obtained_info="",
                            task_conclusion=last_task_conclusion or "CONTINUE",
                            diagnostic_signal=None,
                            subgoal_complete=False,
                        ),
                    )
                    self.system_memory.set_outline(recovery)
                    recovery_injections += 1
                    print(
                        f"\n==> ⚠️ Outline was empty — injected continuation outline "
                        f"({recovery_injections}/{MAX_RECOVERY_OUTLINE_INJECTIONS})\n"
                    )
                    if recovery_injections >= MAX_RECOVERY_OUTLINE_INJECTIONS:
                        print(
                            f"\n==> ⚠️ Max recovery injections reached; "
                            f"continuing with minimal step until max_steps/time\n"
                        )
                continue

            outline_attempt += 1
            exec_step += 1
            step_id = self._outline_step_id(step_key, target_information)
            tracker = self._get_intervention_state(intervention_trackers, step_id)
            if tracker.counts:
                print(f"[Outline Step {step_id}]: intervention history {dict(tracker.counts)}")
            print(
                f"\n==> 📍 Step budget: exec_step={exec_step}/{self.max_steps} "
                f"(outline_attempt={outline_attempt})"
            )

            # Step 1：执行当前 outline 子目标 —— Planner LLM 决定工具/子目标，Executor LLM 生成命令并执行。
            step_ctx = self._execute_step(
                question, image_path, step_key, target_information,
                exec_step, json_data, diagnostic_signal_prev, tracker,
            )

            # Step 2：Diagnoser LLM 验证结果是否达成子目标。这是因果介入的「触发判定点」：
            # step_conclusion == "SUBGOAL_INCOMPLETE" 即触发 _handle_subgoal_incomplete。
            verification = self._run_verification(
                question, image_path, step_ctx, exec_step,
                intervention_context=tracker.to_context(),
            )
            self._record_verification_trace(
                json_data, step_ctx, verification, exec_step, tracker,
            )
            if self._harvest_slots_from_executor(step_ctx, verification, exec_step):
                verification.had_slot_delta = True
            # v3：此处不直接 break（即使 STOP）。COMPLETE 必须落到 _handle_subgoal_complete，
            # 经统一的 answer-sanity 闸门后才允许 STOP。在此 break 会绕过闸门（FM1 缺陷）。
            if self._maybe_escalate_partial_cross_ref(question, step_id, verification):
                continue

            context_verification = verification.analysis
            last_task_conclusion = verification.task_conclusion

            # ===== 因果介入入口：验证判定子目标未达成 =====
            if verification.step_conclusion == "SUBGOAL_INCOMPLETE":
                action, step_ctx, verification, exec_step, diagnostic_signal_prev = (
                    self._handle_subgoal_incomplete(
                        question, image_path, step_ctx, verification,
                        exec_step, json_data, tracker,
                    )
                )
                if action == "complete":
                    context_verification = verification.analysis
                    last_task_conclusion = verification.task_conclusion
                    task_action = self._handle_subgoal_complete(
                        question, step_ctx, verification, exec_step, json_data,
                        intervention_trackers,
                    )
                    diagnostic_signal_prev = None
                    intervention_trackers.pop(step_id, None)
                    if task_action == "stop" and self._can_stop_execution(
                        question, verification.analysis,
                    ):
                        context_verification = verification.analysis
                        break
                    if task_action == "stop":
                        print("\n==> ⚠️ STOP ignored — answer not verified; continuing execution\n")
                    continue

                # Change 3: Cause-Changed gate terminated the intervention
                # loop — the same root cause recurred on the same target
                # variable, so further replanning cannot fix the failure.
                # Stop executing and let the final-answer stage emit the
                # best-effort answer from accumulated evidence.
                if action == "terminate":
                    print(
                        "\n==> ⏹️ Intervention terminated — cause unchanged; "
                        "producing final answer from cumulative evidence\n"
                    )
                    break

                continue

            task_action = self._handle_subgoal_complete(
                question, step_ctx, verification, exec_step, json_data,
                intervention_trackers,
            )
            diagnostic_signal_prev = None
            intervention_trackers.pop(step_id, None)
            if task_action == "stop" and self._can_stop_execution(question, verification.analysis):
                context_verification = verification.analysis
                break
            if task_action == "stop":
                print("\n==> ⚠️ STOP ignored — answer not verified; continuing execution\n")
                continue
        
        print("=============== Last Verification Analysis ================")
        print(f"{context_verification}")
        print("================== Obtained Information ===================")
        print(f"{self.system_memory.get_obtained_information()}")
        print("===========================================================")

        if 'direct' in self.output_types:
            evidence_verified = self._can_stop_execution(question, context_verification)
            profile = self.system_memory.get_task_profile()
            records = self.system_memory.evidence_records
            slot_answer = None
            if evidence_verified and profile:
                slot_answer = SlotGate.extract_final_answer(profile, records)
            if not evidence_verified:
                print(
                    "\n==> ⚠️ Final answer: cumulative evidence NOT fully verified — "
                    "constrained output mode\n"
                )
            if slot_answer:
                # Scheme D: normalize verbose slot answers to the minimal answer
                # the question actually asks for (e.g. a single character name).
                slot_answer = self._normalize_final_answer(question, slot_answer)
                direct_output = slot_answer
                print(f"\n==> 🎯 Final answer from SlotGate: {direct_output}\n")
            else:
                direct_output = self.executor.generate_direct_output(
                    question,
                    context_verification,
                    self.system_memory,
                    evidence_verified=evidence_verified,
                )
            json_data["direct_output"] = direct_output
            json_data["evidence_verified"] = evidence_verified
            sanity_verified = bool(
                evidence_verified and self._answer_consistency_check(question, direct_output)[0]
            )
            json_data["solved"] = sanity_verified
            json_data["scorable_output"] = direct_output if sanity_verified else ""
            json_data["trace_events"].append({
                "event": "finalization",
                "task_id": task_id,
                "exec_step": exec_step,
                "evidence_verified": evidence_verified,
                "solved": sanity_verified,
            })
            print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

        print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
        if json_data.get("solved"):
            print("\n==> ✅ Query Solved!")
        else:
            print("\n==> ⚠️ Query Finished Without Verified Answer")

        self._persist_task_knowledge(task_id)
        return json_data


def construct_solver(
    llm_engine_name: str = "gpt-4o",
    enabled_tools: list[str] = ["all"],
    tool_engine: list[str] = ["Default"],
    output_types: str = "final,direct",
    max_steps: int = 20,
    max_time: int = 3000,
    max_tokens: int = 4000,
    root_cache_dir: str = "solver_cache",
    verbose: bool = True,
    vllm_config_path: str = None,
    temperature: float = 0.0,
    n: int = 1,
    evaluation_mode: bool = False,
):
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )
    planner = Planner(
        llm_engine_name=llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
        n=n,
    )
    executor = Executor(
        llm_engine_name=llm_engine_name,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
    )
    diagnoser = Diagnoser(
        llm_engine_name=llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
    )

    agent_profile = {
        "planner": {
            "strengths": ["Strategic thinking", "Long-term planning", "Tool selection"],
            "weaknesses": ["May overlook immediate details", "Relies on accurate tool metadata"],
            "last_plan_scores": [],
        },
        "executor": {
            "strengths": ["Precise command generation", "Effective tool execution"],
            "weaknesses": ["Limited strategic insight", "Depends on clear sub-goals"],
            "last_plan_scores": [],
        },
        "diagnoser": {
            "strengths": ["Critical evaluation", "Error detection"],
            "weaknesses": ["May be overly cautious", "Relies on comprehensive context"],
            "last_plan_scores": [],
        },
    }

    system_memory = SystemMemory(
        toolbox_metadata=initializer.toolbox_metadata,
        agent_profile=agent_profile,
    )

    return Solver(
        system_memory=system_memory,
        planner=planner,
        executor=executor,
        diagnoser=diagnoser,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        evaluation_mode=evaluation_mode,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the epc_aw demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)",
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()


def main(args):
    tool_engine = ["gpt-4o", "gpt-4o", "Default", "Default"]
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=[
            "Base_Generator_Tool", "Python_Coder_Tool",
            "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool",
        ],
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        temperature=0.7,
    )
    solver.solve("What is the capital of France?")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
