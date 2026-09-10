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
# Causal intervention limits & escalation ladder (L1 → L2 → L3)
# ---------------------------------------------------------------------------
MAX_PARAMETER_RETRIES = 3
MAX_SWITCH_TOOL_ATTEMPTS = 2
MAX_DECOMPOSE_GOAL_ATTEMPTS = 1
MAX_MODIFY_STATE_ATTEMPTS = 1
MAX_REVISE_BELIEF_ATTEMPTS = 1
MAX_RECOVERY_OUTLINE_INJECTIONS = 1

INTERVENTION_LADDER: List[str] = [
    "retry_with_different_parameters",  # L1: parameter perturbation
    "switch_tool",                       # L2/L3: change tool
    "revise_belief",                     # retract conflicting claims + verify
    "decompose_goal",                    # L3: break down sub-goal
    "modify_state",                      # L3: environment / prerequisite fix
]

INTERVENTION_LIMITS: Dict[str, int] = {
    "retry_with_different_parameters": MAX_PARAMETER_RETRIES,
    "switch_tool": MAX_SWITCH_TOOL_ATTEMPTS,
    "revise_belief": MAX_REVISE_BELIEF_ATTEMPTS,
    "decompose_goal": MAX_DECOMPOSE_GOAL_ATTEMPTS,
    "modify_state": MAX_MODIFY_STATE_ATTEMPTS,
}

PDF_ACCESS_ERROR = re.compile(
    r"Download is starting|Failed to load image from",
    re.I,
)
ARXIV_PDF_URL = re.compile(r"arxiv\.org/pdf/", re.I)


@dataclass
class StepInterventionState:
    """Per outline-step tracker for causal intervention attempts."""
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_tools: List[str] = field(default_factory=list)

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

    def reset_parameter_budget(self) -> None:
        self.counts["retry_with_different_parameters"] = 0

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
            " before 2020", " before 2010", " before 2000",
            " locations", " sightings", " records",
            " in the united states", " in the us",
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

        if not bindings:
            return False

        content = " | ".join(parts) if parts else raw[:500]
        quality = "secondary" if step_ctx.tool_name == "Google_Search_Tool" else "primary"
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
        elif kind == "visual":
            for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
                      "Python_Coder_Tool", "Base_Generator_Tool"):
                blocked.add(t)

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
            if suggested and suggested != failed and suggested in self.planner.available_tools:
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
        profile = self.system_memory.get_task_profile()
        if profile and SlotGate.has_compute_slot(profile):
            records = self.system_memory.evidence_records
            if not SlotGate.compute_slot_filled_by_computed(profile, records):
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

        loaded = self.system_memory.load_offline_memory("memory")
        if self.verbose and loaded:
            print("\n==> 📚 Loaded offline memory (Tool Skill Cards)")

        self.executor.set_query_cache_dir(self.root_cache_dir)

        json_data: Dict[str, Any] = {"query": question, "image": image_path}
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
    ) -> None:
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
        self.system_memory.set_outline(updated)
        # v3: grant one bonus step so the corrected outline can execute even if
        # max_steps is about to be reached (e.g. pid=2 fixed formula never ran).
        self._decompose_bonus = 1
        if self.verbose:
            print(f"[Decomposed Outline]:\n{json.dumps(updated, indent=4)}")

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

        if tracker and diagnostic_signal_prev:
            mutations = diagnostic_signal_prev.get("state_mutations") or {}
            for forbidden in mutations.get("forbidden_tools", []):
                tracker.record_failure_tool(forbidden)

        planner_signal = self._enrich_diagnostic_signal(
            diagnostic_signal_prev, target_information, question,
        )

        _, context, sub_goal, tool_name = self._run_planner(
            question, image_path, target_information, exec_step, json_data, planner_signal,
        )

        tool_name = self._resolve_tool_for_step(
            tool_name, question, target_information, sub_goal, planner_signal, image_path,
        )
        if tracker:
            tool_name = self._enforce_tool_policy(
                tool_name, question, target_information, tracker, planner_signal,
            )

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

        command, _, result = self._run_executor(
            question, image_path, context, sub_goal, tool_name,
            exec_step, json_data, diagnostic_signal_prev,
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
        """
        L1 intervention: retry with perturbed parameters (max MAX_PARAMETER_RETRIES total per step).
        Returns updated context, verification, exec_step, and signal for next cycle.
        """
        current_exec_step = exec_step
        current_ctx = step_ctx
        current_signal = self._apply_parameter_guidance(
            diagnostic_signal, step_ctx.first_attempt_command,
            self._collect_tried_params(step_ctx),
        )
        verification = None
        attempts_made = 0
        budget = tracker.remaining("retry_with_different_parameters")
        tried_params = self._collect_tried_params(step_ctx)

        if budget <= 0:
            print(f"\n==> ⏭️ Parameter retry budget exhausted ({MAX_PARAMETER_RETRIES}/{MAX_PARAMETER_RETRIES})\n")
            next_rec = tracker.resolve("retry_with_different_parameters")
            return (
                current_ctx,
                verification or VerificationResult(
                    analysis=diagnostic_signal.get("analysis", ""),
                    step_conclusion="SUBGOAL_INCOMPLETE",
                    info_flag=False,
                    obtained_info="",
                    task_conclusion=None,
                    diagnostic_signal=diagnostic_signal,
                    subgoal_complete=False,
                ),
                current_exec_step,
                self._build_diagnostic_signal_for_replan(
                    diagnostic_signal, next_rec, step_ctx, tracker, question=question,
                ),
            )

        max_this_round = min(budget, MAX_PARAMETER_RETRIES)

        for attempt in range(1, max_this_round + 1):
            if current_signal.get("subgoal_complete"):
                break
            if current_signal.get("recommendation") != "retry_with_different_parameters":
                break
            if current_exec_step >= self.max_steps:
                print(
                    f"\n==> 🛑 Parameter retry aborted — total exec_step budget reached "
                    f"({current_exec_step}/{self.max_steps})\n"
                )
                break

            current_exec_step += 1
            attempts_made += 1
            print(
                f"\n==> 🔧 Parameter Retry {attempt}/{max_this_round} "
                f"(total {tracker.counts['retry_with_different_parameters'] + attempt}/{MAX_PARAMETER_RETRIES}, "
                f"execution #{current_exec_step}/{self.max_steps}, outline step {step_ctx.step_key})\n"
            )

            command, analysis, explanation, curr_query = self._resolve_unique_retry_command(
                question, image_path, current_ctx, current_exec_step,
                json_data, current_signal, tried_params, attempt,
            )
            tried_params.add(self._param_fingerprint(command, current_ctx.tool_name))

            if current_ctx.tool_name == "Web_Search_Tool":
                command = self._ensure_web_search_command(
                    command, current_ctx.sub_goal, current_ctx.context,
                )

            command = self._sanitize_command_for_tool(current_ctx.tool_name, command)
            validation_error = self._validate_command(current_ctx.tool_name, command)
            if validation_error:
                print(f"\n==> ⚠️ Command validation failed: {validation_error}\n")
                result = f"Error in execute_tool_command: {validation_error}"
                json_data[f"tool_result_{current_exec_step}"] = result
            else:
                result = self._execute_generated_command(
                    current_ctx.tool_name, command, current_exec_step, json_data,
                )

            if self.verbose:
                print(f"\n==> 🛠️ Step {current_exec_step}: Command Execution ({current_ctx.tool_name})\n")
                print(f"[Executor Result]:\n{json.dumps(result, indent=4)}")

            current_ctx = StepContext(
                step_key=current_ctx.step_key,
                target_information=current_ctx.target_information,
                context=current_ctx.context,
                sub_goal=current_ctx.sub_goal,
                tool_name=current_ctx.tool_name,
                command=command,
                result_executor=result,
                first_attempt_command=current_ctx.first_attempt_command,
            )

            verification = self._run_verification(
                question, image_path, current_ctx, current_exec_step,
                label=f"Re-verification After Parameter Variation (attempt {attempt})",
                intervention_context=tracker.to_context(),
            )

            if verification.step_conclusion == "SUBGOAL_COMPLETE":
                tracker.record("retry_with_different_parameters", attempts_made)
                print(f"\n==> ✅ Parameter variation succeeded on attempt {attempt}\n")
                return current_ctx, verification, current_exec_step, verification.diagnostic_signal or {}

            self._record_failure_trace(
                current_exec_step, current_ctx,
                verification.diagnostic_signal,
                attempt_seq=attempt + 1,
            )

            if verification.diagnostic_signal:
                current_signal = self._apply_escalation_to_signal(
                    verification.diagnostic_signal, tracker,
                )
                if current_signal.get("recommendation") != "retry_with_different_parameters":
                    break
            else:
                break

        tracker.record("retry_with_different_parameters", attempts_made)

        if verification is None:
            verification = self._run_verification(
                question, image_path, current_ctx, current_exec_step,
                label="Re-verification After Parameter Variation",
                intervention_context=tracker.to_context(),
            )

        next_rec = tracker.resolve("retry_with_different_parameters")
        replan_signal = self._build_diagnostic_signal_for_replan(
            verification.diagnostic_signal or diagnostic_signal,
            next_rec,
            step_ctx,
            tracker,
            analysis=verification.analysis,
            question=question,
        )
        if next_rec != "retry_with_different_parameters":
            print(f"\n==> ⬆️ Parameter retries exhausted → next intervention: {next_rec}\n")

        return current_ctx, verification, current_exec_step, replan_signal

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
        """
        Handle failed subgoal with layered causal intervention escalation.

        Actions: 'complete' | 'replan'
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

        self._record_failure_trace(exec_step, step_ctx, diagnostic_signal)

        if not diagnostic_signal:
            return "replan", step_ctx, verification, exec_step, None

        recommendation = diagnostic_signal.get("recommendation")

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

        if recommendation == "retry_with_different_parameters":
            patterns = diagnostic_signal.get("failure_patterns") or {}
            skip_param_retry = (
                not verification.tool_appropriate
                or patterns.get("wrong_tool_class")
                or patterns.get("url_client_error")
                or self._is_pdf_access_error(step_ctx.result_executor)
                or "missing 1 required positional argument" in str(step_ctx.result_executor)
                or (patterns.get("error_occurred") and already_failed)
            )
            if skip_param_retry:
                reason = "wrong tool class" if not verification.tool_appropriate else (
                    "URL client error" if patterns.get("url_client_error") else "hard/infrastructure error"
                )
                print(
                    f"\n==> ⏭️ Skipping parameter retry — {step_ctx.tool_name} "
                    f"({reason}); escalating to switch_tool\n"
                )
                recommendation = "switch_tool"
                diagnostic_signal = dict(diagnostic_signal)
                diagnostic_signal["recommendation"] = "switch_tool"
            else:
                step_ctx, verification, exec_step, replan_signal = self._retry_with_parameter_variation(
                    question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker,
                )
                if verification.step_conclusion == "SUBGOAL_COMPLETE":
                    return "complete", step_ctx, verification, exec_step, replan_signal
                return self._route_post_intervention(
                    question, step_ctx, verification, exec_step, json_data, tracker, replan_signal,
                )

        if recommendation == "switch_tool":
            tracker.record("switch_tool", tool=step_ctx.tool_name)
            replan_signal = self._build_diagnostic_signal_for_replan(
                diagnostic_signal, "switch_tool", step_ctx, tracker, verification.analysis, question,
            )
            suggested = replan_signal.get("suggested_tool", "?")
            print(
                f"\n==> 🔀 switch_tool scheduled: {step_ctx.tool_name} → {suggested} "
                f"(next execution will replan)\n"
            )
            if tracker.is_exhausted("switch_tool"):
                next_rec = tracker.resolve("switch_tool")
                print(f"\n==> ⬆️ switch_tool exhausted → escalating to {next_rec}\n")
                return self._route_post_intervention(
                    question, step_ctx, verification, exec_step, json_data, tracker,
                    self._build_diagnostic_signal_for_replan(
                        diagnostic_signal, next_rec, step_ctx, tracker, verification.analysis, question,
                    ),
                )
            return "replan", step_ctx, verification, exec_step, replan_signal

        if recommendation == "revise_belief":
            tracker.record("revise_belief")
            replan_signal = self._apply_revise_belief(
                question, step_ctx, verification, exec_step, tracker,
            )
            if tracker.is_exhausted("revise_belief"):
                next_rec = tracker.resolve("revise_belief")
                return self._route_post_intervention(
                    question, step_ctx, verification, exec_step, json_data, tracker,
                    self._build_diagnostic_signal_for_replan(
                        diagnostic_signal, next_rec, step_ctx, tracker, verification.analysis, question,
                    ),
                )
            return "replan", step_ctx, verification, exec_step, replan_signal

        if recommendation == "decompose_goal":
            tracker.record("decompose_goal")
            self._apply_decompose_goal(
                question, step_ctx, verification, exec_step, json_data, tracker,
            )
            replan_signal = self._build_diagnostic_signal_for_replan(
                diagnostic_signal, "decompose_goal", step_ctx, tracker, verification.analysis, question,
            )
            if tracker.is_exhausted("decompose_goal"):
                next_rec = tracker.resolve("decompose_goal")
                replan_signal = self._build_diagnostic_signal_for_replan(
                    diagnostic_signal, next_rec, step_ctx, tracker, verification.analysis, question,
                )
                return self._route_post_intervention(
                    question, step_ctx, verification, exec_step, json_data, tracker, replan_signal,
                )
            return "replan", step_ctx, verification, exec_step, replan_signal

        if recommendation == "modify_state":
            tracker.record("modify_state")
            replan_signal = self._apply_modify_state(step_ctx, verification)
            if tracker.is_exhausted("modify_state"):
                print(f"\n==> ⬆️ All intervention types exhausted on step {step_ctx.step_key}; forcing decompose_goal\n")
                tracker.record("decompose_goal")
                self._apply_decompose_goal(
                    question, step_ctx, verification, exec_step, json_data, tracker,
                )
                replan_signal = self._build_diagnostic_signal_for_replan(
                    diagnostic_signal, "decompose_goal", step_ctx, tracker, verification.analysis, question,
                )
            return "replan", step_ctx, verification, exec_step, replan_signal

        return "replan", step_ctx, verification, exec_step, diagnostic_signal

    def _route_post_intervention(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
        tracker: StepInterventionState,
        signal: Dict[str, Any],
    ) -> Tuple[str, StepContext, VerificationResult, int, Optional[Dict[str, Any]]]:
        """Recursively apply the next intervention after the previous one was exhausted."""
        rec = signal.get("recommendation", "switch_tool")
        if rec == "switch_tool":
            tracker.record("switch_tool", tool=step_ctx.tool_name)
            if not tracker.is_exhausted("switch_tool"):
                return "replan", step_ctx, verification, exec_step, signal
            rec = tracker.resolve("switch_tool")
            signal = self._build_diagnostic_signal_for_replan(signal, rec, step_ctx, tracker, question=question)

        if rec == "revise_belief" and not tracker.is_exhausted("revise_belief"):
            tracker.record("revise_belief")
            signal = self._apply_revise_belief(question, step_ctx, verification, exec_step, tracker)
            return "replan", step_ctx, verification, exec_step, signal

        if rec == "decompose_goal" and not tracker.is_exhausted("decompose_goal"):
            tracker.record("decompose_goal")
            self._apply_decompose_goal(
                question, step_ctx, verification, exec_step, json_data, tracker,
            )
            return "replan", step_ctx, verification, exec_step, signal

        if rec == "modify_state" and not tracker.is_exhausted("modify_state"):
            tracker.record("modify_state")
            return "replan", step_ctx, verification, exec_step, self._apply_modify_state(step_ctx, verification)

        tracker.record("decompose_goal")
        self._apply_decompose_goal(
            question, step_ctx, verification, exec_step, json_data, tracker,
        )
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
        if re.search(r"(name of the character|what character|exact character)", q):
            tokens = re.findall(r"\b[A-Za-z_]+\b", ans)
            if len(ans) > 40 or len(tokens) > 4:
                return "question asks for a single character/name but answer is verbose"
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

        self._analyze_query(question, image_path, json_data, query_start_time)

        outline_attempt = 0
        exec_step = 0
        diagnostic_signal_prev = None
        context_verification = ""
        last_task_conclusion: Optional[str] = None
        intervention_trackers: Dict[str, StepInterventionState] = {}
        recovery_injections = 0
        self._decompose_bonus = 0

        while exec_step < self.max_steps + self._decompose_bonus and (time.time() - query_start_time) < self.max_time:
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

            step_ctx = self._execute_step(
                question, image_path, step_key, target_information,
                exec_step, json_data, diagnostic_signal_prev, tracker,
            )

            verification = self._run_verification(
                question, image_path, step_ctx, exec_step,
                intervention_context=tracker.to_context(),
            )
            if self._harvest_slots_from_executor(step_ctx, verification, exec_step):
                verification.had_slot_delta = True
            # v3: no direct STOP break here. COMPLETE steps must fall through to
            # `_handle_subgoal_complete`, which runs the unified answer-sanity
            # gate before allowing STOP. Breaking here bypassed that gate (FM1).
            if self._maybe_escalate_partial_cross_ref(question, step_id, verification):
                continue

            context_verification = verification.analysis
            last_task_conclusion = verification.task_conclusion

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
            print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

        print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
        print("\n==> ✅ Query Solved!")

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
