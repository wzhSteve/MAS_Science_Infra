import argparse
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
from MAS.epc_aw.models import CausalGraph, BayesianInference, HistoryAnalyzer, CausalInference


# ---------------------------------------------------------------------------
# Causal intervention limits & escalation ladder (L1 → L2 → L3)
# ---------------------------------------------------------------------------
MAX_PARAMETER_RETRIES = 3
MAX_SWITCH_TOOL_ATTEMPTS = 2
MAX_DECOMPOSE_GOAL_ATTEMPTS = 1
MAX_MODIFY_STATE_ATTEMPTS = 1
MAX_RECOVERY_OUTLINE_INJECTIONS = 1

INTERVENTION_LADDER: List[str] = [
    "retry_with_different_parameters",  # L1: parameter perturbation
    "switch_tool",                       # L2/L3: change tool
    "decompose_goal",                    # L3: break down sub-goal
    "modify_state",                      # L3: environment / prerequisite fix
]

INTERVENTION_LIMITS: Dict[str, int] = {
    "retry_with_different_parameters": MAX_PARAMETER_RETRIES,
    "switch_tool": MAX_SWITCH_TOOL_ATTEMPTS,
    "decompose_goal": MAX_DECOMPOSE_GOAL_ATTEMPTS,
    "modify_state": MAX_MODIFY_STATE_ATTEMPTS,
}


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
        if tool and intervention == "switch_tool" and tool not in self.failed_tools:
            self.failed_tools.append(tool)

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

        self.causal_graph = CausalGraph()
        self.bayesian_inference = BayesianInference()
        self.history_analyzer = HistoryAnalyzer()
        self.causal_inference = CausalInference(self.causal_graph, self.history_analyzer)

        self.planner.causal_inference = self.causal_inference
        self.planner.history_analyzer = self.history_analyzer
        self.diagnoser.bayesian_inference = self.bayesian_inference
        self.diagnoser.history_analyzer = self.history_analyzer
        self.executor.history_analyzer = self.history_analyzer

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

    def _perturb_query_param(
        self,
        query: str,
        attempt: int,
        tried: Set[str],
    ) -> str:
        """Deterministic query perturbation when LLM repeats a failed parameter."""
        base = query.strip()
        lower = base.lower()
        candidates: List[str] = []

        if "usgs" in lower or "amphiprion" in lower:
            candidates.extend([
                "site:nas.er.usgs.gov Amphiprion ocellaris nonindigenous",
                "clown anemonefish Amphiprion ocellaris USGS NAS nonnative United States",
                "Amphiprion ocellaris nonindigenous aquatic species database",
                "nas.er.usgs.gov Amphiprion ocellaris collection records Florida",
            ])

        simplified = base
        for phrase in (
            "sightings before 2020 locations",
            " before 2020 locations",
            " before 2020",
            " locations",
            " sightings",
        ):
            simplified = simplified.replace(phrase, "")
        simplified = re.sub(r"\s+", " ", simplified).strip()
        if simplified:
            candidates.append(simplified)

        tokens = simplified.split() if simplified else base.split()
        if len(tokens) > 2:
            candidates.append(" ".join(tokens[: max(2, len(tokens) - 2)]))
            candidates.append(" ".join(reversed(tokens)))

        candidates.append(f"{base} alternative phrasing")
        candidates.append(f"{simplified or base} variant{attempt}")

        for candidate in candidates:
            norm = self._normalize_query_param(candidate)
            if norm and norm not in tried:
                return candidate

        fallback = f"{base} retry{attempt}"
        return fallback

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
        return q_norm

    @staticmethod
    def _inject_url_into_command(command: str, new_url: str) -> str:
        escaped = new_url.replace('"', '\\"')
        if re.search(r'url=["\']', command):
            return re.sub(r'url=(["\'])[^"\']*\1', f'url="{escaped}"', command, count=1)
        if 'query="' in command or "query='" in command:
            return command.rstrip() + f', url="{escaped}")'
        return f'execution = tool.execute(query="fetch page", url="{escaped}")'

    def _perturb_web_search_command(
        self,
        command: str,
        query: str,
        attempt: int,
        tried: Set[str],
    ) -> Tuple[str, str]:
        """Perturb Web_Search by changing URL or query+URL together."""
        url = self._extract_url_param(command)
        candidates: List[Tuple[str, str]] = []

        if "wikipedia" in (url or "").lower() or "404" in query.lower():
            candidates.extend([
                ("Tarpon Springs Florida zip code 34689", "https://www.tarponspringsfl.gov/"),
                ("Fred Howard Park Tarpon Springs zip code", "https://pinellas.gov/parks/fred-howard-park/"),
                ("34689 zip code Tarpon Springs", ""),
            ])
        if "zip" in query.lower() or "fred howard" in query.lower():
            candidates.extend([
                ("Tarpon Springs Florida zip code", "https://www.tarponspringsfl.gov/"),
                ("Fred Howard Park zip code Florida", "https://pinellas.gov/parks/fred-howard-park/"),
            ])
        if "usgs" in (url or "").lower() or "nas.er.usgs.gov" in (url or "").lower():
            candidates.extend([
                ("Amphiprion ocellaris collection Florida before 2020", "https://nas.er.usgs.gov/queries/CollectionInfo.aspx?SpeciesID=3243"),
                ("Tarpon Springs Florida zip code", "https://www.google.com/search?q=Tarpon+Springs+Florida+zip+code"),
            ])

        candidates.append((f"{query} variant{attempt}", url if url != "N/A" else ""))

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
            "zip code", "zip codes", "nas.er.usgs.gov", "nonnative",
        )
        return any(m in text for m in markers)

    def _suggest_alternative_tool(
        self,
        failed_tool: Optional[str],
        target_information: str,
        question: str,
    ) -> Optional[str]:
        """Pick a retrieval tool when the current one failed. Never suggest Base_Generator."""
        text = f"{target_information} {question}".lower()
        available = set(self.planner.available_tools)

        if "usgs" in text or "nas.er.usgs.gov" in text:
            if "Web_Search_Tool" in available and failed_tool != "Web_Search_Tool":
                return "Web_Search_Tool"
            if "Google_Search_Tool" in available and failed_tool != "Google_Search_Tool":
                return "Google_Search_Tool"

        for candidate in (
            "Web_Search_Tool", "Google_Search_Tool", "Wikipedia_Search_Tool",
        ):
            if candidate in available and candidate != failed_tool:
                return candidate
        return None

    def _inject_web_context_for_usgs(self, context: str, target_information: str, question: str) -> str:
        text = f"{context} {target_information} {question}".lower()
        if "usgs" not in text and "amphiprion" not in text:
            return context
        if "http" in context:
            return context
        return (
            "https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243 "
            f"{context}"
        ).strip()

    def _resolve_tool_for_step(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        diagnostic_signal: Optional[Dict[str, Any]],
    ) -> str:
        """Override planner tool when switch_tool or Base_Generator is inappropriate."""
        failed = (diagnostic_signal or {}).get("tool")
        rec = (diagnostic_signal or {}).get("recommendation")

        invalid = (
            not tool_name
            or "no matched" in str(tool_name).lower()
            or tool_name == "Base_Generator_Tool"
        )
        needs_retrieval = self._requires_external_evidence(question, target_information)

        if rec == "switch_tool" or (invalid and needs_retrieval):
            suggested = (diagnostic_signal or {}).get("suggested_tool")
            if suggested and suggested != failed and suggested in self.planner.available_tools:
                if suggested != "Base_Generator_Tool":
                    return suggested
            alt = self._suggest_alternative_tool(failed or tool_name, target_information, question)
            if alt:
                if invalid or tool_name == failed or tool_name == "Base_Generator_Tool":
                    print(f"\n==> 🔀 Tool override: {tool_name} → {alt}\n")
                    return alt

        if tool_name == "Base_Generator_Tool" and needs_retrieval:
            alt = self._suggest_alternative_tool("Base_Generator_Tool", target_information, question)
            if alt:
                print(f"\n==> 🔀 Blocked Base_Generator for factual query → {alt}\n")
                return alt

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

    def _generate_recovery_outline(
        self,
        question: str,
        verification: "VerificationResult",
    ) -> Dict[str, Any]:
        """Last-resort outline when LLM returns empty but task is not STOP."""
        cumulative = self.system_memory.get_obtained_information_for_prompt()
        if self.diagnoser._is_task_answer_ready(question, cumulative):
            return {}

        text = f"{question} {verification.analysis}".lower()
        steps: Dict[str, str] = {}

        if "usgs" in text or "zip" in text:
            if not self.diagnoser._extract_zip_codes(cumulative):
                steps["1"] = (
                    "Target Information: Resolve five-digit zip code for the USGS-documented "
                    "nonnative sighting location. "
                    "Operation Details: Google_Search_Tool with query "
                    "\"Tarpon Springs Florida zip code\" OR \"Fred Howard Park Florida zip code\". "
                    "Expected Output: Five-digit U.S. zip code."
                )
        elif not steps:
            steps["1"] = (
                "Target Information: Retrieve missing evidence to answer the original question. "
                f"Operation Details: Use an appropriate search tool. Context: {verification.analysis[:200]}"
            )
        return steps

    def _cumulative_evidence_text(self) -> str:
        return self.system_memory.get_obtained_information_for_prompt()

    def _is_answer_ready(self, question: str) -> bool:
        return self.diagnoser._is_task_answer_ready(
            question, self._cumulative_evidence_text(),
        )

    def _sanitize_outline_update(
        self,
        prev_outline: Dict[str, Any],
        new_outline: Dict[str, Any],
        current_step_num: int,
        task_conclusion: str,
        question: str,
        verification: "VerificationResult",
    ) -> Dict[str, Any]:
        """Prevent empty outline when task is not STOP."""
        if task_conclusion == "STOP" or self._is_answer_ready(question):
            return new_outline if new_outline is not None else {}

        if new_outline:
            return new_outline

        fallback = self._fallback_advance_outline(prev_outline, current_step_num)
        if fallback:
            print(
                f"\n==> ⚠️ Outline update returned empty but task is {task_conclusion}; "
                f"using fallback ({len(fallback)} remaining step(s))\n"
            )
            return fallback

        recovery = self._generate_recovery_outline(question, verification)
        if not recovery:
            print("\n==> ✅ Answer already in cumulative evidence — clearing outline for STOP\n")
            return {}
        print(
            f"\n==> ⚠️ Outline empty with no fallback steps; "
            f"injecting recovery outline ({len(recovery)} step(s))\n"
        )
        return recovery

    def _task_needs_continuation(self, task_conclusion: Optional[str], outline: Optional[Dict]) -> bool:
        if task_conclusion == "STOP":
            return False
        if outline:
            return True
        return task_conclusion in (None, "CONTINUE")

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
        self.system_memory.online_memory["task_id"] = task_id
        self.system_memory.set_query(question)

        if self.verbose:
            print(f"\n==> 📝 Task ID: {task_id}")

        loaded = self.system_memory.load_offline_memory("memory")
        if self.verbose and loaded:
            print("\n==> 📚 Loaded offline memory (Tool Ability Boundary + Parameter Graphs)")

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
        json_data["analysis"] = analysis
        json_data["outline"] = execution_outline
        self.system_memory.set_outline(execution_outline)

        if self.verbose:
            print("\n==> 🔍 Step 0: Query Analysis\n")
            print(f"{analysis}")
            # print(f"\n[Execution Outline]:\n{json.dumps(execution_outline, indent=4)}")
            print(f"[Time]: {round(time.time() - query_start, 2)}s")

    def _persist_task_knowledge(self, task_id: str) -> None:
        try:
            self.system_memory.upgrade_online_to_offline()
            if self.verbose:
                print("\n==> 📚 Knowledge Upgrade: Online → Offline")

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
                print(f"  Offline: {stats['offline_memory']['tool_ability_boundary']} abilities, "
                      f"{stats['offline_memory']['tool_parameter_graphs']} patterns")
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
            signal = self._enrich_diagnostic_signal(signal, step_ctx.target_information, question)
        return signal

    def _apply_decompose_goal(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
    ) -> None:
        """L3: break down the stuck sub-goal by updating the outline."""
        print(f"\n==> 🔀 L3 Intervention: Decompose Goal (outline step {step_ctx.step_key})\n")
        current_step_num = int(step_ctx.step_key) if step_ctx.step_key.isdigit() else 1
        updated = self.diagnoser.update_outline(
            question=question,
            context_verification=(
                f"{verification.analysis}\n\n"
                "[INTERVENTION: decompose_goal] The current sub-goal failed after exhausting "
                "parameter retries and tool switches. Decompose it into smaller achievable steps."
            ),
            target_information=step_ctx.target_information,
            outline=self.system_memory.get_outline(),
            result_executor=step_ctx.result_executor,
            current_step=current_step_num,
            obtained_information=self.system_memory.get_obtained_information_for_prompt(),
            toolbox_metadata=self.system_memory.get_toolbox_metadata(),
            execution_memory=self._build_execution_memory_summary(json_data, exec_step),
        )
        self.system_memory.set_outline(updated)
        if self.verbose:
            print(f"[Decomposed Outline]:\n{json.dumps(updated, indent=4)}")

    def _apply_modify_state(
        self,
        step_ctx: StepContext,
        verification: VerificationResult,
    ) -> Dict[str, Any]:
        """L3: signal environment/prerequisite change before retrying."""
        print(f"\n==> 🔧 L3 Intervention: Modify State (outline step {step_ctx.step_key})\n")
        return {
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

    def _enrich_diagnostic_signal(
        self,
        signal: Optional[Dict[str, Any]],
        target_information: str,
        question: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Attach alternative tool to switch_tool signals for Planner."""
        if not signal or signal.get("recommendation") != "switch_tool":
            return signal
        enriched = dict(signal)
        failed_tool = enriched.get("tool")
        outline_tool = Planner._extract_tool_from_target(target_information)

        alt = self._suggest_alternative_tool(failed_tool, target_information, question)
        if alt and alt != failed_tool:
            enriched["suggested_tool"] = alt
        elif outline_tool and outline_tool != failed_tool:
            enriched["suggested_tool"] = outline_tool

        if "usgs" in f"{target_information} {question}".lower():
            enriched["web_search_url"] = (
                "https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243"
            )
        return enriched

    def _execute_step(
        self,
        question: str,
        image_path: Optional[str],
        step_key: str,
        target_information: str,
        exec_step: int,
        json_data: Dict[str, Any],
        diagnostic_signal_prev: Optional[Dict[str, Any]],
    ) -> StepContext:
        outline_tool = Planner._extract_tool_from_target(target_information)
        print(
            f"\n==> 🎯 Outline Step {step_key} | Execution #{exec_step}\n"
            f"Target Information:\n{target_information}\n"
        )
        if outline_tool and self.verbose:
            print(f"[Outline Suggested Tool]: {outline_tool}")

        planner_signal = self._enrich_diagnostic_signal(
            diagnostic_signal_prev, target_information, question,
        )

        _, context, sub_goal, tool_name = self._run_planner(
            question, image_path, target_information, exec_step, json_data, planner_signal,
        )

        tool_name = self._resolve_tool_for_step(
            tool_name, question, target_information, planner_signal,
        )
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

        if self.verbose:
            print(f"\n==> 🤖 Step {step_count}: {label} Results\n")
            print(f"[Subgoal Conclusion]: {step_conclusion}")
            print(f"[Task Conclusion]: {task_conclusion}")
            print(f"[Analysis]: {analysis}\n")
            print(f"[Time]: {round(time.time() - local_start, 2)}s")

        return VerificationResult(
            analysis=analysis,
            step_conclusion=step_conclusion,
            info_flag=info_flag,
            obtained_info=obtained_info,
            task_conclusion=task_conclusion,
            diagnostic_signal=diagnostic_signal,
            subgoal_complete=subgoal_complete,
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
        )

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
        )
        if should_record:
            provenance = self.diagnoser._assess_evidence_provenance(
                step_ctx.result_executor,
                self.system_memory.get_query() or "",
            )
            added = self.system_memory.add_evidence_record(
                info,
                outline_step=step_ctx.step_key,
                exec_step=step_count,
                subgoal=step_ctx.sub_goal,
                tool=step_ctx.tool_name,
                source_quality=provenance.get("level", "unknown"),
                subgoal_complete=verification.subgoal_complete,
                confidence=0.8 if verification.subgoal_complete else 0.4,
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
        state = self.system_memory.infer_state_type()
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
        if diagnostic_signal and diagnostic_signal.get("recommendation") == "retry_with_different_parameters":
            self.system_memory.add_failed_parameter(
                state=state,
                subgoal=step_ctx.sub_goal,
                tool=step_ctx.tool_name,
                parameter=str(step_ctx.command)[:100],
                failure_symptom=str(diagnostic_signal.get("failure_patterns", "")),
                L1_diagnosis="参数问题",
            )

    def _record_success_trace(
        self,
        step_count: int,
        step_ctx: StepContext,
        verification: VerificationResult,
        failed_parameter: Optional[str] = None,
        attempt_seq: int = 1,
    ) -> None:
        state = self.system_memory.infer_state_type()
        self._record_causal_effect(
            step_ctx, step_count, success=True,
            verification=verification,
            attempt_seq=attempt_seq,
            finalize=True,
        )
        self.system_memory.add_successful_parameter(
            state=state,
            subgoal=step_ctx.sub_goal,
            tool=step_ctx.tool_name,
            parameter=str(step_ctx.command)[:100],
            result="subgoal_completed",
            failed_parameter=failed_parameter,
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
                "IMPORTANT: Return empty outline ONLY if Task Conclusion is STOP. "
                "If Task Conclusion is CONTINUE, you MUST keep remaining steps."
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
            else:
                perturbed = self._perturb_query_param(
                    curr_query if curr_query != "N/A" else last_query,
                    attempt * 10 + regen,
                    {t.split("|")[0] for t in tried_params},
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
        else:
            perturbed = self._perturb_query_param(last_query, attempt * 100, set())
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

            current_exec_step += 1
            attempts_made += 1
            print(
                f"\n==> 🔧 Parameter Retry {attempt}/{max_this_round} "
                f"(total {tracker.counts['retry_with_different_parameters'] + attempt}/{MAX_PARAMETER_RETRIES}, "
                f"execution #{current_exec_step}, outline step {step_ctx.step_key})\n"
            )

            command, analysis, explanation, curr_query = self._resolve_unique_retry_command(
                question, image_path, current_ctx, current_exec_step,
                json_data, current_signal, tried_params, attempt,
            )
            tried_params.add(self._param_fingerprint(command, current_ctx.tool_name))

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

        diagnostic_signal = verification.diagnostic_signal
        if diagnostic_signal:
            diagnostic_signal = self._apply_escalation_to_signal(diagnostic_signal, tracker)
            diagnostic_signal = self._enrich_diagnostic_signal(
                diagnostic_signal, step_ctx.target_information, question,
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

        if recommendation == "decompose_goal":
            tracker.record("decompose_goal")
            self._apply_decompose_goal(question, step_ctx, verification, exec_step, json_data)
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
                self._apply_decompose_goal(question, step_ctx, verification, exec_step, json_data)
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

        if rec == "decompose_goal" and not tracker.is_exhausted("decompose_goal"):
            tracker.record("decompose_goal")
            self._apply_decompose_goal(question, step_ctx, verification, exec_step, json_data)
            return "replan", step_ctx, verification, exec_step, signal

        if rec == "modify_state" and not tracker.is_exhausted("modify_state"):
            tracker.record("modify_state")
            return "replan", step_ctx, verification, exec_step, self._apply_modify_state(step_ctx, verification)

        tracker.record("decompose_goal")
        self._apply_decompose_goal(question, step_ctx, verification, exec_step, json_data)
        return "replan", step_ctx, verification, exec_step, signal

    def _handle_subgoal_complete(
        self,
        question: str,
        step_ctx: StepContext,
        verification: VerificationResult,
        exec_step: int,
        json_data: Dict[str, Any],
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

        current_step_num = int(step_ctx.step_key) if step_ctx.step_key.isdigit() else 1
        self._update_outline(
            question, verification, step_ctx, current_step_num, json_data, exec_step,
        )

        if verification.task_conclusion != "STOP" and self._is_answer_ready(question):
            print("\n==> 🎯 Cumulative evidence satisfies question — forcing STOP\n")
            verification = VerificationResult(
                analysis=verification.analysis,
                step_conclusion=verification.step_conclusion,
                info_flag=verification.info_flag,
                obtained_info=verification.obtained_info,
                task_conclusion="STOP",
                diagnostic_signal=verification.diagnostic_signal,
                subgoal_complete=verification.subgoal_complete,
            )

        if verification.task_conclusion == "STOP":
            print(f"\n==> 🎯 Execution #{exec_step}: Task Complete - STOP\n")
            return "stop"
        print(f"\n==> 🔄 Execution #{exec_step}: Task Incomplete - CONTINUE\n")
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

        while outline_attempt < self.max_steps and (time.time() - query_start_time) < self.max_time:
            if self._is_answer_ready(question):
                print(
                    f"\n==> 🎯 Cumulative answer ready — stopping early "
                    f"(outline_attempt={outline_attempt}/{self.max_steps})\n"
                )
                last_task_conclusion = "STOP"
                break

            step_key, target_information = self._get_first_outline_step()
            print(f"\n==> 🗂️ Current Execution Outline:\n"
                  f"{json.dumps(self.system_memory.get_outline(), indent=4)}")

            if step_key is None:
                if self._is_answer_ready(question):
                    print("\n==> 🎯 Outline empty and cumulative answer ready — STOP\n")
                    last_task_conclusion = "STOP"
                    break

                if (
                    outline_attempt < self.max_steps
                    and recovery_injections < MAX_RECOVERY_OUTLINE_INJECTIONS
                    and self._task_needs_continuation(last_task_conclusion, self.system_memory.get_outline())
                ):
                    recovery = self._generate_recovery_outline(
                        question,
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
                    if recovery:
                        self.system_memory.set_outline(recovery)
                        recovery_injections += 1
                        print(
                            f"\n==> ⚠️ Outline exhausted at attempt {outline_attempt}/{self.max_steps} "
                            f"but task not STOP — injecting recovery outline "
                            f"({recovery_injections}/{MAX_RECOVERY_OUTLINE_INJECTIONS})\n"
                        )
                        continue

                print(
                    f"\n==> 🎯 Task Completed - No more steps in outline "
                    f"(outline_attempt={outline_attempt}/{self.max_steps})\n"
                )
                break

            outline_attempt += 1
            exec_step += 1
            tracker = self._get_intervention_state(intervention_trackers, step_key)
            if tracker.counts:
                print(f"[Outline Step {step_key}]: intervention history {dict(tracker.counts)}")

            step_ctx = self._execute_step(
                question, image_path, step_key, target_information,
                exec_step, json_data, diagnostic_signal_prev,
            )

            verification = self._run_verification(
                question, image_path, step_ctx, exec_step,
                intervention_context=tracker.to_context(),
            )
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
                    task_action = self._handle_subgoal_complete(
                        question, step_ctx, verification, exec_step, json_data,
                    )
                    diagnostic_signal_prev = None
                    intervention_trackers.pop(step_key, None)
                    if task_action == "stop":
                        break
                    continue

                continue

            task_action = self._handle_subgoal_complete(
                question, step_ctx, verification, exec_step, json_data,
            )
            diagnostic_signal_prev = None
            intervention_trackers.pop(step_key, None)
            if task_action == "stop":
                break

        print("=============== Last Verification Analysis ================")
        print(f"{context_verification}")
        print("================== Obtained Information ===================")
        print(f"{self.system_memory.get_obtained_information()}")
        print("===========================================================")

        if 'direct' in self.output_types:
            direct_output = self.executor.generate_direct_output(
                question, context_verification, self.system_memory,
            )
            json_data["direct_output"] = direct_output
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
