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
            self.system_memory.get_obtained_information(),
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

        result = self.executor.execute_tool_command(tool_name, command)
        result = make_json_serializable_truncated(result)
        json_data[f"tool_result_{step_count}"] = result

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
            signal = self._enrich_diagnostic_signal(signal, step_ctx.target_information)
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
            obtained_information=self.system_memory.get_obtained_information(),
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
    ) -> Optional[Dict[str, Any]]:
        """Attach outline-suggested tool to switch_tool signals for Planner."""
        if not signal or signal.get("recommendation") != "switch_tool":
            return signal
        enriched = dict(signal)
        suggested = Planner._extract_tool_from_target(target_information)
        failed_tool = enriched.get("tool")
        if suggested and suggested != failed_tool:
            enriched["suggested_tool"] = suggested
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

        planner_signal = self._enrich_diagnostic_signal(diagnostic_signal_prev, target_information)

        _, context, sub_goal, tool_name = self._run_planner(
            question, image_path, target_information, exec_step, json_data, planner_signal,
        )

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
            self.system_memory.get_obtained_information(),
            step_ctx.sub_goal,
            step_ctx.tool_name,
            intervention_context=intervention_context,
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
            print(f"\n==> ℹ️ Step {step_count}: New relevant information added to system memory.\n")
            self.system_memory.add_obtained_information(info)
            print(f"\n[New Obtained Information]:\n{info}\n")
        else:
            print(f"\n==> ℹ️ Step {step_count}: No new relevant information obtained.\n")

    def _record_failure_trace(
        self,
        step_count: int,
        step_ctx: StepContext,
        diagnostic_signal: Optional[Dict[str, Any]],
    ) -> None:
        state_type = "无完整信息"
        effects = [{
            "attempt_seq": 1,
            "tool": step_ctx.tool_name,
            "parameter": str(step_ctx.command)[:100],
            "result": {"success": False},
            "symptom": diagnostic_signal.get("failure_patterns", "") if diagnostic_signal else "Unknown",
            "L1_diagnosis": "参数问题",
            "success": False,
        }]
        self.system_memory.record_causal_trace(
            step=step_count,
            state=state_type,
            subgoal=step_ctx.sub_goal,
            effects=effects,
            L2_diagnosis=diagnostic_signal.get("root_cause_analysis", {}) if diagnostic_signal else {},
        )
        if diagnostic_signal and diagnostic_signal.get("recommendation") == "retry_with_different_parameters":
            self.system_memory.add_failed_parameter(
                state=state_type,
                subgoal=step_ctx.sub_goal,
                tool=step_ctx.tool_name,
                parameter=str(step_ctx.command)[:100],
                failure_symptom=diagnostic_signal.get("failure_patterns", ""),
                L1_diagnosis="参数问题",
            )

    def _record_success_trace(self, step_count: int, step_ctx: StepContext) -> None:
        state_type = "无完整信息"
        effects = [{
            "attempt_seq": 1,
            "tool": step_ctx.tool_name,
            "parameter": str(step_ctx.command)[:100],
            "result": {"success": True},
            "symptom": None,
            "L1_diagnosis": "成功",
            "success": True,
        }]
        self.system_memory.record_causal_trace(
            step=step_count,
            state=state_type,
            subgoal=step_ctx.sub_goal,
            effects=effects,
            L2_diagnosis={"root_cause": "success"},
        )
        self.system_memory.add_successful_parameter(
            state=state_type,
            subgoal=step_ctx.sub_goal,
            tool=step_ctx.tool_name,
            parameter=str(step_ctx.command)[:100],
            result="subgoal_completed",
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
        updated = self.diagnoser.update_outline(
            question=question,
            context_verification=verification.analysis,
            target_information=step_ctx.target_information,
            outline=self.system_memory.get_outline(),
            result_executor=step_ctx.result_executor,
            current_step=current_step_num,
            obtained_information=self.system_memory.get_obtained_information(),
            toolbox_metadata=self.system_memory.get_toolbox_metadata(),
            execution_memory=self._build_execution_memory_summary(json_data, step_count),
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
    ) -> Dict[str, Any]:
        signal = dict(diagnostic_signal)
        failed_param = self._extract_query_param(failed_command)
        if failed_param != "N/A":
            guidance = signal.setdefault("parameter_guidance", {})
            guidance["failed_parameter"] = failed_param
            guidance["avoid_exact_match"] = f'Do NOT use query="{failed_param}"'
        return signal

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
        )
        verification = None
        attempts_made = 0
        budget = tracker.remaining("retry_with_different_parameters")

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
                    diagnostic_signal, next_rec, step_ctx, tracker,
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

            command, _, result = self._run_executor(
                question, image_path,
                current_ctx.context, current_ctx.sub_goal, current_ctx.tool_name,
                current_exec_step, json_data, current_signal,
            )

            prev_query = self._extract_query_param(current_ctx.first_attempt_command)
            curr_query = self._extract_query_param(command)
            print(f"\n[Parameter Comparison]: '{prev_query}' → '{curr_query}'")
            if prev_query == curr_query:
                print("  ⚠️ WARNING: Parameters are IDENTICAL!")

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
                diagnostic_signal, step_ctx.target_information,
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
                diagnostic_signal, "switch_tool", step_ctx, tracker, verification.analysis,
            )
            if tracker.is_exhausted("switch_tool"):
                next_rec = tracker.resolve("switch_tool")
                print(f"\n==> ⬆️ switch_tool exhausted → escalating to {next_rec}\n")
                return self._route_post_intervention(
                    question, step_ctx, verification, exec_step, json_data, tracker,
                    self._build_diagnostic_signal_for_replan(
                        diagnostic_signal, next_rec, step_ctx, tracker, verification.analysis,
                    ),
                )
            return "replan", step_ctx, verification, exec_step, replan_signal

        if recommendation == "decompose_goal":
            tracker.record("decompose_goal")
            self._apply_decompose_goal(question, step_ctx, verification, exec_step, json_data)
            replan_signal = self._build_diagnostic_signal_for_replan(
                diagnostic_signal, "decompose_goal", step_ctx, tracker, verification.analysis,
            )
            if tracker.is_exhausted("decompose_goal"):
                next_rec = tracker.resolve("decompose_goal")
                replan_signal = self._build_diagnostic_signal_for_replan(
                    diagnostic_signal, next_rec, step_ctx, tracker, verification.analysis,
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
                    diagnostic_signal, "decompose_goal", step_ctx, tracker, verification.analysis,
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
            signal = self._build_diagnostic_signal_for_replan(signal, rec, step_ctx, tracker)

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

        self._record_success_trace(exec_step, step_ctx)
        self._record_obtained_information(exec_step, verification, step_ctx)

        current_step_num = int(step_ctx.step_key) if step_ctx.step_key.isdigit() else 1
        self._update_outline(
            question, verification, step_ctx, current_step_num, json_data, exec_step,
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
        intervention_trackers: Dict[str, StepInterventionState] = {}

        while outline_attempt < self.max_steps and (time.time() - query_start_time) < self.max_time:
            step_key, target_information = self._get_first_outline_step()
            print(f"\n==> 🗂️ Current Execution Outline:\n"
                  f"{json.dumps(self.system_memory.get_outline(), indent=4)}")

            if step_key is None:
                print("\n==> 🎯 Task Completed - No more steps in outline")
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
