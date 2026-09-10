"""EPC_AW Solver — thin orchestrator (not a fourth LLM role).

Decision exits live in mixins:
  tool_policy / plan_controller / evidence_binder / intervention /
  command_ops / answer_gate. Causal ladder constants: models.intervention.
"""

import argparse
import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from MAS.epc_aw.models.initializer import Initializer
from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.executor import Executor
from MAS.epc_aw.models.utils import make_json_serializable_truncated
from MAS.epc_aw.models import BayesianInference, HistoryAnalyzer, CausalInference
from MAS.epc_aw.models.task_profile import SlotGate
from MAS.epc_aw.models.tool_router import ToolRouter, infer_subgoal_kind
from MAS.epc_aw.models.solver_types import VerificationResult, StepContext
from MAS.epc_aw.models.command_ops import CommandOpsMixin
from MAS.epc_aw.models.answer_gate import AnswerGateMixin
from MAS.epc_aw.models.tool_policy import ToolPolicyMixin
from MAS.epc_aw.models.plan_controller import PlanControllerMixin
from MAS.epc_aw.models.evidence_binder import EvidenceBinderMixin
from MAS.epc_aw.models.intervention import (
    InterventionMixin,
    StepInterventionState,
    MAX_RECOVERY_OUTLINE_INJECTIONS,
)
from MAS.epc_aw.models.ablation import AblationConfig, resolve_ablation


class Solver(CommandOpsMixin, AnswerGateMixin, InterventionMixin, ToolPolicyMixin, PlanControllerMixin, EvidenceBinderMixin):
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
        ablation: Optional[Any] = None,
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
        self.ablation: AblationConfig = resolve_ablation(ablation)
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
        self.planner.ablation = self.ablation
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
        self.diagnoser.ablation = self.ablation
        self.executor.history_analyzer = self.history_analyzer
        self.executor.system_memory = system_memory
        self.executor.ablation = self.ablation

        # v3: decompose_goal bonus step budget. When a corrected outline is
        # injected by decompose_goal, allow at most 1 extra execution step so the
        # fix actually runs before max_steps exhausts. Reset per task in solve().
        self._decompose_bonus = 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------


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
                    f"Step {i}: FORBIDDEN — Screenshot_Tool on PDF URL (triggers download)"
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
        mem_dir = self.ablation.memory_dir()
        if self.ablation.should_load_offline(self.evaluation_mode):
            loaded = self.system_memory.load_offline_memory(mem_dir)
        if self.verbose and loaded:
            print(f"\n==> 📚 Loaded offline memory from {mem_dir} (preset={self.ablation.preset})")

        self.executor.set_query_cache_dir(self.root_cache_dir)

        json_data: Dict[str, Any] = {
            "query": question,
            "image": image_path,
            "task_id": task_id,
            "trace_events": [],
            "layer_telemetry": [],
            "ablation": self.ablation.to_dict(),
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
            mem_dir = self.ablation.memory_dir()
            if self.ablation.enable_evolve:
                self.system_memory.evolve_tool_knowledge()
                if self.verbose:
                    print("\n==> 🧬 Tool Knowledge Memory: evolved from successful traces")

                if not self.evaluation_mode:
                    self.system_memory.persist_offline_memory(mem_dir)
                    if self.verbose:
                        print(f"✅ Persisted offline memory for next task ({mem_dir})")
            elif self.verbose:
                print("\n==> 🧬 Tool Knowledge Memory: evolve skipped (ablation.enable_evolve=False)")

            self.system_memory.persist_online_memory(task_id, mem_dir)
            if self.verbose:
                print(f"✅ Persisted online memory for task {task_id}")

            stats = self.system_memory.get_memory_stats(mem_dir)
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

    def _emit_layer_telemetry(
        self,
        json_data: Dict[str, Any],
        event: str,
        **fields: Any,
    ) -> None:
        """Product-track layer metrics (also mirrored into trace_events)."""
        if not getattr(self, "ablation", None) or not self.ablation.emit_layer_telemetry:
            return
        record = {"event": event, **fields}
        json_data.setdefault("layer_telemetry", []).append(record)
        self._append_trace_event(
            json_data,
            f"layer_{event}",
            exec_step=fields.get("exec_step"),
            outline_step=fields.get("outline_step"),
            tool=str(fields.get("tool") or ""),
            root_cause=str(
                fields.get("reason")
                or fields.get("root_cause")
                or fields.get("recommendation")
                or ""
            ),
        )

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
        obtained_for_prompt = self.system_memory.get_obtained_information_for_prompt()
        plan, _ = self.planner.generate_next_step(
            question,
            image_path,
            target_information,
            step_count,
            self.max_steps,
            obtained_for_prompt,
            json_data,
            diagnostic_signal=diagnostic_signal,
        )
        context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(
            plan,
            target_information=target_information,
            question=question,
            obtained_information=obtained_for_prompt,
        )
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
        target_information: str = "",
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

        if tool_name == "Web_Search_Tool":
            command = self._ensure_web_search_command(command, sub_goal, context)

        command = self._align_command_with_subgoal(
            tool_name, command, question, context, sub_goal,
        )
        command = self._sanitize_command_for_tool(tool_name, command)

        # Retrieval tools: synthesize a seed query when LLM command parse failed.
        _SEARCH_TOOLS = {
            "Google_Search_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool",
        }
        if tool_name in _SEARCH_TOOLS:
            q = self._extract_query_param(command)
            cmd_l = str(command or "").lower()
            if (
                self._is_garbage_retry_query(q)
                or "no command found" in cmd_l
                or "tool.execute" not in str(command)
            ):
                seed = self._seed_query_from_step(
                    sub_goal, target_information, question,
                )
                if seed:
                    print(
                        f"\n==> 🌱 Command missing/unparsed — seeding query from "
                        f"sub_goal/outline: {seed!r}\n"
                    )
                    command = self._inject_query_into_command(
                        "execution = tool.execute(query='placeholder')", seed,
                    )
                    if tool_name == "Web_Search_Tool":
                        command = self._ensure_web_search_command(
                            command, sub_goal, context,
                        )

        # Log the final aligned command (what actually executes), not the raw LLM draft.
        if self.verbose:
            print(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
            print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")

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

        # 单一工具裁决出口：resolve + policy（acquisition 绝不强制 Python）。
        tool_name = self._select_tool_for_step(
            tool_name, question, target_information, sub_goal,
            planner_signal, image_path, tracker,
        )

        # 诊断信号里若指定了 preferred_url，前置注入到 context（用于 L2 失败后定向重取）。
        if diagnostic_signal_prev:
            mutations = diagnostic_signal_prev.get("state_mutations") or {}
            preferred_url = mutations.get("preferred_url")
            if preferred_url and preferred_url not in context:
                context = f"{preferred_url} {context}".strip()

        if tool_name == "Web_Search_Tool":
            context = self._ground_web_context(context, target_information, question)

        if self.verbose:
            print(f"\n==> 🎯 Execution #{exec_step}: Action Prediction ({tool_name})\n")
            print(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")

        # === LLM 调用 #2：Executor.generate_tool_command === 生成具体工具命令并执行。
        # 内部：executor LLM 产出 ToolCommand（结构化）→ executor.execute_tool_command 真正调用工具。
        command, _, result = self._run_executor(
            question, image_path, context, sub_goal, tool_name,
            exec_step, json_data, diagnostic_signal_prev,
            target_information=target_information,
        )
        # Keep raw tool text for provisional finalize (even ABSENCE/refusal).
        try:
            blobs = getattr(self, "_provisional_raw_blobs", None)
            if blobs is not None and result is not None:
                text = str(result)
                if text.strip() and text not in blobs:
                    blobs.append(text[:8000])
                    if len(blobs) > 24:
                        del blobs[:-24]
        except Exception:
            pass
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

    def _verification_intervention_context(
        self,
        step_ctx: StepContext,
        intervention_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach generated_query/command so Diagnoser can detect parse failures."""
        ctx = dict(intervention_context or {})
        gq = self._extract_query_param(step_ctx.command)
        ctx["generated_query"] = "" if gq == "N/A" else gq
        ctx["command"] = str(step_ctx.command or "")
        return ctx

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
            intervention_context=self._verification_intervention_context(
                step_ctx, intervention_context,
            ),
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

        # Single post-verify slot write exit (persist + harvest).
        # obtained_information recording stays on complete/incomplete handlers
        # so candidate-set verifications do not pollute evidence.
        return self.after_verify(
            step_ctx, verification, step_count, record_obtained=False,
        )

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
                "IMPORTANT: NEVER return an empty ExecutionOutline unless evidence "
                "already fully answers the question. "
                "If evidence is sufficient, return empty remaining steps. "
                "Do NOT add a final synthesis/formatting step when the answer is already in evidence. "
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


    # ------------------------------------------------------------------
    # Answer sanity: consistency check + normalization (Scheme D)
    # ------------------------------------------------------------------


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

        self.system_memory.mark_step_done(step_ctx.step_key)

        # Drop the completed head step so leftover acquisition text cannot
        # block STOP via _outline_blocks_stop / force a synthesis rewrite.
        outline = dict(self.system_memory.get_outline() or {})
        numeric_keys = sorted(
            [k for k in outline.keys() if str(k).isdigit()],
            key=lambda x: int(x),
        )
        if numeric_keys:
            outline.pop(numeric_keys[0], None)
            reindexed: Dict[str, Any] = {}
            for i, k in enumerate(sorted(
                [x for x in outline.keys() if str(x).isdigit()],
                key=lambda x: int(x),
            ), start=1):
                reindexed[str(i)] = outline[k]
            self.system_memory.set_outline(reindexed)

        # Decide STOP before rewriting the outline — otherwise a fresh
        # synthesis step can re-block an already-filled SlotGate.
        if self._can_stop_execution(question, verification.analysis):
            if self._answer_sanity_pass(question):
                print(f"\n==> 🎯 Execution #{exec_step}: All required slots filled — STOP\n")
                return "stop"
            return "continue"

        prev_first = self._get_first_outline_step()
        prev_step_id = (
            self._outline_step_id(prev_first[0], prev_first[1])
            if prev_first[0] and prev_first[1] else None
        )

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
        """Orchestrator main loop (not an LLM role).

        init → analyze → while:
          step = plan.next() or plan.recover()
          ctx = execute(planner, tool_policy.select, executor, command_ops.align)
          ver = diagnoser.verify + evidence_binder.after_verify
          if incomplete: action = intervention.run(L1→L2a→L2b→L3)
          else: action = complete_handler(slotgate, answer_gate, plan.update)
          if action == stop: break
        persist memory
        """
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
        # Empty outline after Step0: one free re-analyze before recovery injection.
        empty_outline_analyze_retried = False
        if not (self.system_memory.get_outline() or {}):
            print("\n==> ⚠️ Step0 outline empty — retrying analyze_query once (not a recovery injection)\n")
            self._analyze_query(question, image_path, json_data, query_start_time)
            empty_outline_analyze_retried = True

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
        # Accumulate executor raw text for provisional finalize when slots empty.
        self._provisional_raw_blobs: List[str] = []

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
                    if not empty_outline_analyze_retried:
                        print(
                            "\n==> ⚠️ Outline empty mid-loop — retrying analyze_query "
                            "once (not a recovery injection)\n"
                        )
                        self._analyze_query(
                            question, image_path, json_data, query_start_time,
                        )
                        empty_outline_analyze_retried = True
                        if self._get_first_outline_step()[0]:
                            continue
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
            direct_output = None
            answer_status = "abstain"
            raw_blobs = list(getattr(self, "_provisional_raw_blobs", None) or [])
            for rec in records or []:
                raw_blobs.append(str(rec.get("content") or ""))
            try:
                raw_blobs.append(
                    str(self.system_memory.get_obtained_information_for_prompt() or "")
                )
            except Exception:
                pass
            if context_verification:
                raw_blobs.append(str(context_verification))

            # Prefer SlotGate even when evidence_verified is False — records may
            # already hold a scorable binding that stop-gate rejected.
            if profile:
                slot_answer = SlotGate.extract_final_answer(profile, records)
            if not evidence_verified:
                print(
                    "\n==> ⚠️ Final answer: cumulative evidence NOT fully verified — "
                    "constrained output mode\n"
                )
            if slot_answer:
                evidence_hint = ""
                try:
                    evidence_hint = str(
                        self.system_memory.get_obtained_information_for_prompt() or ""
                    )
                except Exception:
                    evidence_hint = ""
                slot_answer = self._normalize_final_answer(
                    question, slot_answer, evidence_hint=evidence_hint,
                )
                slot_ok, slot_reason = self._answer_consistency_check(
                    question, slot_answer,
                )
                # Reject acquisition JSON / rounding quantum / out-of-band after
                # normalize — treat as slot miss so provisional numeric can run.
                if slot_ok and (
                    ("{" in str(slot_answer) and "}" in str(slot_answer))
                    or self._answer_format_mismatch(question, str(slot_answer))
                ):
                    slot_ok = False
                    slot_reason = slot_reason or "post-normalize format reject"
                if evidence_verified and slot_ok:
                    direct_output = slot_answer
                    answer_status = "verified"
                    print(f"\n==> 🎯 Final answer from SlotGate: {direct_output}\n")
                elif evidence_verified and not slot_ok:
                    # Character unwrap path below may still salvage.
                    direct_output = slot_answer
                    answer_status = "verified"
                    print(f"\n==> 🎯 Final answer from SlotGate: {direct_output}\n")
                elif slot_ok:
                    # Unverified but format-ok slot value — provisional.
                    # For compute profiles, prefer distance×pace numeric first.
                    needs_compute = profile and any(
                        s.name == "computed_count" or s.min_source == "computed"
                        for s in profile.slots
                    )
                    if needs_compute and not evidence_verified:
                        numeric_first = self._provisional_numeric_finalize(
                            question, records, raw_blobs,
                        )
                        if numeric_first:
                            direct_output = numeric_first
                            answer_status = "provisional"
                            print(
                                f"\n==> 🎯 Final answer (provisional numeric "
                                f"before SlotGate): {direct_output}\n"
                            )
                        else:
                            direct_output = slot_answer
                            answer_status = "provisional"
                            print(
                                f"\n==> 🎯 Final answer from SlotGate "
                                f"(provisional): {direct_output}\n"
                            )
                    else:
                        direct_output = slot_answer
                        answer_status = "provisional"
                        print(
                            f"\n==> 🎯 Final answer from SlotGate "
                            f"(provisional): {direct_output}\n"
                        )
                else:
                    # Unverified + format-bad (e.g. "121 minutes" / "1000" quantum)
                    # — ignore and fall through to provisional finalize.
                    print(
                        f"\n==> ⚠️ SlotGate candidate rejected for scorable "
                        f"({slot_reason}): {slot_answer!r}\n"
                    )
                    slot_answer = None
            if not slot_answer and not direct_output:
                provisional = None
                if not evidence_verified:
                    # Compute profiles: numeric (distance×pace) before character.
                    needs_compute = profile and any(
                        s.name == "computed_count" or s.min_source == "computed"
                        for s in profile.slots
                    )
                    if needs_compute:
                        provisional = self._provisional_numeric_finalize(
                            question, records, raw_blobs,
                        )
                        if provisional is None:
                            provisional = self._provisional_character_finalize(
                                question, records, raw_blobs,
                                str(context_verification or ""),
                            )
                    else:
                        provisional = self._provisional_character_finalize(
                            question, records, raw_blobs,
                            str(context_verification or ""),
                        )
                        if provisional is None:
                            provisional = self._provisional_numeric_finalize(
                                question, records, raw_blobs,
                            )
                if provisional:
                    direct_output = provisional
                    answer_status = "provisional"
                    print(
                        f"\n==> 🎯 Final answer (provisional): {direct_output}\n"
                    )
                else:
                    direct_output = self.executor.generate_direct_output(
                        question,
                        context_verification,
                        self.system_memory,
                        evidence_verified=evidence_verified,
                    )
                    if not evidence_verified:
                        from MAS.epc_aw.models.task_profile import _extract_character_name
                        char_name = _extract_character_name(
                            f"{direct_output}\n{context_verification}"
                        )
                        if char_name:
                            direct_output = char_name
                            answer_status = "provisional"
                        else:
                            m = re.search(
                                r"(?m)^\s*(-?\d+)\s*$",
                                str(direct_output or "").strip(),
                            )
                            if not m:
                                m = re.search(
                                    r"(?i)\b(?:answer|final answer)\s*[:\-]?\s*(-?\d+)\b",
                                    str(direct_output or ""),
                                )
                            if m:
                                direct_output = m.group(1)
                                answer_status = "provisional"
                            else:
                                answer_status = "abstain"
                    else:
                        answer_status = "verified"
            json_data["direct_output"] = direct_output
            json_data["evidence_verified"] = evidence_verified
            json_data["answer_status"] = answer_status
            sanity_ok, sanity_reason = self._answer_consistency_check(
                question, direct_output,
            )
            if evidence_verified and sanity_ok:
                scorable = direct_output
            elif answer_status == "provisional" and str(direct_output or "").strip():
                m = re.search(r"-?\d+", str(direct_output))
                # Prefer digit for how-many; keep character names as-is.
                if re.search(r"\bhow many\b", str(question).lower()) and m:
                    scorable = m.group(0)
                else:
                    scorable = str(direct_output).strip()
                sanity_ok = True
            else:
                scorable = ""
            if evidence_verified and not sanity_ok and direct_output:
                from MAS.epc_aw.models.task_profile import _extract_character_name
                unwrapped = _extract_character_name(direct_output)
                if unwrapped:
                    retry_ok, _ = self._answer_consistency_check(question, unwrapped)
                    if retry_ok:
                        print(
                            f"\n==> 🩹 Final answer unwrap: {direct_output!r} → {unwrapped!r} "
                            f"(was rejected: {sanity_reason})\n"
                        )
                        direct_output = unwrapped
                        json_data["direct_output"] = direct_output
                        sanity_ok = True
                        scorable = unwrapped
                        answer_status = "verified"
                        json_data["answer_status"] = answer_status
            sanity_verified = bool(evidence_verified and sanity_ok)
            json_data["solved"] = sanity_verified
            if answer_status == "provisional" and scorable:
                json_data["scorable_output"] = scorable
            else:
                json_data["scorable_output"] = scorable if sanity_verified else ""
            json_data["trace_events"].append({
                "event": "finalization",
                "task_id": task_id,
                "exec_step": exec_step,
                "evidence_verified": evidence_verified,
                "solved": sanity_verified,
                "answer_status": answer_status,
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
    ablation: Optional[Any] = None,
    offline_memory_dir: Optional[str] = None,
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

    ablation_cfg = resolve_ablation(ablation)
    if offline_memory_dir:
        ablation_cfg.offline_memory_dir = offline_memory_dir

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
        ablation=ablation_cfg,
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
    parser.add_argument(
        "--ablation",
        default="full",
        help="Ablation preset: full|no_intervention|no_capability|no_invocation|no_memory|"
             "product_no_l2a|product_no_l2b|product_no_l3",
    )
    parser.add_argument(
        "--offline_memory_dir",
        default=None,
        help="Override offline/online memory root (default: memory or AblationConfig).",
    )
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
        ablation=args.ablation,
        offline_memory_dir=args.offline_memory_dir,
    )
    solver.solve("What is the capital of France?")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
