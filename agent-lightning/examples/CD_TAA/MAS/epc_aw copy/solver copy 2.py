import argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import re
from typing import Any, Dict, List, Tuple
from MAS.epc_aw.models.initializer import Initializer
from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.executor import Executor
from MAS.epc_aw.models.utils import make_json_serializable_truncated
# Day 3 Integration: Import causal reasoning modules
from MAS.epc_aw.models import CausalGraph, BayesianInference, HistoryAnalyzer, CausalInference


feasibility_criteria = """
You are an evaluator assessing the feasibility of a single plan. 
Score the plan from 1 to 5 based on its intrinsic feasibility and reliability.
Do NOT compare with other plans.

Scoring Rules:

Score 5 — Exceptional Feasibility
- The plan is internally coherent, precise, and well-justified.
- Tool selection and parameters are fully specified and theoretically sufficient
  to achieve the stated sub-goal with minimal epistemic uncertainty.
- Reasoning is complete, logically tight, and uses the available context optimally.
- No implicit assumptions or missing steps are required to interpret the plan.

Score 4 — Near-Perfect Feasibility
- The plan is coherent and well-aligned with the sub-goal.
- Tool selection is correct; parameters are appropriate but may allow minor refinement.
- Reasoning is sound, though some details could be made more explicit.
- The plan is interpretable without major inference.

Score 3 — Strong Feasibility
- The plan is plausible and addresses the sub-goal directly.
- Tool selection is mostly correct; some parameters or steps require mild inference.
- Reasoning is generally sound but may be shallow or partially underspecified.
- The plan remains interpretable, though not maximally precise.

Score 2 — Mostly Feasible
- The plan is relevant but exhibits notable epistemic gaps.
- Tool selection is reasonable, but parameters are under-specified or ambiguous.
- Reasoning relies on implicit assumptions or missing details.
- Additional clarification would be required to confidently interpret the plan.

Score 1 — Weak Feasibility
- The plan shows limited coherence or weak alignment with the sub-goal.
- Tool selection or parameter specification is incomplete or mismatched.
- Reasoning is vague, fragmented, or poorly grounded in the given context.
- The plan’s intended effect is difficult to infer epistemically.

Additional Tool-Calling Validity Constraints:
- Google Search may be used for any open-domain query.
- Wikipedia Search is valid only when context provides exactly one encyclopedic keyword.
- Web Search is valid only when the context contains a valid URL.

Calibration and Focus Constraints:
- Scores reflect the plan’s intrinsic feasibility and reliability, independent of other plans.
- Focus ONLY on:
    1) Justification correctness based on the given context
    2) Whether the selected tool and parameters can realistically achieve the sub-goal
"""

NEGATIVE_PATTERNS: List[str] = [
    r"\bno information\b",
    r"\bno relevant information\b",
    r"\bno relevant evidence\b",
    r"\bnot mentioned\b",
    r"\bdoes not mention\b",
    r"\bnot found\b",
    r"\bnothing found\b",
    r"\bno results\b",
    r"\bsearch returned no\b",
    r"\bno data available\b",
    r"\bunable to find\b",
    r"\bcould not find\b",
    r"\bno evidence\b"
]

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
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose

        # Day 3 Integration: Initialize causal reasoning system
        self.causal_graph = CausalGraph()
        self.bayesian_inference = BayesianInference()
        self.history_analyzer = HistoryAnalyzer()
        self.causal_inference = CausalInference(self.causal_graph, self.history_analyzer)

        # Share inference systems with submodules
        self.planner.causal_inference = self.causal_inference
        self.planner.history_analyzer = self.history_analyzer
        self.diagnoser.bayesian_inference = self.bayesian_inference
        self.diagnoser.history_analyzer = self.history_analyzer
        self.executor.history_analyzer = self.history_analyzer
    
    def process_next_step(self, next_step_list: list) -> tuple[list, list]:
        
        pattern_with_capture = r"(Feasibility Score:?\s*([-+]?\d*\.?\d+)(?:\s*\n)?)"
        
        cleaned_list = {}
        score_list = []
        
        for i, item in enumerate(next_step_list):
            match = re.search(pattern_with_capture, item, flags=re.IGNORECASE)
            
            if match:
                full_match = match.group(1) 
                score_str = match.group(2)
                try:
                    score_list.append(int(score_str))
                except ValueError:
                    score_list.append(None) 
                
                cleaned_item = item.replace(full_match, "")
            
            else:
                score_list.append(None)
                cleaned_item = item
                
            cleaned_list[str(i)] = cleaned_item
            
        return cleaned_list, score_list

    def solve(self, question: str, image_path: Optional[str] = None):
        """
        Solve a single problem with integrated causal graph memory system.

        Memory workflow:
        1. Load offline memory (tool ability boundary + parameter graphs)
        2. Execute task (record online memory)
        3. Upgrade online → offline knowledge
        4. Persist offline memory for next task
        """
        # Helper function to extract query parameter from command
        import re
        def extract_query_param(cmd_str):
            match = re.search(r'query=["\']([^"\']+)["\']', str(cmd_str))
            return match.group(1) if match else "N/A"

        # ============================================================================
        # 【步骤0】初始化task_id与加载离线内存
        # ============================================================================
        import uuid
        from datetime import datetime

        # 生成唯一的task_id
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.system_memory.online_memory["task_id"] = task_id

        if self.verbose:
            print(f"\n==> 📝 Task ID: {task_id}")

        # 加载离线内存 - 跨任务知识库
        loaded = self.system_memory.load_offline_memory("memory")
        if self.verbose and loaded:
            print(f"\n==> 📚 Loaded offline memory (Tool Ability Boundary + Parameter Graphs)")

        # Update cache directory for the executor
        self.executor.set_query_cache_dir(self.root_cache_dir)

        # Initialize json_data with basic problem information
        json_data = {
            "query": question,
            "image": image_path
        }
        if self.verbose:
            print(f"\n==> 🔍 Received Query: {question}")
            if image_path:
                print(f"\n==> 🖼️ Received Image: {image_path}")

        # Generate base response if requested
        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print(f"\n==> 📝 Base Response from LLM:\n\n{base_response}")

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            return json_data

        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):
            # if self.verbose:
            #     print(f"\n==> 🐙 Reasoning Steps from AF-MAS (Deep Thinking...)")

            # [1] Analyze query
            # ==================================== Query Analysis ==================================== 
            query_start_time = time.time()
            Analysis, ExecutionOutline = self.planner.analyze_query(question, image_path)
            json_data["analysis"] = Analysis
            json_data["outline"] = ExecutionOutline

            self.system_memory.set_outline(ExecutionOutline)

            if self.verbose:
                print(f"\n==> 🔍 Step 0: Query Analysis\n")
                print(f"{Analysis}")
                print(f"\n[Execution Outline]:\n{json.dumps(ExecutionOutline, indent=4)}")
                print(f"[Time]: {round(time.time() - query_start_time, 2)}s")
            
            preceding_step = 1

            # Initialize diagnostic signal (from previous cycle)
            diagnostic_signal_prev = None

            # Main execution loop
            step_count = 0
            action_times = []
            context_verification = ""
            while step_count < self.max_steps and (time.time() - query_start_time) < self.max_time:
                # Get the next step from outline (always the first one after updates)
                current_outline = self.system_memory.get_outline()
                print(f"\n==> 🗂️ Current Execution Outline:\n{json.dumps(current_outline, indent=4)}")

                # Find the first available step in the outline
                available_steps = sorted([k for k in current_outline.keys() if k.isdigit()], key=lambda x: int(x))

                if not available_steps:
                    print(f"\n==> 🎯 Task Completed - No more steps in outline")
                    break

                # Always use the first available step
                next_step_key = available_steps[0]
                target_information = current_outline[next_step_key]
                print(f"\n==> 🎯 Target Information for Step {step_count + 1}:\n{target_information}\n")
                step_count += 1
                step_start_time = time.time()
                local_start_time = time.time()

                # ==================================== Game-based Plan Selection ====================================
                plan, prompt_generate_next_step = self.planner.generate_next_step(
                    question,
                    image_path,
                    target_information,
                    feasibility_criteria,
                    step_count,
                    self.max_steps,
                    self.system_memory.get_obtained_information(),
                    json_data
                )

                # ==================================== Executor Process ====================================
                local_start_time = time.time()
                print(f"\n==> 🧩 Step {step_count}: Plan Parsing\n")
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(plan)
                
                if 'none' in tool_name.lower() or tool_name is None:
                        tool_name = "Base_Generator_Tool"

                if self.verbose:
                    print(f"\n==> 🎯 Step {step_count}: Action Prediction ({tool_name})\n")
                    print(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
                    print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"\n==> 🚫 Error: Tool '{tool_name}' is not available or not found.")
                    command = "No command was generated because the tool was not found."
                    result_executor = "No result was generated because the tool was not found."
                else:
                    # [3] Generate the tool command
                    local_start_time = time.time()
                    tool_command = self.executor.generate_tool_command(
                        question,
                        image_path,
                        context,
                        sub_goal,
                        tool_name,
                        self.system_memory.toolbox_metadata[tool_name],
                        step_count,
                        json_data,
                        diagnostic_signal_prev
                    )
                    
                    print(f"\n==> 🛠️ Step {step_count}: Tool Command Generation ({tool_name})\n")
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    if self.verbose:
                        print(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
                        print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
                    # [4] Execute the tool command
                    local_start_time = time.time()
                    result_executor = self.executor.execute_tool_command(tool_name, command)
                    result_executor = make_json_serializable_truncated(result_executor) # Convert to JSON serializable format
                    json_data[f"tool_result_{step_count}"] = result_executor

                    # 【新增】保存首次执行的command，用于后续参数对比
                    first_attempt_command = command

                    if self.verbose:
                        print(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
                        print(f"[Executor Result]:\n{json.dumps(result_executor, indent=4)}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                
                # Track execution time for the current step
                execution_time_step = round(time.time() - step_start_time, 2)
                action_times.append(execution_time_step)

                
                # ==================================== Diagnoser Process ====================================
                # [5] Two-layer verification: subgoal completion + task completion

                local_start_time = time.time()
                print(f"\n==> 🩺 Step {step_count}: Two-Layer Verification\n")

                # Call enhanced verificate_context with two-layer judgment
                context_verification, step_conclusion, add_obtained_information_flag, obtained_information, task_conclusion, diagnostic_signal_t = self.diagnoser.verificate_context(
                    question,
                    image_path,
                    target_information,
                    self.system_memory.get_outline(),
                    result_executor,
                    step_count,
                    self.system_memory.get_obtained_information(),
                    sub_goal,
                    tool_name
                )

                if self.verbose:
                    print(f"\n==> 🤖 Step {step_count}: Verification Results\n")
                    print(f"[Subgoal Conclusion]: {step_conclusion}")
                    print(f"[Task Conclusion]: {task_conclusion}")
                    print(f"[Analysis]: {context_verification}\n")
                    print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                # LAYER 1: Check if current subgoal is completed
                if step_conclusion == "SUBGOAL_INCOMPLETE":
                    # Subgoal not completed - generate causal diagnostic signal
                    print(f"\n==> 🕵️‍♂️ Step {step_count}: Causal Diagnosis (Subgoal Incomplete)\n")

                    if diagnostic_signal_t:
                        self.system_memory.set_diagnostic_signal(diagnostic_signal_t)
                        if self.verbose:
                            print(f"[Diagnostic Signal]: {json.dumps(diagnostic_signal_t, indent=2, ensure_ascii=False)}\n")

                    # 【新增】Memory集成：记录失败的执行轨迹为因果日志
                    # Layer 1: 因果执行轨迹（一个因→多个果）
                    state_type = "无完整信息"  # 从outline推导简化状态
                    effects = [{
                        "attempt_seq": 1,
                        "tool": tool_name,
                        "parameter": str(command)[:100],
                        "result": {"success": False},
                        "symptom": diagnostic_signal_t.get("failure_patterns", "") if diagnostic_signal_t else "Unknown",
                        "L1_diagnosis": "参数问题",
                        "success": False
                    }]

                    self.system_memory.record_causal_trace(
                        step=step_count,
                        state=state_type,
                        subgoal=sub_goal,
                        effects=effects,
                        L2_diagnosis=diagnostic_signal_t.get("root_cause_analysis", {}) if diagnostic_signal_t else {}
                    )

                    # Layer 2: 任务本地参数记忆
                    if diagnostic_signal_t and diagnostic_signal_t.get("recommendation") == "retry_with_different_parameters":
                        self.system_memory.add_failed_parameter(
                            state=state_type,
                            subgoal=sub_goal,
                            tool=tool_name,
                            parameter=str(command)[:100],
                            failure_symptom=diagnostic_signal_t.get("failure_patterns", ""),
                            L1_diagnosis="参数问题"
                        )

                    # Update memory and continue to next cycle
                    print(f"\n==> ℹ️ Step {step_count}: Recording failed attempt for causal learning\n")
                    # 【已删除】self.diagnoser.memory.add_action() - Diagnoser不持有memory
                    # 诊断结果由SystemMemory的record_causal_trace()记录

                    # Update diagnostic signal for next cycle
                    diagnostic_signal_prev = diagnostic_signal_t

                    # OPTIMIZATION: If recommendation is to retry with different parameters,
                    # skip Planner and directly call Executor with parameter guidance
                    if diagnostic_signal_t.get("recommendation") == "retry_with_different_parameters":
                        # ✅ 二次检查：确认subgoal真的失败了，而不是完成但后续任务未执行
                        if diagnostic_signal_t.get("subgoal_complete"):
                            print(f"\n==> ✅ Step {step_count}: Subgoal ACTUALLY Complete - Skip Parameter Retry\n")
                            if self.verbose:
                                print(f"[Decision]: Subgoal is complete despite retry recommendation.")
                                print(f"[Action]: Continue to next subgoal (let Planner execute the next step).")
                            continue  # 继续正常流程，让Planner执行下一步

                        print(f"\n==> 🔧 Step {step_count}: Retry with Different Parameters (Skip Planner)\n")

                        # ✅ 调试：打印诊断信号的完整内容
                        if self.verbose:
                            print(f"[Diagnostic Signal Details]:")
                            print(f"  recommendation: {diagnostic_signal_t.get('recommendation')}")
                            print(f"  reason: {diagnostic_signal_t.get('reason')}")
                            print(f"  failure_patterns: {diagnostic_signal_t.get('failure_patterns')}")

                            if diagnostic_signal_t.get("parameter_guidance"):
                                param_guidance = diagnostic_signal_t["parameter_guidance"]
                                print(f"\n[Parameter Guidance]:")
                                print(f"  explanation: {param_guidance.get('explanation')}")
                                print(f"  directions:")
                                for d in param_guidance.get('directions', []):
                                    print(f"    - {d}")
                                print(f"  avoid_patterns:")
                                for p in param_guidance.get('avoid_patterns', []):
                                    print(f"    - {p}")
                                if param_guidance.get('examples'):
                                    print(f"  examples:")
                                    for e in param_guidance.get('examples', []):
                                        print(f"    - {e}")
                                print()

                        step_count += 1
                        # Call Executor directly with diagnostic guidance
                        local_start_time = time.time()
                        print(f"\n==> ⚙️ Step {step_count}: Parameter Variant Execution\n")

                        # Extract the failed parameter value to prevent exact repetition
                        failed_param_value = extract_query_param(first_attempt_command)
                        if failed_param_value != "N/A":
                            if "parameter_guidance" not in diagnostic_signal_t:
                                diagnostic_signal_t["parameter_guidance"] = {}
                            # Add explicit instruction to avoid this exact parameter value
                            diagnostic_signal_t["parameter_guidance"]["failed_parameter"] = failed_param_value
                            diagnostic_signal_t["parameter_guidance"]["avoid_exact_match"] = f"Do NOT use query=\"{failed_param_value}\""

                        tool_command = self.executor.generate_tool_command(
                            question,
                            image_path,
                            context,
                            sub_goal,
                            tool_name,
                            self.system_memory.get_toolbox_metadata().get(tool_name, {}),
                            step_count=step_count,
                            json_data=json_data,
                            diagnostic_signal=diagnostic_signal_t  # Pass diagnostic info to guide parameter generation
                        )
                        
                        # Extract pure Python code from the response (same as normal flow)
                        analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)

                        # ✅ 调试：打印最终的prompt（截断显示）
                        if self.verbose:
                            prompt_content = json_data.get(f"tool_commander_{step_count}_prompt", "")
                            if prompt_content:
                                print(f"\n[Prompt Sent to Executor (First 800 chars)]:")
                                print(prompt_content[:800] + "..." if len(prompt_content) > 800 else prompt_content)
                                print()

                        if self.verbose:
                            print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
                            print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                        # ✅ 调试：比较前后参数
                        prev_query = extract_query_param(first_attempt_command)
                        curr_query = extract_query_param(command)
                        print(f"\n[Parameter Comparison]:")
                        print(f"  Previous query: '{prev_query}'")
                        print(f"  Current query:  '{curr_query}'")
                        if prev_query == curr_query:
                            print(f"  ⚠️ WARNING: Parameters are IDENTICAL! Parameter variation failed!")
                        else:
                            print(f"  ✓ Parameters are different")
                        print()

                        # Execute the tool with the new parameters
                        local_start_time = time.time()
                        result_executor = self.executor.execute_tool_command(tool_name, command)
                        result_executor = make_json_serializable_truncated(result_executor)
                        json_data[f"tool_result_{step_count}"] = result_executor
                        if self.verbose:
                            print(f"\n[Executor Result]:\n{json.dumps(result_executor, indent=4)}")
                            print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                        # Track execution time
                        execution_time_step = round(time.time() - step_start_time, 2)
                        action_times.append(execution_time_step)

                        # ✅ 关键修复：参数扰动后必须重新进行诊断验证
                        # 参数扰动就是一个新的"执行"，需要新的"诊断"
                        local_start_time = time.time()
                        print(f"\n==> 🩺 Step {step_count}: Re-verification After Parameter Variation\n")

                        context_verification, step_conclusion, add_obtained_information_flag, obtained_information, task_conclusion, diagnostic_signal_t = self.diagnoser.verificate_context(
                            question,
                            image_path,
                            target_information,
                            self.system_memory.get_outline(),
                            result_executor,
                            step_count,
                            self.system_memory.get_obtained_information(),
                            sub_goal,
                            tool_name
                        )

                        if self.verbose:
                            print(f"\n==> 🤖 Step {step_count}: Post-Parameter Verification Results\n")
                            print(f"[Subgoal Conclusion]: {step_conclusion}")
                            print(f"[Task Conclusion]: {task_conclusion}")
                            print(f"[Analysis]: {context_verification}\n")
                            print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                        # 重新进入主流程的判断逻辑（跳过Planner）
                        # 继续执行下面的if step_conclusion == "SUBGOAL_INCOMPLETE"判断
                        # (此时会根据新的step_conclusion和diagnostic_signal_t重新决定)
                    elif diagnostic_signal_t.get("recommendation") == "continue_to_next_subgoal":
                        # ✅ 新增：当subgoal已完成时，直接继续（不参数扰动）
                        print(f"\n==> ✅ Step {step_count}: Subgoal Complete - Continue to Next Subgoal\n")
                        if self.verbose:
                            print(f"[Decision]: Subgoal is complete. Proceeding to next step in outline.")

                        # ✅ 关键修复：在continue之前，先清理outline中已完成的步骤
                        # 这样下一次循环才能获取正确的下一个subgoal
                        if add_obtained_information_flag:
                            print(f"\n==> ℹ️ Step {step_count}: New relevant information obtained and added to system memory.\n")
                            self.system_memory.add_obtained_information(obtained_information)
                            print(f"\n[New Obtained Information]:\n{obtained_information}\n")
                        else:
                            print(f"\n==> ℹ️ Step {step_count}: No new relevant information obtained.\n")

                        # 清理outline：移除已完成的步骤，为下一个subgoal准备
                        print(f"\n==> 🗂️ Step {step_count}: Execution Outline Update (Subgoal Complete)\n")
                        updated_outline = self.diagnoser.update_outline(
                            question,
                            result_executor,
                            target_information,
                            self.system_memory.get_outline(),
                            result_executor,
                            step_count,
                            self.system_memory.get_obtained_information(),
                            self.system_memory.get_toolbox_metadata()
                        )

                        # 从outline中移除已完成的步骤
                        cleaned_outline = {}
                        current_step_num = int(next_step_key) if next_step_key.isdigit() else 0

                        for step_key, step_value in updated_outline.items():
                            try:
                                step_num = int(step_key)
                                # 只保留当前step之后的步骤
                                if step_num > current_step_num:
                                    # 重新编号：删除了前current_step_num个步骤后重新从1开始编号
                                    new_step_num = step_num - current_step_num
                                    cleaned_outline[str(new_step_num)] = step_value
                            except (ValueError, TypeError):
                                # 如果key不是数字，保留但不重新编号
                                pass

                        if cleaned_outline:
                            updated_outline = cleaned_outline
                            if self.verbose:
                                print(f"[Outline Cleaned]: Removed step '{next_step_key}' (completed)")
                                print(f"[Remaining Steps]: {len(cleaned_outline)}")
                        else:
                            # 没有剩余步骤，清空outline表示任务完成
                            if self.verbose:
                                print(f"[Outline Cleaned]: All steps completed!")

                        self.system_memory.set_outline(updated_outline)
                        if self.verbose:
                            print(f"\n[Updated Execution Outline]:\n{json.dumps(updated_outline, indent=4)}")

                        continue
                    else:
                        # For other recommendations (switch_tool, decompose_goal, modify_state),
                        # Outline will be updated in the next cycle, automatically presenting the next step
                        continue

                # LAYER 2: Subgoal is complete - now check if task is complete
                print(f"\n==> ✅ Step {step_count}: Subgoal Complete - Checking Task Completion\n")

                # 【新增】Memory集成：记录成功的执行轨迹
                state_type = "无完整信息"
                effects_success = [{
                    "attempt_seq": 1,
                    "tool": tool_name,
                    "parameter": str(command)[:100],
                    "result": {"success": True},
                    "symptom": None,
                    "L1_diagnosis": "成功",
                    "success": True
                }]

                self.system_memory.record_causal_trace(
                    step=step_count,
                    state=state_type,
                    subgoal=sub_goal,
                    effects=effects_success,
                    L2_diagnosis={"root_cause": "success"}
                )

                # Layer 2: 记录成功参数
                self.system_memory.add_successful_parameter(
                    state=state_type,
                    subgoal=sub_goal,
                    tool=tool_name,
                    parameter=str(command)[:100],
                    result="subgoal_completed"
                )

                # Record obtained information
                if add_obtained_information_flag:
                    print(f"\n==> ℹ️ Step {step_count}: New relevant information obtained and added to system memory.\n")
                    self.system_memory.add_obtained_information(obtained_information)
                    print(f"\n[New Obtained Information]:\n{obtained_information}\n")
                else:
                    print(f"\n==> ℹ️ Step {step_count}: No new relevant information obtained.\n")

                # Update outline and memory
                print(f"\n==> 🗂️ Step {step_count}: Execution Outline Update\n")
                updated_outline = self.diagnoser.update_outline(
                    question,
                    result_executor,
                    target_information,
                    self.system_memory.get_outline(),
                    result_executor,
                    step_count,
                    self.system_memory.get_obtained_information(),
                    self.system_memory.get_toolbox_metadata()
                )

                self.system_memory.set_outline(updated_outline)
                if self.verbose:
                    print(f"\n[Updated Execution Outline]:\n{json.dumps(updated_outline, indent=4)}")

                # Task conclusion judgment
                if task_conclusion == 'STOP':
                    print(f"\n==> 🎯 Step {step_count}: Task Complete - STOP\n")
                    break
                else:  # task_conclusion == 'CONTINUE'
                    print(f"\n==> 🔄 Step {step_count}: Task Incomplete - CONTINUE\n")
                    # Move to next outline step
                    preceding_step += 1
                    continue
            
            print(f"=============== Last Verification Analysis ================")
            print(f"{context_verification}")
            print(f"================== Obtained Information ===================")
            print(f"{self.system_memory.get_obtained_information()}")
            print(f"===========================================================")

            # Generate direct output if requested
            if 'direct' in self.output_types:
                direct_output = self.executor.generate_direct_output(question, context_verification, self.system_memory)
                json_data["direct_output"] = direct_output
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

            print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
            print(f"\n==> ✅ Query Solved!")

            # ============================================================================
            # 【步骤N】知识提升与持久化 - 从在线→离线，为下一个任务积累知识
            # ============================================================================
            try:
                # Step 1: 升级在线内存到离线内存
                self.system_memory.upgrade_online_to_offline()
                if self.verbose:
                    print(f"\n==> 📚 Knowledge Upgrade: Online → Offline")

                # Step 2: 持久化离线内存（跨任务知识）
                self.system_memory.persist_offline_memory("memory")
                if self.verbose:
                    print(f"✅ Persisted offline memory for next task")

                # Step 3: 持久化在线内存（当前任务轨迹）
                self.system_memory.persist_online_memory(task_id, "memory")
                if self.verbose:
                    print(f"✅ Persisted online memory for task {task_id}")

                # Step 4: 显示内存统计
                stats = self.system_memory.get_memory_stats("memory")
                if self.verbose:
                    print(f"\n[Memory Statistics]:")
                    print(f"  Online:  {stats['online_memory']['causal_execution_traces']} traces, {stats['online_memory']['task_local_parameters']} parameters")
                    print(f"  Offline: {stats['offline_memory']['tool_ability_boundary']} abilities, {stats['offline_memory']['tool_parameter_graphs']} patterns")

            except Exception as e:
                print(f"⚠️ Warning: Knowledge persistence failed: {e}")
                import traceback
                traceback.print_exc()

        return json_data


def construct_solver(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     output_types : str = "final,direct",
                     max_steps : int = 20,
                     max_time : int = 3000,
                     max_tokens : int = 4000,
                     root_cache_dir : str = "solver_cache",
                     verbose : bool = True,
                     vllm_config_path : str = None,
                     temperature: float = 0.0,
                     n: int =1
                     ):
    
    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )
    # Instantiate Planner
    planner = Planner(
        llm_engine_name=llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
        n=n,
    )

    # Instantiate Executor
    executor = Executor(
        # llm_engine_name=llm_engine_name,
        llm_engine_name=llm_engine_name,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
    )

    # Instantiate Diagnoser
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
            "last_plan_scores": [] 
        },
        "executor": {
            "strengths": ["Precise command generation", "Effective tool execution"],
            "weaknesses": ["Limited strategic insight", "Depends on clear sub-goals"],
            "last_plan_scores": []
        },
        "diagnoser": {
            "strengths": ["Critical evaluation", "Error detection"],
            "weaknesses": ["May be overly cautious", "Relies on comprehensive context"],
            "last_plan_scores": []
        }
    }

    # Instantiate System Memory
    system_memory = SystemMemory(
        toolbox_metadata=initializer.toolbox_metadata,
        agent_profile=agent_profile
    )

    solver = Solver(
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
        temperature=temperature
    )
    return solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the epc_aw demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()
    
def main(args):
    tool_engine=["gpt-4o","gpt-4o","Default","Default"]
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool"], # 
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        temperature=0.7
    )

    # Solve the task or problem
    solver.solve("What is the capital of France?")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
