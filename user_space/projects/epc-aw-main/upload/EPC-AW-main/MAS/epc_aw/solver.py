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

    def solve(self, question: str, image_path: Optional[str] = None, parallel: bool = False):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
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
            
            # Main execution loop
            step_count = 0
            action_times = []
            context_verification = ""
            while step_count < self.max_steps and (time.time() - query_start_time) < self.max_time:
                target_information = self.system_memory.get_outline()[str(preceding_step)]
                print(f"\n==> 🎯 Target Information for Step {step_count + 1}:\n{target_information}\n")
                step_count += 1
                step_start_time = time.time()
                local_start_time = time.time()

                # ==================================== Game-based Plan Selection ==================================== 
                plan_list, prompt_generate_next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    target_information, 
                    feasibility_criteria,
                    step_count, 
                    self.max_steps,
                    self.system_memory.get_obtained_information(),
                    json_data
                )
                # Extract context, sub-goal, and tool name
                if isinstance(plan_list, list):
                    cleaned_plan_list, _ = self.process_next_step(plan_list)
                    # =================== Planner evaluate ===================
                    
                    if parallel:
                        print(
                            f"\n==> Step {step_count}: Plan Evaluation (Planner / Executor / Diagnoser, parallel)\n"
                        )
                        _obtained = self.system_memory.get_obtained_information()
                        _agent_profile = self.system_memory.get_agent_profile()
                        _last_step_plan_scores = self.system_memory.get_last_step_plan_scores()
                        _meta = self.system_memory.toolbox_metadata

                        def _timed(name: str, fn):
                            t0 = time.perf_counter()
                            out = fn()
                            return name, out, time.perf_counter() - t0

                        _tasks = [
                            (
                                "planner_rg",
                                lambda: self.planner.report_generation(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    _obtained,
                                ),
                            ),
                            (
                                "planner_bp",
                                lambda: self.planner.belief_prediction(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    agent_profile=_agent_profile,
                                    last_step_plan_scores=_last_step_plan_scores,
                                    obtained_informtion=_obtained,
                                ),
                            ),
                            (
                                "executor_rg",
                                lambda: self.executor.report_generation(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    _obtained,
                                ),
                            ),
                            (
                                "executor_bp",
                                lambda: self.executor.belief_prediction(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    agent_profile=_agent_profile,
                                    last_step_plan_scores=_last_step_plan_scores,
                                    obtained_informtion=_obtained,
                                ),
                            ),
                            (
                                "diagnoser_rg",
                                lambda: self.diagnoser.report_generation(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    _obtained,
                                ),
                            ),
                            (
                                "diagnoser_bp",
                                lambda: self.diagnoser.belief_prediction(
                                    question,
                                    target_information,
                                    cleaned_plan_list,
                                    _meta,
                                    feasibility_criteria,
                                    agent_profile=_agent_profile,
                                    last_step_plan_scores=_last_step_plan_scores,
                                    obtained_informtion=_obtained,
                                ),
                            ),
                        ]

                        _parallel_t0 = time.perf_counter()
                        _results: Dict[str, Any] = {}
                        _task_secs: Dict[str, float] = {}
                        with ThreadPoolExecutor(max_workers=6) as _pool:
                            _futs = {
                                _pool.submit(_timed, name, fn): name for name, fn in _tasks
                            }
                            for _fut in as_completed(_futs):
                                _name, _payload, _sec = _fut.result()
                                _results[_name] = _payload
                                _task_secs[_name] = _sec
                        _parallel_wall = time.perf_counter() - _parallel_t0
                        _sum_task = sum(_task_secs.values())

                        ranked_plan_list_planner, planner_scores = _results["planner_rg"]
                        executor_scores_by_planner, diagnoser_scores_by_planner = _results[
                            "planner_bp"
                        ]
                        ranked_plan_list_executor, executor_scores = _results["executor_rg"]
                        planner_scores_by_executor, diagnoser_scores_by_executor = _results[
                            "executor_bp"
                        ]
                        ranked_plan_list_diagnoser, diagnoser_scores = _results["diagnoser_rg"]
                        planner_scores_by_diagnoser, executor_scores_by_diagnoser = _results[
                            "diagnoser_bp"
                        ]

                        def _scores_json(obj: Any) -> str:
                            return json.dumps(obj, indent=2, ensure_ascii=False)

                        print("######## Planner Scores (JSON) ########\n" + _scores_json(planner_scores))
                        print(
                            "######## Executor Scores by Planner (JSON) ########\n"
                            + _scores_json(executor_scores_by_planner)
                        )
                        print(
                            "######## Diagnoser Scores by Planner (JSON) ########\n"
                            + _scores_json(diagnoser_scores_by_planner)
                        )
                        print(
                            "######## Executor Scores (JSON) ########\n" + _scores_json(executor_scores)
                        )
                        print(
                            "######## Planner Scores by Executor (JSON) ########\n"
                            + _scores_json(planner_scores_by_executor)
                        )
                        print(
                            "######## Diagnoser Scores by Executor (JSON) ########\n"
                            + _scores_json(diagnoser_scores_by_executor)
                        )
                        print(
                            "######## Diagnoser Scores (JSON) ########\n"
                            + _scores_json(diagnoser_scores)
                        )
                        print(
                            "######## Planner Scores by Diagnoser (JSON) ########\n"
                            + _scores_json(planner_scores_by_diagnoser)
                        )
                        print(
                            "######## Executor Scores by Diagnoser (JSON) ########\n"
                            + _scores_json(executor_scores_by_diagnoser)
                        )
                        print(
                            f"[Eval parallel] wall_clock={_parallel_wall:.2f}s | "
                            f"sum(task_times)={_sum_task:.2f}s | "
                            f"approx_speedup_vs_serial={(_sum_task / _parallel_wall):.2f}x"
                        )
                        print(
                            f"[Eval parallel per-task] "
                            f"planner_rg={_task_secs['planner_rg']:.2f}s, "
                            f"planner_bp={_task_secs['planner_bp']:.2f}s, "
                            f"executor_rg={_task_secs['executor_rg']:.2f}s, "
                            f"executor_bp={_task_secs['executor_bp']:.2f}s, "
                            f"diagnoser_rg={_task_secs['diagnoser_rg']:.2f}s, "
                            f"diagnoser_bp={_task_secs['diagnoser_bp']:.2f}s"
                        )
                    
                    else:
                        print(f"\n==> 🧠 Step {step_count}: Plan Evaluation and Selection\n")
                        ranked_plan_list_planner, planner_scores = self.planner.report_generation(
                            question, 
                            target_information, 
                            cleaned_plan_list,
                            self.system_memory.toolbox_metadata,
                            feasibility_criteria,
                            self.system_memory.get_obtained_information(),
                        )

                        print(f"######## Planner Scores:              {planner_scores} ########")

                        executor_scores_by_planner, diagnoser_scores_by_planner = self.planner.belief_prediction(
                            question, 
                            target_information, 
                            cleaned_plan_list,
                            self.system_memory.toolbox_metadata,
                            feasibility_criteria,
                            agent_profile=self.system_memory.get_agent_profile(),
                            last_step_plan_scores=self.system_memory.get_last_step_plan_scores(),
                            obtained_informtion=self.system_memory.get_obtained_information()
                        )
                        print(f"######## Executor Scores by Planner:  {executor_scores_by_planner} ########")
                        print(f"######## Diagnoser Scores by Planner: {diagnoser_scores_by_planner} ########")


                        # =================== Executor evaluate ===================
                        print(f"\n==> 🏃‍♂️ Step {step_count}: Executor Plan Evaluation\n")
                        ranked_plan_list_executor, executor_scores = self.executor.report_generation(
                            question,
                            target_information, 
                            cleaned_plan_list, 
                            self.system_memory.toolbox_metadata, 
                            feasibility_criteria,
                            self.system_memory.get_obtained_information()
                        )
                        print(f"######## Executor Scores:              {executor_scores} ########")
                        planner_scores_by_executor, diagnoser_scores_by_executor = self.executor.belief_prediction(
                            question, 
                            target_information, 
                            cleaned_plan_list,
                            self.system_memory.toolbox_metadata,
                            feasibility_criteria,
                            agent_profile=self.system_memory.get_agent_profile(),
                            last_step_plan_scores=self.system_memory.get_last_step_plan_scores(),
                            obtained_informtion=self.system_memory.get_obtained_information()
                        )
                        print(f"######## Planner Scores by Executor:   {planner_scores_by_executor} ########")
                        print(f"######## Diagnoser Scores by Executor: {diagnoser_scores_by_executor} ########")

                        # =================== Diagnoser evaluate ===================
                        print(f"\n==> 🩺 Step {step_count}: Diagnoser Plan Evaluation\n")
                        ranked_plan_list_diagnoser, diagnoser_scores = self.diagnoser.report_generation(
                            question, 
                            target_information, 
                            cleaned_plan_list,
                            self.system_memory.toolbox_metadata,
                            feasibility_criteria,
                            self.system_memory.get_obtained_information()
                        ) 
                        print(f"######## Diagnoser Scores:             {diagnoser_scores} ########")
                        planner_scores_by_diagnoser, executor_scores_by_diagnoser = self.diagnoser.belief_prediction(
                            question, 
                            target_information, 
                            cleaned_plan_list,
                            self.system_memory.toolbox_metadata,
                            feasibility_criteria,
                            agent_profile=self.system_memory.get_agent_profile(),
                            last_step_plan_scores=self.system_memory.get_last_step_plan_scores(),
                            obtained_informtion=self.system_memory.get_obtained_information()
                        )
                        print(f"######## Planner Scores by Diagnoser:  {planner_scores_by_diagnoser} ########")
                        print(f"######## Ex`ecutor Scores by Diagnoser: {executor_scores_by_diagnoser} ########")

                    # =================== 综合评分 BTS ===================
                    print(f"\n==> 🤖 Step {step_count}: BTS Plan Selection\n")
                    plan_results = self.planner.compute_bts_for_plans(
                        planner_scores,
                        executor_scores,
                        diagnoser_scores,
                        executor_scores_by_planner,
                        diagnoser_scores_by_planner,
                        planner_scores_by_executor,
                        diagnoser_scores_by_executor,
                        planner_scores_by_diagnoser,
                        executor_scores_by_diagnoser,
                    )
                    BTS_scores = {k: v["bts_score"] for k, v in plan_results.items()}
                    print(f"######## BTS scores: {BTS_scores} ########")
                    planner_selected_index = max(
                        planner_scores.keys(),
                        key=lambda k: planner_scores[k]
                    )
                    print(f"######## Planner Selected Plan Index: {planner_selected_index} ########")
                    BTS_selected_index = max(
                                plan_results.keys(),
                                key=lambda k: (
                                    plan_results[k]["bts_score"],
                                    plan_results[k]["A_bar"]
                                ))
                    print(f"######## BTS Selected Plan Index: {BTS_selected_index} ########")
                    BTS_selected_plan = cleaned_plan_list[BTS_selected_index]
                    print(f"======== Selected_plan: =========")
                    print(f"{BTS_selected_plan}")
                    print(f"=================================")
                
                # ==================================== Executor Process ==================================== 
                local_start_time = time.time()
                print(f"\n==> 🧩 Step {step_count}: Plan Parsing\n")
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(BTS_selected_plan)
                
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
                        json_data
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

                    if self.verbose:
                        print(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
                        print(f"[Executor Result]:\n{json.dumps(result_executor, indent=4)}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                
                # Track execution time for the current step
                execution_time_step = round(time.time() - step_start_time, 2)
                action_times.append(execution_time_step)

                
                # ==================================== Diagnoser Process ==================================== 
                # [5] Verify memory (context verification)
                
                local_start_time = time.time()
                print(f"\n==> 🩺 Step {step_count}: Context Verification\n")
                context_verification, add_obtained_information_flag, obtained_information, conclusion = self.diagnoser.verificate_context(
                    question, 
                    image_path, 
                    target_information, 
                    self.system_memory.get_outline(),
                    result_executor,
                    step_count,
                    self.system_memory.get_obtained_information(),
                )

                if add_obtained_information_flag:
                    print(f"\n==> ℹ️ Step {step_count}: New relevant information obtained and added to system memory.\n")
                    self.system_memory.add_obtained_information(obtained_information)
                    print(f"\n[New Obtained Information]:\n{obtained_information}\n")
                else:
                    print(f"\n==> ℹ️ Step {step_count}: No new relevant information obtained.\n")

                if self.verbose:
                    conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
                    
                    print(f"\n==> 🤖 Step {step_count}: Context Verification\n")
                    print(f"[Analysis]: {context_verification}\n))")
                    print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                
                # Break the loop if the context is verified
                if conclusion == 'STOP' and add_obtained_information_flag and step_count > 1:
                    print(f"[Decision]: STOP 🛑")
                    break
                else:
                    print(f"[Decision]: CONTINUE 🛑")
                    
                    print(f"\n==> 🕵️‍♂️ Step {step_count}: EPC-AW Strategy Generation\n")
                    epc_aw_analysis = ""
                    epistemic_constraint = ""
                    if planner_selected_index != BTS_selected_index:
                        planner_selected_plan = cleaned_plan_list[planner_selected_index]
                        epc_aw_analysis, epistemic_constraint = self.diagnoser.epc_aw_diagnosis(
                            planner_selected_plan, 
                            BTS_selected_plan,
                            target_information,
                            self.system_memory.get_outline(),
                            result_executor,
                            step_count,
                            json_data
                        )
                        if self.verbose:
                            print(f"\n==> 🕵️‍♂️ Step {step_count}: EPC-AW Strategy Analysis\n")
                            print(f"[EPC-AW Analysis]: {epc_aw_analysis}")
                            print(f"[Epistemic Constraint]: {epistemic_constraint}")
                        
                        self.planner.memory.add_epistemic_constraint(epistemic_constraint)


                    # [7] Update execution outline
                    print(f"\n==> 🗂️ Step {step_count}: Execution Outline Update\n")
                    updated_outline = self.diagnoser.update_outline(
                        question, 
                        result_executor, 
                        target_information, 
                        self.system_memory.get_outline(),
                        result_executor,
                        step_count,
                        self.system_memory.get_obtained_information(),
                        epc_aw_analysis,
                        self.system_memory.get_toolbox_metadata()
                    )
                    
                    self.system_memory.set_outline(updated_outline)
                    if self.verbose:
                        print(f"\n[Updated Execution Outline]:\n{json.dumps(updated_outline, indent=4)}")
                    
                    if add_obtained_information_flag:                        
                        print(f"\n==> ℹ️ Step {step_count}: Updating Diagnoser Memory with new information.\n")
                        self.diagnoser.memory.add_action(
                            step_count,
                            target_information,
                            BTS_selected_plan,
                            tool_name,
                            command,
                            result_executor,
                            context_verification,
                        )
                    else:
                        print(f"\n==> ℹ️ Step {step_count}: Updating Planner Memory to avoid similar plans in future.\n")
                        self.planner.memory.add_action(
                            step_count,
                            target_information,
                            BTS_selected_plan,
                            tool_name,
                            command,
                            result_executor,
                            context_verification,
                        )

                    print(f"\n==> ℹ️ Step {step_count}: Updating Executor Memory with the action taken.\n")
                    self.executor.memory.add_action(
                        step_count,
                        target_information,
                        BTS_selected_plan,
                        tool_name,
                        command,
                        result_executor,
                    )
            
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
