import importlib
import json
import os
import re
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional


from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import ToolCommand, ToolCommandSet, FinalAnswer
from MAS.epc_aw.models.memory import Memory
from MAS.epc_aw.models.utils import parse_json_from_llm_response
# Day 3 Integration: Import causal reasoning modules
from MAS.epc_aw.models import CausalInference, HistoryAnalyzer


# -----------------------------------------------------------------------------
# Causal Hypothesis → instruction rendering (P0-②).
#
# ``INSTRUCTION_TEXT`` maps each instruction KEY (emitted by the Diagnoser's
# ``_generate_parameter_guidance`` via ``REASON_INSTRUCTION``) to a short
# natural-language directive. Keeping the text here (rather than in the
# Diagnoser) means the Diagnoser stays a pure classifier; the Executor
# owns the LLM-facing prose. Add a new key here when a new reason is
# introduced in REASON_INSTRUCTION — that is the only edit needed.
# -----------------------------------------------------------------------------
INSTRUCTION_TEXT: Dict[str, str] = {
    # query formulation
    "NeedEntityAnchor":     "Add an explicit entity anchor to the query: a publisher name, journal title, year, or numeric identifier. Avoid generic noun-only queries.",
    "DropOneConstraint":    "Drop the least informative limiting constraint from the previous query; keep the most discriminating entity.",
    "AddDomainContext":     "Add domain context (field, publisher, platform) to disambiguate the generic keywords used previously.",
    "FixYear":              "The year in the previous query is wrong or missing; correct it from the task question.",
    "FixPublisher":         "The publisher/journal name in the previous query is wrong or missing; correct it from the task question.",
    # retrieval coverage
    "PreferAuthoritativeSource": "Prefer an authoritative source URL (e.g. nature.com, springer.com, wikipedia.org, official annual report) over a generic search.",
    "UseOpenArchiveOrMirror":    "The likely source is paywalled; try an open archive or mirror (web.archive.org, doi.org, preprint server).",
    "BroadenQueryScope":         "Broaden the query scope; the previous query was too narrow for the available index.",
    # execution / external
    "SkipParameterRetryEscalateToToolSwitch": "This failure is NOT parameter-fixable (thread/async-context error). Do NOT regenerate a near-identical command; the upstream scheduler will switch tools.",
    "ShortenQueryTokens":        "Shorten the query; the backend may be rate-limiting or timing out on long queries.",
    "RegenerateCommandFromScratch": "Regenerate the command from scratch with a different parameter structure, not a wording tweak.",
    "RewriteQueryFromSubgoal": "No usable query was sent. Rebuild the tool command from the current sub-goal / outline Target Information; do NOT emit placeholder queries like retry1.",
    # environment
    "AvoidImportUseStdlib":  "Avoid the rejected import; use stdlib (math/statistics) or numpy instead.",
    "SwitchToolOrEscalateState": "Permission denied; either switch tool or escalate to environment/state modification.",
    # command schema
    "FixArgumentSchema":     "Fix the command argument schema: match the tool's expected parameter names and arity.",
    # tool selection
    "SwitchToolClass":       "The current tool class cannot satisfy the required capability; switch to a different tool class.",
    # planner
    "ReviseBelief":          "Revise the prior belief/claim that conflicts with the new evidence before retrying.",
    "DecomposeGoal":         "Decompose the current sub-goal into smaller, independently retrievable sub-goals.",
    # default
    "DefaultRetry":          "Generate a substantially different parameter value or approach; do NOT repeat the previous parameter.",
}

# Tool name mapping: Static fallback mapping (long external names to internal)
TOOL_NAME_MAPPING_LONG = {
    "Base_Generator_Tool": {
        "class_name": "Base_Generator_Tool",
        "dir_name": "base_generator"
    },
    "Google_Search_Tool": {
        "class_name": "Google_Search_Tool",
        "dir_name": "google_search"
    },
    "Python_Coder_Tool": {
        "class_name": "Python_Coder_Tool",
        "dir_name": "python_coder"
    },
    "Web_Search_Tool": {
        "class_name": "Web_Search_Tool",
        "dir_name": "web_search"
    },
    "Wikipedia_Search_Tool": {
        "class_name": "Wikipedia_Search_Tool",
        "dir_name": "wikipedia_search"
    }
}

# Short to long mapping for fallback
TOOL_NAME_MAPPING_SHORT = {
    "Base_Generator_Tool": "Base_Generator_Tool",
    "Google_Search_Tool": "Google_Search_Tool",
    "Python_Coder_Tool": "Python_Coder_Tool",
    "Web_Search_Tool": "Web_Search_Tool",
    "Wikipedia_Search_Tool": "Wikipedia_Search_Tool"
}

try:
    TimeoutError
except NameError:
    class TimeoutError(Exception):
        pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120,
    max_output_length: int = 100000, verbose: bool = False, base_url: str = None, check_model: bool = True, temperature: float = .0):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.base_url = base_url
        self.check_model = check_model
        self.temperature  = temperature
        # self.memory = ExecutorMemory()  # 【已删除】使用SystemMemory替代
        self.profile = ""
        if base_url is not None:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, base_url=self.base_url, temperature = self.temperature)
        else:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, temperature = self.temperature)

        # Day 3 Integration: Initialize constraint checking and parameter optimization
        self.history_analyzer = HistoryAnalyzer()
        self.system_memory = None
        self.current_tool = None
        self.current_subgoal = None
    
    def get_profile(self) -> str:
        return self.profile

    # Day 3 Integration: Constraint checking and parameter recording methods

    def _record_execution(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]):
        """
        Record execution for learning.
        Updates history analyzer with execution details.
        """
        try:
            success = result.get('success', False)
            error = result.get('error', None)

            self.history_analyzer.record_execution(
                tool=tool_name,
                subgoal=self.current_subgoal or "unknown",
                parameters=params,
                success=success,
                error=error
            )

            if self.verbose:
                status = "✓" if success else "✗"
                print(f"[Executor] {status} Recorded: {tool_name} with params {params} → {success}")

        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to record execution: {e}")

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

    
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None, diagnostic_signal: Dict[str, Any] = None, n_candidates: int = 1) -> Any:
        """
        Generate tool command with support for diagnostic signals from previous cycle.
        If diagnostic signal contains parameter guidance, incorporate it into the prompt prominently.

        When n_candidates > 1, a SINGLE LLM call returns up to n_candidates mutually
        distinct ToolCommand objects (ToolCommandSet). This is the cost-neutral
        mechanism behind Level 2 Intervention's parameter candidate set: the previous
        sequential retry spent 1 LLM call per attempt (N calls for N candidates);
        the multi-candidate mode spends 1 call total for all N candidates.
        """
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "executor", "generate_tool_command.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Build diagnostic hint if available
        diagnostic_hint_prefix = ""
        diagnostic_hint_suffix = ""

        if diagnostic_signal and diagnostic_signal.get("triggered"):
            # Check for parameter guidance from retry analysis
            if diagnostic_signal.get("parameter_guidance"):
                param_guidance = diagnostic_signal["parameter_guidance"]
                directions = param_guidance.get("directions", [])
                avoid_patterns = param_guidance.get("avoid_patterns", [])
                examples = param_guidance.get("examples", [])
                explanation = param_guidance.get("explanation", "")
                failed_parameter = param_guidance.get("failed_parameter")
                avoid_exact_match = param_guidance.get("avoid_exact_match")
                tried_queries = param_guidance.get("tried_queries") or []
                # Causal Hypothesis fields (P0-②). When ``instruction`` is
                # present and non-default, the hint is rendered from
                # INSTRUCTION_TEXT — the legacy directions/avoid/examples
                # are only used as fallback for DefaultRetry/empty reason.
                instruction = param_guidance.get("instruction")
                hypothesis_target = param_guidance.get("target_variable", "")
                hypothesis_reason = param_guidance.get("reason", "")
                hypothesis_conf = param_guidance.get("confidence", "MEDIUM")
                instruction_text = INSTRUCTION_TEXT.get(instruction or "", "")

                # ✅ CRITICAL PREFIX: Placed BEFORE regular prompt to influence LLM strongly
                diagnostic_hint_prefix = f"""
================================================================================
🚨 IMPORTANT - PARAMETER VARIATION REQUIRED 🚨
================================================================================

PREVIOUS ATTEMPT FAILED - You MUST generate DIFFERENT parameters this time!

"""
                # Primary path: instruction-driven hint (Causal Hypothesis Testing).
                if instruction and instruction != "DefaultRetry" and instruction_text:
                    diagnostic_hint_prefix += (
                        f"CAUSAL HYPOTHESIS (testable):\n"
                        f"  target_variable = {hypothesis_target}\n"
                        f"  reason          = {hypothesis_reason}\n"
                        f"  confidence      = {hypothesis_conf}\n"
                        f"  instruction     = {instruction}\n"
                        f"  directive       = {instruction_text}\n\n"
                        f"CRITICAL: DO NOT generate the same parameter again!\n"
                        f"❌ FORBIDDEN parameter value: {failed_parameter}\n"
                        f"{avoid_exact_match}\n"
                    )
                else:
                    # Legacy fallback: explanation-driven hint (kept for
                    # DefaultRetry / unknown reason, preserves old behaviour).
                    diagnostic_hint_prefix += (
                        f"Reason: {explanation}\n\n"
                        f"CRITICAL: DO NOT generate the same parameter again!\n"
                        f"❌ FORBIDDEN parameter value: {failed_parameter}\n"
                        f"{avoid_exact_match}\n"
                    )

                if tried_queries:
                    diagnostic_hint_prefix += "\nALREADY TRIED (do NOT repeat):\n"
                    for tq in tried_queries:
                        diagnostic_hint_prefix += f"  ❌ {tq}\n"

                # Legacy directions are only appended when the instruction
                # path did not fire (i.e. fallback). Under the instruction
                # path, the directive above already conveys the strategy,
                # and stacking both yields a noisy, contradictory prompt.
                if not (instruction and instruction != "DefaultRetry" and instruction_text):
                    diagnostic_hint_prefix += """
REQUIRED PARAMETER VARIATIONS:
"""
                    for i, direction in enumerate(directions, 1):
                        diagnostic_hint_prefix += f"{i}. {direction}\n"

                    if avoid_patterns:
                        diagnostic_hint_prefix += "\nDO NOT use these patterns:\n"
                        for pattern in avoid_patterns:
                            diagnostic_hint_prefix += f"  ❌ {pattern}\n"

                    if examples:
                        diagnostic_hint_prefix += "\nCONCRETE EXAMPLES OF WHAT TO DO:\n"
                        for example in examples:
                            diagnostic_hint_prefix += f"  ✓ {example}\n"

                # ✅ CRITICAL SUFFIX: Strong reminder at the end
                diagnostic_hint_suffix = f"""
================================================================================
⚠️  CRITICAL REQUIREMENT ⚠️
Your generated parameters MUST be substantially different from the previous attempt.
FORBIDDEN EXACT MATCH: query=\"{failed_parameter}\"

Think about alternative approaches:
- Try different keywords or keyword combinations
- Add qualifiers or filters to narrow down results
- Use related terms or synonyms
- Add context or domain information

If you generate: query=\"{failed_parameter}\"
The search will fail again, and the task will not progress!
================================================================================
"""

        memory_hint = self._build_parameter_memory_hint(sub_goal, tool_name)
        if memory_hint:
            diagnostic_hint_prefix += memory_hint

        # Multi-candidate suffix: when n_candidates > 1, instruct the LLM to return
        # N mutually distinct commands in ONE call (cost-neutral vs sequential retry).
        multi_candidate_suffix = ""
        if n_candidates and n_candidates > 1:
            tried_block = ""
            tried_queries = []
            if diagnostic_signal and diagnostic_signal.get("parameter_guidance"):
                tried_queries = diagnostic_signal["parameter_guidance"].get("tried_queries") or []
            if tried_queries:
                tried_block = "\nALREADY TRIED (do NOT repeat any of these):\n"
                for tq in tried_queries:
                    tried_block += f"  - {tq}\n"
            multi_candidate_suffix = f"""
================================================================================
🎯 MULTI-CANDIDATE GENERATION (Level 2 Intervention)
================================================================================
Generate up to {n_candidates} MUTUALLY DISTINCT candidate commands for this subgoal.
Each candidate must use a DIFFERENT parameter strategy (different keywords,
different URL, different query phrasing, different filters). Do NOT duplicate
parameters across candidates.{tried_block}
Return ALL candidates as the `candidates` list. If you can only produce fewer
distinct strategies, return fewer — quality over forced duplication.
================================================================================
"""

        # Construct the full prompt
        prompt_generate_tool_command = diagnostic_hint_prefix + prompt_template.format(
            Question=question,
            Sub_Goal=sub_goal,
            Tool_Name=tool_name,
            Tool_Metadata=tool_metadata,
            Relevant_Data=context
        ) + diagnostic_hint_suffix + multi_candidate_suffix

        response_format = ToolCommandSet if (n_candidates and n_candidates > 1) else ToolCommand
        tool_command = self.llm_generate_tool_command(prompt_generate_tool_command, response_format=response_format)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command

    def _extract_multiple_commands(self, response: Any) -> List[str]:
        """Parse a ToolCommandSet (or fallback single ToolCommand) into a list of
        command strings. Tolerant of LLM returning a single ToolCommand when
        n_candidates>1 was requested — degrades gracefully to a 1-element list.
        """
        commands: List[str] = []
        try:
            if isinstance(response, ToolCommandSet):
                for cand in (response.candidates or []):
                    cmd = getattr(cand, "command", "") or ""
                    if cmd:
                        commands.append(cmd)
            elif isinstance(response, ToolCommand):
                cmd = getattr(response, "command", "") or ""
                if cmd:
                    commands.append(cmd)
            elif isinstance(response, dict):
                cands = response.get("candidates") or []
                for cand in cands:
                    if isinstance(cand, dict):
                        cmd = cand.get("command") or ""
                    else:
                        cmd = getattr(cand, "command", "") or ""
                    if cmd:
                        commands.append(cmd)
                if not commands and response.get("command"):
                    commands.append(response["command"])
        except Exception:
            pass
        return commands

    def _build_parameter_memory_hint(self, sub_goal: str, tool_name: str) -> str:
        """Inject Tool Invocation Memory factors for (tool, subgoal) — ≤8 lines.

        Pure text: each factor name + its single instruction. No
        scores/counts (per spec). Online failed-parameter blacklist is
        still appended to avoid repeating the same mistake this task.
        """
        if not self.system_memory:
            return ""

        ablation = getattr(self, "ablation", None)
        read_inv = ablation is None or ablation.read_invocation_memory
        use_blacklist = ablation is None or ablation.use_param_blacklist

        entry = None
        if read_inv:
            entry = self.system_memory.retrieve_tool_invocation(tool_name, sub_goal)
        else:
            print("[MemoryHint] TOOL INVOCATION skipped (read_invocation_memory=False)")

        blacklist = []
        if use_blacklist:
            blacklist = self.system_memory.get_failed_parameter_blacklist(
                self.system_memory.infer_state_type(), sub_goal, tool_name,
            )

        lines = [f"\n🧰 TOOL INVOCATION ({tool_name}):"]
        if entry:
            factors = entry.get("factors") or {}
            for fname, fobj in factors.items():
                instr = (fobj or {}).get("instruction", "")
                lines.append(f"  • {fname}: {instr}")
        if blacklist:
            lines.append(f"  Do not repeat: {blacklist[0][:80]}")
        if len(lines) <= 1:
            return ""
        if entry:
            print(f"[MemoryHint] TOOL INVOCATION injected for {tool_name}")
        elif blacklist:
            print(f"[MemoryHint] TOOL INVOCATION blacklist-only for {tool_name}")
        return "\n".join(lines[:8]) + "\n"
    
    
        
    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading/trailing whitespace and triple backticks if present
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        def parse_response(response):
            # Clean weird leading/trailing quotes
            if isinstance(response, str):
                response = response.strip().strip("'").strip()

            # JSON first
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                    response_obj = ToolCommand(**response_dict)
                    analysis = response_obj.analysis.strip()
                    explanation = response_obj.explanation.strip()
                    command = response_obj.command.strip()
                    return analysis, explanation, normalize_code(command)
                except Exception as e:
                    pass  # continue to regex parsing

                # --- Regex parsing fallback ---
                try:
                    # Extract analysis if present
                    analysis_pattern = r"Analysis:(.*?)Command Explanation"
                    match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
                    analysis = match.group(1).strip() if match else "No analysis found."

                    # Extract explanation if present
                    explanation_pattern = r"Command Explanation:(.*?)Generated Command"
                    match = re.search(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
                    explanation = match.group(1).strip() if match else "No explanation found."

                    # --- Extract python code block (robust version) ---
                    code_block_pattern = r"```python\s*(.*?)```"
                    match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
                    if match:
                        command = match.group(1).strip()
                    else:
                        # fallback: any code block
                        any_block_pattern = r"```\s*(.*?)```"
                        match = re.search(any_block_pattern, response, re.DOTALL | re.IGNORECASE)
                        command = match.group(1).strip() if match else "No command found."

                except Exception:
                    analysis = "Parsing error."
                    explanation = "Parsing error."
                    command = "No command found."

                return analysis, explanation, normalize_code(command)

            # Direct ToolCommand object
            elif isinstance(response, ToolCommand):
                return (response.analysis.strip(),
                        response.explanation.strip(),
                        normalize_code(response.command.strip()))

            else:
                return "Invalid response", "Invalid response", "Invalid response"
        
        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."
        analysis, explanation, command = parse_response(response)
        
        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls

        Returns:
            Any: List of execution results or error message
        """

        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Set up the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_time)

            try:
                # Execute the block in the local context
                exec(block, globals(), local_context)
                result = local_context.get('execution')
                signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                return f"Execution timed out after {self.max_time} seconds"
            finally:
                signal.alarm(0)  # Ensure alarm is disabled even if other exceptions occur

        # Import the tool module and instantiate it
        # tool_name could be either short or long name
        # First check if it's a long name
        if tool_name in TOOL_NAME_MAPPING_LONG:
            dir_name = TOOL_NAME_MAPPING_LONG[tool_name]["dir_name"]
            class_name = TOOL_NAME_MAPPING_LONG[tool_name]["class_name"]
        # Then check if it's a short name (convert to long, then get internal)
        elif tool_name in TOOL_NAME_MAPPING_SHORT:
            long_name = TOOL_NAME_MAPPING_SHORT[tool_name]
            if long_name in TOOL_NAME_MAPPING_LONG:
                dir_name = TOOL_NAME_MAPPING_LONG[long_name]["dir_name"]
                class_name = TOOL_NAME_MAPPING_LONG[long_name]["class_name"]
            else:
                # Shouldn't happen, but fallback
                dir_name = tool_name.lower().replace('_tool', '')
                class_name = tool_name
        else:
            # Fallback to original behavior for unmapped tools
            dir_name = tool_name.lower().replace('_tool', '')
            class_name = tool_name

        module_name = f"tools.{dir_name}.tool"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the tool class
            tool_class = getattr(module, class_name)
            
            tool = tool_class()

            # Set the custom output directory
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"

    def _format_history_actions(self, memory: Memory) -> str:
        """Build execution history from legacy actions or SystemMemory causal traces."""
        actions = memory.get_actions()
        if actions:
            return json.dumps(actions, ensure_ascii=False, indent=2)

        online = getattr(memory, "online_memory", None)
        if isinstance(online, dict):
            traces = online.get("causal_execution_traces") or []
            if traces:
                return json.dumps(traces, ensure_ascii=False, indent=2)

        return "No recorded actions."

    def _get_obtained_info_text(self, memory: Memory) -> str:
        if hasattr(memory, "get_obtained_information_for_prompt"):
            return memory.get_obtained_information_for_prompt()
        info = memory.get_obtained_information()
        return str(info) if info else "None"

    def generate_final_output(self, question: str, memory: Memory) -> str:
        obtained_text = self._get_obtained_info_text(memory)
        prompt_generate_final_output = f"""
            Task: Generate the final output based on the query and the results from all tools used.

            Context:
            - **Query:** {question}
            - **Obtained Information:** {obtained_text}

            Instructions:
            1. Review the query and the obtained information.
            2. Incorporate the relevant information to create a coherent, step-by-step final output.
            """

        input_data = [prompt_generate_final_output]
        
        final_output = self.llm_generate_tool_command(input_data)

        return final_output


    def generate_direct_output(
        self,
        question: str,
        last_verification_analysis: str,
        memory: Memory,
        evidence_verified: bool = True,
    ) -> str:
        obtained_information = self._get_obtained_info_text(memory)

        # Validate: Check if we have meaningful obtained information
        has_valid_data = False
        if obtained_information:
            if isinstance(obtained_information, list):
                # Check if list has meaningful content
                for item in obtained_information:
                    if isinstance(item, str) and len(item.strip()) > 10:
                        has_valid_data = True
                        break
                    elif isinstance(item, (dict, list)) and len(str(item)) > 20:
                        has_valid_data = True
                        break
            elif isinstance(obtained_information, str) and len(obtained_information.strip()) > 10:
                has_valid_data = True

        # If no valid data obtained, indicate that explicitly
        if not has_valid_data:
            return "No valid information was obtained to answer the question. The search tools could not retrieve the required data."

        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "executor", "direct_output.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        evidence_constraint = ""
        if not evidence_verified:
            evidence_constraint = (
                "\n\n⚠️ EVIDENCE GATE — INSUFFICIENT VERIFIED EVIDENCE:\n"
                "The cumulative obtained information does NOT fully satisfy the question.\n"
                "You MUST:\n"
                "1. Use ONLY facts explicitly present in Obtained Information.\n"
                "2. Do NOT infer or hallucinate missing facts (e.g., axis labels not in evidence).\n"
                "3. In Process Summary, clearly state which required evidence is still missing.\n"
                "4. In Answer, give the best supported partial answer OR state "
                "\"INSUFFICIENT_EVIDENCE\" if the question cannot be answered from evidence alone.\n"
            )

        prompt_generate_final_output = (
            evidence_constraint
            + prompt_template.format(
                Question=question,
                Obtained_Information=obtained_information,
            )
        )

        if last_verification_analysis:
            prompt_generate_final_output += (
                f"\n\nLast Verification Analysis:\n{last_verification_analysis}\n"
            )

        input_data = [prompt_generate_final_output]

        final_output = self.llm_generate_tool_command(input_data)

        return final_output