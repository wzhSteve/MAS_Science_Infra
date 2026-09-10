import importlib
import json
import os
import re
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional


from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import ToolCommand, FinalAnswer
from MAS.epc_aw.models.memory import Memory
from MAS.epc_aw.models.utils import parse_json_from_llm_response
# Day 3 Integration: Import causal reasoning modules
from MAS.epc_aw.models import CausalInference, HistoryAnalyzer

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

    
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None, diagnostic_signal: Dict[str, Any] = None) -> Any:
        """
        Generate tool command with support for diagnostic signals from previous cycle.
        If diagnostic signal contains parameter guidance, incorporate it into the prompt prominently.
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

                # ✅ CRITICAL PREFIX: Placed BEFORE regular prompt to influence LLM strongly
                diagnostic_hint_prefix = f"""
================================================================================
🚨 IMPORTANT - PARAMETER VARIATION REQUIRED 🚨
================================================================================

PREVIOUS ATTEMPT FAILED - You MUST generate DIFFERENT parameters this time!

Reason: {explanation}

CRITICAL: DO NOT generate the same parameter again!
❌ FORBIDDEN parameter value: {failed_parameter}
{avoid_exact_match}

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

        # Construct the full prompt
        prompt_generate_tool_command = diagnostic_hint_prefix + prompt_template.format(
            Question=question,
            Sub_Goal=sub_goal,
            Tool_Name=tool_name,
            Tool_Metadata=tool_metadata,
            Relevant_Data=context
        ) + diagnostic_hint_suffix

        tool_command = self.llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command
    
    
        
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

    def generate_final_output(self, question: str, memory: Memory) -> str:
        prompt_generate_final_output = f"""
            Task: Generate the final output based on the query and the results from all tools used.

            Context:
            - **Query:** {question}
            - **Actions Taken:** {self._format_history_actions(memory)}
            - **Obtained Information:** {memory.get_obtained_information()}

            Instructions:
            1. Review the query and the results from all tool executions.
            2. Incorporate the relevant information to create a coherent, step-by-step final output.
            """

        input_data = [prompt_generate_final_output]
        
        final_output = self.llm_generate_tool_command(input_data)

        return final_output


    def generate_direct_output(self, question: str, last_verification_analysis: str, memory: Memory) -> str:
        obtained_information = memory.get_obtained_information()

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

        prompt_generate_final_output = prompt_template.format(
            Question=question,
            Obtained_Information=obtained_information,
            History_Actions=self._format_history_actions(memory),
        )

        input_data = [prompt_generate_final_output]

        final_output = self.llm_generate_tool_command(input_data)

        return final_output