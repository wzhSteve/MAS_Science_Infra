# agentflow/tools/python_coder/tool.py

import os
import re
import sys
from io import StringIO
import contextlib

import threading
from MAS.epc_aw.tools.base import BaseTool
from MAS.epc_aw.engine.factory import create_llm_engine

import signal
from contextlib import contextmanager

import platform

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Python_Coder_Tool"

def is_windows_os():
    system=platform.system()
    return system == 'Windows'

# Custom exception for code execution timeout
class TimeoutException(Exception):
    pass

# Custom context manager for code execution timeout
@contextmanager
def timeout(seconds):
    
    if is_windows_os():
        # Windows timeout using threading.Timer
        def raise_timeout():
            raise TimeoutException("Code execution timed out")
        timer = threading.Timer(seconds, raise_timeout)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            
    else:
        def timeout_handler(signum, frame):
            raise TimeoutException("Code execution timed out")

        # Set the timeout handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the original handler and disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Focused on Python computation: arithmetic, algebraic, statistical, date,
   string, and counting operations needed to derive a final numeric/text answer.
2. Allowed imports (whitelist): standard library modules `math`, `statistics`,
   `datetime`, `re`, `collections`, `itertools`, `json`, `decimal`, `fractions`,
   plus third-party `numpy` and `sympy` (for computer-algebra / logic-equivalence
   tasks). Any other import is forbidden.
3. No network access, no file system writes, no subprocess, no OS calls.
4. Input must be provided directly in the query string (numbers, lists, formulas).
5. Output should be a single final result or a small list/tuple of results.
6. Avoid large loop output that floods stdout; print only the final answer.
7. Execution is bounded by a 30-second timeout.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide a clear, specific description of the calculation, including all
   numerical inputs directly in the query.
2. Use the whitelist stdlib modules when needed: `math` for ceil/floor/log,
   `statistics` for aggregates, `datetime` for date arithmetic, `re` for
   pattern matching / counting matches, `collections.Counter` for tallies.
3. Use `numpy` for vectorized arithmetic and `sympy` for symbolic / logic
   equivalence checks (e.g. De Morgan, contrapositive, tautology testing).
4. For counting-over-pages tasks, pair with Browser_Tool which feeds the
   per-page counts into the query string.
5. Print only the final answer with a short descriptive label.
6. Keep tasks self-contained in one snippet; do not rely on external state.
"""

class Python_Coder_Tool(BaseTool):
    require_llm_engine = True
    def __init__(self, model_string=os.getenv("MODEL_Name")):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available.",
            tool_version="1.0.0",
            input_types={
                "query": "str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs."},
            output_type="dict - A dictionary containing the generated code, calculation result, and any error messages.",
            demo_commands=[
                # {
                #     "command": 'execution = tool.execute(query="Calculate the factorial of 5")',
                #     "description": "Generate a Python code snippet to calculate the factorial of 5."
                # },
                {
                    "command": 'execution = tool.execute(query="Find the sum of prime numbers up to 50")',
                    "description": "Generate a Python code snippet to find the sum of prime numbers up to 50."
                },
                {
                    "command": 'query="Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers"\nexecution = tool.execute(query=query)',
                    "description": "Generate a Python function for a specific mathematical operation on a given list of numbers."
                },
            ],
            user_metadata = {
                "limitations": LIMITATION,
                "best_practices": BEST_PRACTICE
            }
        )
        # print(f"Initializing Python_Coder_Tool with model_string: {model_string}")
        # self.llm_engine = create_llm_engine(model_string=model_string, is_multimodal=False, base_url=base_url) if model_string else None

        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=model_string, 
            is_multimodal=False, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            ) if model_string else None

    @staticmethod
    def preprocess_code(code):
        """
        Preprocesses the generated code snippet by extracting it from the response.
        Returns only the first Python code block found.

        Parameters:
            code (str): The response containing the code snippet.

        Returns:
            str: The extracted code snippet from the first Python block.
            
        Raises:
            ValueError: If no Python code block is found.
        """
        # Look for the first occurrence of a Python code block
        match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if not match:
            raise ValueError("No Python code block found in the response")
        return match.group(1).strip()

    def truncate_string(self, text, max_length):
        """
        Truncates a string using middle truncation if it exceeds max_length.

        Parameters:
            text (str): The text to truncate
            max_length (int): Maximum allowed length

        Returns:
            str: Truncated text with middle omission if needed
        """
        if len(text) <= max_length:
            return text

        # Keep first and last portions
        head_size = max_length // 2 - 50  # Leave room for truncation message
        tail_size = max_length // 2 - 50

        return (
            text[:head_size] +
            " ... (truncated: middle content omitted) ... " +
            text[-tail_size:]
        )
    
    def safe_repr(self, obj, max_length=2000):
        """
        Safely represent a variable with truncation for large objects.

        Parameters:
            obj: The object to represent
            max_length (int): Maximum length for representation

        Returns:
            str: Safe string representation of the object
        """
        try:
            # Handle special cases that can be extremely verbose
            import types

            # Skip function objects, modules, classes
            if isinstance(obj, (types.FunctionType, types.ModuleType, type)):
                return f"<{type(obj).__name__}: {getattr(obj, '__name__', 'unnamed')}>"

            # Handle itertools and other iterator objects
            if hasattr(obj, '__iter__') and hasattr(obj, '__next__'):
                return f"<iterator: {type(obj).__name__}>"

            # Convert to string and truncate if needed
            obj_str = str(obj)
            return self.truncate_string(obj_str, max_length)

        except Exception as e:
            return f"<repr error: {type(obj).__name__}>"

    @contextlib.contextmanager
    def capture_output(self):
        """
        Context manager to capture the standard output.

        Yields:
            StringIO: The captured output.
        """
        new_out = StringIO()
        old_out = sys.stdout
        sys.stdout = new_out
        try:
            yield sys.stdout
        finally:
            sys.stdout = old_out

    def execute_code_snippet(self, code, max_head_tail=3500, max_var_length=2000, max_vars=20):
        """
        Executes the given Python code snippet.

        Parameters:
            code (str): The Python code snippet to be executed.
            max_head_tail (int): Maximum length for printed output before truncation
            max_var_length (int): Maximum length for each variable representation
            max_vars (int): Maximum number of variables to include in output

        Returns:
            dict: A dictionary containing the printed output and local variables.
        """
        # Check for dangerous functions and remove them
        dangerous_functions = ['exit', 'quit', 'sys.exit']
        for func in dangerous_functions:
            if func in code:
                print(f"Warning: Removing unsafe '{func}' call from code")
                # Use regex to remove function calls with any arguments
                code = re.sub(rf'{func}\s*\([^)]*\)', 'break', code)

        # Runtime import whitelist enforcement: only allow the stdlib and
        # numeric/symbolic modules declared in LIMITATION/BEST_PRACTICE.
        allowed_modules = {
            "math", "statistics", "datetime", "re", "collections", "itertools",
            "json", "decimal", "fractions", "numpy", "sympy",
        }
        for m in re.finditer(r"^\s*(?:import|from)\s+(\w+)", code, re.MULTILINE):
            mod = m.group(1)
            if mod not in allowed_modules:
                return {"error": f"Import of '{mod}' is not in the whitelist. Allowed: {sorted(allowed_modules)}"}

        try:
            execution_code = self.preprocess_code(code)

            # Re-scan after preprocessing (code block extraction may change text)
            for m in re.finditer(r"^\s*(?:import|from)\s+(\w+)", execution_code, re.MULTILINE):
                mod = m.group(1)
                if mod not in allowed_modules:
                    return {"error": f"Import of '{mod}' is not in the whitelist. Allowed: {sorted(allowed_modules)}"}

            # Execute with 30-second timeout (raised from 10 for numpy/sympy/CAS tasks)
            with timeout(30):
                try:
                    exec(execution_code)
                except TimeoutException:
                    print("Error: Code execution exceeded 30 seconds timeout")
                    return {"error": "Execution timed out after 30 seconds"}
                except Exception as e:
                    print(f"Error executing code: {e}")
                    return {"error": str(e)}
                
            # Capture the output and local variables
            local_vars = {}
            with self.capture_output() as output:
                exec(execution_code, {}, local_vars)
            raw_output = output.getvalue().strip()

            # Truncate printed output using middle truncation
            printed_output = self.truncate_string(raw_output, max_head_tail)

            # Filter and safely represent variables
            """
            Only the variables used in the code are returned,
            excluding built-in variables (which start with '__') and imported modules.
            All variables are safely represented with truncation.
            """
            used_vars = {}
            var_count = 0
            for k, v in local_vars.items():
                if not k.startswith('__') and not isinstance(v, type(sys)):
                    if var_count >= max_vars:
                        used_vars["__truncated__"] = f"... ({len(local_vars) - var_count} more variables omitted)"
                        break
                    # Safely represent the variable with truncation
                    used_vars[k] = self.safe_repr(v, max_var_length)
                    var_count += 1

            return {"printed_output": printed_output, "variables": used_vars, "execution_code":execution_code}
        
        except Exception as e:
            print(f"Error executing code: {e}")
            return {"error": str(e)}

    def execute(self, query):
        """
        Generates and executes Python code based on the provided query.

        Parameters:
            query (str): A query describing the desired operation.

        Returns:
            dict: A dictionary containing the executed output, local variables, or any error message.
        """

        if not self.llm_engine:
            raise ValueError("LLM engine not initialized. Please provide a valid model_string when initializing the tool.")

        task_description = """
        Given a query, generate a Python code snippet that performs the specified computation on the provided data. Think step by step. You may import from the whitelist: standard library `math`, `statistics`, `datetime`, `re`, `collections`, `itertools`, `json`, `decimal`, `fractions`, and third-party `numpy`, `sympy`. Do NOT import any other module. No network, no file writes, no subprocess. Print only the final result with a short descriptive label. The final output must be presented in the following format:

        ```python
        <code snippet>
        ```
        """
        task_description = task_description.strip()
        full_prompt = f"Task:\n{task_description}\n\nQuery:\n{query}"

        response = self.llm_engine(full_prompt)
        result_or_error = self.execute_code_snippet(response)
        return result_or_error

    def get_metadata(self):
        """
        Returns the metadata for the Python_Coder_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        metadata["require_llm_engine"] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/python_coder
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Python_Coder_Tool
    # tool = Python_Coder_Tool()

    # tool = Python_Coder_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Python_Coder_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    tool = Python_Coder_Tool(model_string=os.getenv("MODEL_Name")) # NOTE: weak Qwen2.5-7B model for tool
    # tool = Python_Coder_Tool(model_string="together-Qwen/Qwen2.5-7B-Instruct") # NOTE: weak Qwen2.5-7B model for tool

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Sample query for generating and executing Python code
    queries = [
        # "Given the number list: [1, 2, 3, 4, 5], calculate the sum of all the numbers in the list.",
        # "Print numbers from 1 to 1000 in a loop to test output truncation",
        "Create a list variable containing all numbers from 1 to 5000 to test variable truncation",
    ]
    for query in queries:
        print(f"\n###Query: {query}")
        # Execute the tool with the sample query
        try:
            execution = tool.execute(query=query)
            print("\n###Execution Result:", execution)
        except ValueError as e:
            print(f"Execution failed: {e}")

    print("Done!")
