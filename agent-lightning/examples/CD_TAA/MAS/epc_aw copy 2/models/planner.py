import json
import os
import re
import ast
from typing import Any, Dict, List, Tuple
from PIL import Image

import math

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import MemoryVerification, NextStep, QueryAnalysis, FinalAnswer
from MAS.epc_aw.models.memory import Memory
from MAS.epc_aw.models.utils import parse_json_from_llm_response
# Day 2 Integration: Import causal reasoning modules
from MAS.epc_aw.models import CausalInference, CausalGraph, HistoryAnalyzer

import json
import re



def safe_strip_outer_quotes(s: str) -> str:
    # 去掉外层单引号/双引号（常见的包裹形式）以及智能引号
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    # 智能引号
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return s

def normalize_whitespace(s: str) -> str:
    # 统一换行、去掉重复空行（保留单个空行作为分段）
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s

def remove_markdown_bold(s: str) -> str:
    # 把 **text** -> text
    return re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=re.DOTALL)

def extract_field(text: str, label: str) -> str | None:
    """
    提取格式为 'Label: ...' 的段落，结束条件是遇到下一个以大写字母开头并以 ':' 结尾的标题行，或文本末尾。
    """
    # 允许可变的空格与可选的冒号/拼写变体
    # 主要 lookahead: 下一行是形如 "Word...:" 或到字符串结尾
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Z][A-Za-z0-9 _\-]+?:|\Z)"
    m = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # 备用更宽松匹配（到下一双换行或末尾）
    pattern2 = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n\n|\Z)"
    m2 = re.search(pattern2, text, flags=re.DOTALL | re.MULTILINE)
    if m2:
        return m2.group(1).strip()
    return None

# ----------- 解析主流程 -----------
def parse_response_to_fields(response, available_tools):
    # 如果是对象（例如已经解析成 NextStep），直接取属性
    if isinstance(response, NextStep):
        return response.context.strip(), response.sub_goal.strip(), response.tool_name.strip()

    # 如果是字符串，先尝试 JSON（兼容旧行为）
    if isinstance(response, str):
        # 先尝试 json
        try:
            response_dict = json.loads(response)
            # 假设 JSON 有 fields 对应的 key
            if isinstance(response_dict, dict):
                ctx = response_dict.get("context") or response_dict.get("Context")
                sg = response_dict.get("sub_goal") or response_dict.get("Sub-Goal") or response_dict.get("subGoal")
                tn = response_dict.get("tool_name") or response_dict.get("Tool Name") or response_dict.get("toolName")
                if ctx or sg or tn:
                    return (ctx or "").strip(), (sg or "").strip(), (tn or "").strip()
        except Exception:
            # 不是 JSON，继续走文本解析
            pass

        # 预处理文本
        text = safe_strip_outer_quotes(response)
        text = remove_markdown_bold(text)
        text = normalize_whitespace(text)

        # 提取字段（单独提取更稳健）
        context = extract_field(text, "Context")
        sub_goal = extract_field(text, "Sub-Goal") or extract_field(text, "Sub Goal") or extract_field(text, "Subgoal")
        tool_name = extract_field(text, "Tool Name") or extract_field(text, "ToolName") or extract_field(text, "Tool")

        
        if not tool_name:
            for t in available_tools:
                if t in text:
                    tool_name = t
                    break

        if not tool_name:
            lowered = text.lower()
            for t in available_tools:
                if t.lower() in lowered:
                    tool_name = t
                    break

        
        if not (context or sub_goal or tool_name):
            
            raise ValueError("无法从 response 中解析出 Context / Sub-Goal / Tool Name；请检查输入格式。 原始文本片段（前500字符）：\n" + text[:500])

        
        return (context or "").strip(), (sub_goal or "").strip(), (tool_name or "").strip()

    else:
        raise TypeError("response 类型不是 str 或 NextStep，无法解析。")



class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None,
    verbose: bool = False, is_multimodal: bool = False, check_model: bool = True, temperature : float = .0, n: int =1):
        self.llm_engine_name = llm_engine_name
        self.is_multimodal = is_multimodal
        self.n = n
        # self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, temperature = temperature)
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_name, is_multimodal=False)
        # self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, temperature=temperature, n=n)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        # self.memory = PlannerMemory()  # 【已删除】使用SystemMemory替代
        self.verbose = verbose
        self.temperature = temperature
        self.profile = ""
        self.EPS = 1e-6

        # Day 2 Integration: Initialize causal reasoning system
        self.causal_graph = CausalGraph()
        self.history_analyzer = HistoryAnalyzer()
        self.causal_inference = CausalInference(self.causal_graph, self.history_analyzer)

    def get_profile(self) -> str:
        return self.profile

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = self.get_image_info(image)
         
        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")


        # print("Input data of `generate_base_response()`: ", input_data)
        # self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)
        self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "planner", "analyze_query.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        query_prompt = prompt_template.format(
            Question=question,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=self.toolbox_metadata
        )
        
        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")
        
        # print("Input data of `analyze_query()`: ", input_data)

        self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)
        try:
            parsed = parse_json_from_llm_response(self.query_analysis)
            
            analysis = parsed.get("Analysis", "")
            execution_outline = parsed.get("ExecutionOutline", {})
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {self.query_analysis}") from e
        return analysis, execution_outline

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:

        def normalize_tool_name(tool_name: str) -> str:
            """
            Normalizes a tool name robustly using regular expressions.
            It handles any combination of spaces and underscores as separators.
            """
            def to_canonical(name: str) -> str:
                # Split the name by any sequence of one or more spaces or underscores
                parts = re.split('[ _]+', name)
                # Join the parts with a single underscore and convert to lowercase
                return "_".join(part.lower() for part in parts)

            normalized_input = to_canonical(tool_name)
            
            for tool in self.available_tools:
                if to_canonical(tool) == normalized_input:
                    return tool
                    
            return f"No matched tool given: {tool_name}"
        
        
        try:
            context, sub_goal, tool_name = parse_response_to_fields(response, self.available_tools)
        except Exception as e:
            print("解析 response 失败：", str(e))
            context, sub_goal, tool_name = "", "", ""
        context_split = context.split(", ")
        if "https:" in context_split or "web" in tool_name.lower():
            tool_name = "Web_Search_Tool" 
        elif "wiki" in tool_name.lower():
            tool_name = "Wikipedia_Search_Tool"
        elif "google" in tool_name.lower() or "search" in tool_name.lower():
            tool_name = "Google_Search_Tool"
        tool_name = normalize_tool_name(tool_name)

        return context, sub_goal, tool_name

    def generate_next_step(self, question: str, image: str, target_information: str, step_count: int, max_step_count: int, obtained_informtion: List[str], json_data: Any = None, diagnostic_signal: Dict[str, Any] = None) -> Any:
        """
        Generate a single optimal next step (tool selection) for the current sub-goal.
        Incorporates diagnostic signals from the previous cycle if available.

        Args:
            diagnostic_signal: Optional signal from previous cycle containing:
                - recommendation: 'retry_with_different_parameters' or 'switch_tool'
                - tool: name of tool that failed
                - reason: why it failed
                - failure_patterns: types of failures detected
        """
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "planner", "generate_next_step.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 构建诊断信号提示（如果有 switch_tool 建议）
        diagnostic_hint = ""
        outline_tool = self._extract_tool_from_target(target_information)

        if diagnostic_signal and diagnostic_signal.get("recommendation") == "switch_tool":
            failed_tool = diagnostic_signal.get("tool", "Unknown")
            failure_reason = diagnostic_signal.get("reason", "Unknown")
            suggested = diagnostic_signal.get("suggested_tool") or outline_tool
            web_url = diagnostic_signal.get("web_search_url", "")
            alternatives = [t for t in self.available_tools if t != failed_tool and t != "Base_Generator_Tool"]
            if suggested and suggested in self.available_tools and suggested != failed_tool:
                alternatives_hint = f"Strongly prefer: {suggested}"
            else:
                alternatives_hint = ", ".join(alternatives[:4])

            url_hint = ""
            if web_url and "Web_Search_Tool" in self.available_tools:
                url_hint = f"\nFor USGS data, use Web_Search_Tool with URL: {web_url}\n"

            diagnostic_hint = f"""
⚠️ CRITICAL FEEDBACK FROM PREVIOUS CYCLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous Tool: {failed_tool}
Failure Reason: {failure_reason}
Recommendation: SWITCH TO A DIFFERENT TOOL

⚠️ REQUIREMENT: You MUST select a tool DIFFERENT from {failed_tool}
⚠️ DO NOT use Base_Generator_Tool for factual/database retrieval tasks.

Suggested alternatives: {alternatives_hint}
Outline suggested tool: {outline_tool or "N/A"}
{url_hint}
The previous tool did not work. Do NOT select {failed_tool} again.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        elif diagnostic_signal and diagnostic_signal.get("recommendation") == "decompose_goal":
            diagnostic_hint = f"""
⚠️ GOAL DECOMPOSITION REQUIRED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The current sub-goal is too difficult as a single step.
The outline has been decomposed into smaller steps.

Choose a tool for the NEW FIRST STEP in the updated outline only.
Focus on the smallest achievable sub-task.
Failed tools so far: {diagnostic_signal.get("failed_tools", [])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        elif diagnostic_signal and diagnostic_signal.get("recommendation") == "modify_state":
            state_action = diagnostic_signal.get("state_action", "Fix environment blocker")
            diagnostic_hint = f"""
⚠️ STATE / PREREQUISITE ISSUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state_action}

Try a different approach: alternative tool, simpler query, or prerequisite step.
Avoid repeating the same failing tool call pattern.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        elif outline_tool:
            diagnostic_hint = f"""
📋 OUTLINE TOOL GUIDANCE:
The current plan step specifies Operation Details using: {outline_tool}
You SHOULD use {outline_tool} unless you have strong evidence another tool is better.

"""

        prompt_generate_next_step = diagnostic_hint + prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=self.toolbox_metadata,
            Obtained_Information=obtained_informtion,
            Previous_Steps={},
            Epistemic_Constraint=""
        )

        # Generate single next step (no longer multiple candidates)
        next_step = self.llm_engine_fixed(prompt_generate_next_step, temperature=self.temperature, n=1, response_format=NextStep)

        # next_step is now a single NextStep object, not a list
        # Return it directly along with the prompt
        return next_step, prompt_generate_next_step

    @staticmethod
    def _extract_tool_from_target(target_information: str) -> str | None:
        """Extract tool name from outline Target Information Operation Details."""
        if not target_information:
            return None
        match = re.search(
            r"(Google_Search_Tool|Wikipedia_Search_Tool|Web_Search_Tool|"
            r"Python_Coder_Tool|Base_Generator_Tool|Screenshot_Tool|Vision_OCR_Tool)",
            target_information,
        )
        return match.group(1) if match else None

    # Legacy causal inference methods removed - simplified to single-step planning

