import json
import os
import re
import ast
from typing import Any, Dict, List, Tuple
from PIL import Image

import math

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import MemoryVerification, NextStep, QueryAnalysis, FinalAnswer
from MAS.epc_aw.models.memory import Memory, PlannerMemory

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
        self.memory = PlannerMemory()
        self.verbose = verbose
        self.temperature = temperature
        self.profile = ""
        self.EPS = 1e-6

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
            clean_content = self.query_analysis.strip()
            if clean_content.startswith("```"):
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_content)
                if match:
                    clean_content = match.group(1).strip()
            clean_content = clean_content.replace("\\'", "'")

            parsed = json.loads(clean_content)
            
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

    def generate_next_step(self, question: str, image: str, target_information: str, feasibility_criteria: str, step_count: int, max_step_count: int, obtained_informtion: List[str], json_data: Any = None) -> Any:
        
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "planner", "generate_next_step.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_generate_next_step = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=self.toolbox_metadata,
            Obtained_Information=obtained_informtion,
            Previous_Steps=self.memory.get_actions(),
            Epistemic_Constraint=self.memory.get_epistemic_constraint()
        )
        
        next_step = self.llm_engine_fixed(prompt_generate_next_step, temperature=self.temperature, n=self.n, response_format=NextStep)
        return next_step, prompt_generate_next_step

    def report_generation(self, question: str, target_information: str, plan_list: Dict[str, Any], toolbox_metadata: Dict[str, Any], feasibility_criteria: str, obtained_informtion: List[str],) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "planner", "report_generation.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_generate_next_step = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=self.toolbox_metadata,
            Obtained_Information=obtained_informtion,
            Previous_Steps=self.memory.get_actions(),
            Proposed_Plans=plan_list,
            Number_of_Plans=len(plan_list),
            Index_Range = len(plan_list) - 1,
            Feasibility_Criteria=feasibility_criteria
        )
            
        llm_response = self.llm_engine_fixed(prompt_generate_next_step, temperature=0, n=1, response_format=FinalAnswer)        
        try:
            parsed = json.loads(llm_response)
            scores = parsed["scores"]
            ranked_plan_indices = parsed["ranked_plan_indices"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return ranked_plan_indices, scores

    def belief_prediction(self, question: str, target_information: str, plan_list: Dict[str, Any], toolbox_metadata: Dict[str, Any], feasibility_criteria: str, agent_profile: Dict[str, Any], last_step_plan_scores: Dict[str, Any], obtained_informtion: List[str],) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "planner", "belief_prediction.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_generate_next_step = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Proposed_Plans=plan_list,
            Obtained_Information=obtained_informtion,
            Number_of_Plans=len(plan_list),
            Index_Range=len(plan_list) - 1,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=self.toolbox_metadata,
            Agent_profile=agent_profile,
            Last_Step_Plan_Scores=last_step_plan_scores,
            Feasibility_Criteria=feasibility_criteria,
            Previous_Steps=self.memory.get_actions()
        )
            
        llm_response = self.llm_engine_fixed(prompt_generate_next_step, temperature=0, n=1, response_format=FinalAnswer)        
        try:
            parsed = json.loads(llm_response)
            executor_scores = parsed["executor_beliefs"]
            diagnoser_scores = parsed["diagnoser_beliefs"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return executor_scores, diagnoser_scores


    def compute_bts_for_plans(
        self,
        planner_scores: Dict[str, float],
        executor_scores: Dict[str, float],
        diagnoser_scores: Dict[str, float],

        executor_scores_by_planner: Dict[str, float],
        diagnoser_scores_by_planner: Dict[str, float],

        planner_scores_by_executor: Dict[str, float],
        diagnoser_scores_by_executor: Dict[str, float],

        planner_scores_by_diagnoser: Dict[str, float],
        executor_scores_by_diagnoser: Dict[str, float],
    ):
        """
        Returns:
            plan_results[k] = {
                'A_bar': mean answer score,
                'bts_score': BTS score,
                'final_score': BTS score (used for ranking)
            }
        """

        plans = planner_scores.keys()
        eps = 1e-12

        plan_results = {}

        for k in plans:
            # ----- Answer reports -----
            A_P = planner_scores[k]
            A_E = executor_scores[k]
            A_D = diagnoser_scores[k]

            # ----- Prediction reports (mean over the other two agents) -----
            P_P = 0.5 * (
                executor_scores_by_planner[k] +
                diagnoser_scores_by_planner[k]
            )

            P_E = 0.5 * (
                planner_scores_by_executor[k] +
                diagnoser_scores_by_executor[k]
            )

            P_D = 0.5 * (
                planner_scores_by_diagnoser[k] +
                executor_scores_by_diagnoser[k]
            )

            # ----- BTS scores -----
            bts_P = math.log(A_P + eps) - math.log(P_P + eps)
            bts_E = math.log(A_E + eps) - math.log(P_E + eps)
            bts_D = math.log(A_D + eps) - math.log(P_D + eps)

            bts_score = (bts_P + bts_E + bts_D) / 3.0

            # ----- Mean feasibility (tie-breaker) -----
            A_bar = (A_P + A_E + A_D) / 3.0

            plan_results[k] = {
                "A_bar": A_bar,
                "bts_score": bts_score,
                "final_score": bts_score
            }

        return plan_results

