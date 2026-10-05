import json
import os
import re
import ast
from typing import Any, Dict, List, Tuple
from PIL import Image

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import MemoryVerification, NextStep, QueryAnalysis, FinalAnswer
from MAS.epc_aw.models.memory import Memory, DiagnoserMemory

import json
import re



def safe_strip_outer_quotes(s: str) -> str:
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return s

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s

def remove_markdown_bold(s: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=re.DOTALL)

def extract_field(text: str, label: str) -> str | None:
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Z][A-Za-z0-9 _\-]+?:|\Z)"
    m = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    pattern2 = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n\n|\Z)"
    m2 = re.search(pattern2, text, flags=re.DOTALL | re.MULTILINE)
    if m2:
        return m2.group(1).strip()
    return None

def parse_response_to_fields(response, available_tools):
    if isinstance(response, NextStep):
        return response.context.strip(), response.sub_goal.strip(), response.tool_name.strip()

    if isinstance(response, str):
        try:
            response_dict = json.loads(response)
            if isinstance(response_dict, dict):
                ctx = response_dict.get("context") or response_dict.get("Context")
                sg = response_dict.get("sub_goal") or response_dict.get("Sub-Goal") or response_dict.get("subGoal")
                tn = response_dict.get("tool_name") or response_dict.get("Tool Name") or response_dict.get("toolName")
                if ctx or sg or tn:
                    return (ctx or "").strip(), (sg or "").strip(), (tn or "").strip()
        except Exception:
            pass

        text = safe_strip_outer_quotes(response)
        text = remove_markdown_bold(text)
        text = normalize_whitespace(text)

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



class Diagnoser:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, 
    verbose: bool = False, is_multimodal: bool = False, check_model: bool = True, temperature : float = .0, n: int =1):
        self.llm_engine_name = llm_engine_name
        self.is_multimodal = is_multimodal
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_name, is_multimodal=False)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.memory = DiagnoserMemory()
        self.verbose = verbose
        self.n = n    
        self.profile = ""
        self.temperature = temperature

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


    def report_generation(self, question: str, target_information: str, plan_list: Dict[str, Any], toolbox_metadata: Dict[str, Any], feasibility_criteria: str, obtained_informtion: List[str]) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "report_generation.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_report_generation = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Available_Tools=self.available_tools,
            Toolbox_Metadata=toolbox_metadata,
            Previous_Steps=self.memory.get_actions(),
            Proposed_Plans=plan_list,
            Obtained_Information_So_Far=obtained_informtion,
            Number_of_Plans=len(plan_list),
            Index_Range=range(len(plan_list)),
            Feasibility_and_Scoring_Criteria=feasibility_criteria
        )
            
        llm_response = self.llm_engine_fixed(prompt_report_generation, temperature=0, n=1, response_format=FinalAnswer)        
        try:
            parsed = json.loads(llm_response)
            scores = parsed["scores"]
            ranked_plan_indices = parsed["ranked_plan_indices"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return ranked_plan_indices, scores

    def belief_prediction(self, question: str, target_information: str, plan_list: Dict[str, Any], toolbox_metadata: Dict[str, Any], feasibility_criteria: str, agent_profile: Dict[str, Any], last_step_plan_scores: Dict[str, Any], obtained_informtion: List[str]) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "belief_prediction.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_report_generation = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Proposed_Plans=plan_list,
            Number_of_Plans=len(plan_list),
            Index_Range=range(len(plan_list)),
            Toolbox_Metadata=toolbox_metadata,
            Agent_Profile=agent_profile,
            Last_Step_Plan_Scores=last_step_plan_scores,
            Obtained_Information_So_Far=obtained_informtion,
            Feasibility_Criteria=feasibility_criteria,
            Previous_Steps=self.memory.get_actions()
        )

        llm_response = self.llm_engine_fixed(prompt_report_generation, temperature=0, n=1, response_format=FinalAnswer)        
        try:
            parsed = json.loads(llm_response)
            planner_scores = parsed["planner_beliefs"]
            executor_scores = parsed["executor_beliefs"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return planner_scores, executor_scores

    def verificate_context(self, question: str, image: str, target_information: str, outline: Dict[str, Any], result_executor: str, step_count: int = 0, obtained_information: Any = None) -> Any:
        image_info = self.get_image_info(image)
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "verificate_context.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_verificate_context = prompt_template.format(
            Question=question,
            Target_Information=target_information,
            Outline=outline,
            Result_Executor=result_executor,
            Memory=self.memory.get_actions(),
            Obtained_Information_So_Far=obtained_information
        )
        
        input_data = [prompt_verificate_context]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_fixed(input_data, response_format=MemoryVerification)
        try:
            parsed = json.loads(llm_response)
            Analysis = parsed["Analysis"]
            Obtained_Information_Flag = parsed["New_Obtained_Information_Flag"]
            Obtained_Information = parsed["New_Obtained_Information"]
            Conclusion = parsed["Conclusion"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        if Obtained_Information_Flag == True or "true" in str(Obtained_Information_Flag).lower():
            Obtained_Information_Flag = True
        else:
            Obtained_Information_Flag = False
        return Analysis, Obtained_Information_Flag, Obtained_Information, Conclusion

    def update_outline(self, question: str, context_verification: str, target_information: str, outline: Dict[str, Any], result_executor: str, step_count: int = 0, obtained_information: Any = None, epc_aw_analysis=None, toolbox_metadata=None) -> Any:
        
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "update_outline.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_update_outline = prompt_template.format(
            Question=question,
            Current_Verification_Analysis=context_verification,
            Target_Information=target_information,
            Outline=outline,
            Result_Executor=result_executor,
            Memory=self.memory.get_actions(),
            Obtained_Information_So_Far=obtained_information,
            EPC_AW_Analysis=epc_aw_analysis if epc_aw_analysis else "N/A",
            Toolbox_Metadata=toolbox_metadata if toolbox_metadata else "N/A"
        )

        input_data = [prompt_update_outline]
        
        llm_response = self.llm_engine_fixed(input_data, response_format=MemoryVerification)
        try:
            parsed = json.loads(llm_response)
            updated_outline = parsed["ExecutionOutline"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return updated_outline

    def epc_aw_diagnosis(self, planner_selected_plan: str, BTS_selected_plan: str, target_information: str, outline: Dict[str, Any], result_executor: str, step_count: int = 0, json_data: Any = None) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "epc_aw_diagnosis.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_epc_aw_diagnosis = prompt_template.format(
            Planner_Plan=planner_selected_plan,
            BTS_Plan=BTS_selected_plan,
            Target_Information=target_information,
            Executor_Result=result_executor,
            Verified_Memory=self.memory.get_actions()
        )

        input_data = [prompt_epc_aw_diagnosis]

        llm_response = self.llm_engine_fixed(input_data, response_format=MemoryVerification)

        try:
            parsed = json.loads(llm_response)
            epc_aw_analysis = parsed["epc_aw_analysis"]
            epistemic_constraint = parsed["epistemic_constraint"]
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e
        
        return epc_aw_analysis, epistemic_constraint

    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, bytes):
            try:
                response = response.decode('utf-8', errors='ignore')
            except Exception:
                response = response.decode(errors='ignore')

        if isinstance(response, dict):
            analysis = response.get('analysis', '')
            stop_field = response.get('stop_signal', None)
            if stop_field is None:
                stop_field = response.get('stop', None)
            if stop_field is None:
                stop_field = response.get('conclusion', None)

            if isinstance(stop_field, bool):
                return (analysis, 'STOP' if stop_field else 'CONTINUE')
            if isinstance(stop_field, str):
                sf = stop_field.strip().upper()
                if sf in ('STOP', 'CONTINUE'):
                    return (analysis, sf)

            # if no definitive stop_field, fallthrough to text parsing of analysis (or full response)
            response_text = analysis if analysis else json.dumps(response, ensure_ascii=False)

        else:
            # If object with attributes like .analysis / .stop_signal / .conclusion
            if hasattr(response, 'analysis') or hasattr(response, 'stop_signal') or hasattr(response, 'conclusion'):
                analysis = getattr(response, 'analysis', '') or ''
                stop_field = getattr(response, 'stop_signal', None)
                if stop_field is None:
                    stop_field = getattr(response, 'stop', None)
                if stop_field is None:
                    stop_field = getattr(response, 'conclusion', None)

                if isinstance(stop_field, bool):
                    return (analysis, 'STOP' if stop_field else 'CONTINUE')
                if isinstance(stop_field, str):
                    sf = stop_field.strip().upper()
                    if sf in ('STOP', 'CONTINUE'):
                        return (analysis, sf)

                # fallthrough to text parsing using analysis if present
                response_text = analysis if analysis else str(response)
            else:
                response_text = str(response)

        response_text = response_text.strip()
        if len(response_text) >= 2 and ((response_text[0] == response_text[-1] == "'") or (response_text[0] == response_text[-1] == '"')):
            response_text = response_text[1:-1].strip()

        conclusion_pattern = re.compile(r'(?:^|\n)\s*conclusion\s*[:\-]?\s*(STOP|CONTINUE)\b', re.IGNORECASE)
        m = conclusion_pattern.search(response_text)
        if m:
            conclusion = m.group(1).upper()
            analysis_text = response_text[:m.start()].strip()
            if not analysis_text:
                parts = conclusion_pattern.split(response_text)
                if parts:
                    analysis_text = parts[0].strip()
                else:
                    analysis_text = response_text.strip()
            return analysis_text, conclusion

        generic_pattern = re.compile(r'\b(STOP|CONTINUE)\b', re.IGNORECASE)
        all_matches = list(generic_pattern.finditer(response_text))
        if all_matches:
            last = all_matches[-1]
            conclusion = last.group(1).upper()
            analysis_text = response_text[:last.start()].strip()
            if not analysis_text:
                analysis_text = response_text.strip()
            return analysis_text, conclusion

        low = response_text.lower()
        if 'stop' in low:
            return response_text.strip(), 'STOP'
        if 'continue' in low:
            return response_text.strip(), 'CONTINUE'

        return response_text.strip(), 'CONTINUE'
    
    