from typing import Dict, Any, List, Union, Optional
import os

class Memory:

    def __init__(self):
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._init_file_types()

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, **kwargs) -> None:
        """
        Base add_action: accepts any keyword fields.
        Subclasses can override and enforce their own schema.
        """
        action = kwargs
        step_name = f"Action Step {step_count}"
        self.actions[step_name] = action

    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        return self.actions


class SystemMemory(Memory):
    """
    SystemMemory 的 Docstring
    outline: 记录解题思路，每一步骤包含所需要的目标信息，以及如何获得这些信息的计划
    agent_profile: 记录各个 agent 的角色设定和能力范围
    obtained_information: 记录在解题过程中获得的信息和数据
    last_step_plan_scores: 记录上一步骤中各个计划的评分情况
    """
    def __init__(self, toolbox_metadata: Dict[str, Any], agent_profile: Dict[str, Any] = None):
        super().__init__()
        self.outline: Optional[Dict[str, Any]] = None
        self.obtained_information: List[Any] = []
        self.toolbox_metadata = toolbox_metadata
        self.agent_profile = agent_profile
        self.last_step_plan_scores = {
            "plan_list": [],
            "planner": {},
            "executor": {},
            "diagnoser": {}
        }
    
    def set_outline(self, outline: Dict[str, Any]) -> None:
        self.outline = outline

    def get_outline(self) -> Optional[Dict[str, Any]]:
        return self.outline
    
    def set_agent_profile(self, agent_profile: Dict[str, Any]) -> None:
        self.agent_profile = agent_profile

    def get_agent_profile(self) -> Optional[Dict[str, Any]]:
        return self.agent_profile
    
    def get_obtained_information(self) -> List[Any]:
        return self.obtained_information
    
    def add_obtained_information(self, info: Any) -> None:
        self.obtained_information.append(info)

    def set_last_step_plan_scores(self, plan_list: List[str], planner_scores: Dict[str, float], executor_scores: Dict[str, float], diagnoser_scores: Dict[str, float]) -> None:
        self.last_step_plan_scores = {
            "plan_list": plan_list,
            "planner": planner_scores,
            "executor": executor_scores,
            "diagnoser": diagnoser_scores
        }

    def get_last_step_plan_scores(self) -> Dict[str, Any]:
        return self.last_step_plan_scores

    def get_toolbox_metadata(self) -> Dict[str, Any]:
        return self.toolbox_metadata

    def add_action(
        self,
        step_count: int,
        supporting_evidence: Optional[str] = None,
        **extra
    ):
        action = {
            'role': 'system',
            'supporting_evidence': supporting_evidence,
        }
        action.update(extra)
        step_name = f"Synthesizer Step {step_count}"
        self.actions[step_name] = action


class PlannerMemory(Memory):
    """
    PlannerMemory 的 Docstring
    action schema:记录planner和BTS决策一致的步骤信息
    其中包括目标(target_information)、计划内容(plan)、评分(score)、执行后的结果(result)等字段
    """
    def __init__(self):
        super().__init__()
        self.epistemic_constraint = []

    def get_epistemic_constraint(self) -> List[str]:
        return self.epistemic_constraint
    
    def add_epistemic_constraint(self, instruction: str) -> None:
        self.epistemic_constraint.append(instruction)
    
    def add_action(
        self, step_count: int,
        target_information:str,
        plan: str,
        tool_name: str,
        tool_input: Any,
        result: str,
        context_verification: Any,
        **extra
    ):
        action = {
            'target_information': target_information,
            'plan': plan,
            'tool_name': tool_name,     # 调用的工具
            'tool_input': tool_input,   # 工具输入
            'result': result, #
            'context_verification': context_verification,
        }
        action.update(extra)
        step_name = f"Planner Step {step_count}"
        self.actions[step_name] = action


class ExecutorMemory(Memory):
    """
    ExecutorMemory 的 Docstring
    action schema:记录executor执行计划的步骤信息
    其中包括目标(target_information)、计划内容(plan)、工具名称(tool_name)、工具输入(tool_input)、执行结果(result)等字段
    """
    def __init__(self):
        super().__init__()

    def add_action(
        self,
        step_count: int,
        target_information: str,
        plan: str,
        tool_name: str,
        tool_input: Any,
        context_verification: Any,
        **extra
    ):
        action = {
            'target_information': target_information,
            'plan': plan,               # executor 接收到的计划
            'tool_name': tool_name,     # 调用的工具
            'tool_input': tool_input,   # 工具输入
            'result': context_verification,
        }
        action.update(extra)
        step_name = f"Executor Step {step_count}"
        self.actions[step_name] = action


class DiagnoserMemory(Memory):
    """
    DiagnoserMemory 的 Docstring
    action schema:记录diagnoser诊断步骤的信息
    其中包括诊断结果(diagnosis)、检测到的问题(detected_issue)等字段
    """
    def __init__(self):
        super().__init__()

    def add_action(
        self,
        step_count: int,
        target_information: str,
        plan: str,
        tool_name: str,
        tool_input: Any,
        result: Any,
        context_verification: Any,
        **extra
    ):
        action = {
            'target_information': target_information,
            'plan': plan,
            'tool_name': tool_name,     # 调用的工具
            'tool_input': tool_input,   # 工具输入
            'result': result,   # 工具未整理的返回
            'context_verification': context_verification,
        }
        action.update(extra)
        step_name = f"Diagnoser Step {step_count}"
        self.actions[step_name] = action
