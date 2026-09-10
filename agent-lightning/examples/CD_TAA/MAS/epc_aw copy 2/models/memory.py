from typing import Dict, Any, List, Union, Optional, Tuple
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

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
    SystemMemory - 统一的因果图内存系统

    四层因果图设计:
    【在线内存 - 单任务执行轨迹】
    - Layer 1: causal_execution_traces - 因→果执行日志
    - Layer 2: task_local_parameters - 失败vs成功参数对照

    【离线内存 - 跨任务知识库】
    - Layer 1: tool_ability_boundary - 工具能力边界(能做什么、不能做什么)
    - Layer 2: tool_parameter_graphs - 参数设计模式(如何设计参数)
    """
    def __init__(self, toolbox_metadata: Dict[str, Any], agent_profile: Dict[str, Any] = None):
        super().__init__()
        self.outline: Optional[Dict[str, Any]] = None
        self.obtained_information: List[Any] = []
        # Canonical structured evidence store (obtained_information mirrors content field)
        self.evidence_records: List[Dict[str, Any]] = []
        self.toolbox_metadata = toolbox_metadata
        self.agent_profile = agent_profile
        self.last_step_plan_scores = {
            "plan_list": [],
            "planner": {},
            "executor": {},
            "diagnoser": {}
        }
        self.diagnostic_signal: Optional[Dict[str, Any]] = None

        # ============================================================================
        # 【在线内存】单任务执行轨迹
        # ============================================================================
        self.online_memory = {
            "task_id": None,

            # Layer 1: Causal Execution Traces - 因果执行日志
            # 结构: [{outline_step, factor: {state, subgoal}, effects: [...], L2_diagnosis, status}]
            "causal_execution_traces": [],

            # Layer 2: Task-Local Parameter Memory - 任务本地参数记忆
            # 结构: {(state_type, subgoal_type): {tool: {failed_parameters, successful_parameters, parameter_pairs}}}
            "task_local_parameters": {}
        }

        # ============================================================================
        # 【离线内存】跨任务知识库
        # ============================================================================
        self.offline_memory = {
            # Layer 1: Tool Ability Boundary Graphs - 工具能力边界
            # 结构: {(state_type, subgoal_type): {tool_name: {capable: bool, contexts: {success/failure}}}}
            "tool_ability_boundary": {},

            # Layer 2: Tool Parameter Graphs - 工具参数设计图
            # 结构: {(state_type, subgoal_type): {tool_name: {successful_parameters, failed_parameters}}}
            "tool_parameter_graphs": {}
        }

        # 诊断信号历史
        self.diagnostic_signal_history: List[Dict[str, Any]] = []

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

    def get_obtained_information_for_prompt(self) -> str:
        """Format evidence records for LLM prompts (structured + readable)."""
        if not self.evidence_records:
            if not self.obtained_information:
                return "None"
            return "\n".join(f"- {item}" for item in self.obtained_information)

        lines = []
        for rec in self.evidence_records:
            quality = rec.get("source_quality", "unknown")
            step = rec.get("outline_step", "?")
            content = rec.get("content", "")
            lines.append(f"- [Step {step}|{quality}] {content}")
        return "\n".join(lines)

    def add_obtained_information(self, info: Any) -> None:
        """Legacy flat append; prefer add_evidence_record for new writes."""
        if isinstance(info, dict) and "content" in info:
            self.add_evidence_record(info)
            return
        text = str(info).strip()
        if not text:
            return
        if text in self.obtained_information:
            return
        self.obtained_information.append(text)

    def add_evidence_record(
        self,
        content: str,
        *,
        outline_step: str = "",
        exec_step: int = 0,
        subgoal: str = "",
        tool: str = "",
        source_quality: str = "unknown",
        subgoal_complete: bool = True,
        confidence: float = 0.5,
    ) -> bool:
        """
        Add structured evidence. Returns False if duplicate or empty.

        source_quality: primary | secondary | inferred | unknown
        """
        text = str(content).strip()
        if not text or len(text) < 10:
            return False

        content_hash = hashlib.md5(text[:500].encode()).hexdigest()
        for rec in self.evidence_records:
            if rec.get("content_hash") == content_hash:
                return False

        record = {
            "id": f"ev_{len(self.evidence_records) + 1}",
            "outline_step": outline_step,
            "exec_step": exec_step,
            "subgoal": subgoal,
            "tool": tool,
            "content": text,
            "content_hash": content_hash,
            "source_quality": source_quality,
            "subgoal_complete": subgoal_complete,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self.evidence_records.append(record)
        self.obtained_information.append(text)
        return True

    def infer_state_type(self) -> str:
        """Derive information-state label for causal graph factor.state."""
        outline = self.get_outline() or {}
        remaining = len(outline)
        evidence_count = len(self.evidence_records) or len(self.obtained_information)

        if evidence_count == 0:
            return "初始状态_无已知信息"
        if remaining == 0:
            return "信息完备_待最终确认"
        if evidence_count == 1:
            return f"部分完成_已获首条证据_剩余{remaining}步"
        return f"部分完成_已获{evidence_count}条证据_剩余{remaining}步"

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

    # ============================================================================
    # 【在线内存 Layer 1】因果执行日志操作（因→果关系）
    # ============================================================================

    def record_causal_trace(
        self,
        step: int,
        state: str,
        subgoal: str,
        effects: List[Dict[str, Any]],
        L2_diagnosis: Optional[Dict[str, str]] = None,
        outline_step: str = "",
        status: str = "completed",
    ) -> None:
        """
        记录因果执行轨迹（完整 trace 一次性写入，兼容旧调用）。

        因：一个状态(state)下的一个需求(subgoal)
        果：为了解决这个需求，执行的多个工具调用
        """
        trace = {
            "outline_step": outline_step,
            "step": step,
            "factor": {
                "state": state,
                "subgoal": subgoal,
            },
            "effects": effects,
            "L2_diagnosis": L2_diagnosis or {},
            "status": status,
        }
        self.online_memory["causal_execution_traces"].append(trace)

    def _find_active_causal_trace(
        self,
        outline_step: str,
        subgoal: str,
    ) -> Optional[Dict[str, Any]]:
        """Find in-progress trace for the same outline step + subgoal."""
        for trace in reversed(self.online_memory["causal_execution_traces"]):
            if trace.get("status") != "in_progress":
                continue
            factor = trace.get("factor") or {}
            if trace.get("outline_step") == outline_step and factor.get("subgoal") == subgoal:
                return trace
        return None

    def append_causal_effect(
        self,
        outline_step: str,
        exec_step: int,
        state: str,
        subgoal: str,
        effect: Dict[str, Any],
        L2_diagnosis: Optional[Dict[str, Any]] = None,
        finalize: bool = False,
    ) -> None:
        """
        Append one effect (因→果) to an existing in-progress trace, or open a new one.

        Design: multiple attempts for the same subgoal share one factor node.
        """
        trace = self._find_active_causal_trace(outline_step, subgoal)
        if trace is None:
            trace = {
                "outline_step": outline_step,
                "step": exec_step,
                "factor": {"state": state, "subgoal": subgoal},
                "effects": [],
                "L2_diagnosis": {},
                "status": "in_progress",
            }
            self.online_memory["causal_execution_traces"].append(trace)

        trace["effects"].append(effect)
        trace["step"] = exec_step
        trace["factor"]["state"] = state
        if L2_diagnosis:
            trace["L2_diagnosis"] = L2_diagnosis
        if finalize:
            trace["status"] = "completed"

    def get_causal_traces(
        self,
        state: Optional[str] = None,
        subgoal: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查询因果执行日志（支持过滤）"""
        traces = self.online_memory["causal_execution_traces"]

        if state:
            traces = [t for t in traces if t["factor"]["state"] == state]
        if subgoal:
            traces = [t for t in traces if t["factor"]["subgoal"] == subgoal]

        return traces

    # ============================================================================
    # 【在线内存 Layer 2】任务本地参数记忆操作
    # ============================================================================

    def record_task_local_parameters(
        self,
        state: str,
        subgoal: str,
        tool: str,
        failed_params: Optional[List[Dict[str, Any]]] = None,
        successful_params: Optional[List[Dict[str, Any]]] = None,
        parameter_pairs: Optional[List[Dict[str, Any]]] = None,
        merge: bool = True,
    ) -> None:
        """
        记录任务本地参数记忆。

        failed_params: [{"parameter": "...", "failure_symptom": "...", "L1_diagnosis": "..."}]
        successful_params: [{"parameter": "...", "result": "..."}]
        parameter_pairs: [{"failed": "...", "successful": "...", "insight": "..."}]
        """
        key = (state, subgoal)

        if key not in self.online_memory["task_local_parameters"]:
            self.online_memory["task_local_parameters"][key] = {}

        existing = self.online_memory["task_local_parameters"][key].get(tool, {})

        if merge and tool in self.online_memory["task_local_parameters"][key]:
            merged_failed = list(existing.get("failed_parameters", []))
            merged_success = list(existing.get("successful_parameters", []))
            merged_pairs = list(existing.get("parameter_pairs", []))

            if failed_params:
                merged_failed.extend(failed_params)
            if successful_params:
                merged_success.extend(successful_params)
            if parameter_pairs:
                merged_pairs.extend(parameter_pairs)

            self.online_memory["task_local_parameters"][key][tool] = {
                "failed_parameters": merged_failed,
                "successful_parameters": merged_success,
                "parameter_pairs": merged_pairs,
            }
        else:
            self.online_memory["task_local_parameters"][key][tool] = {
                "failed_parameters": failed_params or [],
                "successful_parameters": successful_params or [],
                "parameter_pairs": parameter_pairs or [],
            }

    def get_task_local_parameters(
        self,
        state: str,
        subgoal: str,
        tool: str
    ) -> Dict[str, Any]:
        """获取特定(state,subgoal,tool)的任务本地参数记忆"""
        key = (state, subgoal)
        return self.online_memory["task_local_parameters"]\
            .get(key, {})\
            .get(tool, {})

    def get_failed_parameter_blacklist(
        self,
        state: str,
        subgoal: str,
        tool: str
    ) -> List[str]:
        """获取失败参数黑名单（供Executor规避）"""
        params = self.get_task_local_parameters(state, subgoal, tool)
        return [p["parameter"] for p in params.get("failed_parameters", [])]

    def add_failed_parameter(
        self,
        state: str,
        subgoal: str,
        tool: str,
        parameter: str,
        failure_symptom: str,
        L1_diagnosis: Optional[str] = None
    ) -> None:
        """在任务本地参数中追加失败参数（合并而非覆盖）。"""
        params = self.get_task_local_parameters(state, subgoal, tool)
        failed_list = list(params.get("failed_parameters", []))

        if not any(p.get("parameter") == parameter for p in failed_list):
            failed_list.append({
                "parameter": parameter,
                "failure_symptom": failure_symptom,
                "L1_diagnosis": L1_diagnosis or "",
            })

        self.record_task_local_parameters(
            state, subgoal, tool,
            failed_params=failed_list,
            successful_params=params.get("successful_parameters", []),
            parameter_pairs=params.get("parameter_pairs", []),
            merge=False,
        )

    def add_successful_parameter(
        self,
        state: str,
        subgoal: str,
        tool: str,
        parameter: str,
        result: Any = None,
        failed_parameter: Optional[str] = None,
    ) -> None:
        """在任务本地参数中追加成功参数，并可选记录 failed→successful 对照。"""
        params = self.get_task_local_parameters(state, subgoal, tool)
        successful_list = list(params.get("successful_parameters", []))
        pairs = list(params.get("parameter_pairs", []))

        if not any(p.get("parameter") == parameter for p in successful_list):
            successful_list.append({
                "parameter": parameter,
                "result": result,
            })

        if failed_parameter and failed_parameter != parameter:
            pair = {
                "failed": failed_parameter,
                "successful": parameter,
                "insight": "parameter_variation_succeeded",
            }
            if pair not in pairs:
                pairs.append(pair)

        self.record_task_local_parameters(
            state, subgoal, tool,
            failed_params=params.get("failed_parameters", []),
            successful_params=successful_list,
            parameter_pairs=pairs,
            merge=False,
        )

    # ============================================================================
    # 【离线内存 Layer 1】工具能力边界操作
    # ============================================================================

    def update_tool_ability_boundary(
        self,
        state: str,
        subgoal: str,
        tool: str,
        capable: bool,
        contexts: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        更新工具能力边界。

        Args:
            capable: 该工具能否解决这类问题
            contexts: {
                "success": ["场景1", "场景2"],  # 能成功的场景
                "failure": ["场景3"]             # 无法处理的场景
            }
        """
        key = (state, subgoal)

        if key not in self.offline_memory["tool_ability_boundary"]:
            self.offline_memory["tool_ability_boundary"][key] = {}

        self.offline_memory["tool_ability_boundary"][key][tool] = {
            "capable": capable,
            "contexts": contexts or {"success": [], "failure": []}
        }

    def query_tool_ability(
        self,
        state: str,
        subgoal: str
    ) -> Dict[str, bool]:
        """
        查询某(state,subgoal)下哪些工具可用。
        返回: {tool_name: capable(bool)}
        """
        key = (state, subgoal)
        result = {}

        tools_info = self.offline_memory["tool_ability_boundary"].get(key, {})
        for tool_name, info in tools_info.items():
            result[tool_name] = info.get("capable", False)

        return result

    # ============================================================================
    # 【离线内存 Layer 2】工具参数设计图操作
    # ============================================================================

    def update_tool_parameter_graphs(
        self,
        state: str,
        subgoal: str,
        tool: str,
        successful_parameters: Optional[List[Dict[str, Any]]] = None,
        failed_parameters: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        更新工具参数设计图。

        successful_parameters: [
            {
                "pattern": "参数设计模式",
                "example": "具体参数",
                "success_rate": 0.95,
                "count": 8
            }
        ]
        failed_parameters: [
            {
                "pattern": "失败模式",
                "example": "具体参数",
                "common_failure": "失败症状",
                "count": 3,
                "reason": "根因"
            }
        ]
        """
        key = (state, subgoal)

        if key not in self.offline_memory["tool_parameter_graphs"]:
            self.offline_memory["tool_parameter_graphs"][key] = {}

        self.offline_memory["tool_parameter_graphs"][key][tool] = {
            "successful_parameters": successful_parameters or [],
            "failed_parameters": failed_parameters or []
        }

    def get_tool_parameter_patterns(
        self,
        state: str,
        subgoal: str,
        tool: str
    ) -> Dict[str, Any]:
        """获取工具的参数设计模式（供Executor参考）"""
        key = (state, subgoal)
        return self.offline_memory["tool_parameter_graphs"]\
            .get(key, {})\
            .get(tool, {})

    # ============================================================================
    # 【诊断信号】操作
    # ============================================================================

    def get_diagnostic_signal(self) -> Optional[Dict[str, Any]]:
        """获取当前诊断信号"""
        return self.diagnostic_signal

    def set_diagnostic_signal(self, signal: Dict[str, Any]) -> None:
        """设置本周期的诊断信号"""
        self.diagnostic_signal = signal
        self.diagnostic_signal_history.append(signal)

    def clear_diagnostic_signal(self) -> None:
        """清除诊断信号"""
        self.diagnostic_signal = None

    def get_diagnostic_signal_history(self) -> List[Dict[str, Any]]:
        """获取诊断信号历史"""
        return self.diagnostic_signal_history

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

    # ============================================================================
    # 【知识提升】从在线→离线
    # ============================================================================

    def upgrade_online_to_offline(self) -> None:
        """
        任务完成时，从在线内存提升知识到离线内存。

        提升逻辑：
        1. 从task_local_parameters统计工具的成功/失败率 → 更新tool_ability_boundary
        2. 从task_local_parameters提升参数设计模式 → 更新tool_parameter_graphs
        """

        # 【提升1】从task_local_parameters统计 → 生成工具能力边界
        # 统计：如果工具有成功的参数，则capable=True；否则capable=False
        for (state, subgoal), tools in self.online_memory["task_local_parameters"].items():
            for tool, data in tools.items():
                successful = data.get("successful_parameters", [])

                # 判断该工具对此(state,subgoal)是否有能力
                # 标准：只要有至少1个成功的参数，就说明工具有能力
                capable = len(successful) > 0

                # 更新工具能力边界
                if capable:
                    contexts = {
                        "success": [subgoal],  # 该subgoal是成功的
                        "failure": []
                    }
                else:
                    contexts = {
                        "success": [],
                        "failure": [subgoal]   # 该subgoal在这个工具上失败了
                    }

                self.update_tool_ability_boundary(
                    state=state,
                    subgoal=subgoal,
                    tool=tool,
                    capable=capable,
                    contexts=contexts
                )

        # 【提升2】从task_local_parameters → 工具参数设计图
        # 提升：提取参数模式和建议（不是完整的command）
        for (state, subgoal), tools in self.online_memory["task_local_parameters"].items():
            for tool, data in tools.items():
                failed_params = data.get("failed_parameters", [])
                successful_params = data.get("successful_parameters", [])

                # 将参数尝试转换为参数设计图的格式
                # 成功的参数应该包含：模式描述、具体例子、建议
                upgraded_successful = self._extract_parameter_patterns(successful_params, is_success=True)
                # 失败的参数应该包含：模式描述、具体例子、为什么失败
                upgraded_failed = self._extract_parameter_patterns(failed_params, is_success=False)

                self.update_tool_parameter_graphs(
                    state=state,
                    subgoal=subgoal,
                    tool=tool,
                    successful_parameters=upgraded_successful,
                    failed_parameters=upgraded_failed
                )

    def _extract_parameter_patterns(self, params: List[Dict[str, Any]], is_success: bool) -> List[Dict[str, Any]]:
        """
        从参数列表提取参数设计模式。

        目标：提取出有用的参数模式信息，而不是完整的command字符串。
        """
        if not params:
            return []

        patterns = []
        for param_record in params:
            if isinstance(param_record, dict):
                param_str = param_record.get("parameter", "")
                symptom = param_record.get("failure_symptom", "")
            else:
                param_str = str(param_record)
                symptom = None

            if not param_str:
                continue

            # 从参数字符串中提取关键的参数值
            # 例如：从 'query="iPhone", site="apple.com"' 提取参数名和值
            pattern_dict = self._parse_parameter_string(param_str)

            if is_success:
                # 成功的参数 - 记录成功的参数值和特征
                pattern_entry = {
                    "pattern": self._describe_parameter_pattern(pattern_dict),
                    "example": param_str[:100],  # 具体例子（截断）
                    "success_rate": 1.0,  # 单个成功的参数的成功率
                    "context": f"Successfully used in this context"
                }
            else:
                # 失败的参数 - 记录失败的参数值、原因和症状
                pattern_entry = {
                    "pattern": self._describe_parameter_pattern(pattern_dict),
                    "example": param_str[:100],  # 具体例子（截断）
                    "failure_count": 1,
                    "symptom": symptom or "Unknown",
                    "reason": "Failed parameter"
                }

            patterns.append(pattern_entry)

        return patterns

    def _parse_parameter_string(self, param_str: str) -> Dict[str, Any]:
        """
        从参数字符串中提取参数名和值。
        例如：'query="iPhone", site="apple.com"' → {query: "iPhone", site: "apple.com"}
        """
        import re
        result = {}

        # 匹配 param="value" 或 param='value' 或 param=value 的模式
        pattern = r'(\w+)\s*=\s*["\']?([^",\']*)["\']?'
        matches = re.findall(pattern, param_str)

        for key, value in matches:
            result[key] = value

        return result

    def _describe_parameter_pattern(self, pattern_dict: Dict[str, Any]) -> str:
        """
        用人类可读的方式描述参数模式。
        例如：{query: "iPhone", site: "apple.com"} → "Use site parameter to limit domain; specific query term"
        """
        if not pattern_dict:
            return "Unknown parameter pattern"

        descriptions = []
        for key, value in pattern_dict.items():
            if value:
                descriptions.append(f"{key}='{value}'")
            else:
                descriptions.append(f"{key} parameter used")

        return "; ".join(descriptions) if descriptions else "Custom parameters"

    # ============================================================================
    # 【持久化】保存/加载离线内存
    # ============================================================================

    def _ensure_memory_dirs(self, memory_dir: str = "memory") -> Path:
        """确保所有必要的目录存在"""
        memory_path = Path(memory_dir).absolute()
        dirs = [
            memory_path / "offline",
            memory_path / "online",
            memory_path / "screenshots"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return memory_path

    def _serialize_graphs(self, graphs: Dict) -> Dict:
        """将元组键转为字符串（JSON兼容）"""
        serialized = {}
        for key, value in graphs.items():
            if isinstance(key, tuple):
                key_str = f"{key[0]}::{key[1]}"
            else:
                key_str = str(key)
            serialized[key_str] = value
        return serialized

    def _deserialize_graphs(self, serialized: Dict) -> Dict:
        """将字符串键转回元组"""
        deserialized = {}
        for key_str, value in serialized.items():
            if "::" in key_str:
                parts = key_str.split("::", 1)
                key_tuple = (parts[0], parts[1])
            else:
                key_tuple = tuple(key_str.split(",")) if "," in key_str else (key_str,)
            deserialized[key_tuple] = value
        return deserialized

    def persist_offline_memory(self, memory_dir: str = "memory") -> None:
        """将离线因果图持久化到本地"""
        memory_path = self._ensure_memory_dirs(memory_dir)
        offline_dir = memory_path / "offline"

        # Layer 1: Tool Ability Boundary
        boundary_file = offline_dir / "tool_ability_boundary.json"
        with open(boundary_file, "w", encoding="utf-8") as f:
            json.dump(
                self._serialize_graphs(self.offline_memory["tool_ability_boundary"]),
                f, indent=2, ensure_ascii=False
            )

        # Layer 2: Tool Parameter Graphs
        param_file = offline_dir / "tool_parameter_graphs.json"
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                self._serialize_graphs(self.offline_memory["tool_parameter_graphs"]),
                f, indent=2, ensure_ascii=False
            )

        # Manifest
        manifest = {
            "last_updated": datetime.now().isoformat(),
            "offline_memory_files": [
                "tool_ability_boundary.json",
                "tool_parameter_graphs.json"
            ]
        }
        manifest_file = memory_path / "offline_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def persist_online_memory(self, task_id: str, memory_dir: str = "memory") -> None:
        """将在线执行轨迹持久化到本地"""
        memory_path = self._ensure_memory_dirs(memory_dir)
        task_dir = memory_path / "online" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: Causal Execution Traces
        traces_file = task_dir / "causal_execution_traces.json"
        with open(traces_file, "w", encoding="utf-8") as f:
            json.dump(
                self.online_memory["causal_execution_traces"],
                f, indent=2, ensure_ascii=False
            )

        # Layer 2: Task-Local Parameters
        params_file = task_dir / "task_local_parameters.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(
                self._serialize_graphs(self.online_memory["task_local_parameters"]),
                f, indent=2, ensure_ascii=False
            )

        # Structured evidence records
        evidence_file = task_dir / "evidence_records.json"
        with open(evidence_file, "w", encoding="utf-8") as f:
            json.dump(self.evidence_records, f, indent=2, ensure_ascii=False)

        # Metadata
        metadata_file = task_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            metadata = {
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "total_steps": len(self.online_memory["causal_execution_traces"]),
                "evidence_count": len(self.evidence_records),
                "status": "completed"
            }
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_offline_memory(self, memory_dir: str = "memory") -> bool:
        """从本地加载离线因果图到内存"""
        memory_path = Path(memory_dir).absolute()
        offline_dir = memory_path / "offline"

        # Load Layer 1
        boundary_file = offline_dir / "tool_ability_boundary.json"
        if boundary_file.exists():
            try:
                with open(boundary_file, "r", encoding="utf-8") as f:
                    graphs = json.load(f)
                    self.offline_memory["tool_ability_boundary"] = \
                        self._deserialize_graphs(graphs)
            except Exception as e:
                print(f"Warning: Failed to load tool_ability_boundary: {e}")
                return False

        # Load Layer 2
        param_file = offline_dir / "tool_parameter_graphs.json"
        if param_file.exists():
            try:
                with open(param_file, "r", encoding="utf-8") as f:
                    graphs = json.load(f)
                    self.offline_memory["tool_parameter_graphs"] = \
                        self._deserialize_graphs(graphs)
            except Exception as e:
                print(f"Warning: Failed to load tool_parameter_graphs: {e}")
                return False

        return True

    def get_memory_stats(self, memory_dir: str = "memory") -> Dict[str, Any]:
        """获取内存统计信息"""
        memory_path = Path(memory_dir).absolute()

        stats = {
            "online_memory": {
                "causal_execution_traces": len(self.online_memory["causal_execution_traces"]),
                "task_local_parameters": len(self.online_memory["task_local_parameters"]),
                "evidence_records": len(self.evidence_records),
            },
            "offline_memory": {
                "tool_ability_boundary": len(self.offline_memory["tool_ability_boundary"]),
                "tool_parameter_graphs": len(self.offline_memory["tool_parameter_graphs"])
            },
            "disk_state": {
                "offline_dir_exists": (memory_path / "offline").exists(),
                "online_dir_exists": (memory_path / "online").exists(),
                "screenshots_dir_exists": (memory_path / "screenshots").exists()
            }
        }
        return stats
