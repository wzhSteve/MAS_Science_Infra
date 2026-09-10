from typing import Dict, Any, List, Union, Optional, Tuple
import os
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime

from .causal_memory_graph import (
    CausalMemoryGraph,
    abstract_subgoal_type,
    abstract_state_type,
    parse_parameter_command,
)
from .tool_knowledge_memory import (
    ToolCapabilityMemory,
    ToolInvocationMemory,
    MemoryRefiner,
    KeywordMemoryRetriever,
    LLMMemoryRetriever,
    MemoryRetriever,
    MEMORY_SCHEMA_VERSION,
)

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

    【离线内存 - Tool Knowledge Memory，跨任务抽象知识】
    - tool_capability (ToolCapabilityMemory) → Planner：选哪个工具 (Capability)
      {tool_name: {capability_summary, subgoals: [{subgoal, context_summary}]}}
      context_summary = Applicability Conditions（在何种条件下应选此工具）
    - tool_invocation (ToolInvocationMemory) → Executor：如何调用（Planner 不访问）
      {tool_name: {subgoals: [{subgoal, factors: {DecisionDimension: {instruction}}}]}}
      factors = Decision Dimensions（影响参数构造的正交维度），每维恰好一条 instruction

    纯文本抽象知识，不存 scores/counts/confidence/embeddings/metadata/concrete examples。
    schema 保持极简，禁止随意新增字段（Memory Compression 是核心卖点）。
    通过 evolve() 持续 retrieve→merge→generalize→rewrite，保持规模近似恒定。
    """
    def __init__(self, toolbox_metadata: Dict[str, Any], agent_profile: Dict[str, Any] = None):
        super().__init__()
        self.outline: Optional[Dict[str, Any]] = None
        self.obtained_information: List[Any] = []
        # Canonical structured evidence store (obtained_information mirrors content field)
        self.evidence_records: List[Dict[str, Any]] = []
        self.task_profile: Optional[Any] = None  # TaskProfile set at query analysis
        self.task_progress: Dict[str, Any] = {
            "completed_outline_steps": [],
            "active_step": "1",
            "step_status": {},
            "step_attempts": {},
            "phase": "RETRIEVE",
        }
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
        # 【离线内存】Tool Knowledge Memory（抽象、紧凑、可演化）
        #   tool_capability → Planner：选哪个工具
        #   tool_invocation → Executor：如何调用（Planner 永不访问）
        # ============================================================================
        self.offline_memory = {
            "tool_capability": None,   # ToolCapabilityMemory instance
            "tool_invocation": None,   # ToolInvocationMemory instance
        }
        self._tkm_retriever: Optional[MemoryRetriever] = None
        self._tkm_refiner: Optional[MemoryRefiner] = None
        self._init_tool_knowledge_memory()

        # 诊断信号历史
        self.diagnostic_signal_history: List[Dict[str, Any]] = []

        # Canonical causal DAG (nodes + typed edges + keyword indexes)
        self.causal_graph = CausalMemoryGraph()
        self._last_graph_outcome_id: Optional[str] = None
        self._last_graph_parameter_id: Optional[str] = None
        self._last_failed_graph_parameter_id: Optional[str] = None

    def _init_tool_knowledge_memory(self) -> None:
        """Build the retriever/refiner and empty Tool Knowledge Memories.

        Tries an LLM-backed retriever/refiner first (so retrieval and evolution
        are semantic per the spec); falls back to a keyword retriever when the
        LLM provider is unavailable, so the system stays runnable offline.
        Memories start empty; `load_offline_memory` seeds/loads them.
        """
        # Retrieval is decoupled from storage (spec Mod 8) and stays keyword-only
        # by default: find_similar_factor / find_similar_subgoal match against
        # tiny canonical vocabularies (<=4 short names) where Jaccard is ample,
        # so retrieval costs ZERO LLM calls. The LLM engine is still attached to
        # the refiner for the abstraction gate / generalize / merge / boundary
        # steps. LLMMemoryRetriever remains available for opt-in.
        retriever: MemoryRetriever = KeywordMemoryRetriever()
        llm_engine = None
        try:
            from MAS.epc_aw.engine.factory import create_llm_engine
            llm_engine = create_llm_engine(
                os.getenv("MODEL_Name", "gpt-4o-mini"), temperature=0.0,
            )
        except Exception:
            llm_engine = None
        self._tkm_retriever = retriever
        self._tkm_refiner = MemoryRefiner(retriever=retriever, llm_engine=llm_engine)
        self.offline_memory["tool_capability"] = ToolCapabilityMemory(
            retriever=retriever, refiner=self._tkm_refiner,
        )
        self.offline_memory["tool_invocation"] = ToolInvocationMemory(
            retriever=retriever, refiner=self._tkm_refiner,
        )

    def set_outline(self, outline: Dict[str, Any]) -> None:
        self.outline = outline

    def get_outline(self) -> Optional[Dict[str, Any]]:
        return self.outline

    def set_agent_profile(self, agent_profile: Dict[str, Any]) -> None:
        self.agent_profile = agent_profile

    def get_agent_profile(self) -> Optional[Dict[str, Any]]:
        return self.agent_profile

    def get_obtained_information(self) -> List[Any]:
        """Derived read-only view of evidence_records content (no parallel list)."""
        if self.evidence_records:
            return [
                rec.get("content", "")
                for rec in self.evidence_records
                if rec.get("status") != "disputed"
            ]
        return list(self.obtained_information)

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

    def set_task_profile(self, profile: Any) -> None:
        self.task_profile = profile
        if profile and hasattr(profile, "phase"):
            self.task_progress["phase"] = profile.phase

    def get_task_profile(self) -> Optional[Any]:
        return self.task_profile

    def get_task_progress(self) -> Dict[str, Any]:
        return self.task_progress

    def mark_step_done(self, step_key: str) -> None:
        key = str(step_key)
        if key not in self.task_progress["completed_outline_steps"]:
            self.task_progress["completed_outline_steps"].append(key)
        self.task_progress["step_status"][key] = "done"

    def record_step_attempt(self, step_key: str, had_slot_delta: bool) -> int:
        key = str(step_key)
        attempts = self.task_progress.setdefault("step_attempts", {})
        info = attempts.get(key, {"count": 0, "no_delta_streak": 0})
        info["count"] = info.get("count", 0) + 1
        if had_slot_delta:
            info["no_delta_streak"] = 0
        else:
            info["no_delta_streak"] = info.get("no_delta_streak", 0) + 1
        attempts[key] = info
        return info["no_delta_streak"]

    def set_active_step(self, step_key: Optional[str]) -> None:
        if step_key:
            self.task_progress["active_step"] = str(step_key)

    def set_phase(self, phase: str) -> None:
        self.task_progress["phase"] = phase
        if self.task_profile and hasattr(self.task_profile, "phase"):
            self.task_profile.phase = phase

    @staticmethod
    def _parse_entity_keys(content: str) -> List[str]:
        keys: List[str] = []
        for m in re.finditer(r"arxiv[:\s]*(\d{4}\.\d{5})", content, re.I):
            keys.append(f"arxiv:{m.group(1).lower()}")
        for m in re.finditer(r"\b(\d{4}\.\d{5})\b", content):
            keys.append(f"arxiv:{m.group(1).lower()}")
        return list(dict.fromkeys(keys))

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
        claim_type: str = "fact",
        slot_bindings: Optional[Dict[str, str]] = None,
        entity_keys: Optional[List[str]] = None,
        compute_real: Optional[bool] = None,
    ) -> bool:
        """
        Add structured evidence. Returns False if duplicate or empty.

        source_quality: primary | secondary | inferred | computed | unknown
        claim_type: fact | absence | hypothesis
        compute_real: for Python_Coder_Tool records, True only when the executed
            code performed a real derivation referencing retrieved inputs (v3).
        """
        text = str(content).strip()
        if not text or len(text) < 10:
            return False

        if tool == "Base_Generator_Tool":
            source_quality = "inferred"
            if claim_type == "fact":
                claim_type = "hypothesis"

        content_hash = hashlib.md5(text[:500].encode()).hexdigest()
        for rec in self.evidence_records:
            if rec.get("content_hash") == content_hash:
                return False

        parsed_entities = entity_keys or self._parse_entity_keys(text)
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
            "claim_type": claim_type,
            "slot_bindings": slot_bindings or {},
            "entity_keys": parsed_entities,
            "status": "active",
            "timestamp": datetime.now().isoformat(),
        }
        if compute_real is not None:
            record["compute_real"] = compute_real
        self.evidence_records.append(record)
        # obtained_information is derived from evidence_records (see get_obtained_information)
        if self._last_graph_outcome_id:
            self.causal_graph.link_evidence_to_outcome(self._last_graph_outcome_id, record["id"])
        return True

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Find contradictory claims on the same entity_key."""
        by_entity: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self.evidence_records:
            if rec.get("status") == "disputed":
                continue
            for ek in rec.get("entity_keys") or []:
                by_entity.setdefault(ek, []).append(rec)

        conflicts: List[Dict[str, Any]] = []
        neg_markers = (
            "does not contain", "does not have", "no three axis", "no such figure",
            "not contain", "scaling laws", "2d log-log", "2d plot",
        )
        pos_markers = (
            "three axis", "three identified axes", "figure 1", "ai regulation",
            "standardization", "utilitarian", "egalitarianism",
        )
        for entity, recs in by_entity.items():
            if len(recs) < 2:
                continue
            texts = [str(r.get("content", "")).lower() for r in recs]
            has_neg = any(any(n in t for n in neg_markers) for t in texts)
            has_pos = any(any(p in t for p in pos_markers) for t in texts)
            if has_neg and has_pos:
                conflicts.append({
                    "entity_key": entity,
                    "record_ids": [r["id"] for r in recs],
                    "reason": "contradictory_figure_claims",
                })
        return conflicts

    def retract_evidence(self, record_id: str, reason: str = "") -> bool:
        for rec in self.evidence_records:
            if rec.get("id") == record_id:
                rec["status"] = "disputed"
                rec["retract_reason"] = reason
                return True
        return False

    def retract_inferred_for_entity(self, entity_key: str, reason: str = "") -> int:
        count = 0
        for rec in self.evidence_records:
            if rec.get("status") == "disputed":
                continue
            if entity_key in (rec.get("entity_keys") or []):
                if rec.get("source_quality") == "inferred" or rec.get("tool") == "Base_Generator_Tool":
                    rec["status"] = "disputed"
                    rec["retract_reason"] = reason
                    count += 1
        return count

    def claims_for_slot(self, slot_name: str) -> List[Dict[str, Any]]:
        return [
            r for r in self.evidence_records
            if r.get("status") != "disputed"
            and (
                slot_name in (r.get("slot_bindings") or {})
                or slot_name in str(r.get("content", "")).lower()
            )
        ]

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

    def infer_state_keyword(self) -> str:
        """Abstract state keyword for offline indexing."""
        return abstract_state_type(self.infer_state_type())

    def infer_subgoal_keyword(self, subgoal: str) -> str:
        """Abstract subgoal keyword for offline indexing."""
        return abstract_subgoal_type(subgoal)

    def _keyword_key(self, state: str, subgoal: str) -> Tuple[str, str]:
        return (abstract_state_type(state), abstract_subgoal_type(subgoal))

    def init_causal_graph_for_task(self, task_id: str) -> None:
        """Bind online causal graph to a task and reset step-local state."""
        self.causal_graph.ensure_task(task_id)
        self.online_memory["task_id"] = task_id
        self._last_graph_outcome_id = None
        self._last_graph_parameter_id = None
        self._last_failed_graph_parameter_id = None

    def _sync_denormalized_from_graph(self) -> None:
        """Refresh linear traces and task-local dict from canonical graph."""
        self.online_memory["causal_execution_traces"] = self.causal_graph.export_linear_traces()
        exported = self.causal_graph.export_task_local_parameters()
        task_local: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key_str, value in exported.items():
            if "::" in key_str:
                state_part, subgoal_part = key_str.split("::", 1)
                task_local[(state_part, subgoal_part)] = value
        self.online_memory["task_local_parameters"] = task_local

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
        contrast_failed_parameter: Optional[str] = None,
    ) -> None:
        """
        Record one execution attempt into the canonical causal graph and sync exports.

        Chain: State → Subgoal → Tool → Parameter → Outcome → (NextState on success)
        """
        result = (effect or {}).get("result") or {}
        success = bool(result.get("success"))
        intervention = None
        if not success and L2_diagnosis:
            intervention = L2_diagnosis.get("recommendation")

        outline = self.get_outline() or {}
        evidence_ids = [r["id"] for r in self.evidence_records]

        node_ids = self.causal_graph.record_execution(
            outline_step=outline_step,
            exec_step=exec_step,
            subgoal=subgoal,
            tool_name=effect.get("tool", ""),
            raw_command=effect.get("parameter", ""),
            success=success,
            result_preview=result.get("preview", ""),
            symptom=effect.get("symptom"),
            state_label=state,
            evidence_ids=evidence_ids,
            outline_remaining=len(outline),
            attempt_seq=effect.get("attempt_seq", 1),
            intervention=intervention,
            L2_diagnosis=L2_diagnosis if L2_diagnosis else None,
        )
        self._last_graph_outcome_id = node_ids["outcome_id"]
        self._last_graph_parameter_id = node_ids["parameter_id"]

        if not success:
            self._last_failed_graph_parameter_id = node_ids["parameter_id"]
        elif success and self._last_failed_graph_parameter_id:
            self.causal_graph.record_parameter_contrast(
                self._last_failed_graph_parameter_id,
                node_ids["parameter_id"],
            )
            self._last_failed_graph_parameter_id = None

        if contrast_failed_parameter and success:
            failed_parsed = parse_parameter_command(contrast_failed_parameter, effect.get("tool", ""))
            failed_id = f"pm:{hashlib.md5(failed_parsed.get('fingerprint', '').encode()).hexdigest()[:8]}"
            if failed_id in self.causal_graph.nodes:
                self.causal_graph.record_parameter_contrast(failed_id, node_ids["parameter_id"])

        if finalize and success and self._last_graph_outcome_id:
            new_state = self.infer_state_type()
            self.causal_graph.transition_state_on_success(
                self._last_graph_outcome_id,
                new_state,
                evidence_ids=evidence_ids,
                outline_remaining=len(outline),
            )

        self._sync_denormalized_from_graph()

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
        key = self._keyword_key(state, subgoal)

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
        key = self._keyword_key(state, subgoal)
        legacy_key = (state, subgoal)
        bucket = self.online_memory["task_local_parameters"]
        return bucket.get(key, bucket.get(legacy_key, {})).get(tool, {})

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
        self._sync_denormalized_from_graph()

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
        self._sync_denormalized_from_graph()

    # ============================================================================
    # 【离线内存】Tool Knowledge Memory 访问（Planner / Executor）
    # ============================================================================

    def list_capability_tools(self) -> List[str]:
        """All tool names known to the capability memory."""
        mem = self.offline_memory.get("tool_capability")
        return mem.all_tools() if mem else []

    def get_tool_capability_summary(self, tool_name: str) -> str:
        """One-line abstract capability of a tool (for Planner prompts)."""
        mem = self.offline_memory.get("tool_capability")
        tool = mem.get_tool(tool_name) if mem else None
        return (tool or {}).get("capability_summary", "")

    def retrieve_tool_capability(
        self, tool_name: str, current_subgoal: str,
    ) -> Optional[Dict[str, Any]]:
        """Planner: return the most similar subgoal entry for a tool (Capability),
        or None. context_summary are Applicability Conditions.

        Returned entry shape: {"subgoal": str, "context_summary": [str, ...]}.
        Pure text, no scores/counts/confidence (per Tool Knowledge Memory spec).
        """
        mem = self.offline_memory.get("tool_capability")
        if not mem:
            return None
        return mem.retrieve(tool_name, current_subgoal)

    def retrieve_tool_invocation(
        self, tool_name: str, subgoal: str,
    ) -> Optional[Dict[str, Any]]:
        """Executor: return invocation decision dimensions for the closest subgoal
        (Invocation), or None. Planner never calls this.

        Returned entry shape:
            {"subgoal": str, "factors": {DecisionDimension: {"instruction": str}}}.
        """
        mem = self.offline_memory.get("tool_invocation")
        if not mem:
            return None
        return mem.retrieve(tool_name, subgoal)

    def query_online_tools_for_context(
        self, state: str, subgoal: str,
    ) -> List[Dict[str, Any]]:
        """Online graph query: tools attempted in this task for state/subgoal keywords."""
        return self.causal_graph.get_tools_for_context(
            abstract_state_type(state),
            abstract_subgoal_type(subgoal),
        )

    def query_failed_parameters(
        self, state: str, subgoal: str, tool: str,
    ) -> List[Dict[str, Any]]:
        """Combined online graph + task-local failed parameter query."""
        sg_type = abstract_subgoal_type(subgoal)
        graph_failed = self.causal_graph.get_failed_parameters(sg_type, tool)
        if graph_failed:
            return graph_failed
        return self.get_task_local_parameters(state, subgoal, tool).get("failed_parameters", [])

    def query_successful_parameters(
        self, state: str, subgoal: str, tool: str,
    ) -> List[Dict[str, Any]]:
        sg_type = abstract_subgoal_type(subgoal)
        graph_success = self.causal_graph.get_successful_parameters(sg_type, tool)
        if graph_success:
            return graph_success
        return self.get_task_local_parameters(state, subgoal, tool).get("successful_parameters", [])

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
    # 【知识演化】从在线成功轨迹 → Tool Knowledge Memory（evolve）
    # ============================================================================

    def evolve_tool_knowledge(self) -> None:
        """At task end, evolve both Tool Knowledge Memories from successful
        online traces via the ingest abstraction gate (Generalize) and then
        cross-tool boundary refinement (Differentiate). Called once per task
        after the final answer is produced; online memory is updated during
        task processing and is NOT touched here.

        For every (tool, subgoal) with >=1 successful parameter record:
          1. abstract_experience: concrete subgoal/params -> abstract subgoal
             (from the frozen canonical vocabulary) + applicability conditions
             (context_summary) + orthogonal decision-dimension -> instruction
             pairs.
          2. cap.evolve / inv.evolve: merge into the matched canonical slot —
             generalizing applicability conditions, rewriting each decision
             dimension's single instruction into a more general one. Subgoal
             slots are never appended; no fields are added.
          3. boundary_refinement: once after the loop, differentiate overlapping
             capability_summary lines across tools (subgoals stay frozen).

        Knowledge stays abstract: no concrete query/entity/URL/paper values, no
        task examples, no execution-progress state in context_summary. If
        abstraction fails the leaked content is discarded and canonical defaults
        are used. Memory size stays approximately constant (compression, not
        growth).
        """
        self._sync_denormalized_from_graph()
        cap = self.offline_memory.get("tool_capability")
        inv = self.offline_memory.get("tool_invocation")
        if not cap or not inv:
            return
        refiner = cap.refiner
        question = self.query or ""

        # Group ALL successful (subgoal, params) by tool so the abstraction gate
        # can run once per tool (batched) instead of once per (tool, subgoal)
        # pair — a major LLM-cost reduction.
        per_tool: Dict[str, List[Dict[str, Any]]] = {}
        for (_state, subgoal), tools in self.online_memory["task_local_parameters"].items():
            subgoal_text = subgoal if isinstance(subgoal, str) else str(subgoal)
            if not subgoal_text.strip() or self._is_garbage_subgoal(subgoal_text):
                continue
            for tool, data in tools.items():
                successful = data.get("successful_parameters", []) if isinstance(data, dict) else []
                if not successful:
                    continue
                if not cap.data.get(tool, {}).get("subgoals"):
                    continue
                per_tool.setdefault(tool, []).append({
                    "concrete_subgoal": subgoal_text,
                    "question": question,
                    "successful_params": successful,
                })

        # Each tool's pipeline (batched abstraction -> cap.evolve/inv.evolve) is
        # fully isolated: it touches only cap.data[tool] / inv.data[tool], and the
        # refiner/retriever methods are stateless. So we run all tools in PARALLEL
        # to cut latency (ThreadPoolExecutor; LLM calls are IO-bound). This does
        # NOT reduce token cost — the token savings come from batching (O1-O5);
        # parallelism only reduces wall-clock time.
        def _run_tool_pipeline(tool_name: str, tool_items: List[Dict[str, Any]]) -> None:
            canonical = [s["subgoal"] for s in cap.data.get(tool_name, {}).get("subgoals", [])]
            cap_summary = cap.data.get(tool_name, {}).get("capability_summary", "")
            abst_list = refiner.abstract_experiences_batch(
                tool_name, cap_summary, canonical, tool_items,
            )
            for abst in abst_list:
                abstract_subgoal = abst.get("abstract_subgoal", "")
                if not abstract_subgoal:
                    continue
                cap.evolve(tool_name, abstract_subgoal, abst.get("context_summaries", []))
                factors = abst.get("factors", {})
                if factors:
                    inv.evolve(tool_name, abstract_subgoal, factors)

        if len(per_tool) <= 1:
            for tool_name, tool_items in per_tool.items():
                try:
                    _run_tool_pipeline(tool_name, tool_items)
                except Exception as e:
                    print(f"⚠️ evolve pipeline failed for {tool_name}: {e}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(len(per_tool), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_run_tool_pipeline, tn, ti): tn
                    for tn, ti in per_tool.items()
                }
                for fut in as_completed(futures):
                    tn = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"⚠️ evolve pipeline failed for {tn}: {e}")
        # Differentiate overlapping capability summaries once after the batch.
        try:
            refiner.boundary_refinement(cap)
        except Exception as e:
            print(f"⚠️ boundary_refinement skipped: {e}")

    @staticmethod
    def _is_garbage_subgoal(subgoal_text: str) -> bool:
        """Filter out non-subgoal strings that leaked into task_local_parameters.

        Catches executor status markers ("NO VALID TOOL ACTION"), execution
        progress state ("部分完成_..."), and pure state descriptions.
        """
        s = (subgoal_text or "").strip().lower()
        if not s:
            return True
        garbage_markers = (
            "no valid tool action", "no valid action", "none",
            "部分完成", "已获", "剩余", "待最终确认", "初始状态",
        )
        if any(m in s for m in garbage_markers):
            return True
        # Pure execution-progress state labels are not subgoals.
        state_only = (
            s.startswith("初始状态") or s.startswith("信息完备")
            or s.startswith("部分完成")
        )
        return state_only

    def _derive_factor_instructions(
        self, tool: str, successful_params: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Legacy best-effort factor extraction. Retained only as a diagnostic
        helper; the ingest abstraction gate (`MemoryRefiner.abstract_experience`)
        is now the canonical path and subsumes this logic with LLM abstraction.
        """
        pairs: Dict[str, str] = {}
        saw_query = saw_url = saw_site = saw_max = saw_timeout = False
        for rec in successful_params:
            param_str = rec.get("parameter", "") if isinstance(rec, dict) else str(rec)
            if not param_str:
                continue
            parsed = parse_parameter_command(param_str, tool)
            if parsed.get("query"):
                saw_query = True
            if parsed.get("url"):
                saw_url = True
            if parsed.get("site"):
                saw_site = True
            if parsed.get("max_results"):
                saw_max = True
            if parsed.get("timeout"):
                saw_timeout = True
        if saw_query:
            pairs["Entity"] = "Include sufficient identifiers to uniquely specify the target."
        if saw_url:
            pairs["Source"] = "Use a direct URL when the target page is already known."
        if saw_site:
            pairs["Scope"] = "Restrict the search to the requested domain."
        if saw_max:
            pairs["Breadth"] = "Adjust the number of returned results to the task need."
        if saw_timeout:
            pairs["Latency"] = "Set a timeout that tolerates slow responses."
        return pairs

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
        """Persist Tool Knowledge Memory (capability + invocation) as JSON."""
        memory_path = self._ensure_memory_dirs(memory_dir)
        offline_dir = memory_path / "offline"

        cap = self.offline_memory.get("tool_capability")
        inv = self.offline_memory.get("tool_invocation")
        if cap is not None:
            cap.save(str(offline_dir / "tool_capability_memory.json"), fmt="json")
        if inv is not None:
            inv.save(str(offline_dir / "tool_invocation_memory.json"), fmt="json")

        manifest = {
            "last_updated": datetime.now().isoformat(),
            "offline_memory_files": [
                "tool_capability_memory.json",
                "tool_invocation_memory.json",
            ],
            "schema": MEMORY_SCHEMA_VERSION,
        }
        manifest_file = memory_path / "offline_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def persist_online_memory(self, task_id: str, memory_dir: str = "memory") -> None:
        """将在线执行轨迹持久化到本地"""
        memory_path = self._ensure_memory_dirs(memory_dir)
        task_dir = memory_path / "online" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        self._sync_denormalized_from_graph()

        graph_file = task_dir / "causal_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(self.causal_graph.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Layer 1: Causal Execution Traces (denormalized export)
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
                "graph_nodes": len(self.causal_graph.nodes),
                "graph_edges": len(self.causal_graph.edges),
                "status": "completed"
            }
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_offline_memory(self, memory_dir: str = "memory") -> bool:
        """Load Tool Knowledge Memory from disk; seed defaults if absent or stale.

        Files:
          offline/tool_capability_memory.json
          offline/tool_invocation_memory.json
          offline_manifest.json  (carries the schema version)

        Schema-gated reset: if the on-disk schema does not match
        MEMORY_SCHEMA_VERSION, both memories are reset to the frozen canonical
        seed (capability + invocation). This cleans up polluted v1 files
        (task-specific entries) without manual deletion. On a fresh or
        schema-matching disk, the 7 enabled tools are seeded with abstract
        capabilities AND orthogonal invocation factors; `evolve_tool_knowledge`
        then generalizes/differentiates them over time.
        """
        memory_path = Path(memory_dir).absolute()
        offline_dir = memory_path / "offline"
        cap = self.offline_memory.get("tool_capability")
        inv = self.offline_memory.get("tool_invocation")
        cap_file = offline_dir / "tool_capability_memory.json"
        inv_file = offline_dir / "tool_invocation_memory.json"
        manifest_file = memory_path / "offline_manifest.json"

        # Schema check: reset to seed when the on-disk schema is stale/missing.
        disk_schema = None
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    disk_schema = (json.load(f) or {}).get("schema")
            except Exception:
                disk_schema = None
        schema_matches = disk_schema == MEMORY_SCHEMA_VERSION
        if not schema_matches:
            if disk_schema is not None:
                print(f"ℹ️ Offline memory schema {disk_schema!r} != {MEMORY_SCHEMA_VERSION!r}; "
                      f"resetting to frozen canonical seed.")
            self._seed_offline_memory_from_defaults()
            return True

        loaded_any = False
        if cap is not None and cap_file.exists():
            try:
                cap.load(str(cap_file))
                loaded_any = bool(cap.data)
            except Exception as e:
                print(f"Warning: Failed to load tool_capability_memory: {e}")

        if inv is not None and inv_file.exists():
            try:
                inv.load(str(inv_file))
                loaded_any = loaded_any or bool(inv.data)
            except Exception as e:
                print(f"Warning: Failed to load tool_invocation_memory: {e}")

        # Seed if still empty (e.g. files missing on a schema-matching disk).
        if (cap is not None and not cap.data) or (inv is not None and not inv.data):
            self._seed_offline_memory_from_defaults()
            loaded_any = True
        return loaded_any

    def _seed_offline_memory_from_defaults(self) -> None:
        """Replace both offline memories with the frozen canonical seed."""
        cap = ToolCapabilityMemory.seed_default_capabilities(
            retriever=self._tkm_retriever, refiner=self._tkm_refiner,
        )
        inv = ToolInvocationMemory.seed_default_invocations(
            retriever=self._tkm_retriever, refiner=self._tkm_refiner,
        )
        self.offline_memory["tool_capability"] = cap
        self.offline_memory["tool_invocation"] = inv

    def get_memory_stats(self, memory_dir: str = "memory") -> Dict[str, Any]:
        """获取内存统计信息（无数值型指标，仅条目计数用于运维）。"""
        memory_path = Path(memory_dir).absolute()
        cap = self.offline_memory.get("tool_capability")
        inv = self.offline_memory.get("tool_invocation")

        cap_tools = cap.all_tools() if cap else []
        cap_subgoals = sum(len(t.get("subgoals", [])) for t in (cap.data.values() if cap else []))
        inv_tools = list(inv.data.keys()) if inv else []
        inv_subgoals = sum(len(t.get("subgoals", [])) for t in (inv.data.values() if inv else []))

        stats = {
            "online_memory": {
                "causal_execution_traces": len(self.online_memory["causal_execution_traces"]),
                "task_local_parameters": len(self.online_memory["task_local_parameters"]),
                "evidence_records": len(self.evidence_records),
                "causal_graph_nodes": len(self.causal_graph.nodes),
                "causal_graph_edges": len(self.causal_graph.edges),
            },
            "offline_memory": {
                "capability_tools": len(cap_tools),
                "capability_subgoals": cap_subgoals,
                "invocation_tools": len(inv_tools),
                "invocation_subgoals": inv_subgoals,
            },
            "disk_state": {
                "offline_dir_exists": (memory_path / "offline").exists(),
                "online_dir_exists": (memory_path / "online").exists(),
                "screenshots_dir_exists": (memory_path / "screenshots").exists()
            }
        }
        return stats
