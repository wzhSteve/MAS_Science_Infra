import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.utils import parse_json_from_llm_response


@dataclass
class ExpertConfig:
    name: str
    role: str
    interface_file: Optional[str] = None
    memory_file: Optional[str] = None
    is_integrator: bool = False
    temperature: float = 0.0


class Expert:
    def __init__(self, llm_engine_name: str, config: ExpertConfig, verbose: bool = False):
        self.llm_engine_name = llm_engine_name
        self.config = config
        self.verbose = verbose
        self.llm = create_llm_engine(
            model_string=llm_engine_name,
            is_multimodal=False,
            temperature=config.temperature,
        )
        self.private_profile = self._load_private_profile(config.interface_file)
        self.private_memory = self._load_private_memory(config.memory_file)

    def _load_private_profile(self, interface_file: Optional[str]) -> Dict[str, Any]:
        if not interface_file:
            return {
                "interface_file": None,
                "loaded": False,
                "private_info": "No interface file configured.",
            }
        if not os.path.exists(interface_file):
            return {
                "interface_file": interface_file,
                "loaded": False,
                "private_info": "Interface file does not exist.",
            }

        try:
            with open(interface_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if interface_file.endswith(".json"):
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw_text": raw}
            else:
                data = {"raw_text": raw}
            return {
                "interface_file": interface_file,
                "loaded": True,
                "private_info": data,
            }
        except Exception as e:
            return {
                "interface_file": interface_file,
                "loaded": False,
                "private_info": f"Failed to load interface file: {e}",
            }

    def _profile_text(self) -> str:
        return json.dumps(self.private_profile, ensure_ascii=False, indent=2)

    def _load_private_memory(self, memory_file: Optional[str]) -> Dict[str, Any]:
        if not memory_file:
            return {
                "memory_file": None,
                "loaded": False,
                "memory": "No memory file configured.",
            }
        if not os.path.exists(memory_file):
            return {
                "memory_file": memory_file,
                "loaded": False,
                "memory": "Memory file does not exist.",
            }
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            return {
                "memory_file": memory_file,
                "loaded": True,
                "memory": text,
            }
        except Exception as e:
            return {
                "memory_file": memory_file,
                "loaded": False,
                "memory": f"Failed to load memory file: {e}",
            }

    def _memory_text(self) -> str:
        return json.dumps(self.private_memory, ensure_ascii=False, indent=2)

    def evaluate(
        self,
        task: str,
        document_text: str,
        metrics: Dict[str, Dict[str, Any]],
        round_index: int,
        shared_context: Optional[Dict[str, Any]] = None,
        peer_opinions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)
        shared_context_json = json.dumps(shared_context or {}, ensure_ascii=False, indent=2)
        peer_json = json.dumps(peer_opinions or [], ensure_ascii=False, indent=2)

        prompt = f"""
你是文档质量评估中的专家，请严格按 JSON 输出结果。
重要要求：所有自然语言字段必须使用中文，不要输出英文句子。

【专家信息】
- 姓名: {self.config.name}
- 角色: {self.config.role}
- 是否整合专家: {self.config.is_integrator}
- 个人私有信息（仅你可见）:
{self._profile_text()}
- 专家初始化记忆（仅你可见）:
{self._memory_text()}

【任务】
{task}

【待评估文档】
{document_text}

【评估指标】
{metrics_json}

【当前轮次】
{round_index}

【上一轮整合上下文（可为空）】
{shared_context_json}

【其他专家观点（可为空）】
{peer_json}

请输出如下 JSON（不要输出其它文字）：
{{
  "expert_name": "...",
  "role": "...",
  "round": {round_index},
  "overall_score": 0-10 的数字,
  "metric_scores": {{
    "metric_key": {{
      "score": 0-10 的数字,
      "reason": "简要理由"
    }}
  }},
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "evidence_quotes": ["从文档中引用的短句1", "短句2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "reply_to_peers": [
    {{
      "peer": "专家名",
      "agreement": "你认同的点",
      "challenge": "你不同意或补充的点"
    }}
  ],
  "updated_stance": "综合结论（用于下一轮讨论，中文）"
}}
"""
        raw = self.llm(prompt, temperature=self.config.temperature, n=1)
        parsed = self._safe_parse_json(raw)
        parsed.setdefault("expert_name", self.config.name)
        parsed.setdefault("role", self.config.role)
        parsed.setdefault("round", round_index)
        parsed.setdefault("overall_score", 0)
        parsed.setdefault("metric_scores", {})
        parsed.setdefault("strengths", [])
        parsed.setdefault("weaknesses", [])
        parsed.setdefault("evidence_quotes", [])
        parsed.setdefault("suggestions", [])
        parsed.setdefault("reply_to_peers", [])
        parsed.setdefault("updated_stance", "")
        return parsed

    def integrate_consensus(
        self,
        task: str,
        document_text: str,
        metrics: Dict[str, Dict[str, Any]],
        round_index: int,
        expert_opinions: List[Dict[str, Any]],
        previous_consensus: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.config.is_integrator:
            raise ValueError(f"Expert {self.config.name} is not configured as integrator.")

        prompt = f"""
你是多专家评估中的整合专家。请综合各专家观点，输出本轮共识。
重要要求：所有自然语言字段必须使用中文，不要输出英文句子。

【整合专家私有信息】
{self._profile_text()}
【整合专家初始化记忆】
{self._memory_text()}

【任务】
{task}

【待评估文档】
{document_text}

【指标】
{json.dumps(metrics, ensure_ascii=False, indent=2)}

【轮次】
{round_index}

【上一轮共识（可为空）】
{json.dumps(previous_consensus or {}, ensure_ascii=False, indent=2)}

【本轮各专家意见】
{json.dumps(expert_opinions, ensure_ascii=False, indent=2)}

请仅输出 JSON：
{{
  "round": {round_index},
  "consensus_summary": "本轮共识摘要",
  "agreed_points": ["达成一致1", "达成一致2"],
  "disputed_points": ["分歧1", "分歧2"],
  "action_items_next_round": ["下一轮聚焦点1", "聚焦点2"],
  "consensus_scores": {{
    "metric_key": 0-10 的数字
  }},
  "overall_consensus_score": 0-10 的数字,
  "stop_discussion": true 或 false,
  "final_verdict": "阶段结论（中文）"
}}
"""
        raw = self.llm(prompt, temperature=self.config.temperature, n=1)
        parsed = self._safe_parse_json(raw)
        parsed.setdefault("round", round_index)
        parsed.setdefault("consensus_summary", "")
        parsed.setdefault("agreed_points", [])
        parsed.setdefault("disputed_points", [])
        parsed.setdefault("action_items_next_round", [])
        parsed.setdefault("consensus_scores", {})
        parsed.setdefault("overall_consensus_score", 0)
        parsed.setdefault("stop_discussion", False)
        parsed.setdefault("final_verdict", "")
        return parsed

    def _safe_parse_json(self, raw: Any) -> Dict[str, Any]:
        try:
            parsed = parse_json_from_llm_response(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"raw_output": str(parsed)}
        except Exception:
            return {"raw_output": str(raw)}
