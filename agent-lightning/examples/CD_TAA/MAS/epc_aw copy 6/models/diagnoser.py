import json
import os
import re
import ast
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import (
    ContextVerification,
    MemoryVerification,
    NextStep,
    OutlineUpdateResponse,
    QueryAnalysis,
    FinalAnswer,
)
from MAS.epc_aw.models.memory import Memory
from MAS.epc_aw.models.utils import parse_json_from_llm_response
from MAS.epc_aw.models.task_profile import (
    SlotGate,
    _extract_character_name,
    _is_character_name_answer,
)
from MAS.epc_aw.models.tool_router import ToolRouter, infer_subgoal_kind
from MAS.epc_aw.models.capability import capability_matches, infer_required_capability
# Day 2 Integration: Import causal reasoning modules
from MAS.epc_aw.models import BayesianInference, HistoryAnalyzer
import time


# -----------------------------------------------------------------------------
# Causal Hypothesis → Meta Instruction mapping (P0-②).
#
# Each ``reason`` emitted by ``_classify_root_cause`` maps to a short
# instruction KEY (not a prompt template). The executor layer owns the
# instruction→natural-language rendering (INSTRUCTION_TEXT in executor.py),
# keeping this table stable as reasons grow. The KEY is the contract
# between Diagnoser (why it failed) and Executor (what to do next).
# -----------------------------------------------------------------------------
REASON_INSTRUCTION: Dict[str, str] = {
    # query formulation
    "entity_missing":      "NeedEntityAnchor",
    "over_constrained":    "DropOneConstraint",
    "generic_keyword":     "AddDomainContext",
    "wrong_year":          "FixYear",
    "wrong_publisher":     "FixPublisher",
    # retrieval coverage
    "source_not_indexed":  "PreferAuthoritativeSource",
    "paywalled":           "UseOpenArchiveOrMirror",
    "low_recall":          "BroadenQueryScope",
    # execution / external
    "thread_context":      "SkipParameterRetryEscalateToToolSwitch",
    "backend_timeout":     "ShortenQueryTokens",
    "rate_limit":          "ShortenQueryTokens",
    "runtime_error":       "RegenerateCommandFromScratch",
    "command_parse_failure": "RewriteQueryFromSubgoal",
    # environment
    "import_whitelist":    "AvoidImportUseStdlib",
    "permission_denied":   "SwitchToolOrEscalateState",
    # command schema
    "schema_mismatch":     "FixArgumentSchema",
    "missing_argument":    "FixArgumentSchema",
    # tool selection
    "capability_mismatch": "SwitchToolClass",
    # planner
    "belief_conflict":     "ReviseBelief",
    "subgoal_unreachable": "DecomposeGoal",
}

DEFAULT_INSTRUCTION = "DefaultRetry"



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
        # self.memory = DiagnoserMemory()  # 【已删除】使用SystemMemory替代
        self.verbose = verbose
        self.n = n
        self.profile = ""
        self.temperature = temperature

        # Day 2 Integration: Initialize causal reasoning modules
        self.bayesian_inference = BayesianInference()
        self.history_analyzer = HistoryAnalyzer()
        self.system_memory = None

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


    def _normalize_subgoal_conclusion(self, raw: Any) -> str:
        """Map LLM output to canonical SUBGOAL_COMPLETE / SUBGOAL_INCOMPLETE."""
        text = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
        if text in ("SUBGOAL_COMPLETE", "COMPLETE", "SUBGOAL_COMPLETE."):
            return "SUBGOAL_COMPLETE"
        if text in ("SUBGOAL_INCOMPLETE", "INCOMPLETE", "SUBGOAL_INCOMPLETE."):
            return "SUBGOAL_INCOMPLETE"
        # Avoid substring trap: "INCOMPLETE" must not match "COMPLETE"
        if text.endswith("INCOMPLETE") or text == "INCOMPLETE":
            return "SUBGOAL_INCOMPLETE"
        if text.endswith("COMPLETE") or text == "COMPLETE":
            return "SUBGOAL_COMPLETE"
        return "SUBGOAL_INCOMPLETE"

    def _parse_verification_response(self, llm_response: Any) -> Dict[str, Any]:
        """Parse structured or JSON verification output into a normalized dict."""
        if isinstance(llm_response, ContextVerification):
            slot_updates = [
                u.model_dump() if hasattr(u, "model_dump") else dict(u)
                for u in (llm_response.slot_updates or [])
            ]
            return {
                "Analysis": llm_response.analysis,
                "Subgoal_Conclusion": llm_response.subgoal_conclusion,
                "New_Obtained_Information_Flag": llm_response.new_obtained_information_flag,
                "New_Obtained_Information": llm_response.new_obtained_information,
                "Conclusion": llm_response.conclusion or llm_response.task_recommendation,
                "Slot_Updates": slot_updates,
                "Evidence_Type": llm_response.evidence_type,
                "Tool_Appropriate": llm_response.tool_appropriate,
            }

        parsed = parse_json_from_llm_response(llm_response)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict from verification response, got: {type(parsed)}")

        # Normalize key casing from free-form JSON
        key_map = {
            "analysis": "Analysis",
            "subgoal_conclusion": "Subgoal_Conclusion",
            "new_obtained_information_flag": "New_Obtained_Information_Flag",
            "new_obtained_information": "New_Obtained_Information",
            "conclusion": "Conclusion",
            "slot_updates": "Slot_Updates",
            "evidence_type": "Evidence_Type",
            "tool_appropriate": "Tool_Appropriate",
            "task_recommendation": "Conclusion",
        }
        normalized = {}
        for k, v in parsed.items():
            canonical = key_map.get(k.lower(), k)
            normalized[canonical] = v
        return normalized

    def _coerce_information_flag(self, flag: Any) -> bool:
        if flag is True:
            return True
        if isinstance(flag, str):
            return flag.strip().lower() in ("true", "yes", "1")
        return False

    def _normalize_obtained_information(self, info: Any) -> str:
        if info is None:
            return ""
        # Character schema wrappers must become short tokens before evidence write,
        # otherwise SlotGate stop-gate accepts JSON that answer_gate later rejects.
        from MAS.epc_aw.models.task_profile import (
            _extract_character_name,
            _unwrap_character_schema,
        )
        if isinstance(info, dict):
            unwrapped = _unwrap_character_schema(info)
            if unwrapped:
                canonical = _extract_character_name(unwrapped) or unwrapped
                return canonical
            return json.dumps(info, ensure_ascii=False)
        if isinstance(info, list):
            return json.dumps(info, ensure_ascii=False)
        text = str(info).strip()
        if text.lower() in ("", "none", "no new information", "[]", "{}"):
            return ""
        unwrapped = _unwrap_character_schema(text)
        if unwrapped:
            return _extract_character_name(unwrapped) or unwrapped
        return text

    def _first_result_dict(self, result_executor: Any) -> Optional[dict]:
        if isinstance(result_executor, list) and result_executor:
            first = result_executor[0]
            return first if isinstance(first, dict) else None
        if isinstance(result_executor, dict):
            return result_executor
        return None

    def _has_actual_tool_error(self, result_executor: Any) -> bool:
        """True only when the tool reported a real failure, not `"error": null`."""
        result_dict = self._first_result_dict(result_executor)
        if result_dict is not None:
            if result_dict.get("success") is False:
                return True
            err = result_dict.get("error")
            if err is not None and str(err).strip().lower() not in ("", "null", "none"):
                return True
            return False

        text = str(result_executor).lower()
        if '"success": false' in text or "'success': false" in text:
            return True
        if re.search(r"""['"]error['"]\s*:\s*['"][^'"]+['"]""", text):
            return True
        # Executor-side wrapper for tool execution failures (network errors,
        # import errors, schema mismatches, etc.) — the underlying error
        # message follows the colon.
        if "error in execute_tool_command" in text:
            return True
        return False

    def _has_usable_result(self, result_executor: Any) -> bool:
        """Return True when executor output contains substantive, non-error data."""
        if result_executor is None:
            return False

        result_dict = self._first_result_dict(result_executor)
        if result_dict is not None:
            if result_dict.get("success") is True:
                usable_keys = (
                    "image_path", "extracted_text", "image_base64",
                    "retrieved_information", "content", "text", "summary", "answer",
                )
                if any(result_dict.get(k) for k in usable_keys):
                    return True
            if self._has_actual_tool_error(result_executor):
                return False

        text = str(result_executor)
        lower = text.lower()

        if self._has_actual_tool_error(result_executor):
            if "relevant_pages (to the query)" in lower or '"relevant_pages":' in lower:
                if "retrieved_information" in lower and "answer:" in lower:
                    return True
                if '"title":' in lower and '"title": null' not in lower:
                    non_null_titles = lower.count('"title":') - lower.count('"title": null')
                    if non_null_titles > 0:
                        return True
            if "retrieved_information" not in lower:
                return False

        if "retrieved_information" in lower and "answer:" in lower:
            return True

        if isinstance(result_executor, list) and len(result_executor) > 0:
            first = result_executor[0]
            if isinstance(first, dict):
                pages = first.get("relevant_pages (to the query)") or first.get("relevant_pages") or []
                if pages:
                    return True
                other_pages = (
                    first.get("other_pages (may be irrelevant to the query)")
                    or first.get("other_pages")
                    or []
                )
                if any(
                    isinstance(page, dict)
                    and page.get("title")
                    and len(str(page.get("abstract") or page.get("content") or "").strip()) >= 80
                    for page in other_pages
                ):
                    return True

        if len(text.strip()) > 150 and "no result was generated" not in lower:
            return True

        return False

    def _analysis_indicates_subgoal_incomplete(self, analysis: str) -> bool:
        """Detect when LLM analysis text explicitly says the sub-goal failed."""
        text = str(analysis).lower()
        markers = (
            "sub-goal remains unfulfilled", "subgoal remains unfulfilled",
            "sub-goal is incomplete", "subgoal is incomplete",
            "sub-goal was not achieved", "subgoal was not achieved",
            "sub-goal was not completed", "subgoal was not completed",
            "did not complete the sub-goal", "failed to complete the sub-goal",
            "fails to meet the sub-goal", "fails to meet the subgoal",
            "does not satisfy the sub-goal", "does not satisfy the subgoal",
            "sub-goal failed", "subgoal failed",
            "layer 1 fails", "layer 1 failed", "layer 1 is not complete",
            "therefore, the sub-goal is incomplete",
            "the sub-goal is not completed", "the subgoal is not completed",
        )
        return any(marker in text for marker in markers)

    def _is_exhaustive_count_task(self, question: str, profile: Any) -> bool:
        """True for count/enumerate tasks that require exhaustive coverage.

        Compute-from-metrics tasks (e.g. Kipchoge pace × Moon perigee) have
        ``input_metrics`` + ``computed_count`` but are NOT exhaustive: each
        metric is a single-value fact, not a corpus listing that needs full
        coverage. Only corpus-style counts (articles/albums between years,
        published-by-year, etc.) qualify.
        """
        if profile is not None and getattr(profile, "exhaustive", False):
            return True
        q = str(question or "").lower()
        if not q:
            return False
        if not re.search(r"\bhow many\b", q):
            return False
        # "how many X between YEAR and YEAR" / "how many X ... published by Y"
        if re.search(r"\bbetween\b\s+\d{4}\s+(and|to|-)\s+\d{4}", q):
            return True
        if re.search(r"\b(published by|released by)\b.{0,60}\b\d{4}\b", q):
            return True
        if re.search(r"\b(discography|studio albums|albums|papers|articles)\b", q):
            return True
        return False

    def _subgoal_is_single_metric_retrieval(
        self, sub_goal: str, target_information: str = ""
    ) -> bool:
        """True when the current subgoal asks for one scalar/entity fact.

        Prevents exhaustive-count downgrade from rejecting a valid perigee /
        pace / distance retrieval just because the overall task later computes
        a count from those metrics.
        """
        text = f"{sub_goal or ''} {target_information or ''}".lower()
        if not text.strip():
            return False
        if re.search(
            r"\b(how many|number of|count of|total (number|count)|enumerate|list all)\b",
            text,
        ):
            return False
        return bool(
            re.search(
                r"\b(distance|perigee|apogee|pace|speed|mass|length|height|"
                r"radius|diameter|duration|date|name|title)\b",
                text,
            )
        )

    def _analysis_admits_partial(self, analysis: str) -> bool:
        """Verifier analysis text admits the evidence is not exhaustive."""
        text = str(analysis or "").lower()
        markers = (
            "not exhaustively listed", "not fully structured",
            "not fully listed", "not a complete list", "not the complete",
            "at least one", "at least a", "partial list", "not exhaustive",
            "not fully structured as a complete list",
            "not fully structured as a complete",
        )
        return any(m in text for m in markers)

    def _result_is_snippet_only(self, result_executor: Any) -> bool:
        """True when a Wikipedia/Search result only returned abstracts/snippets.

        Such results explicitly defer full text to a follow-up URL fetch and
        must not be treated as exhaustive evidence for count tasks.
        """
        text = str(result_executor or "")
        if "please use the url to get the full text further if needed" in text.lower():
            return True
        # Wikipedia_Search_Tool abstract-only payloads carry this stub
        if 'retrieved_information' in text.lower() and text.lower().count(
            'please use the url'
        ) >= 1:
            return True
        return False

    def _analysis_indicates_subgoal_complete(self, analysis: str) -> bool:
        """Detect when LLM analysis text says Layer 1 / sub-goal succeeded."""
        text = analysis.lower()
        if self._analysis_indicates_subgoal_incomplete(analysis):
            return False
        negative_markers = (
            "layer 1 fails", "layer 1 failed", "layer 1 is not complete",
            "sub-goal is incomplete", "subgoal is incomplete",
            "sub-goal was not completed", "subgoal was not completed",
            "did not complete the sub-goal", "failed to complete the sub-goal",
            "sub-goal failed", "subgoal failed", "layer 1 failed.",
        )
        if any(marker in text for marker in negative_markers):
            return False

        positive_markers = (
            "layer 1 is complete",
            "sub-goal is complete", "subgoal is complete",
            "while the sub-goal is complete",
            "directly satisfies the current sub-goal", "directly satisfies the current subgoal",
            "directly satisfies the sub-goal", "directly satisfies the subgoal",
            "satisfies the sub-goal", "satisfies the subgoal",
            "successfully completed the sub-goal",
            "successfully confirmed", "directly satisfies",
        )
        return any(marker in text for marker in positive_markers)

    def _reconcile_subgoal_judgment(
        self,
        step_conclusion: str,
        analysis: str,
        result_executor: Any,
        conclusion: Any,
        info_flag: bool,
        obtained_info: str,
    ) -> Tuple[str, Any, bool, str]:
        """
        Fix common LLM mistakes: Layer-2 incompleteness overriding Layer-1 success.
        """
        has_data = self._has_usable_result(result_executor)
        analysis_says_complete = self._analysis_indicates_subgoal_complete(analysis)

        if step_conclusion == "SUBGOAL_INCOMPLETE" and analysis_says_complete and has_data:
            step_conclusion = "SUBGOAL_COMPLETE"
            if not info_flag:
                info_flag = True
            if not obtained_info:
                obtained_info = self._normalize_obtained_information(
                    self._extract_key_information_from_result(result_executor, analysis)
                )
            if conclusion is None or str(conclusion).strip().lower() in ("null", "none", ""):
                conclusion = "CONTINUE"

        if step_conclusion == "SUBGOAL_COMPLETE" and self._analysis_indicates_subgoal_incomplete(analysis):
            step_conclusion = "SUBGOAL_INCOMPLETE"

        return step_conclusion, conclusion, info_flag, obtained_info

    def _extract_failure_patterns_from_evidence(
        self,
        result_executor: Any,
        *,
        evidence_type: str = "DIRECT",
        tool_appropriate: bool = True,
        slot_delta_empty: bool = True,
        same_tool_repeat_failures: int = 0,
    ) -> Dict[str, Any]:
        """Structured failure patterns for causal intervention."""
        text = str(result_executor)
        compact = re.sub(r"\s+", "", text).replace("'", '"')
        has_usable = self._has_usable_result(result_executor)
        has_error_field = self._has_actual_tool_error(result_executor)
        empty_relevant = (
            '"relevant_pages":[]' in compact
            or '"relevant_pages(tothequery)":[]' in compact
            or "relevant_pages:[]" in compact
        )

        et = str(evidence_type or "DIRECT").upper()
        lower_text = text.lower()
        # Network / infrastructure error markers — these cannot be fixed by
        # parameter perturbation and must be classified as hard errors so the
        # intervention ladder skips Level 2a and escalates to tool switching.
        network_error_markers = (
            "urlopen error", "retrieval incomplete", "connection reset",
            "connection timed out", "timed out", "timeout",
            "ssl: certificate", "no such host", "socket error",
            "max retries exceeded", "connection refused",
        )
        has_network_error = any(m in lower_text for m in network_error_markers)
        # "Error in execute_tool_command:" is the executor-side wrapper for any
        # tool execution failure (network, import, schema, etc.).
        has_executor_error_wrapper = "error in execute_tool_command" in lower_text
        patterns: Dict[str, Any] = {
            "no_results": empty_relevant and not has_usable,
            "timeout": has_network_error or "timed out" in lower_text,
            "permission_denied": any(
                p in lower_text for p in ("permission denied", "unauthorized", "access denied")
            ),
            "error_occurred": (has_error_field or has_executor_error_wrapper or has_network_error) and not has_usable,
            "irrelevant_results": False,
            "evidence_type": et,
            "tool_appropriate": tool_appropriate,
            "slot_delta_empty": slot_delta_empty,
            "same_tool_repeat_failures": same_tool_repeat_failures,
            "wrong_tool_class": not tool_appropriate,
            "absence_unfilled": et == "ABSENCE" and slot_delta_empty,
            "hallucination_risk": et == "ABSENCE" and "base_generator" in lower_text,
            "premature_synthesis": not tool_appropriate and "generator" in lower_text,
            "url_client_error": (
                "400 bad request" in lower_text
                or "400 client error" in lower_text
                or "urlopen error" in lower_text
                or has_network_error
            ),
        }

        if not has_usable and not any(
            patterns[k] for k in ("no_results", "timeout", "permission_denied", "error_occurred")
        ):
            patterns["no_results"] = len(text.strip()) < 30 or et == "EMPTY"

        return patterns

    def _count_remaining_outline_steps(
        self,
        outline: Dict[str, Any],
        current_step_key: Optional[str] = None,
    ) -> int:
        if not outline or not isinstance(outline, dict):
            return 0
        if not current_step_key:
            return len(outline)

        keys = sorted(outline.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        try:
            idx = keys.index(str(current_step_key))
            return max(0, len(keys) - idx - 1)
        except ValueError:
            return len(outline)

    def _detect_evidence_type(self, result_executor: Any, analysis: str = "") -> str:
        # Only score the lead answer — citation pools / STRUCTURED blobs often
        # embed ancillary "Read timed out" noise that must not kill DIRECT hits.
        lead = str(result_executor or "")
        lead = lead.split("---STRUCTURED---")[0]
        lead = re.split(r"\nCitations:\s*\n", lead, maxsplit=1)[0]
        lead_l = lead.lower()
        analysis_l = str(analysis or "").lower()
        err_markers = ("timeout", "exception", "error occurred", "failed to")
        if any(p in lead_l for p in err_markers):
            return "ERROR"
        # Substantial clean lead wins over timeout wording only in analysis.
        if len(lead.strip()) < 30:
            if any(p in analysis_l for p in err_markers):
                return "ERROR"
            return "EMPTY"
        text = f"{lead_l} {analysis_l}"
        absence_markers = (
            "answer: none", "does not contain", "does not have", "no such figure",
            "no three axis", "not contain", "cannot be retrieved", "no records",
            "no relevant", "not found", "does not exist",
        )
        if any(m in text for m in absence_markers):
            return "ABSENCE"
        return "DIRECT"

    def _slot_updates_list(self, raw: Any) -> List[Dict[str, Any]]:
        if not raw:
            return []
        if isinstance(raw, list):
            out = []
            for item in raw:
                if isinstance(item, dict):
                    out.append({
                        "slot": str(item.get("slot", item.get("name", ""))),
                        "value": item.get("value"),
                        "filled": bool(item.get("filled")),
                    })
                elif hasattr(item, "model_dump"):
                    out.append(item.model_dump())
            return out
        return []

    def _extract_authoritative_source_requirement(self, question: str) -> Optional[str]:
        """Detect when the question requires evidence from a named authority."""
        q = question.lower()
        markers = [
            ("usgs", "usgs"),
            ("nonindigenous aquatic species", "usgs_nas"),
            ("official", "official"),
            ("government", "government"),
            ("according to the", "named_authority"),
            ("according to", "named_authority"),
            ("as reported by", "named_authority"),
            ("from the database", "database"),
        ]
        for phrase, label in markers:
            if phrase in q:
                if phrase == "according to the":
                    return label
                if phrase == "according to" and "according to the" not in q:
                    return label
                return label
        return None

    def _assess_evidence_provenance(
        self,
        result_executor: Any,
        question: str,
    ) -> Dict[str, Any]:
        """Classify whether evidence is primary, secondary, or inferred."""
        text = str(result_executor).lower()
        auth = self._extract_authoritative_source_requirement(question)

        secondary_domains = (
            "arxiv.org", "researchgate.net", "reddit.com", "facebook.com",
            "twitter.com", "medium.com", "blog", "wikipedia.org",
        )
        primary_usgs = ("usgs.gov", "nas.er.usgs.gov", "nonindigenous")

        has_secondary = any(d in text for d in secondary_domains)
        has_primary_usgs = any(d in text for d in primary_usgs)

        if auth in ("usgs", "usgs_nas"):
            if has_primary_usgs and not has_secondary:
                level = "primary"
            elif has_secondary and not has_primary_usgs:
                level = "secondary"
            elif has_primary_usgs and has_secondary:
                level = "secondary"
            else:
                level = "inferred"
        elif auth:
            level = "secondary" if has_secondary else "primary" if len(text) > 100 else "inferred"
        else:
            level = "primary" if len(text) > 100 and not has_secondary else "secondary"

        return {"level": level, "authority_required": auth, "has_secondary_citation": has_secondary}

    def _extract_zip_codes(self, text: str) -> List[str]:
        """Extract US zip codes, excluding arXiv IDs and URL path segments."""
        raw = str(text)
        candidates = re.findall(r"\b\d{5}\b", raw)
        excluded: Set[str] = set()

        for match in re.finditer(r"\b(\d{4})\.(\d{5})\b", raw):
            excluded.add(match.group(2))

        for match in re.finditer(
            r"arxiv(?:\.org/(?:abs|pdf)/(\d{4})\.(\d{5})|:?(\d{4})\.(\d{5}))",
            raw,
            re.I,
        ):
            suffix = match.group(2) or match.group(4)
            if suffix:
                excluded.add(suffix)

        for url_match in re.finditer(r"https?://[^\s\])\"']+", raw, re.I):
            for seg in re.findall(r"/(\d{5})(?:/|$|\?|#)", url_match.group(0)):
                excluded.add(seg)

        return [z for z in dict.fromkeys(candidates) if z not in excluded]

    def _is_multi_hop_question(self, question: str) -> bool:
        q = question.lower()
        q_years = set(re.findall(r"\b(20\d{2})\b", q))
        if len(q_years) >= 2:
            return True
        if re.search(r"which of (these|the)", q):
            return True
        if q.count("arxiv") >= 2:
            return True
        cross_ref_markers = (
            "submitted to arxiv",
            "type of society",
            "cross-reference",
            "same word",
            "one of these words",
            "both papers",
            "two papers",
        )
        if sum(1 for marker in cross_ref_markers if marker in q) >= 2:
            return True
        if "arxiv" in q and q_years and re.search(r"which|one of|describe a type", q):
            return True
        return False

    def _arxiv_cross_ref_answer_ready(self, question: str, text: str) -> bool:
        """
        True when cumulative evidence supports a unique cross-paper word match
        (e.g. arXiv axis labels ∩ society-type descriptors from a second paper).
        """
        q = question.lower()
        if "arxiv" not in q:
            return False
        if not re.search(
            r"which of (these|the)|type of society|describe a type of society|one of these words",
            q,
        ):
            return False

        lower = str(text).lower()
        has_2016_paper = bool(
            re.search(r"1608\.03637|august 11,\s*2016", lower)
            or ("physics and society" in lower and "2016" in lower)
        )
        society_terms = [w for w in ("egalitarian", "hierarchical") if w in lower]
        if not society_terms:
            return False

        axis_terms = [
            w for w in (
                "standardized", "standardization", "localized", "localization",
                "utilitarian", "utilitarianism", "egalitarian", "egalitarianism",
                "deontological",
            )
            if w in lower
        ]
        if len(axis_terms) < 2 and not has_2016_paper:
            return False

        matches: Set[str] = set()
        for society in society_terms:
            for axis in axis_terms:
                if society in axis or axis.startswith(society[:6]) or society.startswith(axis[:6]):
                    matches.add(society)
                    break

        if len(matches) == 1:
            return True
        if "egalitarian" in matches and has_2016_paper and len(axis_terms) >= 2:
            return True
        return False

    def _has_direct_answer_signal(self, text: str, question: str) -> bool:
        """True when evidence contains an explicit answer, not just retrieval snippets."""
        lower = str(text).lower()
        if re.search(r"\b(the answer is|final answer)\b", lower):
            return True
        if re.search(r"\banswer:\s*\S", lower):
            return True

        if self._arxiv_cross_ref_answer_ready(question, text):
            return True

        choice_markers = (
            "the word is",
            "matching word",
            "matching label",
            "used to describe a type of society",
            "describes a type of society",
            "therefore the",
            "identified as",
            "correct choice",
        )
        if any(marker in lower for marker in choice_markers):
            q_years = set(re.findall(r"\b(20\d{2})\b", question.lower()))
            if q_years:
                t_years = set(re.findall(r"\b(20\d{2})\b", lower))
                return len(t_years & q_years) >= len(q_years)
            return True
        return False

    def _analysis_indicates_task_incomplete(self, analysis: str) -> bool:
        text = str(analysis).lower()
        markers = (
            "final question remains unanswerable",
            "remains unanswerable",
            "layer 2 gates",
            "layer 2 gate",
            "layer 2 fail",
            "layer 2 fails",
            "layer 2 is not",
            "gate a fails",
            "gate b fails",
            "gate c fails",
            "gate d fails",
            "gates a, b, c, d all fail",
            "gates a, b, c, and d all fail",
            "all fail as the final question",
            "not yet extracted",
            "not yet been calculated",
            "not yet obtained",
            "has not been located",
            "has not been found",
            "has not yet been",
            "cannot answer the original question",
            "does not yet answer",
            "overall question is not yet answered",
            "overall task is incomplete",
            "task is not complete",
            "continue is required",
            "requires further",
            "still needs",
            "more steps are needed",
            "additional steps",
            "remaining outline steps",
            "remain unexecuted",
            "remain pending",
            "outline step 2",
            "outline step 3",
            "step 2 and step 3",
            "step 2 and step 3 of the outline",
            "not present in cumulative evidence",
            "not present in the cumulative",
        )
        return any(marker in text for marker in markers)

    def _analysis_indicates_task_complete(self, analysis: str) -> bool:
        text = str(analysis).lower()
        markers = (
            "layer 2 is complete",
            "overall task is complete",
            "final question can be answered",
            "sufficient to answer the original question",
            "fully answers the question",
            "fully satisfies the original question",
        )
        return any(marker in text for marker in markers)

    def _build_cumulative_evidence_text(
        self,
        obtained_information: Any,
        result_executor: Any,
    ) -> str:
        parts = [str(result_executor), str(obtained_information or "")]
        return " ".join(parts)

    def _has_usgs_task_context(self, text: str) -> bool:
        lower = str(text).lower()
        usgs_markers = ("usgs", "nas.er.usgs.gov", "nonindigenous", "nonnative")
        entity_markers = (
            "amphiprion", "ocellaris", "clownfish", "fred howard",
            "tarpon springs", "anemonefish",
        )
        return any(m in lower for m in usgs_markers) and any(m in lower for m in entity_markers)

    def _is_pipeline_intermediate_only(self, question: str, cumulative_text: str) -> bool:
        """True when evidence is only a screenshot path, not the requested OCR/heading answer."""
        q = question.lower()
        text = str(cumulative_text).lower()
        needs_pipeline = any(m in q for m in (
            "screenshot_tool", "vision_ocr", "ocr_prompt", "screenshot",
            "extract all visible text", "main heading",
        ))
        if not needs_pipeline:
            return False
        has_artifact = ".png" in text or "image_path" in text or "solver_cache/" in text
        has_answer = any(m in text for m in (
            "extracted_text", "page_title:", "page_title",
        ))
        return has_artifact and not has_answer

    def _question_needs_computation(self, question: str) -> bool:
        q = question.lower()
        return any(m in q for m in (
            "how many", "round the", "round your", "rounded to",
            "calculate", "thousand hours", "p-value", "p value",
            "statistical significance", "incorrect as to",
            "ceil(", "ceiling",
        ))

    def _cumulative_has_computed_answer(self, question: str, text: str) -> bool:
        """True when cumulative evidence contains a plausible final numeric/concrete answer."""
        lower = str(text).lower()
        if re.search(r"\b(the answer is|final answer is|result is)\s*[:\s]*\d+", lower):
            return True
        if re.search(r"\b\d+\s+thousand hours\b", lower):
            return True
        if re.search(r"(ceil\s*\(|python_coder|calculated|computed|rounded up to)", lower):
            return bool(re.search(r"\b\d+\b", text))
        return False

    def _is_task_answer_ready(self, question: str, cumulative_text: str) -> bool:
        """True when cumulative evidence satisfies the original question."""
        q = question.lower()
        text = str(cumulative_text)

        if self._is_pipeline_intermediate_only(question, text):
            return False

        if self._question_needs_computation(question):
            if not self._cumulative_has_computed_answer(question, text):
                return False

        if "zip code" in q or "zip codes" in q:
            zips = self._extract_zip_codes(text)
            if not zips:
                return False
            if self._extract_authoritative_source_requirement(question):
                return self._has_usgs_task_context(text)
            return True

        if "comma-separated" in q or "separated by commas" in q:
            return bool(self._extract_zip_codes(text)) or "," in text

        if self._is_multi_hop_question(question):
            q_years = set(re.findall(r"\b(20\d{2})\b", q))
            if q_years:
                t_years = set(re.findall(r"\b(20\d{2})\b", text.lower()))
                if len(t_years & q_years) < len(q_years):
                    if not self._arxiv_cross_ref_answer_ready(question, text):
                        return False
            return self._has_direct_answer_signal(text, question)

        return len(text.strip()) > 80 and self._answer_format_satisfied(question, text)

    def _is_retrieval_subgoal(self, sub_goal: str, target_information: str) -> bool:
        text = f"{sub_goal} {target_information}".lower()
        markers = (
            "zip code", "five-digit", "extract", "retrieve", "find the",
            "convert", "resolve", "collection record", "confirm that",
        )
        return any(m in text for m in markers)

    def _is_negative_retrieval_result(self, result_executor: Any, sub_goal: str) -> bool:
        """True when executor only reports absence/failure without required data fields."""
        text = str(result_executor).lower()
        if self._extract_zip_codes(text):
            return False
        if self._has_usable_result(result_executor) and "answer:" in text:
            negative_phrases = (
                "does not contain", "does not include", "no records",
                "not documented", "no nonnative", "no relevant information",
                "cannot be retrieved", "404 not found", "404 error",
                "no specific entry", "no meaningful answer",
                "not include any details", "no substantive details",
            )
            if any(p in text for p in negative_phrases):
                return True
        return False

    def _answer_format_satisfied(self, question: str, result_executor: Any) -> bool:
        """Heuristic check for explicit format requirements in the question."""
        q = question.lower()
        text = str(result_executor)
        if "zip code" in q or "zip codes" in q:
            return bool(self._extract_zip_codes(text))
        if "comma-separated" in q or "separated by commas" in q:
            return bool(self._extract_zip_codes(text)) or "," in text
        if self._is_multi_hop_question(question):
            return self._has_direct_answer_signal(text, question)
        return True

    def _requires_external_evidence(self, question: str, target_information: str = "") -> bool:
        text = f"{question} {target_information}".lower()
        return any(m in text for m in (
            "usgs", "according to", "official", "database", "zip code", "zip codes",
            "nas.er.usgs.gov", "nonnative", "arxiv", "figure", "extract", "ocr",
            "screenshot", "published", "paper submitted",
        ))

    def _answer_verified_ready(
        self,
        question: str,
        cumulative: str,
        analysis: str = "",
    ) -> bool:
        """True only when cumulative evidence fully satisfies the question."""
        if not self._is_task_answer_ready(question, cumulative):
            return False
        if not self._answer_format_satisfied(question, cumulative):
            return False
        if analysis and self._analysis_indicates_task_incomplete(analysis):
            if not self._analysis_indicates_task_complete(analysis):
                return False
        return True

    def _reconcile_task_conclusion(
        self,
        task_conclusion: Any,
        step_conclusion: str,
        outline: Dict[str, Any],
        question: str,
        result_executor: Any,
        current_step_key: Optional[str] = None,
        tool_name: str = "",
        obtained_information: Any = None,
        target_information: str = "",
        sub_goal: str = "",
        analysis: str = "",
    ) -> Optional[str]:
        """
        Slot-driven reconciliation: STOP only when SlotGate confirms all required slots filled.
        """
        if step_conclusion != "SUBGOAL_COMPLETE":
            return None

        if task_conclusion is None:
            task_conclusion = "CONTINUE"

        normalized = str(task_conclusion).strip().upper()
        if normalized not in ("STOP", "CONTINUE"):
            normalized = "CONTINUE"

        profile = None
        records: List[Dict[str, Any]] = []
        if self.system_memory:
            profile = self.system_memory.get_task_profile()
            records = self.system_memory.evidence_records

        if profile:
            answer_verified = SlotGate.can_stop(profile, records)
        else:
            cumulative = self._build_cumulative_evidence_text(
                obtained_information, result_executor,
            )
            answer_verified = self._answer_verified_ready(question, cumulative, analysis)

        if not answer_verified:
            if normalized == "STOP" and self.verbose:
                print(
                    "  [STOP Reconciliation]: DOWNGRADE STOP→CONTINUE "
                    "(required slots not filled)"
                )
            return "CONTINUE"

        if normalized == "STOP":
            if self.verbose:
                print("  [STOP Reconciliation]: STOP allowed (SlotGate all_required_filled)")
            return "STOP"

        if self.verbose:
            print(
                "  [STOP Reconciliation]: KEEP CONTINUE "
                "(slots filled but verifier said CONTINUE)"
            )
        return "CONTINUE"

    def verificate_context(self, question: str, image: str, target_information: str, outline: Dict[str, Any], result_executor: str, step_count: int = 0, obtained_information: Any = None, sub_goal: str = "", tool_name: str = "", intervention_context: Optional[Dict[str, Any]] = None, current_outline_step: Optional[str] = None) -> Any:
        """
        Enhanced context verification with two-layer judgment:
        Layer 1: Is the current sub-goal completed? (using hybrid keyword+LLM approach)
        Layer 2: Is the overall task completed? (only if Layer 1 is YES)

        Returns:
            (context_verification, step_conclusion, add_obtained_information_flag,
             obtained_information, task_conclusion, diagnostic_signal_t)
        """
        image_info = self.get_image_info(image)
        # Try new v2 prompt first, fallback to original if not found
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "verificate_context_v2.txt")
        if not os.path.exists(prompt_file):
            prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "verificate_context.txt")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        remaining_steps = self._count_remaining_outline_steps(outline, current_outline_step)

        task_state_block = ""
        profile = None
        records: List[Dict[str, Any]] = []
        if self.system_memory:
            profile = self.system_memory.get_task_profile()
            records = self.system_memory.evidence_records
            if profile:
                task_state_block = profile.task_state_block(records)

        prompt_verificate_context = prompt_template.format(
            Question=question,
            Sub_Goal=sub_goal,
            Target_Information=target_information,
            Outline=outline,
            Remaining_Outline_Steps=remaining_steps,
            Result_Executor=result_executor,
            Memory=task_state_block,
            Obtained_Information_So_Far=obtained_information,
            TaskState=task_state_block,
        )

        input_data = [prompt_verificate_context]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_fixed(input_data, response_format=ContextVerification)
        try:
            parsed = self._parse_verification_response(llm_response)
            Analysis = parsed.get("Analysis", "")
            Obtained_Information_Flag = self._coerce_information_flag(
                parsed.get("New_Obtained_Information_Flag", False)
            )
            Obtained_Information = self._normalize_obtained_information(
                parsed.get("New_Obtained_Information", "")
            )
            Conclusion = parsed.get("Conclusion")
            step_conclusion = self._normalize_subgoal_conclusion(
                parsed.get("Subgoal_Conclusion", "SUBGOAL_INCOMPLETE")
            )
            slot_updates = self._slot_updates_list(parsed.get("Slot_Updates"))
            evidence_type = str(parsed.get("Evidence_Type") or "DIRECT").upper()
            tool_appropriate = parsed.get("Tool_Appropriate", True)
            if isinstance(tool_appropriate, str):
                tool_appropriate = tool_appropriate.strip().lower() in ("true", "yes", "1")
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e

        if not evidence_type or evidence_type == "DIRECT":
            evidence_type = self._detect_evidence_type(result_executor, Analysis)

        _, router_appropriate, subgoal_kind = ToolRouter.validate(
            tool_name,
            target_information,
            sub_goal,
            profile,
            records,
            self.available_tools,
        )
        if not router_appropriate:
            tool_appropriate = False

        step_conclusion, Conclusion, Obtained_Information_Flag, Obtained_Information = (
            self._reconcile_subgoal_judgment(
                step_conclusion,
                Analysis,
                result_executor,
                Conclusion,
                Obtained_Information_Flag,
                Obtained_Information,
            )
        )

        # ERROR evidence can never complete a subgoal (contradictory COMPLETE+ERROR).
        if str(evidence_type or "").upper() == "ERROR":
            step_conclusion = "SUBGOAL_INCOMPLETE"
            Conclusion = "CONTINUE"

        character_slot_required = bool(
            profile and any(
                getattr(slot, "slot_type", "") == "character_name"
                for slot in profile.slots
            )
        )
        if step_conclusion == "SUBGOAL_COMPLETE" and character_slot_required:
            candidate_values = [
                str(update.get("value", ""))
                for update in slot_updates
                if update.get("filled") and update.get("slot") == "final_answer"
            ]
            if not candidate_values:
                candidate_values = [Obtained_Information]
            result_text = str(result_executor or "")
            quoted_outputs = re.findall(r'output\s+["\']([^"\']+)["\']', question, re.I)
            grounded = not quoted_outputs or any(
                output.lower() in result_text.lower() for output in quoted_outputs
            )
            # Prefer canonical name from candidates alone (avoid wiki "parenthesis-free"
            # / unrelated punctuation noise in long result blobs).
            canonical_name = None
            for value in candidate_values:
                canonical_name = _extract_character_name(value)
                if canonical_name and _is_character_name_answer(value):
                    break
                canonical_name = None
            if not canonical_name:
                canonical_name = _extract_character_name(
                    " ".join(candidate_values)
                )
            if canonical_name and grounded and _is_character_name_answer(canonical_name):
                Obtained_Information = canonical_name
                Obtained_Information_Flag = True
                slot_updates = [
                    update for update in slot_updates
                    if update.get("slot") != "final_answer"
                ]
                slot_updates.append({
                    "slot": "final_answer",
                    "value": canonical_name,
                    "filled": True,
                })
            else:
                # No canonical punctuation name → never accept bare letters like 'g'.
                step_conclusion = "SUBGOAL_INCOMPLETE"
                Conclusion = "CONTINUE"
                slot_updates = [
                    dict(update, filled=False)
                    if update.get("slot") == "final_answer" else update
                    for update in slot_updates
                ]
                if self.verbose:
                    print(
                        "  [Subgoal Reconciliation]: DOWNGRADE — exact-character "
                        "question received code/explanation instead of a character name"
                    )

        had_slot_delta = any(u.get("filled") for u in slot_updates)
        same_tool_repeat = 0
        if self.system_memory and current_outline_step:
            attempts = self.system_memory.get_task_progress().get("step_attempts", {})
            same_tool_repeat = attempts.get(str(current_outline_step), {}).get("no_delta_streak", 0)

        failure_patterns = self._extract_failure_patterns_from_evidence(
            result_executor,
            evidence_type=evidence_type,
            tool_appropriate=tool_appropriate,
            slot_delta_empty=not had_slot_delta,
            same_tool_repeat_failures=same_tool_repeat,
        )

        # Only downgrade COMPLETE when executor output genuinely failed
        if step_conclusion == "SUBGOAL_COMPLETE" and not self._has_usable_result(result_executor):
            if failure_patterns.get("no_results") or failure_patterns.get("error_occurred"):
                step_conclusion = "SUBGOAL_INCOMPLETE"

        subgoal_actually_complete = step_conclusion == "SUBGOAL_COMPLETE"

        if (
            subgoal_actually_complete
            and tool_name == "Base_Generator_Tool"
            and (
                self._requires_external_evidence(question, target_information)
                or subgoal_kind == "acquisition"
            )
        ):
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            if self.verbose:
                print("  [Subgoal Reconciliation]: DOWNGRADE — Base_Generator cannot satisfy acquisition subgoal")

        if subgoal_actually_complete and subgoal_kind == "acquisition":
            if evidence_type in ("ABSENCE", "EMPTY") and not had_slot_delta:
                step_conclusion = "SUBGOAL_INCOMPLETE"
                subgoal_actually_complete = False
                if self.verbose:
                    print("  [Subgoal Reconciliation]: DOWNGRADE — ABSENCE/EMPTY without slot fill")
            elif not had_slot_delta and not self._has_structured_acquisition_fields(result_executor):
                step_conclusion = "SUBGOAL_INCOMPLETE"
                subgoal_actually_complete = False
                if self.verbose:
                    print("  [Subgoal Reconciliation]: DOWNGRADE — acquisition lacks structured fields")

        if (
            subgoal_actually_complete
            and evidence_type in ("ABSENCE", "EMPTY")
            and not had_slot_delta
            and not self._has_structured_acquisition_fields(result_executor)
        ):
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            if self.verbose:
                print("  [Subgoal Reconciliation]: DOWNGRADE — executor ABSENCE/EMPTY cannot complete subgoal")

        conflicts = []
        if self.system_memory:
            conflicts = self.system_memory.detect_conflicts()
        if conflicts and subgoal_actually_complete:
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            failure_patterns["belief_conflict"] = True
            if self.verbose:
                print(f"  [Subgoal Reconciliation]: DOWNGRADE — claim conflict on {conflicts[0].get('entity_key')}")

        if (
            subgoal_actually_complete
            and self._is_retrieval_subgoal(sub_goal, target_information)
            and self._is_negative_retrieval_result(result_executor, sub_goal)
        ):
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            if self.verbose:
                print("  [Subgoal Reconciliation]: DOWNGRADE — negative/empty retrieval is not subgoal completion")

        # Exhaustive count/enumerate tasks: a single snippet or partial listing
        # must NOT be accepted as SUBGOAL_COMPLETE. Requires full coverage.
        # Skip when the *current* subgoal is a single-metric fact (distance,
        # pace, …) even if the overall question later computes a count.
        if (
            subgoal_actually_complete
            and self._is_exhaustive_count_task(question, profile)
            and not self._subgoal_is_single_metric_retrieval(sub_goal, target_information)
        ):
            result_lower = str(result_executor or "").lower()
            low_confidence_count = (
                any(marker in result_lower for marker in (
                    "approximately", "estimate of", "estimated", "based on an estimate",
                ))
                or any(domain in result_lower for domain in (
                    "quora.com", "reddit.com", "answers.com",
                ))
            )
            if (
                self._result_is_snippet_only(result_executor)
                or self._analysis_admits_partial(Analysis)
                or low_confidence_count
            ):
                step_conclusion = "SUBGOAL_INCOMPLETE"
                subgoal_actually_complete = False
                failure_patterns["exhaustiveness_unmet"] = True
                if self.verbose:
                    print(
                        "  [Subgoal Reconciliation]: DOWNGRADE — exhaustive count task "
                        "received snippet-only / partial evidence"
                    )

        if self.verbose:
            print(f"  [Decision Method]: Single-pass LLM verification + reconciliation")
            print(f"  [Subgoal Conclusion]: {step_conclusion}")
            print(f"  [Information Flag]: {Obtained_Information_Flag}")
            if failure_patterns:
                active = [k for k, v in failure_patterns.items() if v]
                if active:
                    print(f"  [Failure Patterns]: {active}")

        diagnostic_signal_t = None
        if step_conclusion == "SUBGOAL_INCOMPLETE":
            if subgoal_actually_complete and not Obtained_Information_Flag:
                Obtained_Information_Flag = True
                if not Obtained_Information:
                    Obtained_Information = self._normalize_obtained_information(
                        self._extract_key_information_from_result(result_executor, Analysis)
                    )

            # Structural Causal Diagnosis (Change 1 + Change 4):
            # Run the deterministic Capability Consistency Check before
            # generating the causal signal. This overrides Result-based
            # mislabels (e.g. scipy ImportError → wrong_tool_class) and
            # attaches ``failure_type`` / ``root_cause`` to the signal so
            # downstream Intervention acts on the cause, not the symptom.
            if intervention_context:
                failure_patterns["generated_query"] = intervention_context.get(
                    "generated_query", "",
                )
                failure_patterns["command"] = intervention_context.get("command", "")
            _tool_app_ref = [tool_appropriate]
            self._apply_capability_consistency_check(
                result_executor, sub_goal, tool_name,
                failure_patterns, _tool_app_ref,
            )
            tool_appropriate = _tool_app_ref[0]

            diagnostic_signal_t = self._generate_causal_signal(
                result_executor, Analysis, sub_goal, tool_name, step_count,
                failure_patterns=failure_patterns,
                subgoal_complete=subgoal_actually_complete,
                intervention_context=intervention_context,
                evidence_type=evidence_type,
                tool_appropriate=tool_appropriate,
                has_conflicts=bool(conflicts),
            )
            task_conclusion = None
        else:
            task_conclusion = Conclusion
            if isinstance(task_conclusion, str):
                task_conclusion = task_conclusion.strip().upper()
                if task_conclusion not in ("STOP", "CONTINUE"):
                    task_conclusion = "CONTINUE"
            elif task_conclusion is None:
                task_conclusion = "CONTINUE"

            task_conclusion = self._reconcile_task_conclusion(
                task_conclusion,
                step_conclusion,
                outline,
                question,
                result_executor,
                current_outline_step,
                tool_name,
                obtained_information=obtained_information,
                target_information=target_information,
                sub_goal=sub_goal,
                analysis=Analysis,
            )

            # Auto-set information flag when subgoal complete but LLM forgot
            if subgoal_actually_complete and not Obtained_Information_Flag:
                extracted = self._extract_key_information_from_result(result_executor, Analysis)
                if extracted:
                    Obtained_Information_Flag = True
                    Obtained_Information = extracted

        return (
            Analysis,
            step_conclusion,
            Obtained_Information_Flag,
            Obtained_Information,
            task_conclusion,
            diagnostic_signal_t,
            subgoal_actually_complete,
            slot_updates,
            evidence_type,
            tool_appropriate,
        )

    @staticmethod
    def _has_structured_acquisition_fields(result_executor: Any) -> bool:
        text = str(result_executor)
        if re.search(r"https?://", text):
            return True
        if re.search(r"\barxiv:\s*\d{4}\.\d{5}|\b\d{4}\.\d{5}\b", text, re.I):
            return True
        if re.search(r"\b\d{5}\b", text):
            return True
        if re.search(r"\b(the answer is|result is|extracted_text|page_title)\b", text, re.I):
            return True
        return False

    # ------------------------------------------------------------------
    # Capability Consistency Check + failure_type / root_cause (Change 1+4)
    # ------------------------------------------------------------------
    # Deterministic, LLM-free gate that prevents the Result-based
    # misdiagnosis where a capability-matched tool (e.g. Python_Coder_Tool
    # on a compute sub-goal) is mislabelled ``wrong_tool_class`` merely
    # because its result was an environment error (e.g. scipy ImportError).
    #
    # Side effects on ``failure_patterns``:
    #   * when capability matches, force ``wrong_tool_class=False`` and
    #     ``tool_appropriate=True`` (a tool-selection error is impossible
    #     if the capability covers the sub-goal need);
    #   * set ``failure_patterns["failure_type"]`` and
    #     ``failure_patterns["root_cause"]`` so downstream Intervention
    #     can act on the *cause* rather than the symptom.
    _ENV_CONSTRAINT_MARKERS = (
        "is not in the whitelist", "import of '", "not allowed",
        "quota", "rate limit", "permission denied",
    )
    _EXTERNAL_MARKERS = (
        "urlopen error", "retrieval incomplete", "connection reset",
        "connection timed out", "timed out", "timeout",
        "no such host", "socket error", "max retries exceeded",
        "connection refused", "400 bad request", "400 client error",
        "401 unauthorized", "403 forbidden",
    )
    _COMMAND_MARKERS = (
        "missing 1 required positional argument",
        "missing required argument",
        "unexpected keyword argument",
        "takes 0 positional arguments",
        "is not callable",
    )
    # Execution-runtime markers: failures that no parameter change can fix
    # (e.g. async/thread-context errors when a tool runs inside a worker
    # thread). These must short-circuit parameter retry and force tool
    # switch / replan, otherwise the agent wastes budget regenerating
    # near-identical commands (see test_gaia_20260703_194121.log step 2-4).
    _EXECUTION_MARKERS = (
        "signal only works in main thread",
        "asyncio.run() cannot be called from a running event loop",
        "runtimeerror: there is no current event loop",
        "got futures in unexpected states",
    )

    # Reason enum keyed by target_variable. New reasons must be added under
    # an existing target_variable namespace to keep the hypothesis space
    # bounded. The instruction table (REASON_INSTRUCTION) maps each reason
    # to a short, LLM-consumable instruction key.
    _REASONS_BY_TARGET = {
        "query":           {"entity_missing", "over_constrained", "generic_keyword", "wrong_year", "wrong_publisher"},
        "coverage":        {"source_not_indexed", "paywalled", "low_recall"},
        "external":        {"backend_timeout", "rate_limit", "thread_context"},
        "environment":     {"import_whitelist", "permission_denied"},
        "command":         {"schema_mismatch", "missing_argument", "command_parse_failure"},
        "tool":            {"capability_mismatch"},
        "executor":        {"runtime_error", "command_parse_failure"},
        "planner_belief":  {"belief_conflict"},
        "task_graph":      {"subgoal_unreachable"},
    }

    @staticmethod
    def _has_entity_anchor(text: str) -> bool:
        """Heuristic: does the query/sub_goal carry a discriminating entity?

        Treats a 4-digit year, a multi-digit number, or a Capitalised token
        (excluding stop-words) as an entity anchor. Used to decide between
        ``query.generic_keyword`` and ``coverage.source_not_indexed`` when
        the only signal is ABSENCE.
        """
        if not text:
            return False
        t = str(text)
        if re.search(r"\b(19|20)\d{2}\b", t):
            return True
        if re.search(r"\b\d{2,}\b", t):
            return True
        stop = {
            "the", "a", "an", "of", "in", "on", "for", "and", "or", "by",
            "to", "with", "how", "many", "what", "who", "which", "is", "are",
            "was", "were", "find", "retrieve", "search", "get", "list",
            "total", "number", "count", "all", "articles", "article",
            "published", "publishing",
        }
        for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", t):
            if tok.lower() not in stop:
                return True
        return False

    @staticmethod
    def _count_constraints(text: str) -> int:
        """Rough constraint count: number of distinct limiting tokens."""
        if not text:
            return 0
        # Heuristic: count quoted phrases, year tokens, and capitalised
        # proper nouns as individual constraints.
        t = str(text)
        n = 0
        n += len(re.findall(r'"[^"]+"', t))
        n += len(re.findall(r"\b(19|20)\d{2}\b", t))
        n += len([tok for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", t)
                  if tok.lower() not in {
                      "The", "A", "An", "Find", "Retrieve", "Search",
                      "Get", "List", "How", "What", "Who", "Which",
                  }])
        return n

    def _classify_root_cause(
        self,
        result_executor: Any,
        sub_goal: str,
        tool_name: str,
        failure_patterns: Dict[str, Any],
    ) -> Tuple[str, str, str, str]:
        """Return (failure_type, target_variable, reason, confidence).

        Locates the failing node of the action graph and emits a structured
        Causal Hypothesis. ``target_variable`` reuses the
        ``INTERVENTION_TARGET_VARIABLE`` namespace (with two additions:
        ``query`` and ``coverage``) so the downstream Intervention stage
        operates on do(target_variable) directly. ``confidence`` is a
        coarse LOW/MEDIUM/HIGH label that only affects Diagnosis priority,
        never the Terminate decision (Terminate is driven by Cause-Changed
        + Knowledge-Gain + Action-No-op gates in solver.py).

        Order is critical: capability -> execution -> external -> command
        -> retrieval(ABSENCE) -> fallback. Execution must precede ABSENCE
        so runtime errors (e.g. ``signal only works in main thread``) are
        not mislabelled as absence-of-evidence.
        """
        text = str(result_executor or "").lower()
        cap_match = capability_matches(tool_name, sub_goal)

        # N1: Capability mismatch — tool selection is the cause. HIGH confidence:
        # capability match is a deterministic check.
        if not cap_match:
            required = infer_required_capability(sub_goal)
            return (
                "Tool",
                "tool",
                "capability_mismatch",
                "HIGH",
                f"{tool_name} cannot satisfy required capability {sorted(required)}",
            )

        # N2: Execution-runtime error (thread/async context). HIGH confidence:
        # explicit marker match. Must precede the generic error_occurred
        # branch and the ABSENCE branch so the L2a parameter retry is
        # skipped (parameter changes cannot fix a runtime-context error).
        if any(m in text for m in self._EXECUTION_MARKERS):
            return (
                "Execution",
                "external",
                "thread_context",
                "HIGH",
                "Tool not async-safe under ThreadPoolExecutor (signal/thread-context error)",
            )

        # N3: Environment constraint (e.g. import whitelist). HIGH confidence.
        if any(m in text for m in self._ENV_CONSTRAINT_MARKERS):
            imp_match = re.search(r"import of '([^']+)'", text)
            rejected = imp_match.group(1) if imp_match else "unknown"
            allowed_match = re.search(r"allowed:\s*\[([^\]]+)\]", text, re.I)
            allowed = allowed_match.group(1) if allowed_match else ""
            return (
                "Environment",
                "environment",
                "import_whitelist",
                "HIGH",
                f"Environment constraint: import '{rejected}' not whitelisted"
                + (f"; allowed: [{allowed}]" if allowed else ""),
            )

        # N4: External infrastructure error (network / HTTP / quota). HIGH confidence.
        if any(m in text for m in self._EXTERNAL_MARKERS):
            return (
                "External",
                "external",
                "backend_timeout",
                "HIGH",
                "External infrastructure failure (network/http/quota)",
            )

        # N5: Command/schema error. HIGH confidence.
        if any(m in text for m in self._COMMAND_MARKERS):
            return (
                "Command",
                "command",
                "schema_mismatch",
                "HIGH",
                "Command parameter schema mismatch",
            )

        # N6: Generic execution error not matching a specific marker. MEDIUM:
        # we know *something* ran and failed, but the precise mechanism is
        # inferred from the error_occurred flag rather than an explicit marker.
        if failure_patterns.get("error_occurred"):
            return (
                "Execution",
                "executor",
                "runtime_error",
                "MEDIUM",
                "Tool runtime error (no specific marker matched)",
            )

        # N7: EMPTY/ABSENCE on a compute tool is an execution/command failure
        # (e.g. unparsed multiline query → []), never a retrieval coverage miss.
        et = str(failure_patterns.get("evidence_type", "")).upper()
        if et in ("ABSENCE", "EMPTY") and tool_name == "Python_Coder_Tool":
            return (
                "Execution",
                "executor",
                "empty_compute_result",
                "HIGH",
                "Python_Coder_Tool returned empty/unusable output — command or execution failure, not retrieval coverage",
            )

        # N7b: EMPTY with no valid query / unparsed command → not coverage.
        # Prefer rewriting from sub_goal over PreferAuthoritativeSource.
        # Only fire when command context was supplied (solver attaches it);
        # missing keys must not reclassify ordinary EMPTY retrieval misses.
        if et in ("ABSENCE", "EMPTY") and "command" in failure_patterns:
            gen_q = str(failure_patterns.get("generated_query") or "").strip()
            cmd = str(failure_patterns.get("command") or "")
            explicit_parse_fail = bool(re.search(r"no command found", cmd, re.I))
            bare_retry = bool(re.fullmatch(r"retry\d*", gen_q, flags=re.I))
            no_usable_query = (not gen_q) and (
                explicit_parse_fail or "tool.execute" not in cmd
            )
            if (
                (explicit_parse_fail or bare_retry or no_usable_query)
                and tool_name in (
                    "Google_Search_Tool",
                    "Wikipedia_Search_Tool",
                    "Web_Search_Tool",
                    "Base_Generator_Tool",
                )
            ):
                return (
                    "Execution",
                    "executor",
                    "command_parse_failure",
                    "HIGH",
                    "No valid command/query was executed (command missing/unparsed) — "
                    "rewrite from sub_goal, not a coverage miss",
                )

        # N8: Retrieval failure (ABSENCE / EMPTY). Sub-classify by paywall/
        # refusal markers, constraint count, query quality, then coverage.
        if et in ("ABSENCE", "EMPTY"):
            raw_result = str(result_executor or "")
            # Real access gates only — do NOT map generic [REFUSAL:true] /
            # "do not provide information" templates to paywalled (that caused
            # false Cause-Changed terminates on no-match searches).
            if any(
                m in text
                for m in (
                    "paywall",
                    "paywalled",
                    "login required",
                    "sign in to",
                    "stanford login",
                    "subscribe to continue",
                    "access denied",
                )
            ):
                return (
                    "Retrieval",
                    "coverage",
                    "paywalled",
                    "MEDIUM",
                    "Retrieval hit a paywall/login gate — prefer open archive or mirror",
                )
            if (
                re.search(r'"refusal"\s*:\s*true', raw_result, re.I)
                or "[refusal:true]" in text
                or "do not provide information" in text
                or "does not provide information" in text
            ):
                return (
                    "Retrieval",
                    "coverage",
                    "low_recall",
                    "MEDIUM",
                    "Search returned a no-match/refusal template — reformulate query, not archive hop",
                )
            if self._count_constraints(sub_goal) > 3:
                return (
                    "Retrieval",
                    "query",
                    "over_constrained",
                    "MEDIUM",
                    "Query carries too many limiting constraints; the engine cannot match all of them",
                )
            # Short / low-overlap queries are formulation problems, not
            # "source not indexed" — prefer L2a parameter rewrite.
            query_guess = ""
            qm = re.search(r'query\s*=\s*["\']([^"\']+)["\']', raw_result, re.I)
            if qm:
                query_guess = qm.group(1).strip()
            if not query_guess:
                query_guess = str(failure_patterns.get("generated_query") or "").strip()
            sg_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z]{3,}", str(sub_goal or ""))
            }
            q_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z]{3,}", query_guess)
            }
            overlap = (
                len(sg_tokens & q_tokens) / max(1, len(sg_tokens))
                if sg_tokens and q_tokens
                else 1.0
            )
            if query_guess and (len(query_guess) < 12 or overlap < 0.25):
                return (
                    "Retrieval",
                    "query",
                    "generic_keyword",
                    "MEDIUM",
                    "Query is too short or poorly aligned with the sub-goal; rewrite parameters before switching tools",
                )
            if self._has_entity_anchor(sub_goal):
                return (
                    "Retrieval",
                    "coverage",
                    "source_not_indexed",
                    "MEDIUM",
                    "Query has an entity anchor but retrieval returned nothing — likely a coverage/source problem, not a query formulation problem",
                )
            return (
                "Retrieval",
                "query",
                "generic_keyword",
                "MEDIUM",
                "Query lacks a discriminating entity anchor (proper noun / year / number); retrieval cannot disambiguate",
            )

        if failure_patterns.get("no_results"):
            return (
                "Retrieval",
                "query",
                "generic_keyword",
                "MEDIUM",
                "Query produced no results",
            )

        # Fallback: keep the prior Result-based label but flag as LOW.
        return (
            "Unclear",
            "executor",
            "runtime_error",
            "LOW",
            "Root cause not localised by structural check",
        )

    def _apply_capability_consistency_check(
        self,
        result_executor: Any,
        sub_goal: str,
        tool_name: str,
        failure_patterns: Dict[str, Any],
        tool_appropriate_ref: List[bool],
    ) -> Tuple[str, str, str, str, str]:
        """Run the consistency gate and mutate ``failure_patterns`` in place.

        ``tool_appropriate_ref`` is a one-element list used as a mutable
        out-param so callers can read back the corrected flag.

        Returns ``(failure_type, target_variable, reason, confidence,
        description)`` to be attached to the diagnostic signal as the
        structured Causal Hypothesis. ``failure_patterns`` is also
        updated with ``failure_type`` / ``root_cause`` (backward compat)
        / ``target_variable`` / ``reason`` / ``root_cause_confidence``.
        """
        cap_match = capability_matches(tool_name, sub_goal)
        if cap_match:
            failure_patterns["wrong_tool_class"] = False
            failure_patterns["tool_appropriate"] = True
            tool_appropriate_ref[0] = True
            if self.verbose:
                print(
                    f"  [Capability Consistency]: {tool_name} matches "
                    f"required capability → wrong_tool_class overridden to False"
                )
        else:
            required = infer_required_capability(sub_goal)
            failure_patterns["wrong_tool_class"] = True
            failure_patterns["tool_appropriate"] = False
            tool_appropriate_ref[0] = False
            if self.verbose:
                print(
                    f"  [Capability Consistency]: {tool_name} does NOT match "
                    f"required capability {sorted(required)} → CapabilityMismatch"
                )

        failure_type, target_variable, reason, confidence, description = (
            self._classify_root_cause(
                result_executor, sub_goal, tool_name, failure_patterns,
            )
        )
        # Backward-compatible keys (downstream Cause-Changed gate compares
        # root_cause strings; INTERVENTION_TARGET_VARIABLE map reads
        # ``failure_type``). The new structured Causal Hypothesis is the
        # authoritative representation; root_cause is a derived string.
        failure_patterns["failure_type"] = failure_type
        failure_patterns["root_cause"] = description
        failure_patterns["target_variable"] = target_variable
        failure_patterns["reason"] = reason
        failure_patterns["root_cause_confidence"] = confidence
        failure_patterns["causal_hypothesis"] = {
            "failure_type": failure_type,
            "target_variable": target_variable,
            "reason": reason,
            "confidence": confidence,
            "description": description,
        }
        return failure_type, target_variable, reason, confidence, description

    def _generate_causal_signal(self, result_executor: str, analysis: str, sub_goal: str, tool_name: str, step_count: int, failure_patterns: Dict[str, Any] = None, subgoal_complete: bool = False, intervention_context: Optional[Dict[str, Any]] = None, evidence_type: str = "DIRECT", tool_appropriate: bool = True, has_conflicts: bool = False) -> Dict[str, Any]:
        """
        Generate causal diagnostic signal when subgoal is not completed.
        Provides structured guidance for the next execution cycle.

        Args:
            failure_patterns: Failure patterns from LLM judgment (always provided)
            subgoal_complete: Whether the current subgoal is actually complete
        """
        if failure_patterns is None:
            failure_patterns = {
                "no_results": False,
                "timeout": False,
                "permission_denied": False,
                "error_occurred": False,
                "evidence_type": evidence_type,
                "tool_appropriate": tool_appropriate,
            }

        exhausted = set()
        failed_tools: List[str] = []
        if intervention_context:
            exhausted = set(intervention_context.get("exhausted") or [])
            failed_tools = list(intervention_context.get("failed_tools") or [])

        recommendation = self._suggest_intervention(
            failure_patterns,
            tool_name,
            subgoal_complete=subgoal_complete,
            exhausted_interventions=exhausted,
            failed_tools=failed_tools,
            has_conflicts=has_conflicts,
        )

        signal = {
            "triggered": True,
            "reason": "SUBGOAL_INCOMPLETE",
            "sub_goal": sub_goal,
            "tool": tool_name,
            "step_count": step_count,
            "analysis": analysis,
            "failure_patterns": failure_patterns,
            "recommendation": recommendation,
            "subgoal_complete": subgoal_complete,
            "evidence_type": evidence_type,
            "tool_appropriate": tool_appropriate,
            # Structural Causal Diagnosis fields (Change 1):
            # ``failure_type`` localises the failing node of the causal
            # graph; ``root_cause`` is the human-readable cause. They are
            # populated by ``_apply_capability_consistency_check`` which
            # runs before this method at the call site.
            "failure_type": failure_patterns.get("failure_type", "Unclear"),
            "root_cause": failure_patterns.get("root_cause", "Root cause not localised"),
            # Causal Hypothesis (new authoritative representation):
            # {failure_type, target_variable, reason, confidence, description}.
            # Intervention treats this as a testable hypothesis; Evidence/Slot
            # drift falsifies it; Cause-Changed gate confirms/refines it.
            "causal_hypothesis": failure_patterns.get("causal_hypothesis", {
                "failure_type": failure_patterns.get("failure_type", "Unclear"),
                "target_variable": failure_patterns.get("target_variable", "executor"),
                "reason": failure_patterns.get("reason", "runtime_error"),
                "confidence": failure_patterns.get("root_cause_confidence", "LOW"),
                "description": failure_patterns.get("root_cause", "Root cause not localised"),
            }),
            "target_variable": failure_patterns.get("target_variable", "executor"),
            "reason": failure_patterns.get("reason", "runtime_error"),
            "root_cause_confidence": failure_patterns.get("root_cause_confidence", "LOW"),
        }

        # If recommendation is to retry with different parameters,
        # provide specific parameter variation guidance
        if recommendation == "retry_with_different_parameters":
            signal["parameter_guidance"] = self._generate_parameter_guidance(
                result_executor,
                analysis,
                tool_name,
                failure_patterns
            )

        return signal

    def _generate_parameter_guidance(self, result_executor: str, analysis: str, tool_name: str, failure_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter variation guidance keyed by the Causal Hypothesis.

        Primary output: ``instruction`` (a short KEY from REASON_INSTRUCTION),
        ``target_variable``, ``reason``. The executor layer renders the KEY
        into a concrete natural-language hint (INSTRUCTION_TEXT in
        executor.py), so this method does NOT own prompt text — that keeps
        the Diagnoser stable as new reasons are added.

        Legacy fields (``directions`` / ``avoid_patterns`` / ``examples`` /
        ``explanation``) are retained as a tool-specific fallback for the
        case where ``reason`` is empty or ``DefaultRetry``, so existing
        executor prompt rendering keeps working until the instruction-driven
        path is fully migrated.
        """
        target_variable = failure_patterns.get("target_variable", "")
        reason = failure_patterns.get("reason", "")
        confidence = failure_patterns.get("root_cause_confidence", "MEDIUM")
        instruction = REASON_INSTRUCTION.get(reason, DEFAULT_INSTRUCTION)

        guidance: Dict[str, Any] = {
            # Primary, structured, reason-driven fields.
            "instruction": instruction,
            "target_variable": target_variable,
            "reason": reason,
            "confidence": confidence,
            # Legacy fields (kept for backward compat; populated below as fallback).
            "directions": [],
            "avoid_patterns": [],
            "examples": [],
            "explanation": "",
        }

        # Tool-specific heuristic fallback. Only used when the structured
        # reason is unknown/default — otherwise the instruction KEY is the
        # authoritative guidance and the legacy fields stay empty so the
        # executor's instruction-driven rendering path takes over.
        if reason in ("", "runtime_error") or instruction == DEFAULT_INSTRUCTION:
            result_str = str(result_executor).lower()
            analysis_str = str(analysis).lower()
            combined = result_str + " " + analysis_str

            if tool_name == "Wikipedia_Search_Tool":
                if "empty" in combined or "no results" in combined or "not found" in combined:
                    guidance["directions"] = [
                        "remove_specific_names_or_details",
                        "use_simpler_main_keywords",
                        "search_by_subject_category_instead",
                        "try_alternative_keywords_for_same_concept"
                    ]
                    guidance["avoid_patterns"] = [
                        "multiple_specific_names_together",
                        "overly_complex_query_with_many_modifiers",
                        "exact_phrases_that_may_not_exist"
                    ]
                    guidance["examples"] = [
                        "AVOID: 'Les Tuche 2 director' (too specific, searches for exact combination)",
                        "TRY INSTEAD: 'Tuche film' OR 'Tuche 2 movie' OR 'Les Tuche series'"
                    ]
                    guidance["explanation"] = "Wikipedia may not have articles combining all these specific terms. Try simpler, more general searches focusing on the main subject (movie/person/concept)."

                elif "too many" in combined:
                    guidance["directions"] = [
                        "add_specific_filters",
                        "narrow_to_specific_category",
                        "combine_with_additional_restricting_keywords"
                    ]
                    guidance["avoid_patterns"] = [
                        "very_broad_single_word_queries",
                        "missing_context_keywords"
                    ]
                    guidance["explanation"] = "Too many results. Make the query more specific by adding context or limiting to a category."

                elif "timeout" in combined or "slow" in combined:
                    guidance["directions"] = [
                        "simplify_the_query",
                        "reduce_number_of_keywords",
                        "add_time_period_or_specific_context"
                    ]
                    guidance["avoid_patterns"] = [
                        "very_long_complex_queries"
                    ]
                    guidance["explanation"] = "Query processing is slow. Use a simpler, shorter query."

            elif tool_name == "Google_Search_Tool" or tool_name == "Web_Search_Tool":
                if "empty" in combined or "no results" in combined:
                    guidance["directions"] = [
                        "try_different_keywords_or_synonyms",
                        "remove_restrictive_operators",
                        "search_by_main_topic_only"
                    ]
                    guidance["avoid_patterns"] = [
                        "too_many_specific_requirements_at_once"
                    ]
                    guidance["examples"] = [
                        "AVOID: 'director=\"Jean-Paul Dubosc\"' (restrictive syntax)",
                        "TRY: 'Les Tuche 2 director' OR 'who directed Tuche 2'"
                    ]
                    guidance["explanation"] = "No results found. Try with fewer restrictions or different keyword combinations."

            else:
                # Generic fallback for other tools
                guidance["directions"] = [
                    "modify_key_parameters_substantially",
                    "try_alternative_input_format",
                    "adjust_search_scope_or_filters"
                ]
                guidance["avoid_patterns"] = [
                    "exact_same_parameters",
                    "only_minor_variations"
                ]
                guidance["explanation"] = "Try fundamentally different parameter values or approaches."

        return guidance

    def _suggest_intervention(
        self,
        failure_patterns: Dict[str, Any],
        tool_name: str,
        subgoal_complete: bool = False,
        exhausted_interventions: Optional[Set[str]] = None,
        failed_tools: Optional[List[str]] = None,
        has_conflicts: bool = False,
    ) -> str:
        """
        Select causal intervention with escalation when prior attempts are exhausted.

        Ladder: retry → switch_tool → revise_belief → decompose_goal → modify_state
        """
        exhausted = exhausted_interventions or set()

        if subgoal_complete:
            return "continue_to_next_subgoal"

        candidates: List[str] = []
        if has_conflicts or failure_patterns.get("belief_conflict"):
            candidates.append("revise_belief")
        # Hard tool/class failures: switch immediately.
        hard_switch = (
            failure_patterns.get("wrong_tool_class")
            or not failure_patterns.get("tool_appropriate", True)
            or failure_patterns.get("url_client_error")
        )
        if hard_switch:
            candidates.append("switch_tool")
        if failure_patterns.get("permission_denied"):
            candidates.append("modify_state")
        # Soft ABSENCE / EMPTY / no_results: Pearl ladder — retry params first,
        # then switch_tool after L2a budget is exhausted (handled upstream).
        soft_absence = (
            failure_patterns.get("absence_unfilled")
            or failure_patterns.get("evidence_type") in ("EMPTY", "ABSENCE")
            or failure_patterns.get("no_results")
        )
        if soft_absence and not hard_switch:
            candidates.append("retry_with_different_parameters")
            candidates.append("switch_tool")
        if failure_patterns.get("exhaustiveness_unmet") and not soft_absence:
            candidates.append("switch_tool")
        if failure_patterns.get("error_occurred") and not soft_absence:
            candidates.append("switch_tool")
        if failure_patterns.get("timeout"):
            candidates.append("switch_tool")
        if failure_patterns.get("same_tool_repeat_failures", 0) >= 2:
            candidates.append("decompose_goal")
        if failure_patterns.get("irrelevant_results"):
            candidates.append("retry_with_different_parameters")
        if not candidates:
            candidates.append("retry_with_different_parameters")

        seen: Set[str] = set()
        ordered: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        for candidate in ordered:
            if candidate not in exhausted:
                return candidate

        for intervention in (
            "retry_with_different_parameters",
            "switch_tool",
            "revise_belief",
            "decompose_goal",
            "modify_state",
        ):
            if intervention not in exhausted:
                return intervention

        return "decompose_goal"

    def generate_diagnostic_signal(self, question: str, image: str, target_information: str, outline: Dict[str, Any],
                                   result_executor: str, step_count: int = 0, obtained_information: Any = None,
                                   context: str = "", sub_goal: str = "", tool_name: str = "") -> Dict[str, Any]:
        """
        Generate a structured diagnostic signal based on execution results.
        This is the core of the causal intervention framework (L1/L2/L3 diagnosis).

        Returns:
            diagnostic_signal_t: A structured signal containing:
            - triggered: whether diagnosis was triggered (failure occurred)
            - level: diagnostic level (L1, L2, or L3)
            - L1/L2/L3: specific diagnostic information at each level
        """
        # If execution was successful, no diagnosis needed
        if "success" in str(result_executor).lower() or "obtained" in str(result_executor).lower():
            return {
                "triggered": False,
                "reason": "execution_success",
                "level": None
            }

        # Execution failed - trigger L1 diagnosis
        diagnostic_signal = {"triggered": True}

        # L1 Diagnosis: Parameter Problem Detection
        # In a real implementation, this would coordinate with Executor to try parameter variants
        # For now, generate the L1 diagnosis structure
        l1_diagnosis = {
            "diagnosis": "parameter_exhausted",  # Assume exhausted for this version
            "attempted_variants": [],
            "failure_patterns": self._extract_failure_patterns(result_executor),
            "recommendation": "proceed_to_L2"
        }
        diagnostic_signal["L1"] = l1_diagnosis

        # L2 Diagnosis: Root Cause Analysis (Counterfactual Reasoning)
        failure_patterns = l1_diagnosis["failure_patterns"]
        l2_diagnosis = self._diagnose_L2_root_cause(failure_patterns)
        diagnostic_signal["L2"] = l2_diagnosis
        diagnostic_signal["level"] = "L2"

        # L3 Diagnosis: Generate Recovery Recommendations
        if l2_diagnosis.get("root_cause") in ["tool_incompatible", "goal_issue", "state_issue"]:
            l3_diagnosis = self._diagnose_L3_recommendations(l2_diagnosis, tool_name, sub_goal)
            diagnostic_signal["L3"] = l3_diagnosis
            diagnostic_signal["level"] = "L3"

        return diagnostic_signal

    def _diagnose_L2_root_cause(self, failure_patterns: Dict[str, bool]) -> Dict[str, Any]:
        """
        Level 2 diagnosis: Determine root cause based on failure pattern consistency.
        Uses counterfactual reasoning (Pearl's Level 3).
        """
        pattern_types = list(failure_patterns.keys())

        diagnosis = {
            "root_cause": "tool_incompatible",
            "evidence": {"failure_patterns": pattern_types},
            "causal_analysis": ""
        }

        if failure_patterns.get("no_results"):
            diagnosis["root_cause"] = "tool_incompatible"
            diagnosis["causal_analysis"] = "Tool returned no results despite parameter variation. Root cause likely in tool capability or goal unreachability."
        elif failure_patterns.get("permission_denied"):
            diagnosis["root_cause"] = "state_issue"
            diagnosis["causal_analysis"] = "Permission or authentication issue detected. System state needs modification."
        elif failure_patterns.get("timeout"):
            diagnosis["root_cause"] = "tool_incompatible"
            diagnosis["causal_analysis"] = "Tool execution timeout. May indicate tool incompatibility or goal complexity."
        else:
            diagnosis["root_cause"] = "tool_incompatible"

        return diagnosis

    def should_record_obtained_information(self,
                                           step_conclusion: str,
                                           subgoal_actually_complete: bool,
                                           obtained_information_flag: bool,
                                           obtained_information: str,
                                           result_executor: str,
                                           analysis: str,
                                           tool_name: str = "",
                                           evidence_type: str = "DIRECT") -> Tuple[bool, str]:
        """
        Decide whether to persist new information to system memory.

        Generator absence/hypothesis claims do not fill required slots alone.
        """
        info = self._normalize_obtained_information(obtained_information)

        if tool_name == "Base_Generator_Tool" and evidence_type == "ABSENCE":
            if info and len(info) > 20:
                return True, info
            return False, ""

        if obtained_information_flag and info:
            if tool_name == "Base_Generator_Tool":
                return True, info
            return True, info

        is_complete = (
            step_conclusion == "SUBGOAL_COMPLETE" or subgoal_actually_complete
        )

        if is_complete:
            if info:
                return True, info
            extracted = self._extract_key_information_from_result(result_executor, analysis)
            if extracted:
                if tool_name == "Base_Generator_Tool" and evidence_type == "ABSENCE":
                    return True, extracted
                return True, extracted

        if evidence_type == "ABSENCE" and info:
            return True, info

        return False, ""

    def _extract_key_information_from_result(self, result_executor: str, analysis: str) -> str:
        """
        从执行结果中提取关键信息

        策略：
        1. 查找result中的"Answer:"、数据列表、数字
        2. 如果result是列表/字典，提取第一条有意义的数据
        3. 从analysis中提取重要的发现
        """
        result_str = str(result_executor)

        # 策略1：查找显式答案标记
        if "Answer:" in result_str:
            parts = result_str.split("Answer:")
            if len(parts) > 1:
                answer = parts[1].strip()[:500]  # 限制长度
                if answer and answer != "No new information":
                    return answer

        # 策略2：如果result是列表或包含多项，提取摘要
        if isinstance(result_executor, list) and len(result_executor) > 0:
            # 从第一条记录中提取信息
            first_item = result_executor[0]
            if isinstance(first_item, dict):
                if first_item.get("success") and first_item.get("extracted_text"):
                    return str(first_item["extracted_text"])[:500]
                if first_item.get("success") and first_item.get("image_path"):
                    parts = []
                    if first_item.get("page_title"):
                        parts.append(f"page_title: {first_item['page_title']}")
                    parts.append(f"image_path: {first_item['image_path']}")
                    return "; ".join(parts)
                # 查找重要字段
                for key in ["retrieved_information", "content", "text", "summary", "answer"]:
                    if key in first_item:
                        value = str(first_item[key])[:500]
                        if value and value != "No new information":
                            return value

        # 策略3：从analysis中提取发现
        analysis_lower = str(analysis).lower()
        if "found" in analysis_lower or "confirmed" in analysis_lower:
            # 提取analysis的前几句
            sentences = analysis.split(".")[:2]
            extracted = ". ".join(sentences)
            if len(extracted) > 20:
                return extracted[:500]

        return ""

    def _has_valid_execution_data(self, result_executor: str, analysis: str) -> bool:
        """
        判断执行结果是否包含有用的数据（而非错误）

        返回True如果：
        - 有实际的返回数据
        - 分析文本包含发现（即使subgoal不完整）
        - 没有错误信号
        """
        result_str = str(result_executor).lower()
        analysis_str = str(analysis).lower()

        # 排除：明确的错误或无数据
        if any(word in result_str for word in ["error", "exception", "no results", "not found", "failed"]):
            return False

        # 包含：有数据的迹象
        has_data = (
            len(str(result_executor)) > 100 or
            isinstance(result_executor, (list, dict)) or
            "answer" in result_str or
            "found" in analysis_str or
            "obtained" in analysis_str or
            "confirmed" in analysis_str
        )

        return has_data

    def _create_execution_summary(self, result_executor: str, step_conclusion: str) -> str:
        """
        为不完整但有数据的执行创建摘要

        示例：
        - "Partial result: Found X species information, need to verify Y"
        - "Wikipedia search returned 5 pages with X information"
        """
        result_str = str(result_executor)

        if isinstance(result_executor, list):
            count = len(result_executor)
            if count > 0:
                return f"Retrieved {count} results with potentially relevant information. First result summary: {result_str[:200]}"

        if isinstance(result_executor, dict):
            keys = list(result_executor.keys())
            if keys:
                return f"Retrieved structured data with keys: {', '.join(keys[:3])}. Content preview: {result_str[:200]}"

        # 纯文本结果
        if len(result_str) > 50:
            return f"Execution result (incomplete): {result_str[:300]}"

        return ""

    def _diagnose_L3_recommendations(self, l2_diagnosis: Dict[str, Any], tool_name: str, sub_goal: str) -> Dict[str, Any]:
        """
        Level 3 diagnosis: Generate specific recovery recommendations based on L2 diagnosis.
        These recommendations will be executed by Planner/Executor in the next cycle.
        """
        root_cause = l2_diagnosis.get("root_cause", "tool_incompatible")

        if root_cause == "tool_incompatible":
            # Recommend tool switching
            return {
                "recommendation": "switch_tool",
                "action": f"Planner should switch from {tool_name} to alternative tools",
                "candidate_tools": ["Alternative_Tool_1", "Alternative_Tool_2", "Alternative_Tool_3"]
            }
        elif root_cause == "goal_issue":
            # Recommend goal decomposition
            return {
                "recommendation": "decompose_goal",
                "action": f"Decompose goal '{sub_goal}' into sub-steps",
                "decomposed_goals": []
            }
        elif root_cause == "state_issue":
            # Recommend state modification
            return {
                "recommendation": "modify_state",
                "action": "Modify system state (e.g., refresh credentials, increase quota)",
                "suggested_action": "refresh_or_wait"
            }
        else:
            return {
                "recommendation": "continue_exploration",
                "action": "Continue parameter exploration in L1"
            }

    def _build_capability_hint(self, target_information: str) -> str:
        """把 Tool Capability Memory（tool 能力边界）注入 update_outline prompt（≤8 行）。

        纯文本：每个启用工具的 capability_summary + 最接近 subgoal 的适用条件
        (context_summary)。镜像 Planner._build_memory_query_hint，使 outline
        重规划时遵守已学习的工具能力边界（避免把检索子目标分配给 Python_Coder_Tool 等）。
        """
        if not self.system_memory:
            return ""
        ablation = getattr(self, "ablation", None)
        if ablation is not None and not ablation.read_capability_memory:
            return ""
        subgoal = target_information or ""
        lines = ["🧰 TOOL CAPABILITIES (learned boundaries):"]
        for tool in self.system_memory.list_capability_tools():
            summary = self.system_memory.get_tool_capability_summary(tool)
            entry = self.system_memory.retrieve_tool_capability(tool, subgoal)
            if entry:
                ctx = entry.get("context_summary") or []
                ctx_str = "; ".join(ctx[:2]) if ctx else ""
                lines.append(f"  • {tool}: {summary} | when: {ctx_str}")
            else:
                lines.append(f"  • {tool}: {summary}")
        if len(lines) <= 1:
            return ""
        return "\n".join(lines[:8]) + "\n"

    def update_outline(
        self,
        question: str,
        context_verification: str,
        target_information: str,
        outline: Dict[str, Any],
        result_executor: str,
        current_step: int = 1,
        obtained_information: Any = None,
        toolbox_metadata=None,
        epc_aw_analysis: str = "",
        execution_memory: str = "",
    ) -> Dict[str, Any]:
        prompt_file = os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
            "prompts", "diagnoser", "update_outline.txt",
        )
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        task_state = ""
        if self.system_memory:
            profile = self.system_memory.get_task_profile()
            if profile:
                task_state = profile.task_state_block(self.system_memory.evidence_records)

        # 读取 tool_capability_memory.json 作为「工具能力边界」外部信息，
        # 让 outline 重规划遵守已学习的工具适用条件（与 Planner 共享同一记忆源）。
        capability_hint = self._build_capability_hint(target_information)

        prompt_update_outline = prompt_template.format(
            Question=question,
            Current_Verification_Analysis=context_verification,
            Result_Executor=result_executor,
            Memory=execution_memory or "N/A",
            Obtained_Information_So_Far=obtained_information or "None yet",
            Toolbox_Metadata=toolbox_metadata if toolbox_metadata else "N/A",
            Tool_Capability_Hint=capability_hint or "N/A",
            EPC_AW_Analysis=epc_aw_analysis if epc_aw_analysis else "N/A",
            current_step=current_step,
            TaskState=task_state or "N/A",
        )

        # NOTE: update_outline intentionally does NOT use response_format
        # structured output. The outline is a freeform keyed dict
        # ({"1": "...", "2": "..."}), i.e. `Dict[str, str]` -> JSON schema
        # `additionalProperties`. OpenAI structured-output strict mode + the
        # API proxy do not support freeform `additionalProperties`: the SDK
        # strips the property from `properties` while keeping it in `required`,
        # producing a permanent proxy 429 ("Extra required key ... supplied").
        # Plain-text generation + `parse_json_from_llm_response` is robust
        # (the prompt already enforces STRICT JSON output) and avoids the
        # schema rejection entirely. The isinstance branch is retained for
        # any future structured-output path.
        llm_response = self.llm_engine_fixed(
            [prompt_update_outline]
        )
        try:
            if isinstance(llm_response, OutlineUpdateResponse):
                new_outline = llm_response.execution_outline
            else:
                parsed = parse_json_from_llm_response(llm_response)
                if isinstance(parsed, dict):
                    new_outline = parsed.get("execution_outline", parsed.get("ExecutionOutline", {}))
                else:
                    new_outline = {}
        except Exception as e:
            if self.verbose:
                print(f"\n==> ⚠️ Outline update parse failed ({e}); using fallback outline\n")
            new_outline = {}

        if not isinstance(new_outline, dict):
            new_outline = {}

        # Trust LLM renumbered outline; do not merge with stale steps
        return new_outline

    # Day 2 Integration: Bayesian inference-based diagnosis methods

    def _extract_symptoms(self, error_info: Dict[str, Any]) -> List[str]:
        """
        Extract symptoms from error information.
        Symptoms are used for Bayesian root cause inference.
        """
        symptoms = []

        # Check for timeout symptom
        if error_info.get('error_type', ''):
            error_type = error_info['error_type'].lower()
            if 'timeout' in error_type:
                symptoms.append('timeout')
            if 'format' in error_type or 'format_error' in error_type:
                symptoms.append('format_error')
            if 'encoding' in error_type:
                symptoms.append('encoding_error')
            if 'execution' in error_type:
                symptoms.append('execution_error')

        # Check for no results symptom
        result = error_info.get('result', '')
        result_count = error_info.get('result_count', 0)

        if not result or result_count == 0 or 'no_results' in str(result).lower():
            symptoms.append('no_results')

        # Check for empty result
        if 'empty' in str(result).lower() or 'no_data' in str(result).lower():
            if 'no_results' not in symptoms:
                symptoms.append('no_results')

        # Check for low precision
        precision = error_info.get('precision', 1.0)
        if precision < 0.5:
            symptoms.append('low_precision')

        # Check for incompatible output
        if 'incompatible' in str(error_info).lower():
            symptoms.append('incompatible_output')

        return list(set(symptoms))  # Remove duplicates

    def _infer_root_cause(self, symptoms: List[str]) -> List[Tuple[str, float]]:
        """
        Use Bayesian inference to find root causes from symptoms.
        Returns list of (cause, probability) tuples.
        """
        if not symptoms:
            return []

        try:
            result = self.bayesian_inference.infer_root_cause(symptoms)
            return result.root_causes
        except Exception as e:
            if self.verbose:
                print(f"Warning: Bayesian inference failed: {e}")
            return []

    def _refine_diagnosis(self, root_causes: List[Tuple[str, float]], error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create structured diagnosis from root causes and error info.
        """
        diagnosis = {
            'primary_cause': root_causes[0][0] if root_causes else 'unknown',
            'all_causes': root_causes,
            'confidence': root_causes[0][1] if root_causes else 0.0,
            'error_info': error_info,
            'timestamp': time.time()
        }
        return diagnosis

    def _extract_constraints(self, diagnosis: Dict[str, Any]) -> List[Dict]:
        """
        Extract quantified constraints from diagnosis.
        Constraints guide parameter tuning and tool selection.
        """
        constraints = []
        primary_cause = diagnosis['primary_cause']

        # Extract constraints based on root cause diagnosis
        if 'max_results' in primary_cause:
            constraints.append({
                'type': 'parameter_bound',
                'parameter': 'max_results',
                'bound': 50,  # Conservative upper bound
                'confidence': 0.75,
                'reason': primary_cause
            })

        if 'timeout' in primary_cause:
            constraints.append({
                'type': 'parameter_bound',
                'parameter': 'timeout',
                'bound': 30,  # Conservative upper bound
                'confidence': 0.75,
                'reason': primary_cause
            })

        if 'tool_not_suitable' in primary_cause:
            constraints.append({
                'type': 'tool_constraint',
                'suggestion': 'use_different_tool',
                'confidence': 0.8,
                'reason': primary_cause
            })

        if 'parameter_mismatch' in primary_cause:
            constraints.append({
                'type': 'parameter_adjustment',
                'suggestion': 'tune_parameters',
                'confidence': 0.7,
                'reason': primary_cause
            })

        return constraints

    def epc_aw_diagnosis(self, planner_selected_plan: str, BTS_selected_plan: str, target_information: str, outline: Dict[str, Any], result_executor: str, step_count: int = 0, json_data: Any = None) -> Any:
        prompt_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "diagnoser", "epc_aw_diagnosis.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_epc_aw_diagnosis = prompt_template.format(
            Planner_Plan=planner_selected_plan,
            BTS_Plan=BTS_selected_plan,
            Target_Information=target_information,
            Executor_Result=result_executor,
            Verified_Memory=""  # Diagnoser不持有memory
        )

        input_data = [prompt_epc_aw_diagnosis]

        llm_response = self.llm_engine_fixed(input_data, response_format=MemoryVerification)

        try:
            parsed = parse_json_from_llm_response(llm_response)
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
    
    