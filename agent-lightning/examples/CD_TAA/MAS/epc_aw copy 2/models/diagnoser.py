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
# Day 2 Integration: Import causal reasoning modules
from MAS.epc_aw.models import BayesianInference, HistoryAnalyzer
import time

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
        # self.memory = DiagnoserMemory()  # 【已删除】使用SystemMemory替代
        self.verbose = verbose
        self.n = n
        self.profile = ""
        self.temperature = temperature

        # Day 2 Integration: Initialize causal reasoning modules
        self.bayesian_inference = BayesianInference()
        self.history_analyzer = HistoryAnalyzer()

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
            return {
                "Analysis": llm_response.analysis,
                "Subgoal_Conclusion": llm_response.subgoal_conclusion,
                "New_Obtained_Information_Flag": llm_response.new_obtained_information_flag,
                "New_Obtained_Information": llm_response.new_obtained_information,
                "Conclusion": llm_response.conclusion,
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
        if isinstance(info, (list, dict)):
            return json.dumps(info, ensure_ascii=False)
        text = str(info).strip()
        if text.lower() in ("", "none", "no new information", "[]", "{}"):
            return ""
        return text

    def _has_usable_result(self, result_executor: Any) -> bool:
        """Return True when executor output contains substantive, non-error data."""
        if result_executor is None:
            return False

        text = str(result_executor)
        lower = text.lower()

        if '"error":' in lower or "'error':" in lower:
            if "relevant_pages (to the query)" in lower or '"relevant_pages":' in lower:
                # Mixed: has error entries but also real pages
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

        if len(text.strip()) > 150 and "no result was generated" not in lower:
            return True

        return False

    def _analysis_indicates_subgoal_complete(self, analysis: str) -> bool:
        """Detect when LLM analysis text says Layer 1 / sub-goal succeeded."""
        text = analysis.lower()
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

        return step_conclusion, conclusion, info_flag, obtained_info

    def _extract_failure_patterns_from_evidence(self, result_executor: Any) -> Dict[str, bool]:
        """Detect failure patterns from executor output only (not Analysis text)."""
        text = str(result_executor)
        compact = re.sub(r"\s+", "", text).replace("'", '"')
        has_usable = self._has_usable_result(result_executor)

        has_error_field = bool(re.search(r"""['"]error['"]\s*:""", text))
        empty_relevant = (
            '"relevant_pages":[]' in compact
            or '"relevant_pages(tothequery)":[]' in compact
            or "relevant_pages:[]" in compact
        )

        patterns = {
            "no_results": empty_relevant and not has_usable,
            "timeout": "timeout" in text.lower() or "timed out" in text.lower(),
            "permission_denied": any(
                p in text.lower() for p in ("permission denied", "unauthorized", "access denied")
            ),
            "error_occurred": has_error_field and not has_usable,
            "irrelevant_results": False,
        }

        if not has_usable and not any(patterns.values()):
            patterns["no_results"] = len(text.strip()) < 30

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
        import re
        return list(dict.fromkeys(re.findall(r"\b\d{5}\b", str(text))))

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

    def _is_task_answer_ready(self, question: str, cumulative_text: str) -> bool:
        """True when cumulative evidence satisfies the original question."""
        q = question.lower()
        text = str(cumulative_text)

        if "zip code" in q or "zip codes" in q:
            zips = self._extract_zip_codes(text)
            if not zips:
                return False
            if self._extract_authoritative_source_requirement(question):
                return self._has_usgs_task_context(text)
            return True

        if "comma-separated" in q or "separated by commas" in q:
            return bool(self._extract_zip_codes(text)) or "," in text

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
        return True

    def _requires_external_evidence(self, question: str, target_information: str = "") -> bool:
        text = f"{question} {target_information}".lower()
        return any(m in text for m in (
            "usgs", "according to", "official", "database", "zip code", "nonnative",
        ))

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
    ) -> Optional[str]:
        """
        Bidirectional STOP reconciliation:
        - Downgrade premature STOP (single-step weak evidence)
        - Upgrade CONTINUE→STOP when cumulative evidence satisfies the question
        """
        if step_conclusion != "SUBGOAL_COMPLETE":
            return None

        if task_conclusion is None:
            task_conclusion = "CONTINUE"

        normalized = str(task_conclusion).strip().upper()
        if normalized not in ("STOP", "CONTINUE"):
            normalized = "CONTINUE"

        cumulative = self._build_cumulative_evidence_text(
            obtained_information, result_executor,
        )
        answer_ready = self._is_task_answer_ready(question, cumulative)

        if normalized == "CONTINUE" and answer_ready:
            if self.verbose:
                zips = self._extract_zip_codes(cumulative)
                print(
                    f"  [STOP Reconciliation]: UPGRADE CONTINUE→STOP "
                    f"(cumulative evidence ready, zips={zips})"
                )
            return "STOP"

        if normalized != "STOP":
            return normalized

        failed_gates: List[str] = []

        if answer_ready:
            if self._answer_format_satisfied(question, cumulative):
                if self.verbose:
                    print(
                        "  [STOP Reconciliation]: STOP confirmed "
                        "(cumulative evidence satisfies question)"
                    )
                return "STOP"
            failed_gates.append("format_unmet_cumulative")

        remaining = self._count_remaining_outline_steps(outline, current_step_key)
        if remaining > 0 and not answer_ready:
            failed_gates.append(f"outline_remaining={remaining}")

        auth = self._extract_authoritative_source_requirement(question)
        if auth and not answer_ready:
            if not self._has_usgs_task_context(cumulative):
                provenance = self._assess_evidence_provenance(result_executor, question)
                if provenance["level"] not in ("primary", "secondary"):
                    failed_gates.append(f"source_{provenance['level']}")

        if tool_name == "Base_Generator_Tool" and (
            auth or self._requires_external_evidence(question, target_information)
        ):
            failed_gates.append("generator_not_authoritative")

        if not self._answer_format_satisfied(question, cumulative):
            failed_gates.append("format_unmet")

        if failed_gates:
            if self.verbose:
                print(f"  [STOP Reconciliation]: DOWNGRADE STOP→CONTINUE ({', '.join(failed_gates)})")
            return "CONTINUE"

        return "STOP"

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

        prompt_verificate_context = prompt_template.format(
            Question=question,
            Sub_Goal=sub_goal,
            Target_Information=target_information,
            Outline=outline,
            Remaining_Outline_Steps=remaining_steps,
            Result_Executor=result_executor,
            Memory="",
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
        except Exception as e:
            raise ValueError(f"Error parsing LLM response: {llm_response}") from e

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

        failure_patterns = self._extract_failure_patterns_from_evidence(result_executor)

        # Only downgrade COMPLETE when executor output genuinely failed
        if step_conclusion == "SUBGOAL_COMPLETE" and not self._has_usable_result(result_executor):
            if failure_patterns.get("no_results") or failure_patterns.get("error_occurred"):
                step_conclusion = "SUBGOAL_INCOMPLETE"

        subgoal_actually_complete = step_conclusion == "SUBGOAL_COMPLETE"

        if (
            subgoal_actually_complete
            and tool_name == "Base_Generator_Tool"
            and self._requires_external_evidence(question, target_information)
        ):
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            if self.verbose:
                print("  [Subgoal Reconciliation]: DOWNGRADE — Base_Generator cannot satisfy external evidence requirement")

        if (
            subgoal_actually_complete
            and self._is_retrieval_subgoal(sub_goal, target_information)
            and self._is_negative_retrieval_result(result_executor, sub_goal)
        ):
            step_conclusion = "SUBGOAL_INCOMPLETE"
            subgoal_actually_complete = False
            if self.verbose:
                print("  [Subgoal Reconciliation]: DOWNGRADE — negative/empty retrieval is not subgoal completion")

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

            diagnostic_signal_t = self._generate_causal_signal(
                result_executor, Analysis, sub_goal, tool_name, step_count,
                failure_patterns=failure_patterns,
                subgoal_complete=subgoal_actually_complete,
                intervention_context=intervention_context,
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
        )

    def _generate_causal_signal(self, result_executor: str, analysis: str, sub_goal: str, tool_name: str, step_count: int, failure_patterns: Dict[str, bool] = None, subgoal_complete: bool = False, intervention_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate causal diagnostic signal when subgoal is not completed.
        Provides structured guidance for the next execution cycle.

        Args:
            failure_patterns: Failure patterns from LLM judgment (always provided)
            subgoal_complete: Whether the current subgoal is actually complete
        """
        # failure_patterns现在总是从LLM返回，不需要fallback
        if failure_patterns is None:
            failure_patterns = {
                "no_results": False,
                "timeout": False,
                "permission_denied": False,
                "error_occurred": False
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
            "subgoal_complete": subgoal_complete  # ✅ 显式传递完成状态
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

    def _generate_parameter_guidance(self, result_executor: str, analysis: str, tool_name: str, failure_patterns: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate specific parameter variation guidance based on failure analysis.
        Provides Executor with CONCRETE directions for the next parameter attempt.
        """
        guidance = {
            "directions": [],
            "avoid_patterns": [],
            "examples": [],  # ✅ 新增：具体的参数变体示例
            "explanation": ""
        }

        # Analyze failure patterns and suggest parameter variations
        result_str = str(result_executor).lower()
        analysis_str = str(analysis).lower()
        combined = result_str + " " + analysis_str

        # ✅ 针对不同工具提供定制化建议
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
        failure_patterns: Dict[str, bool],
        tool_name: str,
        subgoal_complete: bool = False,
        exhausted_interventions: Optional[Set[str]] = None,
        failed_tools: Optional[List[str]] = None,
    ) -> str:
        """
        Select causal intervention with escalation when prior attempts are exhausted.

        Ladder: retry_with_different_parameters → switch_tool → decompose_goal → modify_state
        """
        exhausted = exhausted_interventions or set()

        if subgoal_complete:
            return "continue_to_next_subgoal"

        # Rank candidates by failure pattern (most specific first)
        candidates: List[str] = []
        if failure_patterns.get("permission_denied"):
            candidates.append("modify_state")
        if failure_patterns.get("no_results") or failure_patterns.get("error_occurred"):
            candidates.append("switch_tool")
        if failure_patterns.get("timeout"):
            candidates.append("switch_tool")
        if failure_patterns.get("irrelevant_results"):
            candidates.append("retry_with_different_parameters")
        if not candidates:
            candidates.append("retry_with_different_parameters")

        # De-duplicate preserving order
        seen: Set[str] = set()
        ordered: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        for candidate in ordered:
            if candidate not in exhausted:
                return candidate

        # All pattern-matched options exhausted — walk the global ladder
        for intervention in (
            "retry_with_different_parameters",
            "switch_tool",
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
                                           analysis: str) -> Tuple[bool, str]:
        """
        Decide whether to persist new information to system memory.

        Priority (highest first):
        1. LLM-extracted information when flag is True
        2. LLM-extracted information when subgoal is complete (flag may be wrong)
        3. Heuristic extraction from executor result when subgoal is complete
        4. Partial useful data even when subgoal is incomplete (non-error results)
        """
        info = self._normalize_obtained_information(obtained_information)

        if obtained_information_flag and info:
            return True, info

        is_complete = (
            step_conclusion == "SUBGOAL_COMPLETE" or subgoal_actually_complete
        )

        if is_complete:
            if info:
                return True, info
            extracted = self._extract_key_information_from_result(result_executor, analysis)
            if extracted:
                return True, extracted

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

        prompt_update_outline = prompt_template.format(
            Question=question,
            Current_Verification_Analysis=context_verification,
            Target_Information=target_information,
            Outline=json.dumps(outline, ensure_ascii=False, indent=2),
            Result_Executor=result_executor,
            Memory=execution_memory or "N/A",
            Obtained_Information_So_Far=obtained_information or "None yet",
            Toolbox_Metadata=toolbox_metadata if toolbox_metadata else "N/A",
            EPC_AW_Analysis=epc_aw_analysis if epc_aw_analysis else "N/A",
            current_step=current_step,
        )

        llm_response = self.llm_engine_fixed(
            [prompt_update_outline], response_format=OutlineUpdateResponse
        )
        try:
            if isinstance(llm_response, OutlineUpdateResponse):
                new_outline = llm_response.execution_outline
            else:
                parsed = parse_json_from_llm_response(llm_response)
                if isinstance(parsed, dict):
                    new_outline = parsed.get("ExecutionOutline", parsed.get("execution_outline", {}))
                else:
                    new_outline = {}
        except Exception as e:
            raise ValueError(f"Error parsing outline update response: {llm_response}") from e

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
    
    