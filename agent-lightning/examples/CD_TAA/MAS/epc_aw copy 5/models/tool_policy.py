"""Single tool-selection exit for EPC_AW Solver."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.task_profile import SlotGate
from MAS.epc_aw.models.tool_router import ToolRouter, infer_subgoal_kind
from MAS.epc_aw.models.intervention import StepInterventionState


class ToolPolicyMixin:
    """Mixin: one select_tool path (resolve + policy)."""

    def select_tool(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        sub_goal: str,
        diagnostic_signal: Optional[Dict[str, Any]],
        image_path: str = "",
        tracker: Optional[StepInterventionState] = None,
    ) -> str:
        """Public single exit — same as _select_tool_for_step."""
        return self._select_tool_for_step(
            tool_name, question, target_information, sub_goal,
            diagnostic_signal, image_path, tracker,
        )

    def _suggest_alternative_tool(
        self,
        failed_tool: Optional[str],
        target_information: str,
        question: str,
    ) -> Optional[str]:
        """Pick a retrieval tool when the current one failed. Never suggest Base_Generator.

        Generic priority order — no sample-specific tool routing.
        """
        if infer_subgoal_kind(target_information, question) == "compute":
            return None
        available = set(self.planner.available_tools)
        # Prefer tools that do not require a grounded URL. Web_Search without a
        # concrete URL must not become the default alternative (no Wikipedia root).
        for candidate in (
            "Google_Search_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool",
        ):
            if candidate in available and candidate != failed_tool:
                return candidate
        return None

    def _enforce_tool_policy(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        tracker: StepInterventionState,
        diagnostic_signal: Optional[Dict[str, Any]],
        sub_goal: str = "",
    ) -> str:
        outline_tool = Planner._extract_tool_from_target(target_information)
        rec = (diagnostic_signal or {}).get("recommendation")
        blocked = set(tracker.failed_tools)

        mutations = (diagnostic_signal or {}).get("state_mutations") or {}
        for forbidden in mutations.get("forbidden_tools", []):
            blocked.add(forbidden)

        # Prefer the live planner sub_goal over a stale diagnostic_signal copy
        # so acquisition steps are not treated as compute from outline text alone.
        kind_sub = sub_goal or (diagnostic_signal or {}).get("sub_goal", "")
        kind = infer_subgoal_kind(target_information, kind_sub)
        profile = self.system_memory.get_task_profile()
        records = getattr(self.system_memory, "evidence_records", None) or []
        # Outline may say "Calculate …" before distance∧pace (or base_count) are
        # ready — keep acquisition tools available and do not force Python.
        if (
            kind == "compute"
            and profile
            and SlotGate.has_compute_slot(profile)
            and not SlotGate.needs_compute_step(profile, records)
        ):
            print(
                "\n==> 🔀 Compute deferred — input slots incomplete; "
                "keeping acquisition tools available\n"
            )
            kind = "acquisition"
        # Distance/pace hour questions: require parseable distance AND pace.
        if (
            kind == "compute"
            and hasattr(self, "_distance_pace_inputs_ready")
            and not self._distance_pace_inputs_ready(question)
        ):
            print(
                "\n==> 🔀 Compute deferred — distance/pace not both parseable; "
                "routing to acquisition\n"
            )
            kind = "acquisition"
        if kind == "acquisition" and tool_name == "Python_Coder_Tool":
            acq_alt = next(
                (
                    t for t in (
                        "Google_Search_Tool",
                        "Wikipedia_Search_Tool",
                        "Web_Search_Tool",
                    )
                    if t in self.planner.available_tools and t not in blocked
                ),
                None,
            )
            if acq_alt:
                print(
                    f"\n==> 🔀 Python blocked until compute inputs ready → {acq_alt}\n"
                )
                return acq_alt
        if kind == "compute":
            for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
                      "Base_Generator_Tool"):
                blocked.add(t)
        # elif kind == "visual":
        #     for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool",
        #               "Python_Coder_Tool", "Base_Generator_Tool"):
        #         blocked.add(t)

        # switch_tool replan: apply suggested alternative before other rules
        if rec == "switch_tool":
            failed = (diagnostic_signal or {}).get("tool")
            suggested = (diagnostic_signal or {}).get("suggested_tool")
            if not suggested or suggested == failed:
                suggested = self._suggest_alternative_tool(failed, target_information, question)
            if (
                suggested
                and suggested in self.planner.available_tools
                and suggested not in blocked
                and suggested != failed
            ):
                if tool_name != suggested:
                    print(
                        f"\n==> 🔀 switch_tool replan: {failed or tool_name} → {suggested}\n"
                    )
                return suggested

        if (
            outline_tool
            and tool_name == outline_tool
            and tool_name not in blocked
            and rec != "switch_tool"
        ):
            return tool_name

        if tool_name in blocked:
            if kind == "compute" and "Python_Coder_Tool" in self.planner.available_tools:
                return "Python_Coder_Tool"
            alt = self._suggest_alternative_tool(tool_name, target_information, question)
            if alt and alt not in blocked:
                print(f"\n==> 🚫 Blocked {tool_name} (failed/blocked) → {alt}\n")
                return alt

        combined = f"{target_information} {question}".lower()
        sub_goal = ((diagnostic_signal or {}).get("sub_goal") or "").lower()
        combined = f"{combined} {sub_goal}"
        if tool_name == "Web_Search_Tool" and "arxiv" in combined:
            figure_markers = ("figure 1", "axis label", "endpoint label", "figure caption", "three axes")
            if any(m in combined for m in figure_markers):
                alt = "Google_Search_Tool"
                if alt in self.planner.available_tools and alt not in blocked:
                    print(
                        "\n==> 🚫 Web_Search cannot extract arXiv figure/axis labels "
                        f"→ {alt}\n"
                    )
                    return alt

        # Tool capability memory is offline-only; not used in select hot path.

        preferred = mutations.get("preferred_tools") or []
        if tool_name in blocked or (
            rec == "switch_tool" and tool_name == (diagnostic_signal or {}).get("tool")
        ):
            for candidate in preferred:
                if candidate in self.planner.available_tools and candidate not in blocked:
                    print(f"\n==> 🔀 Preferred tool from state mutation: {tool_name} → {candidate}\n")
                    return candidate

        if (
            outline_tool
            and outline_tool in self.planner.available_tools
            and outline_tool not in blocked
            and tool_name != outline_tool
            and rec not in ("switch_tool",)
        ):
            print(f"\n==> 📋 Enforce outline tool: {tool_name} → {outline_tool}\n")
            return outline_tool

        return tool_name

    def _select_tool_for_step(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        sub_goal: str,
        diagnostic_signal: Optional[Dict[str, Any]],
        image_path: str = "",
        tracker: Optional[StepInterventionState] = None,
    ) -> str:
        """Single tool-selection exit: ToolRouter resolve then policy enforce."""
        tool_name = self._resolve_tool_for_step(
            tool_name, question, target_information, sub_goal, diagnostic_signal, image_path,
        )
        if tracker is not None:
            tool_name = self._enforce_tool_policy(
                tool_name, question, target_information, tracker, diagnostic_signal, sub_goal,
            )
        return tool_name

    def _resolve_tool_for_step(
        self,
        tool_name: str,
        question: str,
        target_information: str,
        sub_goal: str,
        diagnostic_signal: Optional[Dict[str, Any]],
        image_path: str = "",
    ) -> str:
        """Override planner tool via ToolRouter and diagnostic signals."""
        failed = (diagnostic_signal or {}).get("tool")
        rec = (diagnostic_signal or {}).get("recommendation")

        profile = self.system_memory.get_task_profile()
        records = self.system_memory.evidence_records
        resolved, appropriate, kind = ToolRouter.validate(
            tool_name,
            target_information,
            sub_goal,
            profile,
            records,
            self.planner.available_tools,
        )
        if not appropriate and resolved != tool_name:
            print(f"\n==> 🔀 ToolRouter redirect: {tool_name} → {resolved} (kind={kind})\n")
            tool_name = resolved

        # Defense-in-depth（防幻觉关键守卫）：即使分类器把检索子目标误判成 kind=compute，
        # 也绝不允许把 Python_Coder_Tool 强加到 acquisition 子目标上 ——
        # Python_Coder_Tool.execute(query) 会用内部 LLM 「凭 query 生成代码」并凭空捏造事实
        # （例如把本应检索的计数直接幻觉成"3 篇文章"）。改路由到真正的检索工具。
        live_kind = infer_subgoal_kind(target_information, sub_goal)
        inputs_ready = (
            not profile
            or not SlotGate.has_compute_slot(profile)
            or SlotGate.needs_compute_step(profile, records)
        )
        pace_ready = (
            not hasattr(self, "_distance_pace_inputs_ready")
            or self._distance_pace_inputs_ready(question)
        )
        if tool_name == "Python_Coder_Tool" and (
            live_kind == "acquisition" or not inputs_ready or not pace_ready
        ):
            acq_alt = next(
                (t for t in ("Google_Search_Tool", "Web_Search_Tool", "Wikipedia_Search_Tool")
                 if t in self.planner.available_tools),
                None,
            )
            if acq_alt:
                print(f"\n==> 🔀 Python-blocked for retrieval subgoal → {acq_alt}\n")
                tool_name = acq_alt

        # v3 task-level backstop. Only force Python when the *live* sub_goal is
        # compute (not acquisition) and the computed slot is still empty.
        if (
            profile
            and live_kind == "compute"
            and kind == "compute"
            and SlotGate.has_compute_slot(profile)
            and inputs_ready
            and pace_ready
        ):
            if not SlotGate.compute_slot_filled_by_computed(profile, records):
                # Require input slots when present (base_count / input_metrics).
                missing = profile.missing_slot_names(records)
                input_missing = [s for s in missing if s in ("base_count", "input_metrics")]
                if input_missing:
                    pass  # still retrieving inputs — do not force compute
                elif tool_name != "Python_Coder_Tool" and "Python_Coder_Tool" in self.planner.available_tools:
                    print(f"\n==> 🔀 Forced compute tool (compute slot unfilled): {tool_name} → Python_Coder_Tool\n")
                    tool_name = "Python_Coder_Tool"
        if kind == "visual" and image_path and tool_name not in ("Screenshot_Tool", "Vision_OCR_Tool"):
            if "Screenshot_Tool" in self.planner.available_tools:
                print(f"\n==> 🔀 Forced visual tool (image present): {tool_name} → Screenshot_Tool\n")
                tool_name = "Screenshot_Tool"

        if rec == "switch_tool" or rec == "revise_belief":
            suggested = (diagnostic_signal or {}).get("suggested_tool")
            if (
                kind != "compute"
                and suggested
                and suggested != failed
                and suggested in self.planner.available_tools
            ):
                if suggested != "Base_Generator_Tool":
                    if tool_name != suggested:
                        print(f"\n==> 🔀 Tool override ({rec}): {tool_name} → {suggested}\n")
                    return suggested
            alt = self._suggest_alternative_tool(failed or tool_name, target_information, question)
            if alt and (tool_name == failed or tool_name == "Base_Generator_Tool"):
                print(f"\n==> 🔀 Tool override: {tool_name} → {alt}\n")
                return alt

        if tool_name == "Base_Generator_Tool":
            forbidden = SlotGate.forbidden_tools(profile, records) if profile else []
            if "Base_Generator_Tool" in forbidden or infer_subgoal_kind(target_information, sub_goal) != "synthesis":
                alt = self._suggest_alternative_tool("Base_Generator_Tool", target_information, question)
                if alt:
                    print(f"\n==> 🔀 Blocked Base_Generator (phase={getattr(profile, 'phase', '?')}) → {alt}\n")
                    return alt

        text = f"{target_information} {sub_goal}".lower()
        if tool_name == "Web_Search_Tool" and (
            "arxiv.org/pdf/" in text or ("axis label" in text and "arxiv" in text)
        ):
            if "Google_Search_Tool" in self.planner.available_tools:
                print("\n==> 🔀 Axis/PDF subgoal — prefer Google_Search_Tool over Web_Search_Tool\n")
                return "Google_Search_Tool"

        return tool_name

