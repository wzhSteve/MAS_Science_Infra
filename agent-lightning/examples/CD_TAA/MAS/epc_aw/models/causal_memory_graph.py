"""
Causal Memory Graph — explicit node/edge store for CD-TTA execution chains.

Canonical chain: State → Subgoal → Tool → Parameter → Outcome → NextState
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class GraphNodeType(str, Enum):
    STATE = "State"
    SUBGOAL = "Subgoal"
    TOOL = "Tool"
    PARAMETER = "Parameter"
    OUTCOME = "Outcome"
    EVIDENCE = "Evidence"
    INTERVENTION = "Intervention"


class ExecutionEdgeType(str, Enum):
    STATE_REQUIRES_SUBGOAL = "state_requires_subgoal"
    SUBGOAL_SELECTS_TOOL = "subgoal_selects_tool"
    TOOL_INVOKES_PARAMETER = "tool_invokes_parameter"
    PARAMETER_YIELDS_OUTCOME = "parameter_yields_outcome"
    OUTCOME_TRANSITIONS_STATE = "outcome_transitions_state"
    OUTCOME_EXTRACTS_EVIDENCE = "outcome_extracts_evidence"
    EVIDENCE_CONTRIBUTES_STATE = "evidence_contributes_state"
    PARAMETER_CONTRASTS = "parameter_contrasts"
    OUTCOME_DIAGNOSES = "outcome_diagnoses"


def _short_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def abstract_subgoal_type(subgoal: str) -> str:
    """Rule-based subgoal keyword (offline index)."""
    text = (subgoal or "").strip().lower()
    if not text:
        return "empty_subgoal"
    rules = [
        (r"scientific name|species|identify.*fish|clownfish", "identify_species"),
        (r"usgs|nonnative|non-indigenous|invasive|occurrence|nas", "fetch_usgs_occurrence"),
        (r"zip code|postal|address", "fetch_zip_code"),
        (r"confirm|verify|validate", "verify_fact"),
        (r"search|find|lookup|retrieve|get ", "retrieve_information"),
        (r"extract|parse", "extract_field"),
        (r"wikipedia|encyclopedia", "encyclopedia_lookup"),
    ]
    for pattern, label in rules:
        if re.search(pattern, text):
            return label
    return f"subgoal_{_short_hash(text, 6)}"


def abstract_state_type(state_label: str) -> str:
    """Normalize coarse state label to keyword."""
    label = (state_label or "").strip()
    if label.startswith("初始状态"):
        return "initial_no_info"
    if label.startswith("信息完备"):
        return "complete_pending_final"
    if "首条证据" in label:
        return "partial_first_evidence"
    if "部分完成" in label:
        return "partial_multi_evidence"
    return f"state_{_short_hash(label, 6)}"


def parse_parameter_command(raw_command: str, tool_name: str = "") -> Dict[str, Any]:
    """Structured parameter from executor command string."""
    raw = str(raw_command or "").strip()
    parsed: Dict[str, Any] = {"raw_command": raw, "tool_name": tool_name}

    query_match = re.search(r'query\s*=\s*["\']([^"\']*)["\']', raw)
    if query_match:
        parsed["query"] = query_match.group(1).strip()

    url_match = re.search(r'url\s*=\s*["\']([^"\']*)["\']', raw)
    if url_match:
        parsed["url"] = url_match.group(1).strip()

    for key in ("site", "max_results", "timeout"):
        m = re.search(rf'{key}\s*=\s*["\']?([^,"\']+)["\']?', raw)
        if m:
            parsed[key] = m.group(1).strip()

    fingerprint_base = parsed.get("query") or parsed.get("url") or raw
    normalized = re.sub(r"\s+", " ", fingerprint_base.lower()).strip()
    parsed["fingerprint"] = f"{tool_name}|{normalized}" if tool_name else normalized
    return parsed


class CausalMemoryGraph:
    """
    Task-scoped causal DAG: nodes + typed edges + keyword indexes.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.indexes: Dict[str, Dict[str, List[str]]] = {
            "by_tool": {},
            "by_subgoal_type": {},
            "by_state_type": {},
            "by_parameter_fingerprint": {},
        }
        self.task_id: Optional[str] = None
        self.current_state_id: Optional[str] = None
        self._state_seq: int = 0
        self._active_outline: Dict[str, Dict[str, str]] = {}
        self._last_parameter_id: Optional[str] = None
        self._last_failed_parameter_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Node / edge helpers
    # ------------------------------------------------------------------

    def _index_node(self, node_id: str, node: Dict[str, Any]) -> None:
        ntype = node.get("type")
        if ntype == GraphNodeType.TOOL.value:
            name = node.get("tool_name", "")
            self.indexes["by_tool"].setdefault(name, [])
            if node_id not in self.indexes["by_tool"][name]:
                self.indexes["by_tool"][name].append(node_id)
        elif ntype == GraphNodeType.SUBGOAL.value:
            sg = node.get("subgoal_type", "")
            self.indexes["by_subgoal_type"].setdefault(sg, [])
            if node_id not in self.indexes["by_subgoal_type"][sg]:
                self.indexes["by_subgoal_type"][sg].append(node_id)
        elif ntype == GraphNodeType.STATE.value:
            st = node.get("state_type", "")
            self.indexes["by_state_type"].setdefault(st, [])
            if node_id not in self.indexes["by_state_type"][st]:
                self.indexes["by_state_type"][st].append(node_id)
        elif ntype == GraphNodeType.PARAMETER.value:
            fp = node.get("fingerprint", "")
            if fp:
                self.indexes["by_parameter_fingerprint"].setdefault(fp, [])
                if node_id not in self.indexes["by_parameter_fingerprint"][fp]:
                    self.indexes["by_parameter_fingerprint"][fp].append(node_id)

    def add_node(self, node_id: str, node_type: GraphNodeType, **fields: Any) -> str:
        node = {"type": node_type.value, **fields}
        self.nodes[node_id] = node
        self._index_node(node_id, node)
        return node_id

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: ExecutionEdgeType,
        **attrs: Any,
    ) -> None:
        self.edges.append({"src": src, "dst": dst, "type": edge_type.value, **attrs})

    def ensure_task(self, task_id: str) -> None:
        if self.task_id != task_id:
            self.task_id = task_id
            self._state_seq = 0
            self.current_state_id = None
            self._active_outline.clear()

    def ensure_initial_state(
        self,
        state_label: str,
        evidence_ids: Optional[List[str]] = None,
        outline_remaining: int = 0,
    ) -> str:
        if self.current_state_id and self.current_state_id in self.nodes:
            return self.current_state_id

        state_type = abstract_state_type(state_label)
        node_id = f"st:{self.task_id or 'task'}:{self._state_seq}"
        self.add_node(
            node_id,
            GraphNodeType.STATE,
            state_type=state_type,
            state_label=state_label,
            evidence_ids=list(evidence_ids or []),
            outline_remaining=outline_remaining,
        )
        self.current_state_id = node_id
        return node_id

    def _subgoal_id(self, subgoal_text: str) -> str:
        return f"sg:{_short_hash(subgoal_text)}"

    def _tool_id(self, tool_name: str) -> str:
        return f"tool:{tool_name}"

    def _parameter_id(self, parsed: Dict[str, Any]) -> str:
        return f"pm:{_short_hash(parsed.get('fingerprint', parsed.get('raw_command', '')))}"

    def _outcome_id(self, exec_step: int) -> str:
        return f"oc:{exec_step}"

    def _evidence_node_id(self, evidence_id: str) -> str:
        return f"ev:{evidence_id}"

    # ------------------------------------------------------------------
    # Execution recording
    # ------------------------------------------------------------------

    def begin_outline_step(
        self,
        outline_step: str,
        subgoal: str,
        state_label: str,
        evidence_ids: Optional[List[str]] = None,
        outline_remaining: int = 0,
        *,
        required_fact_names: Optional[List[str]] = None,
        preferred_tools: Optional[List[str]] = None,
        acquisition: bool = True,
    ) -> Tuple[str, str]:
        """Ensure State → Subgoal chain for a new outline step."""
        state_id = self.ensure_initial_state(state_label, evidence_ids, outline_remaining)
        sg_id = self._subgoal_id(subgoal)
        subgoal_type = abstract_subgoal_type(subgoal)

        if sg_id not in self.nodes:
            self.add_node(
                sg_id,
                GraphNodeType.SUBGOAL,
                subgoal_type=subgoal_type,
                text=subgoal,
                required_fact_names=list(required_fact_names or []),
                preferred_tools=list(preferred_tools or []),
                acquisition=bool(acquisition),
                satisfied=False,
            )
        else:
            node = self.nodes[sg_id]
            if required_fact_names is not None:
                node["required_fact_names"] = list(required_fact_names)
            if preferred_tools is not None:
                node["preferred_tools"] = list(preferred_tools)
            node.setdefault("acquisition", bool(acquisition))
            node.setdefault("satisfied", False)

        if not any(
            e["src"] == state_id and e["dst"] == sg_id
            and e["type"] == ExecutionEdgeType.STATE_REQUIRES_SUBGOAL.value
            for e in self.edges
        ):
            self.add_edge(
                state_id, sg_id, ExecutionEdgeType.STATE_REQUIRES_SUBGOAL,
                outline_step=outline_step,
            )

        self._active_outline[outline_step] = {
            "state_id": state_id,
            "subgoal_id": sg_id,
            "subgoal_type": subgoal_type,
            "subgoal_text": subgoal,
        }
        return state_id, sg_id

    def record_execution(
        self,
        *,
        outline_step: str,
        exec_step: int,
        subgoal: str,
        tool_name: str,
        raw_command: str,
        success: bool,
        result_preview: str = "",
        symptom: Optional[str] = None,
        state_label: str = "",
        evidence_ids: Optional[List[str]] = None,
        outline_remaining: int = 0,
        attempt_seq: int = 1,
        intervention: Optional[str] = None,
        L2_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Record State→Subgoal→Tool→Parameter→Outcome chain for one exec attempt.
        Returns dict of node ids created.
        """
        state_id, sg_id = self.begin_outline_step(
            outline_step, subgoal, state_label, evidence_ids, outline_remaining,
        )

        tool_id = self._tool_id(tool_name)
        if tool_id not in self.nodes:
            self.add_node(tool_id, GraphNodeType.TOOL, tool_name=tool_name)

        self.add_edge(
            sg_id, tool_id, ExecutionEdgeType.SUBGOAL_SELECTS_TOOL,
            outline_step=outline_step, attempt=attempt_seq,
        )

        parsed = parse_parameter_command(raw_command, tool_name)
        pm_id = self._parameter_id(parsed)
        if pm_id not in self.nodes:
            self.add_node(
                pm_id,
                GraphNodeType.PARAMETER,
                tool_name=tool_name,
                fingerprint=parsed.get("fingerprint"),
                query=parsed.get("query"),
                url=parsed.get("url"),
                raw_command=parsed.get("raw_command", "")[:300],
            )

        self.add_edge(
            tool_id, pm_id, ExecutionEdgeType.TOOL_INVOKES_PARAMETER,
            outline_step=outline_step, attempt=attempt_seq,
        )

        oc_id = self._outcome_id(exec_step)
        self.add_node(
            oc_id,
            GraphNodeType.OUTCOME,
            success=success,
            symptom=symptom,
            preview=str(result_preview)[:500],
            exec_step=exec_step,
            attempt_seq=attempt_seq,
        )
        self.add_edge(
            pm_id, oc_id, ExecutionEdgeType.PARAMETER_YIELDS_OUTCOME,
            success=success,
        )

        if not success and intervention:
            iv_id = f"iv:{outline_step}:{exec_step}"
            self.add_node(
                iv_id, GraphNodeType.INTERVENTION,
                recommendation=intervention,
            )
            self.add_edge(oc_id, iv_id, ExecutionEdgeType.OUTCOME_DIAGNOSES)

        if L2_diagnosis:
            self.nodes[oc_id]["L2_diagnosis"] = L2_diagnosis

        if not success:
            self._last_failed_parameter_id = pm_id
        self._last_parameter_id = pm_id

        return {
            "state_id": state_id,
            "subgoal_id": sg_id,
            "tool_id": tool_id,
            "parameter_id": pm_id,
            "outcome_id": oc_id,
        }

    def record_parameter_contrast(self, failed_pm_id: str, success_pm_id: str, insight: str = "") -> None:
        self.add_edge(
            failed_pm_id, success_pm_id, ExecutionEdgeType.PARAMETER_CONTRASTS,
            insight=insight or "parameter_variation_succeeded",
        )

    def transition_state_on_success(
        self,
        outcome_id: str,
        new_state_label: str,
        evidence_ids: Optional[List[str]] = None,
        outline_remaining: int = 0,
        information_delta: str = "",
    ) -> str:
        """Outcome(success) → NextState; optional evidence links."""
        self._state_seq += 1
        state_type = abstract_state_type(new_state_label)
        new_state_id = f"st:{self.task_id or 'task'}:{self._state_seq}"
        self.add_node(
            new_state_id,
            GraphNodeType.STATE,
            state_type=state_type,
            state_label=new_state_label,
            evidence_ids=list(evidence_ids or []),
            outline_remaining=outline_remaining,
            information_delta=information_delta,
        )
        self.add_edge(
            outcome_id, new_state_id, ExecutionEdgeType.OUTCOME_TRANSITIONS_STATE,
        )

        for ev_id in evidence_ids or []:
            ev_node = self._evidence_node_id(ev_id)
            if ev_node not in self.nodes:
                self.add_node(ev_node, GraphNodeType.EVIDENCE, evidence_id=ev_id)
            self.add_edge(outcome_id, ev_node, ExecutionEdgeType.OUTCOME_EXTRACTS_EVIDENCE)
            self.add_edge(ev_node, new_state_id, ExecutionEdgeType.EVIDENCE_CONTRIBUTES_STATE)

        self.current_state_id = new_state_id
        return new_state_id

    def link_evidence_to_outcome(
        self,
        outcome_id: str,
        evidence_id: str,
        *,
        verification_status: str = "satisfied",
        live: bool = True,
        fact_names: Optional[List[str]] = None,
        claim_type: str = "fact",
    ) -> None:
        ev_node = self._evidence_node_id(evidence_id)
        if ev_node not in self.nodes:
            self.add_node(
                ev_node,
                GraphNodeType.EVIDENCE,
                evidence_id=evidence_id,
                verification_status=verification_status,
                live=live,
                fact_names=list(fact_names or []),
                claim_type=claim_type,
            )
        else:
            node = self.nodes[ev_node]
            node["verification_status"] = verification_status
            node["live"] = live
            node["fact_names"] = list(fact_names or node.get("fact_names") or [])
            node["claim_type"] = claim_type
        self.add_edge(outcome_id, ev_node, ExecutionEdgeType.OUTCOME_EXTRACTS_EVIDENCE)
        if self.current_state_id and live:
            self.add_edge(
                ev_node, self.current_state_id, ExecutionEdgeType.EVIDENCE_CONTRIBUTES_STATE
            )

    def mark_evidence_invalidated(self, evidence_id: str, reason: str = "") -> None:
        ev_node = self._evidence_node_id(evidence_id)
        node = self.nodes.get(ev_node)
        if not node:
            return
        node["live"] = False
        node["verification_status"] = "invalidated"
        if reason:
            node["invalidate_reason"] = reason

    def update_state_valid_facts(
        self,
        state_id: str,
        valid_facts: Dict[str, Any],
    ) -> None:
        """Attach compact valid-fact summary to a State node (not raw evidence text)."""
        node = self.nodes.get(state_id)
        if not node or node.get("type") != GraphNodeType.STATE.value:
            return
        # Keep summaries short and abstract-ish for graph dumps.
        compact: Dict[str, Any] = {}
        for name, value in (valid_facts or {}).items():
            text = str(value)
            compact[str(name)] = text if len(text) <= 160 else text[:157] + "..."
        node["valid_facts"] = compact
        if compact:
            node["state_label"] = node.get("state_label") or (
                f"部分完成_已验证{len(compact)}项事实"
            )
        # Mark subgoals whose required facts are now met.
        for sg_id, sg in self.nodes.items():
            if sg.get("type") != GraphNodeType.SUBGOAL.value:
                continue
            if self.subgoal_requirements_met(sg_id, compact):
                sg["satisfied"] = True

    def attach_subgoal_contract(
        self,
        subgoal_text: str,
        *,
        required_fact_names: Optional[List[str]] = None,
        preferred_tools: Optional[List[str]] = None,
        acquisition: bool = True,
    ) -> str:
        """Create/update a Subgoal node with VerificationSpec-like contract fields."""
        sg_id = self._subgoal_id(subgoal_text)
        if sg_id not in self.nodes:
            self.add_node(
                sg_id,
                GraphNodeType.SUBGOAL,
                subgoal_type=abstract_subgoal_type(subgoal_text),
                text=subgoal_text,
                required_fact_names=list(required_fact_names or []),
                preferred_tools=list(preferred_tools or []),
                acquisition=bool(acquisition),
                satisfied=False,
            )
        else:
            node = self.nodes[sg_id]
            if required_fact_names is not None:
                node["required_fact_names"] = list(required_fact_names)
            if preferred_tools is not None:
                node["preferred_tools"] = list(preferred_tools)
            node["acquisition"] = bool(acquisition)
        return sg_id

    def subgoal_requirements_met(
        self,
        sg_id: str,
        valid_facts: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """True when required_fact_names ⊆ valid_facts (or subgoal already satisfied)."""
        node = self.nodes.get(sg_id) or {}
        if node.get("type") != GraphNodeType.SUBGOAL.value:
            return False
        if node.get("satisfied"):
            return True
        required = [str(x) for x in (node.get("required_fact_names") or []) if x]
        if not required:
            # No explicit contract: treat as met only after a successful state transition
            # linked from an outcome that extracted evidence for this subgoal path.
            return self._subgoal_has_successful_outcome(sg_id)
        facts = valid_facts
        if facts is None:
            state = self.nodes.get(self.current_state_id or "") or {}
            facts = state.get("valid_facts") or {}
        return all(name in facts and facts[name] not in (None, "") for name in required)

    def _subgoal_has_successful_outcome(self, sg_id: str) -> bool:
        tool_ids = [
            e["dst"] for e in self.edges
            if e["src"] == sg_id
            and e["type"] == ExecutionEdgeType.SUBGOAL_SELECTS_TOOL.value
        ]
        for tool_id in tool_ids:
            for pm_id in self._parameters_for_tool(tool_id):
                for edge in self.edges:
                    if (
                        edge["src"] == pm_id
                        and edge["type"] == ExecutionEdgeType.PARAMETER_YIELDS_OUTCOME.value
                        and self._outcome_success(edge["dst"])
                    ):
                        return True
        return False

    def open_acquisition_subgoals(
        self,
        valid_facts: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return unsatisfied acquisition Subgoal nodes still needing facts."""
        open_items: List[Dict[str, Any]] = []
        for sg_id, node in self.nodes.items():
            if node.get("type") != GraphNodeType.SUBGOAL.value:
                continue
            if node.get("acquisition") is False:
                continue
            if self.subgoal_requirements_met(sg_id, valid_facts):
                continue
            open_items.append(
                {
                    "id": sg_id,
                    "text": node.get("text", ""),
                    "subgoal_type": node.get("subgoal_type", ""),
                    "required_fact_names": list(node.get("required_fact_names") or []),
                    "preferred_tools": list(node.get("preferred_tools") or []),
                }
            )
        return open_items

    def project_outline_from_open_subgoals(
        self,
        valid_facts: Optional[Dict[str, Any]] = None,
        *,
        max_steps: int = 4,
    ) -> Dict[str, str]:
        """Build a compatibility outline projection from open acquisition subgoals."""
        outline: Dict[str, str] = {}
        for index, item in enumerate(self.open_acquisition_subgoals(valid_facts)[:max_steps], start=1):
            tools = item.get("preferred_tools") or []
            tool_hint = tools[0] if tools else "an appropriate tool"
            req = item.get("required_fact_names") or []
            expected = ", ".join(req) if req else "verified facts for this subgoal"
            outline[str(index)] = (
                f"Target Information: {item.get('text') or item.get('subgoal_type')} "
                f"Operation Details: {tool_hint}. "
                f"Expected Output: {expected}."
            )
        return outline

    def count_live_evidence_nodes(self) -> int:
        return sum(
            1
            for node in self.nodes.values()
            if node.get("type") == GraphNodeType.EVIDENCE.value and node.get("live", True)
        )

    def count_diagnosis_edges(self) -> int:
        return sum(
            1
            for edge in self.edges
            if edge.get("type") == ExecutionEdgeType.OUTCOME_DIAGNOSES.value
        )

    def current_valid_facts(self) -> Dict[str, Any]:
        state = self.nodes.get(self.current_state_id or "") or {}
        facts = state.get("valid_facts") or {}
        return dict(facts) if isinstance(facts, dict) else {}

    # ------------------------------------------------------------------
    # Query API (Planner / Executor)
    # ------------------------------------------------------------------

    def get_paths_for_subgoal_type(self, subgoal_type: str) -> List[List[str]]:
        """Return node-id paths from State to Outcome for a subgoal keyword."""
        sg_nodes = self.indexes["by_subgoal_type"].get(subgoal_type, [])
        paths: List[List[str]] = []
        for sg_id in sg_nodes:
            for edge in self.edges:
                if edge["dst"] == sg_id and edge["type"] == ExecutionEdgeType.STATE_REQUIRES_SUBGOAL.value:
                    state_id = edge["src"]
                    paths.extend(self._extend_path([state_id, sg_id]))
        return paths

    def _extend_path(self, prefix: List[str]) -> List[List[str]]:
        last = prefix[-1]
        extended: List[List[str]] = []
        for edge in self.edges:
            if edge["src"] == last:
                extended.append(prefix + [edge["dst"]])
        return extended if extended else [prefix]

    def get_tools_for_context(
        self, state_type: str, subgoal_type: str,
    ) -> List[Dict[str, Any]]:
        """Online: tools used in this task for matching keywords."""
        sg_ids = set(self.indexes["by_subgoal_type"].get(subgoal_type, []))
        tools: Dict[str, Dict[str, Any]] = {}
        for edge in self.edges:
            if edge["type"] != ExecutionEdgeType.SUBGOAL_SELECTS_TOOL.value:
                continue
            if edge["src"] not in sg_ids:
                continue
            tool_id = edge["dst"]
            tool_node = self.nodes.get(tool_id, {})
            name = tool_node.get("tool_name", tool_id)
            successes = sum(
                1 for e in self.edges
                if e["src"] in self._parameters_for_tool(tool_id)
                and self._outcome_success(e["dst"])
            )
            attempts = sum(
                1 for e in self.edges
                if e["src"] in self._parameters_for_tool(tool_id)
            )
            tools[name] = {
                "tool_name": name,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts if attempts else 0.0,
            }
        return sorted(tools.values(), key=lambda x: -x["success_rate"])

    def _parameters_for_tool(self, tool_id: str) -> List[str]:
        return [
            e["dst"] for e in self.edges
            if e["src"] == tool_id and e["type"] == ExecutionEdgeType.TOOL_INVOKES_PARAMETER.value
        ]

    def _outcome_success(self, outcome_id: str) -> bool:
        node = self.nodes.get(outcome_id, {})
        return bool(node.get("success"))

    def get_failed_parameters(
        self, subgoal_type: str, tool_name: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        tool_id = self._tool_id(tool_name)
        pm_ids = self._parameters_for_tool(tool_id)
        for pm_id in pm_ids:
            pm = self.nodes.get(pm_id, {})
            for edge in self.edges:
                if edge["src"] != pm_id or edge["type"] != ExecutionEdgeType.PARAMETER_YIELDS_OUTCOME.value:
                    continue
                oc = self.nodes.get(edge["dst"], {})
                if oc.get("success"):
                    continue
                sg_match = any(
                    e["dst"] == tool_id
                    and e["type"] == ExecutionEdgeType.SUBGOAL_SELECTS_TOOL.value
                    and self.nodes.get(e["src"], {}).get("subgoal_type") == subgoal_type
                    for e in self.edges
                )
                if sg_match:
                    results.append({
                        "parameter_id": pm_id,
                        "query": pm.get("query"),
                        "raw_command": pm.get("raw_command"),
                        "symptom": oc.get("symptom"),
                    })
        return results

    def find_tools_for_subgoal(self, subgoal: str) -> List[str]:
        """Adapter for legacy CausalInference API."""
        sg_type = abstract_subgoal_type(subgoal)
        return [t["tool_name"] for t in self.get_tools_for_context("", sg_type)]

    def get_success_rate(self, tool: str, subgoal: str) -> float:
        """Adapter for legacy CausalInference API."""
        sg_type = abstract_subgoal_type(subgoal)
        for entry in self.get_tools_for_context("", sg_type):
            if entry["tool_name"] == tool:
                return entry["success_rate"]
        return 0.5

    def get_successful_parameters(
        self, subgoal_type: str, tool_name: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        tool_id = self._tool_id(tool_name)
        for pm_id in self._parameters_for_tool(tool_id):
            pm = self.nodes.get(pm_id, {})
            for edge in self.edges:
                if edge["src"] != pm_id:
                    continue
                oc = self.nodes.get(edge["dst"], {})
                if not oc.get("success"):
                    continue
                results.append({
                    "parameter_id": pm_id,
                    "query": pm.get("query"),
                    "raw_command": pm.get("raw_command"),
                })
        return results

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "current_state_id": self.current_state_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "indexes": self.indexes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalMemoryGraph":
        g = cls()
        g.task_id = data.get("task_id")
        g.current_state_id = data.get("current_state_id")
        g.nodes = data.get("nodes", {})
        g.edges = data.get("edges", [])
        g.indexes = data.get("indexes") or {
            "by_tool": {}, "by_subgoal_type": {}, "by_state_type": {},
            "by_parameter_fingerprint": {},
        }
        if g.current_state_id:
            m = re.search(r":(\d+)$", g.current_state_id)
            if m:
                g._state_seq = int(m.group(1))
        return g

    def export_linear_traces(self) -> List[Dict[str, Any]]:
        """Backward-compatible factor/effects view derived from the graph."""
        traces: Dict[str, Dict[str, Any]] = {}

        for edge in self.edges:
            if edge["type"] != ExecutionEdgeType.STATE_REQUIRES_SUBGOAL.value:
                continue
            outline_step = str(edge.get("outline_step", ""))
            sg_id = edge["dst"]
            sg = self.nodes.get(sg_id, {})
            st = self.nodes.get(edge["src"], {})
            key = f"{outline_step}::{sg_id}"
            if key not in traces:
                traces[key] = {
                    "outline_step": outline_step,
                    "factor": {
                        "state": st.get("state_label", st.get("state_type", "")),
                        "state_type": st.get("state_type", ""),
                        "subgoal": sg.get("text", ""),
                        "subgoal_type": sg.get("subgoal_type", ""),
                    },
                    "effects": [],
                    "L2_diagnosis": {},
                    "status": "in_progress",
                }

        attempt_counters: Dict[str, int] = {}

        for edge in self.edges:
            if edge["type"] != ExecutionEdgeType.PARAMETER_YIELDS_OUTCOME.value:
                continue
            pm_id = edge["src"]
            oc_id = edge["dst"]
            oc = self.nodes.get(oc_id, {})
            pm = self.nodes.get(pm_id, {})

            tool_id = None
            for te in self.edges:
                if te["dst"] == pm_id and te["type"] == ExecutionEdgeType.TOOL_INVOKES_PARAMETER.value:
                    tool_id = te["src"]
                    break
            tool_name = self.nodes.get(tool_id or "", {}).get("tool_name", "")

            sg_id = None
            outline_step = ""
            for se in self.edges:
                if se["dst"] == tool_id and se["type"] == ExecutionEdgeType.SUBGOAL_SELECTS_TOOL.value:
                    sg_id = se["src"]
                    for sre in self.edges:
                        if sre["dst"] == sg_id and sre["type"] == ExecutionEdgeType.STATE_REQUIRES_SUBGOAL.value:
                            outline_step = str(sre.get("outline_step", ""))
                            break
                    break

            key = f"{outline_step}::{sg_id}"
            if key not in traces:
                continue

            attempt_counters[key] = attempt_counters.get(key, 0) + 1
            effect = {
                "attempt_seq": attempt_counters[key],
                "tool": tool_name,
                "parameter": pm.get("raw_command", ""),
                "parameter_struct": {
                    "query": pm.get("query"),
                    "url": pm.get("url"),
                    "fingerprint": pm.get("fingerprint"),
                },
                "result": {"success": oc.get("success"), "preview": oc.get("preview", "")},
                "symptom": oc.get("symptom"),
                "L1_diagnosis": "成功" if oc.get("success") else "参数问题",
            }
            traces[key]["effects"].append(effect)
            if oc.get("L2_diagnosis"):
                traces[key]["L2_diagnosis"] = oc["L2_diagnosis"]
            if oc.get("success"):
                traces[key]["status"] = "completed"

        return list(traces.values())

    def export_task_local_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Derive task-local parameter store keyed by state_type::subgoal_type."""
        store: Dict[str, Dict[str, Any]] = {}

        for sg_id, sg in self.nodes.items():
            if sg.get("type") != GraphNodeType.SUBGOAL.value:
                continue
            subgoal_type = sg.get("subgoal_type", "")
            subgoal_text = sg.get("text", "")

            state_type = ""
            for e in self.edges:
                if e["dst"] == sg_id and e["type"] == ExecutionEdgeType.STATE_REQUIRES_SUBGOAL.value:
                    st = self.nodes.get(e["src"], {})
                    state_type = st.get("state_type", "")
                    state_label = st.get("state_label", state_type)
                    break
            else:
                state_label = state_type

            key = f"{state_label}::{subgoal_text}"
            if key not in store:
                store[key] = {}

            for te in self.edges:
                if te["src"] != sg_id or te["type"] != ExecutionEdgeType.SUBGOAL_SELECTS_TOOL.value:
                    continue
                tool_name = self.nodes.get(te["dst"], {}).get("tool_name", "")
                if tool_name not in store[key]:
                    store[key][tool_name] = {
                        "failed_parameters": [],
                        "successful_parameters": [],
                        "parameter_pairs": [],
                    }

                bucket = store[key][tool_name]
                for pe in self.edges:
                    if pe["src"] != te["dst"] or pe["type"] != ExecutionEdgeType.TOOL_INVOKES_PARAMETER.value:
                        continue
                    pm = self.nodes.get(pe["dst"], {})
                    for oe in self.edges:
                        if oe["src"] != pe["dst"] or oe["type"] != ExecutionEdgeType.PARAMETER_YIELDS_OUTCOME.value:
                            continue
                        oc = self.nodes.get(oe["dst"], {})
                        param_str = pm.get("raw_command", "")
                        if oc.get("success"):
                            entry = {"parameter": param_str, "result": "subgoal_completed"}
                            if entry not in bucket["successful_parameters"]:
                                bucket["successful_parameters"].append(entry)
                        else:
                            entry = {
                                "parameter": param_str,
                                "failure_symptom": oc.get("symptom"),
                                "L1_diagnosis": "参数问题",
                            }
                            if entry not in bucket["failed_parameters"]:
                                bucket["failed_parameters"].append(entry)

                for ce in self.edges:
                    if ce["type"] != ExecutionEdgeType.PARAMETER_CONTRASTS.value:
                        continue
                    fail_pm = self.nodes.get(ce["src"], {})
                    succ_pm = self.nodes.get(ce["dst"], {})
                    pair = {
                        "failed": fail_pm.get("raw_command", ""),
                        "successful": succ_pm.get("raw_command", ""),
                        "insight": ce.get("insight", ""),
                    }
                    if pair not in bucket["parameter_pairs"]:
                        bucket["parameter_pairs"].append(pair)

        return store
