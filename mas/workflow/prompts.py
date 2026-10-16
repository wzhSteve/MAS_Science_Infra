"""Single entry point for prompt resolution (MAS refactor Phase 4).

The six storage locations stay physical (canvas freedom + template reuse);
what converges here is *resolution logic* (MAS_design §7):

  ========== ============================== =====================================
  Location    Content                        Read by
  ========== ============================== =====================================
  tir_agent.py constants (SYSTEM_PROMPT/   TirAgent training hot path
  FINALIZE/RESUME_REACT)                    (verbatim, untouched)
  experiments/<id>/workflow.yaml            authority for hub.system_prompt
  + agents[].system_prompt                  + per-node overrides (UI saves here)
  mas/specs/hub_react.yaml                  default spec (fallback resolution)
  mas/specs/templates/*.yaml (x6)           canvas templates (templates.py)
  runtime composition                       candidate menu / verifier feedback
                                           injection / JSON-only tail — HERE
  legacy/structagent/epc_aw/prompts/*.txt   StructAgent stack only (legacy
                                           internal _load_prompt)
  ========== ============================== =====================================

Empty-string rule (formalized to match code reality — MAS_design §7.1):
  None / missing        → fall through to the next tier
  whitespace-only       → fall through (``(x or "").strip()`` semantics)
  non-empty string      → use it

Resolution chains:
  planner_prompt = node.system_prompt → hub.system_prompt → DEFAULT_PLANNER_PROMPT
                   (or → None for the single-hub TirAgent path, empty allowed)
  verifier_prompt = node.system_prompt → DEFAULT_VERIFIER_PROMPT
                   (no verifier node → "")
  candidate_menu = runtime routable candidates → planner *user*-text segment
                   (explicitly NOT part of the system prompt, MAS_design §7.3)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

DEFAULT_PLANNER_PROMPT = (
    "把题目拆成下一步。next 必须是候选 tool-agent 的 id。"
    "只输出 JSON，键为 next、args、sub_goal、done。"
    "done 为 true 且 next 为空时结束。"
)
DEFAULT_VERIFIER_PROMPT = (
    "根据工具结果判断子目标是否完成。"
    "只输出 JSON，键为 ok、reason、step_conclusion、slot_updates、answer。"
    "step_conclusion 只能是 COMPLETE 或 INCOMPLETE。"
    "完成时 answer 给出简短答案。"
)


def _usable(text: Optional[str]) -> bool:
    """Empty-string rule: None / missing / whitespace-only are unusable."""
    return bool((text or "").strip())


def planner_prompt(spec: Any, compiled: Any) -> str:
    """Resolve the planner system prompt. Byte-identical to the pre-refactor
    inline chain in centralized_runtime.py (``node or hub or DEFAULT``)."""
    node = compiled.agents.get("planner")
    node_prompt = getattr(node, "system_prompt", None) if node is not None else None
    if _usable(node_prompt):
        return str(node_prompt)
    hub_prompt = getattr(getattr(spec, "hub", None), "system_prompt", None)
    if _usable(hub_prompt):
        return str(hub_prompt)
    return DEFAULT_PLANNER_PROMPT


def verifier_prompt(compiled: Any) -> str:
    """Resolve the verifier system prompt. No verifier node → empty string.

    An empty prompt on a verifier node falls back to ``DEFAULT_VERIFIER_PROMPT``
    so a freshly dropped verifier still emits the JSON contract.
    """
    verifier_id = None
    for aid, node in compiled.agents.items():
        if getattr(node, "kind", None) == "verifier":
            verifier_id = aid
            break
    if not verifier_id:
        return ""
    node = compiled.agents.get(verifier_id)
    node_prompt = getattr(node, "system_prompt", None) if node is not None else None
    if _usable(node_prompt):
        return str(node_prompt)
    return DEFAULT_VERIFIER_PROMPT


def candidate_menu(router_ids_run: Iterable[str], routers: Any) -> List[str]:
    """The planner user-text candidate list (routable agent ids), de-duplicated
    in router order. Caller joins this into the *user* segment, never the
    system prompt (MAS_design §7.3)."""
    seen: Dict[str, None] = {}
    for rid in router_ids_run:
        router = routers.get(rid)
        for cand in list(getattr(router, "candidates", None) or []):
            seen.setdefault(str(cand), None)
    return list(seen.keys())


def planner_user_text(
    *,
    question: str,
    candidates: Optional[List[str]] = None,
    direct_executor: str = "",
    feedback: str = "",
) -> str:
    """Compose the planner *user* segment (three-part rule, MAS_design §7.3).

    A feedback turn keeps the candidate menu, so a rejected ``next`` is not
    followed by a prompt that no longer names the legal tool-agent ids.
    """
    if feedback:
        constraint = ""
        if candidates:
            constraint = f" next must be one of {candidates}."
        elif direct_executor:
            constraint = f" next must be {direct_executor}."
        return (
            f"{question}\n\n[Verifier feedback from previous turn]\n{feedback}\n"
            "Decide the next step."
            f"{constraint}"
            " If the feedback already answers the question, set done to true and next to empty."
            " Output JSON with keys: next, args, sub_goal, done."
        )
    if candidates:
        return (
            f"{question}\n\nDecide the next step and pick one tool from "
            f"{candidates}. Output JSON with keys: next, args, sub_goal, done."
        )
    return (
        f"{question}\n\nDecide the next step. next must be {direct_executor}. "
        "Output JSON with keys: next, args, sub_goal, done."
    )


def prompt_provenance(spec: Any, compiled: Any) -> Dict[str, Any]:
    """Audit metadata for Trajectory.meta (MAS_design §7.2).

    Returns ``{"prompt_hash": {...}, "prompt_source": {...}}`` keyed by the
    window role (planner / verifier), with the source tier named
    node / hub / default / none.
    """
    import hashlib

    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    node = compiled.agents.get("planner")
    node_prompt = getattr(node, "system_prompt", None) if node is not None else None
    hub_prompt = getattr(getattr(spec, "hub", None), "system_prompt", None)
    if _usable(node_prompt):
        p_text, p_src = str(node_prompt), "node"
    elif _usable(hub_prompt):
        p_text, p_src = str(hub_prompt), "hub"
    else:
        p_text, p_src = DEFAULT_PLANNER_PROMPT, "default"

    verifier_node = next(
        (item for item in compiled.agents.values() if getattr(item, "kind", None) == "verifier"),
        None,
    )
    verifier_node_prompt = getattr(verifier_node, "system_prompt", None) if verifier_node is not None else None
    if verifier_node is None:
        v_text, v_src = "", "none"
    elif _usable(verifier_node_prompt):
        v_text, v_src = str(verifier_node_prompt), "node"
    else:
        v_text, v_src = DEFAULT_VERIFIER_PROMPT, "default"

    return {
        "prompt_hash": {"planner": _sha(p_text), "verifier": _sha(v_text)},
        "prompt_source": {"planner": p_src, "verifier": v_src},
    }
