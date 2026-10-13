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
  verifier_prompt = node.system_prompt → "" (single terse default)
  candidate_menu = runtime routable candidates → planner *user*-text segment
                   (explicitly NOT part of the system prompt, MAS_design §7.3)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

DEFAULT_PLANNER_PROMPT = "Decompose the task into the next step and pick one tool. Output JSON only."


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
    """Resolve the verifier system prompt; absent verifier → empty string."""
    verifier_id = None
    for aid, node in compiled.agents.items():
        if getattr(node, "kind", None) == "verifier":
            verifier_id = aid
            break
    node = compiled.agents.get(verifier_id) if verifier_id else None
    node_prompt = getattr(node, "system_prompt", None) if node is not None else None
    return str(node_prompt) if node_prompt is not None else ""


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

    Byte-identical with the pre-refactor inline branches in
    centralized_runtime.py:
      feedback present → question + feedback header + JSON keys hint
      router fan-out   → question + candidate menu + JSON keys hint
      direct executor  → question + fixed-next hint
    """
    if feedback:
        return (
            f"{question}\n\n[Verifier feedback from previous turn]\n{feedback}\n"
            "Decide the next step. Output JSON with keys: next, args, sub_goal, done."
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

    v_text = verifier_prompt(compiled)
    v_src = "node" if _usable(v_text) else "none"

    return {
        "prompt_hash": {"planner": _sha(p_text), "verifier": _sha(v_text)},
        "prompt_source": {"planner": p_src, "verifier": v_src},
    }
