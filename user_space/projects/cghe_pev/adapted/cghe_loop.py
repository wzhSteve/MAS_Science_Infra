"""Step-1 coverage controller for one CGHE episode.

Coverage is the number of ACTIVE plans whose static first node is v.
Out-degree is not coverage. Ordinary failures do not prune plans.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from cghe_types import CausalRelation, Constraint, PlanStatus


def validate_dag(graph: Any) -> list[str]:
    """Schema, references, acyclicity, and a non-empty Step-1 on every plan."""
    errors: list[str] = []
    known = {node.node_id for node in graph.nodes()}
    edges: dict[str, set[str]] = {}
    for plan_id in graph.candidate_plan_ids:
        sequence = graph.plan_node_ids(plan_id)
        if not sequence:
            errors.append(f"{plan_id}: missing step 1")
            continue
        for node_id in sequence:
            if node_id not in known:
                errors.append(f"{plan_id}: unknown node {node_id}")
        first = graph.node(sequence[0])
        if not str(getattr(first, "subgoal", "") or "").strip():
            errors.append(f"{plan_id}: step 1 has no subgoal")
        for left, right in zip(sequence, sequence[1:]):
            edges.setdefault(left, set()).add(right)
    visiting: set[str] = set()
    visited: set[str] = set()

    def _walk(node_id: str) -> None:
        if node_id in visited or errors and "cycle" in errors[-1]:
            return
        if node_id in visiting:
            errors.append(f"cycle at {node_id}")
            return
        visiting.add(node_id)
        for nxt in edges.get(node_id, ()):
            _walk(nxt)
        visiting.discard(node_id)
        visited.add(node_id)

    for node_id in list(edges):
        _walk(node_id)
    return errors


def make_regeneration_context(
    *,
    query: str,
    mas_state: Any = None,
    active_plans: Sequence[Any] = (),
    verified_counterexamples: Sequence[Any] = (),
    verified_successes: Sequence[Any] = (),
    unverified_failures: Sequence[Any] = (),
    observations: Sequence[Any] = (),
    previous_attempts: Sequence[Any] = (),
    trigger_reasons: Sequence[str] = (),
    global_attempts_remaining: int = 0,
) -> dict[str, Any]:
    return {
        "query": query,
        "mas_state": mas_state,
        "active_plans": list(active_plans),
        "verified_counterexamples": list(verified_counterexamples),
        "verified_successes": list(verified_successes),
        "unverified_failures": list(unverified_failures),
        "unverified_failures_are_hard_constraints": False,
        "observations": list(observations),
        "previous_attempts": list(previous_attempts),
        "trigger_reasons": list(trigger_reasons),
        "global_attempts_remaining": int(global_attempts_remaining),
    }


def drop_contradicted_plans(plans: Sequence[Any], counterexamples: Sequence[Any]) -> list[Any]:
    """Drop plans whose step-1 query repeats a verified snippet or access failure."""
    banned = set()
    for item in counterexamples:
        constraint = getattr(item, "constraint", None)
        if isinstance(item, dict):
            constraint = item.get("constraint") or item
        elif constraint is None and getattr(item, "condition", None) is not None:
            # Bare Constraint passed from planning_records / unit tests.
            constraint = item
        argument = ""
        if constraint is not None and not isinstance(constraint, dict):
            argument = str(getattr(getattr(constraint, "condition", None), "argument", "") or "")
        elif isinstance(constraint, dict):
            argument = str((constraint.get("condition") or {}).get("argument") or constraint.get("argument") or "")
        if argument:
            banned.add(argument.casefold())
    if not banned:
        return list(plans)
    kept = []
    for plan in plans:
        steps = list(getattr(plan, "steps", ()) or ())
        if not steps:
            continue
        argument = str(getattr(steps[0], "argument", "") or getattr(steps[0], "action", "")).casefold()
        if argument in banned:
            continue
        kept.append(plan)
    return kept


def context_guidance(ctx: dict[str, Any]) -> str:
    """Text for the planner. Unverified failures stay labeled as non-binding."""
    lines = [
        "Verified counterexamples are hard constraints. Do not reuse those assumptions.",
        "Verified successes and their outputs may be reused.",
        "Unverified failures are diagnostic hints only. They are NOT hard constraints.",
        f"Trigger reasons: {', '.join(ctx.get('trigger_reasons') or []) or 'none'}.",
        f"Global step-1 attempts remaining: {ctx.get('global_attempts_remaining')}.",
    ]
    for item in ctx.get("verified_counterexamples") or []:
        lines.append(f"HARD verified counterexample: {_brief(item)}")
    for item in ctx.get("verified_successes") or []:
        lines.append(f"Verified success: {_brief(item)}")
    for item in ctx.get("unverified_failures") or []:
        lines.append(f"HINT only, not a constraint: {_brief(item)}")
    for item in ctx.get("observations") or []:
        lines.append(f"Observation: {_brief(item)}")
    return "\n".join(lines)


def _inherit_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("record_id") or item.get("constraint_id") or item.get("evidence_id") or "")
    return str(getattr(item, "record_id", None) or getattr(item, "constraint_id", None) or "")


def _brief(item: Any) -> str:
    if isinstance(item, dict):
        return str({key: item[key] for key in list(item)[:6]})[:400]
    text = getattr(item, "note", None) or getattr(item, "action", None) or item
    return str(text)[:400]


class CGHESession:
    """Per-episode counters. The graph's own attempt_count resets on regeneration.

    Round-based mode: each ``regenerate`` seals the previous DAG and installs a
    brand-new ``PlanningGraph`` with a new version id. Same-DAG prune+SELECT is
    not the next-round strategy.
    """

    def __init__(self, controller: Any, *, settings: Optional[dict[str, Any]] = None) -> None:
        self.controller = controller
        raw = dict(settings or {})
        self.max_step1_attempts = int(raw.get("max_step1_attempts") or 3)
        # Round-based regen needs more planner rounds than the old same-DAG budget.
        default_regen = 5 if bool(raw.get("round_regen", True)) else 2
        self.max_regenerations = int(raw.get("max_regenerations") or default_regen)
        self.max_total = int(raw.get("max_total_step1_attempts") or raw.get("tool_calls") or 12)
        self.round_regen = bool(raw.get("round_regen", True))
        self.global_attempts = 0
        self.regeneration_count = 0
        self.planning_round = 0
        self.events: list[dict[str, Any]] = []
        self.archived_attempts: list[Any] = []
        self.archived_dags: list[dict[str, Any]] = []
        self.stop_reason = ""
        # Make session_for(controller) reuse this instance.
        try:
            controller.cghe_session = self
        except Exception:
            pass

    def bind_graph(self, graph: Any) -> None:
        graph.max_attempts = self.max_step1_attempts
        if not getattr(graph, "attempt_history", None):
            graph.attempt_history = []
        self.controller.graph = graph

    def archive_active_dag(self, *, reason: str = "") -> Optional[dict[str, Any]]:
        """Seal the current DAG as immutable history; return its snapshot."""
        graph = getattr(self.controller, "graph", None)
        if graph is None:
            return None
        if hasattr(graph, "seal"):
            snap = graph.seal()
        else:
            from hypothesis_round_memory import freeze_dag_snapshot

            snap = freeze_dag_snapshot(graph)
        snap = dict(snap or {})
        snap["archive_reason"] = str(reason or "")
        snap["planning_round"] = int(self.planning_round)
        self.archived_dags.append(snap)
        memory = getattr(self.controller, "hypothesis_memory", None)
        if memory is not None:
            version = str(snap.get("version") or "")
            if version and version not in memory.archived_dag_versions:
                memory.archived_dag_versions.append(version)
        self.events.append({
            "event": "dag_archived",
            "version": snap.get("version"),
            "reason": reason,
            "planning_round": self.planning_round,
            "nodes": len(snap.get("nodes") or []),
            "plans": len(snap.get("plans") or []),
        })
        return snap

    def note_step1_execution(
        self,
        graph: Any,
        node_id: str,
        *,
        attempt_id: str,
        selected_plan_ids: Sequence[str],
        coverage: int,
        replay: bool = False,
    ) -> Any:
        """Count one real Executor call. Replays and repeat ids do not count."""
        if replay or not attempt_id or graph is None:
            return None
        if not graph.is_step1_node(node_id):
            return None
        for existing in graph.attempt_history:
            if existing.attempt_id == attempt_id:
                return existing
        record = graph.record_step1_attempt(
            node_id,
            attempt_id=attempt_id,
            selected_plan_ids=selected_plan_ids,
            coverage=coverage,
        )
        self.global_attempts += 1
        self.events.append({
            "event": "step1_executed",
            "attempt_id": attempt_id,
            "dag_id": graph.dag_id,
            "node_id": node_id,
            "global_attempts": self.global_attempts,
            "attempt_count": graph.attempt_count,
        })
        return record

    def log_selection(self, graph: Any, node_id: str, coverage: int, eligible: dict[str, int]) -> None:
        from cghe_dag_log import log_selection as emit_selection, plans_text

        plan_ids = list(graph.plans_starting_with(node_id))
        self.events.append({
            "event": "step1_selected",
            "dag_id": graph.dag_id,
            "node_id": node_id,
            "coverage": int(coverage),
            "coverage_map": dict(graph.compute_step1_coverage()),
            "eligible_coverage": dict(eligible),
            "selected_plan_ids": plan_ids,
            "attempt_count": graph.attempt_count,
            "plans_text": plans_text(graph),
        })
        emit_selection(graph, node_id, coverage, plan_ids)

    def next_decision(self) -> tuple[str, Any]:
        """One controller step. Does not call the Executor and does not loop."""
        if self.stop_reason:
            return self.stop_reason, None
        if self.global_attempts >= self.max_total:
            self.stop_reason = "GLOBAL_BUDGET_EXHAUSTED"
            return self.stop_reason, None
        graph = getattr(self.controller, "graph", None)
        if graph is None:
            return "NEED_GRAPH", None
        reasons = list(graph.regeneration_reasons())
        if reasons:
            if self.regeneration_count >= self.max_regenerations:
                self.stop_reason = "MAX_REGENERATIONS_REACHED"
                return self.stop_reason, reasons
            return "REGENERATE", reasons
        selection = graph.select_step1()
        if selection is None:
            if self.regeneration_count >= self.max_regenerations:
                self.stop_reason = "MAX_REGENERATIONS_REACHED"
                return self.stop_reason, ["NO_EXECUTABLE_CANDIDATE"]
            return "REGENERATE", ["NO_EXECUTABLE_CANDIDATE"]
        return "SELECT", selection

    def regenerate(self, plans: Sequence[Any], ctx: dict[str, Any]) -> Any:
        """Install at most one NEW DAG. Planner/schema failure does not count an attempt.

        The previous active DAG is sealed and archived; it is never reused as the
        next-round mutable graph.
        """
        from planning_graph import PlanningGraph

        from cghe_dag_log import log_dag_build, log_regen, plans_text

        reasons = list(ctx.get("trigger_reasons") or [])
        old = getattr(self.controller, "graph", None)
        old_id = getattr(old, "dag_id", "")
        old_version = str(getattr(old, "version", "") or old_id)
        old_snapshot = None
        if old is not None and not getattr(old, "sealed", False):
            old_snapshot = self.archive_active_dag(reason=",".join(reasons) or "regenerate")
        self.events.append({
            "event": "dag_regeneration_triggered",
            "old_dag_id": old_id,
            "old_version": old_version,
            "trigger_reasons": reasons,
            "coverage_map": dict(old.compute_step1_coverage()) if old is not None else {},
            "attempt_count": int(getattr(old, "attempt_count", 0) or 0),
            "global_attempts": self.global_attempts,
            "planning_round": self.planning_round,
        })
        if not plans:
            log_regen(old_id, "", reasons, [])
            return None
        self.planning_round = int(self.planning_round) + 1
        try:
            graph = PlanningGraph(list(plans), version=self.planning_round)
        except (TypeError, ValueError):
            return None
        errors = validate_dag(graph)
        if errors:
            self.events.append({
                "event": "dag_validation_failed",
                "errors": errors,
                "counted_attempt": False,
            })
            return None
        if old is not None:
            self.archived_attempts.extend(list(old.attempt_history))
        # Identity check: new object, new version.
        if old is not None and graph is old:
            raise RuntimeError("regenerate must install a new PlanningGraph instance")
        if old_version and str(graph.version) == old_version:
            # Force distinct version string if collision.
            graph._version = f"{self.planning_round}-{graph.round_id}"
        graph.attempt_count = 0
        graph.regeneration_count = self.regeneration_count + 1
        graph.max_attempts = self.max_step1_attempts
        self.regeneration_count = graph.regeneration_count
        self.bind_graph(graph)
        memory = getattr(self.controller, "hypothesis_memory", None)
        if memory is not None:
            memory.round_index = self.planning_round
            memory.register_dag_hypotheses(graph)
        inherited = [_inherit_id(item) for item in (ctx.get("verified_counterexamples") or [])]
        inherited += [_inherit_id(item) for item in (ctx.get("verified_successes") or [])]
        inherited_ids = [item for item in inherited if item]
        self.events.append({
            "event": "dag_regenerated",
            "new_dag_id": graph.dag_id,
            "new_version": graph.version,
            "old_dag_id": old_id,
            "old_version": old_version,
            "old_sealed": bool(old_snapshot and old_snapshot.get("sealed")),
            "trigger_reasons": reasons,
            "inherited_ids": inherited_ids,
            "attempt_count": graph.attempt_count,
            "global_attempts": self.global_attempts,
            "regeneration_count": self.regeneration_count,
            "planning_round": self.planning_round,
            "unverified_are_hard_constraints": False,
            "plans_text": plans_text(graph),
            "coverage_map": dict(graph.compute_step1_coverage()),
        })
        print(
            f"[CGHE][ROUND {self.planning_round}] Generated DAG version={graph.version} "
            f"plans={len(list(graph.active_plan_ids()))} "
            f"(prev={old_version or '-'} sealed={bool(old_snapshot)})",
            flush=True,
        )
        log_regen(old_id, graph.dag_id, reasons, inherited_ids)
        log_dag_build(graph)
        return graph

    def handoff_after_step1_success(self, graph: Any, node_id: str, result: Any) -> dict[str, Any]:
        """Step-1 success is not task completion. Remaining steps go to ordinary PEV."""
        selected = [
            plan_id for plan_id in graph.active_plan_ids()
            if node_id in graph.plan_node_ids(plan_id)
        ]
        graph.mark_satisfied(node_id)
        remaining: list[dict[str, str]] = []
        for plan_id in selected:
            sequence = graph.plan_node_ids(plan_id)
            if node_id not in sequence:
                continue
            start = sequence.index(node_id) + 1
            for nxt in sequence[start:]:
                node = graph.node(nxt)
                remaining.append({
                    "plan_id": plan_id,
                    "node_id": nxt,
                    "subgoal": str(node.subgoal or ""),
                    "tool": str(node.tool or ""),
                    "action": str(node.action or ""),
                })
        self.events.append({
            "event": "verification_completed",
            "verifier_status": "SUCCESS",
            "node_id": node_id,
            "dag_id": graph.dag_id,
            "task_completed": False,
        })
        memory = getattr(self.controller, "memory", None)
        if memory is not None and hasattr(memory, "add_causal_edge"):
            memory.add_causal_edge(
                node_id,
                f"success:{node_id}",
                CausalRelation.SUPPORTS,
                evidence_id=str(getattr(result, "evidence_id", "") or node_id),
                direct=True,
            )
            for step in remaining:
                memory.add_causal_edge(node_id, step["node_id"], CausalRelation.ENABLES, direct=True)
        return {
            "decision": "continue_pev",
            "reason": "step1 success handed off to pev",
            "task_completed": False,
            "node_id": node_id,
            "selected_plan_ids": selected,
            "remaining": remaining,
            "graph_cleared": False,
        }


def prune_by_counterexample(
    graph: Any,
    constraint: Optional[Constraint],
    *,
    evidence: Any = None,
    executed_plan_id: str = "",
) -> list[str]:
    """Prune only plans the existing admission gate says explicitly depend on the evidence."""
    from hypothesis_pruner import HypothesisPruner

    if constraint is None:
        return []
    result = HypothesisPruner().prune(
        graph,
        constraint,
        evidence=evidence,
        executed_plan_id=str(executed_plan_id or ""),
        apply=True,
    )
    graph.refresh_orphans()
    # Snippet gaps retire the executed plan only; surface those ids too.
    return list(result.pruned_plan_ids) + list(result.instance_retired_plan_ids)


def record_plan_dependencies(memory: Any, graph: Any) -> None:
    if memory is None or not hasattr(memory, "add_causal_edge"):
        return
    for plan_id in graph.candidate_plan_ids:
        sequence = graph.plan_node_ids(plan_id)
        for left, right in zip(sequence, sequence[1:]):
            memory.add_causal_edge(left, right, CausalRelation.REQUIRES, direct=True)


def session_settings_from_runtime(settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    raw = dict(settings or {})
    round_regen = bool(raw.get("round_regen", True))
    return {
        "max_step1_attempts": int(raw.get("max_step1_attempts") or 3),
        "max_regenerations": int(raw.get("max_regenerations") or (5 if round_regen else 2)),
        "max_total_step1_attempts": int(raw.get("max_total_step1_attempts") or raw.get("tool_calls") or 12),
        "tool_calls": int(raw.get("tool_calls") or 12),
        "round_regen": round_regen,
    }


def session_for(controller: Any, settings: Optional[dict[str, Any]] = None) -> CGHESession:
    existing = getattr(controller, "cghe_session", None)
    if existing is not None:
        return existing
    session = CGHESession(controller, settings=session_settings_from_runtime(settings))
    controller.cghe_session = session
    return session
