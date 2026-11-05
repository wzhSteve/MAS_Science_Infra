"""Human-readable CGHE DAG / prune lines for stdout and MAS Infra UI.

``note_window_progress`` flattens whitespace, so multi-line views must call
``emit`` once per line (never one blob with newlines).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

_HUMAN_LINES: List[str] = []
_HUMAN_CAP = 400


def _clip(text: Any, limit: int = 120) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def human_lines(*, limit: int = 80) -> List[str]:
    if limit <= 0:
        return list(_HUMAN_LINES)
    return list(_HUMAN_LINES[-limit:])


def clear_human_lines() -> None:
    _HUMAN_LINES.clear()


def emit(line: str) -> None:
    text = " ".join(str(line or "").split())
    if not text:
        return
    _HUMAN_LINES.append(text)
    del _HUMAN_LINES[:-_HUMAN_CAP]
    print(text, flush=True)
    try:
        from workflow.user_gateway.public import report_progress

        report_progress(text)
    except Exception:
        pass


def emit_block(title: str, lines: Iterable[str]) -> None:
    head = " ".join(str(title or "").split())
    if head:
        emit(head)
    for line in lines:
        emit(str(line or ""))


def _tool_short(tool: Any) -> str:
    name = str(tool or "").casefold()
    if "bing" in name or "search" in name:
        return "Bing"
    if "fetch" in name:
        return "Fetch"
    if "generalist" in name or "generator" in name:
        return "Extract"
    if "python" in name:
        return "Python"
    return str(tool or "?")[:16]


def _node_label(node: Any) -> str:
    tool = _tool_short(getattr(node, "tool", ""))
    argument = str(getattr(node, "argument", "") or getattr(node, "action", "") or getattr(node, "subgoal", "") or "")
    return f"{tool}({_clip(argument, 72)})"


def format_plan_line(graph: Any, plan_id: str) -> str:
    parts: List[str] = []
    for node_id in list(graph.plan_node_ids(plan_id) or ()):
        try:
            parts.append(_node_label(graph.node(node_id)))
        except Exception:
            parts.append(str(node_id))
    return f"{plan_id}: " + (" -> ".join(parts) if parts else "(empty)")


def format_dag(graph: Any, *, selected_id: str = "") -> List[str]:
    if graph is None:
        return ["(no graph)"]
    lines: List[str] = []
    active = list(graph.active_plan_ids())
    lines.append(
        f"dag_id={getattr(graph, 'dag_id', '')} active_plans={len(active)} "
        f"attempt_count={getattr(graph, 'attempt_count', 0)}"
    )
    coverage = {}
    try:
        coverage = dict(graph.compute_step1_coverage())
    except Exception:
        coverage = {}
    if coverage:
        ranked = sorted(coverage.items(), key=lambda item: (-int(item[1]), str(item[0])))
        cov_bits = []
        for node_id, count in ranked[:8]:
            try:
                label = _clip(getattr(graph.node(node_id), "action", node_id), 40)
            except Exception:
                label = str(node_id)
            mark = "*" if selected_id and node_id == selected_id else ""
            cov_bits.append(f"{label}={count}{mark}")
        lines.append("coverage: " + " | ".join(cov_bits))
    for plan_id in active:
        lines.append(format_plan_line(graph, plan_id))
    shared = []
    try:
        for node in graph.nodes():
            ids = list(getattr(node, "candidate_plan_ids", ()) or ())
            if len(ids) < 2:
                continue
            shared.append(f"{_node_label(node)} shared_by={ids}")
    except Exception:
        pass
    for item in shared[:6]:
        lines.append("share " + item)
    return lines


def format_selection(graph: Any, node_id: str, coverage: int, plan_ids: Sequence[str]) -> List[str]:
    try:
        node = graph.node(node_id)
        label = _node_label(node)
    except Exception:
        label = str(node_id)
    return [
        f"node={node_id} {_clip(label, 100)}",
        f"coverage={int(coverage)} plans={list(plan_ids)}",
    ]


def format_prune(
    before_ids: Sequence[str],
    pruned: Sequence[str],
    after_ids: Sequence[str],
    constraint: Any = None,
) -> List[str]:
    predicate = ""
    argument = ""
    scope = ""
    status = ""
    if constraint is not None:
        status = str(getattr(constraint, "evidence_status", "") or "")
        condition = getattr(constraint, "condition", None)
        predicate = str(getattr(condition, "predicate", "") or "")
        argument = str(getattr(condition, "argument", "") or "")
        scope = _clip(getattr(constraint, "scope", ""), 160)
    lines = [
        f"before={list(before_ids)}",
        f"predicate={predicate or '-'} arg={_clip(argument, 80) or '-'} status={status or '-'}",
    ]
    if scope:
        lines.append(f"scope={scope}")
    lines.append(f"pruned={list(pruned)} keep={list(after_ids)}")
    return lines


def format_regen(
    old_id: str,
    new_id: str,
    reasons: Sequence[str],
    inherited: Sequence[str],
) -> List[str]:
    return [
        f"old_dag={old_id or '-'} new_dag={new_id or '-'}",
        f"reasons={list(reasons)}",
        f"inherited={list(inherited)}",
    ]


def format_advance(parent_id: str, child_id: str, coverage: int, graph: Any = None) -> List[str]:
    parent = parent_id
    child = child_id
    if graph is not None:
        try:
            parent = f"{parent_id}:{_node_label(graph.node(parent_id))}"
        except Exception:
            pass
        try:
            child = f"{child_id}:{_node_label(graph.node(child_id))}"
        except Exception:
            pass
    return [f"from={_clip(parent, 100)}", f"to={_clip(child, 100)} coverage={int(coverage)}"]


def log_dag_build(graph: Any, *, selected_id: str = "") -> None:
    body = format_dag(graph, selected_id=selected_id)
    emit(f"[CGHE][DAG] {body[0]}" if body else "[CGHE][DAG] (empty)")
    for line in body[1:]:
        emit(f"[CGHE][DAG] {line}")


def log_selection(graph: Any, node_id: str, coverage: int, plan_ids: Sequence[str]) -> None:
    for line in format_selection(graph, node_id, coverage, plan_ids):
        emit(f"[CGHE][SELECT] {line}")


def log_prune(
    before_ids: Sequence[str],
    pruned: Sequence[str],
    after_ids: Sequence[str],
    constraint: Any = None,
    *,
    verified: bool = True,
) -> None:
    tag = "[CGHE][PRUNE]" if verified else "[CGHE][FAIL]"
    if not verified:
        emit(f"{tag} unverified hypothesis; no plan deleted")
        emit(f"{tag} before={list(before_ids)} keep={list(after_ids)}")
        return
    for line in format_prune(before_ids, pruned, after_ids, constraint):
        emit(f"{tag} {line}")


def log_regen(old_id: str, new_id: str, reasons: Sequence[str], inherited: Sequence[str]) -> None:
    for line in format_regen(old_id, new_id, reasons, inherited):
        emit(f"[CGHE][REGEN] {line}")


def log_advance(parent_id: str, child_id: str, coverage: int, graph: Any = None) -> None:
    for line in format_advance(parent_id, child_id, coverage, graph):
        emit(f"[CGHE][ADVANCE] {line}")


def plans_text(graph: Any) -> List[str]:
    if graph is None:
        return []
    return [format_plan_line(graph, plan_id) for plan_id in graph.active_plan_ids()]


def constraint_brief(constraint: Any) -> dict[str, str]:
    if constraint is None:
        return {}
    condition = getattr(constraint, "condition", None)
    return {
        "predicate": str(getattr(condition, "predicate", "") or ""),
        "argument": str(getattr(condition, "argument", "") or ""),
        "tool": str(getattr(condition, "tool", "") or ""),
        "status": str(getattr(constraint, "evidence_status", "") or ""),
        "scope": _clip(getattr(constraint, "scope", ""), 200),
    }
