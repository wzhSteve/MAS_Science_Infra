"""Hypothesis-indexed planning graph for one fixed candidate universe.

The graph is built from ``N`` plans that are already structured. After
construction the universe does not gain paths. Nodes merge only when the
normalized action/tool, atomic preconditions and required state are exactly
compatible. Node choice among executable nodes is

    score = |active plans that still contain the node|

with a lower estimated cost, then a stable ``node_id``, as tie breaks.
Downstream node count is not a score.
"""

from __future__ import annotations

import copy
import re
from typing import Iterable, Optional, Sequence

from cghe_types import (
    AtomicPremise,
    CandidatePlan,
    GraphNode,
    NodeStatus,
    PlanStatus,
    PlanStep,
    Scope,
    canonical_action,
    new_id,
    norm,
    normalize_state,
)


class UniverseFrozenError(RuntimeError):
    """Raised when a caller tries to add a path after the round is built."""


def _step_merge_key(step: PlanStep) -> tuple:
    action, tool = canonical_action(step.action, step.tool)
    premises = tuple(sorted(premise.normalized().key() for premise in step.preconditions))
    state = normalize_state(step.required_state)
    usage = (
        norm(step.resource),
        norm(step.identity),
        norm(step.parameter),
        norm(step.argument),
        norm(step.schema_signature),
    )
    return (action, tool, premises, state, usage)


def _slug(tool: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in norm(tool))
    cleaned = cleaned.strip("_")
    return cleaned[:24] or "node"


class PlanningGraph:
    """DAG of one recovery round. ``round_id`` identifies that frozen universe."""

    def __init__(self, plans: Sequence[CandidatePlan], *, round_id: str = "") -> None:
        if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
            raise TypeError("plans must be a sequence of CandidatePlan")
        self.round_id = round_id or new_id("round")
        self._frozen = True
        self._plans: dict[str, CandidatePlan] = {}
        self._plan_order: list[str] = []
        copied: list[CandidatePlan] = []
        for plan in plans:
            if not isinstance(plan, CandidatePlan):
                raise TypeError("each plan must be a CandidatePlan")
            if plan.plan_id in self._plans:
                raise ValueError(f"duplicate plan_id: {plan.plan_id}")
            item = copy.deepcopy(plan)
            copied.append(item)
            self._plans[item.plan_id] = item
            self._plan_order.append(item.plan_id)
        self._nodes: dict[str, GraphNode] = {}
        self._plan_nodes: dict[str, tuple[str, ...]] = {}
        self._predecessors: dict[tuple[str, str], tuple[str, ...]] = {}
        self._build(copied)

    @classmethod
    def from_plans(cls, plans: Sequence[CandidatePlan], *, round_id: str = "") -> "PlanningGraph":
        return cls(plans, round_id=round_id)

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def candidate_plan_ids(self) -> tuple[str, ...]:
        return tuple(self._plan_order)

    def add_candidate(self, plan: CandidatePlan) -> None:
        """The universe is fixed at construction. This always fails."""
        raise UniverseFrozenError(
            f"round {self.round_id} is frozen with {len(self._plan_order)} plans; "
            "open a new recovery round instead of adding a candidate"
        )

    def plan(self, plan_id: str) -> CandidatePlan:
        return copy.deepcopy(self._plans[plan_id])

    def plans(self) -> tuple[CandidatePlan, ...]:
        return tuple(copy.deepcopy(self._plans[plan_id]) for plan_id in self._plan_order)

    def active_plans(self) -> tuple[CandidatePlan, ...]:
        return tuple(plan for plan in self.plans() if plan.status == PlanStatus.ACTIVE)

    def active_plan_ids(self) -> tuple[str, ...]:
        return tuple(
            plan_id
            for plan_id in self._plan_order
            if self._plans[plan_id].status == PlanStatus.ACTIVE
        )

    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(copy.deepcopy(self._nodes[node_id]) for node_id in sorted(self._nodes))

    def node(self, node_id: str) -> GraphNode:
        return copy.deepcopy(self._nodes[node_id])

    def plan_node_ids(self, plan_id: str) -> tuple[str, ...]:
        return self._plan_nodes[plan_id]

    def predecessors(self, plan_id: str, node_id: str) -> tuple[str, ...]:
        return self._predecessors.get((plan_id, node_id), ())

    def sharing_score(self, node_id: str) -> int:
        """Number of still-active plans that contain this node."""
        node = self._nodes[node_id]
        active = set(self.active_plan_ids())
        return sum(1 for plan_id in node.candidate_plan_ids if plan_id in active)

    def executable_node_ids(self) -> tuple[str, ...]:
        ready = [node_id for node_id in sorted(self._nodes) if self._is_executable(node_id)]
        return tuple(ready)

    def select_node(self, *, policy: str = "shared") -> Optional[GraphNode]:
        """Pick the next executable node.

        ``shared`` scores nodes by the number of active plans that contain
        them, then lower estimated cost, then stable node id. ``first_plan``
        keeps plan order and takes the first executable node of the first
        still-active plan, which is the ablation that turns sharing off.
        """
        ready = list(self.executable_node_ids())
        if not ready:
            return None
        if policy == "first_plan":
            ready_set = set(ready)
            for plan_id in self.active_plan_ids():
                for node_id in self.plan_node_ids(plan_id):
                    if node_id in ready_set:
                        return copy.deepcopy(self._nodes[node_id])
            return None
        if policy == "frontier":
            return self._select_frontier()
        if policy != "shared":
            raise ValueError(f"unknown node selection policy: {policy}")
        ready.sort(
            key=lambda node_id: (
                -self.sharing_score(node_id),
                self._nodes[node_id].estimated_cost,
                node_id,
            )
        )
        return copy.deepcopy(self._nodes[ready[0]])

    def mark_satisfied(self, node_id: str) -> None:
        self._nodes[node_id].status = NodeStatus.SATISFIED

    def mark_failed(self, node_id: str) -> None:
        self._nodes[node_id].status = NodeStatus.FAILED

    def drop_suffixes(self, node_id: str) -> tuple[str, ...]:
        """Fail one node and retire every active outline that still contains it.

        Later steps on those outlines leave with the plan. Other outlines stay.
        """
        if node_id not in self._nodes:
            return ()
        self.mark_failed(node_id)
        victims = [
            plan_id
            for plan_id in self.active_plan_ids()
            if node_id in self.plan_node_ids(plan_id)
        ]
        return self.retire_plans(victims)

    def retire_plans(
        self,
        plan_ids: Iterable[str],
        *,
        status: PlanStatus = PlanStatus.PRUNED,
    ) -> tuple[str, ...]:
        """Drop plans from the active set. Unknown ids are ignored.

        Structural ``candidate_plan_ids`` on nodes stay in place, so a later
        score query stops counting the retired plans.
        """
        if isinstance(status, str):
            status = PlanStatus(status)
        if status == PlanStatus.ACTIVE:
            raise ValueError("retire_plans cannot mark a plan active")
        changed: list[str] = []
        for plan_id in plan_ids:
            plan = self._plans.get(plan_id)
            if plan is None or plan.status != PlanStatus.ACTIVE:
                continue
            plan.status = status
            changed.append(plan_id)
        return tuple(sorted(changed))

    def _is_executable(self, node_id: str) -> bool:
        node = self._nodes[node_id]
        if node.status != NodeStatus.PENDING:
            return False
        if self.sharing_score(node_id) <= 0:
            return False
        for plan_id in self.active_plan_ids():
            if node_id not in self._plan_nodes.get(plan_id, ()):
                continue
            preds = self._predecessors.get((plan_id, node_id), ())
            if all(self._nodes[pred].status == NodeStatus.SATISFIED for pred in preds):
                return True
        return False

    def frontier_node_id(self, plan_id: str) -> Optional[str]:
        """First unfinished step of one outline. Later steps are not eligible."""
        if plan_id not in self.active_plan_ids():
            return None
        for node_id in self.plan_node_ids(plan_id):
            node = self._nodes[node_id]
            if node.status != NodeStatus.PENDING:
                continue
            preds = self._predecessors.get((plan_id, node_id), ())
            if all(self._nodes[pred].status == NodeStatus.SATISFIED for pred in preds):
                return node_id
        return None

    def _vote_key(self, node_id: str) -> tuple[str, str]:
        node = self._nodes[node_id]
        return norm(node.subgoal), norm(node.tool)

    def frontier_vote(self, node_id: str) -> int:
        """Outlines whose current first step has this subgoal and tool.

        This is not the node's out-degree and not the count of downstream nodes.
        """
        key = self._vote_key(node_id)
        count = 0
        for plan_id in self.active_plan_ids():
            frontier = self.frontier_node_id(plan_id)
            if frontier is not None and self._vote_key(frontier) == key:
                count += 1
        return count

    def _select_frontier(self) -> Optional[GraphNode]:
        candidates = []
        for plan_id in self.active_plan_ids():
            frontier = self.frontier_node_id(plan_id)
            if frontier is not None and frontier not in candidates:
                candidates.append(frontier)
        if not candidates:
            return None
        candidates.sort(
            key=lambda node_id: (
                -self.frontier_vote(node_id),
                self._nodes[node_id].estimated_cost,
                node_id,
            )
        )
        return copy.deepcopy(self._nodes[candidates[0]])

    def _build(self, plans: Sequence[CandidatePlan]) -> None:
        grouped: dict[tuple, list[tuple[str, int, PlanStep]]] = {}
        for plan in plans:
            for index, step in enumerate(plan.steps):
                grouped.setdefault(_step_merge_key(step), []).append((plan.plan_id, index, step))

        key_to_node: dict[tuple, str] = {}
        for offset, key in enumerate(sorted(grouped), start=1):
            members = grouped[key]
            action, tool = key[0], key[1]
            node_id = f"n-{offset:04d}-{_slug(tool or action)}"
            key_to_node[key] = node_id
            premises = _shared_premises(members)
            plan_ids = tuple(sorted({plan_id for plan_id, _index, _step in members}))
            subgoal = _canonical_subgoal(members)
            cost = min(step.estimated_cost for _plan_id, _index, step in members)
            sample = members[0][2]
            self._nodes[node_id] = GraphNode(
                node_id=node_id,
                subgoal=subgoal,
                action=action,
                tool=tool,
                preconditions=premises,
                dependencies=(),
                candidate_plan_ids=plan_ids,
                status=NodeStatus.PENDING,
                estimated_cost=cost,
                required_state=key[3],
                resource=norm(sample.resource),
                identity=norm(sample.identity),
                parameter=norm(sample.parameter),
                argument=norm(sample.argument),
                schema_signature=norm(sample.schema_signature),
            )

        for plan in plans:
            step_ids = _assign_step_ids(plan.steps)
            node_ids: list[str] = []
            for index, step in enumerate(plan.steps):
                node_ids.append(key_to_node[_step_merge_key(step)])
            self._plan_nodes[plan.plan_id] = tuple(node_ids)
            id_to_index = {step_id: index for index, step_id in enumerate(step_ids)}
            for index, step in enumerate(plan.steps):
                node_id = node_ids[index]
                if step.predecessor_ids:
                    missing = [item for item in step.predecessor_ids if item not in id_to_index]
                    if missing:
                        raise ValueError(
                            f"plan {plan.plan_id} step {step_ids[index]} "
                            f"names unknown predecessors: {missing}"
                        )
                    preds = tuple(node_ids[id_to_index[item]] for item in step.predecessor_ids)
                elif step.links_previous and index > 0:
                    preds = (node_ids[index - 1],)
                else:
                    preds = ()
                self._predecessors[(plan.plan_id, node_id)] = preds

        for node_id, node in self._nodes.items():
            preds = set()
            for plan_id in node.candidate_plan_ids:
                preds.update(self._predecessors.get((plan_id, node_id), ()))
            node.dependencies = tuple(sorted(preds))


def _assign_step_ids(steps: Sequence[PlanStep]) -> tuple[str, ...]:
    assigned: list[str] = []
    seen = set()
    for index, step in enumerate(steps):
        step_id = step.step_id or f"s{index}"
        if step_id in seen:
            raise ValueError(f"duplicate step_id: {step_id}")
        seen.add(step_id)
        assigned.append(step_id)
    return tuple(assigned)


def _shared_premises(members: Sequence[tuple[str, int, PlanStep]]) -> tuple[AtomicPremise, ...]:
    by_key: dict[str, AtomicPremise] = {}
    for _plan_id, _index, step in members:
        for premise in step.preconditions:
            item = premise.normalized()
            by_key.setdefault(item.key(), item)
    return tuple(by_key[key] for key in sorted(by_key))


def _canonical_subgoal(members: Sequence[tuple[str, int, PlanStep]]) -> str:
    ordered = sorted(members, key=lambda item: (item[0], item[1]))
    return ordered[0][2].subgoal


_URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+")
_QUERY_RE = re.compile(r"query\s*=\s*(['\"])(.+?)\1", re.I)
_HEADER_RE = re.compile(r"results for:\s*(.+)", re.I)
_BING = "Bing_Search_Tool"
_FETCH = "Web_Fetch_Tool"


def plans_for_snippet_gap(action: dict, *, environment: str, limit: int = 3) -> tuple[CandidatePlan, ...]:
    """Build a fixed universe from one Bing result the verifier already rejected.

    The repeated query carries ``SnippetsSufficient`` for that query. Fetch
    plans do not, so admissible pruning removes only the query that just failed.
    """
    if not isinstance(action, dict):
        return ()
    if "bing" not in norm(action.get("tool_name")):
        return ()
    text = action.get("raw_tool_result", action.get("result"))
    blob = text if isinstance(text, str) else str(text or "")
    urls = []
    for match in _URL_RE.findall(blob):
        url = match.rstrip(".,")
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    query = ""
    found = _QUERY_RE.search(str(action.get("command") or ""))
    if found:
        query = found.group(2).strip()
    else:
        header = _HEADER_RE.search(blob)
        if header:
            query = header.group(1).split("\n", 1)[0].strip()
    if not query or not urls:
        return ()
    premise = AtomicPremise(
        predicate="SnippetsSufficient",
        tool=_BING,
        parameter="query",
        argument=query,
    )
    answer = PlanStep(
        subgoal="answer the question from the retrieved evidence",
        action="answer",
        tool="Generalist_Solution_Generator_Tool",
        estimated_cost=1,
        links_previous=True,
    )
    search = CandidatePlan(
        "same_query",
        (PlanStep(
            subgoal=f"search again: {query}",
            action=query,
            tool=_BING,
            preconditions=(premise,),
            parameter="query",
            argument=query,
            estimated_cost=2,
            links_previous=False,
        ), answer),
        context=Scope(
            environment=environment,
            tool=_BING,
            parameter="query",
            argument=query,
        ),
    )
    fetches = []
    for index, url in enumerate(urls):
        fetches.append(CandidatePlan(
            f"fetch_{index}",
            (PlanStep(
                subgoal=f"open {url}",
                action=url,
                tool=_FETCH,
                parameter="url",
                argument=url,
                estimated_cost=1,
                links_previous=False,
            ), answer),
            context=Scope(environment=environment),
        ))
    return (search, *fetches)


def render_planner_graph_markdown(
    *,
    question: str,
    n: int,
    step_count: int,
    max_steps: int,
    plans: Sequence[CandidatePlan],
    graph: Optional[PlanningGraph] = None,
) -> str:
    """Markdown view of one planner universe. The vote is not the out-degree."""
    remaining = max(1, int(max_steps) - int(step_count))
    lines = [
        "# Planner 规划图",
        "",
        f"- 题目：{question}",
        f"- n：{int(n)}",
        f"- step_count：{int(step_count)}",
        f"- max_steps：{int(max_steps)}",
        f"- 剩余步数：{remaining}",
        "",
        "票数是有多少条 outline 把该步当作当前第一步。它不是出度。",
        "",
    ]
    if not plans:
        lines.append("Planner 没有返回可解析的 outline。这一步退回原来的单条下一步。")
        return "\n".join(lines) + "\n"
    built = graph or PlanningGraph(list(plans))
    lines.append("## Outlines")
    lines.append("")
    for plan in plans:
        lines.append(f"### {plan.plan_id}")
        lines.append("")
        for index, step in enumerate(plan.steps, 1):
            lines.append(f"{index}. `{step.subgoal}` · `{step.tool}`")
        lines.append("")
    lines.append("## 合并后的 DAG")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    successors: dict[str, set[str]] = {node.node_id: set() for node in built.nodes()}
    for node in built.nodes():
        shared = len(node.candidate_plan_ids) > 1
        label = f"{node.subgoal} · {node.tool}"
        if shared:
            label += " · shared"
        safe = label.replace('"', "'")
        lines.append(f'  {node.node_id}["{safe}"]')
        for pred in node.dependencies:
            successors.setdefault(pred, set()).add(node.node_id)
            lines.append(f"  {pred} --> {node.node_id}")
    lines.append("```")
    lines.append("")
    lines.append("## 前沿")
    lines.append("")
    chosen = built.select_node(policy="frontier")
    seen = []
    for plan_id in built.active_plan_ids():
        node_id = built.frontier_node_id(plan_id)
        if node_id and node_id not in seen:
            seen.append(node_id)
    if not seen:
        lines.append("没有可执行的前沿。")
    for node_id in seen:
        node = built.node(node_id)
        mark = "，选中" if chosen is not None and chosen.node_id == node_id else ""
        lines.append(
            f"- `{node_id}` `{node.subgoal}` · `{node.tool}`"
            f" · 票数 {built.frontier_vote(node_id)}"
            f" · 出度 {len(successors.get(node_id, ()))}"
            f"{mark}"
        )
    lines.append("")
    return "\n".join(lines)
