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
import os
import re
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse

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

_BAD_HOST_MARKERS = (
    "pinterest.", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "reddit.com", "quora.com", "youtube.com", "youtu.be",
    "linkedin.com", "amazon.com", "ebay.com", "shopify.", "blogspot.",
)
_PDF_RE = re.compile(r"\.pdf(?:$|[?#])", re.I)
_ROLE_RE = re.compile(r"role\s*=\s*([a-z_]+)", re.I)


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
    # instance_key keeps open/extract path instances from sharing completion state.
    return (
        action,
        tool,
        premises,
        state,
        usage,
        normalize_state(step.consumes),
        normalize_state(step.produces),
        norm(step.task_id),
        norm(step.instance_key),
    )


def _slug(tool: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in norm(tool))
    cleaned = cleaned.strip("_")
    return cleaned[:24] or "node"


class PlanningGraph:
    """DAG of one planning round. ``round_id`` / ``version`` identify that universe.

    After ``seal()``, the graph is historical: callers must not treat it as the
    next-round active DAG. Round-based regeneration builds a *new* PlanningGraph.
    """

    def __init__(self, plans: Sequence[CandidatePlan], *, round_id: str = "", version: int | str = "") -> None:
        if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
            raise TypeError("plans must be a sequence of CandidatePlan")
        self.round_id = round_id or new_id("round")
        self._version: str = str(version) if version not in ("", None) else self.round_id
        self._frozen = True
        self._sealed = False
        self._seal_snapshot: Optional[dict] = None
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
        self._artifacts: dict[str, dict] = {}
        self._bindings: dict[str, dict] = {}
        self._feedback: dict[str, dict] = {}
        self.attempt_count = 0
        self.max_attempts = 3
        self.regeneration_count = 0
        self.attempt_history: list = []
        self._step1_attempts: dict[str, int] = {}
        self.protect_children_of = ""
        self._build(copied)

    @property
    def dag_id(self) -> str:
        return self.round_id

    @property
    def version(self) -> str:
        """Stable DAG version id for round-based regeneration."""
        return self._version

    @property
    def sealed(self) -> bool:
        return bool(self._sealed)

    def seal(self) -> dict:
        """Mark this DAG historical and capture an immutable structural snapshot."""
        from hypothesis_round_memory import freeze_dag_snapshot

        self._sealed = True
        self._seal_snapshot = freeze_dag_snapshot(self)
        return dict(self._seal_snapshot)

    def seal_snapshot(self) -> dict:
        if self._seal_snapshot is None:
            return self.seal()
        return copy.deepcopy(self._seal_snapshot)

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
        if policy == "max_out_degree":
            return self._select_max_out_degree()
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

    def active_successors(self, node_id: str) -> tuple[str, ...]:
        """Distinct PENDING direct successors still used by active outlines."""
        found: list[str] = []
        seen = set()
        for plan_id in self.active_plan_ids():
            node_ids = self.plan_node_ids(plan_id)
            for index, current in enumerate(node_ids):
                if current != node_id or index + 1 >= len(node_ids):
                    continue
                nxt = node_ids[index + 1]
                if nxt in seen:
                    continue
                if self._nodes[nxt].status != NodeStatus.PENDING:
                    continue
                seen.add(nxt)
                found.append(nxt)
        return tuple(found)

    def out_degree(self, node_id: str) -> int:
        """Active pending out-degree. Failed or retired children do not count."""
        if node_id not in self._nodes:
            return 0
        return len(self.active_successors(node_id))

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

    def prune_unique_downstream(self, node_id: str) -> tuple[str, ...]:
        """Fail the node and remove successors exclusive to its outlines.

        Shared successors used by other still-active outlines stay. Victims are
        the active outlines that contain the failed node.
        """
        if node_id not in self._nodes:
            return ()
        victims = [
            plan_id
            for plan_id in self.active_plan_ids()
            if node_id in self.plan_node_ids(plan_id)
        ]
        keep = set(self.active_plan_ids()) - set(victims)
        exclusive: list[str] = []
        for plan_id in victims:
            node_ids = self.plan_node_ids(plan_id)
            if node_id not in node_ids:
                continue
            start = node_ids.index(node_id)
            for downstream in node_ids[start:]:
                still_shared = any(
                    downstream in self.plan_node_ids(other) for other in keep
                )
                if still_shared:
                    continue
                if downstream not in exclusive:
                    exclusive.append(downstream)
        for item in exclusive:
            self.mark_failed(item)
        if node_id not in exclusive:
            self.mark_failed(node_id)
        return self.retire_plans(victims)

    def retire_plans(
        self,
        plan_ids: Iterable[str],
        *,
        status: PlanStatus = PlanStatus.PRUNED,
        prune_reason: Optional[str] = None,
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
            if prune_reason and status == PlanStatus.PRUNED:
                plan.prune_reason = str(prune_reason)
            changed.append(plan_id)
        self.refresh_orphans()
        return tuple(sorted(changed))

    def refresh_orphans(self) -> tuple[str, ...]:
        """Mark PENDING nodes that no ACTIVE plan still references.

        Satisfied nodes and their outputs stay. Failed nodes stay failed.
        """
        active = set(self.active_plan_ids())
        changed: list[str] = []
        for node_id, node in self._nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            referenced = any(node_id in self._plan_nodes.get(plan_id, ()) for plan_id in active)
            if referenced:
                continue
            node.status = NodeStatus.ORPHANED
            changed.append(node_id)
        return tuple(changed)

    def complete_plans(self, plan_ids: Iterable[str]) -> tuple[str, ...]:
        changed: list[str] = []
        for plan_id in plan_ids:
            plan = self._plans.get(plan_id)
            if plan is None or plan.status != PlanStatus.ACTIVE:
                continue
            plan.status = PlanStatus.COMPLETED
            changed.append(plan_id)
        self.refresh_orphans()
        return tuple(sorted(changed))

    def step1_node_id(self, plan_id: str) -> Optional[str]:
        """First not-yet-executed node on an ACTIVE plan. Satisfied prefixes are skipped."""
        nodes = self._plan_nodes.get(plan_id) or ()
        for node_id in nodes:
            status = self._nodes[node_id].status
            if status == NodeStatus.SATISFIED:
                continue
            if status != NodeStatus.PENDING:
                return None
            return node_id
        return None

    def is_step1_node(self, node_id: str) -> bool:
        return any(self.step1_node_id(plan_id) == node_id for plan_id in self._plan_order)

    def plans_starting_with(self, node_id: str, *, active_only: bool = True) -> tuple[str, ...]:
        found: list[str] = []
        for plan_id in self._plan_order:
            if active_only and self._plans[plan_id].status != PlanStatus.ACTIVE:
                continue
            if self.step1_node_id(plan_id) == node_id:
                found.append(plan_id)
        return tuple(found)

    def compute_step1_coverage(self) -> dict[str, int]:
        """Count ACTIVE plans whose current step 1 is v. Not out-degree."""
        coverage: dict[str, int] = {}
        for plan_id in self.active_plan_ids():
            first = self.step1_node_id(plan_id)
            if not first:
                continue
            coverage[first] = coverage.get(first, 0) + 1
        return coverage

    def is_eligible_step1(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        if self.compute_step1_coverage().get(node_id, 0) <= 0:
            return False
        return self._is_executable(node_id)

    def step1_attempt_count(self, node_id: str) -> int:
        return int(self._step1_attempts.get(node_id, 0))

    def select_step1(self) -> Optional[tuple[str, int, dict[str, int]]]:
        """argmax Coverage over eligible static Step-1 nodes.

        Tie-break is least attempts, never-attempted first, then stable node id.
        This runs only when coverage saturation did not already demand regeneration.
        """
        coverage = self.compute_step1_coverage()
        eligible = {
            node_id: count
            for node_id, count in coverage.items()
            if self.is_eligible_step1(node_id)
        }
        if not eligible:
            return None
        max_count = max(eligible.values())
        tied = [node_id for node_id, count in eligible.items() if count == max_count]

        def _key(node_id: str) -> tuple:
            attempts = self.step1_attempt_count(node_id)
            never = 0 if attempts == 0 else 1
            return (attempts, never, node_id)

        selected = min(tied, key=_key)
        return selected, max_count, eligible

    def coverage_saturated(self) -> bool:
        if self.attempt_count == 0:
            return False
        coverage = self.compute_step1_coverage()
        eligible = {
            node_id: count
            for node_id, count in coverage.items()
            if self.is_eligible_step1(node_id)
        }
        if len(eligible) < 2:
            return False
        parent = str(getattr(self, "protect_children_of", "") or "")
        if parent and any(self.is_child_of(parent, node_id) for node_id in eligible):
            return False
        top = max(eligible.values())
        return sum(1 for count in eligible.values() if count == top) >= 2

    def is_child_of(self, parent_id: str, node_id: str) -> bool:
        for plan_id in self.active_plan_ids():
            sequence = self._plan_nodes.get(plan_id) or ()
            if parent_id not in sequence or node_id not in sequence:
                continue
            if sequence.index(node_id) == sequence.index(parent_id) + 1:
                return True
        return False

    def budget_reached(self) -> bool:
        return self.attempt_count >= self.max_attempts

    def has_active_executable_step1(self) -> bool:
        coverage = self.compute_step1_coverage()
        return any(self.is_eligible_step1(node_id) for node_id in coverage)

    def regeneration_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.budget_reached():
            reasons.append("MAX_STEP1_ATTEMPTS")
        if self.coverage_saturated():
            reasons.append("COVERAGE_SATURATED")
        if not self.has_active_executable_step1():
            reasons.append("NO_EXECUTABLE_CANDIDATE")
        return reasons

    def record_step1_attempt(
        self,
        node_id: str,
        *,
        attempt_id: str,
        selected_plan_ids: Sequence[str],
        coverage: int,
    ):
        from cghe_types import AttemptRecord

        for existing in self.attempt_history:
            if existing.attempt_id == attempt_id:
                return existing
        self.attempt_count += 1
        self._step1_attempts[node_id] = self.step1_attempt_count(node_id) + 1
        record = AttemptRecord(
            attempt_id=attempt_id,
            dag_id=self.dag_id,
            step1_node_id=node_id,
            selected_plan_ids=tuple(selected_plan_ids),
            coverage_at_selection=int(coverage),
        )
        self.attempt_history.append(record)
        return record

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        execution_status: str,
        verifier_status: Optional[str] = None,
        evidence_id: Optional[str] = None,
        pruning_applied: bool = False,
        pruned_plan_ids: Sequence[str] = (),
    ) -> None:
        for record in self.attempt_history:
            if record.attempt_id != attempt_id:
                continue
            record.execution_status = execution_status
            record.verifier_status = verifier_status
            record.evidence_id = evidence_id
            record.pruning_applied = bool(pruning_applied)
            record.pruned_plan_ids = tuple(pruned_plan_ids)
            return

    def _deps_ready(self, plan_id: str, node_id: str) -> bool:
        preds = self._predecessors.get((plan_id, node_id), ())
        return all(self._nodes[pred].status == NodeStatus.SATISFIED for pred in preds)

    def _search_urls_for(self, node_id: str) -> list[str]:
        urls: list[str] = []
        for pred in self._nodes[node_id].dependencies:
            art = self._artifacts.get(pred) or {}
            if art.get("kind") == "search_results":
                for item in art.get("urls") or []:
                    text = str(item or "").strip()
                    if text and text not in urls:
                        urls.append(text)
        return urls

    def _page_content_for(self, node_id: str) -> str:
        for pred in self._nodes[node_id].dependencies:
            art = self._artifacts.get(pred) or {}
            if art.get("kind") == "page_content":
                return str(art.get("text") or "")
            binding = self._bindings.get(pred) or {}
            if binding.get("page_content"):
                return str(binding.get("page_content") or "")
        return ""

    def _role_for_open(self, node_id: str) -> str:
        node = self._nodes[node_id]
        blob = f"{node.action} {node.subgoal}".lower()
        match = _ROLE_RE.search(blob)
        if match:
            return match.group(1).lower()
        if "wiki" in blob or "encyclopedia" in blob:
            return "wiki_entity"
        if "cast" in blob or "imdb" in blob:
            return "cast"
        if "census" in blob or "population" in blob:
            return "census"
        if "official" in blob or "gov" in blob:
            return "official"
        return "wiki_topic"

    def _search_result_rows_for(self, node_id: str) -> list[dict]:
        rows: list[dict] = []
        for pred in self._nodes[node_id].dependencies:
            art = self._artifacts.get(pred) or {}
            if art.get("kind") != "search_results":
                continue
            for item in list(art.get("results") or []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                rows.append({
                    "title": str(item.get("title") or ""),
                    "url": url,
                    "snippet": str(item.get("snippet") or ""),
                })
            if not rows:
                for url in list(art.get("urls") or []):
                    text = str(url or "").strip()
                    if text:
                        rows.append({"title": "", "url": text, "snippet": ""})
        return rows

    @staticmethod
    def _host(url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower()
        except Exception:
            return ""

    def _url_role_score(self, *, role: str, title: str, url: str, snippet: str, rank: int) -> float:
        host = self._host(url)
        low = f"{title} {snippet} {url}".lower()
        score = max(0.0, 12.0 - float(rank))
        if any(marker in host for marker in _BAD_HOST_MARKERS):
            score -= 20.0
        if _PDF_RE.search(url):
            score -= 8.0
        if "wikipedia.org" in host or "wiki" in host:
            score += 8.0
        if role.startswith("wiki") and ("wikipedia.org" in host or "britannica.com" in host):
            score += 6.0
        if role == "cast" and any(tok in host or tok in low for tok in ("imdb.", "cast", "actor", "actress")):
            score += 7.0
        if role == "census" and any(tok in host or tok in low for tok in ("census", "population", ".gov")):
            score += 7.0
        if role == "official" and (host.endswith(".gov") or "official" in low):
            score += 7.0
        if role in {"wiki_entity", "wiki_topic"} and any(tok in low for tok in ("film", "actor", "born", "county", "nationality")):
            score += 1.5
        # Prefer encyclopedia over social/spam even at worse rank.
        if "wikipedia.org" in host:
            score += 2.0
        return score

    def _try_bind_open(self, node_id: str) -> bool:
        """Assign a concrete search URL to an open node when search artifacts exist."""
        if self._bindings.get(node_id, {}).get("url"):
            return True
        rows = self._search_result_rows_for(node_id)
        if not rows:
            urls = self._search_urls_for(node_id)
            rows = [{"title": "", "url": url, "snippet": ""} for url in urls]
        if not rows:
            return False
        taken = {
            str((self._bindings.get(other) or {}).get("url") or "")
            for other in self._nodes
            if other != node_id
        }
        blocked_meta = self._bindings.get("__blocked_urls__")
        if isinstance(blocked_meta, dict):
            blocked = {str(url) for url, meta in blocked_meta.items() if meta}
        else:
            blocked = set()
        # Optional ablation: ignore role scoring and bind by rank only.
        if str(os.environ.get("CGHE_NO_ROLE_BIND") or "").strip().lower() in {"1", "true", "yes"}:
            chosen = next((row["url"] for row in rows if row["url"] not in taken and row["url"] not in blocked), "")
            if not chosen:
                return False
            self._bindings[node_id] = {
                **(self._bindings.get(node_id) or {}),
                "url": chosen,
                "role": "rank",
                "title": "",
                "snippet": "",
            }
            self._nodes[node_id].argument = chosen
            return True
        role = self._role_for_open(node_id)
        ranked = []
        for index, row in enumerate(rows):
            url = row["url"]
            if url in taken or url in blocked:
                continue
            ranked.append((
                self._url_role_score(
                    role=role,
                    title=row.get("title", ""),
                    url=url,
                    snippet=row.get("snippet", ""),
                    rank=index,
                ),
                -index,
                url,
                row,
            ))
        if not ranked:
            self._bindings[node_id] = {
                **(self._bindings.get(node_id) or {}),
                "fetch_blocked": True,
                "role": role,
            }
            return False
        ranked.sort(reverse=True)
        score, _neg_rank, chosen, row = ranked[0]
        if score < -5:
            # All remaining candidates are junk; mark fetch blocked so caller can prune.
            self._bindings[node_id] = {
                **(self._bindings.get(node_id) or {}),
                "fetch_blocked": True,
                "role": role,
            }
            return False
        self._bindings[node_id] = {
            **(self._bindings.get(node_id) or {}),
            "url": chosen,
            "role": role,
            "title": row.get("title", ""),
            "snippet": row.get("snippet", ""),
            "url_score": score,
        }
        self._nodes[node_id].argument = chosen
        return True

    def block_url(self, url: str) -> None:
        text = str(url or "").strip()
        if not text:
            return
        blocked = self._bindings.get("__blocked_urls__")
        if not isinstance(blocked, dict):
            blocked = {}
        blocked[text] = True
        self._bindings["__blocked_urls__"] = blocked

    def open_url_score(self, node_id: str) -> float:
        meta = self._bindings.get(node_id) or {}
        try:
            return float(meta.get("url_score") or 0.0)
        except Exception:
            return 0.0

    def _inputs_ready(self, node_id: str) -> bool:
        node = self._nodes[node_id]
        consumes = set(node.consumes)
        if "search_url" in consumes:
            if not self._try_bind_open(node_id):
                return False
        if "page_content" in consumes:
            if not self._page_content_for(node_id).strip():
                return False
        return True

    def _is_executable(self, node_id: str) -> bool:
        node = self._nodes[node_id]
        if node.status != NodeStatus.PENDING:
            return False
        if self.sharing_score(node_id) <= 0:
            return False
        deps_ok = False
        for plan_id in self.active_plan_ids():
            if node_id not in self._plan_nodes.get(plan_id, ()):
                continue
            if self._deps_ready(plan_id, node_id):
                deps_ok = True
                break
        if not deps_ok:
            return False
        return self._inputs_ready(node_id)

    def frontier_node_id(self, plan_id: str) -> Optional[str]:
        """First unfinished READY step of one outline. Later steps are not eligible."""
        if plan_id not in self.active_plan_ids():
            return None
        for node_id in self.plan_node_ids(plan_id):
            node = self._nodes[node_id]
            if node.status != NodeStatus.PENDING:
                continue
            if not self._deps_ready(plan_id, node_id):
                continue
            if not self._inputs_ready(node_id):
                return None
            return node_id
        return None

    def store_search_results(self, node_id: str, urls: Sequence[str], *, results: Optional[Sequence[dict]] = None) -> None:
        cleaned = []
        for url in urls:
            text = str(url or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        payload = {"kind": "search_results", "urls": cleaned, "results": list(results or [])}
        if not payload["results"]:
            payload["results"] = [{"title": "", "url": url, "snippet": ""} for url in cleaned]
        self._artifacts[node_id] = payload
        for child_id, child in self._nodes.items():
            if "search_url" not in child.consumes:
                continue
            if node_id not in child.dependencies:
                continue
            self._try_bind_open(child_id)

    def store_page_content(self, node_id: str, text: str) -> None:
        body = str(text or "")
        self._artifacts[node_id] = {"kind": "page_content", "text": body}
        self._bindings[node_id] = {**(self._bindings.get(node_id) or {}), "page_content": body}

    def bound_url(self, node_id: str) -> str:
        return str((self._bindings.get(node_id) or {}).get("url") or self._nodes[node_id].argument or "")

    def tool_input_for(self, node_id: str) -> str:
        """Concrete executor input from artifacts. Never invents a URL."""
        node = self._nodes[node_id]
        if "search_url" in node.consumes:
            self._try_bind_open(node_id)
            return self.bound_url(node_id)
        if "page_content" in node.consumes:
            page = self._page_content_for(node_id)
            if page:
                return (
                    "Extract the asked fact from this page only. "
                    "Do not use search snippets.\n\n"
                    f"{page[:8000]}"
                )
            return ""
        return str(node.action or node.subgoal or "")

    def record_feedback(
        self,
        node_id: str,
        *,
        execution: str,
        evidence: str,
        goal: str,
    ) -> dict:
        item = {
            "node_id": node_id,
            "task_id": self._nodes[node_id].task_id,
            "execution": execution,
            "evidence": evidence,
            "goal": goal,
            "input_url": self.bound_url(node_id) if "search_url" in self._nodes[node_id].consumes else "",
            "has_page": bool(self._page_content_for(node_id) or (self._artifacts.get(node_id) or {}).get("kind") == "page_content"),
        }
        self._feedback[node_id] = item
        return item

    def feedback(self, node_id: str = "") -> dict:
        if node_id:
            return dict(self._feedback.get(node_id) or {})
        return {key: dict(value) for key, value in self._feedback.items()}

    def evidence_sufficient(self, question: str = "") -> bool:
        if any(str(item.get("evidence") or "") == "sufficient" for item in self._feedback.values()):
            return True
        # Prefer pages that contain entity/number clues for the question.
        pages = self.collected_page_texts()
        if not pages:
            return False
        if question:
            try:
                from sft_prompts import page_has_answer_clues

                if any(page_has_answer_clues(question, page) for page in pages):
                    return True
            except Exception:
                pass
            return False
        return any(len(page) >= 400 for page in pages)

    def collected_page_texts(self) -> list[str]:
        pages = []
        for art in self._artifacts.values():
            if art.get("kind") == "page_content":
                text = str(art.get("text") or "").strip()
                if text:
                    pages.append(text)
        return pages

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

    def _progress_rank(self, node_id: str) -> int:
        """Prefer advancing a bound open/extract over starting another search."""
        node = self._nodes[node_id]
        consumes = set(node.consumes)
        if "page_content" in consumes:
            return 0
        if "search_url" in consumes:
            return 1
        return 2

    def _frontier_candidates(self) -> list[str]:
        candidates: list[str] = []
        for plan_id in self.active_plan_ids():
            frontier = self.frontier_node_id(plan_id)
            if frontier is not None and frontier not in candidates:
                candidates.append(frontier)
        return candidates

    def _select_frontier(self) -> Optional[GraphNode]:
        candidates = self._frontier_candidates()
        if not candidates:
            return None
        # Prefer READY open/extract over another search. Among opens, prefer
        # higher role/snippet URL scores. Votes break remaining ties.
        candidates.sort(
            key=lambda node_id: (
                self._progress_rank(node_id),
                -self.open_url_score(node_id),
                -self.frontier_vote(node_id),
                self._nodes[node_id].estimated_cost,
                node_id,
            )
        )
        return copy.deepcopy(self._nodes[candidates[0]])

    def _select_max_out_degree(self) -> Optional[GraphNode]:
        """Pick among READY frontiers: advance open/extract first, else max out-degree.

        Out-degree chooses among competing step1 searches. Once a path has a
        bound open/extract READY, that local progress beats starting another
        search just because it has more children. READY opens prefer higher
        role-scored URLs.
        """
        candidates = self._frontier_candidates()
        if not candidates:
            return None
        candidates.sort(
            key=lambda node_id: (
                self._progress_rank(node_id),
                -self.open_url_score(node_id),
                -self.out_degree(node_id),
                -self.sharing_score(node_id),
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
                task_id=str(sample.task_id or ""),
                consumes=normalize_state(sample.consumes),
                produces=normalize_state(sample.produces),
                instance_key=str(sample.instance_key or ""),
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
