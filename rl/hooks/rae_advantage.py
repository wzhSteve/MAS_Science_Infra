"""RAE / parent-child advantage helpers.

R0: zero advantages on shared prefix tokens for child/probe rows.
R1-lite: soft scale by validate/invalidate/abstain.
R1-full: support-set Renorm π_tgt and A^RAE = log π_tgt - log π_old (row-level softprox).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from verl import DataProto


def _as_list(arr: Any) -> List[Any]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)


def _nt_get(ntb: Any, *keys: str) -> Any:
    for k in keys:
        if k in ntb:
            return ntb.get(k)
    return None


def zero_prefix_advantages_for_children(
    batch: DataProto,
    *,
    role_key: str = "rollout_role",
    boundary_key: str = "resume_boundary",
) -> DataProto:
    """R0: for role in {child, probe}, zero advantage before resume boundary.

    ``resume_boundary`` is the number of prompt+prefix response tokens that are
    shared with the parent (non_tensor int per row). If missing, no-op.
    """
    ntb = batch.non_tensor_batch
    roles = _as_list(_nt_get(ntb, role_key, "role_list"))
    bounds = _as_list(_nt_get(ntb, boundary_key, "resume_boundary_list"))
    if not roles or "advantages" not in batch.batch:
        return batch
    adv = batch.batch["advantages"]
    n = adv.shape[0]
    for i in range(min(n, len(roles))):
        role = str(roles[i] or "parent").lower()
        if role not in ("child", "probe", "branch"):
            continue
        if i >= len(bounds):
            continue
        try:
            b = int(bounds[i] or 0)
        except (TypeError, ValueError):
            continue
        if b <= 0:
            continue
        resp_len = adv.shape[-1]
        cut = min(b, resp_len)
        adv[i, :cut] = 0.0
    batch.batch["advantages"] = adv
    if "returns" in batch.batch:
        ret = batch.batch["returns"]
        for i in range(min(n, len(roles))):
            role = str(roles[i] or "parent").lower()
            if role not in ("child", "probe", "branch"):
                continue
            if i >= len(bounds):
                continue
            try:
                b = int(bounds[i] or 0)
            except (TypeError, ValueError):
                continue
            if b > 0:
                ret[i, : min(b, ret.shape[-1])] = 0.0
        batch.batch["returns"] = ret
    return batch


def apply_rae_verdict_advantages(
    batch: DataProto,
    *,
    p_plus: float = 0.8,
    epsilon_f: float = 1e-3,
) -> DataProto:
    """R1-lite: scale row advantages by verdict without full Renorm π_tgt.

    Expects non_tensor ``verdict_list`` with values validate|invalidate|abstain|none.
    """
    ntb = batch.non_tensor_batch
    verdicts = _as_list(_nt_get(ntb, "verdict_list", "verdict"))
    if not verdicts or "advantages" not in batch.batch:
        return batch
    adv = batch.batch["advantages"]
    mask = batch.batch.get("response_mask")
    n = adv.shape[0]
    log_eps = float(np.log(max(float(epsilon_f), 1e-8)))
    for i in range(min(n, len(verdicts))):
        v = str(verdicts[i] or "none").lower()
        if v == "validate":
            scale = 1.0
            bias = 0.5
        elif v == "invalidate":
            scale = 1.0
            bias = log_eps
        elif v == "abstain":
            scale = 0.25
            bias = 0.0
        else:
            continue
        row = adv[i]
        if mask is not None:
            m = mask[i].to(dtype=row.dtype)
            row = row * scale * m + bias * m
            adv[i] = row
        else:
            adv[i] = row * scale + bias
    batch.batch["advantages"] = adv
    return batch


def renorm_target_probs(
    old_probs: Sequence[float],
    verdicts: Sequence[str],
    *,
    epsilon_f: float = 1e-3,
) -> List[float]:
    """Support-set Renorm: drop invalidate mass, renormalize validate+abstain.

    invalidate mass is removed; validate keeps proportional mass; abstain keeps
    reduced mass. Returns π_tgt over the same index set (sums to 1 if any mass).
    """
    n = len(old_probs)
    if n == 0:
        return []
    kept = []
    for p, v in zip(old_probs, verdicts):
        vv = str(v or "none").lower()
        pp = max(float(p), 0.0)
        if vv == "invalidate":
            kept.append(0.0)
        elif vv == "abstain":
            kept.append(pp * 0.25)
        elif vv == "validate":
            kept.append(pp)
        else:
            kept.append(pp)
    s = sum(kept)
    if s <= 0:
        # all invalidated → uniform over epsilon floor (avoid hard zero)
        eps = float(epsilon_f)
        return [eps / n] * n
    return [k / s for k in kept]


def apply_rae_full_tgt_advantages(
    batch: DataProto,
    *,
    epsilon_f: float = 1e-3,
) -> DataProto:
    """R1-full: A_RAE ≈ log π_tgt - log π_old applied as response-token bias.

    Groups rows by (uid/data_id, action_key) when available; otherwise treats each
    row independently with a synthetic old_prob=1.
    Uses old_logprob tensor if present; else uses uniform proxy.
    """
    ntb = batch.non_tensor_batch
    verdicts = _as_list(_nt_get(ntb, "verdict_list", "verdict"))
    if not verdicts or "advantages" not in batch.batch:
        return batch
    action_keys = _as_list(_nt_get(ntb, "action_key_list", "action_key") or [])
    uids = _as_list(_nt_get(ntb, "uid", "data_id") or [])
    adv = batch.batch["advantages"]
    mask = batch.batch.get("response_mask")
    n = adv.shape[0]
    old_logprobs = batch.batch.get("old_log_probs") or batch.batch.get("old_logprobs")

    groups: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        ak = action_keys[i] if i < len(action_keys) else ""
        uid = uids[i] if i < len(uids) else i
        key = (uid, ak) if ak else ("row", i)
        groups[key].append(i)

    a_rae = [0.0] * n
    for idxs in groups.values():
        # proxy old probs from exp(mean old_logprob) or uniform
        old_ps: List[float] = []
        for i in idxs:
            if old_logprobs is not None:
                row = old_logprobs[i]
                if mask is not None:
                    m = mask[i].to(dtype=row.dtype)
                    denom = float(m.sum().item()) if hasattr(m, "sum") else float(np.sum(m))
                    if denom > 0:
                        mean_lp = float((row * m).sum().item() / denom)
                    else:
                        mean_lp = 0.0
                else:
                    mean_lp = float(row.mean().item()) if hasattr(row, "mean") else 0.0
                old_ps.append(float(np.exp(mean_lp)))
            else:
                old_ps.append(1.0 / max(len(idxs), 1))
        vs = [str(verdicts[i] if i < len(verdicts) else "none") for i in idxs]
        tgt = renorm_target_probs(old_ps, vs, epsilon_f=epsilon_f)
        for j, i in enumerate(idxs):
            p_old = max(float(old_ps[j]), 1e-8)
            p_tgt = max(float(tgt[j]), 1e-8)
            a_rae[i] = float(np.log(p_tgt) - np.log(p_old))

    for i in range(n):
        if abs(a_rae[i]) < 1e-12:
            continue
        bias = float(a_rae[i])
        if mask is not None:
            m = mask[i].to(dtype=adv.dtype)
            adv[i] = adv[i] * m + bias * m
        else:
            adv[i] = adv[i] + bias
    batch.batch["advantages"] = adv
    return batch


def adjudicate_action_group(
    outcomes: List[bool],
    *,
    p_plus: float = 0.8,
    k_min: int = 2,
) -> str:
    """Return validate | invalidate | abstain for one action_key group."""
    n = len(outcomes)
    if n < max(1, int(k_min)):
        return "abstain"
    p = sum(1 for y in outcomes if y) / float(n)
    if p <= 0.0:
        return "invalidate"
    if p >= float(p_plus):
        return "validate"
    return "abstain"


def apply_dead_end_backprop_verdicts(
    verdicts: Sequence[str],
    *,
    roles: Sequence[str],
    parent_ids: Sequence[str],
    data_ids: Sequence[str],
    depth: int = 1,
) -> List[str]:
    """If all children of a parent are invalidate, mark parent invalidate (depth=1)."""
    out = [str(v or "none") for v in verdicts]
    if int(depth) <= 0:
        return out
    # Map parent_id -> child indices within same data_id
    children: Dict[str, List[int]] = defaultdict(list)
    id_to_idx: Dict[str, int] = {}
    for i, (role, pid, did) in enumerate(zip(roles, parent_ids, data_ids)):
        # reconstruct a synthetic leaf id from position for parents without explicit id
        id_to_idx[f"{did}::{i}"] = i
        if str(role).lower() in ("child", "probe", "branch") and pid:
            children[f"{did}::{pid}"].append(i)
    # Also group by resume_parent_id string alone within data_id
    by_parent: Dict[tuple, List[int]] = defaultdict(list)
    for i, (role, pid, did) in enumerate(zip(roles, parent_ids, data_ids)):
        if str(role).lower() in ("child", "probe", "branch") and pid:
            by_parent[(str(did), str(pid))].append(i)
    parent_row: Dict[tuple, int] = {}
    for i, (role, pid, did) in enumerate(zip(roles, parent_ids, data_ids)):
        # parent rows: no resume parent, role parent
        if str(role).lower() in ("parent", "", "root") or not pid:
            # sample's own rollout id is unknown; use matching via parent_ids pointing to meta
            pass
    # Heuristic: for each (data_id, parent_id) child group, if all invalidate,
    # find a parent row in same data_id with role parent and set invalidate.
    parents_by_data: Dict[str, List[int]] = defaultdict(list)
    for i, (role, did) in enumerate(zip(roles, data_ids)):
        if str(role).lower() in ("parent", "root", ""):
            parents_by_data[str(did)].append(i)
    for (did, _pid), idxs in by_parent.items():
        if not idxs:
            continue
        if all(str(out[j]).lower() == "invalidate" for j in idxs):
            for pi in parents_by_data.get(str(did), []):
                if str(out[pi]).lower() in ("none", "abstain", "validate", ""):
                    out[pi] = "invalidate"
    return out


# --- RolloutTree credit assignment (new_framework P2; pure python, no torch) --


def k_hop_cumulative_reward(
    tree: Any,
    node_id: str,
    k: int = 1,
) -> Optional[float]:
    """Equal-weight mean of node rewards along the ancestor path within k hops.

    ``tree`` is a RolloutTree (dict or model). Returns None when no reward is
    present on the window (leaf-only credit keeps legacy behavior when absent).
    """
    nodes = tree.get("nodes") if isinstance(tree, dict) else getattr(tree, "nodes", None)
    if not nodes:
        return None
    by_id = {n.get("node_id") if isinstance(n, dict) else n.node_id: n for n in nodes}
    rewards: List[float] = []
    cur = node_id
    hops = 0
    seen = set()
    while cur is not None and cur not in seen and hops <= max(0, int(k)):
        seen.add(cur)
        node = by_id.get(cur)
        if node is None:
            break
        r = node.get("reward") if isinstance(node, dict) else getattr(node, "reward", None)
        if r is not None:
            try:
                rewards.append(float(r))
            except (TypeError, ValueError):
                pass
        nxt = node.get("parent_id") if isinstance(node, dict) else getattr(node, "parent_id", None)
        cur = nxt
        hops += 1
    if not rewards:
        return None
    return float(sum(rewards) / len(rewards))


def apply_verdicts_to_tree(
    tree: Any,
    verdicts: Dict[str, str],
    node_ids: Optional[List[str]] = None,
) -> int:
    """Write RAE verdicts (validate/invalidate/abstain) onto tree nodes.

    ``verdicts`` maps rollout_id -> verdict; ``node_ids`` optionally maps the
    order of tree children to rollout ids (when node ids were rewritten by the
    Daemon store). Returns the number of nodes updated.
    """
    nodes = tree.get("nodes") if isinstance(tree, dict) else getattr(tree, "nodes", None)
    if not nodes:
        return 0
    if node_ids is not None:
        # remap: children in order were assigned store rollout ids
        children = [n for n in nodes if (n.get("role") if isinstance(n, dict) else getattr(n, "role", "")) != "root"]
        for child, rid in zip(children, node_ids):
            if rid in verdicts:
                v = str(verdicts[rid] or "abstain")
                if isinstance(child, dict):
                    child["verdict"] = v
                else:
                    child.verdict = v
    updated = 0
    for n in nodes:
        nid = n.get("node_id") if isinstance(n, dict) else getattr(n, "node_id", None)
        if nid in verdicts:
            v = str(verdicts[nid] or "abstain")
            if isinstance(n, dict):
                if n.get("verdict") != v:
                    n["verdict"] = v
                    updated += 1
            else:
                if getattr(n, "verdict", None) != v:
                    n.verdict = v
                    updated += 1
    return updated
