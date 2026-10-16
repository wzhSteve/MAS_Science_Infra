"""Effective training configuration, local model discovery and preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from science_infra.env import repo_root

from rl.hooks.overlay import BRANCHING_ALGOS
from .experiments import (
    VALID_ALGOS,
    apply_gpu_selection,
    load_bundle,
    load_llm_section,
    load_secrets_env,
    normalize_rl_data_paths,
    workflow_executable,
)
from .model_resources import (
    ResourceError,
    ResourceSnapshot,
    get_resource,
    list_resources,
    resolve_binding,
)
from .paths import tir_agent_root

_LAUNCH_LOCK = threading.Lock()


class TrainingError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        data: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.data = data or {}


@dataclass(frozen=True)
class TrainingSource:
    source: str
    model_path: str
    resource_id: str | None = None
    resource_revision: int | None = None
    resource_name: str | None = None
    selection_source: str = "legacy"
    agent_id: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_path": self.model_path,
            "resource_id": self.resource_id,
            "resource_revision": self.resource_revision,
            "resource_name": self.resource_name,
            "selection_source": self.selection_source,
            "agent_id": self.agent_id,
        }


@dataclass
class TrainingPlan:
    experiment_id: str
    bundle: dict[str, Any]
    rl: dict[str, Any]
    source: TrainingSource
    algorithm: str
    algorithm_source: str
    profile: str
    gpu_ids: list[int]
    group_n: int
    branch_site_count: int
    trainable_agents: list[str]
    active_agents: list[str]
    checks: list[dict[str, str]]
    blocking_issues: list[dict[str, str]]
    warnings: list[dict[str, str]]
    launch_preview: list[str]
    revision: str

    @property
    def ready(self) -> bool:
        return not self.blocking_issues

    def public(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "ready": self.ready,
            "revision": self.revision,
            "effective": {
                "algorithm": self.algorithm,
                "algorithm_source": self.algorithm_source,
                "profile": self.profile,
                "model": self.source.public(),
                "gpu_ids": self.gpu_ids,
                "group_n": self.group_n,
                "branch_site_count": self.branch_site_count,
                "trainable_agents": self.trainable_agents,
                "active_agents": self.active_agents,
                "data": dict(self.rl.get("data") or {}),
            },
            "checks": self.checks,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "launch_preview": self.launch_preview,
        }


def _workflow_path() -> None:
    path = str(tir_agent_root())
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# RL5: rl.yaml v2 dual-track contract validation (V1-V6) + resolved_algo audit
# ---------------------------------------------------------------------------
# Three enforcement layers (RL_design §5.2, cheapest first):
#   save   — save_section(rl) raises rl_contract_validation (400) via the
#            shared ``validate_rl_contract`` before the yaml hits disk;
#   preflight — build_training_plan records ``rl_contract_validation``
#            blocking issues (same function, plan-time fail-fast);
#   launch — the expansion surface itself (apply_algo_overlay pydantic
#            validators + resolve_pipeline_spec) is the last line, already
#            wired by RL5's overlay rewrite.
# This module hosts the *plan-side* reader: it never imports torch/AGL —
# only rl.loss (declaration layer) does any pydantic work.

# v1 single-select keys (A9: permanent) vs v2 pipeline keys — dual-track
# classification for V5. tir carrier keys that legitimately live alongside a
# composed advantage section are sampling/gate/loss-leg keys, not pipeline
# keys, so they never trip V5.
_V1_SELECT_KEYS = ("algo",)
_V2_PIPELINE_SECTION = "advantage"


def _rl_contract_violations(rl: dict[str, Any]) -> list[dict[str, str]]:
    """V1-V6 rule scan of a raw rl config dict (v1 / v2 / dual). Read-only.

    Returns a list of ``{"rule", "field", "detail"}`` diagnostics in launch
    body shape (RL_design §5.2 error contract); empty list = contract clean.
    """
    violations: list[dict[str, str]] = []
    algorithm = rl.get("algorithm") if isinstance(rl.get("algorithm"), dict) else {}
    algo = str(rl.get("algo") or algorithm.get("tir_algo") or "grpo").lower().strip()

    from rl.loss import ADVANTAGE_BASES, ADVANTAGE_OPERATORS, expand_preset

    # V1 (required/type) + V2 (enum domain) on the v1 leg: unknown preset
    # names are rejected here — the permanent CLI/applyWorkflowSampling face
    # only ever carries the eight freezing names (A9).
    if rl.get("algo") is not None or algorithm.get("tir_algo") is not None:
        if algo not in ("grpo", "arpo", "appo", "aepo", "igpo", "gigpo", "rae", "flow_grpo"):
            violations.append(
                {
                    "rule": "V2",
                    "field": "algo" if rl.get("algo") is not None else "algorithm.tir_algo",
                    "detail": (
                        f"未知训练算法 {algo!r}；v1 单选合法域: "
                        "grpo/arpo/appo/aepo/igpo/gigpo/rae/flow_grpo (A9 永久保留)"
                    ),
                }
            )
        else:
            try:
                expand_preset(algo)
            except ValueError as error:
                violations.append(
                    {"rule": "V1", "field": "algo", "detail": str(error)}
                )

    # v2 advantage section: V1 types + V2 enum + V3 combination (XOR/no-dup/
    # forbidden stacks) + V4 numeric domain on params, via the declaration
    # layer's own validators (rl.loss) and the single combination enforcer
    # (rl.hooks.advantage.validate_pipeline_spec — kept as the one authority
    # so tensor-side resolution and plan-side preflight share the rule).
    advantage = algorithm.get(_V2_PIPELINE_SECTION)
    if isinstance(advantage, dict) and advantage:
        base = advantage.get("base", "grpo")
        if base not in ADVANTAGE_BASES:
            violations.append(
                {
                    "rule": "V2",
                    "field": "algorithm.advantage.base",
                    "detail": f"base 必须是 {ADVANTAGE_BASES} 之一, got {base!r}",
                }
            )
        seen: set[str] = set()
        ops_raw = advantage.get("operators") if isinstance(advantage.get("operators"), list) else []
        for index, op in enumerate(ops_raw):
            field = f"algorithm.advantage.operators[{index}]"
            if not isinstance(op, dict):
                violations.append(
                    {"rule": "V1", "field": field, "detail": f"算子条目必须是 {{id, params}} 对象, got {op!r}"}
                )
                continue
            op_id = str(op.get("id") or "")
            if op_id not in ADVANTAGE_OPERATORS:
                violations.append(
                    {"rule": "V2", "field": f"{field}.id", "detail": f"未知算子 {op_id!r}; 注册表: {ADVANTAGE_OPERATORS}"}
                )
            elif op_id in seen:
                violations.append(
                    {"rule": "V2", "field": f"{field}.id", "detail": f"算子 {op_id!r} 在栈内重复声明"}
                )
            seen.add(op_id)
            params = op.get("params")
            if params is not None and not isinstance(params, dict):
                violations.append(
                    {"rule": "V1", "field": f"{field}.params", "detail": f"params 必须是键值对象, got {params!r}"}
                )
        if "rae_verdict" in seen and "rae_full_tgt" in seen:
            violations.append(
                {
                    "rule": "V3",
                    "field": "algorithm.advantage.operators",
                    "detail": "rae_verdict 与 rae_full_tgt 互斥 (R1: 两者都消费 verdict_list 且改写方式冲突)",
                }
            )
        if "rae_full_tgt" in seen and "appo_credit" in seen:
            violations.append(
                {
                    "rule": "V3",
                    "field": "algorithm.advantage.operators",
                    "detail": "rae_full_tgt 与 appo_credit 禁止同栈 (R5: 分支行离场与在场语义冲突)",
                }
            )

    # V5 (dual-track mutex): a composed advantage section AND an explicit
    # fresh v1 select (yaml-authored ``algo``) are two declarations of the
    # same pipeline. The runtime resolution priority is defined and declared
    # (composed wins the advantage segment; algo stays as the sampling/gate
    # carrier — A9), but a silent mix is what the contract forbids: flag it
    # so the operator must pick a lane at save/preflight time.
    if isinstance(advantage, dict) and advantage and rl.get("algo") is not None:
        violations.append(
            {
                "rule": "V5",
                "field": "algorithm.advantage",
                "detail": (
                    f"v2 composed 段与 v1 algo={algo!r} 同时声明: advantage 真值取 composed 段、"
                    "algo 仅承载 sampling/门控语义; 请单轨声明（删除 algo 键或删除 advantage 段）"
                ),
            }
        )

    # V4 (numeric domain): future_kl triple on the tir leg + reward component
    # weights, mirroring the declaration-layer validators (rl.loss.LossSpec /
    # rl.rewards.assembly.RewardComponent) for configs that never pass
    # through a Spec construction on the plan path.
    import math

    tir = algorithm.get("tir") if isinstance(algorithm.get("tir"), dict) else {}
    for key in ("future_kl_weight", "decay_rate", "future_kl_clip_ratio"):
        value = tir.get(key)
        if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float))):
            violations.append(
                {"rule": "V4", "field": f"algorithm.tir.{key}", "detail": f"V4 类型错误: 必须为实数, got {value!r}"}
            )
        elif value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            violations.append(
                {"rule": "V4", "field": f"algorithm.tir.{key}", "detail": f"必须为非负有限数 (>=0, finite), got {value!r}"}
            )
    loss_mode = str(tir.get("loss_mode") or algorithm.get("loss_mode") or "vanilla")
    if loss_mode not in ("vanilla", "future_kl"):
        violations.append(
            {
                "rule": "V2",
                "field": "algorithm.tir.loss_mode",
                "detail": f"loss_mode 必须是 vanilla | future_kl, got {loss_mode!r}",
            }
        )
    reward = algorithm.get("reward") if isinstance(algorithm.get("reward"), dict) else {}
    comps = reward.get("components") if isinstance(reward.get("components"), list) else None
    for index, comp in enumerate(comps or []):
        if not isinstance(comp, dict):
            violations.append(
                {"rule": "V1", "field": f"algorithm.reward.components[{index}]", "detail": f"组件必须是 {{id, weight, enabled}} 对象, got {comp!r}"}
            )
            continue
        weight = comp.get("weight")
        if isinstance(weight, bool) or (weight is not None and not isinstance(weight, (int, float))):
            violations.append(
                {"rule": "V4", "field": f"algorithm.reward.components[{index}].weight", "detail": f"V4 类型错误: weight 必须为实数, got {weight!r}"}
            )
        elif weight is not None and (not math.isfinite(float(weight)) or float(weight) < 0.0):
            violations.append(
                {"rule": "V4", "field": f"algorithm.reward.components[{index}].weight", "detail": f"weight 必须为非负有限数 (>=0, finite), got {weight!r}"}
            )

    # V6 (future_kl connectivity, post-RL1-double-write wording): the write
    # side mirrors the FIPO keys into actor_rollout_ref.actor.policy_loss, so
    # a future_kl config is now truly connected. The plan-side check verifies
    # the mirror keys survive on this config and reports the connected state;
    # a *vanilla* config with stale mirror keys is the actionable smell.
    if loss_mode == "future_kl":
        mirror = (
            (rl.get("actor_rollout_ref") or {}).get("actor", {}).get("policy_loss")
            if isinstance((rl.get("actor_rollout_ref") or {}).get("actor"), dict)
            else None
        )
        mirror_mode = str((mirror or {}).get("loss_mode") or loss_mode)
        if mirror_mode != "future_kl":
            violations.append(
                {
                    "rule": "V6",
                    "field": "actor.policy_loss.loss_mode",
                    "detail": (
                        "V6 future_kl 连通: loss_mode=future_kl 但 actor.policy_loss.loss_mode="
                        f"{mirror_mode!r}（写侧镜像将在启动时由 apply_train_signal 补写; 若为用户显式"
                        "覆写导致不一致，请移除覆写或改为 future_kl）"
                    ),
                }
            )
    return violations


def validate_rl_contract(rl: dict[str, Any]) -> list[dict[str, str]]:
    """Public save-layer entry: raise-free scan used by save_section (400 body)."""
    if not isinstance(rl, dict):
        return [
            {
                "rule": "V1",
                "field": "rl",
                "detail": f"V1 类型错误: rl 配置必须是键值对象, got {type(rl).__name__}",
            }
        ]
    return _rl_contract_violations(rl)


def _resolved_algo_payload(rl: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize the resolved_algo audit block (RL_design §3.5.1) off a plan rl dict.

    Expansion happens here — after preflight, before the subprocess — so the
    signature and effective-rl.yaml are same-source, same-moment (no window
    where config changed but the signature did not). None when the rl dict
    cannot yield a pipeline at all (legacy defensive path).
    """
    if not isinstance(rl, dict):
        return None
    algorithm = rl.get("algorithm") if isinstance(rl.get("algorithm"), dict) else {}
    algo = str(rl.get("algo") or algorithm.get("tir_algo") or "grpo").lower().strip()
    try:
        from rl.loss import expand_preset  # declaration layer only (no torch)

        spec = expand_preset(algo)
    except ValueError:
        return None
    tir = algorithm.get("tir") if isinstance(algorithm.get("tir"), dict) else {}
    reward = algorithm.get("reward") if isinstance(algorithm.get("reward"), dict) else {}
    if not reward:
        # plan.rl is the pre-overlay raw yaml (the overlay runs inside the
        # training subprocess); derive the same RewardSpec the RL4 mirror
        # would write so the audit block reflects the delivered spec.
        try:
            from rl.rewards.assembly import reward_spec_from_tir

            comps = [
                {"id": c.id, "weight": c.weight, "enabled": c.enabled}
                for c in reward_spec_from_tir(dict(tir)).components
            ]
        except Exception:
            comps = []
    else:
        comps = [
            {
                "id": str(comp.get("id")),
                "weight": comp.get("weight", 1.0),
                "enabled": bool(comp.get("enabled", True)),
            }
            for comp in (reward.get("components") or [])
            if isinstance(comp, dict)
        ]
    loss_mode = str(tir.get("loss_mode") or "vanilla")
    payload: dict[str, Any] = {
        "preset": algo,
        "advantage": {
            "base": spec.base,
            "operators": [{"id": op.id, "params": dict(op.params)} for op in spec.operators],
        },
        "loss_mode": loss_mode,
        "reward": {"components": comps},
        "v2_composed": bool(
            isinstance(algorithm.get("advantage"), dict) and algorithm.get("advantage")
        ),
        "dual_track_note": "v1 algo 单选永久保留 (A9); v2 段出现时 composed 段为 advantage 真值",
    }
    return payload


def _resolved_algo_summary(payload: dict[str, Any] | None, group_n: int | None) -> str | None:
    """One-line human-readable resolved summary (RL_design §4.4).

    Rendered by TrainingConsole / TrainingHistory (webui); ``meta.algo`` stays
    the fallback for legacy runs that predate the resolved_algo field.
    """
    if not isinstance(payload, dict):
        return None
    advantage = payload.get("advantage") if isinstance(payload.get("advantage"), dict) else {}
    base = str(advantage.get("base") or "grpo")
    operators = ",".join(
        str(op.get("id")) for op in (advantage.get("operators") or []) if isinstance(op, dict)
    )
    reward = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
    comps = ",".join(
        f"{comp.get('id')}"
        + (f"×{comp.get('weight')}" if comp.get("weight") not in (None, 1) else "")
        + ("" if comp.get("enabled", True) else ":off")
        for comp in (reward.get("components") or [])
        if isinstance(comp, dict)
    )
    loss_mode = str(payload.get("loss_mode") or "vanilla")
    summary = f"base={base} + [{operators}], reward=[{comps}], loss={loss_mode}"
    if group_n is not None:
        summary += f", n={group_n}"
    return summary



def _model_path(rl: dict[str, Any]) -> str:
    return str(
        rl.get("model_path")
        or (rl.get("actor_rollout_ref") or {}).get("model", {}).get("path")
        or ""
    ).strip()


def _agent_model_id(
    workflow: dict[str, Any], agent_id: str | None
) -> str | None:
    if not agent_id:
        return None
    for agent in workflow.get("agents") or []:
        if isinstance(agent, dict) and str(agent.get("id") or "") == agent_id:
            value = str(agent.get("model") or "").strip()
            return value if value and value != "inherit" else None
    return None


def resolve_training_source(
    exp_id: str,
    rl: dict[str, Any] | None = None,
    *,
    workflow: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> TrainingSource:
    workflow = workflow or {}
    override_id = _agent_model_id(workflow, agent_id)
    if override_id:
        resource = get_resource(override_id)
        if resource.data["type"] != "training":
            raise ResourceError("Agent 指定的模型不可用于训练。")
        path = str(resource.data["config"]["model_path"]).strip()
        return TrainingSource(
            source="resource",
            model_path=path,
            resource_id=resource.data["id"],
            resource_revision=resource.data["revision"],
            resource_name=resource.data["name"],
            selection_source="agent_override",
            agent_id=agent_id,
        )
    resource = resolve_binding(exp_id, "training")
    if resource is not None:
        path = str(resource.data["config"]["model_path"]).strip()
        return TrainingSource(
            source="resource",
            model_path=path,
            resource_id=resource.data["id"],
            resource_revision=resource.data["revision"],
            resource_name=resource.data["name"],
            selection_source="experiment_default",
            agent_id=agent_id,
        )
    config = rl if rl is not None else load_bundle(exp_id)["rl"]
    return TrainingSource(
        source="legacy",
        model_path=_model_path(config),
        selection_source="legacy",
        agent_id=agent_id,
    )


def _allowed_model_roots() -> list[Path]:
    roots = [repo_root() / "LLM"]
    configured = os.environ.get("SCIENCE_MODEL_ROOTS", "")
    roots.extend(
        Path(item).expanduser()
        for item in configured.split(os.pathsep)
        if item.strip()
    )
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _candidate_model_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    candidates = [root] if (root / "config.json").is_file() else []
    candidates.extend(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "config.json").is_file()
    )
    return candidates[:100]


def _registered_training_paths() -> dict[str, str]:
    page = list_resources(type="training", offset=0, limit=100)
    registered: dict[str, str] = {}
    for resource in page["items"]:
        path = str((resource.get("config") or {}).get("model_path") or "").strip()
        if path:
            registered[str(Path(path).expanduser().resolve())] = resource["id"]
    return registered


def discover_local_models() -> list[dict[str, Any]]:
    registered = _registered_training_paths()
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in _allowed_model_roots():
        for candidate in _candidate_model_dirs(root):
            resolved = candidate.resolve()
            if resolved in seen or (resolved != root and root not in resolved.parents):
                continue
            seen.add(resolved)
            try:
                config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.append(
                {
                    "path": str(resolved),
                    "name": resolved.name,
                    "architectures": list(config.get("architectures") or []),
                    "model_type": config.get("model_type"),
                    "torch_dtype": config.get("torch_dtype"),
                    "registered_resource_id": registered.get(str(resolved)),
                }
            )
    return sorted(found, key=lambda item: (item["name"].lower(), item["path"]))


def local_model_candidate(model_path: str) -> dict[str, Any]:
    resolved = Path(model_path).expanduser().resolve()
    candidate = next(
        (item for item in discover_local_models() if Path(item["path"]) == resolved),
        None,
    )
    if candidate is None:
        raise ResourceError("本地模型不在允许的发现目录中。")
    return candidate


def _training_env(exp_id: str) -> dict[str, str]:
    """Match main's overrides without using the independent inference binding."""
    llm = load_llm_section(exp_id)
    secrets = load_secrets_env(exp_id)
    env: dict[str, str] = {}
    if secrets.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
    base = str(llm.get("base_url") or "").strip()
    model = str(llm.get("model") or "").strip()
    if base:
        env.update(OPENAI_API_BASE=base, OPENAI_BASE_URL=base)
    if model:
        env.update(OPENAI_MODEL=model, MODEL=model)
    return env


def _dependency_status() -> tuple[list[str], list[str]]:
    required = {
        "agentlightning": "Agent-Lightning",
        "verl": "VERL",
        "ray": "Ray",
        "torch": "PyTorch",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
    }
    present: list[str] = []
    missing: list[str] = []
    for module, label in required.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        (present if available else missing).append(label)
    return present, missing


def _data_path(value: Any) -> Path | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def build_training_plan(exp_id: str) -> TrainingPlan:
    from .process_manager import PROCS
    from .services import _sync_workflow_sampling_into_rl, list_gpus

    bundle = load_bundle(exp_id)
    raw_rl = deepcopy(bundle["rl"])
    workflow = bundle.get("workflow") or {}
    candidate_agents: list[str] = []
    try:
        _workflow_path()
        from workflow.compiler import trainable_agents as compile_trainable_agents
        from workflow.spec import MASSpec

        candidate_agents = compile_trainable_agents(MASSpec.model_validate(workflow))
    except Exception:
        candidate_agents = []
    active_agents = list(candidate_agents)
    source_error: str | None = None
    agent_sources: list[TrainingSource] = []
    for agent_id in active_agents or [None]:
        try:
            agent_sources.append(resolve_training_source(
                exp_id,
                raw_rl,
                workflow=workflow,
                agent_id=agent_id,
            ))
        except ResourceError as error:
            source_error = str(error)
            break
    source = agent_sources[0] if agent_sources else TrainingSource(
        source="invalid",
        model_path="",
        selection_source="agent_override",
        agent_id=active_agents[0] if active_agents else None,
    )
    distinct_models = {
        (item.resource_id or "", str(Path(item.model_path).expanduser()))
        for item in agent_sources
    }
    if not source_error and len(distinct_models) > 1:
        source_error = "多个参与训练的 Agent 使用了不同模型；当前单个训练进程只能优化一套权重，请统一模型后启动。"
    ids = [int(value) for value in (raw_rl.get("devices") or {}).get("ids") or [0]]
    try:
        rl = normalize_rl_data_paths(apply_gpu_selection(raw_rl, ids))
    except ValueError as error:
        raise TrainingError("invalid_data_path", str(error)) from error
    try:
        rl = _sync_workflow_sampling_into_rl(rl, workflow)
    except ValueError as error:
        raise TrainingError("invalid_sampling", f"Sampling 配置无效：{error}") from error
    algorithm_source = "workflow.sampling" if workflow.get("sampling") else "rl"
    # The script reads top-level algo first; Sampling has already synchronized it.
    algorithm = str(rl.get("algo") or "grpo").lower()
    rl["algo"] = algorithm
    rl["algorithm"] = {**(rl.get("algorithm") or {}), "tir_algo": algorithm}
    profile = str(rl.get("profile") or "fast")
    profile_note = rl.get("_profile_downgraded")
    rl = {key: value for key, value in rl.items() if not str(key).startswith("_")}
    rl.pop("active_agent", None)
    rl["model_path"] = source.model_path
    actor_rollout_ref = dict(rl.get("actor_rollout_ref") or {})
    model = dict(actor_rollout_ref.get("model") or {})
    model["path"] = source.model_path
    actor_rollout_ref["model"] = model
    rl["actor_rollout_ref"] = actor_rollout_ref

    rollout = dict(actor_rollout_ref.get("rollout") or {})
    group_n = int(rollout.get("n") or rl.get("rollout_per_gpu") or (2 if profile == "fast" else 4))
    rollout["n"] = group_n
    actor_rollout_ref["rollout"] = rollout
    rl["rollout_per_gpu"] = group_n
    sites = ((rl.get("algorithm") or {}).get("tir") or {}).get("sites")
    branch_site_count = sum(1 for site in sites or [] if site.get("enabled", True))
    if algorithm in BRANCHING_ALGOS and sites is None:
        branch_site_count = 1

    checks: list[dict[str, str]] = []
    blocking: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def record(
        check_id: str,
        label: str,
        status: str,
        message: str,
        *,
        hint: str | None = None,
    ) -> None:
        checks.append(
            {"id": check_id, "label": label, "status": status, "message": message}
        )
        if status == "error":
            issue = {"code": check_id, "message": message}
            if hint:
                issue["hint"] = hint
            blocking.append(issue)
        elif status == "warning":
            warning = {"code": check_id, "message": message}
            if hint:
                warning["hint"] = hint
            warnings.append(warning)

    record(
        "algorithm", "训练算法",
        "pass" if algorithm in VALID_ALGOS else "error",
        algorithm.upper() if algorithm in VALID_ALGOS else f"训练算法必须是 {' / '.join(VALID_ALGOS)}。",
    )
    # RL5: V1-V6 contract scan inlined into preflight (fail-fast layer 2 —
    # layer 1 is the save path, layer 3 the launch-path validators). The
    # scan runs on the *effective* rl dict (post sampling-sync / gpu-select)
    # so it sees exactly what `_launch_training` would serialize.
    # Overlay expansion writes algorithm.advantage next to algo. That pair is
    # the preset, not a second user pipeline, so it is not a V5 conflict.
    from rl.hooks.overlay import _is_expanded_advantage

    scan_rl = rl
    expanded = ((rl.get("algorithm") or {}).get("advantage"))
    if _is_expanded_advantage(expanded):
        scan_rl = deepcopy(rl)
        scan_rl["algorithm"] = {
            key: value for key, value in (scan_rl.get("algorithm") or {}).items() if key != "advantage"
        }
    for violation in _rl_contract_violations(scan_rl):
        record(
            f"rl_contract_{violation['rule'].lower()}",
            "RL 合同校验",
            "error",
            f"{violation['field']}: {violation['rule']} {violation['detail']}",
            hint=str(violation),
        )
    valid_profile = profile in ("fast", "a800", "a800_2gpu")
    record(
        "profile", "运行档位",
        "error" if not valid_profile else "warning" if profile_note else "pass",
        "运行档位必须是 fast / a800 / a800_2gpu。" if not valid_profile else str(profile_note or profile),
    )

    model_path = Path(source.model_path).expanduser() if source.model_path else None
    if source_error:
        record("model", "训练模型", "error", source_error)
    elif model_path and model_path.is_dir() and (model_path / "config.json").is_file():
        record("model", "训练模型", "pass", f"{source.source} · {model_path}")
    else:
        record(
            "model",
            "训练模型",
            "error",
            "训练模型目录或 config.json 不存在。",
            hint=source.model_path or "请绑定训练模型或配置 model_path。",
        )

    data = dict(rl.get("data") or {})
    for key, label in (("train_files", "训练数据"), ("val_files", "验证数据")):
        path = _data_path(data.get(key))
        if path and path.is_file():
            record(key, label, "pass", str(path))
        else:
            record(
                key,
                label,
                "error",
                f"{label}文件不存在。",
                hint=str(path) if path else f"请配置 data.{key}。",
            )
    test_raw = data.get("test_files")
    if test_raw:
        test_path = _data_path(test_raw)
        if test_path and test_path.is_file():
            record("test_files", "测试数据", "pass", str(test_path))
        else:
            record(
                "test_files",
                "测试数据",
                "warning",
                "测试数据文件不存在。",
                hint=str(test_path) if test_path else "请配置 data.test_files。",
            )

    gpu_info = list_gpus()
    gpu_count = int(gpu_info.get("count") or 0)
    invalid_ids = [gpu_id for gpu_id in ids if gpu_id < 0 or gpu_id >= gpu_count]
    if gpu_count == 0:
        record("gpu", "GPU", "error", gpu_info.get("error") or "未检测到 GPU。")
    elif invalid_ids:
        record("gpu", "GPU", "error", f"GPU {invalid_ids} 超出当前机器范围。")
    elif profile == "a800_2gpu" and len(ids) < 2:
        record("gpu", "GPU", "error", "a800_2gpu 需要至少选择两张 GPU。")
    else:
        names = {
            int(item["id"]): str(item["name"])
            for item in gpu_info.get("gpus") or []
        }
        record(
            "gpu",
            "GPU",
            "pass",
            "，".join(f"{gpu_id}: {names.get(gpu_id, 'GPU')}" for gpu_id in ids),
        )

    executable = workflow_executable(bundle.get("workflow") or {})
    trainable_agents: list[str] = []
    if executable.get("ok"):
        _workflow_path()
        from workflow.compiler import trainable_agents as compile_trainable_agents
        from workflow.spec import MASSpec

        spec = MASSpec.model_validate(bundle["workflow"])
        trainable_agents = compile_trainable_agents(spec)
        if trainable_agents:
            record(
                "workflow",
                "Workflow",
                "pass",
                "可训练 Agent：" + "、".join(trainable_agents),
            )
        else:
            record("workflow", "Workflow", "error", "Workflow 没有可训练 Agent。")
        if trainable_agents:
            record(
                "training_scope",
                "训练范围",
                "pass",
                "、".join(trainable_agents),
            )
    else:
        record(
            "workflow",
            "Workflow",
            "error",
            str(executable.get("reason") or "Workflow 不可执行。"),
        )
    if executable.get("ok") and algorithm in BRANCHING_ALGOS:
        from workflow.contracts import BranchSite
        from workflow.site_policy import site_capability
        from workflow.spec import MASSpec

        spec = MASSpec.model_validate(workflow)
        if workflow.get("sampling") and not ((rl.get("algorithm") or {}).get("tir") or {}).get(
            "expand_in_runner", True
        ):
            record(
                "branch_executor", "分支执行方式", "error",
                "当前声明的 Sampling Sites 只能由 expand_in_runner 执行；关闭后旧 Daemon 分支路径不会读取这些站点。",
            )
        if sites is None:
            record("branch_site_legacy", "兼容站点", "pass", "首次 Tool 返回后的通用分支。")
        elif not sites:
            record("branch_sites_disabled", "分支位置", "pass", "没有启用的分支位置；剩余候选独立采样。")
        for raw_site in sites or []:
            site = BranchSite.model_validate(raw_site)
            if site.enabled:
                # Sites belong to the RL algorithm's sampling strategy
                # (ARPO/APPO adapters declare agent-turn opportunities;
                # the workflow's default grpo_n does not), so validate
                # against tir_algo instead of spec.sampling.mode.
                status, message = site_capability(
                    site, spec, strategy=str(algorithm or "") or None
                )
                record(f"branch_site_{site.id}", f"站点 {site.id}", status, message)
        from workflow.runtime import agent_window_rollout

        if agent_window_rollout(spec):
            enabled_sites = [
                BranchSite.model_validate(raw_site)
                for raw_site in (sites or [])
                if isinstance(raw_site, dict) and raw_site.get("enabled", True)
            ]
            known_agents = {agent.id for agent in spec.agents}
            hub_sites = [
                site.id for site in enabled_sites
                if site.anchor.agent_id and site.anchor.agent_id not in known_agents
            ]
            if hub_sites:
                record(
                    "window_branch_anchor", "窗口分支锚点", "error",
                    "窗口图的分支站点必须落在图里的 agent 上，不能指向 hub：" + "、".join(hub_sites),
                )
            planner_site = any(
                site.anchor.kind == "after_agent_turn" and site.anchor.agent_id == "planner"
                for site in enabled_sites
            )
            if algorithm == "arpo" and not planner_site:
                record(
                    "window_planner_site", "Planner 分支", "error",
                    "AgentFlow 窗口上的 ARPO 需要一个 after_agent_turn / planner 站点，才能从该步续写。",
                )
            elif algorithm == "arpo":
                record("window_planner_site", "Planner 分支", "pass", "planner 回合后可续写。")

    _, missing = _dependency_status()
    if missing:
        record(
            "dependencies",
            "训练依赖",
            "error",
            "缺少训练依赖：" + "、".join(missing),
            hint="请在启动 Control 的 Python 环境准备训练依赖。",
        )
    else:
        record("dependencies", "训练依赖", "pass", "训练依赖可发现。")

    active_train = PROCS.active("train")
    if active_train is not None:
        record(
            "process",
            "运行冲突",
            "error",
            f"实验 {active_train.experiment_id} 的训练 {active_train.run_id} 正在运行。",
        )
    else:
        record("process", "运行冲突", "pass", "当前没有活动训练。")
    active_llm = PROCS.active("llm")
    if active_llm is not None:
        record(
            "local_llm",
            "本地模型服务",
            "warning",
            "启动训练时将停止当前本地模型服务以释放 GPU。",
        )

    launch_preview = [
        sys.executable,
        str(tir_agent_root() / "train_tir_agent.py"),
        profile,
        "--algo",
        algorithm,
        "--rl-yaml",
        f"<effective:{exp_id}:rl.yaml>",
        "--workflow-yaml",
        f"<effective:{exp_id}:workflow.yaml>",
        "--model",
        source.model_path or "<missing>",
    ]
    if rl.get("n_runners") is not None:
        launch_preview.extend(["--n-runners", str(int(rl["n_runners"]))])
    if active_agents:
        launch_preview.extend(["--active-agents", ",".join(active_agents)])
    revision_payload = {
        "experiment_id": exp_id,
        "rl": rl,
        "workflow": bundle.get("workflow"),
        "model": source.public(),
        "active_agents": active_agents,
        "gpu_ids": ids,
    }
    revision = hashlib.sha256(
        json.dumps(revision_payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return TrainingPlan(
        experiment_id=exp_id,
        bundle=bundle,
        rl=rl,
        source=source,
        algorithm=algorithm,
        algorithm_source=algorithm_source,
        profile=profile,
        gpu_ids=ids,
        group_n=group_n,
        branch_site_count=branch_site_count,
        trainable_agents=trainable_agents,
        active_agents=active_agents,
        checks=checks,
        blocking_issues=blocking,
        warnings=warnings,
        launch_preview=launch_preview,
        revision=revision,
    )


def training_preflight(exp_id: str) -> dict[str, Any]:
    return build_training_plan(exp_id).public()


def persisted_rl(plan: TrainingPlan) -> dict[str, Any]:
    """Persist normalized training fields without copying a bound resource path."""
    output = deepcopy(plan.rl)
    if plan.source.source != "resource":
        return output
    original = plan.bundle["rl"]
    if "model_path" in original:
        output["model_path"] = original["model_path"]
    else:
        output.pop("model_path", None)
    original_actor = original.get("actor_rollout_ref") or {}
    original_model = original_actor.get("model")
    actor = dict(output.get("actor_rollout_ref") or {})
    if isinstance(original_model, dict):
        actor["model"] = deepcopy(original_model)
    else:
        actor.pop("model", None)
    output["actor_rollout_ref"] = actor
    return output


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _launch_training(
    exp_id: str,
    *,
    request_id: str,
    preflight_revision: str,
    stop_local_llm: bool,
) -> dict[str, Any]:
    from .process_manager import PROCS

    request_id = request_id.strip()
    if not request_id:
        raise TrainingError("invalid_request_id", "训练请求缺少 request_id。")
    existing = PROCS.find_request(
        kind="train", experiment_id=exp_id, request_id=request_id
    )
    if existing:
        return {**existing, "reused": True}

    plan = build_training_plan(exp_id)
    if preflight_revision != plan.revision:
        raise TrainingError(
            "preflight_stale",
            "训练配置已变化，请重新执行启动检查。",
            status=409,
            data={"preflight": plan.public()},
        )
    if not plan.ready:
        conflict = next(
            (
                issue
                for issue in plan.blocking_issues
                if issue["code"] == "process"
            ),
            None,
        )
        active = PROCS.active("train")
        raise TrainingError(
            "training_conflict" if conflict else "preflight_failed",
            conflict["message"]
            if conflict
            else "训练启动检查未通过。",
            status=409 if conflict else 400,
            data={
                "preflight": plan.public(),
                "active_run": (
                    PROCS.status(active.run_id) if active is not None else None
                ),
            },
        )

    active_llm = PROCS.active("llm")
    if active_llm is not None and not stop_local_llm:
        raise TrainingError(
            "local_llm_conflict",
            "本地模型服务正在占用 GPU，请确认停止后再训练。",
            status=409,
            data={"active_run": PROCS.status(active_llm.run_id)},
        )

    run_id = uuid4().hex[:12]
    run_dir = PROCS.run_dir(exp_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    rl_path = run_dir / "effective-rl.yaml"
    workflow_path = run_dir / "effective-workflow.yaml"
    launch_path = run_dir / "launch.json"
    metadata = {
        "request_id": request_id,
        "profile": plan.profile,
        "algo": plan.algorithm,
        "resolved_algorithm": _resolved_algo_summary(
            _resolved_algo_payload(plan.rl), plan.group_n
        ),
        "preflight_revision": plan.revision,
        "training_source": plan.source.public(),
        "cuda_visible_devices": ",".join(str(value) for value in plan.gpu_ids),
        "n_gpus": len(plan.gpu_ids),
        "n_runners": int(plan.rl.get("n_runners") or 1),
        "group_n": plan.group_n,
        "active_agents": plan.active_agents,
        "rollout_tree": {
            "schema_version": 3, "store_mode": "v1",
            "directory": str(run_dir / "rollout-trees"),
        },
        "snapshot": {
            "rl": str(rl_path),
            "workflow": str(workflow_path),
            "launch": str(launch_path),
        },
    }
    PROCS.write_preparing(
        run_id=run_id,
        kind="train",
        experiment_id=exp_id,
        meta=metadata,
    )

    try:
        env = _training_env(exp_id)
        env.update(
            TIR_EXPERIMENT_ID=exp_id,
            TIR_RUN_ID=run_id,
            TIR_ROLLOUT_TREE_DIR=str(run_dir / "rollout-trees"),
            TIR_EXECUTION_TREE_VERSION="3",
        )
        _write_yaml(rl_path, plan.rl)
        _write_yaml(workflow_path, plan.bundle["workflow"])
        train_script = tir_agent_root() / "train_tir_agent.py"
        argv = [
            sys.executable,
            str(train_script),
            plan.profile,
            "--algo",
            plan.algorithm,
            "--rl-yaml",
            str(rl_path),
            "--workflow-yaml",
            str(workflow_path),
            "--model",
            plan.source.model_path,
        ]
        if plan.rl.get("n_runners") is not None:
            argv.extend(
                ["--n-runners", str(int(plan.rl["n_runners"]))]
            )
        if plan.active_agents:
            argv.extend(["--active-agents", ",".join(plan.active_agents)])
        launch = {
            "schema_version": 1,
            "run_id": run_id,
            "request_id": request_id,
            "experiment_id": exp_id,
            "git_commit": _git_commit(),
            "created_at": time.time(),
            "preflight_revision": plan.revision,
            "algorithm": plan.algorithm,
            "algorithm_source": plan.algorithm_source,
            "resolved_algo": _resolved_algo_payload(plan.rl),
            "profile": plan.profile,
            "gpu_ids": plan.gpu_ids,
            "model": plan.source.public(),
            "argv": argv,
            "snapshot": metadata["snapshot"],
        }
        _write_json(launch_path, launch)

        if active_llm is not None:
            PROCS.stop_run(
                active_llm.run_id,
                experiment_id=active_llm.experiment_id,
                reason="stopped_for_training",
            )

        root = str(tir_agent_root().parent)
        env["PYTHONPATH"] = os.pathsep.join(
            [root, str(tir_agent_root()), env.get("PYTHONPATH", "")]
        )
        env["CUDA_VISIBLE_DEVICES"] = metadata[
            "cuda_visible_devices"
        ]
        env["VLLM_USE_V1"] = env.get("VLLM_USE_V1") or "1"
        process = PROCS.start(
            kind="train",
            experiment_id=exp_id,
            run_id=run_id,
            argv=argv,
            cwd=tir_agent_root(),
            env=env,
            meta=metadata,
            replace=False,
        )
        return {
            **PROCS.status(process.run_id),
            "argv": argv,
            "preflight_revision": plan.revision,
            "reused": False,
        }
    except TrainingError:
        raise
    except Exception as error:
        current = PROCS.status(run_id)
        stage = (
            str(current.get("failure_stage"))
            if current and current.get("failure_stage")
            else "prepare"
        )
        PROCS.mark_failed(
            run_id=run_id,
            experiment_id=exp_id,
            kind="train",
            stage=stage,
            message=str(error),
            meta=metadata,
        )
        raise TrainingError(
            "process_start_failed",
            f"训练进程启动失败：{error}",
            data={"run_id": run_id},
        ) from error


def launch_training(
    exp_id: str,
    *,
    request_id: str,
    preflight_revision: str,
    stop_local_llm: bool,
) -> dict[str, Any]:
    with _LAUNCH_LOCK:
        return _launch_training(
            exp_id,
            request_id=request_id,
            preflight_revision=preflight_revision,
            stop_local_llm=stop_local_llm,
        )


def stop_training_run(
    exp_id: str, run_id: str
) -> dict[str, Any]:
    from .process_manager import PROCS, TERMINAL_STATES

    row = PROCS.status(run_id, exp_id)
    if row is None:
        raise TrainingError("run_not_found", "训练运行不存在。", status=404)
    if row.get("experiment_id") != exp_id or row.get("kind") != "train":
        raise TrainingError(
            "run_experiment_mismatch",
            "训练运行不属于当前实验。",
            status=409,
        )
    if row.get("state") in TERMINAL_STATES:
        return row
    try:
        stopped = PROCS.stop_run(
            run_id,
            experiment_id=exp_id,
            reason="user_requested",
        )
    except RuntimeError as error:
        raise TrainingError(
            "stop_cleanup_failed", str(error), status=500, data={"run_id": run_id}
        ) from error
    if stopped is None:
        raise TrainingError(
            "run_not_stoppable",
            "训练运行无法验证为当前 Control 管理的活动进程，不能停止。",
            status=409,
        )
    return stopped
