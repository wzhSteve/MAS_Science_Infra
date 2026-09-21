"""Effective training configuration, local model discovery and preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_infra.env import repo_root

from .experiments import (
    apply_gpu_selection,
    load_bundle,
    normalize_rl_data_paths,
    workflow_executable,
)
from .model_resources import ResourceSnapshot, list_resources, resolve_binding
from .paths import tir_agent_root


@dataclass(frozen=True)
class TrainingSource:
    source: str
    model_path: str
    resource_id: str | None = None
    resource_revision: int | None = None
    resource_name: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_path": self.model_path,
            "resource_id": self.resource_id,
            "resource_revision": self.resource_revision,
            "resource_name": self.resource_name,
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


def _model_path(rl: dict[str, Any]) -> str:
    return str(
        rl.get("model_path")
        or (rl.get("actor_rollout_ref") or {}).get("model", {}).get("path")
        or ""
    ).strip()


def resolve_training_source(
    exp_id: str, rl: dict[str, Any] | None = None
) -> TrainingSource:
    resource = resolve_binding(exp_id, "training")
    if resource is not None:
        path = str(resource.data["config"]["model_path"]).strip()
        return TrainingSource(
            source="resource",
            model_path=path,
            resource_id=resource.data["id"],
            resource_revision=resource.data["revision"],
            resource_name=resource.data["name"],
        )
    config = rl if rl is not None else load_bundle(exp_id)["rl"]
    return TrainingSource(source="legacy", model_path=_model_path(config))


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
        raise ValueError("本地模型不在允许的发现目录中。")
    return candidate


def _apply_sampling(
    rl: dict[str, Any], workflow: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    sampling = workflow.get("sampling") if isinstance(workflow, dict) else None
    if not sampling:
        return rl, "rl"
    _workflow_path()
    from rl.hooks.overlay import apply_sample_policy

    return apply_sample_policy(rl, sampling), "workflow.sampling"


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
    from .services import list_gpus

    bundle = load_bundle(exp_id)
    raw_rl = deepcopy(bundle["rl"])
    source = resolve_training_source(exp_id, raw_rl)
    ids = [int(value) for value in (raw_rl.get("devices") or {}).get("ids") or [0]]
    rl = normalize_rl_data_paths(apply_gpu_selection(raw_rl, ids))
    rl, algorithm_source = _apply_sampling(rl, bundle.get("workflow") or {})
    rl["model_path"] = source.model_path
    actor_rollout_ref = dict(rl.get("actor_rollout_ref") or {})
    model = dict(actor_rollout_ref.get("model") or {})
    model["path"] = source.model_path
    actor_rollout_ref["model"] = model
    rl["actor_rollout_ref"] = actor_rollout_ref

    algorithm = str(
        (rl.get("algorithm") or {}).get("tir_algo") or rl.get("algo") or "grpo"
    ).lower()
    profile = str(rl.get("profile") or "fast")
    rollout = actor_rollout_ref.get("rollout") or {}
    group_n = int(rollout.get("n") or rl.get("rollout_per_gpu") or 1)
    sites = (
        ((rl.get("algorithm") or {}).get("tir") or {}).get("sites")
        or ((bundle.get("workflow") or {}).get("sampling") or {}).get("sites")
        or []
    )
    branch_site_count = sum(
        1 for site in sites if isinstance(site, dict) and site.get("enabled", True)
    )

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

    model_path = Path(source.model_path).expanduser() if source.model_path else None
    if model_path and model_path.is_dir() and (model_path / "config.json").is_file():
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
    else:
        record(
            "workflow",
            "Workflow",
            "error",
            str(executable.get("reason") or "Workflow 不可执行。"),
        )

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
    if active_train is not None and active_train.popen.poll() is None:
        record(
            "process",
            "运行冲突",
            "error",
            f"实验 {active_train.experiment_id} 的训练 {active_train.run_id} 正在运行。",
        )
    else:
        record("process", "运行冲突", "pass", "当前没有活动训练。")
    active_llm = PROCS.active("llm")
    if active_llm is not None and active_llm.popen.poll() is None:
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
        "--model",
        source.model_path or "<missing>",
    ]
    if rl.get("n_runners") is not None:
        launch_preview.extend(["--n-runners", str(int(rl["n_runners"]))])
    revision_payload = {
        "experiment_id": exp_id,
        "rl": rl,
        "workflow": bundle.get("workflow"),
        "model": source.public(),
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
