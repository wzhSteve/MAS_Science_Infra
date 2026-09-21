"""Experiment YAML bundle: load / save / validate / templates."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from science_infra.control.paths import experiments_root, repo_data_dir, tir_agent_root

VALID_ALGOS = ("grpo", "arpo", "aepo", "igpo", "gigpo", "rae")
HARNESS_PLUGINS = (
    "log_error",
    "loss_volatility",
    "cognitive_convergence",
    "reward_hacking",
    "epc_aw_consensus",
)
STUB_HARNESS = frozenset({"epc_aw_consensus"})


class ExperimentMeta(BaseModel):
    id: str
    seed: int = 42
    refs: Dict[str, str] = Field(
        default_factory=lambda: {
            "llm": "llm.yaml",
            "workflow": "workflow.yaml",
            "rl": "rl.yaml",
            "harness": "harness.yaml",
        }
    )
    pipeline: List[str] = Field(default_factory=lambda: ["collect", "diagnose", "train"])
    name: str = ""
    agl_metrics_url: str = "/agl/metrics"

    model_config = {"extra": "allow"}


def _safe_id(exp_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", exp_id):
        raise ValueError(f"invalid experiment id: {exp_id!r}")
    return exp_id


def exp_dir(exp_id: str) -> Path:
    return experiments_root() / _safe_id(exp_id)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return raw


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def default_llm() -> Dict[str, Any]:
    return {
        "kind": "api",
        "model": "",
        "base_url": "",
        "port": 8000,
        "model_path": "/root/autodl-tmp/LLM/Qwen3-4B",
        "gpu_memory_utilization": 0.45,
        "api_key_set": False,
    }


def default_workflow() -> Dict[str, Any]:
    hub_path = tir_agent_root() / "specs" / "hub_react.yaml"
    if hub_path.is_file():
        data = _read_yaml(hub_path)
        data.setdefault("schema_version", "0.1.0")
        data.setdefault("agents", [])
        data.setdefault("edges", [])
        data.setdefault("entry_agent", "hub")
        return data
    return {
        "schema_version": "0.1.0",
        "topology": "hub_react",
        "hub": {"role": "orchestrator", "skills": ["react_loop"]},
        "tools": ["web_search", "wikipedia_search", "execute_python"],
        "llm": {"kind": "api", "model": "", "base_url": ""},
        "memory": {"agent": "messages", "system": "none"},
        "archive": {"window": "post_first_tool"},
        "agents": [],
        "edges": [],
        "entry_agent": "hub",
    }


def default_rl() -> Dict[str, Any]:
    return {
        "profile": "fast",
        "algo": "grpo",
        "n_runners": 1,
        "rollout_per_gpu": 2,
        "devices": {"ids": [0]},
        "model_path": "/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B",
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "tir_algo": "grpo",
        },
        "data": {
            "train_files": str(repo_data_dir() / "train.parquet"),
            "val_files": str(repo_data_dir() / "val.parquet"),
            "train_batch_size": 2,
            "max_prompt_length": 2048,
            "max_response_length": 512,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 2,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.35,
            },
            "actor": {
                "ppo_mini_batch_size": 2,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": 1},
            "model": {
                "path": "/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "tensorboard"],
            "project_name": "AgentLightning",
            "experiment_name": "tir_agent",
            "nnodes": 1,
            "test_freq": 1,
            "total_epochs": 1,
            "total_training_steps": 1,
        },
    }


def normalize_rl_data_paths(rl: Dict[str, Any]) -> Dict[str, Any]:
    """Point relative ``data/*.parquet`` paths at ``MAS_Science_Infra/data``."""
    out = deepcopy(rl)
    data = out.get("data")
    if not isinstance(data, dict):
        return out
    root = repo_data_dir()
    for key, default_name in (("train_files", "train.parquet"), ("val_files", "val.parquet")):
        raw = data.get(key)
        if raw is None or raw == "":
            data[key] = str(root / default_name)
            continue
        p = Path(str(raw)).expanduser()
        if p.is_file():
            data[key] = str(p.resolve())
            continue
        # Relative data/... or bare filename → repo data dir
        name = p.name if p.name.endswith(".parquet") else default_name
        cand = root / name
        if cand.is_file() or not p.is_absolute():
            data[key] = str(cand)
    return out


def default_harness() -> Dict[str, Any]:
    return {
        "plugins": [
            "log_error",
            "loss_volatility",
            "cognitive_convergence",
            "reward_hacking",
        ],
    }


def ensure_experiment(exp_id: str, *, seed: int = 42, name: str = "") -> Path:
    root = exp_dir(exp_id)
    (root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    meta_path = root / "experiment.yaml"
    if not meta_path.is_file():
        meta = ExperimentMeta(id=exp_id, seed=seed, name=name or exp_id)
        _write_yaml(meta_path, {"experiment": meta.model_dump()})
    if not (root / "llm.yaml").is_file():
        _write_yaml(root / "llm.yaml", default_llm())
    if not (root / "workflow.yaml").is_file():
        _write_yaml(root / "workflow.yaml", default_workflow())
    if not (root / "rl.yaml").is_file():
        _write_yaml(root / "rl.yaml", default_rl())
    if not (root / "harness.yaml").is_file():
        _write_yaml(root / "harness.yaml", default_harness())
    return root


def create_experiment(exp_id: str, *, seed: int = 42, name: str = "") -> Path:
    root = exp_dir(exp_id)
    if (root / "experiment.yaml").is_file():
        raise ValueError(f"experiment already exists: {exp_id}")
    return ensure_experiment(exp_id, seed=seed, name=name)


def require_experiment(exp_id: str) -> Path:
    root = exp_dir(exp_id)
    if not (root / "experiment.yaml").is_file():
        raise FileNotFoundError(f"experiment not found: {exp_id}")
    return root


def list_experiments() -> List[str]:
    root = experiments_root()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "experiment.yaml").is_file():
            out.append(p.name)
    return out


def load_llm_section(exp_id: str) -> Dict[str, Any]:
    root = require_experiment(exp_id)
    raw = _read_yaml(root / "experiment.yaml")
    meta = ExperimentMeta.model_validate(raw.get("experiment") or raw)
    return _read_yaml(root / meta.refs.get("llm", "llm.yaml"))


def load_bundle(exp_id: str) -> Dict[str, Any]:
    root = require_experiment(exp_id)
    meta_raw = _read_yaml(root / "experiment.yaml")
    exp = meta_raw.get("experiment") or meta_raw
    meta = ExperimentMeta.model_validate(exp)
    llm = _read_yaml(root / meta.refs.get("llm", "llm.yaml"))
    workflow = _read_yaml(root / meta.refs.get("workflow", "workflow.yaml"))
    rl = _read_yaml(root / meta.refs.get("rl", "rl.yaml"))
    harness = _read_yaml(root / meta.refs.get("harness", "harness.yaml"))
    from .llm_config import (
        public_endpoint,
        resolve_legacy_llm_config,
        resource_llm_config,
    )
    from .model_resources import binding_context

    bindings, resource = binding_context(exp_id)
    binding_error = bindings["inference"].get("error")
    if binding_error:
        model_summary = {
            "kind": "api",
            "model": "",
            "base_url": "",
            "api_key_set": False,
            "credential_source": "none",
            "config_revision": "",
            "resource_id": bindings["inference"]["resource_id"],
            "resource_revision": None,
            "resource_name": None,
            "binding_error": binding_error,
        }
    else:
        model_summary = (
            resource_llm_config(resource)
            if resource
            else resolve_legacy_llm_config(exp_id, llm=llm)
        ).public()
    llm = dict(llm)
    llm.pop("api_key", None)
    llm["base_url"] = public_endpoint(str(llm.get("base_url") or ""))
    if resource:
        llm = dict(resource.data["config"])
    llm.update(model_summary)
    if isinstance(workflow.get("llm"), dict):
        workflow["llm"] = dict(workflow["llm"])
        workflow["llm"].pop("api_key", None)
        workflow["llm"]["base_url"] = public_endpoint(
            str(workflow["llm"].get("base_url") or "")
        )
    return {
        "id": meta.id,
        "meta": meta.model_dump(),
        "llm": llm,
        "workflow": workflow,
        "rl": rl,
        "harness": harness,
        "path": str(root),
        "executable": workflow_executable(workflow),
        "model_bindings": bindings,
    }


def save_section(exp_id: str, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
    root = require_experiment(exp_id)
    meta_raw = _read_yaml(root / "experiment.yaml")
    exp = meta_raw.get("experiment") or meta_raw
    meta = ExperimentMeta.model_validate(exp)
    section = section.lower().strip()
    if section == "experiment" or section == "meta":
        merged = {**meta.model_dump(), **{k: v for k, v in data.items() if k != "id"}}
        merged["id"] = exp_id
        meta = ExperimentMeta.model_validate(merged)
        if isinstance(meta_raw.get("experiment"), dict):
            document = deepcopy(meta_raw)
            document["experiment"] = meta.model_dump()
        else:
            document = meta.model_dump()
        _write_yaml(root / "experiment.yaml", document)
    elif section == "llm":
        from .model_resources import ResourceError, resolve_binding

        if resolve_binding(exp_id, "inference") is not None:
            raise ResourceError(
                "当前实验已绑定模型资源，请编辑资源或先解除绑定。",
                409,
            )
        clean = dict(data)
        api_key = clean.pop("api_key", None)
        for field in (
            "api_key_set",
            "credential_source",
            "config_revision",
            "resource_id",
            "resource_revision",
            "resource_name",
            "binding_error",
        ):
            clean.pop(field, None)
        _write_yaml(root / meta.refs.get("llm", "llm.yaml"), clean)
        if isinstance(api_key, str) and api_key.strip() and not api_key.startswith("••"):
            _write_secret(root, "OPENAI_API_KEY", api_key.strip())
        # Mirror into workflow.llm when present
        wf_path = root / meta.refs.get("workflow", "workflow.yaml")
        wf = _read_yaml(wf_path)
        wf["llm"] = {
            "kind": clean.get("kind", "api"),
            "model": clean.get("model", ""),
            "base_url": clean.get("base_url") or "",
        }
        _write_yaml(wf_path, wf)
    elif section == "workflow":
        current_llm = _read_yaml(root / meta.refs.get("llm", "llm.yaml"))
        workflow = dict(data)
        workflow["llm"] = {
            "kind": current_llm.get("kind", "api"),
            "model": current_llm.get("model", ""),
            "base_url": current_llm.get("base_url") or "",
        }
        _write_yaml(root / meta.refs.get("workflow", "workflow.yaml"), workflow)
    elif section == "rl":
        data = normalize_rl_data_paths(data)
        _write_yaml(root / meta.refs.get("rl", "rl.yaml"), data)
    elif section == "harness":
        _write_yaml(root / meta.refs.get("harness", "harness.yaml"), data)
    else:
        raise ValueError(f"unknown section: {section}")
    return load_bundle(exp_id)


def _write_secret(root: Path, key: str, value: str) -> None:
    path = root / ".secrets.env"
    existing: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip().strip('"')
    existing[key] = value
    lines = [f'{k}="{v}"' for k, v in existing.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


def load_secrets_env(exp_id: str) -> Dict[str, str]:
    path = exp_dir(exp_id) / ".secrets.env"
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"')
    return out


def artifacts_dir(exp_id: str) -> Path:
    return ensure_experiment(exp_id) / "artifacts"


def collect_path(exp_id: str) -> Path:
    return artifacts_dir(exp_id) / "collect.json"


def diagnose_path(exp_id: str) -> Path:
    return artifacts_dir(exp_id) / "diagnose.json"


def workflow_executable(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Whether current runtime can run this workflow (via Topology Compiler)."""
    try:
        import sys

        tir = str(tir_agent_root())
        if tir not in sys.path:
            sys.path.insert(0, tir)
        from workflow.spec import MASSpec

        spec = MASSpec.model_validate(workflow)
        ok, reason = spec.is_executable()
        return {"ok": ok, "reason": reason, "topology": spec.topology, "entry_agent": spec.entry_agent}
    except Exception as e:
        topology = str(workflow.get("topology") or "hub_react")
        return {"ok": False, "reason": str(e), "topology": topology}


def apply_gpu_selection(rl: Dict[str, Any], ids: List[int]) -> Dict[str, Any]:
    """Normalize devices.ids, n_gpus_per_node, and 1-GPU profile downgrade."""
    out = deepcopy(rl)
    ids = [int(x) for x in ids]
    if not ids:
        ids = [0]
    devices = dict(out.get("devices") or {})
    devices["ids"] = ids
    out["devices"] = devices
    n = len(ids)
    trainer = dict(out.get("trainer") or {})
    trainer["n_gpus_per_node"] = n
    out["trainer"] = trainer
    if n < 2 and str(out.get("profile") or "") == "a800_2gpu":
        out["profile"] = "fast"
        out["_profile_downgraded"] = "a800_2gpu→fast (need 2 GPUs)"
    rpg = out.get("rollout_per_gpu")
    if rpg is not None:
        arr = dict(out.get("actor_rollout_ref") or {})
        rollout = dict(arr.get("rollout") or {})
        rollout["n"] = int(rpg)
        arr["rollout"] = rollout
        out["actor_rollout_ref"] = arr
    return out


def recommend_rl(gpu_count: int) -> Dict[str, Any]:
    n = max(1, int(gpu_count or 1))
    if n <= 1:
        return {
            "profile": "fast",
            "n_runners": 1,
            "rollout_per_gpu": 2,
            "devices": {"ids": [0]},
            "actor_rollout_ref.rollout.n": 2,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
            "trainer.n_gpus_per_node": 1,
        }
    ids = list(range(min(n, 2)))
    return {
        "profile": "a800_2gpu" if n >= 2 else "a800",
        "n_runners": 4 if n >= 2 else 2,
        "rollout_per_gpu": 4,
        "devices": {"ids": ids},
        "trainer.n_gpus_per_node": len(ids),
    }


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overlay.items():
        if k in ("profile", "algo", "n_runners", "model_path"):
            out[k] = v
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
