"""Ablation / product feature flags for EPC_AW dual-track validation.

Paper track (ICLR main table):
  full | no_intervention | no_capability | no_invocation | no_memory

Product track (layer kill-switches + telemetry; not paper main ablation):
  product_no_l2a | product_no_l2b | product_no_l3
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Union


PAPER_PRESETS = frozenset(
    {
        "full",
        "no_intervention",
        "no_capability",
        "no_invocation",
        "no_memory",
    }
)

PRODUCT_PRESETS = frozenset(
    {
        "product_no_l2a",
        "product_no_l2b",
        "product_no_l3",
        "no_verified_memory",
        "no_struct_control",
    }
)


@dataclass
class AblationConfig:
    """Single source of truth for paper ablations and product layer switches."""

    # Paper Track
    enable_intervention: bool = True
    read_capability_memory: bool = True
    read_invocation_memory: bool = True
    enable_evolve: bool = True
    offline_memory_dir: Optional[str] = None

    # Product Track (defaults all on; not used in paper main table)
    enable_l2a: bool = True
    enable_l2b: bool = True
    enable_l3: bool = True
    emit_layer_telemetry: bool = True

    # Task-local failed-param blacklist stays on even for no_invocation
    # (offline factors off; in-task contrast still helps product). For
    # no_memory paper preset we also disable blacklist to avoid leakage.
    use_param_blacklist: bool = True

    # StructAgent-inspired verified memory controls (product track).
    # Defaults on: sole commit via commit_progress; compact/evolve prefer ledger.
    enable_verified_memory_commit: bool = True
    enable_verified_only_evolve: bool = True
    enable_compact_task_view: bool = True

    # StructAgent-inspired control-plane: STOP / intervention / final audit.
    enable_struct_stop_gate: bool = True
    enable_struct_intervention_routing: bool = True
    enable_final_audit: bool = True

    preset: str = "full"

    def should_load_offline(self, evaluation_mode: bool) -> bool:
        """Load offline JSON when reading Cap/Inv, or when a snapshot dir is set."""
        if not (self.read_capability_memory or self.read_invocation_memory):
            return False
        if self.offline_memory_dir:
            return True
        return not evaluation_mode

    def memory_dir(self) -> str:
        return self.offline_memory_dir or "memory"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Preset → field overrides (unlisted fields keep AblationConfig defaults).
PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_intervention": {
        "enable_intervention": False,
    },
    "no_capability": {
        "read_capability_memory": False,
    },
    "no_invocation": {
        "read_invocation_memory": False,
        # Keep task-local blacklist (plan default for NoInv).
        "use_param_blacklist": True,
    },
    "no_memory": {
        "read_capability_memory": False,
        "read_invocation_memory": False,
        "enable_evolve": False,
        "use_param_blacklist": False,
    },
    "product_no_l2a": {
        "enable_l2a": False,
    },
    "product_no_l2b": {
        "enable_l2b": False,
    },
    "product_no_l3": {
        "enable_l3": False,
    },
    "no_verified_memory": {
        "enable_verified_memory_commit": False,
        "enable_verified_only_evolve": False,
        "enable_compact_task_view": False,
    },
    "no_struct_control": {
        "enable_struct_stop_gate": False,
        "enable_struct_intervention_routing": False,
        "enable_final_audit": False,
    },
}


def resolve_ablation(
    ablation: Optional[Union[str, AblationConfig, Mapping[str, Any]]] = None,
    **overrides: Any,
) -> AblationConfig:
    """Build AblationConfig from preset name, mapping, instance, or kwargs."""
    if ablation is None:
        cfg = AblationConfig()
    elif isinstance(ablation, AblationConfig):
        cfg = AblationConfig(**asdict(ablation))
    elif isinstance(ablation, str):
        key = ablation.strip().lower().replace("-", "_")
        if key not in PRESETS:
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown ablation preset '{ablation}'. Known: {known}")
        cfg = AblationConfig(preset=key, **PRESETS[key])
    elif isinstance(ablation, Mapping):
        data = dict(ablation)
        preset = str(data.pop("preset", "full")).strip().lower().replace("-", "_")
        merged = {**PRESETS.get(preset, {}), **data, "preset": preset}
        allowed = {f.name for f in fields(AblationConfig)}
        cfg = AblationConfig(**{k: v for k, v in merged.items() if k in allowed})
    else:
        raise TypeError(f"Unsupported ablation type: {type(ablation)!r}")

    allowed = {f.name for f in fields(AblationConfig)}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Unknown AblationConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def list_presets() -> Dict[str, list]:
    return {
        "paper": sorted(PAPER_PRESETS),
        "product": sorted(PRODUCT_PRESETS),
    }
