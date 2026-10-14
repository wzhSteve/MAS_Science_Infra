"""Per-algorithm Hydra/dict overlays. VERL adv_estimator stays 'grpo'."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from rl.train_signal import TrainSignal

VALID_ALGOS = ("grpo", "arpo", "appo", "aepo", "igpo", "gigpo", "rae", "flow_grpo")
# Algorithms that fork extra rollouts. APPO shares ARPO's expansion path; credit differs later.
BRANCHING_ALGOS = ("arpo", "appo", "aepo", "rae")

DEFAULT_TIR: Dict[str, Any] = {
    "initial_rollouts": 2,
    "beam_size": 2,
    "branch_probability": 0.5,
    "entropy_weight": 0.5,
    "entropy_threshold": 0.15,
    "entropy_tokens": 8,
    "gamma": 1.0,
    "info_gain_type": "log_prob_diff",
    "step_advantage_w": 1.0,
    "gigpo_mode": "mean_std",
    "entropy_adv_coef": 0.1,
    "aepo_beta": 1.0,
    "branch_penalty_slope": 0.25,
    "multi_tool_bonus": 0.1,
    "max_branch_depth": 2,
    "expand_in_runner": True,
    "use_official_arpo_gate": True,
    "sites": None,
    "zero_prefix_adv_for_children": True,
    "ready_batch": False,
    "rae_p_plus": 0.8,
    "rae_k_min": 2,
    "rae_epsilon_f": 1e-3,
    "rae_full_tgt": False,
    "rae_dead_end_backprop": True,
    "rae_dead_end_depth": 1,
}


def apply_sample_policy(config: Dict[str, Any], sampling: Any) -> Dict[str, Any]:
    """Map MAS SamplePolicy onto algorithm.tir / rollout.n (no AGL import)."""
    if sampling is None:
        return config
    cfg = deepcopy(config)
    if hasattr(sampling, "mode"):
        mode = str(getattr(sampling, "mode", None) or "grpo_n")
        group_n = getattr(sampling, "group_n", None)
        beam = getattr(sampling, "beam_size", None)
        init = getattr(sampling, "initial_rollouts", None)
        max_depth = getattr(sampling, "max_branch_depth", None)
        expand = getattr(sampling, "expand_in_runner", None)
    elif isinstance(sampling, dict):
        mode = str(sampling.get("mode") or "grpo_n")
        group_n = sampling.get("group_n")
        beam = sampling.get("beam_size")
        init = sampling.get("initial_rollouts")
        max_depth = sampling.get("max_branch_depth")
        expand = sampling.get("expand_in_runner")
    else:
        return config
    from workflow.contracts import SamplePolicy

    policy = sampling if isinstance(sampling, SamplePolicy) else SamplePolicy.model_validate(sampling)
    mode = mode.lower().strip()
    algo_map = {
        "grpo_n": "grpo",
        "grpo": "grpo",
        "arpo": "arpo",
        "aepo": "aepo",
        "appo": "appo",
        "rae": "rae",
    }
    algo = algo_map.get(mode, mode if mode in VALID_ALGOS else "grpo")
    cfg = apply_algo_overlay(cfg, algo)
    from workflow.sampling.registry import sampling_adapters

    adapter = sampling_adapters.resolve(algo)
    cfg = adapter.training_overlay(cfg, policy)
    tir = dict(cfg["algorithm"].get("tir") or {})
    if group_n is not None:
        n = max(1, int(group_n))
        roll = dict(cfg.get("actor_rollout_ref") or {})
        r = dict(roll.get("rollout") or {})
        r["n"] = n
        roll["rollout"] = r
        cfg["actor_rollout_ref"] = roll
        cfg["rollout_per_gpu"] = n
    if beam is not None:
        tir["beam_size"] = max(1, int(beam))
    if init is not None:
        tir["initial_rollouts"] = max(1, int(init))
    if max_depth is not None:
        tir["max_branch_depth"] = max(1, int(max_depth))
    if expand is not None:
        tir["expand_in_runner"] = bool(expand)
    tir["sites"] = [site.model_dump(mode="json") for site in policy.resolved_sites()]
    cfg["algorithm"]["tir"] = tir
    return cfg


def _user_advantage_section(algo_block: Dict[str, Any]) -> Dict[str, Any] | None:
    """Read a user-authored v2 ``algorithm.advantage`` section, if any.

    Only a non-empty dict counts as a v2 composed declaration (the empty
    dict is what pass-2 of apply_train_signal re-applying would see right
    after pass-1 wrote the *expanded* preset — see _is_expanded_advantage;
    during that window user_tir alone must not trip V5).
    """
    section = algo_block.get("advantage")
    if isinstance(section, dict) and section:
        return section
    return None


def _is_expanded_advantage(section: Any) -> bool:
    """Fingerprint of an expand_preset-product section (mirror of itself).

    apply_train_signal re-applies the overlay twice around the user-yaml
    deep-merge (train_tir_agent.py:370-374): pass-1 writes the expanded
    ``algorithm.advantage``; the user yaml then deep-merges on top. A
    pre-existing advantage section that exactly equals one of the eight
    expanded presets is our own write, not a composed user override —
    recognized so that V5 does not fire on idempotent re-application and
    applications of the saved effective yaml.
    """
    if not isinstance(section, dict):
        return False
    for algo in VALID_ALGOS:
        preset = _expand_preset(algo)
        if section == _advantage_section_from_spec(preset):
            return True
    return False


def _expand_preset(algo: str):
    """Formal v1 expansion hook — re-export of rl.loss.expand_preset.

    Validation contract: the expanded snapshot passes V1/V2/V3 by
    construction (registry members, no duplicates, snapshot royalty);
    the snapshot can never produce an illegal stack, so the expand step
    itself needs no further checks (exactly the property RL_design §5.2
    grants the v1 leg). The V5 dual-track check below is what actually
    gates at the expansion exit.
    """
    from rl.loss import expand_preset

    return expand_preset(algo)


def _advantage_section_from_spec(spec) -> Dict[str, Any]:
    """Serialize an AdvantagePipelineSpec into the ``algorithm.advantage`` v2 section."""
    return {
        "base": spec.base,
        "operators": [
            {"id": op.id, "params": dict(op.params)} for op in spec.operators
        ],
    }


def _validate_algo_overlay_v_track(
    *,
    algo: str,
    preset,
    user_algo: Dict[str, Any],
    user_tir: Dict[str, Any],
) -> list[dict[str, str]]:
    """V-track checks at the expansion exit (save-layer core, RL5).

    V5 (dual-track hedging): a user-composed ``algorithm.advantage`` section
    and the v1 single-select are mutually exclusive declarations of the
    advantage pipeline. Racing priorities mid-flight would produce silent
    config drift — the design demands the conflict be *rejected*, and the
    rewrite below keeps ``tir_algo`` (sampling/gate semantics stay v1, A9
    permanent) while making the composed section the advantage truth in
    full. Re-applying this overlay is idempotent (expanded presets are
    recognized as ours, not user compositions).
    User-``tir`` operator-relevant keys (loss_mode / reward_scale_discount /
    zero_prefix_adv_for_children) are NOT blocked: RL_design §3.4.2 lists
    "v2 与 v1 同键不同值" pairs, but those keys are *loss/sampling-leg* keys
    in this codebase's split (tir carries sampling+loss, advantage 匹配
    carries pipeline) — cross-blocking them would reject every existing v1
    yaml during its own expansion, double-punishing the legacy track. The
    actual operator-relevanttir key set is empty today; recorded here so
    the rule can be tightened when D3 fully migrates the tir dict (P2-phase
    concern, out of RL5 scope).
    """
    user_section = _user_advantage_section(user_algo)
    if user_section is None:
        violations: list[dict[str, str]] = []
    elif _is_expanded_advantage(user_section):
        # Idempotent re-apply of our own expanded write (apply_train_signal
        # pass-2 / effective-rl.yaml reload): not a user composition.
        violations = []
    else:
        # Composed declaration wins the advantage segment; the preset
        # expansion result just written is demoted to sampling/gate-only.
        # Emitted diagnostics tell the maintainer which leg carried what.
        violations = [
            {
                "rule": "V5",
                "field": "algorithm.advantage",
                "detail": (
                    f"v2 composed 段与 v1 algo={algo!r} 同时存在: advantage 真值取"
                    " composed 段, tir_algo 保留承载 sampling/门控语义 (A9 v1 永久保留)。"
                    " preset 展开结果不参与 resolution"
                ),
            }
        ]
    return violations


def apply_algo_overlay(config: Dict[str, Any], algo: str) -> Dict[str, Any]:
    """v1/v2 双轨 overlay（RL5 重构）: 「展开 preset → deep-merge 用户覆写」两步。

    v1（algo 单选，永久保留——A9）：``algo`` 合法则查 expand_preset 快照表
    展开为 AdvantagePipelineSpec，经 V1-V5 校验后写入 ``algorithm.advantage``
    v2 段（展开结果即 v2 形态——v1→v2 单向无损，RL_design §3.4.2），
    再按既有 ``tir.update(user_tir)`` 语义合并用户覆写。
    v2（composed 声明）：yaml 携带 ``algorithm.advantage`` 段时该段为
    advantage 真值（校验后透传），algo 降级为采样/门控键的载体语义；
    V5 双轨互斥由 ``_validate_algo_overlay_v_track`` 前置裁决。
    """
    algo = algo.lower().strip()
    user_algo = dict(config.get("algorithm") or {})
    user_tir = dict(user_algo.get("tir") or {})
    user_advantage = _user_advantage_section(user_algo)
    if user_advantage is None and algo not in VALID_ALGOS:
        raise ValueError(f"Unknown --algo {algo!r}. Expected one of {VALID_ALGOS}")
    cfg = deepcopy(config)
    algo_block = dict(cfg.get("algorithm") or {})
    algo_block["adv_estimator"] = "grpo"
    # Respect an explicit use_kl_in_reward from the rl-yaml; only default to
    # False when the key is absent (e.g. plain Hydra configs without a yaml).
    if "use_kl_in_reward" not in algo_block:
        algo_block["use_kl_in_reward"] = False
    algo_block["tir_algo"] = algo
    tir = dict(DEFAULT_TIR)
    # RL5 two-step expansion — the pre-existing ``tir.update(user_tir)`` is
    # the same deep-merge shape, generalized per RL_design §3.4.2: expand the
    # preset first, THEN merge user overrides on top so user keys always win.
    expand_violations: list[dict[str, str]] = []
    if algo in VALID_ALGOS:
        preset = _expand_preset(algo)
        expand_violations = _validate_algo_overlay_v_track(
            algo=algo, preset=preset, user_algo=algo_block, user_tir=user_tir,
        )
        if user_advantage is not None and not _is_expanded_advantage(user_advantage):
            # v2 composed declaration is the advantage truth (validated in
            # place by the preflight layer; keep user key order intact) —
            # do NOT overwrite it with the preset expansion product.
            algo_block["advantage"] = user_advantage
        else:
            algo_block["advantage"] = _advantage_section_from_spec(preset)
    tir.update(user_tir)
    if algo in ("arpo", "appo", "aepo", "rae"):
        n = int(cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 4))
        tir["initial_rollouts"] = max(1, min(tir["initial_rollouts"], n - 1 if n > 1 else 1))
        tir["ready_batch"] = bool(tir.get("ready_batch", True))
    if algo == "arpo" and "loss_mode" not in user_tir:
        tir["loss_mode"] = "vanilla"
    if algo == "appo":
        if "loss_mode" not in user_tir:
            tir["loss_mode"] = "future_kl"
        if "reward_scale_discount" not in user_tir:
            tir["reward_scale_discount"] = 0.9
    if algo == "rae":
        tir["rae_full_tgt"] = bool(tir.get("rae_full_tgt", False))
        tir["rae_dead_end_backprop"] = bool(tir.get("rae_dead_end_backprop", True))
    algo_block["tir"] = tir
    # RL4: mirror the tir multi_tool_bonus into a structured RewardSpec under
    # algorithm.reward — the RL5 v2 yaml segment slot. This CONNECTS the
    # previously-dead DEFAULT_TIR["multi_tool_bonus"] config (overlay.py:29,
    # never consumed by collector.py's default_reward_fn call chain) to a
    # live, switchable/weightable component spec. Dual-read window:
    # algorithm.tir.multi_tool_bonus keeps its legacy read path (nothing
    # consumes it today besides this mirror), and the spec section below is
    # the canonical delivery surface going forward.
    algo_block["reward"] = _reward_section_from_tir(tir, user_tir)
    cfg["algorithm"] = algo_block
    if algo == "flow_grpo":
        # Trajectory outcome is already broadcast onto every turn by the VERL daemon.
        # These are AgentFlow's actor defaults; rollout.n stays the group size.
        # An explicit use_kl_loss=False in the rl-yaml survives the re-apply
        # (single-GPU memory layouts move KL into use_kl_in_reward instead).
        roll = dict(cfg.get("actor_rollout_ref") or {})
        actor = dict(roll.get("actor") or {})
        explicit_no_kl_loss = actor.get("use_kl_loss") is False
        if not explicit_no_kl_loss:
            actor["use_kl_loss"] = True
            actor["kl_loss_coef"] = 0.001
        actor["entropy_coeff"] = 0.0
        actor["clip_ratio_low"] = 0.2
        actor["clip_ratio_high"] = 0.3
        roll["actor"] = actor
        cfg["actor_rollout_ref"] = roll
    trainer = dict(cfg.get("trainer") or {})
    base_name = str(trainer.get("experiment_name") or "tir_agent")
    if not base_name.endswith(algo):
        trainer["experiment_name"] = f"{base_name}_{algo}"
    cfg["trainer"] = trainer
    return cfg


def _reward_section_from_tir(tir: Dict[str, Any], user_tir: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``algorithm.reward`` section (RewardSpec-shaped) from the tir dict.

    The effective multi_tool_bonus magnitude (DEFAULT_TIR 0.1 baseline, user
    yaml override wins — both live in ``tir`` at this point) becomes the
    multi_tool_bonus component's weight; weight 0 means disabled. format_penalty
    and accuracy_reward carry their frozen defaults (on; accuracy not
    switchable). ``user_tir`` is consulted only to document provenance — an
    explicit user ``multi_tool_bonus`` key lands in the section's ``meta``
    for auditability (RL5 resolved_algo will serialize this section).
    """
    from rl.rewards.assembly import reward_spec_from_tir

    spec = reward_spec_from_tir(tir)
    section: Dict[str, Any] = {
        "outcome": spec.outcome,
        "components": [
            {"id": comp.id, "weight": float(comp.weight), "enabled": bool(comp.enabled)}
            for comp in spec.components
        ],
    }
    if "multi_tool_bonus" in user_tir:
        section["meta"] = {"multi_tool_bonus_source": "user_tir"}
    return section


def apply_train_signal(config: Dict[str, Any], signal: "TrainSignal") -> Dict[str, Any]:
    """Map RL TrainSignal onto Hydra/VERL dict. Does not import DataProto."""
    algo = str(signal.advantage.name or "grpo").lower()
    cfg = apply_algo_overlay(config, algo)
    actor = dict(cfg.get("actor_rollout_ref", {}).get("actor") or {})
    actor["clip_ratio_low"] = float(signal.loss.clip_ratio_low)
    actor["clip_ratio_high"] = float(signal.loss.clip_ratio_high)
    actor["entropy_coeff"] = float(signal.loss.entropy_coeff)
    actor["kl_loss_coef"] = float(signal.loss.kl_loss_coef)
    if algo == "flow_grpo":
        # Preserve an explicit use_kl_loss=False from the rl-yaml (see
        # apply_algo_overlay); the re-apply here must not resurrect the flag.
        if actor.get("use_kl_loss") is not False:
            actor["use_kl_loss"] = True
        if float(signal.loss.kl_loss_coef) == 0.0 and actor.get("use_kl_loss") is True:
            actor["kl_loss_coef"] = 0.001
    roll = dict(cfg.get("actor_rollout_ref") or {})
    roll["actor"] = actor
    cfg["actor_rollout_ref"] = roll
    tir = dict(cfg["algorithm"].get("tir") or {})
    tir["gamma"] = float(signal.advantage.gamma)
    tir["reward_scale_discount"] = float(signal.advantage.reward_scale_discount)
    loss_mode = str(signal.loss.loss_mode or "vanilla")
    if algo == "appo" and loss_mode == "vanilla":
        loss_mode = str(tir.get("loss_mode") or "future_kl")
    tir["loss_mode"] = loss_mode
    tir["future_kl_weight"] = float(signal.loss.future_kl_weight)
    tir["decay_rate"] = float(signal.loss.decay_rate)
    tir["future_kl_clip_ratio"] = float(signal.loss.future_kl_clip_ratio)
    cfg["algorithm"]["tir"] = tir
    # RL1 D2 write-side double-write: mirror the FIPO keys into the config
    # subtree the vendored read side actually dispatches on
    # (dp_actor.py:392 fipo_cfg = self.config.get("policy_loss", {}) —
    # fsdp_workers.py:574 passes self.config.actor, i.e. the mirror target is
    # actor_rollout_ref.actor.policy_loss). Optional keys with read-side
    # defaults (chunk_size / safety_thresh / future_kl_clip_high_only) pass
    # through only when the TIR config / LossSpec.extra carries them, so
    # readers fall back to dp_actor defaults unchanged.
    if loss_mode == "future_kl":
        roll = dict(cfg.get("actor_rollout_ref") or {})
        actor = dict(roll.get("actor") or {})
        policy_loss_block = dict(actor.get("policy_loss") or {})
        for opt_key in ("chunk_size", "safety_thresh", "future_kl_clip_high_only"):
            if opt_key in tir and opt_key not in policy_loss_block:
                policy_loss_block[opt_key] = tir[opt_key]
            elif opt_key in (signal.loss.extra or {}) and opt_key not in policy_loss_block:
                policy_loss_block[opt_key] = signal.loss.extra[opt_key]
        # Existing keys win (explicit user yaml / previous-pass mirror
        # values re-applied idempotently); absent keys get the mirror value.
        policy_loss_block.setdefault("loss_mode", loss_mode)
        policy_loss_block.setdefault("future_kl_weight", float(signal.loss.future_kl_weight))
        policy_loss_block.setdefault("decay_rate", float(signal.loss.decay_rate))
        policy_loss_block.setdefault("future_kl_clip_ratio", float(signal.loss.future_kl_clip_ratio))
        actor["policy_loss"] = policy_loss_block
        roll["actor"] = actor
        cfg["actor_rollout_ref"] = roll
    else:
        # Retract a stale mirror from a previous pass. apply_train_signal runs
        # twice around the user-yaml deep-merge (train_tir_agent.py:370-374);
        # pass 1 mirrors future_kl, the user yaml may then flip tir.loss_mode
        # to vanilla, so pass 2 must withdraw the mirror or the vendored
        # reader (dp_actor.py:392) would still dispatch future_kl. Only a
        # block that exactly fingerprints the pass-1 mirror is withdrawn;
        # user-authored policy_loss blocks are left untouched.
        roll = dict(cfg.get("actor_rollout_ref") or {})
        actor = dict(roll.get("actor") or {})
        block = actor.get("policy_loss")
        mirror_keys = {
            "loss_mode",
            "future_kl_weight",
            "decay_rate",
            "future_kl_clip_ratio",
            "chunk_size",
            "safety_thresh",
            "future_kl_clip_high_only",
        }
        if (
            isinstance(block, dict)
            and set(block.keys()) <= mirror_keys
            and str(block.get("loss_mode")) == "future_kl"
            and float(block.get("future_kl_weight", -1.0)) == float(signal.loss.future_kl_weight)
            and float(block.get("decay_rate", -1.0)) == float(signal.loss.decay_rate)
            and float(block.get("future_kl_clip_ratio", -1.0)) == float(signal.loss.future_kl_clip_ratio)
        ):
            actor.pop("policy_loss", None)
            roll["actor"] = actor
            cfg["actor_rollout_ref"] = roll
    return ensure_trainer_horizon(cfg)


def ensure_trainer_horizon(config: Dict[str, Any]) -> Dict[str, Any]:
    """Make VERL actually run ``total_training_steps``.

    AgentLightningTrainer iterates ``range(total_epochs)`` over the dataloader
    and only uses ``total_training_steps`` as an early-stop. One epoch of the
    5-row GSM8K sample with ``train_batch_size=5`` is a single step, so
    ``steps=3, epochs=1`` exits after 1/3 with returncode 0.
    """
    cfg = deepcopy(config)
    trainer = dict(cfg.get("trainer") or {})
    steps = trainer.get("total_training_steps")
    if steps is None:
        return cfg
    steps_i = int(steps)
    epochs = int(trainer.get("total_epochs") or 1)
    if steps_i > 0 and epochs < steps_i:
        trainer["total_epochs"] = steps_i
        cfg["trainer"] = trainer
    return cfg
