"""Hand-verifiable numerics for GRPO/ARPO/APPO/flow_grpo advantage + reward anchors. CPU only.

Contract being locked (task3 report anchors):
- grpo:  advantage == (score-mean)/std per uid group (VERL compute_grpo_outcome_advantage,
          core_algos.py:246-308: unbiased std, +1e-6 eps in DENOMINATOR only; the row scalar is
          broadcast over the WHOLE masked row via scores.unsqueeze(-1)*response_mask).
          Then `_maybe_zero_prefix` (advantage.py:82-84, 182-190 → rae_advantage.py:34-83) default ON
          zeroes shared-prefix advantages (and returns) of child/probe rows.
- arpo:  identical to stock GRPO on a mixed parent+child batch. arpo's difference lives in
          branch sampling, NOT in the credit formula (advantage.py:74-75 passthrough).
- appo:  apply_appo_credit (advantage.py:118-179): branch rows zeroed ENTIRELY (adv+returns),
          parents gain discount*mean(child outcomes) added inside response_mask.
- flow_grpo: overlay (overlay.py:133-153) forces use_kl_loss=True + kl_loss_coef=0.001 unless
          the rl-yaml explicitly set use_kl_loss=False; episode outcome broadcast happens
          daemon-side (contrib env_verl daemon.py:754-844: final_reward copied to every turn row).
- future_kl (appo preset): mirrored into actor_rollout_ref.actor.policy_loss by
          apply_train_signal (RL1 double-write) — the vendored reader (dp_actor.py:392)
          dispatches future_kl for the first time; vanilla presets write no subtree.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(MAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W = 4  # token width of every fake row
EPS = 1e-6  # VERL GRPO epsilon (denominator only)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grpo_row_scale(outcomes):
    """Hand value of the VERL GRPO row scalar for a group with the given outcomes.

    mean = mean(outcomes); std = torch.std (unbiased, ddof=1) of outcomes;
    scalar_i = (outcome_i - mean) / (std + 1e-6). Computed in pure python (float
    math, same formula) so the test is a hand-verifiable oracle, independent of
    the torch code path being tested.
    """
    vals = [float(v) for v in outcomes]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # ddof=1
    std = var ** 0.5
    return mean, std, [(v - mean) / (std + EPS) for v in vals]


def _scores_batch(outcomes, uids, *, width=W, mask=None, adv=None):
    """token_level_scores with the outcome written on the LAST token of each row.

    VERL semantics: scores = token_level_rewards.sum(-1), so the row scalar == the
    outcome regardless of which token holds it.
    """
    n = len(outcomes)
    scores = torch.zeros(n, width)
    for i, v in enumerate(outcomes):
        scores[i, -1] = float(v)
    tensors = {
        "token_level_scores": scores,
        "token_level_rewards": scores.clone(),
        "response_mask": mask if mask is not None else torch.ones(n, width),
    }
    if adv is not None:
        tensors["advantages"] = adv
        tensors["returns"] = adv.clone()
    batch = DataProto.from_dict(tensors=tensors)
    # Production trainer sets uid=data_id_list before compute_advantage
    # (agentlightning/verl/trainer.py:309). GRPO groups by uid.
    batch.non_tensor_batch["uid"] = np.array(list(uids), dtype=object)
    return batch


@contextmanager
def _patched_compute_advantage(mode: str):
    """Swap advantage_mod.compute_advantage the way the trainer monkey-patch does.

    `rl/hooks/advantage.py` imports compute_advantage from verl at module import;
    `rl/hooks/trainer.py:37` rebinds it on the module. Both "identity" (existing
    tests) and "real" (genuine stock GRPO group math) are supported.
    """
    from rl.hooks import advantage as advantage_mod

    original = advantage_mod.compute_advantage
    if mode == "identity":
        advantage_mod.compute_advantage = lambda batch, **_: batch
    elif mode == "real":
        from verl.trainer.ppo.ray_trainer import compute_advantage as real_fn

        advantage_mod.compute_advantage = real_fn
    else:
        raise ValueError(mode)
    try:
        yield advantage_mod
    finally:
        advantage_mod.compute_advantage = original


def _apply(batch, algo, *, tir=None, patch="identity", norm_adv_by_std_in_grpo=True, num_repeat=4):
    """Run the production dispatch apply_tir_advantages with a chosen compute_advantage."""
    with _patched_compute_advantage(patch) as mod:
        return mod.apply_tir_advantages(
            batch,
            adv_estimator="grpo",
            gamma=1.0,
            lam=1.0,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config={"tir_algo": algo, "tir": dict(tir or {})},
        )


def _role_batch(outcomes, roles, parents, rollouts, *, data_ids=None, boundaries=None, adv=None, mask=None):
    batch = _scores_batch(outcomes, data_ids or ["t"] * len(outcomes), adv=adv, mask=mask)
    n = len(outcomes)
    batch.non_tensor_batch["rollout_role"] = np.array(roles, dtype=object)
    batch.non_tensor_batch["data_id_list"] = np.array(data_ids or ["t"] * n, dtype=object)
    batch.non_tensor_batch["rollout_id_list"] = np.array(rollouts, dtype=object)
    batch.non_tensor_batch["resume_parent_id"] = np.array(parents, dtype=object)
    batch.non_tensor_batch["resume_boundary"] = np.array(
        boundaries if boundaries is not None else [0 if r == "parent" else 2 for r in roles], dtype=object
    )
    return batch


# ---------------------------------------------------------------------------
# 1. grpo
# ---------------------------------------------------------------------------


class TestGrpoNumerics(unittest.TestCase):
    def test_outcome_advantage_matches_hand_computed_zscore(self):
        """4 trajectories, outcomes [1.0, 0.0, 0.5, 0.5], ONE group.

        Hand: mean=0.5; unbiased std=sqrt((0.25+0.25+0+0)/3)=sqrt(1/6)=0.4082482905;
        scalars = ±1.2247428 / 0 / 0 (epsilon=1e-6 in denominator only).
        VERL broadcasts the scalar across the whole masked row.
        """
        outcomes = [1.0, 0.0, 0.5, 0.5]
        mean, std, scalars = _grpo_row_scale(outcomes)
        self.assertAlmostEqual(mean, 0.5, places=12)
        self.assertAlmostEqual(std, 0.408248290463863, places=12)
        expected = torch.tensor([[s] * W for s in scalars])
        torch.testing.assert_close(expected[0, 0], torch.tensor(1.2247418), rtol=1e-5, atol=1e-7)

        batch = _scores_batch(outcomes, ["t"] * 4)
        out = _apply(batch, "grpo", patch="real", num_repeat=1)
        torch.testing.assert_close(out.batch["advantages"], expected, rtol=1e-5, atol=1e-7)
        # returns == advantages for outcome-only GRPO
        torch.testing.assert_close(out.batch["returns"], expected, rtol=1e-5, atol=1e-7)

    def test_shared_prefix_zeros_child_prefix_only(self):
        """_maybe_zero_prefix default ON: shared 2-token prefix of the child zeroed.

        outcomes [2.0, 0.0]: mean=1, unbiased std=sqrt(2)=1.4142135624.
        parent scalar=+0.7071067, child scalar=-0.7071067.
        child advantages: first 2 (shared) tokens → 0, last 2 → scalar.
        returns prefix zeroed identically; parent row untouched.
        """
        outcomes = [2.0, 0.0]
        _, _, scalars = _grpo_row_scale(outcomes)
        batch = _role_batch(outcomes, ["parent", "child"], ["", "r0"], ["r0", "r1"], boundaries=[0, 2])
        out = _apply(batch, "grpo", patch="real", num_repeat=1)

        child_adv = out.batch["advantages"][1]
        self.assertTrue(torch.equal(child_adv[:2], torch.zeros(2)), "shared prefix must be 0")
        torch.testing.assert_close(child_adv[2:], torch.full((2,), scalars[1]), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            out.batch["advantages"][0], torch.full((W,), scalars[0]), rtol=1e-5, atol=1e-7
        )
        self.assertTrue(torch.equal(out.batch["returns"][1][:2], torch.zeros(2)))
        torch.testing.assert_close(out.batch["returns"][1][2:], torch.full((2,), scalars[1]), rtol=1e-5, atol=1e-7)

    def test_single_row_group_advantage_equals_raw_score(self):
        """VERL fallback (core_algos.py:292-295): group of ONE → mean=0, std=1.

        scalar = (0.7 - 0)/(1 + 1e-6) = 0.6999993 — i.e. the raw score survives
        as its own advantage when the group collapses to a single trajectory.
        """
        batch = _scores_batch([0.7], ["solo"])
        out = _apply(batch, "grpo", patch="real", num_repeat=1)
        torch.testing.assert_close(
            out.batch["advantages"][0], torch.full((W,), 0.7 / (1 + 1e-6)), rtol=1e-5, atol=1e-7
        )

    def test_zero_prefix_off_keeps_full_child_rows(self):
        """Explicit zero_prefix_adv_for_children=False → child row keeps full scalar."""
        outcomes = [2.0, 0.0]
        _, _, scalars = _grpo_row_scale(outcomes)
        batch = _role_batch(outcomes, ["parent", "child"], ["", "r0"], ["r0", "r1"], boundaries=[0, 2])
        out = _apply(
            batch, "grpo", tir={"zero_prefix_adv_for_children": False}, patch="real", num_repeat=1
        )
        torch.testing.assert_close(
            out.batch["advantages"][1], torch.full((W,), scalars[1]), rtol=1e-5, atol=1e-7
        )


# ---------------------------------------------------------------------------
# 2. arpo == soft GRPO (credit identical)
# ---------------------------------------------------------------------------


class TestArpoIsStockGrpoOnMixedBatch(unittest.TestCase):
    """Branch rows + global rollouts share one uid group; arpo returns the stock
    GRPO output untouched (advantage.py:74-75) → advantages identical to plain grpo
    with prefix-zeroing disabled. Locks "difference is in sampling, not credit"."""


    OUTCOMES = [1.0, 0.0, 1.0, 0.5]
    ROLES = ["parent", "child", "parent", "child"]
    PARENTS = ["", "r0", "", "r2"]
    ROLLOUTS = ["r0", "r1", "r2", "r3"]

    def _mk(self):
        return _role_batch(
            self.OUTCOMES,
            self.ROLES,
            self.PARENTS,
            self.ROLLOUTS,
            boundaries=[0, 2, 0, 1],
        )

    def test_mixed_batch_same_advantages_as_grpo(self):
        arpo_out = _apply(self._mk(), "arpo", patch="real", num_repeat=1)
        grpo_out = _apply(
            self._mk(), "grpo", tir={"zero_prefix_adv_for_children": False}, patch="real", num_repeat=1
        )
        torch.testing.assert_close(arpo_out.batch["advantages"], grpo_out.batch["advantages"], rtol=0, atol=0)
        torch.testing.assert_close(arpo_out.batch["returns"], grpo_out.batch["returns"], rtol=0, atol=0)
        # hand-check row0 scalar: mean=0.625, unbiased std=0.4787135538
        _, _, scalars = _grpo_row_scale(self.OUTCOMES)
        torch.testing.assert_close(arpo_out.batch["advantages"][0, -1], torch.tensor(scalars[0]), rtol=1e-5, atol=1e-7)

    def test_arpo_does_not_zero_branch_rows(self):
        """Contrast with appo: arpo keeps branch rows carrying FULL group advantage.

        Real stock GRPO on the mixed batch → branch rows must NOT be whole-row-zeroed
        (they hold ±z group scalars), unlike appo which zeroes them entirely.
        """
        _, _, scalars = _grpo_row_scale(self.OUTCOMES)
        out = _apply(self._mk(), "arpo", patch="real", num_repeat=1)
        adv = out.batch["advantages"]
        self.assertFalse(torch.equal(adv[1], torch.zeros(W)))
        self.assertFalse(torch.equal(adv[3], torch.zeros(W)))
        torch.testing.assert_close(adv[1], torch.full((W,), scalars[1]), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(adv[3], torch.full((W,), scalars[3]), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. appo (core)
# ---------------------------------------------------------------------------


class TestAppoCreditNumerics(unittest.TestCase):
    """Scenario (hand-checkable): parent r0 + 2 branches (r1, r2) + 1 lone parent r3,
    same data_id. outcomes r0=1.0, r1=2.0, r2=4.0, r3=0.5. Base advantages all 1.

    apply_appo_credit(discount=0.9):
      branch rows 1,2 → advantages=0 (WHOLE row), returns=0
      parent r0 → 1 + 0.9*mean(2,4) = 1 + 2.7 = 3.7 (inside mask)
      parent r3 → no children → unchanged
    """

    BATCH_OUTCOMES = [1.0, 2.0, 4.0, 0.5]

    def _mixed_batch(self, **kw):
        return _role_batch(
            self.BATCH_OUTCOMES,
            ["parent", "branch", "branch", "parent"],
            ["", "r0", "r0", ""],
            ["r0", "r1", "r2", "r3"],
            adv=torch.ones(4, W),
            **kw,
        )

    def test_branch_rows_fully_zeroed_and_parent_gets_discounted_mean(self):
        from rl.hooks.advantage import apply_appo_credit

        out = apply_appo_credit(self._mixed_batch(), discount=0.9)
        for branch_row in (1, 2):
            self.assertTrue(torch.equal(out.batch["advantages"][branch_row], torch.zeros(W)), branch_row)
            self.assertTrue(torch.equal(out.batch["returns"][branch_row], torch.zeros(W)), branch_row)
        self.assertTrue(torch.equal(out.batch["advantages"][0], torch.full((W,), 3.7)))
        self.assertTrue(torch.equal(out.batch["returns"][0], torch.ones(W)), "parent returns kept")
        self.assertTrue(torch.equal(out.batch["advantages"][3], torch.ones(W)))
        self.assertTrue(torch.equal(out.batch["returns"][3], torch.ones(W)))

    def test_mask_limits_the_bonus_to_valid_tokens(self):
        """Parent valid tokens = first 3; bonus lands there, masked tail stays base 1."""
        from rl.hooks.advantage import apply_appo_credit

        mask = torch.ones(4, W)
        mask[0, 3] = 0.0
        batch = self._mixed_batch(mask=mask)
        out = apply_appo_credit(batch, discount=0.9)
        torch.testing.assert_close(out.batch["advantages"][0], torch.tensor([3.7, 3.7, 3.7, 1.0]), rtol=0, atol=0)

    def test_discount_parameter_flows_through(self):
        """discount=0.5 → bonus 0.5*mean(2,4)=1.5 → parent 2.5. Locks the param chain."""
        from rl.hooks.advantage import apply_appo_credit

        out = apply_appo_credit(self._mixed_batch(), discount=0.5)
        self.assertTrue(torch.equal(out.batch["advantages"][0], torch.full((W,), 2.5)))

    def test_child_role_also_treated_as_branch(self):
        """Roles child/probe/branch all zeroed (advantage.py:148-150)."""
        from rl.hooks.advantage import apply_appo_credit

        batch = _role_batch(
            [1.0, 3.0, 5.0],
            ["parent", "child", "probe"],
            ["", "r0", "r0"],
            ["r0", "r1", "r2"],
            data_ids=["t", "t", "t"],
            adv=torch.ones(3, W),
        )
        out = apply_appo_credit(batch, discount=0.9)
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(W)))
        self.assertTrue(torch.equal(out.batch["advantages"][2], torch.zeros(W)))
        self.assertTrue(torch.equal(out.batch["advantages"][0], torch.full((W,), 1 + 0.9 * 4.0)))

    def test_full_appo_path_carries_discount_from_config(self):
        """apply_tir_advantages(algo=appo) reads tir.reward_scale_discount → apply_appo_credit
        (advantage.py:76-80). Same scenario directly through the dispatch."""
        out = _apply(self._mixed_batch(), "appo", tir={"reward_scale_discount": 0.9})
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(W)))
        self.assertTrue(torch.equal(out.batch["advantages"][2], torch.zeros(W)))
        self.assertTrue(torch.equal(out.batch["advantages"][0], torch.full((W,), 3.7)))

    def test_overlay_default_discount_chain(self):
        """Config chain: overlay('appo') defaults reward_scale_discount=0.9 (overlay.py:127-131);
        AdvantageSpec default is 0.9 as well (rl/loss.py:27-29)."""
        from rl.hooks.overlay import apply_algo_overlay
        from rl.loss import AdvantageSpec

        cfg = apply_algo_overlay(
            {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}}, "trainer": {}}, "appo"
        )
        self.assertEqual(cfg["algorithm"]["tir"]["reward_scale_discount"], 0.9)
        self.assertEqual(AdvantageSpec(name="appo").reward_scale_discount, 0.9)


# ---------------------------------------------------------------------------
# 4. flow_grpo
# ---------------------------------------------------------------------------


class TestFlowGrpoOverlayForcesKl(unittest.TestCase):
    def _base(self, **actor):
        return {
            "algorithm": {},
            "actor_rollout_ref": {"rollout": {"n": 8}, "actor": dict(actor)},
            "trainer": {},
        }

    def test_yaml_silent_forces_kl_loss_on(self):
        """overlay.py:133-153: actor block WITHOUT explicit use_kl_loss → forced True + 0.001."""
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(self._base(use_kl_loss=None), "flow_grpo")
        actor = cfg["actor_rollout_ref"]["actor"]
        self.assertTrue(actor["use_kl_loss"])
        self.assertEqual(actor["kl_loss_coef"], 0.001)
        self.assertEqual(actor["entropy_coeff"], 0.0)
        self.assertEqual(actor["clip_ratio_low"], 0.2)
        self.assertEqual(actor["clip_ratio_high"], 0.3)

    def test_happy_path_when_actor_block_missing(self):
        """No actor block at all → overlay creates one with forced KL."""
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(
            {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}}, "trainer": {}}, "flow_grpo"
        )
        self.assertTrue(cfg["actor_rollout_ref"]["actor"]["use_kl_loss"])
        self.assertEqual(cfg["actor_rollout_ref"]["actor"]["kl_loss_coef"], 0.001)

    def test_explicit_kl_zero_coef_still_forced_to_0_001(self):
        """use_kl_loss unset/true-like with coef 0 → coef forced to 0.001."""
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(self._base(kl_loss_coef=0.0), "flow_grpo")
        actor = cfg["actor_rollout_ref"]["actor"]
        self.assertTrue(actor["use_kl_loss"])
        self.assertEqual(actor["kl_loss_coef"], 0.001)


class TestFlowGrpoEpisodeBroadcast(unittest.TestCase):
    """Daemon-side broadcast (agent-lightning contrib env_verl daemon.py:754-844):
    every TURN of an episode becomes its own row, and with use_final_reward_as_step_reward=
    True the episode final_reward is stamped into `scores` for EVERY turn row (830-837:
    ONE row per (rollout, turn) with final_reward) — that IS the broadcast. GRPO group math
    then sees per-turn rows carrying the episode scalar."""

    @staticmethod
    def _broadcast_rows(ep_rewards, turn_counts):
        """Pure-python oracle of daemon behavior: final_reward copied per turn row."""
        rows = []
        for fr, n_turns in zip(ep_rewards, turn_counts):
            rows.extend([float(fr)] * n_turns)
        return rows

    def test_episode_reward_repeated_on_every_turn(self):
        rows = self._broadcast_rows([1.0, -1.0], [3, 2])
        self.assertEqual(rows, [1.0, 1.0, 1.0, -1.0, -1.0])

    def test_broadcast_rows_flow_through_stock_group_math(self):
        """Broadcast rows → stock GRPO outcome advantage == hand math.

        uid group holds TURN-rows of the same episode. Distinct episodes in one group
        give non-trivial std. episodes: A(scalar 1.0, 2 turns) B(-1.0, 1 turn) C(0.5, 1 turn)
        → 4 rows [1, 1, -1, 0.5]; mean=0.375, unbiased std=sqrt(0.8958333)=0.9464847243;
        scalars=[0.6603375, 0.6603375, -1.4527425, 0.1320675].
        """
        rows = self._broadcast_rows([1.0, -1.0, 0.5], [2, 1, 1])
        _, _, scalars = _grpo_row_scale(rows)
        batch = _scores_batch(rows, ["task"] * len(rows))
        out = _apply(batch, "flow_grpo", patch="real", num_repeat=1)
        torch.testing.assert_close(
            out.batch["advantages"], torch.tensor([[s] * W for s in scalars]), rtol=1e-5, atol=1e-7
        )

    def test_same_episode_broadcast_group_has_zero_advantage(self):
        """One episode → every row carries the SAME broadcast scalar → std=0.

        VERL single-suite: len==1 → mean0/std1; >1 identical scores → std=0 →
        advantage = 0/(0+1e-6) = 0 exactly. Broadcast quirk: rows of the same
        episode end up with zero advantage (the signal lives across episodes).
        """
        rows = self._broadcast_rows([1.0], [3])  # one episode, 3 turns, scalar 1.0
        self.assertEqual(len(rows), 3)
        batch = _scores_batch(rows, ["t"] * len(rows))
        out = _apply(batch, "flow_grpo", patch="real", num_repeat=1)
        torch.testing.assert_close(out.batch["advantages"], torch.zeros(len(rows), W), rtol=0, atol=0)

    def test_flow_grpo_path_is_stock_grpo_no_branch_credit(self):
        """Advantage path: apply_tir_advantages(flow_grpo) returns the GRPO batch untouched
        (advantage.py:71-73) — no prefix zeroing (zero_prefix ignored) and no branch credit."""
        outcomes = [2.0, 0.0]
        _, _, scalars = _grpo_row_scale(outcomes)
        batch = _role_batch(outcomes, ["parent", "child"], ["", "r0"], ["r0", "r1"], boundaries=[0, 2])
        out = _apply(batch, "flow_grpo", tir={"zero_prefix_adv_for_children": True}, patch="real", num_repeat=1)
        torch.testing.assert_close(
            out.batch["advantages"][1], torch.full((W,), scalars[1]), rtol=1e-5, atol=1e-7
        )


class TestFutureKlMirror(unittest.TestCase):
    """RL1: apply_train_signal mirrors algorithm.tir loss keys into
    actor_rollout_ref.actor.policy_loss — the config subtree the vendored
    reader dispatches on (dp_actor.py:392 fipo_cfg = self.config.get(
    "policy_loss", {}); fsdp_workers.py:574 passes self.config.actor).
    Vanilla runs must get NO policy_loss subtree so the read side keeps its
    dp_actor defaults (loss_mode "vanilla") unchanged."""

    def _base(self):
        return {
            "algorithm": {},
            "actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}},
            "trainer": {"total_training_steps": 8},
        }

    def _signal(self, algo, **loss_kw):
        from rl.loss import AdvantageSpec, LossSpec
        from rl.train_signal import TrainSignal

        return TrainSignal(
            advantage=AdvantageSpec(name=algo),
            loss=LossSpec(name=algo, **loss_kw),
        )

    def test_future_kl_mirror_written_to_actor_policy_loss(self):
        """appo preset + LossSpec(loss_mode="future_kl") → cfg contains the four
        mirrored keys with the LossSpec values (decay_rate aligned to the
        read-side default 128.0 when LossSpec uses its own default)."""
        from rl.hooks.overlay import apply_train_signal

        signal = self._signal("appo", loss_mode="future_kl")
        cfg = apply_train_signal(self._base(), signal)
        self.assertEqual(cfg["algorithm"]["tir"]["loss_mode"], "future_kl")
        pol = cfg["actor_rollout_ref"]["actor"]["policy_loss"]
        self.assertEqual(pol["loss_mode"], "future_kl")
        self.assertEqual(pol["future_kl_weight"], 1.0)
        self.assertEqual(pol["decay_rate"], 128.0)  # LossSpec default = read-side default
        self.assertEqual(pol["future_kl_clip_ratio"], 0.2)
        # the deprecated downgrade marker is gone with the double-write
        self.assertNotIn("loss_applied_to_actor", cfg["algorithm"]["tir"])

    def test_vanilla_run_gets_no_policy_loss_subtree(self):
        """Non-future_kl LossSpec presets (grpo / arpo) → no actor.policy_loss
        subtree, so the vendored read point keeps dp_actor defaults (vanilla)."""
        from rl.hooks.overlay import apply_train_signal

        for algo in ("grpo", "arpo"):
            cfg = apply_train_signal(self._base(), self._signal(algo))
            self.assertEqual(cfg["algorithm"]["tir"]["loss_mode"], "vanilla", algo)
            self.assertNotIn("policy_loss", cfg["actor_rollout_ref"]["actor"], algo)

    def test_explicit_user_yaml_override_beats_mirror(self):
        """Explicit user-yaml actor.policy_loss values survive the mirror.
        The two-pass flow (train_tir_agent.py:370-374: apply → _deep_merge
        user yaml → re-apply) must keep user-authored keys — the mirror only
        fills absent keys (setdefault semantics), never overwrites."""
        from rl.hooks.overlay import apply_train_signal

        signal = self._signal("appo", loss_mode="future_kl")

        # Pass 1 mirrors; user yaml then sets an explicit policy_loss key.
        cfg = apply_train_signal(self._base(), signal)
        cfg["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] = "vanilla"
        cfg["actor_rollout_ref"]["actor"]["policy_loss"]["future_kl_weight"] = 0.5
        cfg = apply_train_signal(cfg, signal)  # re-apply like the real flow
        pol = cfg["actor_rollout_ref"]["actor"]["policy_loss"]
        # explicit values beat the mirror on every re-apply pass
        self.assertEqual(pol["loss_mode"], "vanilla")
        self.assertEqual(pol["future_kl_weight"], 0.5)
        # absent keys still receive mirror values
        self.assertEqual(pol["decay_rate"], 128.0)
        self.assertEqual(pol["future_kl_clip_ratio"], 0.2)

    def test_explicit_tir_vanilla_withdraws_stale_mirror(self):
        """User yaml flipping tir.loss_mode to vanilla (the one documented
        escape hatch, §2.1) must withdraw the pass-1 mirror entirely — the
        vendored reader would otherwise still dispatch future_kl."""
        from rl.hooks.overlay import apply_train_signal

        # Real flow: train_tir_agent builds LossSpec(name=algo) with the
        # loss_mode default "vanilla"; the appo revival guard pulls the
        # effective mode out of tir. Mirror on pass 1, flip on merge, the
        # re-apply pass must withdraw the mirror.
        signal = self._signal("appo")
        cfg = apply_train_signal(self._base(), signal)
        self.assertEqual(cfg["algorithm"]["tir"]["loss_mode"], "future_kl")
        self.assertIn("policy_loss", cfg["actor_rollout_ref"]["actor"])
        # user yaml deep-merges tir.loss_mode=vanilla (flip after mirror)
        cfg["algorithm"]["tir"]["loss_mode"] = "vanilla"
        cfg = apply_train_signal(cfg, signal)  # re-apply pass 2
        self.assertEqual(cfg["algorithm"]["tir"]["loss_mode"], "vanilla")
        self.assertNotIn("policy_loss", cfg["actor_rollout_ref"]["actor"])

    def test_mirror_only_fills_optional_keys_when_carried(self):
        """Optional keys (chunk_size/safety_thresh/future_kl_clip_high_only)
        pass through only when the TIR config carries them; otherwise the
        reader falls back to dp_actor defaults."""
        from rl.hooks.overlay import apply_train_signal

        signal = self._signal("appo", loss_mode="future_kl")
        cfg = apply_train_signal(self._base(), signal)
        pol = cfg["actor_rollout_ref"]["actor"]["policy_loss"]
        self.assertNotIn("chunk_size", pol)
        self.assertNotIn("safety_thresh", pol)
        self.assertNotIn("future_kl_clip_high_only", pol)

        base_with_tir = self._base()
        base_with_tir["algorithm"]["tir"] = {"chunk_size": 256}
        cfg2 = apply_train_signal(base_with_tir, signal)
        pol2 = cfg2["actor_rollout_ref"]["actor"]["policy_loss"]
        self.assertEqual(pol2["chunk_size"], 256)
        self.assertNotIn("safety_thresh", pol2)


# ---------------------------------------------------------------------------
# 5. reward anchors (rl/rewards/outcome.py:149-167)
# ---------------------------------------------------------------------------


class TestRewardAnchors(unittest.TestCase):
    """test_outcome_hotpot.py:65-82 already covers: format-gate −1, truncated-think 0,
    exact alias F1 1.0, empty prediction 0. New anchors here: partial-F1 hand value,
    gsm8k numeric path, and the two-tool +0.1 bonus."""

    def test_partial_f1_hand_value(self):
        """pred 'the Prussian soldier' vs gold 'Prussian' (hotpot → QA F1).
        normalize: pred tokens [prussian soldier], gold [prussian];
        common={prussian}: prec=1/2, rec=1/1 → F1=2*(0.5*1)/(1.5)=2/3≈0.6667."""
        from rl.rewards.outcome import compute_outcome_reward, f1_score

        self.assertAlmostEqual(f1_score("the Prussian soldier", ["Prussian"]), 2 / 3, places=6)
        r = compute_outcome_reward(
            "<answer>the Prussian soldier</answer>",
            "Prussian",
            source="hotpot",
            format_ok=True,
            n_search=0,
            n_python=0,
        )
        self.assertAlmostEqual(r, 2 / 3, places=6)  # Acc>0 → Acc, no tools → no bonus

    def test_two_tool_bonus_composes_with_f1(self):
        """Both tools used → reward = F1 + 0.1 (outcome.py:163-166)."""
        from rl.rewards.outcome import compute_outcome_reward

        r = compute_outcome_reward(
            "<answer>the Prussian soldier</answer>",
            "Prussian",
            source="hotpot",
            format_ok=True,
            n_search=2,
            n_python=1,
        )
        self.assertAlmostEqual(r, 2 / 3 + 0.1, places=6)
        # single-tool families → no bonus even when several calls
        r_single = compute_outcome_reward(
            "<answer>the Prussian soldier</answer>",
            "Prussian",
            source="hotpot",
            format_ok=True,
            n_search=3,
            n_python=0,
        )
        self.assertAlmostEqual(r_single, 2 / 3, places=6)

    def test_gsm8k_numeric_match_and_bonus(self):
        """gsm8k → 0/1 numeric match: '#### 16' vs '16' exact → 1 (+0.1 both tools)."""
        from rl.rewards.outcome import compute_outcome_reward

        self.assertEqual(
            compute_outcome_reward(
                "#### 16", "16", source="gsm8k", format_ok=True, n_search=0, n_python=0
            ),
            1.0,
        )
        self.assertAlmostEqual(
            compute_outcome_reward(
                "#### 17", "16", source="gsm8k", format_ok=True, n_search=1, n_python=1
            ),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            compute_outcome_reward(
                "#### 16", "16", source="gsm8k", format_ok=True, n_search=1, n_python=1
            ),
            1.1,
            places=6,
        )

    def test_bing_free_synthetic_anchor(self):
        """Task2-independent synthetic anchor (no Bing/search dependency): a bespoke
        source string still resolves through the QA F1 branch with hand math."""
        from rl.rewards.outcome import compute_outcome_reward

        # source not 'gsm8k' → F1 branch. exact → 1.0
        self.assertEqual(
            compute_outcome_reward(
                "<answer>hello world</answer>",
                "hello world",
                source="synthetic_bing_free",
                format_ok=True,
                n_search=0,
                n_python=0,
            ),
            1.0,
        )
        # 'hello there' vs 'hello world': pred [hello there], gold [hello world];
        # common={hello}: prec=1/2, rec=1/2 → F1=0.5
        self.assertAlmostEqual(
            compute_outcome_reward(
                "<answer>hello there</answer>",
                "hello world",
                source="synthetic_bing_free",
                format_ok=True,
                n_search=0,
                n_python=0,
            ),
            0.5,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
