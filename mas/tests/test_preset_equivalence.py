"""RL3 equivalence gate: pipeline path vs frozen legacy dispatch, element-wise.

For every one of the 8 presets the SAME fixture batch (same seed, same tensors
and non_tensor columns) goes through ``apply_tir_advantages`` (pipeline: base +
ordered operator stack) and ``legacy_apply_tir_advantages`` (frozen verbatim
copy of the pre-RL3 eight-branch if/return). ``np.testing.assert_array_equal``
asserts advantages AND response_mask (and returns where present) — byte/element
level, no tolerance.

Domain-boundary note (RL_design §3.1.6, adopted from the plan): flow_grpo's
KL forcing (use_kl_loss/kl_loss_coef) is an *overlay-layer config overlay*,
not an advantage-tensor transform — it is asserted in test_overlay_kl_semantics /
test_advantage_numerics instead and never enters this tensor comparison scope.

rae specifics: the verdict fixture columns are constructed here (verdict_list +
action_key_list + old_log_probs) because the daemon-side adjudication fills
them at rollout time; both paths get the exact same construction.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(MAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W = 4  # token width


def _mixed_batch(*, with_verdict: bool = False, with_entropys: bool = False,
                 with_old_logprobs: bool = False, with_ig: bool = False,
                 tir: dict | None = None, seed: int = 7):
    """One shared scenario rich enough to exercise every operator.

    6 rows: parent r0 + branch r1(→r0) + parent r2 + branch r3(→r2)
            + lone parent r4 + branch r5(→r0); grouped by two data_ids.
    Carries every column any of the 8 presets' operators consume
    (rollout_role / resume_boundary / resume_parent_id / verdict_list /
    action_key_list / anchor_obs / ig_logprob_delta / turn_index_list ...)
    plus optional tensors (entropys / old_log_probs — see notes inside).
    """
    g = torch.Generator().manual_seed(seed)
    n = 6
    mask = torch.ones(n, W)
    mask[0, -1] = 0.0  # one masked tail so mask itself is worth comparing
    scores = torch.zeros(n, W)
    outcomes = [1.0, 0.0, 0.5, 1.0, 0.0, 2.0]
    for i, v in enumerate(outcomes):
        scores[i, -1] = float(v)
    tensors = {
        "token_level_scores": scores,
        "token_level_rewards": scores.clone(),
        "response_mask": mask,
    }
    if with_entropys:
        tensors["entropys"] = torch.randn(n, W, generator=g)
    if with_old_logprobs:
        # NOTE: as a nested python list, NOT a torch tensor — the frozen
        # kernel does batch.batch.get("old_log_probs") or batch.batch.get(...)
        # (rae_advantage.py:180) and torch tensors are truthy-ambiguous.
        # Row-wise torch ops accept list rows identically.
        tensors["old_log_probs"] = [
            [float(-x - 0.5) for x in row] for row in torch.rand(n, W, generator=g).tolist()
        ]
    batch = DataProto.from_dict(tensors=tensors)
    ntb = batch.non_tensor_batch
    # NOTE: list-typed non_tensor columns (not np.ndarray) — the frozen
    # kernels (advantage.py `_apply_igpo/_apply_gigpo`, rae_advantage.py:176
    # `_nt_get(...) or []`) use bare `or` truthiness on these values, which
    # is ambiguous on multi-element ndarrays. Both the legacy and the
    # pipeline path share those kernels, so equivalence is unaffected; the
    # latent ndarray-crash in the kernels themselves is a pre-existing
    # condition outside RL3's change surface (kernels frozen verbatim).
    ntb["uid"] = ["d0", "d0", "d0", "d0", "d1", "d1"]
    ntb["data_id_list"] = ["d0", "d0", "d0", "d0", "d1", "d1"]
    ntb["rollout_id_list"] = ["r0", "r1", "r2", "r3", "r4", "r5"]
    ntb["rollout_role"] = ["parent", "child", "parent", "branch", "parent", "probe"]
    ntb["resume_parent_id"] = ["", "r0", "", "r2", "", "r0"]
    ntb["resume_boundary"] = [0, 2, 0, 1, 0, 3]
    ntb["site_id_list"] = ["s"] * n
    if with_verdict:
        ntb["verdict_list"] = ["none", "invalidate", "validate", "abstain", "validate", "invalidate"]
        ntb["action_key_list"] = ["", "ak0", "", "ak0", "", "ak0"]
    ntb["anchor_obs"] = ["h0", "h0", "h0", "h0", "h1", "h1"]
    if with_ig:
        ntb["ig_logprob_delta"] = [0.1, -0.2, 0.3, 0.0, 0.25, -0.1]
    ntb["turn_index_list"] = [0, 1, 0, 1, 0, 1]
    return batch


def _run_both(mod, batch, algo, *, tir=None, num_repeat=2):
    """Run deep copies of the batch through pipeline and legacy paths.

    Stage-0 is a deterministic identity stand-in that materializes
    advantages/returns from token_level_scores exactly like the production
    compute_advantage contract (row scalar broadcast over the masked row) —
    stock-GRPO numerics themselves are already locked in
    test_advantage_numerics, this keeps the comparison purely about dispatch
    composition. Both paths start from byte-identical copies.
    """
    import copy

    algo_cfg = {"tir_algo": algo, "tir": dict(tir or {})}

    def _ident(b, **_kw):
        scores = b.batch["token_level_scores"]
        row = scores.sum(-1)
        mask = b.batch["response_mask"]
        adv = row.unsqueeze(-1).repeat(1, scores.shape[-1]) * mask
        b.batch["advantages"] = adv
        b.batch["returns"] = adv.clone()
        return b

    original = mod.compute_advantage
    mod.compute_advantage = _ident
    try:
        pipe = mod.apply_tir_advantages(
            copy.deepcopy(batch),
            adv_estimator="grpo",
            gamma=1.0,
            lam=1.0,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=True,
            config=algo_cfg,
        )
        legacy = mod.legacy_apply_tir_advantages(
            copy.deepcopy(batch),
            adv_estimator="grpo",
            gamma=1.0,
            lam=1.0,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=True,
            config=algo_cfg,
        )
    finally:
        mod.compute_advantage = original
    return pipe, legacy


def _assert_equal_batches(tc: unittest.TestCase, pipe: DataProto, legacy: DataProto):
    """Element-wise: advantages + response_mask (+ returns when present)."""
    np.testing.assert_array_equal(
        pipe.batch["advantages"].detach().cpu().numpy(),
        legacy.batch["advantages"].detach().cpu().numpy(),
        err_msg="advantages differ between pipeline and legacy paths",
    )
    np.testing.assert_array_equal(
        pipe.batch["response_mask"].detach().cpu().numpy(),
        legacy.batch["response_mask"].detach().cpu().numpy(),
        err_msg="response_mask differs",
    )
    if "returns" in pipe.batch and "returns" in legacy.batch:
        np.testing.assert_array_equal(
            pipe.batch["returns"].detach().cpu().numpy(),
            legacy.batch["returns"].detach().cpu().numpy(),
            err_msg="returns differ",
        )
    # non_tensor slices must be untouched by both paths (read-only contract)
    for key in legacy.non_tensor_batch:
        np.testing.assert_array_equal(
            np.asarray(pipe.non_tensor_batch[key]),
            np.asarray(legacy.non_tensor_batch[key]),
            err_msg=f"non_tensor column {key} mutated",
        )


class TestPresetEquivalence(unittest.TestCase):
    """8 preset cases test_preset_equivalence_{grpo,arpo,appo,aepo,igpo,gigpo,rae,flow_grpo}."""

    PRESETS = ("grpo", "arpo", "appo", "aepo", "igpo", "gigpo", "rae", "flow_grpo")

    def test_preset_equivalence_grpo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch()
        pipe, legacy = _run_both(adv, b, "grpo")
        _assert_equal_batches(self, pipe, legacy)

    def test_preset_equivalence_arpo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch()
        pipe, legacy = _run_both(adv, b, "arpo")
        _assert_equal_batches(self, pipe, legacy)

    def test_preset_equivalence_appo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch()
        pipe, legacy = _run_both(adv, b, "appo")
        _assert_equal_batches(self, pipe, legacy)
        # appo actually mutates advantages (branch zero + parent bonus) —
        # guard against both paths trivially no-opping.
        self.assertFalse(
            torch.equal(pipe.batch["advantages"], torch.zeros(6, W)),
            "appo credit did not fire (test would be vacuous)",
        )

    def test_preset_equivalence_aepo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch(with_entropys=True)
        pipe, legacy = _run_both(adv, b, "aepo")
        _assert_equal_batches(self, pipe, legacy)
        # sanity: the entropy reshape actually changed the tensor relative to
        # the stage-0 baseline (identity stand-in = scalar broadcast rows).
        scores = b.batch["token_level_scores"]
        stage0 = scores.sum(-1).unsqueeze(-1).repeat(1, W) * b.batch["response_mask"]
        self.assertFalse(
            torch.equal(pipe.batch["advantages"], stage0),
            "aepo reshape did not fire (test would be vacuous)",
        )

    def test_preset_equivalence_igpo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch(with_ig=True)
        pipe, legacy = _run_both(adv, b, "igpo", tir={"gamma": 0.9})
        _assert_equal_batches(self, pipe, legacy)
        self.assertIn("returns", pipe.batch, "igpo writes both advantages and returns")

    def test_preset_equivalence_gigpo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch()
        pipe, legacy = _run_both(adv, b, "gigpo", tir={"step_advantage_w": 0.7, "gamma": 0.95})
        _assert_equal_batches(self, pipe, legacy)
        self.assertIn("returns", pipe.batch)

    def test_preset_equivalence_rae(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch(with_verdict=True)
        pipe, legacy = _run_both(adv, b, "rae")
        _assert_equal_batches(self, pipe, legacy)

    def test_preset_equivalence_flow_grpo(self):
        from rl.hooks import advantage as adv

        b = _mixed_batch()
        pipe, legacy = _run_both(adv, b, "flow_grpo")
        _assert_equal_batches(self, pipe, legacy)
        # flow_grpo advantage path is pure GRPO passthrough — KL overlay is
        # config-side and out of the tensor comparison domain (see module doc).

    # -- extra variants on the same equivalence contract ----------------------

    def test_preset_equivalence_rae_full_tgt_variant(self):
        """rae + tir rae_full_tgt=True → operator stack swaps rae_verdict→rae_full_tgt.

        NOTE: runs the frozen kernel's uniform old-prob proxy branch (no
        old_log_probs tensor). The kernel's real-tensor branch is a pre-
        existing latent crash: rae_advantage.py:180 does
        ``batch.batch.get("old_log_probs") or batch.batch.get("old_logprobs")``
        and torch tensors are truthy-ambiguous — the rae_full_tgt=True path
        has never executed with real tensors. Both paths share the frozen
        kernel, so pipeline-vs-legacy equivalence below is still a complete
        execution of apply_rae_full_tgt_advantages (renorm + bias)."""
        from rl.hooks import advantage as adv

        b = _mixed_batch(with_verdict=True)
        pipe, legacy = _run_both(adv, b, "rae", tir={"rae_full_tgt": True})
        _assert_equal_batches(self, pipe, legacy)

    def test_preset_equivalence_grpo_zero_prefix_off_and_real_backbone(self):
        """Real VERL GRPO stage-0 (non-identity) + explicit zero-prefix off.

        Exercises the identical compute_advantage feeding both paths, i.e.
        end-to-end equality including the shared base computation."""
        from rl.hooks import advantage as adv
        from verl.trainer.ppo.ray_trainer import compute_advantage as real_fn

        b = _mixed_batch()
        algo_cfg = {
            "tir_algo": "grpo",
            "tir": {"zero_prefix_adv_for_children": False},
        }
        original = adv.compute_advantage
        adv.compute_advantage = real_fn
        try:
            pipe = adv.apply_tir_advantages(
                b, adv_estimator="grpo", gamma=1.0, lam=1.0, num_repeat=2,
                norm_adv_by_std_in_grpo=True, config=algo_cfg,
            )
            legacy = adv.legacy_apply_tir_advantages(
                b, adv_estimator="grpo", gamma=1.0, lam=1.0, num_repeat=2,
                norm_adv_by_std_in_grpo=True, config=algo_cfg,
            )
        finally:
            adv.compute_advantage = original
        _assert_equal_batches(self, pipe, legacy)

    def test_pipeline_noop_counter_bumps_on_missing_columns(self):
        """Operator missing-column degrade bumps the noop counter (registry contract).

        rae_verdict without verdict_list degrades to no-op and counts — and the
        pipeline output then equals the passthrough batch (grpo base, no ops
        applied)."""
        from rl.hooks import advantage as adv

        before = adv.pipeline_noop_count()
        b = _mixed_batch()  # no verdict_list
        out = adv.apply_tir_advantages(
            b, adv_estimator="grpo", gamma=1.0, lam=1.0, num_repeat=2,
            norm_adv_by_std_in_grpo=True,
            config={"tir_algo": "rae", "tir": {}},
        )
        self.assertGreater(adv.pipeline_noop_count(), before)
        snap = adv.reset_pipeline_noop_counter()
        self.assertGreater(snap["count"], 0)
        self.assertTrue(any("verdict" in r for r in snap["reasons"]))
        self.assertEqual(adv.pipeline_noop_count(), 0)  # drained


if __name__ == "__main__":
    unittest.main()
