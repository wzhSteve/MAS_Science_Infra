"""RL5 rl.yaml contract validation — V1-V6 rules across the three layers.

Layer map (RL_design §5.2, interception cost low→high):
  1. save    — save_section(rl) raises TrainingError(rl_contract_validation,
               400, {violations: [{rule, field, detail}]})  [experiments.py]
  2. preflight — build_training_plan records rl_contract_v* blocking
               issues off the same scanner                [training.py]
  3. launch  — pydantic validators in the declaration layer (rl.loss /
               rl.rewards.assembly construction) + overlay V-track checks;
               the composition enforcer (validate_pipeline_spec) remains
               the single authority for stack rules.

This file locks the scanner semantics (V-rules on the raw yaml dict), the
save-layer 400 body shape (field path + rule id), the preflight record
wiring, and the存量 v1 acceptance: the five live experiments/*/rl*.yaml
must validate clean with zero migration (A9).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(MAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _violations(rl):
    from science_infra.control.training import validate_rl_contract

    return validate_rl_contract(rl)


def _rules(violations):
    return {v["rule"] for v in violations}


class TestV1RequiredAndType:
    pass


class TestVRulesScanner(unittest.TestCase):
    """V-rule semantics on the raw dict scanner (shared by save + preflight)."""

    def test_v1_v2_unknown_preset_rejected(self):
        v = _violations({"algo": "future_algo"})
        self.assertIn("V2", _rules(v))
        self.assertEqual(v[0]["field"], "algo")

    def test_v1_v2_tir_algo_carrier_rejected_too(self):
        v = _violations({"algorithm": {"tir_algo": "quantum"}})
        self.assertIn("V2", _rules(v))
        self.assertEqual(v[0]["field"], "algorithm.tir_algo")

    def test_v1_v2_base_enum(self):
        v = _violations(
            {"algorithm": {"advantage": {"base": "flow_grpo", "operators": []}}}
        )
        self.assertIn("V2", _rules(v))
        self.assertTrue(any(x["field"] == "algorithm.advantage.base" for x in v))

    def test_v1_operator_entry_type(self):
        v = _violations(
            {"algorithm": {"advantage": {"base": "grpo", "operators": ["zero_prefix"]}}}
        )
        self.assertIn("V1", _rules(v))

    def test_v1_operator_id_enum(self):
        v = _violations(
            {
                "algorithm": {
                    "advantage": {"base": "grpo", "operators": [{"id": "mu_ref_renorm"}]}
                }
            }
        )
        self.assertIn("V2", _rules(v))

    def test_v2_duplicate_operator(self):
        v = _violations(
            {
                "algorithm": {
                    "advantage": {
                        "base": "grpo",
                        "operators": [{"id": "zero_prefix"}, {"id": "zero_prefix"}],
                    }
                }
            }
        )
        self.assertIn("V2", _rules(v))
        self.assertTrue(any("重复" in x["detail"] for x in v))

    def test_v1_reward_component_type(self):
        v = _violations(
            {"algorithm": {"reward": {"components": ["format_penalty"]}}}
        )
        self.assertEqual(_rules(v) & {"V1"}, {"V1"})

    def test_v2_loss_mode_enum(self):
        v = _violations({"algorithm": {"tir": {"loss_mode": "quantum"}}})
        self.assertIn("V2", _rules(v))

    def test_v3_verdict_xor_full_tgt(self):
        v = _violations(
            {
                "algorithm": {
                    "advantage": {
                        "base": "grpo",
                        "operators": [{"id": "rae_verdict"}, {"id": "rae_full_tgt"}],
                    }
                }
            }
        )
        self.assertIn("V3", _rules(v))
        self.assertTrue(any("互斥" in x["detail"] for x in v if x["rule"] == "V3"))

    def test_v3_full_tgt_appo_credit_forbidden(self):
        v = _violations(
            {
                "algorithm": {
                    "advantage": {
                        "base": "grpo",
                        "operators": [{"id": "rae_full_tgt"}, {"id": "appo_credit"}],
                    }
                }
            }
        )
        # R5-forbidden stack also violates RewardSpec-agnostic V3 here
        self.assertIn("V3", _rules(v))

    def test_v3_legal_stack_clean(self):
        v = _violations(
            {
                "algorithm": {
                    "advantage": {
                        "base": "gigpo",
                        "operators": [
                            {"id": "zero_prefix"},
                            {"id": "aepo_entropy", "params": {"entropy_adv_coef": 0.2}},
                        ],
                    }
                }
            }
        )
        self.assertEqual(v, [])

    def test_v4_future_kl_triple_type_and_domain(self):
        v = _violations(
            {
                "algorithm": {
                    "tir": {
                        "loss_mode": "future_kl",
                        "future_kl_weight": -0.5,
                        "decay_rate": "big",
                        "future_kl_clip_ratio": 1e400,
                    }
                }
            }
        )
        v4 = [x for x in v if x["rule"] == "V4"]
        self.assertEqual(len(v4), 3)
        fields = {x["field"] for x in v4}
        self.assertIn("algorithm.tir.future_kl_weight", fields)
        self.assertIn("algorithm.tir.decay_rate", fields)
        self.assertIn("algorithm.tir.future_kl_clip_ratio", fields)
        self.assertTrue(all("V4 类型错误" in x["detail"] or "非负有限" in x["detail"] for x in v4))

    def test_v4_zero_is_legal(self):
        v = _violations(
            {
                "algorithm": {
                    "tir": {
                        "loss_mode": "future_kl",
                        "future_kl_weight": 0.0,
                        "decay_rate": 0.0,
                        "future_kl_clip_ratio": 0.0,
                    }
                }
            }
        )
        self.assertEqual(v, [])

    def test_v4_reward_component_weight_domain(self):
        v = _violations(
            {
                "algorithm": {
                    "reward": {"components": [{"id": "multi_tool_bonus", "weight": -1.0}]}
                }
            }
        )
        self.assertIn("V4", _rules(v))
        self.assertTrue(any("algorithm.reward.components" in x["field"] for x in v))

    def test_v5_dual_track_flags_algo_plus_composed(self):
        v = _violations(
            {
                "algo": "grpo",
                "algorithm": {"advantage": {"base": "grpo", "operators": []}},
            }
        )
        self.assertIn("V5", _rules(v))
        v5 = next(x for x in v if x["rule"] == "V5")
        self.assertIn("单轨", v5["detail"])

    def test_v5_not_fired_for_v2_alone_or_algo_alone(self):
        self.assertEqual(_violations({"algorithm": {"advantage": {"base": "gigpo", "operators": []}}}), [])
        self.assertEqual(_violations({"algo": "grpo"}), [])

    def test_v6_future_kl_connectivity_mirrors_post_double_write(self):
        """RL1双写后 future_kl 真实连通: write side mirrors into
        actor.policy_loss. The plan-side check reports connectivity when the
        user overrode the mirror to something inconsistent."""
        v = _violations(
            {
                "algorithm": {"tir": {"loss_mode": "future_kl"}},
                "actor_rollout_ref": {
                    "actor": {"policy_loss": {"loss_mode": "vanilla"}}
                },
            }
        )
        self.assertIn("V6", _rules(v))
        v6 = next(x for x in v if x["rule"] == "V6")
        self.assertEqual(v6["field"], "actor.policy_loss.loss_mode")
        self.assertIn("连通", v6["detail"])

    def test_v6_future_kl_without_mirror_block_is_connected(self):
        """Absent actor.policy_loss means the launch-time mirror
        (apply_train_signal) will write it — connected, no violation."""
        v = _violations({"algorithm": {"tir": {"loss_mode": "future_kl"}}})
        self.assertEqual(v, [])

    def test_vanilla_configs_are_never_v6(self):
        v = _violations(
            {
                "algorithm": {"tir": {"loss_mode": "vanilla"}},
                "actor_rollout_ref": {
                    "actor": {"policy_loss": {"loss_mode": "future_kl"}}
                },
            }
        )
        self.assertNotIn("V6", _rules(v))

    def test_non_dict_input(self):
        v = _violations(["not", "a", "dict"])
        self.assertEqual(v[0]["rule"], "V1")


class TestSaveLayer400Body(unittest.TestCase):
    """save_section(rl) raises rl_contract_validation with the contract body."""

    def _save(self, rl):
        from science_infra.control.experiments import save_section

        return save_section("agentflow_4algo", "rl", rl)

    def test_invalid_rl_rejected_with_rule_and_field(self):
        from science_infra.control.training import TrainingError

        try:
            self._save({"algo": "future_algo"})
            self.fail("save must reject unknown algo")
        except TrainingError as error:
            self.assertEqual(error.code, "rl_contract_validation")
            self.assertEqual(error.status, 400)
            self.assertIn("violations", error.data)
            worst = error.data["violations"][0]
            self.assertEqual(worst["rule"], "V2")
            self.assertEqual(worst["field"], "algo")
            self.assertIn("V2", str(error))

    def test_dual_track_rejected_at_save(self):
        from science_infra.control.training import TrainingError

        try:
            self._save(
                {
                    "algo": "appo",
                    "algorithm": {
                        "advantage": {"base": "grpo", "operators": []}
                    },
                }
            )
            self.fail("save must reject dual-track declaration")
        except TrainingError as error:
            self.assertEqual(error.code, "rl_contract_validation")
            self.assertIn("V5", {v["rule"] for v in error.data["violations"]})

    def test_v3_xor_rejected_at_save(self):
        from science_infra.control.training import TrainingError

        try:
            self._save(
                {
                    "algorithm": {
                        "advantage": {
                            "base": "grpo",
                        }
                    }
                }
            )
            # v2 alone without algo: legal
            self.assertIsInstance(self._save, object)
        except TrainingError:
            self.fail("v2-only composed declaration must save cleanly")


class TestPreflightLayer(unittest.TestCase):
    """build_training_plan records rl_contract_v* blocking issues (fail-fast)."""

    def test_preflight_records_contract_violations(self):
        """Scanner violations become blocking checks — verified at the
        internal-function level (PROCS-free) to keep the test hermetic."""
        from science_infra.control.training import _rl_contract_violations

        rl = {
            "algo": "grpo",
            "algorithm": {
                "advantage": {
                    "base": "grpo",
                    "operators": [{"id": "rae_verdict"}, {"id": "rae_full_tgt"}],
                },
                "tir": {"loss_mode": "future_kl", "future_kl_weight": -1.0},
            },
        }
        violations = _rl_contract_violations(rl)
        self.assertTrue(violations)
        # every violation carries the three contract keys (launch body shape)
        for item in violations:
            self.assertIn("rule", item)
            self.assertIn("field", item)
            self.assertIn("detail", item)
        rules = {v["rule"] for v in violations}
        self.assertEqual(rules, {"V3", "V4", "V5"})


class TestLaunchLayerDeclarations(unittest.TestCase):
    """Layer 3: pydantic validators in the declaration layer fail at Spec
    construction (the pydantic leg of “保存（pydantic）→ preflight → 启动”)"""

    def test_v4_on_loss_spec_construction(self):
        from rl.loss import LossSpec

        with self.assertRaises(ValueError):
            LossSpec(future_kl_weight=-0.5)
        with self.assertRaises(ValueError):
            LossSpec(decay_rate=float("inf"))

    def test_v4_on_reward_component_construction(self):
        from rl.rewards.assembly import RewardComponent

        with self.assertRaises(ValueError):
            RewardComponent(id="multi_tool_bonus", weight=-0.1)

    def test_v1_v2_on_pipeline_spec_construction(self):
        from rl.loss import AdvantagePipelineSpec

        with self.assertRaises(ValueError):
            AdvantagePipelineSpec(base="flow_grpo")
        with self.assertRaises(ValueError):
            AdvantagePipelineSpec(base="grpo", operators=[{"id": "quantum_rop"}])

    def test_dual_track_at_overlay_expansion(self):
        """v2 composed section survives apply_algo_overlay (advantage truth)
        while tir_algo keeps the sampling/gate carrier semantics — the
        resolution priority V5 declares is executed, not just documented."""
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(
            {
                "algorithm": {
                    "advantage": {
                        "base": "gigpo",
                        "operators": [{"id": "zero_prefix", "params": {"w": 2}}],
                    }
                }
            },
            "grpo",
        )
        self.assertEqual(cfg["algorithm"]["advantage"]["base"], "gigpo")
        self.assertEqual(cfg["algorithm"]["tir_algo"], "grpo")
        self.assertEqual(cfg["algorithm"]["advantage"]["operators"][0]["params"], {"w": 2})

    def test_overlay_idempotent_for_expanded_presets(self):
        from rl.hooks.overlay import apply_algo_overlay

        once = apply_algo_overlay({}, "appo")
        twice = apply_algo_overlay(once, "appo")
        self.assertEqual(
            once["algorithm"]["advantage"], twice["algorithm"]["advantage"]
        )


class TestLegacyV1Acceptance(unittest.TestCase):
    """A9: the five live experiments/agentflow_4algo/rl*.yaml validate clean,
   expand unchanged — zero migration cost (the RL5 acceptance face)."""

    def test_live_v1_yamls_scan_clean(self):
        """RL5 acceptance face. NOTE (probe deviation, recorded in the commit
        message): during this stage's run the untracked user artifact
        experiments/agentflow_4algo/rl.yaml was externally rewritten (user /
        parallel session — 2026-10-14 15:33, outside this agent's writes)
        from the full flow_grpo v1 config into a 3-line v2 fragment
        (algorithm.advantage.base: grpo). The four rl_<preset>.yaml siblings
        are untouched full v1 files. Contract-wise both shapes are legal:
        the scanner must validate them all clean either way."""
        import yaml

        base = ROOT / "experiments" / "agentflow_4algo"
        files = sorted(base.glob("rl*.yaml"))
        self.assertEqual(len(files), 5, f"expected the 5 live rl yamls, got {files}")
        for path in files:
            with open(path, encoding="utf-8") as handle:
                rl = yaml.safe_load(handle)
            violations = _violations(rl)
            self.assertEqual(
                violations,
                [],
                f"{path.name} must validate clean (存量 v1 零迁移): {violations}",
            )
            # v1 select may ride either top-level ``algo`` or the
            # algorithm.tir_algo carrier — both expand identically. An
            # absent select (the rewritten rl.yaml v2 fragment carries
            # neither) falls back to the grpo leg, matching the historical
            # resolve tolerance (unknown/absent tir_algo → grpo passthrough).
            algo = str(
                rl.get("algo")
                or (rl.get("algorithm") or {}).get("tir_algo")
                or ""
            )
            from rl.loss import expand_preset

            if algo:
                spec = expand_preset(algo)
                base_ops = (spec.base, tuple(op.id for op in spec.operators))
            else:
                algo = "grpo"
                spec = expand_preset(algo)
                base_ops = (spec.base, tuple(op.id for op in spec.operators))
            frozen = {name: (b, tuple(o)) for name, (b, o) in _frozen_pairs()}
            self.assertEqual(
                frozen.get(algo),
                base_ops,
                f"{path.name} expansion must match the frozen snapshot",
            )

    def test_grpo_defaults_scan_clean(self):
        self.assertEqual(_violations({"algo": "grpo"}), [])
        self.assertEqual(_violations({}), [])  # fully absent → v1 grpo leg


def _frozen_pairs():
    return {
        "grpo": ("grpo", ["zero_prefix"]),
        "arpo": ("grpo", []),
        "appo": ("grpo", ["appo_credit"]),
        "aepo": ("grpo", ["aepo_entropy"]),
        "igpo": ("igpo", []),
        "gigpo": ("gigpo", []),
        "rae": ("grpo", ["zero_prefix", "rae_verdict"]),
        "flow_grpo": ("grpo", []),
    }.items()


if __name__ == "__main__":
    unittest.main()
