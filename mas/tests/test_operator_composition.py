"""RL3 composition rules: illegal operator stacks / bases are rejected.

Mirrors RL_design §3.1.4 (R1/R5 + registry membership + duplicate check) via
``validate_pipeline_spec`` on AdvantagePipelineSpec, plus the preset→spec
expansion table assertions (expansion products must exactly match the frozen
snapshot table — RL_design §3.1.6/§3.4.4 is the single authority).
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


class TestIllegalComposition(unittest.TestCase):
    """validate_pipeline_spec rejects duplicate ops / R1 XOR / R5 / stale ids."""

    def _spec(self, base, ops):
        from rl.loss import AdvantageOperatorSpec, AdvantagePipelineSpec

        return AdvantagePipelineSpec(
            base=base,
            operators=[AdvantageOperatorSpec(id=o, params=dict(p or {})) for o, p in ops],
        )

    def _resolve(self, spec):
        from rl.hooks.advantage import validate_pipeline_spec

        return validate_pipeline_spec(spec)

    def test_duplicate_operators_rejected(self):
        spec = self._spec("grpo", [("zero_prefix", None), ("zero_prefix", None)])
        with self.assertRaises(ValueError, msg="operators must be unique in the stack"):
            self._resolve(spec)

    def test_r1_verdict_and_full_tgt_mutually_exclusive(self):
        """R1: rae_verdict XOR rae_full_tgt — matches the current fork at
        advantage.py (legacy rae picks one of the two, never both)."""
        spec = self._spec("grpo", [("rae_verdict", None), ("rae_full_tgt", None)])
        with self.assertRaises(ValueError, msg="R1 XOR violated"):
            self._resolve(spec)

    def test_r1_variant_order_irrelevant(self):
        spec = self._spec("grpo", [("rae_full_tgt", None), ("rae_verdict", {"p_plus": 0.9})])
        with self.assertRaises(ValueError, msg="R1 XOR violated (reversed order)"):
            self._resolve(spec)

    def test_r5_full_tgt_with_appo_credit_rejected(self):
        spec = self._spec("grpo", [("appo_credit", None), ("rae_full_tgt", None)])
        with self.assertRaises(ValueError, msg="R5 forbidden stack"):
            self._resolve(spec)

    def test_unknown_operator_id_rejected(self):
        spec = self._spec("grpo", [("mu_ref_renorm", None)])
        with self.assertRaises(ValueError, msg="operator not in the five-value registry"):
            self._resolve(spec)

    def test_base_outside_enum_rejected(self):
        spec = self._spec("flow_grpo", [])
        with self.assertRaises(ValueError, msg="flow_grpo is NOT a base — snapshot not base"):
            self._resolve(spec)
        spec = self._spec("Gigantic", [])
        with self.assertRaises(ValueError, msg="base outside {grpo,igpo,gigpo}"):
            self._resolve(spec)

    def test_duplicate_detection_reports_the_offending_id(self):
        from rl.hooks.advantage import validate_pipeline_spec
        from rl.loss import AdvantageOperatorSpec, AdvantagePipelineSpec

        spec = AdvantagePipelineSpec(
            base="grpo",
            operators=[
                AdvantageOperatorSpec(id="aepo_entropy"),
                AdvantageOperatorSpec(id="aepo_entropy", params={"coef": 0.2}),
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline_spec(spec)
        self.assertIn("aepo_entropy", str(ctx.exception))

    def test_r4_redundant_but_legal_stack_passes(self):
        """R4: zero_prefix + appo_credit is legal-but-redundant (allowed, warned
        by convention at the config layer — the validator must NOT reject it)."""
        spec = self._spec("grpo", [("zero_prefix", None), ("appo_credit", None)])
        self._resolve(spec)  # must not raise

    def test_r6_igpo_with_appo_credit_allowed(self):
        """R6: base igpo + appo_credit (new-experiment surface) is allowed."""
        spec = self._spec("igpo", [("appo_credit", None)])
        self._resolve(spec)

    def test_gigpo_with_operator_stack_allowed(self):
        spec = self._spec("gigpo", [("zero_prefix", None), ("aepo_entropy", None)])
        self._resolve(spec)


class TestExpansionTable(unittest.TestCase):
    """v1 preset → pipeline expansion matches the frozen snapshot table exactly."""

    EXPECTED = {
        "grpo": ("grpo", ["zero_prefix"]),
        "arpo": ("grpo", []),
        "appo": ("grpo", ["appo_credit"]),
        "aepo": ("grpo", ["aepo_entropy"]),
        "igpo": ("igpo", []),
        "gigpo": ("gigpo", []),
        "rae": ("grpo", ["zero_prefix", "rae_verdict"]),
        "flow_grpo": ("grpo", []),
    }

    def test_all_eight_presets_expand_to_frozen_products(self):
        from rl.hooks.advantage import resolve_pipeline_spec

        for algo, (base, ops) in self.EXPECTED.items():
            spec = resolve_pipeline_spec({"tir_algo": algo, "tir": {}})
            self.assertEqual(spec.base, base, f"{algo}: base mismatch")
            self.assertEqual(
                [o.id for o in spec.operators], ops,
                f"{algo}: operator stack mismatch",
            )

    def test_rae_full_tgt_flag_swaps_trailing_operator(self):
        """tir rae_full_tgt=True swaps rae_verdict→rae_full_tgt in place
        (mirrors the legacy fork at advantage.py:92-99 — swap, not append)."""
        from rl.hooks.advantage import resolve_pipeline_spec

        spec = resolve_pipeline_spec({"tir_algo": "rae", "tir": {"rae_full_tgt": True}})
        self.assertEqual([o.id for o in spec.operators], ["zero_prefix", "rae_full_tgt"])

    def test_unknown_algo_falls_back_to_grpo_passthrough(self):
        """Legacy dispatch ended with a bare ``return batch`` for unknown algo;
        the resolver must keep that shape (grpo base, no operators)."""
        from rl.hooks.advantage import resolve_pipeline_spec

        spec = resolve_pipeline_spec({"tir_algo": "future_algo", "tir": {}})
        self.assertEqual(spec.base, "grpo")
        self.assertEqual([o.id for o in spec.operators], [])

    def test_registry_surface_frozen(self):
        """Five-value operator registry + three-value base enum are the contract."""
        from rl.hooks.advantage import BASE_CONSUMED_COLUMNS, OPERATOR_CONSUMED_COLUMNS
        from rl.loss import ADVANTAGE_BASES, ADVANTAGE_OPERATORS

        self.assertEqual(set(ADVANTAGE_BASES), {"grpo", "igpo", "gigpo"})
        self.assertEqual(
            set(ADVANTAGE_OPERATORS),
            {"zero_prefix", "appo_credit", "aepo_entropy", "rae_verdict", "rae_full_tgt"},
        )
        self.assertEqual(set(OPERATOR_CONSUMED_COLUMNS), set(ADVANTAGE_OPERATORS))
        self.assertEqual(set(BASE_CONSUMED_COLUMNS), set(ADVANTAGE_BASES))


if __name__ == "__main__":
    unittest.main()
