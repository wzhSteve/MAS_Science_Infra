"""RL5 formal expand_preset API — frozen snapshot products & stable dumps.

``expand_preset(algo) -> AdvantagePipelineSpec`` (rl/loss.py) is the single
authority implementing the RL_design §3.4.4 preset snapshot table (post
RL0-errata §2.3 truth table). Every v1 ``algo`` single-select expansion —
overlay-side (apply_algo_overlay v2 section) and tensor-side
(resolve_pipeline_spec) — must consult exactly this table. These tests pin
the products against a frozen literal list and assert yaml/json dumps are
order-stable so audit signatures stay diffable across runs.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(MAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Frozen expected expansion (RL_design §3.4.4 snapshot table, verbatim).
FROZEN = {
    "grpo": ("grpo", ["zero_prefix"]),
    "arpo": ("grpo", []),
    "appo": ("grpo", ["appo_credit"]),
    "aepo": ("grpo", ["aepo_entropy"]),
    "igpo": ("igpo", []),
    "gigpo": ("gigpo", []),
    "rae": ("grpo", ["zero_prefix", "rae_verdict"]),
    "flow_grpo": ("grpo", []),
}


class TestExpandPresetProducts(unittest.TestCase):
    def _spec(self, algo):
        from rl.loss import expand_preset

        return expand_preset(algo)

    def test_eight_presets_match_frozen_products(self):
        for algo, (base, ops) in FROZEN.items():
            spec = self._spec(algo)
            self.assertEqual(spec.base, base, f"{algo}: base mismatch")
            self.assertEqual(
                [op.id for op in spec.operators], ops, f"{algo}: operator stack mismatch"
            )
            # params stay empty at expansion: values live in the tir dict
            # read at execution time (param chain preserved, single source).
            for op in spec.operators:
                self.assertEqual(op.params, {}, f"{algo}/{op.id}: params must stay empty")

    def test_operators_are_pydantic_operator_specs(self):
        from rl.loss import AdvantageOperatorSpec

        spec = self._spec("rae")
        for op in spec.operators:
            self.assertIsInstance(op, AdvantageOperatorSpec)

    def test_case_and_whitespace_normalization(self):
        self.assertEqual(self._spec(" GRPO ").base, "grpo")
        self.assertEqual(self._spec("Aepo").operators[0].id, "aepo_entropy")

    def test_unknown_preset_raises_v1(self):
        from rl.loss import expand_preset

        for bad in ("", "future_algo", "grpo_n", "mu_ref_renorm"):
            with self.assertRaises(ValueError, msg=f"unknown preset {bad!r} must raise"):
                expand_preset(bad)
        with self.assertRaises(ValueError, msg="None resolves to empty → unknown"):
            expand_preset(None or "")

    def test_expanded_spec_passes_combination_rules(self):
        """Snapshot royalty: every expanded product is legal by construction —
        the composition enforcer must accept all eight."""
        from rl.hooks.advantage import validate_pipeline_spec

        for algo in FROZEN:
            validate_pipeline_spec(self._spec(algo))

    def test_expansion_tables_have_single_physical_copy(self):
        """rl/loss.py PRESET_EXPANSION_* IS the authority; advantage.py's
        _PIPELINE_EXPANSION mirrors it as legacy-dispatch documentation.
        The two drifting apart would fork the snapshot — test forces sync."""
        from rl.hooks.advantage import (
            _PIPELINE_EXPANSION,
            _PIPELINE_EXPANSION_BASE,
        )
        from rl.loss import PRESET_EXPANSION_BASE, PRESET_EXPANSION_OPERATORS

        self.assertEqual(_PIPELINE_EXPANSION_BASE, PRESET_EXPANSION_BASE)
        self.assertEqual(_PIPELINE_EXPANSION, PRESET_EXPANSION_OPERATORS)

    def test_apply_algo_overlay_writes_expanded_section(self):
        """overlay-side expansion consults the same table (v1→v2 single
        direction lossless: expanded result IS the v2 form)."""
        from rl.hooks.overlay import apply_algo_overlay

        for algo, (base, ops) in FROZEN.items():
            cfg = apply_algo_overlay({}, algo)
            section = cfg["algorithm"]["advantage"]
            self.assertEqual(section["base"], base, algo)
            self.assertEqual([op["id"] for op in section["operators"]], ops, algo)

    def test_resolve_pipeline_spec_consults_the_same_table(self):
        """tensor-side expansion also routes through the formal API."""
        from rl.hooks.advantage import resolve_pipeline_spec

        for algo, (base, ops) in FROZEN.items():
            spec = resolve_pipeline_spec({"tir_algo": algo, "tir": {}})
            self.assertEqual(spec.base, base, algo)
            self.assertEqual([op.id for op in spec.operators], ops, algo)


class TestStableSerialization(unittest.TestCase):
    """yaml/json dumps must be order-stable for audit diffing."""

    def test_json_dump_stable(self):
        from rl.loss import expand_preset

        first = json.dumps(
            {algo: expand_preset(algo).model_dump() for algo in sorted(FROZEN)},
            sort_keys=True,
            ensure_ascii=False,
        )
        second = json.dumps(
            {algo: expand_preset(algo).model_dump() for algo in sorted(FROZEN)},
            sort_keys=True,
            ensure_ascii=False,
        )
        self.assertEqual(first, second)

    def test_yaml_dump_stable_sorting(self):
        import yaml

        from rl.loss import expand_preset

        first = yaml.safe_dump(
            {algo: expand_preset(algo).model_dump() for algo in sorted(FROZEN)},
            sort_keys=True,
            allow_unicode=True,
        )
        second = yaml.safe_dump(
            {algo: expand_preset(algo).model_dump() for algo in sorted(FROZEN)},
            sort_keys=True,
            allow_unicode=True,
        )
        self.assertEqual(first, second)
        parsed = yaml.safe_load(first)
        self.assertEqual(parsed["appo"]["operators"][0]["id"], "appo_credit")

    def test_spec_model_dump_round_trip_reconstructs(self):
        from rl.loss import AdvantagePipelineSpec, expand_preset

        spec = expand_preset("rae")
        rebuilt = AdvantagePipelineSpec.model_validate(spec.model_dump())
        self.assertEqual(rebuilt.base, spec.base)
        self.assertEqual([op.id for op in rebuilt.operators], [op.id for op in spec.operators])


if __name__ == "__main__":
    unittest.main()
