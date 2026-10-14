"""RL5 resolved_algo audit — launch.json snapshot read-back & legacy compat.

RL_design §3.5.1/P4 acceptance: the launch.json gains a ``resolved_algo``
field (semantic serialization of the expanded pipeline: base/operators/
reward components/loss mode), expanded after preflight and before the
subprocess so signature and effective-rl.yaml are same-source, same-moment.
Consumers must tolerate old runs whose launch.json predates the field
(TrainingHistory-style missing-field fallback, ADR-6: resolved_algo and
preflight_revision stay separate layers).
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


def _resolved_algo(rl):
    from science_infra.control.training import _resolved_algo_payload

    return _resolved_algo_payload(rl)


class TestResolvedAlgoPayload(unittest.TestCase):
    """The serialization block itself."""

    def test_v1_preset_expands_into_payload(self):
        payload = _resolved_algo({"algo": "appo"})
        self.assertEqual(payload["preset"], "appo")
        self.assertEqual(payload["advantage"]["base"], "grpo")
        self.assertEqual(
            [op["id"] for op in payload["advantage"]["operators"]],
            ["appo_credit"],
        )
        self.assertEqual(payload["loss_mode"], "vanilla")  # absent tir → vanilla
        self.assertFalse(payload["v2_composed"])

    def test_eight_presets_cover_the_snapshot_table(self):
        expected = {
            "grpo": ("grpo", ["zero_prefix"]),
            "arpo": ("grpo", []),
            "appo": ("grpo", ["appo_credit"]),
            "aepo": ("grpo", ["aepo_entropy"]),
            "igpo": ("igpo", []),
            "gigpo": ("gigpo", []),
            "rae": ("grpo", ["zero_prefix", "rae_verdict"]),
            "flow_grpo": ("grpo", []),
        }
        for algo, (base, ops) in expected.items():
            payload = _resolved_algo({"algo": algo})
            self.assertEqual(payload["preset"], algo)
            self.assertEqual(payload["advantage"]["base"], base, algo)
            self.assertEqual(
                [op["id"] for op in payload["advantage"]["operators"]], ops, algo
            )

    def test_reward_components_fall_back_to_tir_derivation(self):
        """plan.rl is the pre-overlay raw yaml; without an algorithm.reward
        section the payload derives the RL4 mirror spec from the tir dict
        (multi_tool_bonus 0.1 default vs user override)."""
        payload = _resolved_algo(
            {"algo": "grpo", "algorithm": {"tir": {"multi_tool_bonus": 0.2}}}
        )
        comps = {c["id"]: c for c in payload["reward"]["components"]}
        self.assertAlmostEqual(comps["multi_tool_bonus"]["weight"], 0.2)
        self.assertTrue(comps["multi_tool_bonus"]["enabled"])
        self.assertTrue(comps["accuracy_reward"]["enabled"])  # 不可关主奖励
        self.assertFalse(comps["cost_penalty"]["enabled"])  # 预留位 born disabled

        default_payload = _resolved_algo({"algo": "grpo"})
        default_comps = {c["id"]: c for c in default_payload["reward"]["components"]}
        self.assertAlmostEqual(default_comps["multi_tool_bonus"]["weight"], 0.1)

    def test_v2_composed_flag_reflects_advantage_section(self):
        payload = _resolved_algo(
            {
                "algo": "grpo",
                "algorithm": {
                    "advantage": {"base": "gigpo", "operators": [{"id": "zero_prefix"}]}
                },
            }
        )
        self.assertTrue(payload["v2_composed"])

    def test_unknown_algo_yields_none_legacy_defensive(self):
        self.assertIsNone(_resolved_algo({"algo": "no_such_algo"}))
        self.assertIsNone(_resolved_algo("not a dict"))
        # tir_algo carrier also resolves when top-level algo is absent
        payload = _resolved_algo({"algorithm": {"tir_algo": "gigpo"}})
        self.assertEqual(payload["preset"], "gigpo")


class TestLaunchJsonRoundTrip(unittest.TestCase):
    """launch.json carries resolved_algo alongside every pre-existing field;
    json round-trip keeps both, and legacy readers never break."""

    BASE_LAUNCH = {
        "schema_version": 1,
        "run_id": "abc123def456",
        "request_id": "req-1",
        "experiment_id": "agentflow_4algo",
        "git_commit": "7186f7e",
        "created_at": 1760422800.0,
        "preflight_revision": "deadbeef" * 8,
        "algorithm": "appo",
        "algorithm_source": "rl",
        "profile": "a800_2gpu",
        "gpu_ids": [0, 1],
        "model": {"source": "legacy", "model_path": "/LLM/Qwen3-4B"},
        "argv": ["python", "train_tir_agent.py"],
        "snapshot": {"rl": "effective-rl.yaml", "launch": "launch.json"},
    }

    def test_new_launch_has_resolved_algo_and_keeps_legacy_fields(self):
        launch = {
            **self.BASE_LAUNCH,
            "resolved_algo": _resolved_algo({"algo": "appo"}),
        }
        encoded = json.dumps(launch, ensure_ascii=False, indent=2)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["resolved_algo"]["advantage"]["base"], "grpo")
        self.assertEqual(
            [op["id"] for op in decoded["resolved_algo"]["advantage"]["operators"]],
            ["appo_credit"],
        )
        for legacy_key, legacy_value in self.BASE_LAUNCH.items():
            self.assertEqual(decoded[legacy_key], legacy_value, legacy_key)

    def test_legacy_launch_without_resolved_algo_renders_with_fallback(self):
        """TrainingHistory-style consumer: old runs predate the field. The
        render path falls back to meta.algo (挂 TrainingConsole 现状) and
        must not raise — the consumer contract is ``launch.get`` everywhere."""
        legacy_launch = dict(self.BASE_LAUNCH)  # no resolved_algo key
        render = lambda launch: launch.get("resolved_algo") or launch.get("algorithm") or "unknown"
        self.assertEqual(render(legacy_launch), "appo")

        with_resolved = {**self.BASE_LAUNCH, "resolved_algo": {"preset": "gigpo"}}
        self.assertEqual(render(with_resolved), {"preset": "gigpo"})

    def test_resolved_algo_and_preflight_revision_stay_separate(self):
        """ADR-6: preflight_revision answers “config inputs identical?”,
        resolved_algo answers “expanded semantics identical?” — never merged."""
        launch = {
            **self.BASE_LAUNCH,
            "resolved_algo": _resolved_algo({"algo": "appo"}),
        }
        self.assertIsInstance(launch["preflight_revision"], str)
        self.assertIsInstance(launch["resolved_algo"], dict)
        self.assertNotIn("preflight_revision", launch["resolved_algo"])


class TestExpandSerializationStability(unittest.TestCase):
    """Same rl → byte-stable serialization (audit diffing depends on it)."""

    def test_payload_is_json_stable_across_calls(self):
        a = json.dumps(_resolved_algo({"algo": "rae"}), sort_keys=True, ensure_ascii=False)
        b = json.dumps(_resolved_algo({"algo": "rae"}), sort_keys=True, ensure_ascii=False)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
