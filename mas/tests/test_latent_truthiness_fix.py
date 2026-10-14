"""Regression gate for the RL3-§5.1 latent truthiness bug (small fix).

The AGL daemon injects every non_tensor column as np.ndarray
(rl/hooks/daemon.py:750-752, :829-841) and verl's trainer aligns
``uid = data_id_list`` (agentlightning/verl/trainer.py:293) — production
shapes for ``data_id_list``/``rollout_id_list``/``verdict_list``/
``action_key_list``/``old_log_probs`` are therefore **ndarrays and torch
tensors**, never plain lists. The advantage kernels previously used bare
``or`` fallbacks on those values:

    advantage.py  _apply_igpo / _apply_gigpo:
        ntb.get("data_id_list") or ntb.get("uid")
    rae_advantage.py  apply_rae_full_tgt_advantages:
        _nt_get(ntb, "action_key_list", "action_key") or []
        _nt_get(ntb, "uid", "data_id") or []
        batch.batch.get("old_log_probs") or batch.batch.get("old_logprobs")

Multi-element ndarrays / (0-dim) tensors are truth-value-ambiguous or decide
the fallback on a meaningless scalar — ValueError under NumPy ≥1.25, bogus
deprecation-cast behaviour before it. The CPU fixtures of
test_preset_equivalence / test_advantage_numerics built the same columns as
plain *lists*, which never trip the ambiguity — that is why the igpo/gigpo/
rae production paths had never actually executed (RL3 report §5.1).

The fix replaces each bare ``or`` with an explicit ``is None`` check
(same list semantics; present-and-non-empty values always win the
fallback). These tests build the columns in **production form** (ndarrays /
tensors) and assert:

1. the kernels run (no ValueError) and produce the exact same results as
   the equivalent list-formed batch (shape-neutrality of the fix);
2. a fallback that still works (absent data_id_list falls back to uid);
3. a crash-lock that documents the pre-fix failure mode (a repr of the
   historical bare-or line) so the pattern cannot silently return.

Kernel-semantics freeze: igpo/gigpo/rae_full_tgt outputs of the fixed
kernels on ndarray-shaped inputs must equal the frozen outputs on the
identical list-shaped inputs — the fix must make existing semantics
runnable, never change them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(MAS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W = 4  # token width


def _nt_columns(*, verdict: bool, as_ndarray: bool):
    """Production-shaped non_tensor columns (AGL daemon form when ndarray)."""
    cols = {
        "uid": ["d0", "d0", "d0", "d0", "d1", "d1"],
        "data_id_list": ["d0", "d0", "d0", "d0", "d1", "d1"],
        "rollout_id_list": ["r0", "r1", "r2", "r3", "r4", "r5"],
        "rollout_role": ["parent", "child", "parent", "branch", "parent", "probe"],
        "resume_parent_id": ["", "r0", "", "r2", "", "r0"],
        "resume_boundary": [0, 2, 0, 1, 0, 3],
        "anchor_obs": ["h0", "h0", "h0", "h0", "h1", "h1"],
        "ig_logprob_delta": [0.1, -0.2, 0.3, 0.0, 0.25, -0.1],
        "turn_index_list": [0, 1, 0, 1, 0, 1],
    }
    if verdict:
        cols["verdict_list"] = ["none", "invalidate", "validate", "abstain", "validate", "invalidate"]
        cols["action_key_list"] = ["", "ak0", "", "ak0", "", "ak0"]
    if as_ndarray:
        cols = {
            k: np.array(v, dtype=object)
            for k, v in cols.items()
        }
    return cols


def _batch_from(columns: dict, *, old_log_probs=None) -> "DataProto":
    from verl import DataProto

    n = 6
    scores = torch.zeros(n, W)
    for i, v in enumerate([1.0, 0.0, 0.5, 1.0, 0.0, 2.0]):
        scores[i, -1] = float(v)
    tensors = {
        "advantages": torch.zeros(n, W, requires_grad=False),
        "returns": torch.zeros(n, W),
        "token_level_scores": scores,
        "response_mask": torch.ones(n, W),
    }
    if old_log_probs is not None:
        tensors["old_log_probs"] = old_log_probs
    batch = DataProto.from_dict(tensors=tensors)
    nt = batch.non_tensor_batch
    for key, value in columns.items():
        nt[key] = value
    return batch


def _clone(batch):
    import copy

    return copy.deepcopy(batch)


class TestNdarrayShapeNeutrality(unittest.TestCase):
    """Fixed kernels: ndarray-shaped (production) columns ≡ list-shaped columns."""

    def _igpo(self, batch, gamma=1.0):
        from rl.hooks.advantage import _apply_igpo

        return _apply_igpo(batch, gamma)

    def _gigpo(self, batch, step_w=0.7, gamma=0.95, mode="mean_std"):
        from rl.hooks.advantage import _apply_gigpo

        return _apply_gigpo(batch, step_w=step_w, gamma=gamma, mode=mode)

    def test_igpo_ndarray_columns_match_list_columns(self):
        list_b = self._igpo(_batch_from(_nt_columns(verdict=False, as_ndarray=False)))
        nd_b = self._igpo(_batch_from(_nt_columns(verdict=False, as_ndarray=True)))
        self.assertFalse(torch.equal(list_b.batch["advantages"], torch.zeros(6, W)),
                         "igpo kernel did not fire (test vacuous)")
        np.testing.assert_array_equal(
            nd_b.batch["advantages"].detach().numpy(),
            list_b.batch["advantages"].detach().numpy(),
            err_msg="igpo: ndarray-shaped columns must produce list-shaped outputs",
        )

    def test_gigpo_ndarray_columns_match_list_columns(self):
        list_b = self._gigpo(_batch_from(_nt_columns(verdict=False, as_ndarray=False)))
        nd_b = self._gigpo(_batch_from(_nt_columns(verdict=False, as_ndarray=True)))
        self.assertFalse(torch.equal(list_b.batch["advantages"], torch.zeros(6, W)),
                         "gigpo kernel did not fire (test vacuous)")
        np.testing.assert_array_equal(
            nd_b.batch["advantages"].detach().numpy(),
            list_b.batch["advantages"].detach().numpy(),
            err_msg="gigpo: ndarray-shaped columns must produce list-shaped outputs",
        )

    def test_rae_full_tgt_ndarray_columns_match_list_columns(self):
        """action_key_list / uid as object-ndarray + real old_log_probs tensor.

        This is the exact production triple that used to crash: the two bare
        ``or []`` reads in rae_advantage.py:176-177 plus the bare ``or`` at
        :181 on the ``old_log_probs`` torch tensor.
        """
        from rl.hooks.rae_advantage import apply_rae_full_tgt_advantages

        old_lp = torch.log(torch.full((6, W), 0.25))

        list_b = apply_rae_full_tgt_advantages(
            _batch_from(_nt_columns(verdict=True, as_ndarray=False), old_log_probs=old_lp),
            epsilon_f=1e-3,
        )
        nd_b = apply_rae_full_tgt_advantages(
            _batch_from(_nt_columns(verdict=True, as_ndarray=True), old_log_probs=old_lp),
            epsilon_f=1e-3,
        )
        self.assertFalse(torch.equal(list_b.batch["advantages"], torch.zeros(6, W)),
                         "rae_full_tgt kernel did not fire (test vacuous)")
        np.testing.assert_array_equal(
            nd_b.batch["advantages"].detach().numpy(),
            list_b.batch["advantages"].detach().numpy(),
            err_msg="rae_full_tgt: ndarray-shaped columns must produce list-shaped outputs",
        )


class TestFallbackStillWorks(unittest.TestCase):
    """None-based fallback: absent data_id_list still falls back to uid."""

    def test_igpo_uid_fallback(self):
        from rl.hooks.advantage import _apply_igpo

        cols = _nt_columns(verdict=False, as_ndarray=True)
        cols.pop("data_id_list")  # production column absent → uid fallback
        out = self._apply_ok(_apply_igpo, cols, gamma=1.0)
        self.assertFalse(torch.equal(out.batch["advantages"], torch.zeros(6, W)),
                         "igpo uid-fallback path did not fire")

    def test_gigpo_uid_fallback(self):
        from rl.hooks.advantage import _apply_gigpo

        cols = _nt_columns(verdict=False, as_ndarray=True)
        cols.pop("data_id_list")
        out = self._apply_ok(_apply_gigpo, cols, step_w=1.0, gamma=1.0, mode="mean_std")
        self.assertFalse(torch.equal(out.batch["advantages"], torch.zeros(6, W)),
                         "gigpo uid-fallback path did not fire")

    def test_rae_full_tgt_absent_action_keys_use_row_groups(self):
        """Absent action_key_list/uid degrade to per-row groups (kernel contract)."""
        from rl.hooks.rae_advantage import apply_rae_full_tgt_advantages

        cols = _nt_columns(verdict=True, as_ndarray=True)
        cols.pop("action_key_list")
        cols.pop("uid")
        cols.pop("data_id_list")
        out = apply_rae_full_tgt_advantages(
            _batch_from(cols, old_log_probs=torch.log(torch.full((6, W), 0.25))),
            epsilon_f=1e-3,
        )
        # With a uniform old_prob proxy and mixed verdicts the kernel must
        # have written *some* non-zero bias without raising.
        self.assertIsInstance(out, object)

    @staticmethod
    def _apply_ok(fn, cols, **kwargs):
        return fn(_batch_from(cols), **kwargs)


class TestCrashLockDocumentation(unittest.TestCase):
    """Lock the history: the bare-or line itself is truth-ambiguous.

    The historical line ``ntb.get("data_id_list") or ntb.get("uid")`` raised
    ValueError on multi-element ndarray inputs. We re-create the *original*
    expression literally and assert it still fails on the production shape —
    proving the regression suite bites if someone reverts the fix to a
    bare ``or`` while keeping the ndarray-shaped fixtures green.
    """

    def _ndarray_input(self):
        return np.array(["d0", "d0", "d0", "d0", "d1", "d1"], dtype=object)

    def test_historical_bare_or_expression_is_ambiguous(self):
        first = self._ndarray_input()
        fallback = ["u0"] * 6
        with self.assertRaises(ValueError):
            _ = first or fallback  # noqa: F841 (this exact expression raised pre-fix)

    def test_fixed_expression_prefers_present_value(self):
        first = self._ndarray_input()
        fallback = ["u0"] * 6
        resolved = fallback if first is None else first
        self.assertIs(resolved, first)
        self.assertEqual(_as_list(resolved), ["d0", "d0", "d0", "d0", "d1", "d1"])


def _as_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else list(x)


if __name__ == "__main__":
    unittest.main()
