"""Unit tests: paper ablation presets must not leak disabled memory / intervention paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from MAS.epc_aw.models.ablation import PAPER_PRESETS, PRESETS, resolve_ablation
from MAS.epc_aw.models.executor import Executor
from MAS.epc_aw.models.planner import Planner


def test_paper_presets_defined():
    assert "full" in PRESETS
    assert PAPER_PRESETS == {
        "full",
        "no_intervention",
        "no_capability",
        "no_invocation",
        "no_memory",
    }
    assert "no_verified_memory" in PRESETS


def test_resolve_no_verified_memory_flags():
    cfg = resolve_ablation("no_verified_memory")
    assert cfg.enable_verified_memory_commit is False
    assert cfg.enable_verified_only_evolve is False
    assert cfg.enable_compact_task_view is False
    # Other paper-track memory reads stay on by default.
    assert cfg.read_capability_memory is True
    assert cfg.enable_evolve is True


def test_resolve_no_struct_control_flags():
    cfg = resolve_ablation("no_struct_control")
    assert cfg.enable_struct_stop_gate is False
    assert cfg.enable_struct_intervention_routing is False
    assert cfg.enable_final_audit is False
    # Verified memory stays on; only control-plane switches off.
    assert cfg.enable_verified_memory_commit is True
    assert "no_struct_control" in PRESETS


def test_resolve_no_capability_flags():
    cfg = resolve_ablation("no_capability")
    assert cfg.read_capability_memory is False
    assert cfg.read_invocation_memory is True
    assert cfg.enable_intervention is True


def test_resolve_no_invocation_keeps_blacklist():
    cfg = resolve_ablation("no_invocation")
    assert cfg.read_invocation_memory is False
    assert cfg.use_param_blacklist is True


def test_resolve_no_memory_disables_blacklist_and_evolve():
    cfg = resolve_ablation("no_memory")
    assert cfg.read_capability_memory is False
    assert cfg.read_invocation_memory is False
    assert cfg.enable_evolve is False
    assert cfg.use_param_blacklist is False


def test_resolve_no_intervention():
    cfg = resolve_ablation("no_intervention")
    assert cfg.enable_intervention is False
    assert cfg.enable_l2a and cfg.enable_l2b and cfg.enable_l3


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown ablation"):
        resolve_ablation("not_a_real_preset")


def test_planner_hint_skips_capability_retrieve_when_disabled():
    memory = MagicMock()
    memory.list_capability_tools.return_value = ["Google_Search_Tool"]
    memory.get_tool_capability_summary.return_value = "search"
    memory.retrieve_tool_capability.return_value = {
        "subgoal": "Retrieve external factual knowledge",
        "context_summary": ["open domain"],
    }
    planner = Planner.__new__(Planner)
    planner.system_memory = memory
    planner.ablation = resolve_ablation("no_capability")

    hint = planner._build_memory_query_hint("find the capital")
    assert hint == ""
    memory.retrieve_tool_capability.assert_not_called()
    memory.list_capability_tools.assert_not_called()


def test_planner_hint_calls_capability_when_enabled():
    memory = MagicMock()
    memory.list_capability_tools.return_value = ["Google_Search_Tool"]
    memory.get_tool_capability_summary.return_value = "search"
    memory.retrieve_tool_capability.return_value = {
        "subgoal": "Retrieve external factual knowledge",
        "context_summary": ["open domain"],
    }
    planner = Planner.__new__(Planner)
    planner.system_memory = memory
    planner.ablation = resolve_ablation("full")

    hint = planner._build_memory_query_hint("find the capital")
    assert "TOOL CAPABILITIES" in hint
    memory.retrieve_tool_capability.assert_called()


def test_executor_hint_skips_invocation_retrieve_when_disabled():
    memory = MagicMock()
    memory.retrieve_tool_invocation.return_value = {
        "factors": {"Entity": {"instruction": "name the entity"}},
    }
    memory.infer_state_type.return_value = "unknown"
    memory.get_failed_parameter_blacklist.return_value = ["bad query"]

    executor = Executor.__new__(Executor)
    executor.system_memory = memory
    executor.ablation = resolve_ablation("no_invocation")

    hint = executor._build_parameter_memory_hint("Retrieve facts", "Google_Search_Tool")
    memory.retrieve_tool_invocation.assert_not_called()
    # blacklist still allowed under no_invocation
    memory.get_failed_parameter_blacklist.assert_called()
    assert "Do not repeat" in hint


def test_executor_no_memory_skips_blacklist_and_invocation():
    memory = MagicMock()
    memory.retrieve_tool_invocation.return_value = {
        "factors": {"Entity": {"instruction": "name the entity"}},
    }
    memory.infer_state_type.return_value = "unknown"
    memory.get_failed_parameter_blacklist.return_value = ["bad query"]

    executor = Executor.__new__(Executor)
    executor.system_memory = memory
    executor.ablation = resolve_ablation("no_memory")

    hint = executor._build_parameter_memory_hint("Retrieve facts", "Google_Search_Tool")
    assert hint == ""
    memory.retrieve_tool_invocation.assert_not_called()
    memory.get_failed_parameter_blacklist.assert_not_called()


def test_product_layer_kill_switches():
    assert resolve_ablation("product_no_l2a").enable_l2a is False
    assert resolve_ablation("product_no_l2b").enable_l2b is False
    assert resolve_ablation("product_no_l3").enable_l3 is False


def test_offline_dir_override():
    cfg = resolve_ablation("full", offline_memory_dir="/tmp/mem_snap")
    assert cfg.memory_dir() == "/tmp/mem_snap"
    assert cfg.should_load_offline(evaluation_mode=True) is True
