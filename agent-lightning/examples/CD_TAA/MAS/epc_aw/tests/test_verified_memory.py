"""Unit tests for StructAgent-inspired verified memory ledger."""

from __future__ import annotations

from MAS.epc_aw.models.ablation import resolve_ablation
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.verified_memory import (
    CommitKind,
    MemoryCommitEvent,
    map_verification_to_commit_kind,
)


def _memory() -> SystemMemory:
    mem = SystemMemory(toolbox_metadata={})
    mem.configure_verified_memory(
        enable_verified_memory_commit=True,
        enable_verified_only_evolve=True,
        enable_compact_task_view=True,
    )
    mem.set_query("What is the capital of France?")
    mem.set_outline({"1": "Find capital city"})
    return mem


def test_map_verification_kinds():
    assert (
        map_verification_to_commit_kind(
            step_conclusion="SUBGOAL_COMPLETE",
            subgoal_complete=True,
            claim_type="fact",
        )
        == CommitKind.SATISFIED.value
    )
    assert (
        map_verification_to_commit_kind(
            step_conclusion="SUBGOAL_COMPLETE",
            subgoal_complete=True,
            compute_real=True,
        )
        == CommitKind.VALUE_COMMITTED.value
    )
    assert (
        map_verification_to_commit_kind(
            step_conclusion="SUBGOAL_COMPLETE",
            subgoal_complete=True,
            evidence_type="ABSENCE",
            claim_type="absence",
        )
        == CommitKind.REJECTED.value
    )
    assert (
        map_verification_to_commit_kind(
            step_conclusion="SUBGOAL_INCOMPLETE",
            subgoal_complete=False,
        )
        == CommitKind.REJECTED.value
    )


def test_commit_satisfied_writes_fact_and_evidence():
    mem = _memory()
    result = mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"capital": "Paris"},
            tool_name="Google_Search_Tool",
        ),
        evidence_payload={
            "content": "Paris is the capital of France according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "subgoal": "Find capital city",
            "tool": "Google_Search_Tool",
            "source_quality": "primary",
            "claim_type": "fact",
        },
        slot_bindings={"capital": "Paris"},
    )
    assert result["committed"] is True
    assert result["evidence_id"]
    assert mem.verified.valid_facts()["capital"] == "Paris"
    assert mem.evidence_records[0]["live"] is True
    assert mem.evidence_records[0]["verification_status"] == "satisfied"
    prompt = mem.get_obtained_information_for_prompt()
    assert "Paris" in prompt
    assert "Valid facts" in prompt or "capital" in prompt


def test_commit_rejected_does_not_add_valid_facts():
    mem = _memory()
    result = mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.REJECTED.value,
            step=1,
            reason="empty results",
            outline_step="1",
            tool_name="Google_Search_Tool",
        ),
        evidence_payload={
            "content": "Search results do not contain the requested capital city name.",
            "outline_step": "1",
            "exec_step": 1,
            "tool": "Google_Search_Tool",
            "claim_type": "absence",
        },
    )
    assert result["committed"] is False
    assert mem.verified.valid_facts() == {}
    assert mem.verified.recent_failures
    compact = mem.compact_task_view()
    assert compact["facts"] == {}
    assert compact["recent_failures"]


def test_invalidation_removes_fact_from_compact_view():
    mem = _memory()
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"capital": "Paris"},
        ),
        evidence_payload={
            "content": "Paris is the capital of France according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
        },
        slot_bindings={"capital": "Paris"},
    )
    eid = mem.evidence_records[0]["id"]
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.INVALIDATED.value,
            step=2,
            reason="conflict",
            evidence_id=eid,
            outline_step="1",
        )
    )
    assert mem.evidence_records[0]["status"] == "disputed"
    assert mem.evidence_records[0]["live"] is False
    assert "capital" not in mem.verified.valid_facts()
    prompt = mem.get_obtained_information_for_prompt()
    assert "Paris is the capital" not in prompt


def test_evolve_gate_rejects_unverified_success_params():
    mem = _memory()
    mem.online_memory["task_local_parameters"] = {
        ("partial_first_evidence", "Find capital city"): {
            "Google_Search_Tool": {
                "successful_parameters": ['execute("Google_Search_Tool", query="capital")'],
                "failed_parameters": [],
                "pairs": [],
            }
        }
    }
    # Seed capability slot so evolve would otherwise accept the tool.
    cap = mem.offline_memory["tool_capability"]
    cap.data["Google_Search_Tool"] = {
        "capability_summary": "Web search",
        "subgoals": [
            {
                "subgoal": "Retrieve external factual knowledge",
                "context_summary": ["Open-domain question"],
            }
        ],
    }
    mem.enable_verified_only_evolve = True
    ok, reason = mem._experience_eligible_for_evolve(
        tool="Google_Search_Tool",
        subgoal="Find capital city",
    )
    assert ok is False
    assert reason == "no_verified_evidence_or_contrast"

    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"capital": "Paris"},
        ),
        evidence_payload={
            "content": "Paris is the capital of France according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "subgoal": "Find capital city",
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
        },
    )
    ok2, reason2 = mem._experience_eligible_for_evolve(
        tool="Google_Search_Tool",
        subgoal="Find capital city",
    )
    assert ok2 is True
    assert reason2 == "live_verified_evidence"


def test_add_evidence_record_does_not_dual_write_verified_facts():
    """Raw evidence append must not mint verified facts without commit_progress."""
    mem = _memory()
    added = mem.add_evidence_record(
        "Berlin is the capital of Germany according to encyclopedia.",
        outline_step="1",
        exec_step=1,
        subgoal="Find capital",
        tool="Google_Search_Tool",
        subgoal_complete=True,
        claim_type="fact",
        slot_bindings={"capital": "Berlin"},
    )
    assert added
    assert mem.verified.valid_facts().get("capital") is None
    assert mem.evidence_records[0]["verification_status"] == "satisfied"


def test_add_evidence_record_explicit_dual_write_opt_in():
    mem = _memory()
    added = mem.add_evidence_record(
        "Berlin is the capital of Germany according to encyclopedia.",
        outline_step="1",
        exec_step=1,
        subgoal="Find capital",
        tool="Google_Search_Tool",
        subgoal_complete=True,
        claim_type="fact",
        slot_bindings={"capital": "Berlin"},
        sync_verified_ledger=True,
    )
    assert added
    assert mem.verified.valid_facts().get("capital") == "Berlin"


def test_commit_progress_marks_outline_done():
    mem = _memory()
    result = mem.commit_progress(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"capital": "Paris"},
            tool_name="Google_Search_Tool",
        ),
        evidence_payload={
            "content": "Paris is the capital of France according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "subgoal": "Find capital city",
            "tool": "Google_Search_Tool",
            "source_quality": "primary",
            "claim_type": "fact",
        },
        slot_bindings={"capital": "Paris"},
        mark_outline_done=True,
    )
    assert result["committed"] is True
    assert result.get("progress_api") == "commit_progress"
    assert "1" in mem.task_progress["completed_outline_steps"]
    assert mem.verified.valid_facts()["capital"] == "Paris"


def test_commit_progress_is_sole_live_fact_path():
    mem = _memory()
    # Pre-commit: compact view must not invent facts from raw append.
    mem.add_evidence_record(
        "Raw uncommitted note about Paris being mentioned somewhere online.",
        outline_step="1",
        exec_step=1,
        subgoal="Find capital",
        tool="Google_Search_Tool",
        subgoal_complete=True,
        slot_bindings={"capital": "Paris"},
    )
    assert "capital" not in mem.verified.valid_facts()
    mem.commit_progress(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=2,
            reason="verified",
            outline_step="1",
            facts={"capital": "Paris"},
            tool_name="Google_Search_Tool",
        ),
        evidence_payload={
            "content": "Paris is the capital of France according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 2,
            "subgoal": "Find capital city",
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
        },
        slot_bindings={"capital": "Paris"},
    )
    assert mem.verified.valid_facts()["capital"] == "Paris"


def test_ablation_no_verified_memory_preset():
    cfg = resolve_ablation("no_verified_memory")
    assert cfg.enable_verified_memory_commit is False
    assert cfg.enable_verified_only_evolve is False
    assert cfg.enable_compact_task_view is False
    mem = _memory()
    mem.apply_ablation_memory_flags(cfg)
    assert mem.enable_verified_memory_commit is False
    assert mem.enable_compact_task_view is False
