"""LLM-call cost test for the offline Tool Knowledge Memory evolution.

Mocks the refiner's LLM engine so every `MemoryRefiner._ask` / retriever call
is counted, then drives the SAME grouped-by-tool pipeline that
`SystemMemory.evolve_tool_knowledge` uses (batched abstraction gate ->
cap.evolve / inv.evolve -> boundary_refinement) for a 2-tool x 3-pair scenario.

Asserts the total LLM call count is <= 7 (the plan's target for T=2, P=3),
proving the O1-O5 batching actually reduced calls from the old ~19-35.

Run:  python test_evolve_cost.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MAS.epc_aw.models.tool_knowledge_memory import (  # noqa: E402
    ToolCapabilityMemory,
    ToolInvocationMemory,
    MemoryRefiner,
    KeywordMemoryRetriever,
)


class CountingEngine:
    """A fake LLM engine that records every generate() call and returns
    plausible JSON so the batching code paths are exercised. Also categorizes
    each call so the test can print a per-stage breakdown."""

    def __init__(self):
        self.calls = 0
        self.by_stage = {}

    def _tick(self, stage: str):
        self.calls += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1

    def generate(self, user, system_prompt=""):
        p = (system_prompt + "\n" + user)
        pl = p.lower()
        if "one object per experience" in pl:
            self._tick("abstract_batch")
            # Count "[Experience N]" blocks to return the right array length.
            n_items = p.count("[Experience ")
            n_items = max(1, n_items)
            return json.dumps([
                {
                    "abstract_subgoal": "Retrieve external factual knowledge",
                    "context_summary": ["External knowledge is unavailable."],
                    "factors": {
                        "Entity": "Include sufficient identifiers to uniquely specify the target.",
                        "Freshness": "Prefer recent sources when temporal relevance matters.",
                    },
                }
                for _ in range(n_items)
            ])
        if "collapse semantically equivalent dimensions" in pl:
            self._tick("merge_factors_batch")
            return json.dumps({
                "Entity": {"instruction": "Include sufficient identifiers to uniquely specify the target."},
                "Freshness": {"instruction": "Prefer recent sources when temporal relevance matters."},
            })
        if "maximize inter-tool discrimination" in pl or "discrimination" in pl:
            self._tick("boundary_batch")
            return json.dumps({
                "Google_Search_Tool": "Retrieve open-domain factual information via web search.",
                "Wikipedia_Search_Tool": "Look up encyclopedic entities on Wikipedia.",
            })
        if "generalize and merge a list of short applicability" in pl:
            self._tick("generalize_context")
            return json.dumps(["External knowledge is unavailable.", "Open-domain question"])
        # single-item abstract fallback (_abstract_with_llm)
        if "map a concrete successful tool experience" in pl or "return strict json" in pl:
            self._tick("abstract_single")
            return json.dumps({
                "abstract_subgoal": "Retrieve external factual knowledge",
                "context_summary": ["External knowledge is unavailable."],
                "factors": {
                    "Entity": "Include sufficient identifiers to uniquely specify the target.",
                    "Freshness": "Prefer recent sources when temporal relevance matters.",
                },
            })
        self._tick("other")
        return "{}"


def build_seeded_memories(engine):
    cap = ToolCapabilityMemory.seed_default_capabilities(
        retriever=KeywordMemoryRetriever(),
        refiner=MemoryRefiner(retriever=KeywordMemoryRetriever(), llm_engine=engine),
    )
    inv = ToolInvocationMemory.seed_default_invocations(
        retriever=KeywordMemoryRetriever(),
        refiner=cap.refiner,
    )
    return cap, inv


def run_evolve_pipeline(cap, inv, per_tool_items):
    """Replicates SystemMemory.evolve_tool_knowledge's grouped-by-tool pipeline."""
    refiner = cap.refiner
    for tool, items in per_tool_items.items():
        canonical = [s["subgoal"] for s in cap.data.get(tool, {}).get("subgoals", [])]
        cap_summary = cap.data.get(tool, {}).get("capability_summary", "")
        abst_list = refiner.abstract_experiences_batch(tool, cap_summary, canonical, items)
        for abst in abst_list:
            asg = abst.get("abstract_subgoal", "")
            if not asg:
                continue
            cap.evolve(tool, asg, abst.get("context_summaries", []))
            factors = abst.get("factors", {})
            if factors:
                inv.evolve(tool, asg, factors)
    refiner.boundary_refinement(cap)


def main():
    engine = CountingEngine()
    cap, inv = build_seeded_memories(engine)

    # Plan scenario: 2 tools x 3 successful pairs TOTAL (2 on Google, 1 on Wiki).
    per_tool = {
        "Google_Search_Tool": [
            {"concrete_subgoal": "Find the capital of France",
             "question": "What is the capital of France?",
             "successful_params": [{"parameter": "capital of France"}]},
            {"concrete_subgoal": "Find the current PM of UK",
             "question": "Who is the PM of UK?",
             "successful_params": [{"parameter": "current UK prime minister"}]},
        ],
        "Wikipedia_Search_Tool": [
            {"concrete_subgoal": "Resolve the entity Einstein",
             "question": "Who was Einstein?",
             "successful_params": [{"parameter": "Albert Einstein"}]},
        ],
    }

    run_evolve_pipeline(cap, inv, per_tool)

    n = engine.calls
    print(f"LLM calls for 2 tools x 3 pairs total (P=3): {n}")
    print("Per-stage breakdown:", engine.by_stage)
    # Invariants
    for tool, td in cap.data.items():
        assert len(td.get("subgoals", [])) <= 5, f"cap subgoal cap violated: {tool}"
        for sg in td["subgoals"]:
            assert len(sg.get("context_summary", [])) <= 5, f"context cap violated: {tool}"
    for tool, td in inv.data.items():
        for sg in td.get("subgoals", []):
            assert len(sg.get("factors", {})) <= 4, f"factor cap violated: {tool}"
            for fname, fobj in sg["factors"].items():
                assert "instruction" in fobj and isinstance(fobj["instruction"], str)
    # Leakage: no task-specific entity should appear in stored memory.
    blob = json.dumps(cap.to_dict()) + json.dumps(inv.to_dict())
    for leak in ["einstein", "newton", "bohr", "france", "japan"]:
        assert leak not in blob.lower(), f"LEAK: {leak} found in offline memory"

    assert n <= 7, f"FAIL: expected <= 7 LLM calls, got {n}"
    print(f"PASS: LLM call count {n} <= 7 (old path was ~19-35 for this scenario)")
    # Show the per-stage breakdown
    print("capability_summary after boundary:",
          cap.data["Google_Search_Tool"]["capability_summary"][:60], "...")


def test_systemmemory_pipeline():
    """Drive the REAL SystemMemory.evolve_tool_knowledge (incl. O6 parallelism)
    with a mock engine by injecting online_memory + query and replacing the
    refiner engine with a CountingEngine."""
    from MAS.epc_aw.models.memory import SystemMemory

    engine = CountingEngine()
    sm = SystemMemory(toolbox_metadata={}, agent_profile={})
    # In real runs the causal graph is populated during the task and
    # _sync_denormalized_from_graph rebuilds task_local_parameters from it. Here
    # we stub the sync so we can inject task_local_parameters directly and
    # exercise evolve_tool_knowledge's grouping + parallel + evolve logic.
    sm._sync_denormalized_from_graph = lambda: None
    # Force the refiner to use the counting engine (overrides whatever __init__
    # created, so no real API key is needed).
    sm._tkm_refiner._engine = engine
    sm.offline_memory["tool_capability"] = ToolCapabilityMemory.seed_default_capabilities(
        retriever=KeywordMemoryRetriever(), refiner=sm._tkm_refiner,
    )
    sm.offline_memory["tool_invocation"] = ToolInvocationMemory.seed_default_invocations(
        retriever=KeywordMemoryRetriever(), refiner=sm._tkm_refiner,
    )
    sm.query = "What is the capital of France and who was Einstein?"
    # Build task_local_parameters in the shape evolve_tool_knowledge reads.
    sm.online_memory["task_local_parameters"] = {
        ("initial_no_info", "Find the capital of France"): {
            "Google_Search_Tool": {"successful_parameters": [{"parameter": "capital of France"}]},
        },
        ("initial_no_info", "Find the current PM of UK"): {
            "Google_Search_Tool": {"successful_parameters": [{"parameter": "current UK PM"}]},
        },
        ("initial_no_info", "Resolve the entity Einstein"): {
            "Wikipedia_Search_Tool": {"successful_parameters": [{"parameter": "Albert Einstein"}]},
        },
    }
    sm.evolve_tool_knowledge()
    n = engine.calls
    print(f"\n[SystemMemory.evolve_tool_knowledge] LLM calls (P=3, 2 tools): {n}")
    print("Per-stage breakdown:", engine.by_stage)
    assert n <= 8, f"FAIL: SystemMemory path expected <= 8 LLM calls, got {n}"
    # Invariants
    cap = sm.offline_memory["tool_capability"]
    inv = sm.offline_memory["tool_invocation"]
    for tool, td in cap.data.items():
        assert len(td.get("subgoals", [])) <= 5
    for tool, td in inv.data.items():
        for sg in td.get("subgoals", []):
            assert len(sg.get("factors", {})) <= 4
    blob = json.dumps(cap.to_dict()) + json.dumps(inv.to_dict())
    for leak in ["einstein", "france"]:
        assert leak not in blob.lower(), f"LEAK: {leak}"
    print(f"PASS: SystemMemory path {n} calls, invariants + leakage OK")


if __name__ == "__main__":
    main()
    test_systemmemory_pipeline()
