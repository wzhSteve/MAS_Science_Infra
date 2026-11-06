# Round-based Planning DAG Regeneration

## 1. Original architecture

PEV windows in [`adapted/context.py`](adapted/context.py) drive Planner → Executor → Verifier.

CGHE layered planning:

1. `build_outlines_oneshot` → `PlanningGraph`
2. `select_step1` by max ACTIVE path coverage ([`planning_graph.py`](adapted/planning_graph.py))
3. On verifier failure: `FailureAnalyzer` → optional `HypothesisPruner` **on the same DAG** → SELECT sibling Step-1 **or** regenerate when empty/saturated ([`cghe_loop.py`](adapted/cghe_loop.py))
4. Soft-Bing (prior optimization): Bing/Fetch soft misses advance along the same plan without hard prune

Formally the failure path was \(G_t^{(k+1)} = \mathrm{Prune}(G_t^k, F_t^k)\).

## 2. New architecture

| Concept | Lifecycle |
|---------|-----------|
| Planning DAG | **Temporary**, versioned per planning round; sealed when the round ends |
| HypothesisMemory | **Persistent** across rounds |
| VerifierFeedback | Structured evidence for memory updates (does not emit the next DAG) |

Within a round: Soft-Bing / success still advances Bing→Fetch→Extract on the **same** DAG (ordinary multi-step execution).

When a round ends (no soft successor, or hard CE, or subgoal failure under round-regen):

1. Build `VerifierFeedback`
2. `HypothesisMemory.update_from_feedback`
3. `PlanningGraph.seal()` → immutable historical snapshot
4. Planner generates a **new** `PlanningGraph(version=t+1)` guided by memory
5. Coverage `select_step1` runs independently on the new DAG

Disable same-DAG prune+SELECT as the next-round strategy when `round_regen: true` (default). Legacy path: `CGHE_ROUND_REGEN=0`.

## 3. Modified / added files

| File | Role |
|------|------|
| [`adapted/hypothesis_round_memory.py`](adapted/hypothesis_round_memory.py) | **New** HypothesisMemory, VerifierFeedback, statuses |
| [`adapted/planning_graph.py`](adapted/planning_graph.py) | `version`, `seal()`, immutable snapshot |
| [`adapted/cghe_loop.py`](adapted/cghe_loop.py) | Archive sealed DAGs; regenerate installs new version |
| [`adapted/context.py`](adapted/context.py) | Round end → regen; memory-guided planner; Soft-Bing within round |
| [`adapted/recovery_controller.py`](adapted/recovery_controller.py) | Attaches `hypothesis_memory` |
| [`contracts/runtime.yaml`](contracts/runtime.yaml) | `round_regen: true`, `max_regenerations: 5` |
| [`adapted/test_round_regen.py`](adapted/test_round_regen.py) | Tests 1–10 |

## 4. New data structures

- `HypothesisStatus`: `UNKNOWN | SUPPORTED | REFUTED | UNVERIFIED`
- `PlanningHypothesis`: id, description, tool/argument, status, evidence, dag versions, env fingerprint
- `VerifierFeedback`: subgoal/execution status, failure_type, planning_implication (`KEEP|REVISE|AVOID|INVESTIGATE`), evidence
- `HypothesisMemory`: upsert, update_from_feedback, filter_plans (drop REFUTED), planner_guidance

**Critical rule:** `ExecutionFailure ≠ REFUTED`. Timeouts / soft verifier misses → `UNVERIFIED`. Only strong CE codes (`SNIPPETS_INSUFFICIENT`, `PARAM_ERROR`, …) with `AVOID` → `REFUTED`.

## 5. Control flow

```
init HypothesisMemory
round = 1
G = Planner(S, H, F_prev)          # version=round
loop:
  select_step1(G)                   # coverage unchanged
  execute / verify
  if success and has successor:
      advance along G               # within-round
  elif soft Bing/Fetch miss and has successor:
      soft-advance; H ← UNVERIFIED  # within-round
  elif task complete:
      stop
  else:
      H ← Update(H, F)
      seal(G)                       # immutable history
      round += 1
      G = Planner(S, H, F)          # NEW DAG, new version
```

## 6–7. How memory steers the next DAG

- Soft guidance text in `build_outlines_oneshot(..., guidance=...)`
- Hard filter: `HypothesisMemory.filter_plans` drops Step-1 keys that are REFUTED under the current environment fingerprint
- UNVERIFIED / SUPPORTED / UNKNOWN may be reused or reorganized; Planner may invent new queries (not a copy of surviving nodes)

## 8–9. Tests

`python -m unittest test_round_regen` — Tests 1–10 (new DAG, persistence, UNVERIFIED timeout, REFUTED filter, new hyp D, immutability, no repeat REFUTED, coverage select, SUPPORTED, multi-round).

Also green with `test_cghe_core`, `test_cghe_spec`, `test_cghe_recovery`.

## 10. Guarantees

| Requirement | Mechanism |
|-------------|-----------|
| Each round a new DAG | `CGHESession.regenerate` builds a new `PlanningGraph`; asserts `graph is not old` |
| No cross-round prune-as-strategy | `_round_regen_enabled` → `_end_round_for_regen` instead of SELECT siblings |
| Evidence persists | `HypothesisMemory` on `RecoveryController` |
| REFUTED not re-explored | `filter_plans` + guidance AVOID lines; env fingerprint can reopen REFUTED if state changes |
| New hypotheses allowed | Planner free to emit new queries; only REFUTED keys filtered |

## Limitations

- Soft-Bing within-round still advances without ending the round (intentional; matches “do not regenerate after every successful/soft step”).
- Same-DAG `HypothesisPruner` may still record CE for memory/logging before round end; the next active DAG is always newly planned.
- Full online eval under SoftBing was in flight separately; this refactor is covered by unit/integration tests above.
