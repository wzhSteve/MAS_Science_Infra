from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.executor import Executor
from MAS.epc_aw.models.formatters import PlannerDecision
from MAS.epc_aw.models.initializer import Initializer
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.planner import Planner
from MAS.epc_aw.models.state import (
    EventKind,
    FailureRecord,
    MilestoneStatus,
    VerifierEvent,
)


class Solver:
    """QA-native StructAgent orchestration.

    Planner and actor only propose progress. The verifier emits events and
    ``AgentState.apply_verifier_event`` is the sole progress commit path.
    """

    def __init__(
        self,
        planner: Planner,
        system_memory: SystemMemory,
        executor: Executor,
        diagnoser: Diagnoser,
        output_types: str = "base,final,direct",
        max_steps: int = 20,
        max_time: int = 3000,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = 0.0,
        actor_burst_size: int = 3,
        max_audit_retries: int = 2,
    ):
        self.planner = planner
        self.system_memory = system_memory
        self.executor = executor
        self.diagnoser = diagnoser
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir
        self.verbose = verbose
        self.temperature = temperature
        self.actor_burst_size = max(1, actor_burst_size)
        self.max_audit_retries = max(0, max_audit_retries)
        self.output_types = [item.strip().lower() for item in output_types.split(",")]
        if not all(item in {"base", "final", "direct"} for item in self.output_types):
            raise ValueError("output_types supports only base, final, and direct")

    def solve(
        self,
        question: str,
        image_path: Optional[str] = None,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        del parallel  # retained for EPC-AW call compatibility; BTS is removed.
        started = time.time()
        self.executor.set_query_cache_dir(self.root_cache_dir)
        result: Dict[str, Any] = {"query": question, "image": image_path}

        if "base" in self.output_types:
            result["base_response"] = self.planner.generate_base_response(
                question, image_path, self.max_tokens
            )
        if set(self.output_types) == {"base"}:
            return result

        state = self.planner.initialize_state(question, image_path)
        self.system_memory.set_state(state)
        result["analysis"] = "Question decomposed into verifier-gated milestones."
        result["outline"] = {
            str(index + 1): milestone.description
            for index, milestone in enumerate(state.milestones)
        }

        step = 0
        audit_failures = 0
        completed = False
        while step < self.max_steps and (time.time() - started) < self.max_time:
            if state.can_attempt_done():
                answer = self.executor.generate_direct_output(question, state)
                audit = self.diagnoser.audit_final_answer(state, answer)
                state.audit = (
                    audit.model_dump() if hasattr(audit, "model_dump") else audit.dict()
                )
                if audit.verdict == "PASS":
                    state.final_answer = answer
                    completed = True
                    result["direct_output"] = answer
                    result["final_output"] = answer
                    break
                audit_failures += 1
                self._handle_audit_failure(state, audit.reason, step)
                if audit_failures > self.max_audit_retries:
                    break

            reachable = state.reachable_milestones()
            if not reachable:
                state.failures.append(
                    FailureRecord(
                        step=step,
                        milestone_id=None,
                        attributed_to="planner",
                        reason="No reachable milestone remains while DONE is blocked.",
                        recovery_hint="Review milestone dependencies and verification specs.",
                    )
                )
                break

            try:
                plan = self.planner.select_subgoal(state)
            except Exception as exc:
                state.failures.append(
                    FailureRecord(
                        step=step,
                        milestone_id=None,
                        attributed_to="planner",
                        reason=f"{type(exc).__name__}: {exc}",
                        recovery_hint="Select one reachable milestone and available tool.",
                    )
                )
                step += 1
                continue

            milestone = state.get_milestone(plan.milestone_id)
            if milestone is None:  # guarded by Planner; defensive for custom planners
                step += 1
                continue

            for _ in range(self.actor_burst_size):
                if step >= self.max_steps or (time.time() - started) >= self.max_time:
                    break
                step += 1
                if self.verbose:
                    print(
                        f"[StructAgent step {step}/{self.max_steps}] "
                        f"target={milestone.id} ({milestone.description}) | "
                        f"subgoal={plan.subgoal} | tool={plan.tool_name}"
                    )
                trace = self._run_actor_turn(state, milestone, plan, step)
                state.add_traces([trace])
                event, verification = self.diagnoser.verify(state, milestone, trace)
                state.apply_verifier_event(event)
                if self.verbose:
                    obtained = json.dumps(
                        event.facts, ensure_ascii=False, default=str
                    )
                    print(
                        f"[StructAgent step {step}/{self.max_steps} result] "
                        f"target={milestone.id} | verdict={event.kind} | "
                        f"obtained={obtained} | reason={event.reason}"
                    )
                if event.kind in {
                    EventKind.SATISFIED.value,
                    EventKind.INVALIDATED.value,
                }:
                    break

                failure = self.diagnoser.attribute_failure(
                    state, milestone, trace, verification
                )
                state.failures.append(failure)
                if failure.attributed_to == "actor":
                    plan = self._plan_with_recovery_hint(plan, failure.recovery_hint)
                    continue
                break

        result.setdefault("direct_output", "")
        result.setdefault("final_output", result["direct_output"])
        result.update(
            {
                "completed": completed,
                "state": state.to_dict(),
                "events": [asdict(item) for item in state.events],
                "execution_traces": [
                    asdict(item) for item in state.execution_traces
                ],
                "failure_attributions": [
                    asdict(item) for item in state.failures
                ],
                "audit": state.audit,
                "step_count": step,
                "execution_time": round(time.time() - started, 3),
            }
        )
        if self.verbose:
            self._print_run_summary(state, step, completed)
        return result

    def _print_run_summary(self, state, step: int, completed: bool) -> None:
        verified = [
            milestone
            for milestone in state.milestones
            if milestone.status == MilestoneStatus.VERIFIED
        ]
        print(
            f"[StructAgent summary] total_steps={step}/{self.max_steps} | "
            f"completed={completed} | "
            f"verified_targets={len(verified)}/{len(state.milestones)}"
        )
        for milestone in state.milestones:
            facts = {
                name: fact.value
                for name, fact in state.facts.items()
                if fact.valid and fact.source_milestone_id == milestone.id
            }
            print(
                f"[StructAgent target info] target={milestone.id} | "
                f"status={milestone.status.value} | "
                f"obtained_at_step={milestone.last_updated_step} | "
                f"facts={json.dumps(facts, ensure_ascii=False, default=str)}"
            )
        print(
            "[StructAgent final answer] "
            f"{state.final_answer if state.final_answer is not None else '<none>'}"
        )

    def _run_actor_turn(self, state, milestone, plan, step):
        try:
            action = self.executor.generate_action(state, milestone, plan)
            return self.executor.execute_action(
                action,
                step=step,
                milestone_id=milestone.id,
                subgoal=plan.subgoal,
            )
        except Exception as exc:
            from MAS.epc_aw.models.state import ExecutionTrace

            return ExecutionTrace(
                step=step,
                milestone_id=milestone.id,
                subgoal=plan.subgoal,
                tool_name=plan.tool_name,
                arguments={},
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _plan_with_recovery_hint(
        plan: PlannerDecision, recovery_hint: str
    ) -> PlannerDecision:
        payload = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        payload["subgoal"] = (
            f"{payload['subgoal']}\nRecovery guidance: {recovery_hint}"
        )
        if hasattr(PlannerDecision, "model_validate"):
            return PlannerDecision.model_validate(payload)
        return PlannerDecision.parse_obj(payload)

    @staticmethod
    def _handle_audit_failure(state, reason: str, step: int) -> None:
        leaves = state.leaf_milestones()
        target = next(
            (
                item
                for item in reversed(leaves)
                if item.status == MilestoneStatus.VERIFIED
            ),
            None,
        )
        if target is not None:
            state.apply_verifier_event(
                VerifierEvent(
                    kind=EventKind.INVALIDATED.value,
                    milestone_id=target.id,
                    step=step,
                    reason=f"Final auditor rejected DONE: {reason}",
                    confidence="high",
                )
            )
        state.failures.append(
            FailureRecord(
                step=step,
                milestone_id=target.id if target else None,
                attributed_to="verifier",
                reason=reason,
                recovery_hint="Gather stronger evidence for the unsupported final claim.",
            )
        )


def construct_solver(
    llm_engine_name: str = "gpt-4o",
    enabled_tools: Optional[List[str]] = None,
    tool_engine: Optional[List[str]] = None,
    output_types: str = "final,direct",
    max_steps: int = 20,
    max_time: int = 3000,
    max_tokens: int = 4000,
    root_cache_dir: str = "solver_cache",
    verbose: bool = True,
    vllm_config_path: Optional[str] = None,
    temperature: float = 0.0,
    n: int = 1,
    planner_model: Optional[str] = None,
    verifier_model: Optional[str] = None,
    auditor_model: Optional[str] = None,
    actor_burst_size: int = 3,
    max_audit_retries: int = 2,
) -> Solver:
    enabled_tools = enabled_tools or ["all"]
    tool_engine = tool_engine or []
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )
    planner_name = planner_model or llm_engine_name
    verifier_name = verifier_model or llm_engine_name
    planner = Planner(
        llm_engine_name=planner_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
        n=n,
    )
    executor = Executor(
        llm_engine_name=llm_engine_name,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        toolbox_metadata=initializer.toolbox_metadata,
        tool_instances=initializer.tool_instances,
    )
    diagnoser = Diagnoser(
        llm_engine_name=verifier_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
        auditor_model=auditor_model,
    )
    memory = SystemMemory(toolbox_metadata=initializer.toolbox_metadata)
    return Solver(
        planner=planner,
        system_memory=memory,
        executor=executor,
        diagnoser=diagnoser,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        actor_burst_size=actor_burst_size,
        max_audit_retries=max_audit_retries,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA-native StructAgent")
    parser.add_argument(
        "--llm_engine_name",
        default=os.getenv("MODEL_NAME") or os.getenv("MODEL_Name") or "gpt-4o",
    )
    parser.add_argument("--output_types", default="base,final,direct")
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_time", type=int, default=300)
    parser.add_argument("--actor_burst_size", type=int, default=3)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=["all"],
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        actor_burst_size=args.actor_burst_size,
    )
    print(json.dumps(solver.solve("What is the capital of France?"), indent=2))


if __name__ == "__main__":
    main(parse_arguments())
