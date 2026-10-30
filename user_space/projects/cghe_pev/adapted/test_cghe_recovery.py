"""Unit tests for evidence memory and the recovery controller."""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cghe_types import (  # noqa: E402
    NEEDS_REPLAN_MESSAGE,
    AtomicPremise,
    Budget,
    CandidatePlan,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    ExecutionObservation,
    FailureType,
    PlanStep,
    Scope,
)
from evidence_memory import EvidenceMemory, default_evidence_db_path  # noqa: E402
from failure_analyzer import FailureAnalyzer  # noqa: E402
from planning_graph import UniverseFrozenError  # noqa: E402
from recovery_controller import (  # noqa: E402
    BING_TOOL,
    CONTINUE_PEV_MESSAGE,
    FETCH_TOOL,
    GENERALIST_TOOL,
    PYTHON_TOOL,
    STOP_RECOVERY_ROUNDS,
    STOP_REPLANS,
    STOP_TOOL_CALLS,
    RecoveryController,
    RecoveryDecision,
    call_fingerprint,
    should_enter_recovery,
)


def _step(
    tool: str,
    premises: tuple[AtomicPremise, ...] = (),
    *,
    cost: float = 1.0,
    argument: str = "",
    parameter: str = "",
    resource: str = "",
    identity: str = "",
) -> PlanStep:
    return PlanStep(
        subgoal=tool,
        action=tool,
        tool=tool,
        preconditions=premises,
        estimated_cost=cost,
        argument=argument,
        parameter=parameter,
        resource=resource,
        identity=identity,
    )


def _plan(plan_id: str, steps: tuple[PlanStep, ...], *, environment: str = "sandbox") -> CandidatePlan:
    return CandidatePlan(plan_id, steps, context=Scope(environment=environment))


def _exists(tool: str) -> AtomicPremise:
    return AtomicPremise(predicate="ToolExists", tool=tool)


def _valid(tool: str, argument: str) -> AtomicPremise:
    return AtomicPremise(predicate="ValidID", tool=tool, parameter="id", argument=argument)


def _access(tool: str, resource: str, identity: str = "alice") -> AtomicPremise:
    return AtomicPremise(predicate="CanAccess", tool=tool, resource=resource, identity=identity)


class EvidenceMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "evidence.sqlite"
        self.memory = EvidenceMemory(self.db_path)

    def tearDown(self) -> None:
        self.memory.close()
        self._tmp.cleanup()

    def test_default_path_is_under_project_artifacts_and_can_be_overridden(self) -> None:
        path = default_evidence_db_path()
        self.assertEqual(path.parent.name, "artifacts")
        self.assertEqual(path.name, "cghe_evidence.sqlite")
        self.assertEqual(Path(self.memory.db_path), self.db_path)
        self.assertTrue(self.db_path.is_file())

    def test_three_statuses_round_trip_and_hypotheses_stay_out_of_pruning(self) -> None:
        observation = ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result={"error_code": "TOOL_NOT_FOUND"},
            llm_hypothesis="maybe the resource is gone",
            execution_log="tool returned TOOL_NOT_FOUND",
        )
        verified = self.memory.record_counterexample(
            FailureAnalyzer().analyze(observation).constraint,
            task_id="task-1",
            task_type="c1",
            task_goal="read the record",
            environment="sandbox",
            action="read",
            command="read.execute",
            observation=observation,
        )
        success = self.memory.record_success(
            task_id="task-1",
            task_type="c1",
            task_goal="export",
            environment="sandbox",
            action="export",
            tool="export",
            observation=ExecutionObservation(tool_name="export", environment="sandbox", tool_result="ok"),
            source="verifier",
        )
        hypothesis = self.memory.record_hypothesis(
            task_id="task-1",
            task_type="c1",
            environment="sandbox",
            action="guess",
            tool="read",
            observation=observation,
            note="m_cf",
            source="llm_hypothesis",
            origin=EvidenceOrigin.INFERENCE,
        )
        self.assertEqual(verified.evidence_status, EvidenceStatus.VERIFIED_COUNTEREXAMPLE)
        self.assertEqual(success.evidence_status, EvidenceStatus.VERIFIED_SUCCESS)
        self.assertEqual(hypothesis.evidence_status, EvidenceStatus.UNVERIFIED_HYPOTHESIS)
        self.assertIn("maybe the resource is gone", json.dumps(hypothesis.observation))

        reopened = EvidenceMemory(self.db_path)
        pruning = reopened.constraints_for_pruning(environment="sandbox", task_type="c1")
        self.assertEqual(tuple(item.constraint_id for item in pruning), (verified.constraint.constraint_id,))
        planning = reopened.records_for_planning(environment="sandbox", task_type="c1")
        self.assertEqual(tuple(item.record_id for item in planning.verified_successes), (success.record_id,))
        self.assertEqual(len(planning.verified_counterexamples), 1)
        self.assertEqual(tuple(item.record_id for item in planning.diagnostic_hypotheses), (hypothesis.record_id,))
        self.assertEqual(planning.pruning_constraints(), pruning)
        self.assertFalse(any(
            item.evidence_status == EvidenceStatus.UNVERIFIED_HYPOTHESIS for item in pruning
        ))
        other_env = reopened.constraints_for_pruning(environment="other", task_type="c1")
        self.assertEqual(other_env, ())
        other_task = reopened.constraints_for_pruning(environment="sandbox", task_type="c5")
        self.assertEqual(other_task, ())
        state = reopened.export_state()
        blob = json.dumps(state)
        self.assertIn(self.memory.db_path, blob)
        self.assertNotIn("sqlite3", blob)
        self.assertNotIn("_conn", blob)
        self.assertIsInstance(state["records"], list)
        self.assertEqual(len(state["records"]), 3)
        reopened.close()

    def test_export_state_reopens_without_keeping_a_connection(self) -> None:
        self.memory.record_success(environment="sandbox", action="export", tool="export", task_type="c1")
        raw = pickle.dumps(self.memory)
        restored = pickle.loads(raw)
        self.assertIsNone(restored._conn)
        self.assertEqual(restored.db_path, self.memory.db_path)
        self.assertEqual(len(restored.records()), 1)
        self.assertNotIsInstance(restored._conn, type(None))
        restored.close()


class RecoveryControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "evidence.sqlite")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _controller(self, **kwargs: object) -> RecoveryController:
        kwargs.setdefault("db_path", self.db_path)
        kwargs.setdefault("environment", "sandbox")
        kwargs.setdefault("task_id", "task-1")
        kwargs.setdefault("task_type", "c1")
        return RecoveryController(**kwargs)  # type: ignore[arg-type]

    def _missing_read(self, **kwargs: object) -> ExecutionObservation:
        base = dict(
            tool_name="read",
            environment="sandbox",
            tool_result={"error_code": "TOOL_NOT_FOUND", "tool": "read"},
            llm_hypothesis="the resource does not exist",
            verifier_label="failure",
        )
        base.update(kwargs)
        return ExecutionObservation(**base)

    def _read_plans(self) -> list[CandidatePlan]:
        return [
            _plan("p1", (_step("read", (_exists("read"),)),)),
            _plan("p2", (_step("read", (_exists("read"),)),)),
            _plan("p3", (_step("export", (_exists("export"),)),)),
        ]

    def test_trigger_requires_explicit_verifier_failure_and_an_explorable_type(self) -> None:
        self.assertFalse(should_enter_recovery(verifier_label="uncertain", failure_type=FailureType.VERIFIER_UNCERTAIN).intervene)
        self.assertIn("uncertainty", should_enter_recovery(verifier_label="uncertain").reason)
        low = should_enter_recovery(
            verifier_label="failure",
            verifier_satisfied=False,
            failure_type=FailureType.TRANSIENT_ERROR,
            confidence=0.2,
            verified_counterexample=False,
        )
        self.assertFalse(low.intervene)
        self.assertIn("low-confidence", low.reason)
        ordinary = should_enter_recovery(
            verifier_label="不满足",
            failure_type=FailureType.SUBGOAL_NOT_SATISFIED,
            verified_counterexample=False,
            confidence=1.0,
        )
        self.assertFalse(ordinary.intervene)
        self.assertIn("original PEV", ordinary.reason)
        silent = should_enter_recovery(
            failure_type=FailureType.TOOL_UNAVAILABLE,
            verified_counterexample=True,
            confidence=1.0,
        )
        self.assertFalse(silent.intervene)
        entered = should_enter_recovery(
            verifier_label="failure",
            failure_type=FailureType.TOOL_UNAVAILABLE,
            verified_counterexample=True,
            confidence=1.0,
        )
        self.assertTrue(entered.intervene)

    def test_ordinary_and_uncertain_failures_do_not_open_a_graph(self) -> None:
        controller = self._controller()
        uncertain = controller.handle_failure(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            verifier_label="uncertain",
            tool_result="a normal page",
        ))
        self.assertEqual(uncertain.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertEqual(uncertain.message, CONTINUE_PEV_MESSAGE)
        self.assertFalse(uncertain.intervene)
        self.assertIsNone(controller.graph)
        self.assertEqual(controller.budget.recovery_rounds_used, 0)
        self.assertEqual(controller.memory.constraints_for_pruning(environment="sandbox", task_type="c1"), ())
        self.assertTrue(controller.planning_records().diagnostic_hypotheses)

        generic = controller.handle_failure(
            ExecutionObservation(tool_name="read", environment="sandbox", tool_result="Error: timeout talking to the host"),
            verifier_label="failure",
            plans=self._read_plans(),
        )
        self.assertEqual(generic.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertIsNone(controller.graph)
        self.assertEqual(generic.analysis.evidence.failure_type, FailureType.TRANSIENT_ERROR)

        subgoal = controller.handle_failure(
            ExecutionObservation(tool_name="read", environment="sandbox", tool_result="a normal page", verifier_label="不满足"),
            plans=self._read_plans(),
        )
        self.assertEqual(subgoal.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertIn("original PEV", subgoal.reason)

        guessed = controller.handle_failure(
            ExecutionObservation(
                tool_name="read",
                environment="sandbox",
                tool_result="page text without an error code",
                llm_hypothesis="TOOL_NOT_FOUND",
                verifier_label="failure",
            ),
            plans=self._read_plans(),
        )
        self.assertEqual(guessed.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertFalse(guessed.analysis.verified_counterexample)
        self.assertEqual(controller.memory.constraints_for_pruning(environment="sandbox"), ())

    def test_tool_error_without_verifier_failure_stays_on_pev(self) -> None:
        controller = self._controller()
        outcome = controller.handle_failure(
            self._missing_read(verifier_label=""),
            plans=self._read_plans(),
        )
        self.assertEqual(outcome.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertFalse(outcome.intervene)
        self.assertIsNone(controller.graph)
        self.assertEqual(controller.budget.recovery_rounds_used, 0)
        self.assertEqual(controller.memory.constraints_for_pruning(environment="sandbox"), ())

    def test_verified_counterexample_prunes_and_selects_the_shared_survivor(self) -> None:
        controller = self._controller()
        asking = controller.handle_failure(self._missing_read())
        self.assertEqual(asking.decision, RecoveryDecision.NEED_UNIVERSE)
        self.assertEqual(controller.budget.recovery_rounds_used, 0)
        self.assertIsNone(controller.graph)
        self.assertTrue(controller.memory.constraints_for_pruning(environment="sandbox", task_type="c1"))

        outcome = controller.begin_round(self._read_plans(), round_id="round-a")
        self.assertEqual(outcome.decision, RecoveryDecision.CONTINUE_EXECUTION)
        self.assertEqual(outcome.round_id, "round-a")
        self.assertIsNotNone(outcome.selected_node)
        assert outcome.selected_node is not None
        self.assertEqual(outcome.selected_node.tool, "export")
        self.assertEqual(controller.graph.active_plan_ids(), ("p3",))
        self.assertEqual(controller.budget.recovery_rounds_used, 1)
        self.assertEqual(controller.budget.replans_used, 0)
        self.assertFalse(outcome.task_impossible)
        log = outcome.logs[0]
        self.assertEqual(log.active_before, 3)
        self.assertEqual(log.active_after, 1)
        self.assertEqual(log.evidence_source, "tool_result")
        self.assertTrue(log.reason)
        self.assertTrue(log.reason_code)
        with self.assertRaises(UniverseFrozenError):
            controller.graph.add_candidate(_plan("extra", (_step("export", (_exists("export"),)),)))
        with self.assertRaises(UniverseFrozenError):
            controller.begin_round([_plan("other", (_step("ping", (_exists("ping"),)),))])

    def test_permission_error_prunes_only_the_matching_access(self) -> None:
        controller = self._controller()
        secret = _access("read", "secret")
        other = _access("read", "other")
        plans = [
            _plan("secret_a", (_step("read", (secret,), resource="secret", identity="alice", cost=5),)),
            _plan("secret_b", (_step("read", (secret,), resource="secret", identity="alice", cost=5),)),
            _plan("other", (_step("read", (other,), resource="other", identity="alice", cost=1),)),
            _plan("export", (_step("export", (_exists("export"),), cost=1),)),
            _plan(
                "foreign",
                (_step("read", (secret,), resource="secret", identity="alice", cost=5),),
                environment="other-env",
            ),
        ]
        outcome = controller.handle_failure(
            ExecutionObservation(
                tool_name="read",
                environment="sandbox",
                resource="secret",
                identity="alice",
                tool_result={
                    "error_code": "PERMISSION_DENIED",
                    "resource": "secret",
                    "identity": "alice",
                },
                verifier_label="failure",
                plan_id="secret_a",
            ),
            plans=plans,
        )
        self.assertEqual(outcome.decision, RecoveryDecision.CONTINUE_EXECUTION)
        self.assertEqual(outcome.analysis.mapped_predicate, "CanAccess")
        self.assertFalse(outcome.analysis.constraint.disables_tool)
        self.assertEqual(set(outcome.prune_result.pruned_plan_ids), {"secret_a", "secret_b"})
        self.assertEqual(set(controller.graph.active_plan_ids()), {"other", "export", "foreign"})
        self.assertFalse(outcome.task_impossible)
        log = outcome.logs[0]
        self.assertEqual(log.active_before, 5)
        self.assertEqual(log.active_after, 3)
        self.assertEqual(log.reason_code, "admissible_prune")
        self.assertEqual(log.evidence_source, "tool_result")
        self.assertTrue(log.reason)
        self.assertTrue(log.admissible)

    def test_transient_network_error_does_not_permanently_prune(self) -> None:
        controller = self._controller()
        opened = controller.handle_failure(self._missing_read(), plans=self._read_plans())
        self.assertEqual(opened.decision, RecoveryDecision.CONTINUE_EXECUTION)
        self.assertEqual(controller.graph.active_plan_ids(), ("p3",))
        stored = controller.memory.constraints_for_pruning(environment="sandbox", task_type="c1")
        self.assertEqual(len(stored), 1)
        transient = controller.handle_failure(
            ExecutionObservation(
                tool_name="export",
                environment="sandbox",
                tool_result="Error: HTTP 503 timeout talking to the host",
                verifier_label="failure",
                plan_id="p3",
            )
        )
        self.assertEqual(transient.decision, RecoveryDecision.CONTINUE_PEV)
        self.assertFalse(transient.intervene)
        self.assertEqual(transient.analysis.evidence.failure_type, FailureType.TRANSIENT_ERROR)
        self.assertFalse(transient.analysis.verified_counterexample)
        self.assertEqual(controller.graph.active_plan_ids(), ("p3",))
        self.assertEqual(
            controller.memory.constraints_for_pruning(environment="sandbox", task_type="c1"),
            stored,
        )
        later = self._controller()
        again = later.begin_round(self._read_plans(), round_id="later")
        self.assertIn("p3", later.graph.active_plan_ids())
        self.assertNotIn("p1", later.graph.active_plan_ids())
        self.assertNotIn("p2", later.graph.active_plan_ids())
        self.assertFalse(again.task_impossible)
        self.assertTrue(all(
            item.observation_code != "TRANSIENT"
            for item in later.memory.constraints_for_pruning(environment="sandbox", task_type="c1")
        ))
        later.memory.close()

    def test_parameter_error_does_not_disable_the_tool(self) -> None:
        controller = self._controller()
        plans = [
            _plan("bad", (_step("read", (_valid("read", "bad"),), argument="bad", parameter="id"),)),
            _plan("other", (_step("read", (_valid("read", "good"),), argument="good", parameter="id"),)),
        ]
        outcome = controller.handle_failure(
            ExecutionObservation(
                tool_name="read",
                environment="sandbox",
                parameters={"id": "bad"},
                tool_result="parameter error: invalid id 'bad'",
                verifier_label="failure",
                plan_id="bad",
            ),
            plans=plans,
        )
        self.assertEqual(outcome.decision, RecoveryDecision.CONTINUE_EXECUTION)
        self.assertEqual(controller.graph.active_plan_ids(), ("other",))
        self.assertFalse(outcome.analysis.constraint.disables_tool)
        self.assertEqual(outcome.selected_node.tool, "read")
        self.assertEqual(outcome.selected_node.argument, "good")

    def test_exhausted_universe_opens_another_round_until_budget(self) -> None:
        controller = self._controller()
        self.assertEqual(controller.budget.tool_calls, 12)
        self.assertEqual(controller.budget.replans, 1)
        self.assertEqual(controller.budget.recovery_rounds, 2)
        first = controller.handle_failure(
            self._missing_read(),
            plans=[
                _plan("p1", (_step("read", (_exists("read"),)),)),
                _plan("p2", (_step("read", (_exists("read"),)),)),
            ],
        )
        self.assertEqual(first.decision, RecoveryDecision.NEED_UNIVERSE)
        self.assertEqual(first.message, NEEDS_REPLAN_MESSAGE)
        self.assertFalse(first.task_impossible)
        self.assertEqual(controller.budget.recovery_rounds_used, 1)
        self.assertNotEqual(first.logs[0].active_before, first.logs[0].active_after)
        self.assertEqual(first.logs[0].active_after, 0)

        second = controller.begin_round(
            [_plan("p3", (_step("export", (_exists("export"),)),))],
            round_id="round-b",
        )
        self.assertEqual(second.decision, RecoveryDecision.CONTINUE_EXECUTION)
        self.assertEqual(second.round_id, "round-b")
        self.assertNotEqual(second.round_id, first.round_id)
        self.assertEqual(second.selected_node.tool, "export")
        self.assertEqual(controller.budget.replans_used, 1)
        self.assertEqual(controller.budget.recovery_rounds_used, 2)
        self.assertFalse(second.task_impossible)

        again = controller.handle_failure(
            ExecutionObservation(
                tool_name="export",
                environment="sandbox",
                tool_result={"error_code": "TOOL_NOT_FOUND", "tool": "export"},
                verifier_label="failure",
            )
        )
        self.assertEqual(again.decision, RecoveryDecision.STOP_BUDGET)
        self.assertEqual(again.reason, STOP_REPLANS)
        self.assertFalse(again.task_impossible)
        self.assertNotIn("impossible", again.reason.casefold())

    def test_recovery_round_budget_stops_with_an_explicit_reason(self) -> None:
        controller = self._controller(recovery_rounds=1, replans=1)
        outcome = controller.handle_failure(
            self._missing_read(),
            plans=[_plan("p1", (_step("read", (_exists("read"),)),))],
        )
        self.assertEqual(outcome.decision, RecoveryDecision.STOP_BUDGET)
        self.assertEqual(outcome.reason, STOP_RECOVERY_ROUNDS)
        self.assertFalse(outcome.task_impossible)

    def test_retry_policy_fingerprint_and_tool_budget(self) -> None:
        controller = self._controller(tool_calls=2)
        python_first = controller.authorize_tool_call(PYTHON_TOOL, command="print(1)")
        self.assertTrue(python_first.allowed)
        controller.commit_tool_call(python_first)
        python_replay = controller.may_auto_replay("Python_Coder_Tool", command="print(1)")
        self.assertFalse(python_replay.allowed)
        self.assertIn("not proven idempotent", python_replay.reason)
        python_again = controller.authorize_tool_call("Python", command="print(1)", replay=False)
        self.assertFalse(python_again.allowed)

        generalist = controller.may_auto_replay("Base_Generator_Tool", command="draft")
        self.assertFalse(generalist.allowed)
        self.assertIn("Generalist", generalist.reason)
        self.assertFalse(controller.may_auto_replay(GENERALIST_TOOL, command="draft").allowed)

        search = controller.authorize_tool_call(BING_TOOL, command="cghe", parameters={"q": "cghe"})
        self.assertTrue(search.allowed)
        self.assertTrue(search.safe_to_retry)
        controller.commit_tool_call(search)
        self.assertEqual(controller.budget.tool_calls_used, 2)
        replay = controller.may_auto_replay(BING_TOOL, command="cghe", parameters={"q": "cghe"})
        self.assertFalse(replay.allowed)
        self.assertEqual(replay.reason, STOP_TOOL_CALLS)
        fetch = controller.authorize_tool_call(FETCH_TOOL, command="https://example.test")
        self.assertFalse(fetch.allowed)
        self.assertEqual(fetch.reason, STOP_TOOL_CALLS)

        room = self._controller(tool_calls=4, db_path=str(Path(self._tmp.name) / "other.sqlite"))
        first = room.authorize_tool_call(BING_TOOL, command="same")
        room.commit_tool_call(first)
        second = room.may_auto_replay(FETCH_TOOL, command="https://example.test")
        self.assertFalse(second.allowed)
        allowed = room.may_auto_replay(BING_TOOL, command="same")
        self.assertTrue(allowed.allowed)
        room.commit_tool_call(allowed)
        hammer = room.may_auto_replay(BING_TOOL, command="same")
        self.assertFalse(hammer.allowed)
        self.assertIn("fingerprint", hammer.reason)
        self.assertNotEqual(
            call_fingerprint(BING_TOOL, command="same"),
            call_fingerprint(BING_TOOL, command="other"),
        )
        self.assertEqual(
            call_fingerprint("Bing_Search_Tool", command="same"),
            call_fingerprint("bing", command="same"),
        )

    def test_state_round_trip_keeps_the_db_path_and_not_the_connection(self) -> None:
        controller = self._controller()
        controller.handle_failure(self._missing_read())
        outcome = controller.begin_round(self._read_plans(), round_id="round-a")
        self.assertEqual(outcome.selected_node.tool, "export")
        controller.record_verified_success(
            ExecutionObservation(tool_name="export", environment="sandbox", tool_result="ok"),
            action="export",
            subgoal="export the record",
        )
        state = controller.export_state()
        blob = json.dumps(state)
        self.assertIn(self.db_path, blob)
        self.assertNotIn("_conn", blob)
        self.assertNotIn("sqlite3.Connection", blob)
        for value in state.values():
            self.assertNotIsInstance(value, sqlite3.Connection)

        restored = RecoveryController(db_path=str(Path(self._tmp.name) / "restored-unused.sqlite"))
        restored.restore_state(state)
        self.assertEqual(restored.memory.db_path, self.db_path)
        self.assertEqual(restored.graph.active_plan_ids(), ("p3",))
        self.assertEqual(restored.graph.round_id, "round-a")
        self.assertEqual(restored.budget.recovery_rounds_used, 1)
        planning = restored.planning_records()
        self.assertEqual(len(planning.verified_successes), 1)
        self.assertTrue(planning.pruning_constraints())
        self.assertFalse(planning.diagnostic_hypotheses)
        pickled = pickle.loads(pickle.dumps(restored))
        self.assertEqual(pickled.memory.db_path, self.db_path)
        self.assertIsNone(pickled.memory._conn)
        statuses = {item.evidence_status for item in pickled.memory.records()}
        self.assertIn(EvidenceStatus.VERIFIED_SUCCESS, statuses)
        pickled.memory.close()
        restored.memory.close()

    def test_budget_defaults_match_the_shared_budget_type(self) -> None:
        controller = self._controller()
        default = Budget()
        self.assertEqual(controller.budget.tool_calls, default.tool_calls)
        self.assertEqual(controller.budget.replans, default.replans)
        self.assertEqual(controller.budget.recovery_rounds, default.recovery_rounds)


if __name__ == "__main__":
    unittest.main()
