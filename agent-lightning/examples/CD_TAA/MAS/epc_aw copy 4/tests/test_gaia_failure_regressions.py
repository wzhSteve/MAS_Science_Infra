"""Regression tests for the GAIA failure modes observed in July 2026."""
import threading

from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.task_profile import SlotGate, _extract_character_name
from MAS.epc_aw.solver import (
    Solver,
    StepContext,
    StepInterventionState,
    VerificationResult,
)
from MAS.epc_aw.tools.wikipedia_search.tool import (
    Select_Relevant_Queries,
    select_relevant_queries,
)


class _Memory:
    def __init__(self, obtained="None"):
        self.obtained = obtained

    def get_obtained_information_for_prompt(self):
        return self.obtained


def _solver_without_init():
    solver = Solver.__new__(Solver)
    solver.system_memory = _Memory()
    return solver


def test_candidate_set_executes_on_main_thread():
    solver = _solver_without_init()
    seen_threads = []

    class Executor:
        def execute_tool_command(self, tool_name, command):
            seen_threads.append(threading.current_thread())
            return {"ok": command}

    solver.executor = Executor()
    solver._validate_command = lambda _tool, _command: None
    data = {}

    results = solver._execute_candidate_set(
        "Google_Search_Tool",
        [
            'execution = tool.execute(query="first")',
            'execution = tool.execute(query="second")',
        ],
        2,
        data,
    )

    assert len(results) == 2
    assert seen_threads == [threading.main_thread(), threading.main_thread()]


def test_candidate_verification_uses_separate_intervention_budget():
    solver = _solver_without_init()
    solver._candidate_passes_prefilter = lambda _tool, _result: True
    solver._run_verification = lambda *args, **kwargs: VerificationResult(
        analysis="failed",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion=None,
        diagnostic_signal={},
        subgoal_complete=False,
    )
    tracker = StepInterventionState()
    step = StepContext(
        step_key="1",
        target_information="Retrieve a value",
        context="context",
        sub_goal="Retrieve a value",
        tool_name="Google_Search_Tool",
        command='execution = tool.execute(query="value")',
        result_executor=[],
        first_attempt_command='execution = tool.execute(query="value")',
    )

    winner, final_exec_step = solver._verify_candidate_set(
        "question",
        None,
        step,
        [("a", ["one"]), ("b", ["two"])],
        tracker,
        exec_step=7,
    )

    assert winner is None
    assert final_exec_step == 7
    assert tracker.intervention_attempts == 2


def test_compute_command_includes_verified_numeric_inputs():
    solver = _solver_without_init()
    solver.system_memory = _Memory('{"article_count": 8000, "p_value": 0.04}')

    command = solver._align_command_with_subgoal(
        "Python_Coder_Tool",
        'execution = tool.execute(query="statistical significance")',
        "How many false positives?",
        "Nature articles",
        "Calculate the result.",
    )

    assert "8000" in command
    assert "0.04" in command
    assert "prefer math" in command


def test_deterministic_compute_uses_slot_values():
    solver = _solver_without_init()

    class Memory(_Memory):
        evidence_records = [{
            "status": "active",
            "slot_bindings": {"base_count": "8000"},
        }]

    solver.system_memory = Memory()
    question = (
        "If all Nature articles in 2020 relied on statistical significance and "
        "they on average came to a p-value of 0.04, how many papers would be incorrect?"
    )
    command = solver._align_command_with_subgoal(
        "Python_Coder_Tool",
        'execution = tool.execute(query="compute")',
        question,
        "Nature articles",
        "Calculate incorrect papers from base count and p-value.",
    )

    assert "base_count = 8000" in command
    assert "p_value = 0.04" in command
    assert "math.ceil" in command


def test_level2b_compute_routes_to_python_coder():
    solver = Solver.__new__(Solver)
    solver.planner = type("Planner", (), {
        "available_tools": [
            "Python_Coder_Tool", "Google_Search_Tool", "Web_Search_Tool",
        ],
    })()
    solver.executor = type("Executor", (), {
        "generate_tool_command": lambda *args, **kwargs: "command",
        "extract_explanation_and_command": lambda raw: ("", "", 'execution = tool.execute(query="x")'),
    })()
    solver.system_memory = _Memory()
    solver.system_memory.toolbox_metadata = {"Python_Coder_Tool": {}}
    solver._validate_command = lambda *_args, **_kwargs: None
    solver._sanitize_command_for_tool = lambda _tool, cmd: cmd
    solver._align_command_with_subgoal = lambda *_args, **kwargs: 'execution = tool.execute(query="x")'
    solver._execute_generated_command = lambda *_args, **_kwargs: "error"
    solver._candidate_passes_prefilter = lambda *_args, **_kwargs: False
    solver._run_verification = lambda *args, **kwargs: VerificationResult(
        analysis="",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion=None,
        diagnostic_signal={},
        subgoal_complete=False,
    )
    solver.MAX_VERIFY_PER_ROUND = 1
    solver._suggest_alternative_tool = Solver._suggest_alternative_tool.__get__(solver, Solver)

    step = StepContext(
        step_key="3",
        target_information="Compute final answer with Python_Coder_Tool",
        context="inputs ready",
        sub_goal="Calculate incorrect papers",
        tool_name="Google_Search_Tool",
        command='execution = tool.execute(query="bad")',
        result_executor="error",
        first_attempt_command='execution = tool.execute(query="bad")',
    )
    tracker = StepInterventionState()
    seen_tools = []

    class RecordingExecutor:
        def generate_tool_command(self, question, image_path, context, sub_goal, tool_name, *args, **kwargs):
            seen_tools.append(tool_name)
            return "raw"

        def extract_explanation_and_command(self, raw):
            return "", "", 'execution = tool.execute(query="x")'

    solver.executor = RecordingExecutor()

    solver._run_intervention_level2b(
        "question",
        None,
        step,
        4,
        {},
        {"suggested_tool": "Web_Search_Tool"},
        tracker,
    )

    assert seen_tools == ["Python_Coder_Tool"]


def test_retrieval_command_is_reanchored_to_current_subgoal():
    solver = _solver_without_init()

    command = solver._align_command_with_subgoal(
        "Google_Search_Tool",
        'execution = tool.execute(query="minimum perigee distance of the Moon")',
        "question",
        "context",
        "Retrieve Eliud Kipchoge marathon pace in minutes",
    )

    assert "Eliud Kipchoge marathon pace" in command
    assert "perigee" not in command


def test_exact_character_query_keeps_output_and_code_anchors():
    solver = _solver_without_init()
    question = (
        'In Unlambda, what exact charcter is needed to output "For penguins"?\n'
        '`r`````.F.o.r.si'
    )

    command = solver._align_command_with_subgoal(
        "Google_Search_Tool",
        'execution = tool.execute(query="Unlambda correction")',
        question,
        "Unlambda",
        "Identify the exact character needed",
    )

    assert "For penguins" in command
    assert ".F.o.r.si" in command


def test_exact_character_profile_rejects_code_fragment():
    question = (
        "In Unlambda, what exact charcter needs to be added? "
        "If it is a character, answer with the name of the character."
    )
    profile = SlotGate.infer_profile(question)
    assert profile.slots[0].slot_type == "character_name"

    bad_record = {
        "status": "active",
        "source_quality": "secondary",
        "content": '`λx. x "For penguins"`',
        "slot_bindings": {"final_answer": '`λx. x "For penguins"`'},
    }
    good_record = {
        "status": "active",
        "source_quality": "secondary",
        "content": "backtick",
        "slot_bindings": {"final_answer": "backtick"},
    }

    assert not SlotGate.can_stop(profile, [bad_record])
    assert SlotGate.can_stop(profile, [good_record])
    assert Solver._answer_format_mismatch(question, bad_record["content"])
    assert Solver._answer_format_mismatch(question, "backtick") is None
    assert _extract_character_name("also called the backquote character") == "backtick"


def test_compute_result_harvests_computed_slot():
    solver = _solver_without_init()
    profile = SlotGate.infer_profile("How many hours? Calculate and round the result.")

    class Memory(_Memory):
        evidence_records = []

        def get_task_profile(self):
            return profile

        def add_evidence_record(self, content, **kwargs):
            self.evidence_records.append({"content": content, **kwargs})
            return True

    solver.system_memory = Memory()
    solver._python_execution_is_real_computation = lambda _result: True
    step = StepContext(
        step_key="1",
        target_information="Compute result",
        context="distance and pace",
        sub_goal="Calculate total time",
        tool_name="Python_Coder_Tool",
        command='execution = tool.execute(query="calculate")',
        result_executor=[{
            "printed_output": "17",
            "execution_code": "result = 363300 / 21",
        }],
        first_attempt_command='execution = tool.execute(query="calculate")',
    )
    verification = VerificationResult(
        analysis="computed",
        step_conclusion="SUBGOAL_COMPLETE",
        info_flag=True,
        obtained_info="17",
        task_conclusion="STOP",
        diagnostic_signal=None,
        subgoal_complete=True,
    )

    assert solver._harvest_slots_from_executor(step, verification, 2)
    record = solver.system_memory.evidence_records[-1]
    assert record["slot_bindings"]["computed_count"] == "17"
    assert record["source_quality"] == "computed"
    assert record["compute_real"] is True


def test_wikipedia_structured_selection_and_fallback():
    class StructuredEngine:
        def generate(self, prompt, response_format):
            return Select_Relevant_Queries(
                matched_queries=["Lunar distance"],
                matched_query_ids=[1],
            )

    queries, ids = select_relevant_queries(
        "minimum lunar distance",
        ["Moon", "Lunar distance", "Apollo"],
        StructuredEngine(),
    )
    assert queries == ["Lunar distance"]
    assert ids == [1]

    class EmptyEngine:
        def generate(self, prompt, response_format):
            return {"matched_queries": [], "matched_query_ids": []}

    queries, ids = select_relevant_queries(
        "minimum lunar distance",
        ["Lunar distance", "Apollo"],
        EmptyEngine(),
    )
    assert ids == [0]
    assert queries == ["Lunar distance"]


def test_diagnoser_accepts_substantive_other_pages():
    diagnoser = Diagnoser.__new__(Diagnoser)
    result = [{
        "relevant_pages (to the query)": [],
        "other_pages (may be irrelevant to the query)": [{
            "title": "Lunar distance",
            "abstract": (
                "The actual Earth-Moon distance varies through the orbit. "
                "The smallest distance is 356352.93 km according to the cited table."
            ),
        }],
    }]

    assert diagnoser._has_usable_result(result)
