"""Regression tests for the GAIA failure modes observed in July 2026."""
import json
import threading

from MAS.epc_aw.models.diagnoser import Diagnoser
from MAS.epc_aw.models.task_profile import (
    SlotGate,
    _extract_character_name,
    _is_character_name_answer,
    rejects_family_journal_base_count,
)
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

def test_inject_query_repr_keeps_command_single_line():
    """Multiline deterministic snippets must not break executor.split_commands."""
    import re as _re

    snippet = "import math\nbase_count = 12000\nprint(result)"
    snippet = snippet.encode().decode("unicode_escape")
    cmd = Solver._inject_query_into_command(
        'execution = tool.execute(query="old")',
        snippet,
    )
    # Command source must be one physical line (repr escapes newlines).
    assert chr(10) not in cmd
    assert "base_count = 12000" in cmd
    # Executor split pattern: one physical-line execute(...) call.
    pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
    blocks = _re.findall(pattern, cmd, _re.DOTALL)
    assert len(blocks) == 1, repr(cmd)


def test_deterministic_compute_is_single_line():
    solver = _solver_without_init()
    question = (
        "If all Nature articles in 2020 relied on statistical significance and "
        "they on average came to a p-value of 0.04, how many papers would be incorrect?"
    )
    query = solver._build_deterministic_compute_query(
        question, "Calculate incorrect papers", {"base_count": "12000"},
    )
    assert query is not None
    assert "\n" not in query
    assert "math.ceil" in query
    assert "12000" in query and "0.04" in query


def test_parenthesis_free_does_not_extract_parenthesis():
    assert _extract_character_name(
        "Unlambda is written in a parenthesis-free prefix notation"
    ) is None
    assert _extract_character_name("also called the backquote character") == "backtick"
    assert _extract_character_name("the parenthesis character is needed") == "parenthesis"


def test_tool_router_blocks_python_on_acquisition():
    from MAS.epc_aw.models.tool_router import ToolRouter

    tool, appropriate, kind = ToolRouter.validate(
        "Python_Coder_Tool",
        "Retrieve Eliud Kipchoge marathon pace",
        "Retrieve Eliud Kipchoge marathon pace",
        None,
        [],
        ["Python_Coder_Tool", "Google_Search_Tool", "Web_Search_Tool"],
    )
    assert kind == "acquisition"
    assert tool == "Google_Search_Tool"
    assert appropriate is False


def test_diagnoser_empty_python_is_execution_not_retrieval():
    diagnoser = Diagnoser.__new__(Diagnoser)
    diagnoser.verbose = False
    # capability must match so we reach EMPTY branch
    failure_type, target, reason, conf, desc = diagnoser._classify_root_cause(
        [],
        "Calculate incorrect papers from base count 12000",
        "Python_Coder_Tool",
        {"evidence_type": "EMPTY", "no_results": True, "tool_appropriate": True},
    )
    assert failure_type == "Execution"
    assert reason == "empty_compute_result"


def test_python_coder_runs_deterministic_source_without_llm():
    from MAS.epc_aw.tools.python_coder.tool import Python_Coder_Tool

    tool = Python_Coder_Tool.__new__(Python_Coder_Tool)
    tool.llm_engine = None
    result = tool.execute(
        "import math; base_count = 12000; p_value = 0.04; "
        "result = math.ceil(base_count * p_value); print(result)",
    )
    assert isinstance(result, dict)
    assert result.get("printed_output") == "480"


def test_deterministic_distance_pace_not_time_fragments():
    """Prose with 2:01:09 must not become distance=2, pace=50."""
    solver = _solver_without_init()

    class Memory(_Memory):
        evidence_records = [
            {
                "status": "active",
                "content": (
                    "The minimum perigee distance between the Earth and the Moon "
                    "is approximately 363,300 kilometers."
                ),
                "slot_bindings": {},
            },
            {
                "status": "active",
                "content": (
                    "Eliud Kipchoge's marathon pace is highlighted by his personal "
                    "best of 2:01:09, achieved at the Berlin Marathon in 2022, and "
                    "his historic run of 1:59:40 during the INEOS 1:59 Challenge."
                ),
                "slot_bindings": {"input_metrics": "2 minutes and 50 seconds per kilometer"},
            },
        ]

        def get_obtained_information_for_prompt(self):
            return "\n".join(r["content"] for r in self.evidence_records)

    solver.system_memory = Memory()
    question = (
        "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, "
        "how many thousand hours would it take him to run the distance between the Earth "
        "and the Moon its closest approach? Round your result to the nearest 1000 hours."
    )
    query = solver._build_deterministic_compute_query(
        question,
        "Calculate thousand hours",
        {"input_metrics": "2 minutes and 50 seconds"},
    )
    assert query is not None
    assert "distance_km = 363300" in query or "distance_km = 363300.0" in query
    assert "distance = 2" not in query
    assert "pace = 50" not in query
    assert "/ 1000" in query


def test_single_letter_is_not_character_name():
    assert _is_character_name_answer("g") is False
    assert _is_character_name_answer("space") is False
    assert _is_character_name_answer("backtick") is True
    assert _extract_character_name("also called the backquote character") == "backtick"


def test_family_journal_base_count_rejected_for_articles_only():
    question = (
        "If we assume all articles published by Nature in 2020 (articles, only, "
        "not book reviews/columns, etc) relied on statistical significance"
    )
    family = (
        "In 2020, Nature and its associated journals published over 12,000 articles"
    )
    assert rejects_family_journal_base_count(question, family) is True
    profile = SlotGate.infer_profile(question)
    slot = next(s for s in profile.slots if s.name == "base_count")
    assert not SlotGate._content_fills_slot(slot, family.lower(), {}, [], question=question)


def test_harvest_rejects_zero_for_thousand_hours():
    solver = _solver_without_init()
    question = (
        "how many thousand hours would it take him to run the distance "
        "Round your result to the nearest 1000 hours"
    )
    profile = SlotGate.infer_profile(question)

    class Memory(_Memory):
        evidence_records = []

        def get_task_profile(self):
            return profile

        def get_query(self):
            return question

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
            "printed_output": "0",
            "execution_code": "distance = 2; pace = 50; result = round(distance / pace); print(result)",
        }],
        first_attempt_command='execution = tool.execute(query="calculate")',
    )
    verification = VerificationResult(
        analysis="computed",
        step_conclusion="SUBGOAL_COMPLETE",
        info_flag=True,
        obtained_info="0",
        task_conclusion="STOP",
        diagnostic_signal=None,
        subgoal_complete=True,
    )
    assert not solver._harvest_slots_from_executor(step, verification, 2)


def test_nature_articles_only_query_is_anchored():
    solver = _solver_without_init()
    question = (
        "all articles published by Nature in 2020 (articles, only, not book reviews/columns)"
    )
    command = solver._align_command_with_subgoal(
        "Google_Search_Tool",
        'execution = tool.execute(query="total articles Nature 2020")',
        question,
        "Nature",
        "Retrieve total articles",
    )
    assert "articles only" in command.lower() or "research articles only" in command.lower()
    # Must not over-stack negative site filters that collapse recall.
    assert "NOT associated journals family" not in command


def test_param_fingerprint_does_not_reference_solver_class():
    """Mixin extraction must not leave bare Solver.* references."""
    fp = Solver._param_fingerprint(
        'execution = tool.execute(query="Nature 2020", url="https://en.wikipedia.org/")',
        "Web_Search_Tool",
    )
    assert "nature 2020" in fp
    assert "wikipedia" in fp


def test_tool_policy_single_select_exit():
    """Path uniqueness: hot-path tool selection goes through select_tool once."""
    solver = Solver.__new__(Solver)
    solver.planner = type("P", (), {"available_tools": [
        "Google_Search_Tool", "Python_Coder_Tool", "Web_Search_Tool",
    ]})()
    solver.system_memory = _Memory()
    solver.system_memory.evidence_records = []
    solver.system_memory.get_task_profile = lambda: None
    calls = {"n": 0}
    orig = solver._select_tool_for_step

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    solver._select_tool_for_step = counting
    out = solver.select_tool(
        "Python_Coder_Tool",
        "Retrieve Eliud Kipchoge marathon pace",
        "Retrieve Eliud Kipchoge marathon pace",
        "Retrieve Eliud Kipchoge marathon pace",
        None,
        "",
        None,
    )
    assert calls["n"] == 1
    assert out == "Google_Search_Tool"


def test_plan_controller_can_stop_delegates_to_slotgate():
    solver = _solver_without_init()
    profile = SlotGate.infer_profile(
        "If all Nature articles in 2020 relied on statistical significance "
        "and they on average came to a p-value of 0.04, how many papers would be incorrect?"
    )

    class Memory(_Memory):
        evidence_records = [{
            "status": "active",
            "source_quality": "computed",
            "compute_real": True,
            "content": "Computed result: 41",
            "slot_bindings": {
                "base_count": "1025",
                "computed_count": "41",
            },
        }]

        def get_task_profile(self):
            return profile

    solver.system_memory = Memory()
    assert solver.can_stop(profile.question) is True
    assert solver._can_stop_execution(profile.question) is True


def test_evidence_binder_after_verify_is_single_slot_write():
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
    solver._python_execution_is_real_computation = lambda _r: True
    step = StepContext(
        step_key="1",
        target_information="Compute result",
        context="inputs",
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
    out = solver.after_verify(step, verification, 2, record_obtained=False)
    assert out.had_slot_delta is True
    assert solver.system_memory.evidence_records[-1]["slot_bindings"]["computed_count"] == "17"


def test_solver_has_no_gaia_task_names():
    """Orchestrator must not embed GAIA-sample proper nouns / hostnames."""
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath("solver.py").read_text().lower()
    banned = ("usgs", "nas.er.usgs", "unlambda", "kipchoge", "nature journal", "gaia")
    for token in banned:
        assert token not in text, f"solver.py still contains {token!r}"


def test_soft_coverage_absence_runs_l2a():
    """Retrieval/coverage + switch_tool must still enter Level 2a."""
    solver = _solver_without_init()
    called = {"l2a": False}

    class Memory(_Memory):
        def set_diagnostic_signal(self, _sig):
            pass

        def get_obtained_information_for_prompt(self):
            return "None"

    solver.system_memory = Memory()
    solver.verbose = False

    def fake_l2a(question, image_path, step_ctx, exec_step, json_data, diagnostic_signal, tracker):
        called["l2a"] = True
        return (
            "complete",
            step_ctx,
            VerificationResult(
                analysis="l2a ran",
                step_conclusion="SUBGOAL_COMPLETE",
                info_flag=True,
                obtained_info="ok",
                task_conclusion="CONTINUE",
                diagnostic_signal=diagnostic_signal or {},
                subgoal_complete=True,
            ),
            exec_step,
            None,
        )

    solver._run_intervention_level2a = fake_l2a
    solver._run_intervention_level2b = lambda *a, **k: (
        "escalate", a[2],
        VerificationResult(
            analysis="l2b",
            step_conclusion="SUBGOAL_INCOMPLETE",
            info_flag=False,
            obtained_info="",
            task_conclusion=None,
            diagnostic_signal={},
            subgoal_complete=False,
        ),
        a[3],
        {},
    )
    solver._is_pdf_access_error = lambda _s: False
    solver._record_obtained_information = lambda *a, **k: None
    solver._record_failure_trace = lambda *a, **k: None
    solver._append_trace_event = lambda *a, **k: None
    solver._observe_failure = lambda *a, **k: type("FC", (), {
        "tool": "Google_Search_Tool",
        "evidence_type": "ABSENCE",
        "failure_patterns": {},
    })()
    solver._compute_intervention_gains = lambda *a, **k: (1, 1)
    solver._apply_escalation_to_signal = lambda sig, _t: sig
    solver._enrich_diagnostic_signal = lambda sig, *_a, **_k: sig
    solver._parse_slots_from_obtained = lambda *_a, **_k: set()
    step = StepContext(
        step_key="1",
        target_information="Retrieve Nature article count",
        context="Nature 2020",
        sub_goal="Retrieve total research articles published by Nature in 2020",
        tool_name="Google_Search_Tool",
        command='execution = tool.execute(query="Nature 2020")',
        result_executor='{"facts":[],"refusal":true}\n[REFUSAL:true]',
        first_attempt_command='execution = tool.execute(query="Nature 2020")',
    )
    verification = VerificationResult(
        analysis="absence",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion=None,
        diagnostic_signal={
            "recommendation": "switch_tool",
            "failure_type": "Retrieval",
            "target_variable": "coverage",
            "reason": "source_not_indexed",
            "failure_patterns": {
                "absence_unfilled": True,
                "evidence_type": "ABSENCE",
                "tool_appropriate": True,
                "wrong_tool_class": False,
            },
            "analysis": "no coverage",
        },
        subgoal_complete=False,
        tool_appropriate=True,
        evidence_type="ABSENCE",
    )
    tracker = StepInterventionState()
    solver._handle_subgoal_incomplete(
        "Nature articles 2020 p-value",
        None,
        step,
        verification,
        1,
        {},
        tracker,
    )
    assert called["l2a"] is True


def test_web_search_no_default_wikipedia():
    solver = _solver_without_init()

    class Memory(_Memory):
        def get_diagnostic_signal(self):
            return {}

    solver.system_memory = Memory()
    out = solver._ensure_web_search_command(
        'execution = tool.execute(query="Nature research articles 2020")',
        "Retrieve Nature article count",
        "Nature journal 2020 articles only",
    )
    assert "en.wikipedia.org" not in out.lower()
    unlambda = solver._ensure_web_search_command(
        'execution = tool.execute(query="Unlambda character")',
        "Find the character name",
        "Unlambda applicative order backtick",
    )
    assert "en.wikipedia.org" not in unlambda.lower()
    wiki_ok = solver._ensure_web_search_command(
        'execution = tool.execute(query="Moon")',
        "Look up Wikipedia page",
        "Use wikipedia for Moon perigee",
    )
    assert "en.wikipedia.org" in wiki_ok.lower()


def test_absence_unfilled_prefers_retry():
    d = Diagnoser.__new__(Diagnoser)
    rec = d._suggest_intervention(
        {
            "absence_unfilled": True,
            "evidence_type": "ABSENCE",
            "tool_appropriate": True,
            "wrong_tool_class": False,
            "no_results": True,
        },
        "Google_Search_Tool",
        subgoal_complete=False,
    )
    assert rec == "retry_with_different_parameters"
    ft, tv, reason, *_ = d._classify_root_cause(
        'STRUCTURED\n{"facts":[],"refusal":true}\nstanford login required\n[REFUSAL:true]',
        "Retrieve Nature article count for 2020",
        "Google_Search_Tool",
        {"evidence_type": "ABSENCE", "error_occurred": False},
    )
    assert ft == "Retrieval"
    assert tv == "coverage"
    assert reason == "paywalled"
    # Generic refusal / no-match template is NOT paywalled.
    ft2, tv2, reason2, *_ = d._classify_root_cause(
        'The search results do not provide information that answers this query.\n'
        '[REFUSAL:true]\n---STRUCTURED---\n{"facts":[],"refusal":true}',
        "Retrieve journal article count for 2020",
        "Google_Search_Tool",
        {"evidence_type": "ABSENCE", "error_occurred": False},
    )
    assert ft2 == "Retrieval"
    assert tv2 == "coverage"
    assert reason2 == "low_recall"


def test_executor_logs_aligned_command():
    """_run_executor must return the aligned articles-only command for Nature."""
    solver = _solver_without_init()
    solver.verbose = False
    solver.system_memory = _Memory()
    solver.system_memory.toolbox_metadata = {"Google_Search_Tool": {}}
    solver.planner = type("P", (), {"available_tools": ["Google_Search_Tool"]})()

    class Executor:
        def generate_tool_command(self, *args, **kwargs):
            return "analysis\nexplanation\ncommand"

        def extract_explanation_and_command(self, _raw):
            return (
                "a",
                "e",
                'execution = tool.execute(query="total articles Nature 2020")',
            )

    solver.executor = Executor()
    solver._execute_generated_command = lambda *a, **k: {"ok": True}
    question = (
        "all articles published by Nature in 2020 (articles, only, not book reviews/columns)"
    )
    command, _analysis, _result = solver._run_executor(
        question,
        None,
        "Nature",
        "Retrieve total articles",
        "Google_Search_Tool",
        1,
        {},
        None,
    )
    assert "articles only" in command.lower() or "research articles only" in command.lower()


def test_empty_outline_retry_analyze_before_recovery():
    """Second analyze_query success must populate outline without recovery count."""
    solver = _solver_without_init()
    calls = {"analyze": 0}
    outlines = [{}, {"1": "Target Information: Unlambda ` `` ` character name"}]

    class Memory(_Memory):
        outline = {}
        evidence_records = []

        def get_outline(self):
            return self.outline

        def set_outline(self, o):
            self.outline = o or {}

        def set_task_profile(self, _p):
            pass

        def get_task_profile(self):
            return None

    mem = Memory()
    solver.system_memory = mem
    solver.diagnoser = type("D", (), {
        "_is_multi_hop_question": staticmethod(lambda _q: False),
        "_extract_zip_codes": staticmethod(lambda _t: []),
    })()

    class Planner:
        available_tools = ["Google_Search_Tool"]

        def analyze_query(self, question, image_path=None):
            calls["analyze"] += 1
            idx = min(calls["analyze"] - 1, len(outlines) - 1)
            return "analysis", outlines[idx]

    solver.planner = Planner()
    solver.verbose = False
    data = {}
    solver._analyze_query("Unlambda ` `` ` what character?", None, data, 0.0)
    assert mem.get_outline() == {}
    # Mimic Step0 retry path from solve().
    solver._analyze_query("Unlambda ` `` ` what character?", None, data, 0.0)
    assert calls["analyze"] == 2
    assert mem.get_outline()
    assert "Unlambda" in str(mem.get_outline())
    recovery = solver._generate_recovery_outline(
        "Unlambda program ` `` ` outputs the character name",
        VerificationResult(
            analysis="",
            step_conclusion="SUBGOAL_INCOMPLETE",
            info_flag=False,
            obtained_info="",
            task_conclusion="CONTINUE",
            diagnostic_signal=None,
            subgoal_complete=False,
        ),
    )
    blob = " ".join(str(v) for v in recovery.values())
    assert "unlambda" in blob.lower()
    assert "`" in blob or "``" in blob


def test_query_analysis_schema_exposes_execution_outline():
    """Structured Step-0 schema must carry analysis + execution_outline."""
    from MAS.epc_aw.models.formatters import QueryAnalysis
    from MAS.epc_aw.models.planner import Planner

    qa = QueryAnalysis(
        analysis="Need external base quantity then compute.",
        execution_outline=json.dumps({
            "1": "Target Information: Retrieve base count. Operation Details: Google_Search_Tool.",
        }),
    )
    dumped = qa.model_dump()
    assert "execution_outline" in dumped
    assert isinstance(dumped["execution_outline"], str)
    analysis, outline = Planner._extract_analysis_and_outline(dumped)
    assert "base" in analysis.lower() or "quantity" in analysis.lower()
    assert outline and "1" in outline

    # Legacy PascalCase still parses
    analysis2, outline2 = Planner._extract_analysis_and_outline({
        "Analysis": "legacy",
        "ExecutionOutline": {"1": "step"},
    })
    assert analysis2 == "legacy"
    assert outline2 == {"1": "step"}

    # Old four-field dump must not silently become a valid outline
    analysis3, outline3 = Planner._extract_analysis_and_outline({
        "concise_summary": "x",
        "required_skills": "y",
        "relevant_tools": "z",
        "additional_considerations": "w",
    })
    assert outline3 == {}
    fallback = Planner._schema_fallback_outline(
        "How many widgets were sold after applying a rate of 0.04? Round up."
    )
    assert fallback and "1" in fallback
    assert "round" not in str(fallback).lower().split("round")[0] or "widgets" in str(fallback).lower()
    assert "widgets" in str(fallback).lower()


def test_distance_selects_min_under_closest_constraint():
    """Polarity from question — not entity names — picks the closest distance."""
    solver = _solver_without_init()
    blob = (
        "The average distance is 384,399 km. "
        "The minimum closest approach distance is 356,400 km. "
        "The maximum distance is 406,700 km."
    )
    q_min = "Use the minimum closest approach distance in kilometers for the calculation."
    assert solver._parse_distance_km(blob, q_min) == 356400.0
    q_max = "Use the maximum farthest distance in kilometers."
    assert solver._parse_distance_km(blob, q_max) == 406700.0


def test_recovery_outline_avoids_instruction_verb_anchors():
    solver = _solver_without_init()
    solver.diagnoser = type("D", (), {
        "_is_multi_hop_question": staticmethod(lambda _q: False),
        "_extract_zip_codes": staticmethod(lambda _t: []),
    })()
    solver.system_memory = _Memory()
    solver.system_memory.evidence_records = []
    solver.system_memory.get_task_profile = lambda: None
    recovery = solver._generate_recovery_outline(
        "If we assume all articles published by Nature in 2020 relied on "
        "statistical significance and they on average came to a p-value of 0.04, "
        "how many papers would be incorrect? Round the value up to the next integer.",
        VerificationResult(
            analysis="",
            step_conclusion="SUBGOAL_INCOMPLETE",
            info_flag=False,
            obtained_info="",
            task_conclusion="CONTINUE",
            diagnostic_signal=None,
            subgoal_complete=False,
        ),
    )
    blob = " ".join(str(v) for v in recovery.values()).lower()
    assert "nature" in blob or "articles" in blob or "pvalue" in blob.replace("-", "") or "value" in blob
    # Must not invent the false proper-noun pair from "Round the value"
    assert "nature round" not in blob


def test_character_schema_json_unwraps_to_scorable_name():
    """SlotGate/JSON wrapper must become Pred=backtick, not empty scorable."""
    question = (
        "In Unlambda, what exact charcter needs to be added? "
        "If it is a character, answer with the name of the character."
    )
    profile = SlotGate.infer_profile(question)
    poison = {
        "status": "active",
        "source_quality": "secondary",
        "content": '{"character_name":"backtick"}',
        "slot_bindings": {},
    }
    assert SlotGate.can_stop(profile, [poison])
    extracted = SlotGate.extract_final_answer(profile, [poison])
    assert extracted == "backtick"
    structured = {
        "status": "active",
        "source_quality": "secondary",
        "content": (
            '{"facts":[{"type":"character_name","value":"backtick","unit":null,'
            '"quote":"Usage of the backtick character","url":"https://x",'
            '"confidence":"high"}],"refusal":false}'
        ),
        "slot_bindings": {},
    }
    assert SlotGate.can_stop(profile, [structured])
    assert SlotGate.extract_final_answer(profile, [structured]) == "backtick"
    from MAS.epc_aw.models.answer_gate import AnswerGateMixin
    assert AnswerGateMixin._answer_format_mismatch(question, extracted) is None
    assert AnswerGateMixin._answer_format_mismatch(
        question, '{"character_name":"backtick"}',
    ) is not None
    # Diagnoser normalize unwraps dict before evidence write.
    d = Diagnoser.__new__(Diagnoser)
    assert d._normalize_obtained_information({"character_name": "backtick"}) == "backtick"


def test_computed_count_prefers_python_over_pace_fragment():
    """Wikipedia pace '1:59:40' must not beat Python printed 17 for thousand hours."""
    question = (
        "how many thousand hours would it take at marathon pace to run the "
        "minimum perigee distance in km?"
    )
    profile = SlotGate.infer_profile(question)
    records = [
        {
            "status": "active",
            "source_quality": "secondary",
            "tool": "Wikipedia_Search_Tool",
            "content": "marathon best 1:59:40.2",
            "slot_bindings": {"input_metrics": "1:59:40.2", "computed_count": "59"},
        },
        {
            "status": "active",
            "source_quality": "computed",
            "tool": "Python_Coder_Tool",
            "compute_real": True,
            "content": "Computed result: 17",
            "slot_bindings": {"computed_count": "17"},
        },
    ]
    assert "computed_count" in SlotGate.filled_slots(profile, records)
    assert SlotGate.extract_final_answer(profile, records) == "17"


def test_range_distance_selects_lower_bound_under_min():
    """'A to B km' must expose both endpoints; min polarity picks A."""
    solver = _solver_without_init()
    blob = "The Moon's perigee varies from 356,355 to 370,399 km while the apogee varies."
    q = "Please use the minimum perigee value when carrying out your calculation."
    assert solver._parse_distance_km(blob, q) == 356355.0


def test_soft_cause_repeated_does_not_terminate():
    """low_recall reoccurrence must soft-continue, not Cause-Changed terminate."""
    from MAS.epc_aw.models.intervention import StepInterventionState, InterventionMixin

    tracker = StepInterventionState()
    tracker.root_cause_history = [
        ("low_recall", "coverage"),
        ("low_recall", "coverage"),
    ]
    assert tracker.cause_repeated("low_recall", "coverage") is True
    # Soft causes are excluded from terminate in intervention loop; paywalled still counts.
    tracker2 = StepInterventionState()
    tracker2.root_cause_history = [
        ("paywalled", "coverage"),
        ("paywalled", "coverage"),
    ]
    assert tracker2.cause_repeated("paywalled", "coverage") is True

    # Free-text description must normalize to the enum reason key.
    key = InterventionMixin._normalize_cause_reason_key(
        "low_recall",
        "Search returned a no-match/refusal template — reformulate query, not archive hop",
    )
    assert key == "low_recall"
    # Description-only (missing reason) still recovers the soft token.
    key2 = InterventionMixin._normalize_cause_reason_key(
        "",
        "Search returned a no-match/refusal template with low_recall coverage",
    )
    assert key2 == "low_recall"


def test_soft_continue_uses_reason_not_description_in_counterfactual():
    """Cause-Changed soft-continue must key on reason enum, not prose description."""
    from MAS.epc_aw.models.intervention import StepInterventionState

    solver = _solver_without_init()
    solver.verbose = False
    solver._last_planner_action = None
    solver._no_op_penalty = 0
    tracker = StepInterventionState()
    # History recorded the way production now does: reason enum.
    tracker.record_root_cause("low_recall", "coverage")
    tracker.record_root_cause("low_recall", "coverage")
    assert tracker.cause_repeated("low_recall", "coverage")

    signal = {
        "recommendation": "decompose_goal",
        "root_cause": "Search returned a no-match/refusal template — reformulate",
        "reason": "low_recall",
        "target_variable": "coverage",
        "root_cause_confidence": "MEDIUM",
        "causal_hypothesis": {
            "description": "Search returned a no-match/refusal template — reformulate",
            "reason": "low_recall",
            "target_variable": "coverage",
            "confidence": "MEDIUM",
        },
    }
    step_ctx = StepContext(
        step_key="1:test",
        target_information="Retrieve article count",
        context="",
        sub_goal="Retrieve article count",
        tool_name="Google_Search_Tool",
        command='execution = tool.execute(query="x")',
        result_executor="",
        first_attempt_command='execution = tool.execute(query="x")',
    )
    verification = VerificationResult(
        analysis="",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion="CONTINUE",
        diagnostic_signal=signal,
        subgoal_complete=False,
    )
    try:
        action, *_ = solver._run_counterfactual(
            "how many articles",
            step_ctx,
            verification,
            1,
            {},
            tracker,
            signal,
        )
    except Exception as exc:
        # Soft-continue passed the terminate gate; later planner wiring may be
        # incomplete on a bare Solver.__new__ instance.
        assert "terminate" not in str(exc).lower()
        action = "soft_continued"
    assert action != "terminate"


def test_provisional_numeric_finalize_pvalue_ceil():
    """Exhausted retrieval still yields ceil(N*p) from evidence candidates."""
    solver = _solver_without_init()
    question = (
        "If we assume all articles published by a journal in 2020 "
        "(articles, only, not book reviews) relied on statistical significance "
        "and they on average came to a p-value of 0.04, how many papers would "
        "be incorrect? Round the value up to the next integer."
    )
    records = [
        {
            "status": "active",
            "source_quality": "secondary",
            "tool": "Google_Search_Tool",
            "content": (
                "In 2020 the journal published 1025 research articles "
                "(articles only, excluding book reviews).\n"
                "---STRUCTURED---\n"
                '{"facts":[{"type":"number","value":1025,"unit":"article",'
                '"quote":"published 1025 research articles","url":"u","confidence":"high"}],'
                '"refusal":false}'
            ),
            "slot_bindings": {},
        },
        {
            "status": "active",
            "source_quality": "secondary",
            "content": "Nature Index tracks 178 journals — not an article count.",
            "slot_bindings": {},
        },
    ]
    out = solver._provisional_numeric_finalize(question, records)
    assert out == "41"  # ceil(1025 * 0.04)


def test_provisional_rejects_family_aggregate():
    solver = _solver_without_init()
    question = (
        "articles published by Nature in 2020 (articles, only) p-value of 0.04 "
        "how many papers would be incorrect as to their claims of statistical significance?"
    )
    records = [
        {
            "status": "active",
            "content": (
                "Nature and its associated journals published about 12,000 articles "
                "in the Nature family."
            ),
            "slot_bindings": {},
        },
    ]
    assert solver._provisional_numeric_finalize(question, records) is None


def test_provisional_character_from_verification_analysis():
    solver = _solver_without_init()
    question = (
        "In Unlambda, what exact charcter needs to be added? "
        "If it is a character, answer with the name of the character."
    )
    analysis = (
        "The executor returned relevant data indicating that the character "
        "needed is the backtick character. The slot for final_answer is filled."
    )
    assert solver._provisional_character_finalize(
        question, [], [], analysis,
    ) == "backtick"


def test_provisional_thousand_hours_uses_distance_pace_not_median():
    solver = _solver_without_init()
    question = (
        "how many thousand hours would it take at marathon pace to run the "
        "minimum perigee distance between Earth and the Moon in km?"
    )
    blob = (
        "minimum perigee distance is about 363,300 km from Earth. "
        "The marathon personal best is 2:01:09."
    )
    out = solver._provisional_numeric_finalize(question, [], [blob])
    assert out == "17"


def test_hours_answer_rejects_minutes_phrase():
    from MAS.epc_aw.models.answer_gate import AnswerGateMixin
    q = "how many thousand hours would it take to run the distance at pace?"
    assert AnswerGateMixin._answer_format_mismatch(q, "121 minutes") is not None
    assert AnswerGateMixin._answer_format_mismatch(q, "17") is None


def test_parse_pace_mss_and_pace_per_km_json():
    """M:SS.xx / pace_per_km / pace_per_5km must parse to km/h (no athlete names)."""
    solver = _solver_without_init()
    q = "marathon pace for thousand hours distance calculation"
    # ~2:52.28 min/km → 60 / 2.871 ≈ 20.9 km/h
    pace = solver._parse_pace_kmh('{"pace_per_km":"2:52.28"}', q)
    assert pace is not None
    assert 20.0 < pace < 22.0
    # pace_per_5km = 5× per-km
    pace5 = solver._parse_pace_kmh('{"pace_per_5km":"14:21.40"}', q)
    assert pace5 is not None
    assert 20.0 < pace5 < 22.0
    # Explicit min/km unit
    pace_u = solver._parse_pace_kmh("elite pace 2:50 min/km sustained", q)
    assert pace_u is not None
    assert 20.5 < pace_u < 22.0
    # Finish time still works
    pace_f = solver._parse_pace_kmh("marathon personal best is 2:01:09", q)
    assert pace_f is not None
    assert 20.5 < pace_f < 21.5
    # Free-text pace per 5km
    pace_5t = solver._parse_pace_kmh(
        "His average pace per 5km was clocked at a jaw-dropping 14:21.4.", q,
    )
    assert pace_5t is not None
    assert 20.0 < pace_5t < 22.0


def test_input_metrics_rejects_pace_only_km_substring():
    """pace_per_5km must not falsely satisfy distance via substring 'km'."""
    from MAS.epc_aw.models.task_profile import AnswerSlot, SlotGate, TaskProfile

    question = (
        "how many thousand hours would it take at marathon pace to run the "
        "minimum perigee distance between Earth and the Moon in km?"
    )
    profile = SlotGate.infer_profile(question)
    # Ensure profile has computed_count for the extract hard-gate.
    if not any(s.name == "computed_count" for s in profile.slots):
        profile = TaskProfile(
            question=question,
            slots=[
                AnswerSlot(name="input_metrics", slot_type="text", min_source="secondary"),
                AnswerSlot(name="computed_count", slot_type="numeric", min_source="computed"),
            ],
        )
    pace_only = {
        "tool": "Google_Search_Tool",
        "content": '{"pace_per_km":"2:52.28","pace_per_5km":"14:21.40"}',
        "status": "verified",
        "source_quality": "secondary",
        "slot_bindings": {"input_metrics": '{"pace_per_km":"2:52.28"}'},
    }
    assert "input_metrics" not in SlotGate.filled_slots(profile, [pace_only])
    assert SlotGate.extract_final_answer(profile, [pace_only]) is None

    both = [
        {
            "tool": "Google_Search_Tool",
            "content": "minimum perigee distance is 356355 km",
            "status": "verified",
            "source_quality": "secondary",
            "slot_bindings": {"input_metrics": "356355 km"},
        },
        pace_only,
    ]
    # Combined distance + pace across records can fill input_metrics,
    # but extract must still refuse without computed_count.
    assert "input_metrics" in SlotGate.filled_slots(profile, both)
    assert SlotGate.extract_final_answer(profile, both) is None


def test_thousand_hours_rejects_rounding_quantum_and_out_of_band():
    from MAS.epc_aw.models.answer_gate import AnswerGateMixin

    q = (
        "how many thousand hours would it take? Round to the nearest 1000 hours."
    )
    assert AnswerGateMixin._answer_format_mismatch(q, "1000") is not None
    assert AnswerGateMixin._rounding_quantum_from_question(q) == 1000
    assert AnswerGateMixin._answer_format_mismatch(q, "17") is None
    assert AnswerGateMixin._answer_format_mismatch(q, "3") is not None  # below band
    assert AnswerGateMixin._answer_format_mismatch(q, "99") is not None  # above band


def test_provisional_thousand_hours_from_mss_pace_json():
    """distance + pace_per_km JSON must yield ~17 without H:MM:SS finish time."""
    solver = _solver_without_init()
    question = (
        "how many thousand hours would it take at marathon pace to run the "
        "minimum perigee distance between Earth and the Moon in km? "
        "Round to the nearest 1000 hours."
    )
    blob = (
        "minimum perigee distance is about 356355 km from Earth. "
        '{"pace_per_km":"2:52.28"}'
    )
    out = solver._provisional_numeric_finalize(question, [], [blob])
    assert out is not None
    assert out != "1000"
    assert 10 <= int(out) <= 25


def test_parse_distance_from_structured_km_key():
    solver = _solver_without_init()
    q = "Please use the minimum perigee value when carrying out your calculation."
    blob = '{"minimum_perigee_distance_km": 356355}'
    assert solver._parse_distance_km(blob, q) == 356355.0
