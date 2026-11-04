"""Wire MAS-ProVe Best-of-N + Judge into pure_pev windows."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from prove_search import sample_candidates, select_best
from prove_types import PROVE_DEFAULTS, ProveDecision


def load_prove_settings(runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(PROVE_DEFAULTS)
    block = (runtime or {}).get("prove") if isinstance(runtime, dict) else None
    if isinstance(block, dict):
        cfg.update(block)

    enabled_env = os.environ.get("PROVE_ENABLED", "").strip().lower()
    if enabled_env in ("1", "true", "yes", "on"):
        cfg["enabled"] = True
    elif enabled_env in ("0", "false", "no", "off"):
        cfg["enabled"] = False

    n_env = os.environ.get("PROVE_N", "").strip()
    if n_env.isdigit():
        cfg["n"] = max(1, int(n_env))

    gran = os.environ.get("PROVE_GRANULARITY", "").strip().lower()
    if gran in ("plan", "plan_and_command", "iteration"):
        cfg["granularity"] = gran

    temp_env = os.environ.get("PROVE_CANDIDATE_TEMPERATURE", "").strip()
    if temp_env:
        try:
            cfg["candidate_temperature"] = float(temp_env)
        except ValueError:
            pass

    backend = os.environ.get("PROVE_JUDGE_BACKEND", "").strip().lower()
    if backend in ("inline", "tcp"):
        cfg["judge_backend"] = backend

    # iteration maps to plan-level selection on this PEV host
    if str(cfg.get("granularity") or "") == "iteration":
        cfg["granularity"] = "plan"

    cfg["n"] = max(1, int(cfg.get("n") or 1))
    cfg["enabled"] = bool(cfg.get("enabled"))
    return cfg


def plan_site_enabled(settings: Dict[str, Any]) -> bool:
    if not settings.get("enabled"):
        return False
    gran = str(settings.get("granularity") or "plan")
    return gran in ("plan", "plan_and_command", "iteration")


def command_site_enabled(settings: Dict[str, Any]) -> bool:
    if not settings.get("enabled"):
        return False
    return str(settings.get("granularity") or "") == "plan_and_command"


def _memory_actions_summary(memory: Any, *, limit: int = 8) -> str:
    actions = getattr(memory, "actions", None)
    if not isinstance(actions, dict) or not actions:
        return ""
    keys = sorted(
        actions.keys(),
        key=lambda k: int(str(k).rsplit(" ", 1)[-1]) if str(k).rsplit(" ", 1)[-1].isdigit() else 0,
    )
    lines: List[str] = []
    for key in keys[-limit:]:
        action = actions.get(key)
        if not isinstance(action, dict):
            lines.append(f"{key}: {action}")
            continue
        tool = action.get("tool_name") or action.get("tool") or ""
        sub = str(action.get("sub_goal") or "")[:200]
        result = str(action.get("result") or "")[:400]
        lines.append(f"{key} tool={tool} sub_goal={sub}\nresult={result}")
    return "\n\n".join(lines)


def build_prove_context(
    *,
    question: str,
    memory: Any,
    query_analysis: str = "",
    mode: str = "full_memory",
) -> str:
    if mode == "step_only":
        return f"Question: {question}"
    parts = [f"Question: {question}"]
    if query_analysis:
        parts.append(f"Query analysis:\n{str(query_analysis)[:1200]}")
    outline = getattr(memory, "outline", None)
    if outline:
        parts.append(f"Outline: {outline}")
    obtained = getattr(memory, "obtained_information", None)
    if obtained:
        parts.append(f"Obtained information: {obtained}")
    summary = _memory_actions_summary(memory)
    if summary:
        parts.append(f"Executed steps so far:\n{summary}")
    return "\n\n".join(parts)


@contextmanager
def _temporary_temperature(engine: Any, temperature: float) -> Iterator[None]:
    if engine is None or not hasattr(engine, "temperature"):
        yield
        return
    previous = float(getattr(engine, "temperature", 0.0) or 0.0)
    try:
        engine.temperature = float(temperature)
        yield
    finally:
        engine.temperature = previous


def _make_judge_llm_call(engine: Any) -> Callable[[str, str], str]:
    def _call(system: str, user: str) -> str:
        if engine is None:
            return ""
        # Prefer generate(system_prompt=...) surface used by LocalChatEngine.
        if hasattr(engine, "generate"):
            return str(engine.generate(user, system_prompt=system) or "")
        return str(engine(user, system_prompt=system) or "")

    return _call


def select_plan_step(
    *,
    settings: Dict[str, Any],
    question: str,
    memory: Any,
    query_analysis: str,
    sampler: Callable[[], str],
    judge_engine: Any,
    planner_engine: Any = None,
) -> ProveDecision:
    """Sample N next-step texts and Judge-select the best."""
    if not plan_site_enabled(settings):
        text = sampler()
        return ProveDecision(
            enabled=False,
            n=1,
            best_idx=0,
            rankings=[0],
            skipped_reason="disabled",
            site="plan",
            candidates=[text],
        )

    n = int(settings["n"])
    context = build_prove_context(
        question=question,
        memory=memory,
        query_analysis=query_analysis,
        mode=str(settings.get("context_mode") or "full_memory"),
    )
    cand_temp = float(settings.get("candidate_temperature") or 0.7)

    def _sample(_i: int) -> str:
        with _temporary_temperature(planner_engine, cand_temp):
            return str(sampler() or "")

    candidates = sample_candidates(_sample, n, parallel=False)
    if n <= 1:
        return ProveDecision(
            enabled=True,
            n=1,
            best_idx=0,
            rankings=[0],
            skipped_reason="n_eq_1",
            site="plan",
            candidates=[candidates[0].text if candidates else ""],
            context=context,
        )

    return select_best(
        candidates,
        question=question,
        context=context,
        task_type=str(settings.get("task_type") or "qa"),
        site="plan",
        judge_backend=str(settings.get("judge_backend") or "inline"),
        llm_call=_make_judge_llm_call(judge_engine),
        tcp_host=str(settings.get("tcp_host") or "127.0.0.1"),
        tcp_port=int(settings.get("tcp_port") or 5555),
    )


def select_tool_command(
    *,
    settings: Dict[str, Any],
    question: str,
    memory: Any,
    query_analysis: str,
    sampler: Callable[[], str],
    judge_engine: Any,
    executor_engine: Any = None,
) -> ProveDecision:
    """Optional Best-of-N over ``generate_tool_command`` only (not tool execution)."""
    if not command_site_enabled(settings):
        text = sampler()
        return ProveDecision(
            enabled=False,
            n=1,
            best_idx=0,
            rankings=[0],
            skipped_reason="command_site_off",
            site="command",
            candidates=[text],
        )

    n = int(settings["n"])
    context = build_prove_context(
        question=question,
        memory=memory,
        query_analysis=query_analysis,
        mode=str(settings.get("context_mode") or "full_memory"),
    )
    cand_temp = float(settings.get("candidate_temperature") or 0.7)

    def _sample(_i: int) -> str:
        with _temporary_temperature(executor_engine, cand_temp):
            return str(sampler() or "")

    candidates = sample_candidates(_sample, n, parallel=False)
    if n <= 1:
        return ProveDecision(
            enabled=True,
            n=1,
            best_idx=0,
            rankings=[0],
            skipped_reason="n_eq_1",
            site="command",
            candidates=[candidates[0].text if candidates else ""],
            context=context,
        )

    return select_best(
        candidates,
        question=question,
        context=context,
        task_type=str(settings.get("task_type") or "qa"),
        site="command",
        judge_backend=str(settings.get("judge_backend") or "inline"),
        llm_call=_make_judge_llm_call(judge_engine),
        tcp_host=str(settings.get("tcp_host") or "127.0.0.1"),
        tcp_port=int(settings.get("tcp_port") or 5555),
    )
