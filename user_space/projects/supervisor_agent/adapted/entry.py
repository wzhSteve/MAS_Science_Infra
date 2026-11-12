"""Run a supervised Smolagents team inside one Science Studio window."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
VENDORED = PROJECT / "upload" / "SupervisorAgent" / "smolagents_SMAS" / "src"


def _smolagents() -> Any:
    loaded = sys.modules.get("smolagents")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if not origin or not Path(origin).resolve().is_relative_to(VENDORED):
            raise RuntimeError("A different smolagents is already loaded; use a separate Python process.")
        return loaded
    sys.path.insert(0, str(VENDORED))
    try:
        import smolagents

        return smolagents
    finally:
        sys.path.remove(str(VENDORED))


def _build_model(config: dict[str, Any], stats: dict[str, Any]) -> Any:
    smolagents = _smolagents()
    try:
        from workflow.llm_scope import current_rollout_llm
    except ModuleNotFoundError as exc:
        if exc.name not in {"workflow", "workflow.llm_scope"}:
            raise
        binding = {}
    else:
        binding = current_rollout_llm()

    endpoint = binding.get("endpoint") or os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model_id = binding.get("model") or os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
    if not endpoint or not model_id:
        raise ValueError("Configure a rollout model or set OPENAI_API_BASE_URL and MODEL_NAME.")

    class MeasuredModel(smolagents.OpenAIServerModel):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            stats["model_calls"] += 1
            message = super().generate(*args, **kwargs)
            if message.token_usage is not None:
                stats["input_tokens"] += message.token_usage.input_tokens
                stats["output_tokens"] += message.token_usage.output_tokens
            else:
                stats["usage_complete"] = False
            return message

    return MeasuredModel(
        model_id=model_id,
        api_base=endpoint,
        api_key=binding.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        client_kwargs={"timeout": float(config["request_timeout"]), "max_retries": 0},
        temperature=float(config["temperature"]),
        max_tokens=int(config["max_tokens"]),
    )


def _supervisor_callback(supervisor: Any, question: str, stats: dict[str, Any]) -> Any:
    def review(step: Any, agent: Any) -> None:
        observation = step.observations or ""
        tool_name = step.tool_calls[0].name if step.tool_calls else ""
        local_task = agent.task if agent.name == "search_agent" else ""
        supervisor.record_step(
            step=step, agent_name=agent.name, local_task=local_task,
            parent_agent_name="manager_agent" if agent.name == "search_agent" else None,
        )
        if tool_name != "final_answer" and "## Search Results" in observation:
            step.observations = supervisor.filter_search_results(observation)

        should_review = (
            step.error is not None
            or len(observation) > 3000
            or supervisor._check_for_inefficiency(agent.name)[0]
            or "### 1. Task outcome (short version):" in observation
        )
        if tool_name == "web_search" and step.error is None and len(observation) < 5000:
            should_review = False
        if should_review:
            stats["supervisor_reviews"] += 1
            supervisor.supervise_and_correct(
                step=step, agent_name=agent.name, local_task=local_task, global_task=question,
            )

    return review


def _run_team(question: str, config: dict[str, Any], model: Any, stats: dict[str, Any]) -> Any:
    smolagents = _smolagents()

    def web_tools() -> list[Any]:
        if not config["web_tools"]:
            return []
        return [smolagents.GoogleSearchTool(provider="serper"), smolagents.VisitWebpageTool()]

    def count_step(step: Any) -> None:
        stats["action_steps"] += 1

    callbacks = [count_step]
    if config["supervisor_enabled"]:
        verifier = smolagents.ToolCallingAgent(
            tools=web_tools(), model=model, max_steps=int(config["verifier_max_steps"]),
            step_callbacks=[count_step], verbosity_level=0, name="verification_agent",
        )
        supervisor = smolagents.SupervisorAgent(model=model, summary_model=model, verification_agent=verifier)
        callbacks.append(_supervisor_callback(supervisor, question, stats))

    worker = smolagents.ToolCallingAgent(
        tools=web_tools(), model=model, max_steps=int(config["worker_max_steps"]),
        name="search_agent", description="Investigate a subquestion and report your findings.",
        provide_run_summary=True, verbosity_level=0, step_callbacks=callbacks,
    )
    manager = smolagents.ToolCallingAgent(
        tools=[], model=model, max_steps=int(config["manager_max_steps"]),
        managed_agents=[worker], name="manager_agent", verbosity_level=0, step_callbacks=callbacks,
    )
    return manager.run(question, return_full_result=True)


def solve(question: str, *, supervisor_enabled: bool | None = None) -> dict[str, Any]:
    import yaml

    question = question.strip()
    if not question:
        raise ValueError("SupervisorAgent requires a non-empty question.")
    config = yaml.safe_load((PROJECT / "contracts" / "runtime.yaml").read_text(encoding="utf-8"))
    if supervisor_enabled is not None:
        config["supervisor_enabled"] = supervisor_enabled
    stats = {
        "supervisor_enabled": config["supervisor_enabled"],
        "supervisor_reviews": 0,
        "action_steps": 0,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "usage_complete": True,
    }
    model = _build_model(config, stats)
    started = perf_counter()
    result = _run_team(question, config, model, stats)
    return {
        "answer": str(result.output) if result.output is not None else "",
        "state": result.state,
        "elapsed_seconds": perf_counter() - started,
        **stats,
    }


class UserAgent:
    def run_window(self, message: Any) -> dict[str, Any]:
        payload = message.get("payload", message) if isinstance(message, dict) else message.payload
        question = str(payload.get("question") or payload.get("input") or "")
        result = solve(question)
        answer = result.pop("answer")
        return {
            "kind": "plan_step",
            "payload": {
                "next": "", "args": {}, "sub_goal": "", "done": True,
                "answer": answer, "trace": result,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--no-supervisor", action="store_true")
    args = parser.parse_args()
    print(json.dumps(solve(args.question, supervisor_enabled=False if args.no_supervisor else None), ensure_ascii=False))
