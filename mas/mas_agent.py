"""LitAgent wrapper: train MAS_structagent with Agent-lightning + VERL.

LLM endpoint/model come from ``resources["main_llm"]`` (VERL), not from
``tir_agent/.env`` (dmxapi). All ChatOpenAI clients inside Solver/tools are
forced onto that endpoint via ``llm_runtime_override``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, cast

import agentlightning as agl

from rl.rewards import compute_mas_outcome_reward, parse_alias_field

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_MAS_TOOLS: List[str] = [
    "Base_Generator_Tool",
    "Python_Coder_Tool",
    "Wikipedia_Search_Tool",
    "Web_Search_Tool",
    "Google_Search_Tool", 
]


def bootstrap_mas_package() -> Path:
    """Register ``MAS_structagent`` as importable package name ``MAS``."""
    mas_root = HERE / "MAS_structagent"
    if not mas_root.is_dir():
        raise FileNotFoundError(f"MAS_structagent not found at {mas_root}")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    existing = sys.modules.get("MAS")
    if existing is None:
        pkg = types.ModuleType("MAS")
        pkg.__file__ = str(mas_root / "__init__.py")
        pkg.__path__ = [str(mas_root)]  # type: ignore[attr-defined]
        sys.modules["MAS"] = pkg
    elif not getattr(existing, "__path__", None):
        existing.__path__ = [str(mas_root)]  # type: ignore[attr-defined]
    return mas_root


@contextmanager
def bind_training_llm_env(*, endpoint: str, model: str, api_key: str) -> Iterator[None]:
    """Temporarily bind OPENAI_* / MODEL_* so leaky clients cannot hit .env dmxapi."""
    keys = (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "MODEL_Name",
        "MODEL_NAME",
        "SERVER_MODEL",
        "MAS_OPENAI_COMPATIBLE",
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["OPENAI_API_BASE_URL"] = endpoint
        os.environ["OPENAI_API_BASE"] = endpoint
        os.environ["OPENAI_BASE_URL"] = endpoint
        os.environ["OPENAI_API_KEY"] = api_key or "dummy"
        os.environ["MODEL_Name"] = model
        os.environ["MODEL_NAME"] = model
        server_models = [m.strip() for m in (prev.get("SERVER_MODEL") or "").split(",") if m.strip()]
        if model not in server_models:
            server_models.append(model)
        os.environ["SERVER_MODEL"] = ",".join(server_models)
        os.environ["MAS_OPENAI_COMPATIBLE"] = "1"
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class LitMASAgent(agl.LitAgent[Dict[str, Any]]):
    """One parquet row → one ``solver.solve()`` episode → ``emit_reward``."""

    def __init__(
        self,
        max_steps: int = 8,
        max_time: int = 600,
        max_tokens: int = 2048,
        val_temperature: Optional[float] = None,
        enabled_tools: Optional[Sequence[str]] = None,
        root_cache_dir: str = "mas_solver_cache",
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.val_temperature = val_temperature
        self.enabled_tools = list(enabled_tools) if enabled_tools is not None else list(DEFAULT_MAS_TOOLS)
        self.root_cache_dir = root_cache_dir

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        bootstrap_mas_package()
        from MAS.epc_aw.engine.runtime_override import llm_runtime_override
        from MAS.epc_aw.solver import construct_solver

        question = str(task["question"])
        ground_truth = str(task.get("answer") or "")
        source = str(task.get("source") or "gsm8k")
        start = time.time()
        llm = cast(agl.LLM, resources["main_llm"])
        rollout_id = rollout.rollout_id

        if rollout.mode == "train":
            temperature = float(llm.sampling_parameters.get("temperature", 0.7))
        else:
            temperature = (
                self.val_temperature
                if self.val_temperature is not None
                else float(llm.sampling_parameters.get("temperature", 0.0))
            )

        endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        model = llm.model
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        os.environ.setdefault("MAS_MAX_COMPLETION_TOKENS", str(self.max_tokens))

        logger.info(
            "[MAS Rollout %s] source=%s endpoint=%s model=%s q=%s",
            rollout_id,
            source,
            endpoint,
            model,
            question[:180],
        )

        prediction: Optional[str] = None
        result: Dict[str, Any] = {}
        try:
            with llm_runtime_override(
                base_url=endpoint,
                model=model,
                api_key=api_key,
                openai_compatible=True,
            ), bind_training_llm_env(endpoint=endpoint, model=model, api_key=api_key):
                tool_engine = [model] * len(self.enabled_tools)
                solver = construct_solver(
                    llm_engine_name=model,
                    enabled_tools=self.enabled_tools,
                    tool_engine=tool_engine,
                    output_types="final,direct",
                    max_steps=self.max_steps,
                    max_time=self.max_time,
                    max_tokens=self.max_tokens,
                    root_cache_dir=os.path.join(self.root_cache_dir, str(rollout_id)),
                    verbose=False,
                    temperature=temperature,
                    n=1,
                )
                result = solver.solve(question) or {}
            prediction = (
                result.get("direct_output")
                or result.get("final_output")
                or (result.get("state") or {}).get("final_answer")
                or ""
            )
            if prediction is not None:
                prediction = str(prediction).strip() or None
        except Exception as e:
            logger.exception("[MAS Rollout %s] Solver failed: %s", rollout_id, e)
            prediction = None
            result = {}

        reward = compute_mas_outcome_reward(prediction, task, result)
        logger.info(
            "[MAS Rollout %s] src=%s pred=%r gt=%r completed=%s R=%.3f t=%.2fs",
            rollout_id,
            source,
            prediction,
            ground_truth,
            result.get("completed"),
            reward,
            time.time() - start,
        )
        agl.emit_reward(reward)
        return None


def debug_mas_agent() -> None:
    """Trainer.dev smoke test without VERL."""
    bootstrap_mas_package()
    records: List[Dict[str, Any]] = [
        {
            "id": "debug-mas-py-1",
            "question": "What is 12 multiplied by 15?",
            "answer": "180",
            "answers": ["180"],
            "source": "gsm8k",
        },
        {
            "id": "debug-mas-qa-1",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "answers": ["Paris"],
            "source": "hotpot",
        },
    ]

    endpoint = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if not endpoint:
        raise RuntimeError("Set OPENAI_API_BASE (or OPENAI_BASE_URL) to an OpenAI-compatible endpoint.")

    model = os.environ.get("OPENAI_MODEL", os.environ.get("MODEL", "/root/autodl-tmp/LLM/Qwen3-4B"))
    print(f"debug_mas_agent endpoint={endpoint} model={model} n={len(records)}")
    trainer = agl.Trainer(
        n_runners=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=endpoint,
                model=model,
                sampling_parameters={"temperature": 0.3},
            )
        },
    )
    trainer.dev(LitMASAgent(max_steps=6), records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    debug_mas_agent()
