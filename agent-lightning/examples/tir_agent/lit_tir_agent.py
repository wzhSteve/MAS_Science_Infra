"""LitTirAgent + Trainer.dev smoke (requires agentlightning)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, cast

import agentlightning as agl

from tir_agent import ig_deltas_from_messages
from workflow.archive import dump_resume_with_archive, register_archive, Archive
from workflow.collector import default_reward_fn
from workflow.env_load import load_repo_dotenv
from workflow.memory import MemoryStore
from workflow.runtime import LLMConfig, TirRunner, apply_verifier_feedback, episode_to_trajectory, run_episode
from workflow.spec import load_spec

load_repo_dotenv()

agl.setup_logging(apply_to=[__name__])
logger = logging.getLogger(__name__)


class LitTirAgent(agl.LitAgent[Dict[str, Any]]):
    """Training wrapper: shared run_episode + canonical RewardFn + emit_*."""

    def __init__(
        self,
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 8,
        max_tokens: int = 1024,
        max_model_len: Optional[int] = None,
        multi_tool_bonus: float = 0.1,
        spec_path: Optional[str] = None,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        env_len = os.environ.get("TIR_MAX_MODEL_LEN", "").strip()
        self.max_model_len = int(max_model_len or (env_len or 0) or 0) or None
        self.multi_tool_bonus = multi_tool_bonus
        self.spec = load_spec(spec_path)

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = str(task["question"])
        ground_truth = str(task.get("answer") or "")
        source = str(task.get("source") or "gsm8k")
        start = time.time()
        llm = cast(agl.LLM, resources["main_llm"])
        rollout_id = rollout.rollout_id
        resume_parent = str(task.get("resume_parent_id") or "")
        task_run = dict(task)
        task_run["_rollout_id"] = rollout_id

        if not task_run.get("resume_messages") and task_run.get("resume_from"):
            try:
                from workflow.archive import branch_point_to_resume_task_fields
                from workflow.contracts import BranchPoint

                raw = task_run.get("resume_from")
                bp = raw if isinstance(raw, BranchPoint) else BranchPoint.model_validate(raw)
                task_run.update(branch_point_to_resume_task_fields(bp))
                resume_parent = str(task_run.get("resume_parent_id") or resume_parent)
            except Exception as e:
                logger.warning("[Rollout] resume_from resolve failed: %s", e)

        if rollout.mode == "train":
            temperature = float(llm.sampling_parameters.get("temperature", 0.7))
        else:
            temperature = (
                self.val_temperature
                if self.val_temperature is not None
                else float(llm.sampling_parameters.get("temperature", 0.0))
            )

        endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        request_logprobs = os.environ.get("TIR_REQUEST_LOGPROBS", "").strip().lower() in ("1", "true", "yes")
        handler = self.tracer.get_langchain_handler()
        cfg = LLMConfig(
            endpoint=endpoint,
            model=llm.model,
            temperature=temperature,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_model_len=self.max_model_len,
            request_logprobs=request_logprobs,
            enabled_tools=list(self.spec.tools),
            langchain_callbacks=[handler] if handler else None,
        )
        logger.info("[Rollout %s] source=%s q=%s", rollout_id, source, question[:180])
        arch = register_archive(Archive())
        mem = MemoryStore()
        raw = run_episode(task_run, cfg, arch, spec=self.spec, memory=mem)
        raw = apply_verifier_feedback(task_run, raw, cfg, arch, self.spec, mem, TirRunner())
        traj = episode_to_trajectory(
            task_run, raw, arch, collector="lit_tir_agent", spec=self.spec, memory=mem
        )
        if self.multi_tool_bonus != 0.1:
            # still canonical formula; bonus is an argument of RewardFn via traj fields
            pass
        reward = float(default_reward_fn(traj, task_run))
        n_search = traj.n_search
        n_python = traj.n_python
        format_ok = traj.format_ok
        prediction = traj.final_answer
        hashes = list(raw.obs_hashes)
        golds = [ground_truth]
        ig_list: List[float] = []
        if raw.lc_messages is not None:
            from workflow.rewards import parse_alias_field

            aliases = parse_alias_field(task_run.get("answers"))
            golds = [ground_truth] + [a for a in aliases if a and a != ground_truth]
            ig_list = ig_deltas_from_messages(raw.lc_messages, golds)
        logger.info(
            "[Rollout %s] src=%s pred=%r gt=%r fmt=%s R=%.3f search=%d py=%d t=%.2fs",
            rollout_id,
            source,
            prediction,
            ground_truth,
            format_ok,
            reward,
            n_search,
            n_python,
            time.time() - start,
        )
        agl.emit_reward(reward)
        archive_id = ""
        snapshot_id = ""
        tir_algo = os.environ.get("TIR_ALGO", "grpo").strip().lower()
        dump_ok = os.environ.get("TIR_DUMP_RESUME", "").strip().lower() in ("1", "true", "yes") or tir_algo in (
            "arpo",
            "aepo",
        )
        if dump_ok and not raw.error:
            try:
                dumped = dump_resume_with_archive(
                    rollout_id,
                    {
                        "messages": raw.branch_messages or raw.messages,
                        "h_root": raw.h_root,
                        "h_tool": raw.h_tool,
                        "consecutive_high": raw.consecutive_high,
                    },
                    archive=arch,
                )
                archive_id = str(dumped.get("archive_id") or "")
                snapshot_id = str(dumped.get("snapshot_id") or "")
            except Exception as e:
                logger.warning("[Rollout %s] dump_resume failed: %s", rollout_id, e)
        try:
            agl.emit_annotation(
                {
                    "tir.source": source,
                    "tir.n_search": int(n_search),
                    "tir.n_python": int(n_python),
                    "tir.format_ok": bool(format_ok),
                    "tir.obs_hashes": "|".join(hashes),
                    "tir.ig_deltas": "|".join(f"{x:.6f}" for x in ig_list),
                    "tir.h_root": float(raw.h_root),
                    "tir.h_tool": float(raw.h_tool),
                    "tir.consecutive_high": int(raw.consecutive_high),
                    "tir.resume_parent_id": resume_parent,
                    "tir.archive_id": archive_id,
                    "tir.snapshot_id": snapshot_id,
                }
            )
        except Exception as e:
            logger.warning("[Rollout %s] emit_annotation failed: %s", rollout_id, e)
        return None


def debug_tir_agent() -> None:
    """Trainer.dev smoke test: 4 items that should trigger python / search tools."""
    records: List[Dict[str, Any]] = [
        {
            "id": "debug-py-1",
            "question": "What is 12 multiplied by 15? Use the python tool.",
            "answer": "180",
            "answers": ["180"],
            "source": "gsm8k",
        },
        {
            "id": "debug-py-2",
            "question": "What is the square root of 256? Use execute_python.",
            "answer": "16",
            "answers": ["16"],
            "source": "gsm8k",
        },
        {
            "id": "debug-qa-1",
            "question": "What is the capital of France? Look it up with wikipedia or web search.",
            "answer": "Paris",
            "answers": ["Paris"],
            "source": "hotpot",
        },
        {
            "id": "debug-qa-2",
            "question": "Who wrote the play Hamlet? Use wikipedia_search or web_search before answering.",
            "answer": "William Shakespeare",
            "answers": ["William Shakespeare", "Shakespeare"],
            "source": "hotpot",
        },
    ]

    endpoint = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if not endpoint:
        raise RuntimeError("Set OPENAI_API_BASE (or OPENAI_BASE_URL) to an OpenAI-compatible endpoint.")

    model = os.environ.get("OPENAI_MODEL", os.environ.get("MODEL", "/root/autodl-tmp/LLM/Qwen3-4B"))
    print(f"debug_tir_agent endpoint={endpoint} model={model} n={len(records)}")
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
    trainer.dev(LitTirAgent(), records)


if __name__ == "__main__":
    debug_tir_agent()
