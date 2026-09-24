"""LitTirAgent + Trainer.dev smoke (requires agentlightning)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, cast

import agentlightning as agl

from tir_agent import ig_deltas_from_messages
from workflow.active_set import (
    ActiveSetConfig,
    ActiveSetSession,
    dump_local_expansion,
    expansion_payload_from_result,
)
from workflow.archive import dump_resume_with_archive, register_archive, Archive
from workflow.collector import default_reward_fn
from workflow.env_load import load_repo_dotenv
from workflow.llm_diagnostics import log_model_route
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
        self._logged_model_route = False

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

                raw_bp = task_run.get("resume_from")
                bp = raw_bp if isinstance(raw_bp, BranchPoint) else BranchPoint.model_validate(raw_bp)
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
        if not self._logged_model_route:
            log_model_route(f"runner rollout={rollout_id}", endpoint, llm.model)
            self._logged_model_route = True
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

        expand = bool(task_run.get("expand_in_runner")) and int(task_run.get("sampling_budget") or 1) > 1
        branch_local_count = 0
        session_metrics: Dict[str, Any] = {}

        def _run_one(t: Dict[str, Any]):
            raw_one = run_episode(t, cfg, arch, spec=self.spec, memory=mem)
            return apply_verifier_feedback(t, raw_one, cfg, arch, self.spec, mem, TirRunner())

        if expand:
            sites = None
            raw_sites = task_run.get("branch_sites")
            if raw_sites is not None:
                from workflow.contracts import BranchSite

                sites = [s if isinstance(s, BranchSite) else BranchSite.model_validate(s) for s in raw_sites]
            ready_batch = bool(task_run.get("ready_batch", False))
            local_expand = bool(task_run.get("force_local_expand", False)) or (
                ready_batch and str(task_run.get("collect_mode") or "").lower() in ("mock", "collect", "1", "true")
            )
            sess = ActiveSetSession(
                ActiveSetConfig(
                    group_budget=max(1, int(task_run.get("sampling_budget") or 1)),
                    beam_size=max(1, int(task_run.get("tir_beam_size") or 2)),
                    max_branch_depth=max(1, int(task_run.get("max_branch_depth") or 2)),
                    branch_probability=float(task_run.get("tir_branch_probability") or 0.5),
                    entropy_weight=float(task_run.get("tir_entropy_weight") or 0.5),
                    entropy_threshold=float(task_run.get("tir_entropy_threshold") or 0.15),
                    use_official_arpo_gate=bool(task_run.get("tir_use_official_arpo_gate", True)),
                    execute_local=local_expand,
                    parallel_local=ready_batch,
                    run_probes=True,
                    sites=sites,
                    event_kind=str(task_run.get("branch_event_kind") or "after_tool"),
                    agent_id=str(task_run.get("branch_agent_id") or "") or None,
                    tool_id=str(task_run.get("branch_tool_id") or "") or None,
                    verifier_ok=task_run.get("verifier_ok"),
                    tool_ok=task_run.get("tool_ok"),
                    final_failed=bool(task_run.get("final_failed") or False),
                    h_branch=task_run.get("h_branch"),
                    strategy=str(
                        task_run.get("sampling_strategy") or "configured_gate"
                    ),
                )
            )
            if ready_batch and local_expand and bool(task_run.get("use_scheduler", True)):
                from workflow.active_set_scheduler import ActiveSetScheduler

                def _llm_step(payload: Dict[str, Any]) -> Dict[str, Any]:
                    # Episode-granularity step: run remaining episode once (scheduler still
                    # batches READY leaves concurrently via its thread pool).
                    t = dict(payload.get("task") or {})
                    if payload.get("messages"):
                        t["resume_messages"] = payload["messages"]
                    raw_one = _run_one(t)
                    return {
                        "messages": list(getattr(raw_one, "branch_messages", None) or getattr(raw_one, "messages", None) or []),
                        "tool_calls": [],
                        "done": True,
                        "raw_partial": raw_one,
                    }

                def _tool_step(payload: Dict[str, Any], tool_calls: List[Any]) -> Dict[str, Any]:
                    return {"messages": list(payload.get("messages") or []), "done": False}

                sched = ActiveSetScheduler(
                    sess.config,
                    llm_step=_llm_step,
                    tool_step=_tool_step,
                    max_workers=min(8, max(1, int(task_run.get("sampling_budget") or 1))),
                )
                as_result = sched.run(task_run, parent_rollout_id=rollout_id)
            else:
                as_result = sess.run(task_run, _run_one, parent_rollout_id=rollout_id)
            raw = as_result.completed[0].raw
            branch_local_count = int(as_result.branch_local_count)
            session_metrics = dict(as_result.metrics)
            try:
                dump_local_expansion(rollout_id, expansion_payload_from_result(as_result))
            except Exception as e:
                logger.warning("[Rollout %s] dump_local_expansion failed: %s", rollout_id, e)
        else:
            raw = _run_one(task_run)
        traj = episode_to_trajectory(
            task_run, raw, arch, collector="lit_tir_agent", spec=self.spec, memory=mem
        )
        reward = float(default_reward_fn(traj, task_run))
        n_search = traj.n_search
        n_python = traj.n_python
        format_ok = traj.format_ok
        prediction = traj.final_answer
        hashes = list(raw.obs_hashes)
        golds = [ground_truth]
        ig_list: List[float] = []
        if raw.lc_messages is not None:
            from rl.rewards import parse_alias_field

            aliases = parse_alias_field(task_run.get("answers"))
            golds = [ground_truth] + [a for a in aliases if a and a != ground_truth]
            ig_list = ig_deltas_from_messages(raw.lc_messages, golds)
        logger.info(
            "[Rollout %s] src=%s pred=%r gt=%r fmt=%s R=%.3f search=%d py=%d branch_local=%d t=%.2fs",
            rollout_id,
            source,
            prediction,
            ground_truth,
            format_ok,
            reward,
            n_search,
            n_python,
            branch_local_count,
            time.time() - start,
        )
        agl.emit_reward(reward)
        archive_id = ""
        snapshot_id = ""
        tir_algo = os.environ.get("TIR_ALGO", "grpo").strip().lower()
        dump_ok = os.environ.get("TIR_DUMP_RESUME", "").strip().lower() in ("1", "true", "yes") or tir_algo in (
            "arpo",
            "aepo",
            "rae",
        )
        if dump_ok and not raw.error:
            try:
                window_snapshots = list(getattr(raw, "window_snapshots", None) or [])
                dumped = dump_resume_with_archive(
                    rollout_id,
                    {
                        # schema 0.3: prefer window snapshots, fall back to legacy branch_messages
                        "messages": raw.branch_messages or raw.messages,
                        "window_snapshots": window_snapshots,
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
                    "tir.branch_local_count": int(branch_local_count),
                    "tir.expand_in_runner": bool(expand),
                    "tir.session_n_plans": int(session_metrics.get("n_plans") or 0),
                    "tir.ready_batch_size": float(session_metrics.get("ready_batch_size") or 0),
                    "tir.tool_wait_skew": float(session_metrics.get("tool_wait_skew") or 0),
                    "tir.rollout_role": str(task_run.get("rollout_role") or ("child" if resume_parent else "parent")),
                    "tir.site_id": str(task_run.get("site_id") or ""),
                    "tir.sampling_strategy": str(task_run.get("sampling_strategy") or ""),
                    "tir.window_id": str(task_run.get("window_id") or task_run.get("event_id") or ""),
                    "tir.boundary_snapshot_ref": str(task_run.get("snapshot_ref") or ""),
                    "tir.sampling_decision": str(
                        (task_run.get("decision") or {}).get("reason") or ""
                    ),
                    "tir.action_key": str(task_run.get("action_key") or ""),
                    "tir.reward_scheme": str(task_run.get("reward_scheme") or ""),
                    "tir.resume_boundary": int(task_run.get("resume_boundary") or 0),
                    "tir.verdict": str(task_run.get("verdict") or "none"),
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
