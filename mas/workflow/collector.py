"""Collect TirAgent rollouts + rewards without VERL / training GPU.

``workflow/`` must not import agentlightning. Mock path needs no LLM server.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .contracts import Trajectory, TrajectoryBatch
from .env_load import load_repo_dotenv
from .rewards import compute_outcome_reward, parse_alias_field
from .runtime import ExecutionService, LLMConfig
from .spec import load_spec
from uuid import uuid4

RewardFn = Callable[[Trajectory, Dict[str, Any]], float]

# Ensure Agent_Science_Infra/.env is visible before reading OPENAI_*.
load_repo_dotenv()


def default_reward_fn(traj: Trajectory, task: Dict[str, Any]) -> float:
    gold = str(task.get("answer") or "")
    source = str(task.get("source") or "gsm8k")
    aliases = parse_alias_field(task.get("answers"))
    golds = [gold] + [a for a in aliases if a and a != gold]
    pred = traj.final_answer
    format_ok = traj.format_ok
    if pred is not None and not format_ok:
        format_ok = True
    return compute_outcome_reward(
        pred,
        gold,
        source=source,
        aliases=golds,
        format_ok=format_ok,
        n_search=traj.n_search,
        n_python=traj.n_python,
    )


class Collector:
    """MAS data-layer collector: Trajectory + reward, no VERL."""

    def __init__(
        self,
        *,
        mock: bool = False,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_turns: int = 8,
        max_tokens: int = 1024,
        reward_fn: Optional[RewardFn] = None,
        n: int = 1,
        archive_root: Optional[str] = None,
        spec_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.mock = mock
        self.endpoint = endpoint or os_environ_endpoint()
        self.model = model or os_environ_model()
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.reward_fn = reward_fn or default_reward_fn
        self.n = max(1, int(n))
        self.archive_root = archive_root
        self.spec_path = spec_path
        self.api_key = api_key
        self.spec = load_spec(spec_path)

    def _service(self) -> ExecutionService:
        llm = None
        if not self.mock:
            import os

            mlen = os.environ.get("TIR_MAX_MODEL_LEN", "").strip()
            llm = LLMConfig(
                endpoint=self.endpoint or "",
                model=self.model,
                temperature=self.temperature,
                max_turns=self.max_turns,
                max_tokens=self.max_tokens,
                max_model_len=int(mlen) if mlen else None,
                enabled_tools=list(self.spec.tools),
                api_key=self.api_key,
            )
        return ExecutionService(
            mock=self.mock,
            spec=self.spec,
            llm=llm,
            archive_root=self.archive_root,
        )

    def collect_one(self, task: Dict[str, Any]) -> Trajectory:
        traj = self._service().run(task)
        traj.final_reward = float(self.reward_fn(traj, task))
        traj.meta = dict(traj.meta or {})
        traj.meta["rewarded"] = True
        traj.sync_tir_meta()
        return traj

    def collect(self, tasks: Iterable[Dict[str, Any]], *, n: Optional[int] = None) -> TrajectoryBatch:
        sampling = getattr(self.spec, "sampling", None)
        policy_n = int(getattr(sampling, "group_n", 0) or 0) if sampling is not None else 0
        mode = str(getattr(sampling, "mode", None) or "grpo_n") if sampling is not None else "grpo_n"
        # Explicit collect(..., n=) wins; else sampling.group_n when >1; else Collector.n.
        if n is not None:
            group_n = max(1, int(n))
        elif policy_n > 1:
            group_n = policy_n
        else:
            group_n = max(1, int(self.n))
        out: List[Trajectory] = []
        task_list = list(tasks)
        for task in task_list:
            group_id = str(task.get("id") or task.get("data_id") or uuid4().hex)
            for i in range(group_n):
                sample = dict(task)
                sample["_sample_index"] = i
                sample["_group_id"] = group_id
                traj = self.collect_one(sample)
                traj.meta["sample_index"] = i
                traj.meta["group_n"] = group_n
                traj.meta["group_id"] = group_id
                traj.meta["sampling_mode"] = mode
                traj.meta["is_branch"] = bool(traj.branch_parent_id) or bool(traj.resume_from)
                if traj.branch_points:
                    traj.meta["n_branch_points"] = len(traj.branch_points)
                out.append(traj)
        return TrajectoryBatch(
            trajectories=out,
            meta={
                "n_tasks": len(task_list),
                "group_n": group_n,
                "n_trajectories": len(out),
                "mean_reward": _mean_reward(out),
                "sampling_mode": mode,
            },
        )


def _mean_reward(trajs: Sequence[Trajectory]) -> float:
    vals = [float(t.final_reward) for t in trajs if t.final_reward is not None]
    return sum(vals) / len(vals) if vals else 0.0


def os_environ_endpoint() -> Optional[str]:
    import os

    load_repo_dotenv()
    return os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")


def os_environ_model() -> str:
    import os

    load_repo_dotenv()
    return os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "/root/autodl-tmp/LLM/Qwen3-4B"


def collect_with_mock(tasks: Iterable[Dict[str, Any]], *, n: int = 1) -> TrajectoryBatch:
    return Collector(mock=True, n=n).collect(tasks)
