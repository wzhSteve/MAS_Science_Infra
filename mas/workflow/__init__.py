"""TIR workflow layer: contracts + Archive + Collector (no agentlightning import).

Hub ReAct is the default Runtime. Topology Compiler walks graph agents
(route / message / feedback / tool_call) without generating LangGraph code.
"""

from .archive import Archive, dump_resume_with_archive, load_resume_messages
from .collector import Collector, collect_with_mock, default_reward_fn
from .compiler import compile_spec, trainable_agents
from .contracts import ArchiveRef, BranchPoint, BranchSite, MemoryItem, SamplePolicy, Snapshot, Trajectory, TrajectoryBatch
from .harness import HARNESS, Hypothesis, default_harness
from .memory import MemoryStore
from .plugins import REGISTRY, SkillResult
from .runtime import ExecutionService, LLMConfig, run_episode
from .spec import MASSpec, load_spec
from .active_set import ActiveSetConfig, ActiveSetSession, dump_local_expansion, load_local_expansion

__all__ = [
    "ActiveSetConfig",
    "ActiveSetSession",
    "AdvantageSpec",
    "Archive",
    "ArchiveRef",
    "BranchPoint",
    "BranchSite",
    "Collector",
    "compile_spec",
    "ExecutionService",
    "HARNESS",
    "Hypothesis",
    "LLMConfig",
    "LossSpec",
    "MASSpec",
    "MemoryItem",
    "MemoryStore",
    "REGISTRY",
    "SamplePolicy",
    "SkillResult",
    "Snapshot",
    "TrainSignal",
    "Trajectory",
    "TrajectoryBatch",
    "trainable_agents",
    "batch_to_train_signal",
    "collect_with_mock",
    "default_harness",
    "default_reward_fn",
    "dump_local_expansion",
    "dump_resume_with_archive",
    "load_local_expansion",
    "load_resume_messages",
    "load_spec",
    "run_episode",
]


def __getattr__(name: str):
    if name in ("AdvantageSpec", "LossSpec", "TrainSignal", "batch_to_train_signal"):
        from .train_signal import AdvantageSpec, LossSpec, TrainSignal, batch_to_train_signal

        return {
            "AdvantageSpec": AdvantageSpec,
            "LossSpec": LossSpec,
            "TrainSignal": TrainSignal,
            "batch_to_train_signal": batch_to_train_signal,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
