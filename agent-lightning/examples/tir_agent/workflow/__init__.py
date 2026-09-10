"""TIR workflow layer: contracts + Archive + Collector (no agentlightning import).

Hub ReAct is the default Runtime. Topology Compiler walks graph agents
(route / message / feedback / tool_call) without generating LangGraph code.
"""

from .archive import Archive, dump_resume_with_archive, load_resume_messages
from .collector import Collector, collect_with_mock, default_reward_fn
from .compiler import compile_spec, trainable_agents
from .contracts import ArchiveRef, BranchPoint, MemoryItem, Snapshot, Trajectory, TrajectoryBatch
from .harness import HARNESS, Hypothesis, default_harness
from .memory import MemoryStore
from .plugins import REGISTRY, SkillResult
from .runtime import ExecutionService, LLMConfig, run_episode
from .spec import MASSpec, load_spec
from .train_signal import AdvantageSpec, LossSpec, TrainSignal, batch_to_train_signal

__all__ = [
    "AdvantageSpec",
    "Archive",
    "ArchiveRef",
    "BranchPoint",
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
    "dump_resume_with_archive",
    "load_resume_messages",
    "load_spec",
    "run_episode",
]
