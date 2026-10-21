"""Best-effort tree recording at the training boundary, without importing AGL."""

from __future__ import annotations

import logging
import os
import asyncio
import json
import math
import sys
from contextlib import asynccontextmanager, suppress
from aiohttp import ClientError
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from workflow.rollout_tree import RolloutTreeArchive
from workflow.rollout_results import TERMINAL, read_results

logger = logging.getLogger(__name__)
TREE_ENV_KEYS = ("TIR_ROLLOUT_TREE_DIR", "TIR_EXPERIMENT_ID", "TIR_RUN_ID", "TIR_EXECUTION_TREE_VERSION")


def _store_closed(error: BaseException) -> bool:
    text = str(error)
    return "Cannot connect to host" in text or "Connect call failed" in text


class RolloutTreeRecorder:
    def __init__(self, context: Mapping[str, str] | None = None):
        context = context if context is not None else os.environ
        self.archive: RolloutTreeArchive | None = None
        self.validated_attempts: dict[str, tuple[int, str | None]] = {}
        self._reconcile_offset = 0
        if not context.get("TIR_ROLLOUT_TREE_DIR"):
            return
        try:
            if not all(context.get(key) for key in TREE_ENV_KEYS[:3]):
                raise ValueError("Incomplete rollout tree run identity")
            self.archive = RolloutTreeArchive(
                Path(context["TIR_ROLLOUT_TREE_DIR"]),
                context["TIR_EXPERIMENT_ID"], context["TIR_RUN_ID"],
                schema_version=int(context.get("TIR_EXECUTION_TREE_VERSION", "2")),
            )
        except (OSError, ValueError) as error:
            logger.error("RolloutTree initialization failed for run %s: %s", context.get("TIR_RUN_ID"), error)

    def update(self, operation: Callable[[RolloutTreeArchive], None]) -> None:
        if self.archive is None:
            return
        try:
            operation(self.archive)
        except (OSError, ValueError, TypeError) as error:
            self.archive.report_error(str(error))
        finally:
            self.flush()

    def initial(self, samples: Mapping[str, dict[str, Any]], mode: str) -> None:
        def apply(archive: RolloutTreeArchive) -> None:
            for rollout_id, sample in samples.items():
                archive.record(rollout_id, sample, mode)
        self.update(apply)

    def plans(self, sample: dict[str, Any], parent: str, plans: list[dict[str, Any]], remaining: int) -> None:
        self.update(lambda archive: archive.plans(sample, parent, plans, remaining))

    def submitted(self, rollouts: Sequence[Any], samples: Sequence[dict[str, Any]]) -> None:
        def apply(archive: RolloutTreeArchive) -> None:
            if len(rollouts) != len(samples):
                raise ValueError("Store enqueue response length differs from submitted requests")
            # Store guarantees request order. Pair only actual submissions, never plan-tree nodes.
            for rollout, sample in zip(rollouts, samples):
                metadata = rollout.metadata or {}
                if metadata.get("data_id") != sample.get("data_id"):
                    raise ValueError(f"Store group mismatch for {rollout.rollout_id}")
                if sample.get("plan_id") and metadata.get("plan_id") != sample["plan_id"]:
                    raise ValueError(f"Store plan mismatch for {rollout.rollout_id}")
                archive.record(str(rollout.rollout_id), sample, "train", "independent_fill")
        self.update(apply)

    def release_batch(self) -> None:
        if self.archive:
            self.archive.release_batch()
        self.validated_attempts.clear()
        self._reconcile_offset = 0

    def flush(self) -> None:
        if self.archive:
            for event in self.archive.flush():
                try:
                    sys.stdout.write(json.dumps({"__rollout_tree_event__": event}) + "\n")
                    sys.stdout.flush()
                except OSError as error:
                    self.archive.report_error(f"Tree event notification failed: {error}")

    @staticmethod
    def observation(rollout: Any, attempt: Any = None) -> dict[str, Any]:
        attempt = attempt or getattr(rollout, "attempt", None)
        return {
            "attempt_id": attempt.attempt_id if attempt else None,
            "sequence": attempt.sequence_id if attempt else 0,
            "store_status": rollout.status,
            "attempt_status": attempt.status if attempt else None,
            "started_at": attempt.start_time if attempt else rollout.start_time,
            "ended_at": attempt.end_time if attempt else rollout.end_time,
        }

    def completed(self, rollout: Any, sample: dict[str, Any], spans: list[Any],
                  reward: float | None, training_status: str, attempt: Any = None) -> None:
        if self.archive is None:
            return
        observation = self.observation(rollout, attempt)
        previous = self.validated_attempts.get(rollout.rollout_id)
        if previous and observation["sequence"] == previous[0] and observation["attempt_id"] != previous[1]:
            self.archive.report_error(f"Conflicting validated attempt for {rollout.rollout_id}")
            self.flush()
            return
        if previous is None or observation["sequence"] > previous[0]:
            self.validated_attempts[rollout.rollout_id] = (observation["sequence"], observation["attempt_id"])

        def apply(archive: RolloutTreeArchive) -> None:
            archive.observe(rollout.rollout_id, sample, rollout.mode or "train", **observation)
            if reward is not None and not math.isfinite(reward):
                raise ValueError("Non-finite rollout reward")
            results = read_results(spans, rollout.rollout_id, observation["attempt_id"])
            archive.observe(
                rollout.rollout_id, sample, rollout.mode or "train", **observation,
                results=results, reward=reward, training_status=training_status,
            )
        self.update(apply)

    def batch_judgments(self, rollout_ids: Sequence[str], samples: Mapping[str, dict[str, Any]],
                       rows: Sequence[dict[str, Any]], step: int) -> None:
        def apply(archive: RolloutTreeArchive) -> None:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for rid, row in zip(rollout_ids, rows, strict=True):
                grouped.setdefault(str(rid), []).append(row)
            for rid, judgments in grouped.items():
                if rid not in self.validated_attempts:
                    archive.report_error(f"No validated attempt for batch rollout {rid}")
                    continue
                archive.judgments(rid, samples[rid], "train", judgments, step, self.validated_attempts[rid][1])
        self.update(apply)

    async def reconcile(self, store: Any, samples: Mapping[str, dict[str, Any]], mode: str) -> None:
        if self.archive is None:
            return
        snapshot = dict(samples)
        ids = list(snapshot)
        if not ids:
            return
        start = self._reconcile_offset % len(ids)
        ids = ids[start:] + ids[:start]
        ids_page = ids[:100]
        positions = {rid: index for index, rid in enumerate(ids_page)}
        rollouts = await asyncio.wait_for(
            store.query_rollouts(rollout_id_in=ids_page, limit=len(ids_page)), timeout=5,
        )
        self._reconcile_offset = (start + len(ids_page)) % len(ids)
        observations = []
        try:
            for rollout in rollouts:
                if rollout.rollout_id not in positions:
                    raise ValueError("Store returned an unrequested rollout")
                # Advance before I/O so a slow rollout cannot starve later IDs.
                self._reconcile_offset = (start + positions[rollout.rollout_id] + 1) % len(ids)
                if (rollout.metadata or {}).get("data_id") != snapshot[rollout.rollout_id].get("data_id"):
                    raise ValueError("Store reconciliation group mismatch")
                observation = self.observation(rollout)
                if not observation["attempt_id"] and rollout.status not in ("queuing", "requeuing"):
                    attempt = await asyncio.wait_for(store.get_latest_attempt(rollout.rollout_id), timeout=5)
                    observation = self.observation(rollout, attempt)
                results = []
                observations.append((rollout.rollout_id, observation, results))
                if rollout.status in TERMINAL and observation["attempt_id"]:
                    spans = await asyncio.wait_for(store.query_spans(
                        rollout.rollout_id, attempt_id=observation["attempt_id"],
                        name="agentlightning.annotation", limit=64, sort_order="desc",
                    ), timeout=5)
                    results.extend(read_results(spans, rollout.rollout_id, observation["attempt_id"]))
                    if len(spans) == 64 and not results:
                        self.archive.report_error(
                            f"Result not found in bounded annotation tail for {rollout.rollout_id}"
                        )
        finally:
            def apply(archive: RolloutTreeArchive) -> None:
                for rid, observation, results in observations:
                    archive.observe(rid, snapshot[rid], mode, **observation, results=results)
            self.update(apply)

    async def _reconcile_safely(
        self, store: Any, samples: Mapping[str, dict[str, Any]], mode: str, *, closing: bool = False,
    ) -> None:
        try:
            await asyncio.wait_for(self.reconcile(store, samples, mode), timeout=10)
        except (OSError, ValueError, TypeError, RuntimeError, TimeoutError, ClientError) as error:
            # The store process exits with the trainer. That refused connection
            # is not a hole in the trees already written.
            if closing and _store_closed(error):
                logger.warning("Rollout tree store closed with the run: %s", error)
                return
            if self.archive:
                self.archive.report_error(f"Store reconciliation failed: {error}")
                self.flush()
            return
        if self.archive:
            self.archive.clear_errors("Store reconciliation failed:")

    @asynccontextmanager
    async def watching(self, store: Any, samples: Mapping[str, dict[str, Any]], mode: str):
        if self.archive is None:
            yield
            return

        async def poll() -> None:
            while True:
                await self._reconcile_safely(store, samples, mode)
                await asyncio.sleep(3)

        task = asyncio.create_task(poll())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await self._reconcile_safely(store, samples, mode, closing=True)
