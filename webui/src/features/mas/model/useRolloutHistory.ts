import { useCallback, useEffect, useRef, useState } from 'react';
import type { RolloutHistoryItem, RolloutRunContext, RolloutTrajectoryResponse } from '../../../shared/api/types';
import { errorMessage } from '../../../shared/api/http';
import { masApi } from '../api';

export function useRolloutHistory(experimentId: string) {
  const [items, setItems] = useState<RolloutHistoryItem[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [context, setContext] = useState<RolloutRunContext | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trajectory, setTrajectory] = useState<RolloutTrajectoryResponse | null>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState(false);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const listRequest = useRef<AbortController | null>(null);
  const contextRequest = useRef<AbortController | null>(null);
  const traceRequest = useRef<AbortController | null>(null);
  useEffect(() => () => {
    listRequest.current?.abort();
    contextRequest.current?.abort();
    traceRequest.current?.abort();
  }, [experimentId]);

  const load = useCallback(async (cursor?: string | null) => {
    listRequest.current?.abort();
    const controller = new AbortController();
    listRequest.current = controller;
    setListLoading(true); setListError(null);
    try {
      const page = await masApi.rolloutHistory(experimentId, cursor, controller.signal);
      if (controller.signal.aborted) return;
      setItems(previous => cursor ? Array.from(new Map([...previous, ...page.items].map(item => [item.run_id, item])).values()) : page.items);
      setNextCursor(page.next_cursor);
    } catch (reason) {
      if (!controller.signal.aborted) setListError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setListLoading(false);
    }
  }, [experimentId]);

  const select = useCallback(async (runId: string | null) => {
    contextRequest.current?.abort(); traceRequest.current?.abort();
    setSelectedId(runId); setContext(null); setError(null);
    setTrajectory(null); setTrajectoryError(null); setTrajectoryLoading(false);
    setLoading(Boolean(runId));
    if (!runId) return;
    const controller = new AbortController();
    contextRequest.current = controller;
    try {
      const result = await masApi.rolloutContext(experimentId, runId, controller.signal);
      if (!controller.signal.aborted) setContext(result);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, [experimentId]);

  const readTrajectory = useCallback(async () => {
    if (!selectedId) return;
    traceRequest.current?.abort();
    const controller = new AbortController();
    traceRequest.current = controller;
    setTrajectoryLoading(true); setTrajectoryError(null);
    try {
      const result = await masApi.rolloutTrajectory(experimentId, selectedId, controller.signal);
      if (controller.signal.aborted) return;
      setTrajectory(result); setContext(result);
    } catch (reason) {
      if (!controller.signal.aborted) setTrajectoryError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setTrajectoryLoading(false);
    }
  }, [experimentId, selectedId]);

  return { items, nextCursor, listLoading, listError, load, selectedId, context, loading, error, select,
    trajectory, trajectoryLoading, trajectoryError, readTrajectory };
}
