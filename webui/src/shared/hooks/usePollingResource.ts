import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { errorMessage, isAbortError } from '../api/http';
import { shareSnapshot } from '../lib/shareSnapshot';

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
}
interface Snapshot<T> { key: string; data: T | null; error: string | null; loading: boolean }

export function usePollingResource<T>(
  key: string,
  load: (signal: AbortSignal) => Promise<T>,
  interval?: number,
  enabled = true,
): Resource<T> {
  const [snapshot, setSnapshot] = useState<Snapshot<T>>({ key, data: null, error: null, loading: true });
  const loadRef = useRef(load);
  const refreshRef = useRef<() => Promise<void>>(async () => {});
  useLayoutEffect(() => { loadRef.current = load; }, [load]);

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let running: Promise<void> | null = null;
    let queued = false;
    const controller = new AbortController();
    setSnapshot(previous => previous.key === key ? previous : { key, data: null, error: null, loading: true });

    const execute = (): Promise<void> => {
      if (running) return running;
      running = (async () => {
        do {
          queued = false;
          try {
            const data = await loadRef.current(controller.signal);
            if (!active) return;
            setSnapshot(previous => {
              const shared = shareSnapshot(previous.key === key ? previous.data : null, data);
              return previous.key === key && shared === previous.data && !previous.error && !previous.loading
                ? previous : { key, data: shared, error: null, loading: false };
            });
          } catch (error) {
            if (!active || isAbortError(error)) return;
            const message = errorMessage(error);
            setSnapshot(previous => previous.key === key && previous.error === message && !previous.loading
              ? previous : { key, data: previous.key === key ? previous.data : null, error: message, loading: false });
          }
        } while (queued && active);
      })().finally(() => { running = null; });
      return running;
    };
    const refresh = () => {
      if (running) queued = true;
      return execute();
    };
    refreshRef.current = refresh;
    void execute();
    const timer = interval ? setInterval(() => { if (!running) void execute(); }, interval) : undefined;
    return () => {
      active = false;
      controller.abort();
      clearInterval(timer);
      if (refreshRef.current === refresh) refreshRef.current = async () => {};
    };
  }, [key, interval, enabled]);

  const refresh = useCallback(() => refreshRef.current(), []);
  const data = snapshot.key === key ? snapshot.data : null;
  const error = snapshot.key === key ? snapshot.error : null;
  const loading = snapshot.key !== key || snapshot.loading;
  return useMemo(() => ({ data, error, loading, refresh }), [data, error, loading, refresh]);
}
