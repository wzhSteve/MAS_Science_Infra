import { useCallback, useEffect, useState } from 'react';
import { masApi } from '../api';
import { errorMessage } from '../../../shared/api/http';
import type { ModelReadinessResponse } from '../../../shared/api/types';

export function useModelReadiness(expId: string, revision: string | undefined, active: boolean) {
  const [attempt, setAttempt] = useState(0);
  const key = JSON.stringify([expId, revision, active, attempt]);
  const [snapshot, setSnapshot] = useState<{
    key: string; data: ModelReadinessResponse | null; error: string | null;
  } | null>(null);
  const refresh = useCallback(() => setAttempt(value => value + 1), []);

  useEffect(() => {
    setSnapshot(null);
    if (!active) return;
    const controller = new AbortController();
    let current = true;
    void masApi.readiness(expId, controller.signal).then(data => {
      if (!current) return;
      if (data.experiment_id !== expId || (revision && data.model.config_revision !== revision)) {
        throw new Error('模型配置版本已变化，请刷新实验配置后重试。');
      }
      setSnapshot({ key, data, error: null });
    }).catch((error: unknown) => {
      if (current) setSnapshot({ key, data: null, error: errorMessage(error) });
    });
    return () => { current = false; controller.abort(); };
  }, [expId, revision, active, key]);

  const current = snapshot?.key === key ? snapshot : null;
  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    loading: !current,
    refresh,
  };
}

export type ModelReadinessState = ReturnType<typeof useModelReadiness>;
