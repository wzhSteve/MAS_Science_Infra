import { useEffect, useState } from 'react';
import type { ScienceEvent } from '../../../shared/api/types';
import { errorMessage } from '../../../shared/api/http';

function parseEvent(data: string): ScienceEvent {
  const value: unknown = JSON.parse(data);
  if (!value || typeof value !== 'object' || !('type' in value) || typeof value.type !== 'string'
    || !('experiment_id' in value) || typeof value.experiment_id !== 'string') {
    throw new Error('事件格式无效');
  }
  return value as ScienceEvent;
}

export interface ExperimentEvents {
  latest: string;
  error: string | null;
  connected: boolean;
  collectEvents: ScienceEvent[];
}

export function useExperimentEvents(expId: string, onTrain: () => Promise<void>, onEvent: () => Promise<void>) {
  const [state, setState] = useState<ExperimentEvents>({
    latest: '', error: null, connected: false, collectEvents: [],
  });
  useEffect(() => {
    setState({ latest: '', error: null, connected: false, collectEvents: [] });
    const source = new EventSource(`/api/events?experiment_id=${encodeURIComponent(expId)}`);
    source.onopen = () => setState(previous => ({ ...previous, connected: true, error: null }));
    source.onerror = () => setState(previous => previous.error === '事件连接中断，正在重新连接'
      ? previous : { ...previous, connected: false, error: '事件连接中断，正在重新连接' });
    source.onmessage = event => {
      try {
        const payload = parseEvent(event.data);
        if (payload.experiment_id !== expId) return;
        setState(previous => ({
          ...previous, latest: payload.type, error: null,
          collectEvents: payload.type === 'collect_progress'
            ? [...previous.collectEvents.slice(-99), payload] : previous.collectEvents,
        }));
        if (payload.type.includes('train')) void onTrain();
        void onEvent();
      } catch (error) {
        setState(previous => ({ ...previous, error: errorMessage(error) }));
      }
    };
    return () => source.close();
  }, [expId, onTrain, onEvent]);
  return state;
}
