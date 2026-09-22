import { useMemo } from 'react';
import { useCommandStatus, useMonitor, useRuntimeCommands, useRuntimeEvents, useTraining } from '../providers/RuntimeProvider';
import { StatusBadge } from '../../shared/components/StatusBadge';
import { Button } from '../../shared/ui/button';

export function StatusBar() {
  const { data } = useTraining();
  const monitor = useMonitor();
  const events = useRuntimeEvents();
  const { status } = useCommandStatus();
  const { stopTrain } = useRuntimeCommands();
  const tail = useMemo(() => data?.log.split('\n').filter(Boolean).slice(-2).join(' · ') || '', [data?.log]);
  const summary = events.error || tail || events.latest || (status !== 'idle' ? status : '等待事件…');
  return <footer className="status-bar">
    <StatusBadge tone={data?.running ? 'success' : data?.state === 'failed' ? 'danger' : 'neutral'}>
      {data?.state || 'idle'}
    </StatusBadge>
    <span className="mono">reward={monitor.data?.mean_reward ?? '—'}{monitor.error ? ' (stale)' : ''}</span>
    <span className={`status-tail${events.error ? ' field-error' : ''}`} title={summary}>{summary}</span>
    {data?.running && <Button size="sm" variant="danger" onClick={() => void stopTrain()}>Stop</Button>}
  </footer>;
}
