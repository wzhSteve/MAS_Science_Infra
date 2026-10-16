import { memo } from 'react';
import { AglMetricsLink } from '../../../app/layout/MetricsLink';
import { ActionBar } from '../../../shared/components/ActionBar';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { Button } from '../../../shared/ui/button';

type Props = {
  trainRunId: string | null;
  trainState?: string;
  trainRunning: boolean;
  aglOnline: boolean;
  pending: string | null;
  onStart: () => void;
  onStop: () => Promise<void>;
};

export const RlTrainingActions = memo(function RlTrainingActions({ trainRunId, trainState, trainRunning, aglOnline, pending, onStart, onStop }: Props) {
  const stop = useAction();
  return <Section title="训练运行"
    actions={<StatusBadge tone={trainRunning ? 'info' : trainState === 'failed' ? 'danger' : 'neutral'}>
      {trainState || (trainRunning ? 'running' : 'idle')}
    </StatusBadge>}>
    {trainRunId && <p className="field-hint">run=<span className="mono break-all">{trainRunId}</span></p>}
    <ActionBar>
      <Button variant="primary" loading={pending === 'start'} disabled={trainRunning || pending !== null} onClick={onStart}>保存训练配置并确认启动</Button>
      <Button variant="danger" loading={stop.pending !== null} onClick={() => { void stop.run('stop', onStop); }}>停止训练</Button>
      <AglMetricsLink />
    </ActionBar>
    {!aglOnline && <p className="field-hint">LightningStore 未在线时，AGL Metrics 打开磁盘上的历史曲线。</p>}
  </Section>;
});
