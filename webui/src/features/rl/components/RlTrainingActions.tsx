import { memo } from 'react';
import { ActionBar } from '../../../shared/components/ActionBar';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { Button } from '../../../shared/ui/button';

type Props = {
  trainRunId: string | null;
  trainRunning: boolean;
  aglOnline: boolean;
  pending: string | null;
  onStart: () => void;
  onStop: () => Promise<void>;
};

export const RlTrainingActions = memo(function RlTrainingActions({ trainRunId, trainRunning, aglOnline, pending, onStart, onStop }: Props) {
  const stop = useAction();
  return <Section title="训练运行"
    actions={<StatusBadge tone={trainRunning ? 'info' : 'neutral'}>{trainRunning ? 'running' : 'idle'}</StatusBadge>}>
    {trainRunId && <p className="field-hint">run=<span className="mono break-all">{trainRunId}</span></p>}
    <ActionBar>
      <Button variant="primary" loading={pending === 'start'} disabled={trainRunning || pending !== null} onClick={onStart}>保存训练配置并确认启动</Button>
      <Button variant="danger" loading={stop.pending !== null} onClick={() => { void stop.run('stop', onStop); }}>停止训练</Button>
      {aglOnline ? <a className="ui-button ui-button--secondary ui-button--md" href="/agl/metrics" target="_blank" rel="noreferrer">打开 AGL Metrics</a> :
        <Button disabled aria-describedby="rl-metrics-unavailable">Metrics 未就绪</Button>}
    </ActionBar>
    {!aglOnline && <p id="rl-metrics-unavailable" className="field-hint">训练进程拉起 LightningStore 后可用（同源 /agl/metrics）。</p>}
    {stop.notice && <InlineNotice tone={stop.notice.tone}>{stop.notice.message}</InlineNotice>}
  </Section>;
});
