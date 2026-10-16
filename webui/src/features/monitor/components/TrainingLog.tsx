import { memo } from 'react';
import { AglMetricsLink } from '../../../app/layout/MetricsLink';
import { Button } from '../../../shared/ui/button';
import { LogPanel } from '../../../shared/components/LogPanel';
import { StatusBadge } from '../../../shared/components/StatusBadge';

export const MetricsLink = AglMetricsLink;

interface TrainingLogProps {
  trainRunId: string | null;
  trainRunning: boolean;
  trainLog: string;
  aglOnline: boolean;
  onRefreshLog: () => Promise<void>;
}

export const TrainingLog = memo(function TrainingLog({
  trainRunId,
  trainRunning,
  trainLog,
  onRefreshLog,
}: TrainingLogProps) {
  return (
    <LogPanel
      log={trainLog}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge tone={trainRunning ? 'success' : 'neutral'}>
            {trainRunning ? 'running' : 'idle'}
          </StatusBadge>
          {trainRunId ? <span className="mono break-all text-xs">run={trainRunId}</span> : null}
          <Button onClick={onRefreshLog}>刷新日志</Button>
          <AglMetricsLink />
        </div>
      }
      empty={
        <>若已点 Train，日志在 experiments/&lt;id&gt;/artifacts/runs/&lt;run&gt;/stdout.log；重启 UI 后会从磁盘恢复。</>
      }
    />
  );
});
