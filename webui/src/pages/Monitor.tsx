import { memo } from 'react';
import type { MonitorResponse } from '../shared/api/types';
import { Button } from '../shared/ui/button';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { PageHeader } from '../shared/components/PageHeader';
import { MonitorData, MonitorSummary } from '../features/monitor/components/MonitorData';
import { TrainingLog } from '../features/monitor/components/TrainingLog';
import { AglMetricsLink } from '../app/layout/MetricsLink';
import { useRuntimeCommands } from '../app/providers/RuntimeProvider';

export interface MonitorPanelProps {
  expId: string;
  model: MonitorResponse | null;
  error: string | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
  trainRunId: string | null;
  trainRunning: boolean;
  trainLog: string;
  aglOnline: boolean;
  onRefreshLog: () => Promise<void>;
  visible?: boolean;
}

export const MonitorPanel = memo(function MonitorPanel({
  model,
  error,
  loading,
  onRefresh,
  trainRunId,
  trainRunning,
  trainLog,
  aglOnline,
  onRefreshLog,
  visible = true,
}: MonitorPanelProps) {
  const { viewTraining } = useRuntimeCommands();
  const hideData = Boolean(error || (!model && loading));
  return (
    <div className="page-stack">
      <PageHeader
        title="Monitor"
        description="Reward 与 Loss 来自 metrics.jsonl，训练进行中与 LightningStore 合并，保留更完整的曲线。"
        actions={
          <>
            <Button onClick={onRefresh} loading={loading}>刷新</Button>
            <AglMetricsLink />
          </>
        }
      />
      {error ? <InlineNotice tone="danger">{error}</InlineNotice> : null}
      {loading && !model ? <LoadingState label="正在加载 Monitor…" /> : null}
      {!error ? <MonitorSummary model={model} /> : null}
      <div className="action-bar"><span className="field-hint">以下为实验级分析，不保证属于某一次训练。</span>
        <Button size="sm" onClick={() => viewTraining(trainRunId || undefined)}>查看训练日志</Button>
      </div>
      <TrainingLog
        trainRunId={trainRunId}
        trainRunning={trainRunning}
        trainLog={trainLog}
        aglOnline={aglOnline}
        onRefreshLog={onRefreshLog}
      />
      <div hidden={hideData} className={hideData ? 'hidden' : 'contents'}>
        <MonitorData model={model} aglOnline={aglOnline} visible={visible && !hideData} />
      </div>
    </div>
  );
});
