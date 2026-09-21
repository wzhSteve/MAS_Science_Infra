import { memo } from 'react';
import type { MonitorResponse } from '../shared/api/types';
import { Button } from '../shared/ui/button';
import { InlineNotice } from '../shared/components/InlineNotice';
import { LoadingState } from '../shared/components/LoadingState';
import { PageHeader } from '../shared/components/PageHeader';
import { MonitorData, MonitorSummary } from '../features/monitor/components/MonitorData';
import { MetricsLink, TrainingLog } from '../features/monitor/components/TrainingLog';

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
  const hideData = Boolean(error || (!model && loading));
  return (
    <div className="page-stack">
      <PageHeader
        title="Monitor"
        description="训练 Reward 与 AGL Metrics 同源（LightningStore）。Collect 曲线只反映 MAS 采集，不是 GRPO 训练步。"
        actions={
          <>
            <Button onClick={onRefresh} loading={loading}>刷新</Button>
            <MetricsLink online={aglOnline} />
          </>
        }
      />
      {error ? <InlineNotice tone="danger">{error}</InlineNotice> : null}
      {loading && !model ? <LoadingState label="正在加载 Monitor…" /> : null}
      {!error ? <MonitorSummary model={model} /> : null}
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
