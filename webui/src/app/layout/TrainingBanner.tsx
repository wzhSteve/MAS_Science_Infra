import { useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { StatusBadge } from '../../shared/components/StatusBadge';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { MetricsLink } from './MetricsLink';

export function TrainingBanner() {
  const { data, error } = useTraining();
  const { stopTrain } = useRuntimeCommands();
  return <>
    {data?.running && <div className="training-banner">
      <StatusBadge tone="success">训练中</StatusBadge><span className="mono">run={data.runId}</span>
      <Button size="sm" variant="danger" onClick={() => void stopTrain()}>Stop</Button><MetricsLink />
    </div>}
    {error && <div className="runtime-error"><InlineNotice tone="warning">训练状态刷新失败，显示最近结果：{error}</InlineNotice></div>}
  </>;
}
