import { useCommandStatus, useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { Button } from '../../shared/ui/button';
import { StatusBadge } from '../../shared/components/StatusBadge';
import { InlineNotice } from '../../shared/components/InlineNotice';

export function TrainingBanner() {
  const { data, error } = useTraining();
  const { viewTraining } = useRuntimeCommands();
  const { notice, pending } = useCommandStatus();
  return <>
    {data?.running && <div className="training-banner">
      <StatusBadge tone="success">{data.state}</StatusBadge><span className="mono">run={data.runId}</span>
      <Button size="sm" onClick={() => viewTraining(data.runId || undefined)}>查看训练</Button>
    </div>}
    {error && <div className="runtime-error"><InlineNotice tone="warning">训练状态刷新失败，显示最近结果：{error}</InlineNotice></div>}
    {pending === 'train' && <div className="runtime-error" role="status">正在准备训练…</div>}
    {notice?.tone === 'danger' && <div className="runtime-error"><InlineNotice tone="danger">{notice.message}</InlineNotice></div>}
  </>;
}
