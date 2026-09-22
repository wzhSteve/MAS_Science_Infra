import { memo, useCallback, useMemo, useState } from 'react';
import { FileText, RefreshCw } from 'lucide-react';
import { runtimeApi } from '../../runtime/api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { useRuntimeCommands, useTraining } from '../../../app/providers/RuntimeProvider';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { Snapshot, TRAIN_STATE } from './TrainingConsole';

export const TrainingHistory = memo(function TrainingHistory({ experimentId, active }: { experimentId: string; active: boolean }) {
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState('');
  const activity = useTraining();
  const { viewTraining } = useRuntimeCommands();
  const load = useCallback((signal: AbortSignal) => runtimeApi.trainingRuns(experimentId, signal, offset), [experimentId, offset]);
  const history = usePollingResource(`runs:${experimentId}:${offset}:${activity.data?.runId}:${activity.data?.state}`, load, undefined, active);
  const runs = useMemo(() => {
    const term = query.trim().toLowerCase();
    return history.data?.runs.filter(run => !term || `${run.run_id} ${run.meta?.algo || ''} ${TRAIN_STATE[run.state || ''] || ''}`.toLowerCase().includes(term)) || [];
  }, [history.data, query]);
  return <div className="workspace-document-scroll"><div className="workspace-document training-history-page">
    <div className="workspace-document-heading"><h1>训练记录 <span>{history.data?.total ?? '—'}</span></h1>
      <Button size="sm" variant="ghost" onClick={() => void history.refresh()}><RefreshCw size={14} />刷新</Button>
    </div>
    <Input aria-label="搜索当前页训练记录" placeholder="搜索当前页的运行、算法或状态" value={query} onChange={event => setQuery(event.target.value)} />
    {history.error && <InlineNotice tone="danger">{history.error}</InlineNotice>}
    <div className="training-records">
      {runs.map(run => <article key={run.run_id} className="training-record">
        <span className="training-record-icon"><FileText size={18} /></span>
        <div className="training-record-identity"><strong>{String(run.meta?.algo || '训练').toUpperCase()} <code>{run.run_id}</code></strong>
          <span>{run.started_at ? new Date(run.started_at * 1000).toLocaleString() : '等待启动'}{run.meta?.cuda_visible_devices != null ? ` · GPU ${String(run.meta.cuda_visible_devices)}` : ''}</span>
          {run.message && <p>{run.message}</p>}
        </div>
        <StatusBadge tone={run.running ? 'info' : run.state === 'succeeded' ? 'success' : run.state === 'failed' ? 'danger' : 'neutral'}>
          {TRAIN_STATE[run.state || ''] || run.state || '未知'}
        </StatusBadge>
        <Snapshot experimentId={experimentId} runId={run.run_id} />
        <Button size="sm" onClick={() => viewTraining(run.run_id)}>查看日志</Button>
      </article>)}
      {!runs.length && !history.error && <div className="training-console-empty"><FileText size={24} />
        <strong>{history.loading ? '正在读取记录…' : query ? '没有匹配的记录' : '暂无训练记录'}</strong></div>}
    </div>
    {(history.data?.total || 0) > 50 && <div className="training-record-pagination">
      <Button size="sm" disabled={offset === 0} onClick={() => setOffset(value => Math.max(0, value - 50))}>上一页</Button>
      <span>{offset + 1}–{Math.min(offset + 50, history.data?.total || 0)} / {history.data?.total}</span>
      <Button size="sm" disabled={offset + 50 >= (history.data?.total || 0)} onClick={() => setOffset(value => value + 50)}>下一页</Button>
    </div>}
  </div></div>;
});
