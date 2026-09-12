import { useEffect, useState } from 'react';
import { History, RefreshCw } from 'lucide-react';
import type { useRolloutHistory } from '../model/useRolloutHistory';
import { rolloutStatusLabel } from '../model/useMasRolloutRun';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';

export function RolloutHistory({ history, currentRunId }: {
  history: ReturnType<typeof useRolloutHistory>;
  currentRunId?: string;
}) {
  const [open, setOpen] = useState(false);
  useEffect(() => { if (open) void history.load(); }, [open, currentRunId, history.load]);
  return <div className="mas-history">
    <Button size="sm" aria-expanded={open} onClick={() => setOpen(!open)}><History size={14} />最近运行</Button>
    {history.selectedId && <Button size="sm" variant="ghost" onClick={() => void history.select(null)}>返回本页运行</Button>}
    {open && <section className="mas-history-menu" aria-label="最近运行记录">
      <div className="mas-history-heading"><strong>当前实验的运行</strong>
        <Button size="sm" variant="ghost" disabled={history.listLoading} aria-label="刷新运行列表" onClick={() => void history.load()}><RefreshCw size={13} /></Button></div>
      <p className="field-hint">只查看记录，不改变画布或模型配置。</p>
      {history.items.map(item => <button type="button" key={item.run_id} className="mas-history-item"
        disabled={!item.run} aria-current={history.selectedId === item.run_id ? 'true' : undefined}
        onClick={() => { void history.select(item.run_id); setOpen(false); }}>
        {item.run ? <>
          <span><strong>{item.question_preview || '未记录问题'}</strong><small>
            {new Date(item.run.started_at).toLocaleString('zh-CN', { hour12: false })} · {item.run.execution === 'mock' ? 'Mock' : item.run.model?.name || 'Live'}
          </small></span><span>{rolloutStatusLabel[item.run.status]}</span>
        </> : <span><code>{item.run_id}</code><small>{item.record_error || '记录不可用'}</small></span>}
      </button>)}
      {!history.items.length && !history.listLoading && !history.listError && <p className="mas-console-empty">暂无运行记录。</p>}
      {history.listError && <InlineNotice tone="warning">{history.listError}
        <Button size="sm" onClick={() => void history.load(history.nextCursor)}>重试读取</Button>
      </InlineNotice>}
      {history.listLoading && <p className="field-hint" role="status">正在读取记录…</p>}
      {history.nextCursor && <Button size="sm" disabled={history.listLoading} onClick={() => void history.load(history.nextCursor)}>加载更多</Button>}
    </section>}
  </div>;
}
