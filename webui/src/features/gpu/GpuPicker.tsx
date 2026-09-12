import { memo, useCallback } from 'react';
import { RefreshCw } from 'lucide-react';
import { gpuApi } from './api';
import { usePollingResource } from '../../shared/hooks/usePollingResource';
import { useAction } from '../../shared/hooks/useAction';
import { Button } from '../../shared/ui/button';
import { Tooltip } from '../../shared/ui/tooltip';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { useUnsavedChanges } from '../../shared/hooks/useUnsavedChanges';
export type { GpuInfo } from '../../shared/api/types';

export default memo(function GpuPicker({ selected, onChange, compact = false }: {
  selected: number[]; onChange: (ids: number[]) => void | Promise<void>; compact?: boolean;
}) {
  const { data, loading, error, refresh } = usePollingResource('gpus', gpuApi.gpus);
  const { pending, notice, run } = useAction();
  useUnsavedChanges('gpu-selection', {
    label: 'GPU 配置保存', resource: 'rl', dirty: false, busy: pending !== null,
  });
  const toggle = useCallback((id: number) => {
    const ids = selected.includes(id) ? selected.filter(value => value !== id) : [...selected, id].sort((a, b) => a - b);
    void run('select', async () => { await onChange(ids.length ? ids : [id]); });
  }, [selected, onChange, run]);
  const problem = error || data?.error;
  return <div className={`gpu-picker${compact ? ' gpu-picker--compact' : ''}`}>
    <div className="gpu-controls" aria-label="GPU 设备">
      {!compact && <span className="field-hint">GPU（{data?.count ?? 0} 张）</span>}
      {data?.gpus.map(gpu => <Tooltip key={gpu.id} content={<>{gpu.name}<br />{Math.round(gpu.mem_used_mb)} / {Math.round(gpu.mem_total_mb)} MB · {Math.round(gpu.util)}%</>}>
        <button type="button" className={`gpu-option${selected.includes(gpu.id) ? ' is-selected' : ''}`}
          aria-pressed={selected.includes(gpu.id)} disabled={!!pending} onClick={() => toggle(gpu.id)}>
          <span>GPU {gpu.id}<span className="mono">{Math.round(gpu.util)}%</span></span>
          <span className="gpu-meter"><span style={{ width: `${Math.max(0, Math.min(100, gpu.util))}%` }} /></span>
        </button>
      </Tooltip>)}
      {loading ? <span className="field-hint" role="status">探测 GPU…</span> : !data?.gpus.length && <Tooltip content={problem || '本机未检测到 GPU'}>
        <span tabIndex={0} className="gpu-unavailable">未检测到 GPU</span>
      </Tooltip>}
      <Button size="sm" variant="ghost" title="重新探测 GPU" aria-label="重新探测 GPU" onClick={() => void refresh()}><RefreshCw size={13} aria-hidden="true" />探测</Button>
    </div>
    {problem && <details className="gpu-diagnostic"><summary>设备探测详情</summary><span>{problem}</span></details>}
    {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
  </div>;
});
