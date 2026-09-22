import { memo, useMemo } from 'react';
import { DropdownMenu } from 'radix-ui';
import { AlertTriangle, Check, ChevronDown, Cpu, RefreshCw } from 'lucide-react';
import { useTrainingConfig } from '../providers/TrainingConfigProvider';
import { Button } from '../../shared/ui/button';

export const GpuStatusControl = memo(function GpuStatusControl() {
  const { draft, gpu, selectedIds, selectGpus } = useTrainingConfig();
  const detected = gpu.data?.gpus || [];
  const problem = gpu.error || gpu.data?.error;
  const detectedIds = useMemo(() => new Set(detected.map((item) => item.id)), [detected]);
  const missingIds = selectedIds.filter((id) => !detectedIds.has(id));
  const selectedDetected = detected.filter((item) => selectedIds.includes(item.id));
  const utilization = selectedDetected.length
    ? Math.round(selectedDetected.reduce((sum, item) => sum + item.util, 0) / selectedDetected.length)
    : 0;
  const label = gpu.loading && !gpu.data ? 'GPU 检测中'
    : !detected.length ? 'GPU 未检测'
      : missingIds.length ? 'GPU 配置待确认'
        : `${selectedIds.length} / ${detected.length} GPU`;

  const toggle = (id: number) => {
    const next = selectedIds.includes(id)
      ? selectedIds.filter((value) => value !== id)
      : [...selectedIds, id].sort((a, b) => a - b);
    if (next.length) selectGpus(next);
  };

  return <div className="gpu-status-control">
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button type="button" className={`gpu-status-trigger${missingIds.length ? ' has-warning' : ''}`}>
          {missingIds.length ? <AlertTriangle size={14} /> : <Cpu size={14} />}
          <span>{label}</span>
          {detected.length > 0 && <span className="gpu-status-meter" aria-label={`平均占用 ${utilization}%`}>
            <span style={{ width: `${Math.max(0, Math.min(100, utilization))}%` }} />
          </span>}
          {draft.dirty && <span className="gpu-status-dirty" aria-label="训练配置未保存" />}
          <ChevronDown size={12} aria-hidden="true" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="gpu-status-popover" align="start" sideOffset={8} collisionPadding={12}>
          <DropdownMenu.Label className="gpu-status-title">训练设备</DropdownMenu.Label>
          {detected.map((item) => {
            const selected = selectedIds.includes(item.id);
            return <DropdownMenu.CheckboxItem key={item.id} checked={selected}
              className="gpu-status-device" onSelect={(event) => event.preventDefault()}
              onCheckedChange={() => toggle(item.id)}>
              <span className="gpu-status-check">{selected && <Check size={13} />}</span>
              <span className="gpu-status-device-main">
                <strong>GPU {item.id}</strong><span>{item.name}</span>
                <span className="gpu-status-device-meter"><span style={{ width: `${Math.max(0, Math.min(100, item.util))}%` }} /></span>
              </span>
              <span className="gpu-status-device-meta">
                {Math.round(item.mem_used_mb)} / {Math.round(item.mem_total_mb)} MB<br />{Math.round(item.util)}%
              </span>
            </DropdownMenu.CheckboxItem>;
          })}
          {!detected.length && <div className="gpu-status-empty">
            <strong>当前环境未检测到 NVIDIA GPU</strong>
            <span>{problem || '可以继续编辑实验，真实训练需在 GPU 环境执行。'}</span>
          </div>}
          {missingIds.length > 0 && <div className="gpu-status-warning">
            目标配置：{selectedIds.map((id) => `GPU ${id}`).join('、')}<br />
            当前无法验证：{missingIds.map((id) => `GPU ${id}`).join('、')}
          </div>}
          <DropdownMenu.Separator className="gpu-status-separator" />
          <div className="gpu-status-footer">
            <span>{draft.dirty ? '设备选择尚未保存' : `目标设备 ${selectedIds.length} 张`}</span>
          </div>
          <DropdownMenu.Arrow className="workspace-navigation-arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
    <Button size="sm" variant="ghost" className="gpu-status-refresh"
      loading={gpu.loading} aria-label="重新探测 GPU" title="重新探测 GPU"
      onClick={() => { void gpu.refresh(); }}>
      <RefreshCw size={13} />
    </Button>
  </div>;
});
