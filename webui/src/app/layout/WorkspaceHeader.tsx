import { memo, useCallback, useMemo } from 'react';
import { FlaskConical } from 'lucide-react';
import type { Bundle } from '../../shared/api/types';
import { experimentApi } from '../../features/experiment/api';
import GpuPicker from '../../features/gpu/GpuPicker';
import { GlobalActionBar } from './GlobalActionBar';

const DEFAULT_GPU_IDS = [0];

export const WorkspaceHeader = memo(function WorkspaceHeader({ bundle, onReload }: { bundle: Bundle; onReload: () => Promise<void> }) {
  const ids = bundle.rl.devices?.ids || DEFAULT_GPU_IDS;
  const trainable = useMemo(() => bundle.workflow.agents?.filter(agent => agent.trainable !== false).map(agent => agent.id).join(', ')
    || bundle.workflow.entry_agent || 'hub', [bundle.workflow]);
  const saveGpus = useCallback(async (selected: number[]) => {
    await experimentApi.putSection(bundle.id, 'rl', {
      ...bundle.rl, devices: { ids: selected },
      trainer: { ...bundle.rl.trainer, n_gpus_per_node: selected.length },
    });
    await onReload();
  }, [bundle.id, bundle.rl, onReload]);
  return <header className="workspace-header">
    <div className="workspace-context">
      <div className="experiment-heading"><FlaskConical size={16} aria-hidden="true" /><span className="field-hint">实验</span><strong>{bundle.id}</strong><span className="context-divider" /><span className="mono field-hint">seed {bundle.meta.seed ?? '—'}</span></div>
      <dl className="contract-summary" aria-label="实验参数摘要">
        <div><dt>入口</dt><dd>{bundle.workflow.entry_agent || 'hub'}</dd></div>
        <div><dt>可训</dt><dd>{trainable}</dd></div>
        <div><dt>GPU</dt><dd>{ids.join(', ')}</dd></div>
        <div><dt>n</dt><dd>{bundle.rl.rollout_per_gpu ?? bundle.rl.actor_rollout_ref?.rollout?.n ?? 2}</dd></div>
        <div><dt>runners</dt><dd>{bundle.rl.n_runners ?? 1}</dd></div>
      </dl>
    </div>
    <div className="workspace-actions"><GpuPicker compact selected={ids} onChange={saveGpus} /><GlobalActionBar executable={bundle.executable} /></div>
  </header>;
});
