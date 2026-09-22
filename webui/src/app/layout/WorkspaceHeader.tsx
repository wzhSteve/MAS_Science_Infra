import { memo, useContext } from 'react';
import { ArrowLeft, Circle, Database, FlaskConical, Play, Save } from 'lucide-react';
import type { Bundle } from '../../shared/api/types';
import { Button } from '../../shared/ui/button';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { useAction } from '../../shared/hooks/useAction';
import { DraftRegistryContext, useUnsavedChanges } from '../../shared/hooks/useUnsavedChanges';
import { useSettingsStatus } from '../../features/settings/components/SettingsStatus';
import { isExperimentDraft, saveExperimentDrafts } from '../../features/experiment/model/saveExperiment';
import { useCommandStatus, useRuntimeCommands, useTraining } from '../providers/RuntimeProvider';
import { GpuStatusControl } from './GpuStatusControl';
import { WorkspaceMenu } from './WorkspaceMenu';
import type { PanelId, ResourceCategory } from '../navigation';
import type { SettingsSection } from '../../features/settings/model/sections';

const ExperimentActions = memo(function ExperimentActions({ onReload }: { onReload: () => Promise<void> }) {
  const registry = useContext(DraftRegistryContext)!;
  const entries = useSettingsStatus();
  const drafts = entries.filter(isExperimentDraft);
  const dirty = drafts.some(item => item.dirty);
  const action = useAction();
  const commands = useRuntimeCommands();
  const runtime = useCommandStatus();
  const training = useTraining();
  const busy = entries.some(item => item.busy) || action.pending !== null || runtime.pending !== null;
  useUnsavedChanges('experiment-save', { label: '保存实验', resource: 'runtime', dirty: false, busy: action.pending !== null });
  const error = action.notice?.tone === 'danger' ? action.notice.message
    : runtime.notice?.tone === 'danger' ? runtime.notice.message : training.error;
  return <>
    <span className={`experiment-save-status${dirty ? ' is-dirty' : ''}`} role="status">
      <Circle size={6} fill="currentColor" />{action.pending ? '保存中' : dirty ? '未保存' : '已保存'}
      {training.data?.running && <span>下次运行配置</span>}
    </span>
    <div className="experiment-header-actions">
      <Button size="sm" disabled={busy || !dirty} loading={action.pending === 'save'} onClick={() => void action.run('save', async () => {
        await saveExperimentDrafts(registry, onReload);
      })}><Save size={14} />保存实验</Button>
      <Button size="sm" variant="primary" disabled={busy} loading={runtime.pending === 'train'}
        onClick={() => training.data?.running ? commands.viewTraining(training.data.runId || undefined) : void commands.startTrain()}>
        <Play size={14} />{training.data?.running ? '查看训练' : '开始训练'}
      </Button>
    </div>
    {error && <div className="experiment-header-error"><InlineNotice tone="danger">{error}</InlineNotice></div>}
  </>;
});

export const WorkspaceHeader = memo(function WorkspaceHeader({ bundle, onReload, onHome, active, settings, onChangePanel, onResources }: {
  bundle: Bundle; onReload: () => Promise<void>; onHome: () => void;
  active: PanelId; settings?: SettingsSection; onChangePanel: (panel: PanelId) => void;
  onResources: (category: ResourceCategory) => void;
}) {
  return <header className="experiment-header">
    <Button size="sm" variant="ghost" onClick={onHome}><ArrowLeft size={15} /><span>实验</span></Button>
    <span className="experiment-header-divider" />
    <div className="experiment-header-name" title={bundle.meta.name || bundle.id}>
      <FlaskConical size={16} aria-hidden="true" /><span>{bundle.meta.name || bundle.id}</span>
    </div>
    <GpuStatusControl />
    <ExperimentActions onReload={onReload} />
    <nav className="experiment-header-tools" aria-label="资源与工具">
      <Button size="sm" variant="ghost" title="模型与数据" aria-label="模型与数据" onClick={() => onResources('models')}>
        <Database size={15} /><span>模型与数据</span>
      </Button>
      <WorkspaceMenu active={settings === 'diagnostics' ? 'harness' : active} onChange={onChangePanel} />
    </nav>
  </header>;
});
