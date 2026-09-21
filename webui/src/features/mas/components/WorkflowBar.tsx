import { memo } from 'react';
import { Circle, FlaskConical, History, Play, Plus, Save, Settings2, Workflow } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import type { EditorPanel } from '../types';
import type { SettingsSection } from '../../settings/model/sections';
import { SettingsStatus } from '../../settings/components/SettingsStatus';
import { useTraining } from '../../../app/providers/RuntimeProvider';

const TrainingEntry = memo(function TrainingEntry({ onSettings }: { onSettings: (section: SettingsSection) => void }) {
  const training = useTraining();
  return <Button size="sm" variant="primary" onClick={() => onSettings('training')}
    title="打开训练方案和启动确认，不会立即启动训练">
    <Play size={14} />{training.data?.running ? '查看训练' : '开始训练'}
  </Button>;
});

export const WorkflowBar = memo(function WorkflowBar({ expId, dirty, pending, saving, saveError, panel, onSettings, onCloseSettings, libraryOpen, onOpenLibrary, onSave, onDebug, onHistory }: {
  expId: string;
  dirty: boolean;
  pending: string | null;
  saving: boolean;
  saveError: boolean;
  panel: EditorPanel;
  onSettings: (section: SettingsSection) => void;
  onCloseSettings: () => void;
  libraryOpen: boolean;
  onOpenLibrary: () => void;
  onSave: () => Promise<void>;
  onDebug: () => void;
  onHistory: () => void;
}) {
  return <header className="mas-workflow-bar">
    <div className="mas-workflow-identity">
      <Workflow size={19} aria-hidden="true" />
      <span className="mas-workflow-name" title={`实验 ${expId}`}><span>实验 / </span><strong>{expId}</strong></span>
      <span className="mas-workflow-label">MAS 设计</span>
      <span className={`mas-save-state${saveError ? ' is-error' : dirty ? ' is-dirty' : ''}`} role="status">
        <Circle size={6} fill="currentColor" aria-hidden="true" />
        {saving ? '设计保存中…' : saveError ? '设计保存失败' : dirty ? '设计未保存' : '设计已保存'}
      </span>
      <SettingsStatus />
    </div>
    <div className="mas-workflow-actions" aria-label="工作流操作">
      <Button size="sm" aria-expanded={libraryOpen} aria-controls="mas-node-library" onClick={onOpenLibrary}>
        <Plus size={15} aria-hidden="true" />添加
      </Button>
      <Button size="sm" disabled={Boolean(pending)} loading={saving} onClick={onSave} aria-label="保存 Workflow" title="保存 Workflow，不包含训练配置">
        <Save size={14} aria-hidden="true" /><span className="mas-action-label">保存设计</span>
      </Button>
      <Button size="sm" onClick={onDebug} title="打开单题调试面板，不立即执行">
        <FlaskConical size={14} aria-hidden="true" />调试
      </Button>
      <Button id="mas-settings-trigger" size="sm" variant="ghost" className="mas-settings-trigger" aria-controls="experiment-settings"
        title="分组编辑模型、数据、训练与诊断配置" aria-expanded={panel === 'settings'}
        onClick={() => panel === 'settings' ? onCloseSettings() : onSettings('model')}>
        <Settings2 size={15} aria-hidden="true" />实验设置
      </Button>
      <Button size="sm" variant="ghost" onClick={onHistory} title="查看当前实验的单次 Rollout 记录；训练监控在更多功能中">
        <History size={14} />运行记录
      </Button>
      <TrainingEntry onSettings={onSettings} />
    </div>
  </header>;
});
