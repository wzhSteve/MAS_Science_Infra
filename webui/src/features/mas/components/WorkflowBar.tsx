import { memo } from 'react';
import { Circle, Play, Plus, Save, Settings2, Workflow } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import type { EditorPanel } from '../types';

export const WorkflowBar = memo(function WorkflowBar({ expId, dirty, pending, saving, saveError, rlDirty, panel, onPanelChange, libraryOpen, onOpenLibrary, onSave, onRun }: {
  expId: string;
  dirty: boolean;
  pending: string | null;
  saving: boolean;
  saveError: boolean;
  rlDirty: boolean;
  panel: EditorPanel;
  onPanelChange: (panel: EditorPanel) => void;
  libraryOpen: boolean;
  onOpenLibrary: () => void;
  onSave: () => Promise<void>;
  onRun: () => void;
}) {
  return <header className="mas-workflow-bar">
    <div className="mas-workflow-identity">
      <Workflow size={19} aria-hidden="true" />
      <span className="mas-workflow-name" title={`实验 ${expId}`}><span>实验 / </span><strong>{expId}</strong></span>
      <span className="mas-workflow-label">Workflow</span>
      <span className={`mas-save-state${saveError ? ' is-error' : dirty ? ' is-dirty' : ''}`} role="status">
        <Circle size={6} fill="currentColor" aria-hidden="true" />
        {saving ? '保存中…' : saveError ? '保存失败' : dirty ? '未保存' : '已保存'}
      </span>
    </div>
    <div className="mas-workflow-actions" aria-label="工作流操作">
      <Button size="sm" aria-expanded={libraryOpen} aria-controls="mas-node-library" onClick={onOpenLibrary}>
        <Plus size={15} aria-hidden="true" />添加
      </Button>
      <Button size="sm" disabled={Boolean(pending)} loading={saving} onClick={onSave} aria-label="保存 Workflow" title="保存 Workflow，不包含训练配置">
        <Save size={14} aria-hidden="true" /><span className="mas-action-label">保存</span>
      </Button>
      <Button size="sm" variant="primary" onClick={onRun}>
        <Play size={14} aria-hidden="true" />运行
      </Button>
      <Button size="sm" variant="ghost" className="mas-settings-trigger" aria-label={rlDirty ? '工作流设置，训练配置未保存' : '工作流设置'}
        title="工作流设置" aria-expanded={panel === 'settings'}
        onClick={() => onPanelChange(panel === 'settings' ? null : 'settings')}>
        <Settings2 size={16} aria-hidden="true" />
        {rlDirty && <span className="mas-unsaved-dot" />}
      </Button>
    </div>
  </header>;
});
