import type { ReactNode } from 'react';
import { Settings2, X } from 'lucide-react';
import type { WorkflowSpec } from '../../../shared/api/types';
import type { Notice } from '../../../shared/components/InlineNotice';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Button } from '../../../shared/ui/button';

export function WorkflowSettings({ workflow, rlDirty, rlNotice, rlSettings, onClose }: {
  workflow: WorkflowSpec;
  rlDirty: boolean;
  rlNotice: Notice | null;
  rlSettings: ReactNode;
  onClose: () => void;
}) {
  return <aside className="mas-panel mas-inspector" aria-label="工作流设置">
    <div className="mas-panel-heading"><h2><Settings2 size={16} />工作流设置</h2>
      <Button size="sm" variant="ghost" aria-label="关闭工作流设置" onClick={onClose}><X size={16} /></Button>
    </div>
    <div className="mas-panel-body">
      <div className="mas-property-section">
        <h3>技术详情</h3>
        <dl className="mas-settings-details">
          <dt>拓扑</dt><dd><code>{workflow.topology}</code></dd>
          <dt>运行入口</dt><dd><code>{workflow.entry_agent || 'hub'}</code></dd>
          <dt>配置版本</dt><dd><code>{workflow.schema_version || '0.1.0'}</code></dd>
        </dl>
        <p className="field-hint">拓扑根据节点与连线生成，保存至 workflow.yaml。</p>
      </div>
      {rlSettings && <div className="mas-property-section">
        <div className="mas-training-status"><h3>训练配置</h3><span className={rlDirty ? 'is-dirty' : ''}>{rlDirty ? '未保存' : '已保存'}</span></div>
        <p className="field-hint">以下配置单独保存至 rl.yaml，不包含在工作栏的“保存”中。</p>
        {rlNotice && <InlineNotice tone={rlNotice.tone}>{rlNotice.message}</InlineNotice>}
        {rlSettings}
      </div>}
    </div>
  </aside>;
}
