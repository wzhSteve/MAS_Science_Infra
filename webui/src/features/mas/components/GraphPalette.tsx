import { memo, useState, type DragEvent } from 'react';
import { AlertDialog } from 'radix-ui';
import { Bot, Layers, Plus, Search, Wrench, X } from 'lucide-react';
import type { Palette, WorkflowSpec } from '../../../shared/api/types';
import type { GraphNode } from '../types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';

export const NODE_TRANSFER = 'application/science-node';
export type LibraryTab = 'agent' | 'tool' | 'template';

export const GraphPalette = memo(function GraphPalette({ palette, nodes, tab, onTabChange, onAdd, onTemplate, onClose }: {
  palette: Palette;
  nodes: GraphNode[];
  tab: LibraryTab;
  onTabChange: (tab: LibraryTab) => void;
  onAdd: (kind: 'agent' | 'tool', type: string) => void;
  onTemplate: (workflow: WorkflowSpec) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState('');
  const [replacement, setReplacement] = useState<WorkflowSpec | null>(null);
  const matches = (value: string) => value.toLowerCase().includes(query.trim().toLowerCase());
  const items = (tab === 'agent' ? palette.roles || [] : palette.tools || []).filter(matches);
  const templates = (palette.templates || []).filter((t) => matches(t.label) || matches(t.id));
  const drag = (event: DragEvent, kind: 'agent' | 'tool', type: string) => {
    event.dataTransfer.setData(NODE_TRANSFER, JSON.stringify({ kind, type }));
    event.dataTransfer.effectAllowed = 'copy';
  };

  return <aside className="mas-panel mas-library" id="mas-node-library" aria-label="添加节点">
    <div className="mas-panel-heading">
      <h2>添加到画布</h2>
      <Button variant="ghost" size="sm" onClick={onClose} aria-label="关闭节点库"><X size={16} /></Button>
    </div>
    <div className="mas-library-search"><Search size={14} aria-hidden="true" />
      <Input autoFocus placeholder="搜索节点或模板" aria-label="搜索节点或模板" value={query} onChange={(e) => setQuery(e.target.value)} />
    </div>
    <div className="mas-segmented" aria-label="节点分类">
      {([['agent', 'Agent'], ['tool', 'Tool'], ['template', '模板']] as const).map(([value, label]) =>
        <button key={value} type="button" aria-pressed={tab === value} onClick={() => onTabChange(value)}>{label}</button>)}
    </div>
    <div className="mas-panel-body mas-library-items">
      {tab === 'template' ? templates.map((template) =>
        <button type="button" className="mas-library-item" key={template.id} disabled={!template.workflow}
          onClick={() => {
            if (!template.workflow) return;
            if (nodes.length) setReplacement(template.workflow);
            else onTemplate(template.workflow);
          }}>
          <span className="mas-library-icon"><Layers size={17} /></span>
          <span><strong>{template.label.replace(' (executable)', '')}</strong><small>{nodes.length ? '替换当前工作流' : '从此模板开始'}</small></span>
        </button>)
        : items.map((type) => {
          const exists = (tab === 'tool' || type === 'hub') && nodes.some((n) => n.id === type);
          const Icon = tab === 'tool' ? Wrench : Bot;
          return <button type="button" className="mas-library-item" key={type} disabled={exists} draggable={!exists}
            onDragStart={(event) => drag(event, tab, type)} onClick={() => onAdd(tab, type)}>
            <span className="mas-library-icon"><Icon size={17} aria-hidden="true" /></span>
            <span><strong>{type}</strong><small>{exists ? '已添加' : tab === 'tool' ? '工具节点' : '智能体节点'}</small></span>
            {!exists && <Plus size={14} className="mas-library-plus" aria-hidden="true" />}
          </button>;
        })}
      {(tab === 'template' ? !templates.length : !items.length) && <p className="mas-panel-empty">没有匹配的{tab === 'template' ? '模板' : '节点'}</p>}
    </div>
    <p className="mas-panel-footnote">{tab === 'template' ? '模板包含节点、连线和工作流配置。' : '点击或拖拽连续添加，点击右上角关闭工具箱。'}</p>
    <AlertDialog.Root open={Boolean(replacement)} onOpenChange={(open) => { if (!open) setReplacement(null); }}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="dialog-overlay" />
        <AlertDialog.Content className="dialog-content">
          <AlertDialog.Title className="dialog-title">替换当前工作流</AlertDialog.Title>
          <AlertDialog.Description className="dialog-description">当前节点、连线和未保存的修改将被模板替换。此操作暂不支持撤销。</AlertDialog.Description>
          <div className="dialog-actions">
            <AlertDialog.Cancel asChild><Button>取消</Button></AlertDialog.Cancel>
            <AlertDialog.Action asChild><Button variant="primary" onClick={() => { if (replacement) onTemplate(replacement); }}>确认替换</Button></AlertDialog.Action>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  </aside>;
});
