import { ArrowDown, Trash2, X } from 'lucide-react';
import type { GraphEdge } from '../types';
import { EDGE_LABELS } from '../model/workflowGraph';
import { Button } from '../../../shared/ui/button';
import { Select } from '../../../shared/ui/select';
import { FormField } from '../../../shared/components/FormField';

export function EdgeInspector({ edge, kinds, onChange, onClose, onDelete }: {
  edge: GraphEdge;
  kinds: Array<{ kind: string; reason: string }>;
  onChange: (kind: string) => void;
  onClose: () => void;
  onDelete: () => void;
}) {
  return <aside className="mas-panel mas-inspector" aria-label="连线属性">
    <div className="mas-panel-heading"><h2>连线属性</h2><Button size="sm" variant="ghost" aria-label="关闭连线属性" onClick={onClose}><X size={16} /></Button></div>
    <div className="mas-panel-body mas-property-section">
      <div className="mas-edge-endpoints"><code>{edge.source}</code><ArrowDown size={16} /><code>{edge.target}</code></div>
      <FormField label="关系类型">
        <Select value={edge.data?.kind || 'message'} onChange={(e) => onChange(e.target.value)}>
          {kinds.map(({ kind, reason }) => <option key={kind} value={kind} disabled={Boolean(reason)}>{EDGE_LABELS[kind] || kind}{reason ? ` · ${reason}` : ''}</option>)}
        </Select>
      </FormField>
      <p className="field-hint">连线修改随 Workflow 一起保存。</p>
    </div>
    <div className="mas-panel-footer"><Button size="sm" variant="danger" onClick={onDelete}><Trash2 size={14} />删除连线</Button></div>
  </aside>;
}
