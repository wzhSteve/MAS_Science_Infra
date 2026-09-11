import { memo } from 'react';
import { Bot, Flag, Trash2, Wrench, X } from 'lucide-react';
import type { Palette } from '../../../shared/api/types';
import type { GraphNodeData, SelectedGraphNode } from '../types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { Textarea } from '../../../shared/ui/textarea';
import { Checkbox } from '../../../shared/ui/checkbox';
import { FormField } from '../../../shared/components/FormField';

export const NodeInspector = memo(function NodeInspector({ selected, palette, entryId, onPatch, onEntry, onDelete, onClose }: {
  selected: SelectedGraphNode;
  palette: Palette;
  entryId: string;
  onPatch: (patch: Partial<GraphNodeData>) => void;
  onEntry: (id: string) => void;
  onDelete: () => void;
  onClose: () => void;
}) {
  const isTool = selected.type === 'tool';
  const Icon = isTool ? Wrench : Bot;
  const tools = Array.from(new Set([...(palette.tools || []), ...(selected.data.tools || [])]));
  return <aside className="mas-panel mas-inspector" aria-label="节点属性">
    <div className="mas-panel-heading">
      <div><span className="mas-panel-caption">{isTool ? 'Tool' : 'Agent'} 属性</span><h2><Icon size={16} />{selected.id}</h2></div>
      <Button size="sm" variant="ghost" onClick={onClose} aria-label="关闭节点属性"><X size={16} /></Button>
    </div>
    <div className="mas-panel-body">
      {isTool ? <div className="mas-property-section">
        <FormField label="工具标识"><Input value={selected.id} readOnly className="mono" /></FormField>
        <p className="field-hint">从 Agent 拉出连线调用此工具。工具不能作为运行入口。</p>
      </div> : <>
        <div className="mas-property-section">
          <FormField label="角色"><Input value={selected.data.role || ''} onChange={(e) => onPatch({ role: e.target.value })} /></FormField>
          <FormField label="系统提示词" hint="定义 Agent 的职责、行为和输出要求。">
            <Textarea rows={7} value={selected.data.system_prompt || ''} placeholder="描述这个 Agent 应该完成什么…"
              onChange={(e) => onPatch({ system_prompt: e.target.value })} />
          </FormField>
          <FormField label="Skills" hint="多个技能以逗号分隔。">
            <Input value={(selected.data.skills || []).join(', ')} onChange={(e) =>
              onPatch({ skills: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} />
          </FormField>
        </div>
        <fieldset className="mas-property-section">
          <legend>可调用工具</legend>
          {tools.map((tool) => {
            const selectedTools = selected.data.tools || [];
            const checked = selectedTools.includes(tool);
            return <label key={tool} className="mas-checkbox-row">
              <Checkbox checked={checked} onChange={() => onPatch({ tools: checked ? selectedTools.filter((t) => t !== tool) : [...selectedTools, tool] })} />
              <span>{tool}</span>
            </label>;
          })}
          {!tools.length && <p className="field-hint">暂无可用工具。</p>}
        </fieldset>
        <div className="mas-property-section">
          <label className="mas-checkbox-row"><Checkbox checked={selected.data.trainable !== false}
            onChange={(e) => onPatch({ trainable: e.target.checked })} />参与训练</label>
          <Button size="sm" disabled={selected.id === entryId} onClick={() => onEntry(selected.id)}>
            <Flag size={14} />{selected.id === entryId ? '当前运行入口' : '设为运行入口'}
          </Button>
        </div>
        {selected.id === 'hub' && <details className="mas-property-section">
          <summary>高级配置</summary>
          <FormField label="验证技能" hint="启用后通过 Verifier 向 hub 反馈。">
            <Select value={selected.data.verify || ''} onChange={(e) => onPatch({ verify: e.target.value || null })}>
              <option value="">关闭</option>
              {(palette.skills || []).map((skill) => <option key={skill} value={skill}>{skill}</option>)}
            </Select>
          </FormField>
        </details>}
      </>}
    </div>
    <div className="mas-panel-footer"><Button size="sm" variant="danger" onClick={onDelete}><Trash2 size={14} />删除节点</Button></div>
  </aside>;
});
