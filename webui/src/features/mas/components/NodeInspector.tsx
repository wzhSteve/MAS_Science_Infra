import { memo, type ReactNode } from 'react';
import type { Palette } from '../../../shared/api/types';
import type { GraphNodeData, SelectedGraphNode } from '../types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { Textarea } from '../../../shared/ui/textarea';
import { Checkbox } from '../../../shared/ui/checkbox';
import { FormField } from '../../../shared/components/FormField';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { EmptyState } from '../../../shared/components/EmptyState';

export const NodeInspector = memo(function NodeInspector({ selected, palette, entryId, onPatch, onEntry, rlSettings }: {
  selected: SelectedGraphNode | null;
  palette: Palette;
  entryId: string;
  onPatch: (patch: Partial<GraphNodeData>) => void;
  onEntry: (id: string) => void;
  rlSettings?: ReactNode;
}) {
  return <aside className="mas-inspector" aria-label="节点属性">
    <div className="mas-inspector__heading">
      <h2>节点属性</h2>
      {selected && <StatusBadge tone={selected.type === 'tool' ? 'neutral' : 'info'}>{selected.type === 'tool' ? 'Tool' : 'Agent'}</StatusBadge>}
    </div>
    {!selected ? <EmptyState title="选择一个节点">
      选中 Agent 编辑 prompt、skills、tools 和训练参与；将「采集入口」拖到 Agent 设置 Episode 起点。
    </EmptyState>
      : selected.type === 'tool' ? <div className="space-y-3">
        <div className="mono break-all font-semibold">{selected.id}</div>
        <p className="field-hint">工具节点通过 tool_call 边连接 Agent，不作为采集入口。</p>
      </div> : <div className="space-y-4">
        <div className="space-y-3">
          <FormField label="id"><Input value={selected.id} disabled className="mono" /></FormField>
          <FormField label="role"><Input value={selected.data.role || ''}
            onChange={(event) => onPatch({ role: event.target.value })} /></FormField>
        </div>
        <FormField label="system prompt（profile）">
          <Textarea rows={5} value={selected.data.system_prompt || ''}
            onChange={(event) => onPatch({ system_prompt: event.target.value })} />
        </FormField>
        <FormField label="skills（逗号分隔）">
          <Input value={(selected.data.skills || []).join(', ')} onChange={(event) =>
            onPatch({ skills: event.target.value.split(',').map((skill) => skill.trim()).filter(Boolean) })} />
        </FormField>
        <fieldset className="space-y-2">
          <legend className="mb-2 text-xs font-semibold">tools</legend>
          {(palette.tools || []).map((tool) => {
            const tools = selected.data.tools || [];
            const checked = tools.includes(tool);
            return <label key={tool} className="flex items-center gap-2 text-xs">
              <Checkbox checked={checked} onChange={() =>
                onPatch({ tools: checked ? tools.filter((value) => value !== tool) : [...tools, tool] })} />
              <span className="mono break-all">{tool}</span>
            </label>;
          })}
        </fieldset>
        <label className="flex items-center gap-2 text-xs">
          <Checkbox checked={selected.data.trainable !== false}
            onChange={(event) => onPatch({ trainable: event.target.checked })} />
          参与 RL（trainable）
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={() => onEntry(selected.id)}>设为采集入口</Button>
          {selected.id === entryId && <StatusBadge tone="info">当前入口</StatusBadge>}
        </div>
        {selected.id === 'hub' && <FormField label="verify skill（空=关闭）">
          <Select value={selected.data.verify || ''} onChange={(event) => onPatch({ verify: event.target.value || null })}>
            <option value="">(none)</option>
            {(palette.skills || []).map((skill) => <option key={skill} value={skill}>{skill}</option>)}
          </Select>
        </FormField>}
        {selected.id === entryId && rlSettings}
      </div>}
    <p className="field-hint mt-4">图写入 WorkflowSpec YAML；画布位置仅用于当前视图。每题采样条数与采集入口独立。</p>
  </aside>;
});
