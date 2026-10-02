import { memo, useEffect, useMemo, useState } from 'react';
import { Bot, Boxes, Flag, Route, Trash2, Wrench, X } from 'lucide-react';
import type { AgentKind, Config, Palette, RouterStrategy } from '../../../shared/api/types';
import type { GraphNode, GraphNodeData, SelectedGraphNode } from '../types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { Textarea } from '../../../shared/ui/textarea';
import { Checkbox } from '../../../shared/ui/checkbox';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import {
  AgentModelSelect,
  resolveAgentModel,
  type AgentDefaultModel,
  type AgentModelOption,
} from '../../resources/components/AgentModelSelect';

const AGENT_KINDS: Array<{ value: Exclude<AgentKind, 'tool'>; label: string }> = [
  { value: 'planner', label: 'Planner' },
  { value: 'verifier', label: 'Verifier' },
  { value: 'blank', label: '自定义 Agent' },
];

function JsonObjectField({ label, value, onChange }: {
  label: string;
  value: Config;
  onChange: (value: Config) => void;
}) {
  const serialized = JSON.stringify(value, null, 2);
  const [text, setText] = useState(serialized);
  const [error, setError] = useState('');
  useEffect(() => { setText(serialized); setError(''); }, [serialized]);
  const commit = () => {
    try {
      const parsed: unknown = JSON.parse(text || '{}');
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('请输入 JSON 对象');
      onChange(parsed as Config);
      setError('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  return <FormField label={label} hint="JSON 对象；离开输入框时应用。">
    <div>
      <Textarea rows={5} className="mono" value={text} onChange={(event) => setText(event.target.value)} onBlur={commit} />
      {error && <InlineNotice tone="danger">{error}</InlineNotice>}
    </div>
  </FormField>;
}

function AgentModelFields({ data, defaultModel, options, loading, error, onRefresh, onPatch, allowTrainable = true }: {
  data: GraphNodeData;
  defaultModel: AgentDefaultModel;
  options: AgentModelOption[];
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<unknown>;
  onPatch: (patch: Partial<GraphNodeData>) => void;
  allowTrainable?: boolean;
}) {
  const trainable = data.trainable !== false;
  const [trainableError, setTrainableError] = useState('');
  const effective = useMemo(
    () => resolveAgentModel(data.model, options, defaultModel),
    [data.model, defaultModel, options],
  );
  useEffect(() => setTrainableError(''), [data.model]);
  const changeTrainable = (next: boolean) => {
    if (next && (!effective.available || !effective.trainable)) {
      setTrainableError(effective.available
        ? `${effective.name} 没有可训练权重。请先选择本地可训练模型。`
        : `${effective.inherited ? '实验默认模型' : effective.name} 不可用。请先选择本地可训练模型。`);
      return;
    }
    setTrainableError('');
    onPatch({ trainable: next });
  };
  return <>
    <FormField label="模型" hint="每个 Agent 都需要模型；继承表示使用实验默认模型。">
      <AgentModelSelect value={data.model || 'inherit'} defaultModel={defaultModel}
        trainable={trainable} options={options} loading={loading} error={error} onRefresh={onRefresh}
        onChange={(model) => onPatch({ model })} />
    </FormField>
    {allowTrainable ? <>
      <label className="mas-checkbox-row"><Checkbox checked={trainable}
        onChange={(event) => changeTrainable(event.target.checked)} />参与训练</label>
      {trainableError && <InlineNotice tone="warning">{trainableError}</InlineNotice>}
      <p className="field-hint">{trainable
        ? '该 Agent 的模型调用进入训练优化；模型必须具备可训练权重。'
        : '该 Agent 仍参与 Workflow 执行，但模型参数保持冻结。'}</p>
    </> : <p className="field-hint">tool-agent 不可训练，只参与执行。</p>}
  </>;
}

export const NodeInspector = memo(function NodeInspector({
  selected, nodes, palette, entryId, defaultModel, modelOptions, modelOptionsLoading, modelOptionsError,
  onRefreshModelOptions, onPatch, onEntry, onDelete, onClose, onModelResources,
}: {
  selected: SelectedGraphNode;
  nodes: GraphNode[];
  palette: Palette;
  entryId: string;
  defaultModel: AgentDefaultModel;
  modelOptions: AgentModelOption[];
  modelOptionsLoading: boolean;
  modelOptionsError: string | null;
  onRefreshModelOptions: () => Promise<unknown>;
  onPatch: (patch: Partial<GraphNodeData>) => void;
  onEntry: (id: string) => void;
  onDelete: () => void;
  onClose: () => void;
  onModelResources: () => void;
}) {
  const isLegacyTool = selected.type === 'tool';
  const isIntelligentTool = selected.type === 'agent' && selected.data.kind === 'tool';
  const isTool = isLegacyTool || isIntelligentTool;
  const isRouter = selected.type === 'router';
  const isPool = selected.type === 'pool';
  const Icon = isPool ? Boxes : isTool ? Wrench : isRouter ? Route : Bot;
  const members = selected.data.members || [];
  const toolChoices = (palette.tool_agents || []).filter((tool) => !members.includes(tool.id));
  const wraps = typeof selected.data.meta?.wraps === 'string' ? selected.data.meta.wraps : '';
  const userWrap = selected.data.profile?.backend === 'user_space' || selected.data.backend === 'user_space' || wraps.startsWith('EPC-AW.');
  const tools = Array.from(new Set([...(palette.tools || []), ...(selected.data.tools || [])]));
  const agents = nodes.filter((node) => node.type === 'agent' && node.data.kind !== 'tool');
  const routerCandidates = selected.data.candidates || selected.data.members || [];
  const routerStrategy = selected.data.strategy || 'from_plan';
  return <aside className="mas-panel mas-inspector" aria-label="节点属性">
    <div className="mas-panel-heading">
      <div><span className="mas-panel-caption">{isPool ? 'Pool' : isTool ? 'Tool' : isRouter ? 'Router' : 'Agent'} 属性</span><h2><Icon size={16} />{isPool ? 'tool-agent pool' : selected.id}</h2></div>
      <Button size="sm" variant="ghost" onClick={onClose} aria-label="关闭节点属性"><X size={16} /></Button>
    </div>
    <div className="mas-panel-body">
      {isLegacyTool ? <div className="mas-property-section">
        <FormField label="工具标识"><Input value={selected.id} readOnly className="mono" /></FormField>
        <p className="field-hint">{selected.data.description || '确定性工具，通过标准 tool_call 调用。'}</p>
      </div> : isIntelligentTool ? <>
        <div className="mas-property-section">
          <FormField label="工具标识"><Input value={selected.id} readOnly className="mono" /></FormField>
          <p className="field-hint">{selected.data.description || '封装 tool-agent：内部调用工具内核，再用 LLM 完成任务。不可训练。'}</p>
          <FormField label="角色"><Input value={selected.data.role || 'tool'} onChange={(event) => onPatch({ role: event.target.value })} /></FormField>
          <AgentModelFields data={selected.data} defaultModel={defaultModel}
            options={modelOptions} loading={modelOptionsLoading} error={modelOptionsError}
            onRefresh={onRefreshModelOptions} onPatch={onPatch} allowTrainable={false} />
          <FormField label="系统提示词">
            <Textarea rows={7} value={selected.data.system_prompt || ''} onChange={(event) => onPatch({ system_prompt: event.target.value })} />
          </FormField>
          <FormField label="Memory scope">
            <Input value={selected.data.memory_scope || 'agent'} onChange={(event) => onPatch({ memory_scope: event.target.value })} />
          </FormField>
        </div>
        <details className="mas-property-section">
          <summary>扩展字段</summary>
          <JsonObjectField label="profile" value={selected.data.profile || {}} onChange={(profile) => onPatch({ profile })} />
          <JsonObjectField label="meta" value={selected.data.meta || {}} onChange={(meta) => onPatch({ meta })} />
        </details>
      </> : isPool ? <div className="mas-property-section">
        <p className="field-hint">Router 的下游是这个 pool。在这里增加或删除 tool-agent，成员不可训练。</p>
        <ul className="mas-pool-editor">
          {members.map((member) => {
            const tiers = selected.data.memberTiers || {};
            const tier = tiers[member] === 'pro' ? 'pro' : 'lite';
            return <li key={member}>
              <span>{member}</span>
              <Select className="mas-pool-tier" aria-label={`${member} 级别`} value={tier} onChange={(event) => onPatch({
                memberTiers: { ...tiers, [member]: event.target.value === 'pro' ? 'pro' : 'lite' },
              })}>
                <option value="lite">lite</option>
                <option value="pro">pro</option>
              </Select>
              <Button size="sm" variant="ghost" aria-label={`移除 ${member}`} onClick={() => onPatch({
                members: members.filter((item) => item !== member),
              })}>移除</Button>
            </li>;
          })}
          {!members.length && <li className="is-empty">pool 里还没有 tool-agent</li>}
        </ul>
        <FormField label="添加 tool-agent">
          <Select value="" onChange={(event) => {
            const id = event.target.value;
            if (!id || members.includes(id)) return;
            onPatch({ members: [...members, id] });
          }}>
            <option value="">选择要加入的 tool-agent</option>
            {toolChoices.map((tool) => <option key={tool.id} value={tool.id}>{tool.id}</option>)}
          </Select>
        </FormField>
        {!toolChoices.length && <p className="field-hint">可用的 tool-agent 都已在 pool 中。</p>}
      </div> : isRouter ? <>
        <div className="mas-property-section">
          <FormField label="路由策略">
            <Select value={routerStrategy} onChange={(event) => onPatch({ strategy: event.target.value as RouterStrategy })}>
              <option value="from_plan">按计划拆分（from_plan）</option>
              <option value="llm_choice">按计划拆分（llm_choice 别名）</option>
              <option value="score">候选评分</option>
              <option value="round_robin">轮询</option>
            </Select>
          </FormField>
          {routerStrategy === 'score' && <FormField label="Scorer Agent">
            <Select value={selected.data.scorer || ''} onChange={(event) => onPatch({ scorer: event.target.value || null })}>
              <option value="">未指定</option>
              {agents.map((agent) => <option key={agent.id} value={agent.id}>{agent.id}</option>)}
            </Select>
          </FormField>}
        </div>
        <div className="mas-property-section">
          <p className="field-hint">Router 的下游固定是 tool-agent pool。点击 pool 增加或删除成员。当前成员：{(routerCandidates.length ? routerCandidates.join('、') : '无')}。</p>
          <JsonObjectField label="Router meta" value={selected.data.meta || {}} onChange={(meta) => onPatch({ meta })} />
        </div>
      </> : userWrap ? <div className="mas-property-section">
        <p className="field-hint">
          用户封装节点{wraps ? ` · ${wraps}` : ''}。
          代码在 <code>user_space/projects/{String(selected.data.profile?.user_project || '')}</code>，只能由悬浮辅助 AI 修改。
          工具在 executor 窗口内部执行，不在画布上绑定。采样点仍可挂在此节点上。
        </p>
        <dl className="mas-settings-details">
          <dt>显示名</dt><dd>{selected.data.label || selected.id}</dd>
          <dt>角色</dt><dd>{selected.data.role || selected.data.kind || 'agent'}</dd>
          <dt>窗口</dt><dd className="mono">{selected.id}</dd>
          <dt>训练</dt><dd>{selected.data.trainable === false ? '已冻结' : '参与训练'}</dd>
          <dt>模型</dt><dd>继承实验配置</dd>
        </dl>
      </div> : <>
        <div className="mas-property-section">
          <FormField label="Agent 类型">
            <Select value={selected.data.kind || 'blank'} onChange={(event) =>
              onPatch({ kind: event.target.value as AgentKind })}>
              {AGENT_KINDS.map((kind) => <option key={kind.value} value={kind.value}>{kind.label}</option>)}
            </Select>
          </FormField>
          <FormField label="角色"><Input value={selected.data.role || ''} onChange={(e) => onPatch({ role: e.target.value })} /></FormField>
          <FormField label="系统提示词" hint="定义 Agent 的职责、行为和输出要求。">
            <Textarea rows={7} value={selected.data.system_prompt || ''} placeholder="描述这个 Agent 应该完成什么…"
              onChange={(e) => onPatch({ system_prompt: e.target.value })} />
          </FormField>
          <FormField label="Skills" hint="多个技能以逗号分隔。">
            <Input value={(selected.data.skills || []).join(', ')} onChange={(e) =>
              onPatch({ skills: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} />
          </FormField>
          <FormField label="Memory scope">
            <Input value={selected.data.memory_scope || 'agent'} onChange={(event) => onPatch({ memory_scope: event.target.value })} />
          </FormField>
        </div>
        <div className="mas-property-section">
          <h3>模型与训练</h3>
          <AgentModelFields data={selected.data} defaultModel={defaultModel}
            options={modelOptions} loading={modelOptionsLoading} error={modelOptionsError}
            onRefresh={onRefreshModelOptions} onPatch={onPatch} />
          <Button size="sm" variant="ghost" onClick={onModelResources}>管理模型</Button>
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
          {selected.data.kind === 'planner' && <p className="field-hint">Planner 不直接调用工具。把 tool-agent 放进 Router 下游的 pool。</p>}
        </fieldset>
        <div className="mas-property-section">
          <Button size="sm" disabled={selected.id === entryId} onClick={() => onEntry(selected.id)}>
            <Flag size={14} />{selected.id === entryId ? '当前运行入口' : '设为运行入口'}
          </Button>
        </div>
        {selected.data.kind === 'planner' && <details className="mas-property-section">
          <summary>高级配置</summary>
          <FormField label="验证技能" hint="启用后通过 Verifier 沿 feedback 回 Planner。">
            <Select value={selected.data.verify || ''} onChange={(e) => onPatch({ verify: e.target.value || null })}>
              <option value="">关闭</option>
              {(palette.skills || []).map((skill) => <option key={skill} value={skill}>{skill}</option>)}
            </Select>
          </FormField>
        </details>}
        <details className="mas-property-section">
          <summary>扩展字段</summary>
          <JsonObjectField label="profile" value={selected.data.profile || {}} onChange={(profile) => onPatch({ profile })} />
          <JsonObjectField label="meta" value={selected.data.meta || {}} onChange={(meta) => onPatch({ meta })} />
        </details>
      </>}
    </div>
    <div className="mas-panel-footer"><Button size="sm" variant="danger" onClick={onDelete}><Trash2 size={14} />删除节点</Button></div>
  </aside>;
});
