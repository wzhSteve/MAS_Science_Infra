import { memo, useEffect, useMemo, useState } from 'react';
import { Bot, Flag, Route, Trash2, Wrench, X } from 'lucide-react';
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
  { value: 'hub', label: 'Hub' },
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

function AgentModelFields({ data, defaultModel, options, loading, error, onRefresh, onPatch }: {
  data: GraphNodeData;
  defaultModel: AgentDefaultModel;
  options: AgentModelOption[];
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<unknown>;
  onPatch: (patch: Partial<GraphNodeData>) => void;
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
    <label className="mas-checkbox-row"><Checkbox checked={trainable}
      onChange={(event) => changeTrainable(event.target.checked)} />参与训练</label>
    {trainableError && <InlineNotice tone="warning">{trainableError}</InlineNotice>}
    <p className="field-hint">{trainable
      ? '该 Agent 的模型调用进入训练优化；模型必须具备可训练权重。'
      : '该 Agent 仍参与 Workflow 执行，但模型参数保持冻结。'}</p>
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
  const Icon = isTool ? Wrench : isRouter ? Route : Bot;
  const tools = Array.from(new Set([...(palette.tools || []), ...(selected.data.tools || [])]));
  const agents = nodes.filter((node) => node.type === 'agent' && node.data.kind !== 'tool');
  const toolNodes = nodes.filter((node) => node.type === 'tool' || node.data.kind === 'tool');
  const routerCandidates = selected.data.candidates || [];
  const routerStrategy = selected.data.strategy || 'llm_choice';
  return <aside className="mas-panel mas-inspector" aria-label="节点属性">
    <div className="mas-panel-heading">
      <div><span className="mas-panel-caption">{isTool ? 'Tool' : isRouter ? 'Router' : 'Agent'} 属性</span><h2><Icon size={16} />{selected.id}</h2></div>
      <Button size="sm" variant="ghost" onClick={onClose} aria-label="关闭节点属性"><X size={16} /></Button>
    </div>
    <div className="mas-panel-body">
      {isLegacyTool ? <div className="mas-property-section">
        <FormField label="工具标识"><Input value={selected.id} readOnly className="mono" /></FormField>
        <p className="field-hint">{selected.data.description || '确定性工具，通过标准 tool_call 调用。'}</p>
      </div> : isIntelligentTool ? <>
        <div className="mas-property-section">
          <FormField label="工具标识"><Input value={selected.id} readOnly className="mono" /></FormField>
          <p className="field-hint">{selected.data.description || '智能工具可使用模型、Memory 和独立训练配置。'}</p>
          <FormField label="角色"><Input value={selected.data.role || 'tool'} onChange={(event) => onPatch({ role: event.target.value })} /></FormField>
          <AgentModelFields data={selected.data} defaultModel={defaultModel}
            options={modelOptions} loading={modelOptionsLoading} error={modelOptionsError}
            onRefresh={onRefreshModelOptions} onPatch={onPatch} />
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
      </> : isRouter ? <>
        <div className="mas-property-section">
          <FormField label="路由策略">
            <Select value={routerStrategy} onChange={(event) => onPatch({ strategy: event.target.value as RouterStrategy })}>
              <option value="llm_choice">LLM 选择</option>
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
        <fieldset className="mas-property-section">
          <legend>候选 Agent</legend>
          {agents.map((agent) => {
            const aliases = agent.data.kind === 'blank' ? [agent.id, `blank:${agent.id}`] : [agent.id];
            const checked = routerCandidates.some((candidate) => aliases.includes(candidate));
            return <label key={agent.id} className="mas-checkbox-row">
              <Checkbox checked={checked} onChange={() => onPatch({
                candidates: checked
                  ? routerCandidates.filter((candidate) => !aliases.includes(candidate))
                  : [...routerCandidates, agent.id],
              })} />
              <span>{agent.id}<small className="field-hint"> · {agent.data.kind === 'blank' ? '自定义 Agent' : agent.data.kind || 'Agent'}</small></span>
            </label>;
          })}
          {!agents.length && <p className="field-hint">先在画布中添加 Agent。</p>}
        </fieldset>
        <fieldset className="mas-property-section">
          <legend>候选 Tool</legend>
          {toolNodes.map((tool) => {
            const checked = routerCandidates.includes(tool.id);
            return <label key={tool.id} className="mas-checkbox-row">
              <Checkbox checked={checked} onChange={() => onPatch({
                candidates: checked ? routerCandidates.filter((candidate) => candidate !== tool.id) : [...routerCandidates, tool.id],
              })} />
              <span>{tool.id}<small className="field-hint"> · {tool.data.kind === 'tool' ? '智能工具' : '内置工具'}</small></span>
            </label>;
          })}
          {!toolNodes.length && <p className="field-hint">画布中暂无 Tool。</p>}
        </fieldset>
        <div className="mas-property-section">
          <p className="field-hint">使用任务路由连线将上游 Agent 接入 Router；候选集合由这里维护，不创建普通下游边。</p>
          <JsonObjectField label="Router meta" value={selected.data.meta || {}} onChange={(meta) => onPatch({ meta })} />
        </div>
      </> : <>
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
          {selected.id === 'hub' && <p className="field-hint">hub 未显式绑定工具时会使用工作流的全局工具。移除绑定不等于禁止调用。</p>}
        </fieldset>
        <div className="mas-property-section">
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
