import { memo, useCallback, useEffect, useState, type DragEvent } from 'react';
import { AlertDialog } from 'radix-ui';
import { Bot, Layers, Plus, Route, Search, Settings2, Wrench, X } from 'lucide-react';
import type { AgentKind, Palette, WorkflowSpec } from '../../../shared/api/types';
import type { GraphNode, GraphNodePreset } from '../types';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { masApi, type EpcAwRuntimeConfig } from '../api';

export const NODE_TRANSFER = 'application/science-node';
export type LibraryTab = 'agent' | 'tool' | 'router' | 'template' | 'user';

const DEFAULT_TOOLS = [
  'Base_Generator_Tool',
  'Python_Coder_Tool',
  'Wikipedia_Search_Tool',
  'Bing_Search_Tool',
  'Web_Fetch_Tool',
];

const DEFAULT_RUNTIME: EpcAwRuntimeConfig = {
  n: 1,
  max_steps: 20,
  max_time: 3000,
  max_tokens: 4000,
  temperature: 0,
  enabled_tools: [...DEFAULT_TOOLS],
};

function isUserSpaceNode(node: GraphNode): boolean {
  return node.data.backend === 'user_space' || node.data.profile?.backend === 'user_space'
    || String(node.data.meta?.wraps || '').startsWith('EPC-AW');
}

function EpcAwRuntimePanel({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const [form, setForm] = useState<EpcAwRuntimeConfig>(DEFAULT_RUNTIME);
  const [status, setStatus] = useState<'idle' | 'loading' | 'saving' | 'saved' | 'error'>('loading');
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    void masApi.userProjectRuntime(projectId).then((data) => {
      if (cancelled) return;
      setForm({ ...DEFAULT_RUNTIME, ...data });
      setStatus('idle');
    }).catch((err: Error) => {
      if (cancelled) return;
      setError(err.message || '读取失败');
      setStatus('error');
    });
    return () => { cancelled = true; };
  }, [projectId]);

  const save = useCallback(async () => {
    setStatus('saving');
    setError('');
    try {
      const saved = await masApi.putUserProjectRuntime(projectId, form);
      setForm({ ...DEFAULT_RUNTIME, ...saved });
      setStatus('saved');
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存失败');
      setStatus('error');
    }
  }, [form, projectId]);

  const toggleTool = (tool: string) => {
    const current = form.enabled_tools || [];
    const next = current.includes(tool) ? current.filter((item) => item !== tool) : [...current, tool];
    setForm((prev) => ({ ...prev, enabled_tools: next.length ? next : ['Base_Generator_Tool'] }));
  };

  return <div className="mas-epc-runtime" onClick={(event) => event.stopPropagation()}>
    <div className="mas-epc-runtime__head">
      <strong>EPC-AW 运行超参</strong>
      <Button size="sm" variant="ghost" onClick={onClose} aria-label="关闭超参">关闭</Button>
    </div>
    {status === 'loading' ? <p className="field-hint">读取中…</p> : <>
      <label className="mas-epc-runtime__field">n（候选计划数）
        <Input type="number" min={1} max={16} value={form.n ?? 1}
          onChange={(event) => setForm((prev) => ({ ...prev, n: Math.max(1, Math.min(16, Number(event.target.value) || 1)) }))} />
      </label>
      <label className="mas-epc-runtime__field">max_steps
        <Input type="number" min={1} max={40} value={form.max_steps ?? 20}
          onChange={(event) => setForm((prev) => ({ ...prev, max_steps: Math.max(1, Math.min(40, Number(event.target.value) || 20)) }))} />
      </label>
      <label className="mas-epc-runtime__field">max_time（秒）
        <Input type="number" min={30} max={10000} value={form.max_time ?? 3000}
          onChange={(event) => setForm((prev) => ({ ...prev, max_time: Math.max(30, Number(event.target.value) || 3000) }))} />
      </label>
      <label className="mas-epc-runtime__field">max_tokens
        <Input type="number" min={256} max={16384} value={form.max_tokens ?? 4000}
          onChange={(event) => setForm((prev) => ({ ...prev, max_tokens: Math.max(256, Number(event.target.value) || 4000) }))} />
      </label>
      <label className="mas-epc-runtime__field">temperature
        <Input type="number" min={0} max={2} step={0.1} value={form.temperature ?? 0}
          onChange={(event) => setForm((prev) => ({ ...prev, temperature: Number(event.target.value) || 0 }))} />
      </label>
      <fieldset className="mas-epc-runtime__tools">
        <legend>enabled_tools</legend>
        {DEFAULT_TOOLS.map((tool) => (
          <label key={tool} className="mas-checkbox-row">
            <input type="checkbox" checked={(form.enabled_tools || []).includes(tool)} onChange={() => toggleTool(tool)} />
            <span>{tool}</span>
          </label>
        ))}
      </fieldset>
      {error && <p className="field-hint" style={{ color: 'var(--danger, #b42318)' }}>{error}</p>}
      {status === 'saved' && <p className="field-hint">已保存，下次测试/采样生效。</p>}
      <Button size="sm" variant="primary" disabled={status === 'saving'} onClick={() => void save()}>
        {status === 'saving' ? '保存中…' : '保存超参'}
      </Button>
    </>}
  </div>;
}

export const GraphPalette = memo(function GraphPalette({ palette, nodes, poolMembers = [], tab, onTabChange, onAdd, onTemplate, onClose }: {
  palette: Palette;
  nodes: GraphNode[];
  poolMembers?: string[];
  tab: LibraryTab;
  onTabChange: (tab: LibraryTab) => void;
  onAdd: (preset: GraphNodePreset) => void;
  onTemplate: (workflow: WorkflowSpec) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState('');
  const [replacement, setReplacement] = useState<WorkflowSpec | null>(null);
  const [runtimeProject, setRuntimeProject] = useState<string | null>(null);
  const matches = (value: string) => value.toLowerCase().includes(query.trim().toLowerCase());
  const labels: Record<AgentKind, string> = {
    planner: '基础 Planner', verifier: '基础 Verifier', blank: '自定义 Agent', tool: '封装工具',
  };
  const agentKinds: AgentKind[] = ['planner', 'verifier', 'blank'];
  const knownKinds = new Set<AgentKind>([...agentKinds, 'tool']);
  const hasUserPlanner = nodes.some((node) => (node.id === 'planner' || node.data.kind === 'planner') && isUserSpaceNode(node));
  const hasUserVerifier = nodes.some((node) => (node.id === 'verifier' || node.data.kind === 'verifier') && isUserSpaceNode(node));
  const agents: GraphNodePreset[] = [
    ...agentKinds.map((kind) => ({
      nodeType: 'agent' as const,
      id: kind === 'blank' ? 'custom_agent' : kind,
      agentKind: kind,
      role: kind === 'blank' ? 'agent' : kind,
      description: labels[kind],
    })),
    ...(palette.roles || []).filter((role) => !knownKinds.has(role as AgentKind) && role !== 'orchestrator')
      .map((role) => ({ nodeType: 'agent' as const, id: role, agentKind: 'blank' as const, role, description: role })),
  ];
  const toolInfo = new Map((palette.tool_agents || []).map((tool) => [tool.id, tool]));
  const toolIds = Array.from(new Set([...(palette.tools || []), ...toolInfo.keys()]));
  const tools: GraphNodePreset[] = toolIds.map((id) => {
    const info = toolInfo.get(id);
    return {
      nodeType: 'agent' as const,
      id,
      agentKind: 'tool' as const,
      role: 'tool',
      backend: info?.backend || 'llm',
      llmRequired: true,
      description: info?.description || '封装 tool-agent（内核 + LLM）',
    };
  });
  const items = (tab === 'agent' ? agents : tab === 'tool' ? tools
      : [{ nodeType: 'router' as const, id: 'router', description: '把上游计划拆给下游 Agent set' }])
    .filter((item) => matches(item.id) || matches(item.description || ''));
  const templates = (palette.templates || []).filter((t) => matches(t.label) || matches(t.id));
  const userProjects = (palette.user_projects || []).filter((p) => matches(p.title) || matches(p.id));
  const drag = (event: DragEvent, preset: GraphNodePreset) => {
    event.dataTransfer.setData(NODE_TRANSFER, JSON.stringify(preset));
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
      {([['agent', 'Agent'], ['tool', 'Tool'], ['router', 'Router'], ['template', '模板'], ['user', '用户']] as const).map(([value, label]) =>
        <button key={value} type="button" aria-pressed={tab === value} onClick={() => onTabChange(value)}>{label}</button>)}
    </div>
    <div className="mas-panel-body mas-library-items">
      {tab === 'user' ? userProjects.map((project) => {
        const ready = project.status === 'ready' && project.workflow;
        const family = project.family || (String(project.wraps || '').includes('EPC-AW') ? 'epc_aw' : '');
        const isEpc = family === 'epc_aw';
        return <div key={project.id} className="mas-library-user-block">
          <button type="button" className="mas-library-item" disabled={!ready}
            onClick={() => {
              if (!project.workflow) return;
              if (project.mode === 'native_mas') {
                if (nodes.length) setReplacement(project.workflow);
                else onTemplate(project.workflow);
                return;
              }
              const agent = project.workflow.agents?.find((node) => node.id === project.agent_ids?.[0]) || project.workflow.agents?.[0];
              if (!agent) return;
              onAdd({
                nodeType: 'agent',
                id: agent.id,
                agentKind: (agent.kind as AgentKind) || 'blank',
                role: 'agent',
                backend: 'user_space',
                userProject: project.id,
                description: project.title,
                profile: agent.profile,
              });
            }}>
            <span className="mas-library-icon"><Bot size={17} /></span>
            <span><strong>{project.title}</strong>
              <small>{isEpc ? 'EPC-AW · 用户封装' : project.mode === 'native_mas' ? '整图导入 · Mode 2' : 'I/O 模块 · Mode 1'}{ready ? '' : ' · 未就绪'}</small></span>
            {ready && <Plus size={14} className="mas-library-plus" aria-hidden="true" />}
          </button>
          {isEpc && ready && <div className="mas-library-user-actions">
            <Button size="sm" variant="ghost" onClick={() => setRuntimeProject((cur) => cur === project.id ? null : project.id)}>
              <Settings2 size={14} /> 超参
            </Button>
          </div>}
          {runtimeProject === project.id && <EpcAwRuntimePanel projectId={project.id} onClose={() => setRuntimeProject(null)} />}
        </div>;
      })
        : tab === 'template' ? templates.map((template) =>
        <button type="button" className="mas-library-item" key={template.id} disabled={!template.workflow}
          onClick={() => {
            if (!template.workflow) return;
            if (nodes.length) setReplacement(template.workflow);
            else onTemplate(template.workflow);
          }}>
          <span className="mas-library-icon"><Layers size={17} /></span>
          <span><strong>{template.label.replace(' (executable)', '')}</strong><small>{nodes.length ? '替换当前工作流' : '从此模板开始'}</small></span>
        </button>)
        : items.map((item) => {
          const isInfraPlanner = item.agentKind === 'planner' || item.id === 'planner';
          const isInfraVerifier = item.agentKind === 'verifier' || item.id === 'verifier';
          const blockedByUser = (isInfraPlanner && hasUserPlanner) || (isInfraVerifier && hasUserVerifier);
          const exists = isInfraPlanner || isInfraVerifier
            ? nodes.some((node) => node.id === item.id || node.data.kind === item.agentKind)
            : tab === 'tool' && poolMembers.includes(item.id);
          const disabled = exists || blockedByUser;
          const Icon = tab === 'tool' ? Wrench : tab === 'router' ? Route : Bot;
          const hint = blockedByUser
            ? (isInfraPlanner ? '画布已是用户封装 Planner' : '画布已是用户封装 Verifier')
            : exists
              ? (tab === 'tool' ? '已在 pool 中' : '已添加')
              : tab === 'agent'
                ? 'MAS Infra 内置，非用户封装'
                : tab === 'tool' ? '加入 tool-agent pool' : item.description;
          return <button type="button" className="mas-library-item" key={item.id} disabled={disabled} draggable={!disabled}
            onDragStart={(event) => drag(event, item)} onClick={() => onAdd(item)}>
            <span className="mas-library-icon"><Icon size={17} aria-hidden="true" /></span>
            <span><strong>{tab === 'agent' ? item.description : item.id}</strong>
              <small>{hint}</small></span>
            {!disabled && <Plus size={14} className="mas-library-plus" aria-hidden="true" />}
          </button>;
        })}
      {(tab === 'user' ? !userProjects.length : tab === 'template' ? !templates.length : !items.length)
        && <p className="mas-panel-empty">没有匹配的{tab === 'user' ? '用户项目' : tab === 'template' ? '模板' : '节点'}</p>}
    </div>
    <p className="mas-panel-footnote">{tab === 'user' ? '用户封装（如 EPC-AW）只能从这里导入；基础 Planner 在 Agent Tab。' : tab === 'template' ? '模板包含节点、连线和工作流配置。' : tab === 'agent' ? '基础角色属于 MAS Infra；EPC-AW 等用户 MAS 请用「用户」Tab。' : '点击或拖拽连续添加，点击右上角关闭工具箱。'}</p>
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
