import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, BookOpen, ChevronDown, RefreshCw, X } from 'lucide-react';
import { masApi, type UserProjectFlowDoc } from '../api';
import { Button } from '../../../shared/ui/button';

export const FLOW_DOC_PROJECTS = ['agentflow', 'epc-aw-main'] as const;
export type FlowDocProject = typeof FLOW_DOC_PROJECTS[number];

/** 每个 step 可能是共享步骤（agent: "planner/executor/diagnoser"），按包含关系归入各 agent tab。 */
export const FLOW_DOC_TABS = ['planner', 'executor', 'diagnoser'] as const;
export type FlowDocTabId = typeof FLOW_DOC_TABS[number];

const PROJECT_LABELS: Record<FlowDocProject, string> = {
  agentflow: 'AgentFlow',
  'epc-aw-main': 'EPC-AW',
};

const TAB_LABELS: Record<FlowDocTabId, string> = {
  planner: 'Planner 规划器',
  executor: 'Executor 执行器',
  diagnoser: 'Diagnoser 校验器',
};

const tabLabel = (project: FlowDocProject, tab: FlowDocTabId) =>
  project === 'agentflow' && tab === 'diagnoser' ? 'Verifier 校验器' : TAB_LABELS[tab];

/** step.agent / prompt.agent 字段 → tab 归属（复用 flow-doc 面板中 agentBadge 的包含式判定）。 */
export const agentBadge = (agent: string): FlowDocTabId => {
  if (agent.includes('planner')) return 'planner';
  if (agent.includes('executor')) return 'executor';
  return 'diagnoser';
};

export const stepsForAgent = (doc: UserProjectFlowDoc | null, tab: FlowDocTabId) =>
  tab === 'diagnoser'
    ? (doc?.steps || []).filter((step) => agentBadge(step.agent) === tab || step.agent.includes('diagnoser'))
    : (doc?.steps || []).filter((step) => step.agent.includes(tab));

export const promptsForAgent = (doc: UserProjectFlowDoc | null, tab: FlowDocTabId) =>
  (doc?.prompts || []).filter((prompt) => agentBadge(prompt.agent) === tab || prompt.agent.includes(tab));

const RUNTIME_INACTIVE_HINT = '该步骤源码存在，但当前项目运行时不会执行（见步骤说明）。';

function StepTimeline({ doc, tab }: { doc: UserProjectFlowDoc; tab: FlowDocTabId }) {
  const steps = useMemo(() => stepsForAgent(doc, tab), [doc, tab]);
  return <ol className="flowdoc-steps" aria-label="主循环步骤">
    {steps.map(step => {
      const inactive = step.runtime_active === false;
      return <li
        className={`flowdoc-step${inactive ? ' flowdoc-step--inactive' : ''}`}
        key={`${step.seq}-${step.method}`}
        title={inactive ? RUNTIME_INACTIVE_HINT : undefined}
      >
        <span className="flowdoc-step-seq" aria-hidden="true">{String(step.seq).padStart(2, '0')}</span>
        <div className="flowdoc-step-body">
          <p className="flowdoc-step-head">
            <span className={`flowdoc-step-agent flowdoc-step-agent--${agentBadge(step.agent)}`}>{step.agent}</span>
            <code>{step.method}</code>
            {inactive && <span className="flowdoc-step-badge" role="note" title={RUNTIME_INACTIVE_HINT}>运行时不触发</span>}
          </p>
          <p className="flowdoc-step-desc">{step.desc}</p>
          {step.prompt && <p className="flowdoc-step-prompt">prompt · <code>{step.prompt}</code></p>}
        </div>
      </li>;
    })}
  </ol>;
}

function PromptList({ doc, tab }: { doc: UserProjectFlowDoc; tab: FlowDocTabId }) {
  const prompts = useMemo(() => promptsForAgent(doc, tab), [doc, tab]);
  return <div className="flowdoc-prompts">
    {prompts.map(prompt => <details className="flowdoc-prompt" key={prompt.path}>
      <summary><ChevronDown size={13} aria-hidden="true" /><code>{prompt.name}</code>
        {prompt.error && <span className="flowdoc-prompt-error">{prompt.error}</span>}
      </summary>
      {prompt.content
        ? <pre>{prompt.content}</pre>
        : <p className="flowdoc-prompt-empty">{prompt.error || '内容为空'}</p>}
    </details>)}
    {!prompts.length && <p className="field-hint">该 Agent 暂无 prompt 模板。</p>}
  </div>;
}

export function UserProjectFlowDoc({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [projectId, setProjectId] = useState<FlowDocProject>('agentflow');
  const [tab, setTab] = useState<FlowDocTabId>('planner');
  const [doc, setDoc] = useState<UserProjectFlowDoc | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // 手动刷新复用 AbortController：旧请求未中止时可能晚到覆盖新响应
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true); setError('');
    masApi.userProjectFlowDoc(projectId, controller.signal).then(data => {
      setDoc(data); setLoading(false);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return;
      setError(reason instanceof Error ? reason.message : '读取失败');
      setLoading(false);
    });
    return () => { controller.abort(); abortRef.current?.abort(); };
  }, [open, projectId]);
  const promptCount = useMemo(() => doc?.prompts.length ?? 0, [doc]);
  const manualRefresh = () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true); setError('');
    masApi.userProjectFlowDoc(projectId, controller.signal).then(data => {
      if (controller.signal.aborted) return;
      setDoc(data); setLoading(false);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return;
      setError(reason instanceof Error ? reason.message : '读取失败');
      setLoading(false);
    });
  };
  useEffect(() => () => abortRef.current?.abort(), []);
  const openAtRef = useRef(false);
  // 面板关闭重开时清空上次数据，避免先闪旧项目内容
  useEffect(() => {
    if (open && !openAtRef.current) { setDoc(null); }
    openAtRef.current = open;
  }, [open]);
  if (!open) return null;
  return <aside id="user-project-flow-doc" className="workflow-debug flowdoc-panel" aria-label="项目流程文档" onKeyDown={event => {
    if (event.key === 'Escape' && !event.defaultPrevented) {
      event.preventDefault(); event.stopPropagation(); onClose();
    }
  }}>
    <header className="workflow-debug-heading">
      <div><span>FLOW DOC</span><h2><BookOpen size={16} />项目流程文档</h2></div>
      <label className="flowdoc-project-select">
        项目
        <select value={projectId} onChange={event => setProjectId(event.target.value as FlowDocProject)}>
          {FLOW_DOC_PROJECTS.map(id => <option key={id} value={id}>{PROJECT_LABELS[id]}</option>)}
        </select>
      </label>
      <Button size="sm" variant="ghost" aria-label="重新读取" onClick={manualRefresh}><RefreshCw size={14} /></Button>
      <Button size="sm" variant="ghost" aria-label="关闭流程文档" onClick={onClose}><X size={16} /></Button>
    </header>
    <div className="workflow-debug-scroll flowdoc-scroll">
      {loading && <p className="mas-console-empty" role="status">正在读取流程文档…</p>}
      {error && <p className="mas-console-empty" role="alert">读取失败：{error}</p>}
      {!loading && !error && doc && <>
        <p className="field-hint">{doc.title} · 源码 {doc.source_root} · {promptCount} 个 prompt 模板</p>
        <div className="flowdoc-tabs" role="tablist" aria-label="Agent 维度查看">
          {FLOW_DOC_TABS.map(id => {
            const tabSteps = stepsForAgent(doc, id);
            const inactiveCount = tabSteps.filter(step => step.runtime_active === false).length;
            return <button
              key={id} type="button" role="tab" aria-selected={tab === id}
              className={`flowdoc-tab${tab === id ? ' is-active' : ''}`}
              onClick={() => setTab(id)}
            >{tabLabel(projectId, id)}{inactiveCount > 0 && <span className="flowdoc-tab-count" title={`${inactiveCount} 步运行时不触发`}>{inactiveCount}</span>}</button>;
          })}
        </div>
        <section aria-label="Agent 循环" className="flowdoc-loop-section"><AgentFlowStrip doc={doc} /></section>
        <section aria-label={`${tabLabel(projectId, tab)} 流程步骤`}>
          <h3>{tabLabel(projectId, tab)} · 主循环步骤（solver.py 源码顺序）</h3>
          <StepTimeline doc={doc} tab={tab} />
        </section>
        <section aria-label={`${tabLabel(projectId, tab)} Prompt 模板`}>
          <h3>{tabLabel(projectId, tab)} · Prompt 模板（{promptsForAgent(doc, tab).length}）</h3>
          <PromptList doc={doc} tab={tab} />
        </section>
      </>}
      {!loading && !error && !doc && <p className="mas-console-empty">未读取到流程文档。</p>}
    </div>
  </aside>;
}

function AgentFlowStrip({ doc }: { doc: UserProjectFlowDoc }) {
  return <div className="flowdoc-agents" aria-label="三 Agent 协作">
    {doc.agents.map((agent, index) => <div className="flowdoc-agent" key={agent.id}>
      <span className="flowdoc-agent-icon" aria-hidden="true">{index + 1}</span>
      <span><strong>{agent.id}</strong><small>{agent.label}</small></span>
    </div>)}
    <p className="flowdoc-agents-loop" aria-hidden="true">
      planner 生成计划 → executor 执行工具 → diagnoser 校验并反馈 <ArrowRight size={12} /> 循环直至 STOP
    </p>
  </div>;
}
