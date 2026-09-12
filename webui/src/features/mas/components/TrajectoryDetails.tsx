import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Bot, ChevronDown, LocateFixed, MessageSquare, Wrench } from 'lucide-react';
import type { RolloutTrajectoryResponse } from '../../../shared/api/types';
import { errorMessage, request } from '../../../shared/api/http';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { JsonDetails } from '../../../shared/components/JsonDetails';
import {
  adaptTrajectory, agentLabel, elapsed, eventLocation, items, mergeEvents, readEventPage,
  readFullEvent, readMessagePage, record, snippet, text, trajectoryUrl,
  type EventView, type RecordValue, type TrajectoryLocation,
} from '../model/trajectoryView';

type ViewTab = 'executions' | 'conversation' | 'tools' | 'events';
type Props = { data: RolloutTrajectoryResponse; experimentId: string; onLocate: (target: TrajectoryLocation) => void };
const tabs: Array<{ id: ViewTab; label: string }> = [
  { id: 'executions', label: '执行片段' }, { id: 'conversation', label: '对话' },
  { id: 'tools', label: '工具调用' }, { id: 'events', label: '事件' },
];

function Value({ value, label }: { value: unknown; label: string }) {
  if (value === undefined || value === null) return <p className="field-hint">{label}：未记录</p>;
  return typeof value === 'string' ? <details className="mas-trace-value">
    <summary>{label}</summary><pre>{value}</pre>
  </details> : <JsonDetails value={value} label={label} />;
}

function Locate({ event, onLocate }: { event: EventView; onLocate: Props['onLocate'] }) {
  const location = eventLocation(event);
  return location ? <Button size="sm" variant="ghost" title="定位历史记录涉及的实体，不改变画布"
    onClick={() => onLocate(location)}><LocateFixed size={12} />定位</Button> : null;
}

function EventContent({ event, payload }: { event: EventView; payload: RecordValue }) {
  switch (event.kind) {
    case 'agent_enter': {
      const context = record(payload.context);
      return <>
        <p className="field-hint">Agent 进入时的配置；模型实际接收的上下文以各次“模型请求”为准。</p>
        <Value value={context.system_prompt} label="有效系统提示词" />
        <Value value={context.skills} label="声明的 Skills（实际执行见 Skill 事件）" />
        <Value value={payload.enabled_tools ?? context.tool_definitions} label="可用工具" />
        <Value value={context.model} label="模型绑定" />
        <Value value={context.handoff_from_execution_id} label="上游执行片段" />
      </>;
    }
    case 'model_call': return <>
      <p className="field-hint">记录格式：{text(payload.request_format) || '未记录'} · 以下内容来自实际调用边界。</p>
      <Value value={payload.messages} label="实际请求上下文" />
      <Value value={payload.tool_definitions} label="本次请求的工具定义" />
      <Value value={payload.sampling_parameters} label="采样参数" />
      <Value value={payload.context_processing} label="上下文处理与裁剪" />
      {payload.retry_of && <Value value={payload.retry_of} label="重试来源调用" />}
    </>;
    case 'model_result': return <>
      <Value value={payload.message} label="模型可见返回" />
      <Value value={payload.finish_reason} label="结束原因" />
      <Value value={payload.model} label="服务端返回的模型身份" />
      <Value value={payload.usage} label="服务端返回的用量" />
      <Value value={payload.logprobs} label="服务端返回的概率信息（不等于完整分布熵）" />
    </>;
    case 'tool_call': return <Value value={payload.args ?? payload.arguments} label="工具输入参数" />;
    case 'tool_result': return <Value value={payload.content ?? payload.output ?? payload.result} label="工具返回" />;
    case 'memory_read':
    case 'memory_write': return <>
      <p className="field-hint">作用域：{text(payload.scope) || '未记录'} · Owner：{text(payload.owner) || '未记录'}。读写记录不代表全部内容已进入模型上下文。</p>
      <Value value={payload.items ?? payload.item ?? payload.content ?? payload} label="记忆读写内容" />
    </>;
    default: return <Value value={payload} label="记录内容" />;
  }
}

function EventDetails({ event, url, onLocate }: { event: EventView; url: string; onLocate: Props['onLocate'] }) {
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState<RecordValue | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const load = async () => {
    if (!event.id) return;
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setLoading(true); setError(null);
    try {
      const response = await request<unknown>(url, { signal: current.signal });
      if (!current.signal.aborted) setFull(readFullEvent(response, event.id));
    } catch (reason) {
      if (!current.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (!current.signal.aborted) setLoading(false);
    }
  };
  const payload = full ? record(full.payload) : event.payload;
  const description = text(payload.error) || text(payload.name) || text(payload.skill) || text(payload.text)
    || text(payload.termination_reason) || text(record(payload.message).content);
  return <article className={`mas-trace-event${event.kind === 'error' || payload.status === 'failed' || payload.ok === false ? ' has-error' : ''}`}>
    <div className="mas-trace-event-heading">
      <button type="button" aria-expanded={open} onClick={() => {
        const next = !open; setOpen(next);
        if (next && event.preview && event.id && !full && !loading) void load();
      }}><ChevronDown size={13} className={open ? 'is-open' : ''} /><strong>{event.title}</strong>
        <span>{event.status || `#${event.order + 1}`}</span></button>
      <Locate event={event} onLocate={onLocate} />
    </div>
    <div className="mas-trace-event-caption"><span>{agentLabel(event)}</span>
      <span>{event.occurredAt ? `发生时间 ${new Date(event.occurredAt).toLocaleTimeString('zh-CN', { hour12: false })}`
        : event.recordedAt ? `记录时间 ${new Date(event.recordedAt).toLocaleTimeString('zh-CN', { hour12: false })}` : '时间未记录'}</span>
    </div>
    {description && <p className="mas-trace-snippet">{snippet(description)}</p>}
    {open && <div className="mas-trace-event-body">
      {event.preview && !full && <p className="field-hint">{event.previewTruncated ? '此处为已截断的摘要。' : '此处为预览。'}展开后按需读取完整事件。</p>}
      {loading && <p className="field-hint" role="status">正在读取完整事件…</p>}
      {error && <InlineNotice tone="warning">{error}<Button size="sm" onClick={() => void load()}>重试读取</Button></InlineNotice>}
      {Boolean(payload.error) && <InlineNotice tone="danger">{text(payload.error) || '事件记录了结构化错误，详情见原始记录。'}</InlineNotice>}
      <EventContent event={event} payload={payload} />
      <JsonDetails value={full || event.raw} label={full || !event.preview ? '原始事件与关联身份' : '原始事件预览（可能截断）'} />
    </div>}
  </article>;
}

function Message({ value, index }: { value: unknown; index: number }) {
  const message = record(value);
  const role = text(message.role) || 'unknown';
  const names: Record<string, string> = { user: '用户', assistant: '模型', tool: '工具返回', system: '系统提示词' };
  return <article className="mas-trace-message">
    <header><strong>{names[role] || role}</strong><span>消息 {index + 1} · 合并对话，不据此推断 Agent 归属</span></header>
    <Value value={message.content} label={role === 'system' ? '系统提示词（本条消息）' : '消息正文'} />
    {Boolean(message.tool_calls) && <Value value={message.tool_calls} label="工具调用请求" />}
    {Boolean(message.tool_call_id) && <Value value={message.tool_call_id} label="关联调用 ID" />}
  </article>;
}

function Capabilities({ data }: { data: RolloutTrajectoryResponse }) {
  const view = adaptTrajectory(data);
  const coverage = record(view.meta.trace_coverage);
  const capabilities = record(view.modelIdentity.capabilities);
  const labels: Record<string, string> = { available: '服务记录为可用', unsupported: '服务不支持', unknown: '未确认' };
  const coverageLabels: Record<string, string> = {
    messages: '消息', execution_cursor: '执行游标', agent_memory: 'Agent 记忆',
    shared_memory: '共享记忆', external_tool_state: '工具外部状态', policy_identity: '策略身份',
  };
  return <details className="mas-trace-capabilities">
    <summary>记录与能力 · {data.run.trace_status === 'complete' ? '记录完整' : '存在记录限制'}</summary>
    <p className="field-hint">执行成功、记录完整、可恢复和满足训练算法要求是不同状态。此处不提供通用“可训练”结论。</p>
    {data.metadata_truncated && <InlineNotice tone="warning">部分元数据仅为预览，完整记录请使用导出。</InlineNotice>}
    <dl>
      <dt>当前调用记录</dt><dd>{text(coverage.current_calls) || '未标注'}</dd>
      <dt>历史前缀</dt><dd>{text(coverage.historical_prefix) || '未标注'}</dd>
      <dt>模型边界缺失</dt><dd>{typeof coverage.missing_model_boundaries === 'boolean' ? coverage.missing_model_boundaries ? '是' : '记录为否' : '未知'}</dd>
      <dt>策略版本</dt><dd>{text(view.modelIdentity.policy_version) || data.run.model?.policy_version || '未提供'}</dd>
      {(['token_ids', 'logprobs', 'usage', 'tool_calling'] as const).map(key =>
        <div className="mas-trace-dl-row" key={key}><dt>{key}</dt><dd>{labels[String(capabilities[key])] || '未记录'}；具体内容见模型返回</dd></div>)}
      <dt>凭据脱敏</dt><dd>{data.redaction_applied || record(view.meta.redaction).applied === true ? '已发生，不承诺严格复现原始输入' : '未报告'}</dd>
    </dl>
    <p className="field-hint">选中 token 的 logprob 不等于完整分布熵；缺失 token / 概率信息不补造，训练适配器需按算法核对。</p>
    <Value value={view.meta.actual_models} label="服务实际返回的模型名称" />
    <h4>快照范围</h4>
    {!view.snapshots.length && <p className="field-hint">没有可读取的快照范围记录，不代表支持从任意位置续跑。</p>}
    {view.snapshots.map((snapshot, index) => <div className="mas-trace-snapshot" key={snapshot.id || index}>
      <code>{snapshot.id || '标识未记录'}</code>
      {snapshot.error && <p className="field-error">{snapshot.error}</p>}
      <div>{Object.entries(coverageLabels).map(([key, label]) => <span key={key}>
        {label}：{snapshot.coverage[key] === true ? '已包含' : snapshot.coverage[key] === false ? '未包含' : '未知'}
      </span>)}</div>
    </div>)}
    <p className="field-hint">消息快照不等于完整 MAS 状态。此阶段只查看，不提供分支或恢复执行。</p>
    <details><summary>评估与学习信息</summary>
      <p className="field-hint">{typeof view.trajectory.final_reward === 'number' ? `记录中的奖励：${view.trajectory.final_reward}，来源需结合采集配置核对。` : '本轨迹未记录奖励，不表示 0 分。'}</p>
      <Value value={view.meta.train_signal} label="已记录的训练信号" />
    </details>
  </details>;
}

export function TrajectoryDetails(props: Props) {
  // A refresh replaces the first page; all lazy detail state must follow that record.
  return <TrajectoryBrowser key={`${props.data.run.run_id}:${props.data.run.finished_at}:${props.data.event_page?.offset ?? 0}`}
    {...props} />;
}

function TrajectoryBrowser({ data, experimentId, onLocate }: Props) {
  const [tab, setTab] = useState<ViewTab>('executions');
  const [rawEvents, setRawEvents] = useState<unknown[]>(() => items(data.trajectory.events));
  const [nextOffset, setNextOffset] = useState<number | null>(data.event_page?.next_offset ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<unknown[]>(() => data.preview ? [] : items(data.trajectory.messages));
  const [messageOffset, setMessageOffset] = useState<number | null>(data.preview ? 0 : null);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [messagesError, setMessagesError] = useState<string | null>(null);
  const eventsRequest = useRef<AbortController | null>(null);
  const messagesRequest = useRef<AbortController | null>(null);
  const eventBase = `/api/mas/rollout-runs/${encodeURIComponent(data.run.run_id)}/trajectory`;
  const view = useMemo(() => adaptTrajectory(data, rawEvents), [data, rawEvents]);
  const query = `experiment_id=${encodeURIComponent(experimentId)}`;
  useEffect(() => {
    eventsRequest.current?.abort(); messagesRequest.current?.abort();
    setRawEvents(items(data.trajectory.events));
    setNextOffset(data.event_page?.next_offset ?? null);
    setMessages(data.preview ? [] : items(data.trajectory.messages));
    setMessageOffset(data.preview ? 0 : null);
    setError(null); setMessagesError(null); setLoading(false); setMessagesLoading(false);
    return () => { eventsRequest.current?.abort(); messagesRequest.current?.abort(); };
  }, [data]);
  const loadEvents = async () => {
    if (nextOffset === null || loading) return;
    const controller = new AbortController(); eventsRequest.current = controller;
    setLoading(true); setError(null);
    try {
      const response = await request<unknown>(`${trajectoryUrl(data.run.run_id, experimentId)}&view=preview&offset=${nextOffset}&limit=100`, { signal: controller.signal });
      const result = readEventPage(response, data.run.run_id, nextOffset);
      if (controller.signal.aborted) return;
      setRawEvents(previous => mergeEvents(previous, result.events)); setNextOffset(result.page.nextOffset);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  };
  const loadMessages = async () => {
    if (messageOffset === null || messagesLoading) return;
    const controller = new AbortController(); messagesRequest.current = controller;
    setMessagesLoading(true); setMessagesError(null);
    try {
      const response = await request<unknown>(`${eventBase}/messages?${query}&offset=${messageOffset}&limit=20`, { signal: controller.signal });
      const result = readMessagePage(response, messageOffset);
      if (controller.signal.aborted) return;
      setMessages(previous => [...previous, ...result.messages]); setMessageOffset(result.page.nextOffset);
    } catch (reason) {
      if (!controller.signal.aborted) setMessagesError(errorMessage(reason));
    } finally {
      if (!controller.signal.aborted) setMessagesLoading(false);
    }
  };
  const renderEvent = (event: EventView) => <EventDetails key={`${data.run.run_id}:${event.key}`} event={event}
    url={`${eventBase}/events/${encodeURIComponent(event.id || '')}?${query}`} onLocate={onLocate} />;
  return <section className="mas-trajectory" aria-label="Rollout 轨迹详情">
    <Capabilities data={data} />
    <div className="mas-trace-tabs" role="tablist" aria-label="轨迹视图">
      {tabs.map(({ id, label }, index) => <button type="button" key={id} id={`trace-tab-${id}`} role="tab"
        aria-selected={tab === id} aria-controls={`trace-panel-${id}`} tabIndex={tab === id ? 0 : -1}
        onClick={() => setTab(id)} onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
            : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
          setTab(tabs[next].id); document.getElementById(`trace-tab-${tabs[next].id}`)?.focus();
        }}>{label}</button>)}
    </div>
    <p className="field-hint">显示已保存记录，不是实时执行。已加载 {view.events.length} / {data.event_page?.total ?? view.events.length} 个事件。
      {nextOffset !== null ? ' 部分配对或执行片段可能在下一页。' : ''}</p>
    <div role="tabpanel" id="trace-panel-executions" aria-labelledby="trace-tab-executions" hidden={tab !== 'executions'}>
      {view.executions.map(group => <details key={`${data.run.run_id}:${group.key}`} className="mas-trace-execution">
        <summary><Bot size={15} /><strong>{group.agentId || '未确认归属的事件'}</strong>
          <span>{group.events.length} 个事件 · {elapsed(group.entry, group.exit)}</span></summary>
        <div className="mas-trace-execution-body">
          <p className="field-hint">执行片段：{group.executionId || '未记录，不能假定属于同一次 Agent 执行'}</p>
          {group.events.map(renderEvent)}
        </div>
      </details>)}
    </div>
    <div role="tabpanel" id="trace-panel-conversation" aria-labelledby="trace-tab-conversation" hidden={tab !== 'conversation'}>
      <p className="field-hint"><MessageSquare size={12} /> 合并对话只表达消息顺序；按 Agent 查看实际上下文，请展开执行片段中的模型请求。</p>
      {messages.map((message, index) => <Message key={index} value={message} index={index} />)}
      {!messages.length && <p className="mas-console-empty">消息按需读取，系统提示词默认折叠。</p>}
      {messagesError && <InlineNotice tone="warning">{messagesError}</InlineNotice>}
      {messageOffset !== null && <Button size="sm" loading={messagesLoading} disabled={messagesLoading} onClick={() => void loadMessages()}>
        {messagesError ? '重试读取消息' : messages.length ? '加载更多消息' : '读取对话消息'}
      </Button>}
    </div>
    <div role="tabpanel" id="trace-panel-tools" aria-labelledby="trace-tab-tools" hidden={tab !== 'tools'}>
      {!view.tools.length && <p className="mas-console-empty">已加载记录中没有工具调用；绑定工具不等于实际调用。</p>}
      {view.tools.map(tool => {
        const event = tool.call || tool.result;
        if (!event) return null;
        const issues = { missing_ids: '关联 ID 不完整，未强行配对。', missing_result: '未找到返回记录（可能尚未加载）。',
          orphan_result: '返回没有匹配的先前调用。', duplicate_result: '同一调用出现重复返回。' };
        return <section key={tool.key} className="mas-trace-tool">
          <header><Wrench size={15} /><strong>{event.toolName || '工具名称未记录'}</strong>
            <span>{elapsed(tool.call, tool.result)}</span><Locate event={event} onLocate={onLocate} /></header>
          {(tool.issue || tool.duplicateCall) && <p className="field-error"><AlertCircle size={12} />
            {tool.duplicateCall ? '调用标识重复，配对存在歧义。' : tool.issue ? issues[tool.issue] : ''}</p>}
          <p className="field-hint">调用 ID：{event.toolCallId || '未记录'} · 模型调用：{event.modelCallId || '未记录'}</p>
          {tool.call && renderEvent(tool.call)}{tool.result && renderEvent(tool.result)}
        </section>;
      })}
    </div>
    <div role="tabpanel" id="trace-panel-events" aria-labelledby="trace-tab-events" hidden={tab !== 'events'}>
      {view.events.map(renderEvent)}
      {!view.events.length && <p className="mas-console-empty">没有可读取的事件记录。</p>}
    </div>
    {error && <InlineNotice tone="warning">{error}</InlineNotice>}
    {nextOffset !== null && tab !== 'conversation' && <Button size="sm" loading={loading} disabled={loading} onClick={() => void loadEvents()}>
      {error ? '重试读取事件' : '加载更多事件'}
    </Button>}
  </section>;
}
