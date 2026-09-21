import type { RolloutTrajectoryResponse } from '../../../shared/api/types';

export type TrajectoryLocation = { agentId: string; agentExecutionId?: string; toolName?: string };
export type RecordValue = Record<string, unknown>;

export function record(value: unknown): RecordValue {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as RecordValue : {};
}

export function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

export function items(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function snippet(value: unknown, limit = 180): string {
  const content = text(value);
  if (!content) return value == null ? '未记录' : '结构化内容，展开查看';
  return content.length > limit ? `${content.slice(0, limit)}…` : content;
}

export interface EventView {
  key: string;
  id?: string;
  order: number;
  kind: string;
  title: string;
  raw: RecordValue;
  payload: RecordValue;
  agentId?: string;
  recordedAgentId?: string;
  executionId?: string;
  modelCallId?: string;
  toolCallId?: string;
  toolName?: string;
  occurredAt?: string;
  recordedAt?: string;
  preview: boolean;
  previewTruncated?: boolean;
  status: string;
}

export interface ExecutionView {
  key: string;
  executionId?: string;
  agentId?: string;
  entry?: EventView;
  exit?: EventView;
  events: EventView[];
}

export interface ToolView {
  key: string;
  call?: EventView;
  result?: EventView;
  issue?: 'missing_ids' | 'missing_result' | 'orphan_result' | 'duplicate_result';
  duplicateCall: boolean;
}

export interface PageView {
  offset: number;
  total: number;
  nextOffset: number | null;
}

const titles: Record<string, string> = {
  task_start: '任务开始', agent_enter: '进入 Agent', agent_exit: '退出 Agent',
  handoff: 'Agent 交接', model_call: '模型请求', model_result: '模型返回',
  skill_call: 'Skill 调用', skill_result: 'Skill 返回', termination: '运行终止',
  agent_message: 'Agent 消息', tool_call: '工具调用', tool_result: '工具返回',
  final_answer: '最终答案事件', snapshot: '快照记录', error: '错误',
  memory_read: '读取记忆', memory_write: '写入记忆', feedback: '反馈',
};

export function eventStatus(kind: string, payload: RecordValue): string {
  if (payload.recoverable === true && (payload.error || kind === 'error')) return '可恢复的局部错误';
  if (payload.status === 'failed' || payload.ok === false || kind === 'error') return '失败记录';
  if (payload.status === 'succeeded' || payload.ok === true) return '成功记录';
  return '';
}

export function readPage(value: unknown): PageView | undefined {
  const page = record(value);
  const integer = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 0;
  if (!integer(page.offset) || !integer(page.total)
    || !(page.next_offset === null || integer(page.next_offset))) return undefined;
  return { offset: page.offset, total: page.total, nextOffset: page.next_offset };
}

export function readEvent(value: unknown, order: number, preview: boolean): EventView {
  const raw = record(value);
  const payload = record(raw.payload);
  const kind = text(raw.kind) ?? 'unknown';
  const id = text(raw.event_id);
  return {
    key: id ?? `unidentified-event-${order}`, id, order, kind, raw, payload,
    title: titles[kind] ?? `未知事件：${kind}`,
    recordedAgentId: text(raw.agent_id), executionId: text(raw.agent_execution_id),
    modelCallId: text(raw.model_call_id), toolCallId: text(raw.tool_call_id),
    toolName: kind === 'tool_call' || kind === 'tool_result' ? text(payload.name) : undefined,
    occurredAt: text(raw.occurred_at), recordedAt: text(raw.recorded_at) ?? text(raw.ts),
    preview, previewTruncated: typeof raw.preview_truncated === 'boolean' ? raw.preview_truncated : undefined,
    status: eventStatus(kind, payload),
  };
}

export function mergeEvents(previous: unknown[], incoming: unknown[]): unknown[] {
  const ids = new Set(previous.map(value => text(record(value).event_id)).filter(Boolean));
  return [...previous, ...incoming.filter(value => {
    const id = text(record(value).event_id);
    if (!id) return true;
    if (ids.has(id)) return false;
    ids.add(id);
    return true;
  })];
}

export function elapsed(start?: EventView, end?: EventView): string {
  if (!start?.occurredAt || !end?.occurredAt) return '耗时未知';
  const from = Date.parse(start.occurredAt);
  const to = Date.parse(end.occurredAt);
  if (!Number.isFinite(from) || !Number.isFinite(to) || to < from) return '耗时未知';
  const milliseconds = to - from;
  return milliseconds < 1000 ? `${milliseconds} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
}

export function eventLocation(event: EventView): TrajectoryLocation | undefined {
  if (!event.agentId) return undefined;
  return { agentId: event.agentId, agentExecutionId: event.executionId, toolName: event.toolName };
}

export function agentLabel(event: EventView): string {
  return event.agentId ?? (event.recordedAgentId ? `归属未核实（记录为 ${event.recordedAgentId}）` : 'Agent 未标注');
}

export function pairTools(events: EventView[]): ToolView[] {
  const output: ToolView[] = [];
  const buckets = new Map<string, ToolView[]>();
  const callCounts = new Map<string, number>();
  const correlation = (event: EventView) => JSON.stringify([event.executionId, event.modelCallId, event.toolCallId]);
  for (const event of events) {
    if (event.kind === 'tool_call' && event.executionId && event.modelCallId && event.toolCallId) {
      const key = correlation(event);
      callCounts.set(key, (callCounts.get(key) || 0) + 1);
    }
  }
  for (const event of events) {
    if (event.kind !== 'tool_call' && event.kind !== 'tool_result') continue;
    const isCall = event.kind === 'tool_call';
    const row: ToolView = {
      key: event.key, ...(isCall ? { call: event } : { result: event }), duplicateCall: false,
    };
    if (!event.executionId || !event.modelCallId || !event.toolCallId) {
      output.push({ ...row, issue: 'missing_ids' });
      continue;
    }
    const key = correlation(event);
    if ((callCounts.get(key) || 0) > 1) {
      output.push({ ...row, duplicateCall: true });
      continue;
    }
    const calls = buckets.get(key) ?? [];
    if (isCall) {
      row.issue = 'missing_result';
      if (calls.length > 0) {
        row.duplicateCall = true;
        calls.forEach(call => { call.duplicateCall = true; });
      }
      calls.push(row);
      buckets.set(key, calls);
      output.push(row);
    } else {
      // Only earlier calls in this exact correlation bucket are eligible.
      const pending = calls.length === 1 && !calls[0].duplicateCall ? calls[0] : undefined;
      if (pending && !pending.result) {
        pending.result = event;
        pending.issue = undefined;
      } else {
        output.push({ ...row, issue: calls.length ? 'duplicate_result' : 'orphan_result' });
      }
    }
  }
  return output;
}

export function adaptTrajectory(data: RolloutTrajectoryResponse, rawEvents = items(data.trajectory.events)) {
  const trajectory = record(data.trajectory);
  const meta = record(trajectory.meta);
  const eventCounts = new Map<string, number>();
  const events = rawEvents.map((value, order) => {
    const event = readEvent(value, order, data.preview === true);
    const count = eventCounts.get(event.key) || 0;
    eventCounts.set(event.key, count + 1);
    if (count) event.key = `${event.key}-duplicate-${count}`;
    return event;
  });
  const entries = new Map<string, { agentId: string; conflicting: boolean }>();
  for (const event of events) {
    if (event.kind !== 'agent_enter' || !event.executionId) continue;
    if ((event.raw.run_id && event.raw.run_id !== data.run.run_id)
      || (event.raw.trajectory_id && event.raw.trajectory_id !== trajectory.trajectory_id)) continue;
    const context = record(event.payload.context);
    const agentId = text(context.agent_id);
    if (!agentId || context.agent_execution_id !== event.executionId
      || (event.recordedAgentId && event.recordedAgentId !== agentId)) continue;
    const previous = entries.get(event.executionId);
    entries.set(event.executionId, { agentId, conflicting: !!previous && (previous.conflicting || previous.agentId !== agentId) });
  }
  const executions = new Map<string, ExecutionView>();
  for (const event of events) {
    const entry = event.executionId ? entries.get(event.executionId) : undefined;
    const wrongRun = (event.raw.run_id && event.raw.run_id !== data.run.run_id)
      || (event.raw.trajectory_id && event.raw.trajectory_id !== trajectory.trajectory_id);
    const correlated = String(trajectory.schema_version) === '2'
      && event.executionId && event.raw.run_id === data.run.run_id
      && event.raw.trajectory_id === trajectory.trajectory_id;
    if (!wrongRun && entry && !entry.conflicting && (!event.recordedAgentId || event.recordedAgentId === entry.agentId)) {
      event.agentId = entry.agentId;
    } else if (!entry && correlated) {
      event.agentId = event.recordedAgentId;
    }
    const key = event.executionId ?? 'uncorrelated';
    let group = executions.get(key);
    if (!group) {
      group = { key, executionId: event.executionId, agentId: event.agentId, events: [] };
      executions.set(key, group);
    }
    group.events.push(event);
    if (event.kind === 'agent_enter' && !group.entry) group.entry = event;
    if (event.kind === 'agent_exit') group.exit = event;
    if (group.agentId !== event.agentId) group.agentId = undefined;
  }
  return {
    trajectory, meta, events, executions: [...executions.values()], tools: pairTools(events),
    messages: items(trajectory.messages), eventPage: readPage(record(data).event_page),
    messagePage: readPage(record(data).message_page), modelIdentity: record(meta.model_identity),
    snapshots: items(record(data).snapshots).map(value => {
      const snapshot = record(value);
      return { id: text(snapshot.snapshot_id), coverage: record(snapshot.coverage), error: text(snapshot.error) };
    }),
    presence: ['usage', 'logprobs', 'token_ids'].map(field => ({
      field, present: events.some(event => event.kind === 'model_result' && event.payload[field] != null),
    })),
  };
}

export function trajectoryUrl(runId: string, experimentId: string, suffix = ''): string {
  return `/api/mas/rollout-runs/${encodeURIComponent(runId)}/trajectory${suffix}?experiment_id=${encodeURIComponent(experimentId)}`;
}

export function readEventPage(value: unknown, runId: string, offset: number) {
  const response = record(value);
  const trajectory = record(response.trajectory);
  const page = readPage(response.event_page);
  if (record(response.run).run_id !== runId || !Array.isArray(trajectory.events)
    || !page || page.offset !== offset || (page.nextOffset !== null && page.nextOffset <= offset)) {
    throw new Error('事件分页响应格式不符合记录契约');
  }
  return { events: trajectory.events as unknown[], page };
}

export function readMessagePage(value: unknown, offset: number) {
  const response = record(value);
  const page = readPage(response);
  if (!Array.isArray(response.messages) || !page || page.offset !== offset
    || (page.nextOffset !== null && page.nextOffset <= offset)) {
    throw new Error('消息分页响应格式不符合记录契约');
  }
  return { messages: response.messages as unknown[], page };
}

export function readFullEvent(value: unknown, eventId: string): RecordValue {
  const event = record(record(value).event);
  if (event.event_id !== eventId || !text(event.kind)) throw new Error('事件详情响应与请求不一致');
  return event;
}
