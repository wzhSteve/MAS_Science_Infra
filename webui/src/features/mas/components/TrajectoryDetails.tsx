import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, LocateFixed } from 'lucide-react';
import type { RolloutTrajectoryResponse } from '../../../shared/api/types';
import { errorMessage, request } from '../../../shared/api/http';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import {
  adaptTrajectory, elapsed, items, mergeEvents, readEventPage, record, text, trajectoryUrl,
  type EventView, type ExecutionView, type RecordValue, type TrajectoryLocation,
} from '../model/trajectoryView';

type Props = {
  data: RolloutTrajectoryResponse; experimentId: string;
  onLocate: (target: TrajectoryLocation) => void; active?: boolean;
};

function contentText(value: unknown): string {
  if (typeof value === 'string') return value;
  return items(value).map(block => text(record(block).text) || '').filter(Boolean).join('\n');
}

function summary(value: string) {
  return value.length > 800 ? `${value.slice(0, 800)}…` : value;
}

// --- user_space window trace (Fix E/F) --------------------------------------

interface LlmCallView { model?: string; n?: number; promptHead: string; responseHead: string }

export interface WindowTraceView {
  llmCount: number;
  calls: LlmCallView[];
  bts?: { n?: number; selected?: string; plannerSelected?: string; scores?: Record<string, unknown> };
  fields: { tool?: string; subGoal?: string; command?: string; conclusion?: string };
}

/** Read payload.trace defensively: llm_calls / bts / detail fields may be absent in older records. */
function readWindowTrace(payload: unknown): WindowTraceView | undefined {
  const trace = record(record(payload).trace);
  if (!Object.keys(trace).length) return undefined;
  const llm = record(trace.llm_calls);
  const calls = items(llm.calls).map(call => {
    const row = record(call);
    return {
      model: text(row.model),
      n: typeof row.n === 'number' ? row.n : undefined,
      promptHead: text(row.prompt_head) ?? '',
      responseHead: text(row.response_head) ?? '',
    };
  });
  const count = typeof llm.count === 'number' ? llm.count : calls.length;
  const btsRaw = record(trace.bts);
  const bts = Object.keys(btsRaw).length
    ? { n: typeof btsRaw.n === 'number' ? btsRaw.n : undefined, selected: text(btsRaw.selected), plannerSelected: text(btsRaw.planner_selected), scores: record(btsRaw.scores) as Record<string, unknown> }
    : undefined;
  const fields = {
    tool: text(trace.tool) ?? text(trace.tool_name),
    subGoal: text(trace.sub_goal),
    command: text(trace.command),
    conclusion: text(trace.conclusion),
  };
  if (!count && !calls.length && !bts && !Object.values(fields).some(Boolean)) return undefined;
  return { llmCount: count, calls, bts, fields };
}

function scoreEntries(scores: Record<string, unknown>): Array<[string, number]> {
  const out: Array<[string, number]> = [];
  for (const entry of Object.entries(scores)) {
    if (typeof entry[1] === 'number') out.push([entry[0], entry[1]]);
  }
  return out;
}

function btsSummaryLine(trace: WindowTraceView): string | undefined {
  const bts = trace.bts;
  if (!bts) return undefined;
  const parts: string[] = [];
  if (bts.n !== undefined) parts.push(`${bts.n} 候选`);
  const scores = scoreEntries(bts.scores ?? {});
  if (scores.length) parts.push(`BTS ${scores.map(([key, value]) => `${key}:${value}`).join(' ')}`);
  if (bts.plannerSelected !== undefined && bts.selected !== undefined && bts.plannerSelected !== bts.selected) {
    parts.push(`planner 选 ${bts.plannerSelected} ≠ BTS 选 ${bts.selected}`);
  } else if (bts.selected !== undefined) {
    parts.push(`选中 ${bts.selected}`);
  }
  return parts.length ? parts.join(' · ') : undefined;
}

function WindowTraceBlock({ trace }: { trace: WindowTraceView }) {
  const btsLine = btsSummaryLine(trace);
  return <div className="agent-run-llm">
    <p className="agent-run-llm-summary">
      {trace.llmCount > 0 && <span className="status-badge status-badge--info">LLM {trace.llmCount} 次</span>}
      {btsLine && <span className="field-hint">BTS 评分：{btsLine}</span>}
      {trace.llmCount > 0 && trace.calls.length === 0 && <span className="field-hint">提示词明细未记录</span>}
    </p>
    {trace.calls.length > 0 && <details className="agent-run-llm-calls">
      <summary>{trace.calls.length} 条模型调用记录</summary>
      {trace.calls.map((call, index) => <div className="agent-run-llm-call" key={index}>
        <p className="agent-run-llm-call-title">#{index + 1}{call.model ? ` · ${call.model}` : ''}{call.n && call.n > 1 ? ` · n=${call.n}` : ''}</p>
        {call.promptHead && <details>
          <summary>提示词</summary>
          <pre>{call.promptHead}</pre>
        </details>}
        {call.responseHead && <details>
          <summary>返回</summary>
          <pre>{call.responseHead}</pre>
        </details>}
      </div>)}
    </details>}
  </div>;
}

const WINDOW_KIND_LABELS: Record<string, string> = {
  plan_step: '规划', tool_result: '执行', verify: '校验', error: '错误',
};

/** One user_space window: agent_message(phase=window_close) joined with its output message. */
interface UserWindowView {
  key: string;
  eventId: string;
  agentId: string;
  turn: number;
  ok: boolean;
  kind: string;
  output?: string;
  readyToStop?: boolean;
  reason?: string;
  trace?: WindowTraceView;
}

function readUserWindows(events: EventView[], rawMessages: unknown[]): UserWindowView[] {
  const byMsgId = new Map<string, RecordValue>();
  for (const raw of rawMessages) {
    const msg = record(raw);
    const msgId = text(msg.msg_id);
    if (msgId) byMsgId.set(msgId, msg);
  }
  const windows: UserWindowView[] = [];
  for (const event of events) {
    if (event.kind !== 'agent_message') continue;
    const payload = record(event.payload);
    if (payload.phase !== 'window_close') continue;
    const outId = text(payload.output_msg_id);
    const msg = outId ? byMsgId.get(outId) : undefined;
    const msgPayload = record(msg?.payload);
    const kind = text(msg?.kind) || text(payload.agent_id) || 'window';
    const output = text(msgPayload.output);
    windows.push({
      key: `${event.key}:${text(payload.agent_id) || 'window'}:${payload.turn ?? 0}`,
      eventId: event.key,
      agentId: text(payload.agent_id) || text(msg?.src) || 'agent',
      turn: typeof payload.turn === 'number' ? payload.turn : 0,
      ok: payload.ok !== false && msgPayload.ok !== false,
      kind,
      output,
      readyToStop: msgPayload.ready_to_stop === true ? true : undefined,
      reason: text(msgPayload.reason) ?? text(msgPayload.error),
      trace: readWindowTrace(msgPayload) ?? readWindowTrace(payload),
    });
  }
  return windows.sort((a, b) => (a.turn - b.turn) || a.eventId.localeCompare(b.eventId));
}

function UserWindowCard({ window }: { window: UserWindowView }) {
  const trace = window.trace;
  const details: string[] = [];
  if (trace?.fields.tool) details.push(`工具 ${trace.fields.tool}`);
  if (trace?.fields.subGoal) details.push(`目标 ${trace.fields.subGoal}`);
  if (trace?.fields.command) details.push(`命令 ${trace.fields.command}`);
  if (window.reason) details.push(`说明 ${window.reason}`);
  const state = window.kind === 'verify'
    ? window.readyToStop ? '可以停止' : '继续'
    : window.ok ? '成功' : '失败';
  return <article className="agent-run-card agent-run-card--user-window">
    <header>
      <span className="agent-run-icon"><Bot size={16} /></span>
      <strong>{window.agentId}</strong>
      <span className="field-hint">第{window.turn}轮 · {WINDOW_KIND_LABELS[window.kind] ?? '窗口'}</span>
      <span className={`agent-run-state${window.ok ? '' : ' is-failed'}`}>{state}</span>
    </header>
    <div className="agent-run-io">
      {window.output && <section><h4>窗口输出</h4><p>{summary(window.output)}</p></section>}
      {details.length > 0 && <section><h4>窗口摘要</h4><p>{details.join('\n')}</p></section>}
    </div>
    {trace && <WindowTraceBlock trace={trace} />}
  </article>;
}

function UserWindowTimeline({ windows }: { windows: UserWindowView[] }) {
  return <div className="agent-run-user-windows" aria-label="用户封装窗口">
    <h3>用户封装执行窗口</h3>
    <p className="field-hint">每次窗口收尾记录：工具链、BTS 评分、LLM 调用次数与提示词摘要。</p>
    <ol className="agent-run-timeline">
      {windows.map(window => <li key={window.key}><UserWindowCard window={window} /></li>)}
    </ol>
  </div>;
}

const AgentCard = memo(function AgentCard({ group, index, onLocate }: {
  group: ExecutionView; index: number; onLocate: Props['onLocate'];
}) {
  const requestEvent = group.events.find(event => event.kind === 'model_call');
  const userMessages = items(requestEvent?.payload.messages)
    .filter(message => record(message).role === 'user')
    .map(message => contentText(record(message).content)).filter(Boolean);
  const input = userMessages.filter((message, position) => position === 0 || message.startsWith('[Upstream Agent output]')).join('\n\n');
  const agentId = group.agentId;
  const results = group.events.filter(event => event.kind === 'model_result');
  const lastResult = results[results.length - 1];
  const answer = text(group.exit?.payload.final_answer);
  const output = answer || contentText(record(lastResult?.payload.message).content);
  const tools = group.events.filter(event => event.kind === 'tool_call');
  const toolNames = [...new Set(tools.map(event => event.toolName).filter(Boolean))];
  const failed = group.exit?.payload.status === 'failed';
  const completed = group.exit?.payload.status === 'succeeded';
  const preview = group.events.some(event => event.previewTruncated);
  return <article className="agent-run-card">
    <header>
      <span className="agent-run-icon"><Bot size={16} /></span>
      <strong>{group.agentId}</strong>
      <span className="field-hint">执行 {index + 1} · {elapsed(group.entry, group.exit)}</span>
      <span className={`agent-run-state${failed ? ' is-failed' : ''}`}>{failed ? '失败' : completed ? '完成' : '记录未完整加载'}</span>
      {agentId && <Button size="sm" variant="ghost" onClick={() => onLocate({
        agentId, agentExecutionId: group.executionId,
      })}><LocateFixed size={13} />定位</Button>}
    </header>
    <div className="agent-run-io">
      <section><h4>输入 · 任务与交接摘要</h4><p>{input ? summary(input)
        : group.entry?.payload.execution_kind === 'skill_only' ? '本节点仅执行验证技能，没有模型输入；详细输入见下载记录。'
          : '尚无可展示的模型输入，可能位于后续记录中。'}</p></section>
      <section><h4>{answer ? '输出摘要' : '最近输出摘要'}</h4><p>{output ? summary(output)
        : failed ? '本节点未返回有效输出，失败详情见下载记录。' : '尚未记录输出或尚未加载到结束记录。'}</p></section>
    </div>
    {tools.length > 0 && <p className="agent-run-tools">已加载的工具调用：{tools.length} 次{toolNames.length ? ` · ${toolNames.join('、')}` : ''}</p>}
    {(preview || input.length > 800 || output.length > 800) && <p className="field-hint agent-run-preview">内容为摘要，完整上下文和原始输出请下载 JSON 查看。</p>}
  </article>;
});

export function TrajectoryDetails(props: Props) {
  return <AgentOverview key={`${props.data.run.run_id}:${props.data.run.finished_at}`} {...props} />;
}

function AgentOverview({ data, experimentId, onLocate, active = true }: Props) {
  const [events, setEvents] = useState<unknown[]>(() => items(data.trajectory.events));
  const [nextOffset, setNextOffset] = useState<number | null>(data.event_page?.next_offset ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    pending.current?.abort();
    pending.current = null;
    setEvents(items(data.trajectory.events));
    setNextOffset(data.event_page?.next_offset ?? null);
    setLoading(false); setError(null);
    return () => pending.current?.abort();
  }, [data]);
  const view = useMemo(() => adaptTrajectory(data, events), [data, events]);
  const groups = useMemo(() => view.executions.filter(group => group.agentId && group.executionId), [view]);
  const userWindows = useMemo(
    () => readUserWindows(view.events, items(view.messages)),
    [view.events, view.messages],
  );
  const loadMore = async () => {
    if (nextOffset === null || pending.current) return;
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true); setError(null);
    try {
      const result = await request<unknown>(`${trajectoryUrl(data.run.run_id, experimentId)}&view=preview&offset=${nextOffset}&limit=100`, { signal: controller.signal });
      if (controller.signal.aborted) return;
      const page = readEventPage(result, data.run.run_id, nextOffset);
      setEvents(previous => mergeEvents(previous, page.events));
      setNextOffset(page.page.nextOffset);
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (pending.current === controller) pending.current = null;
      if (!controller.signal.aborted) setLoading(false);
    }
  };
  return <section className="agent-run-overview" aria-label="Agent 输入输出" hidden={!active}>
    {active && userWindows.length > 0 && <UserWindowTimeline windows={userWindows} />}
    {active && groups.length > 0 && <ol className="agent-run-timeline">
      {groups.map((group, index) => <li key={group.key}><AgentCard group={group} index={index} onLocate={onLocate} /></li>)}
    </ol>}
    {!groups.length && !userWindows.length && <p className="mas-console-empty">已加载记录中没有可确认归属的 Agent 片段，请加载后续记录或下载完整 JSON。</p>}
    {nextOffset !== null && <div className="agent-run-more">
      <p className="field-hint">过程尚未全部加载，部分 Agent 的输入输出可能在后续记录中。</p>
      <Button size="sm" loading={loading} onClick={() => void loadMore()}>加载后续过程</Button>
    </div>}
    {error && <InlineNotice tone="warning">读取过程失败：{summary(error)}<Button size="sm" onClick={() => void loadMore()}>重试</Button></InlineNotice>}
  </section>;
}
