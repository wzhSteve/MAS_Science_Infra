import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, LocateFixed } from 'lucide-react';
import type { RolloutTrajectoryResponse } from '../../../shared/api/types';
import { errorMessage, request } from '../../../shared/api/http';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import {
  adaptTrajectory, elapsed, items, mergeEvents, readEventPage, record, text, trajectoryUrl,
  type ExecutionView, type TrajectoryLocation,
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
    <h3>Agent 执行过程</h3>
    <p className="field-hint">按 Agent 展示任务交接与输出，同一 Agent 多次执行分别列出。系统提示词、记忆、工具参数等详情仅在下载的记录中查看。</p>
    {active && groups.map((group, index) => <AgentCard key={group.key} group={group} index={index} onLocate={onLocate} />)}
    {!groups.length && <p className="mas-console-empty">已加载记录中没有可确认归属的 Agent 片段，请加载后续记录或下载完整 JSON。</p>}
    {nextOffset !== null && <div className="agent-run-more">
      <p className="field-hint">过程尚未全部加载，部分 Agent 的输入输出可能在后续记录中。</p>
      <Button size="sm" loading={loading} onClick={() => void loadMore()}>加载后续过程</Button>
    </div>}
    {error && <InlineNotice tone="warning">读取过程失败：{summary(error)}<Button size="sm" onClick={() => void loadMore()}>重试</Button></InlineNotice>}
  </section>;
}
