import { useRef, useState, type CSSProperties } from 'react';
import { ChevronDown, ChevronUp, CircleCheck, CircleAlert, LoaderCircle, Terminal, Settings2 } from 'lucide-react';
import type { ScienceEvent } from '../../../shared/api/types';
import { useRuntimeEvents } from '../../../app/providers/RuntimeProvider';
import type { useMasDraft } from '../model/useMasDraft';
import { Button } from '../../../shared/ui/button';
import { DataTable } from '../../../shared/components/DataTable';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { RunConfig } from './RunConfig';

export type ConsoleTab = 'config' | 'results' | 'logs';
type Draft = ReturnType<typeof useMasDraft>;
const tabs: Array<{ id: ConsoleTab; label: string }> = [
  { id: 'config', label: '运行配置' }, { id: 'results', label: '结果' }, { id: 'logs', label: '日志' },
];
const clock = (at: number) => new Date(at).toLocaleTimeString('zh-CN', { hour12: false });

function RunResults({ draft }: { draft: Draft }) {
  const result = draft.lastResult;
  if (!result) return <div className="mas-console-empty">
    {draft.running ? '正在执行，结果将在后端返回后显示。' : '运行完成后，结果将在这里展示。'}
  </div>;
  const previous = result.run.id !== draft.lastRun?.id;
  return <div className="mas-run-results">
    <div className="mas-result-summary">
      <strong>{previous ? '上次运行结果' : '本次运行结果'}</strong>
      <span>{result.run.mock ? '模拟执行' : '真实模型'}</span>
      <span>{result.data.n} 条</span>
      <span>平均奖励 <b>{result.data.mean_reward ?? '—'}</b></span>
      <time>{clock(result.run.endedAt || result.run.startedAt)}</time>
    </div>
    {previous && <p className="field-hint">以下数据来自上次完成的运行，不代表本次运行结果。</p>}
    <DataTable aria-label="运行结果">
      <thead><tr><th>任务</th><th>回答</th><th>奖励</th><th>工具调用</th></tr></thead>
      <tbody>{(result.data.rows || []).map((row, index) => <tr key={`${row.id}-${index}`}>
        <td className="mono">{row.id}</td>
        <td><details className="mas-answer"><summary>{String(row.answer ?? '—')}</summary><p>{String(row.answer ?? '—')}</p></details></td>
        <td className="mono">{row.reward ?? '—'}</td><td>{row.tool === undefined ? '—' : row.tool ? '已调用' : '未调用'}</td>
      </tr>)}</tbody>
    </DataTable>
    {result.data.path && <div className="mas-result-path">产物位置 <code>{result.data.path}</code></div>}
  </div>;
}

function eventDescription(event: ScienceEvent) {
  const data = event.data || {};
  if (data.state === 'done') return `采集完成 · ${data.n ?? '—'} 条 · 平均奖励 ${data.mean_reward ?? '—'}`;
  if (typeof data.index === 'number' && typeof data.total === 'number') {
    return `正在处理第 ${data.index + 1} / ${data.total} 题 · ${data.task_id ?? ''}`;
  }
  if (data.state === 'running') return `采集开始 · ${data.n_tasks ?? '—'} 题 · ${data.mock ? '模拟执行' : '真实模型'}`;
  return JSON.stringify(data);
}

function RunLog({ draft }: { draft: Draft }) {
  const events = useRuntimeEvents();
  return <div className="mas-run-log">
    <section>
      <h3>本页操作记录</h3>
      {!draft.logs.length ? <p className="mas-console-empty">尚无操作记录。</p>
        : <ol>{draft.logs.map((entry) => <li key={entry.id} className={`is-${entry.tone}`}>
          <time>{clock(entry.at)}</time><span>{entry.message}</span>
        </li>)}</ol>}
    </section>
    <section>
      <h3>实验采集事件 <span>{events.connected ? '已连接' : '未连接'}</span></h3>
      <p className="field-hint">来自当前实验的后端事件，可能包含其他页面发起的采集，不作为本次运行的完成依据。</p>
      {events.error && <InlineNotice tone="warning">进度暂不可用：{events.error}。运行结果仍以请求返回为准。</InlineNotice>}
      {!events.collectEvents.length ? <p className="mas-console-empty">暂无后端采集事件。</p>
        : <ol>{events.collectEvents.map((event, index) => <li key={`${event.ts}-${index}`}>
          <time>{event.ts ? clock(event.ts * 1000) : '—'}</time><span>{eventDescription(event)}</span>
        </li>)}</ol>}
    </section>
  </div>;
}

export function RunConsole({ draft, open, onOpenChange, tab, onTabChange, executable }: {
  draft: Draft;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tab: ConsoleTab;
  onTabChange: (tab: ConsoleTab) => void;
  executable: { ok: boolean; reason: string };
}) {
  const root = useRef<HTMLElement>(null);
  const drag = useRef<{ y: number; height: number; parentHeight: number } | null>(null);
  const [height, setHeight] = useState(36);
  const clampHeight = (value: number) => Math.min(65, Math.max(22, value));
  const run = draft.lastRun;
  const result = draft.lastResult;
  const summary = !run ? '尚未运行' : run.status === 'saving' ? '正在保存 Workflow…' : run.status === 'running' ? '正在运行…'
    : run.status === 'failed' ? '运行失败' : `运行完成 · ${result?.data.n ?? '—'} 条`;
  const StatusIcon = draft.running ? LoaderCircle : run?.status === 'failed' ? CircleAlert : CircleCheck;

  return <section ref={root} className={`mas-run-console${open ? ' is-open' : ''}`}
    style={{ '--console-height': `${height}%` } as CSSProperties} aria-label="运行控制台">
    {open && <div className="mas-console-resize" role="separator" tabIndex={0} aria-orientation="horizontal"
      aria-label="调整控制台高度" aria-valuenow={Math.round(height)} aria-valuemin={22} aria-valuemax={65}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        const parentHeight = root.current?.parentElement?.clientHeight;
        if (!parentHeight) return;
        drag.current = { y: event.clientY, height, parentHeight };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const start = drag.current;
        if (start) setHeight(clampHeight(start.height + (start.y - event.clientY) / start.parentHeight * 100));
      }}
      onPointerUp={() => { drag.current = null; }}
      onLostPointerCapture={() => { drag.current = null; }}
      onKeyDown={(event) => {
        if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        setHeight(clampHeight(event.key === 'Home' ? 22 : event.key === 'End' ? 65 : height + (event.key === 'ArrowUp' ? 5 : -5)));
      }} />}
    <div className="mas-console-header">
      <button type="button" className="mas-console-toggle" aria-expanded={open} aria-controls="mas-console-content" onClick={() => onOpenChange(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}<Terminal size={14} /><strong>运行控制台</strong>
      </button>
      <span className={`mas-console-summary${run?.status === 'failed' ? ' is-error' : ''}`} role="status">
        {run && <StatusIcon size={13} className={draft.running ? 'animate-spin' : undefined} aria-hidden="true" />}{summary}
      </span>
      <Button size="sm" variant="ghost" aria-label="打开运行配置" title="运行配置"
        onClick={() => { onOpenChange(true); onTabChange('config'); }}><Settings2 size={14} /></Button>
    </div>
    <div id="mas-console-content" className="mas-console-content" hidden={!open}>
      <div className="mas-console-tabs" role="tablist" aria-label="控制台内容">
        {tabs.map(({ id, label }, index) => <button key={id} type="button" role="tab" id={`mas-console-tab-${id}`}
          aria-controls={`mas-console-panel-${id}`} aria-selected={tab === id} tabIndex={tab === id ? 0 : -1}
          onClick={() => onTabChange(id)} onKeyDown={(event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
              : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            onTabChange(tabs[next].id);
            document.getElementById(`mas-console-tab-${tabs[next].id}`)?.focus();
          }}>
          {label}{id === 'results' && result && <span className="mas-tab-count">{result.data.n}</span>}
        </button>)}
        {run && <span className="mas-run-version">{run.mock ? '模拟' : '真实'} · {run.input === 'parquet' ? '数据集' : '示例'}
          {run.workflow !== draft.workflow ? ' · 草稿有新编辑' : ''}</span>}
      </div>
      {run?.status === 'failed' && <div className="mas-console-error"><InlineNotice tone="danger">{run.error}</InlineNotice></div>}
      {draft.running && <p className="mas-run-hint">本次使用启动时保存的 Workflow，新编辑不会自动加入本次运行。</p>}
      <div className="mas-console-body" role="tabpanel" id="mas-console-panel-config" aria-labelledby="mas-console-tab-config" hidden={tab !== 'config'}>
        <RunConfig draft={draft} executable={executable} />
      </div>
      <div className="mas-console-body" role="tabpanel" id="mas-console-panel-results" aria-labelledby="mas-console-tab-results" hidden={tab !== 'results'}>
        <RunResults draft={draft} />
      </div>
      <div className="mas-console-body" role="tabpanel" id="mas-console-panel-logs" aria-labelledby="mas-console-tab-logs" hidden={tab !== 'logs'}>
        <RunLog draft={draft} />
      </div>
    </div>
  </section>;
}
