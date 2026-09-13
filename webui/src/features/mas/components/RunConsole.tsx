import { memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { ChevronDown, ChevronUp, CircleCheck, CircleAlert, LoaderCircle, Terminal, Settings2 } from 'lucide-react';
import type { useMasDraft } from '../model/useMasDraft';
import { Button } from '../../../shared/ui/button';
import { DataTable } from '../../../shared/components/DataTable';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { RunConfig } from './RunConfig';
import { ModelReadiness } from './ModelReadiness';
import type { ModelReadinessState } from '../model/useModelReadiness';
import { rolloutStatusLabel, useMasRolloutRun } from '../model/useMasRolloutRun';
import { RolloutConfig, RolloutResults } from './RolloutRun';
import type { TrajectoryLocation } from '../model/trajectoryView';
import { RunErrorDetails } from './RunErrorDetails';
import { DownloadText } from '../../../shared/components/DownloadText';

export type ConsoleTab = 'config' | 'results';
type Draft = ReturnType<typeof useMasDraft>;
const tabs: Array<{ id: ConsoleTab; label: string }> = [
  { id: 'config', label: '任务输入' }, { id: 'results', label: '答案与过程' },
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
      <strong>{previous ? '上次数据集采集结果' : '本次数据集采集结果'}</strong>
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

function RetainedContent({ active, children }: { active: boolean; children: ReactNode }) {
  const [visited, setVisited] = useState(active);
  useEffect(() => { if (active) setVisited(true); }, [active]);
  return <div hidden={!active}>{(active || visited) && children}</div>;
}

export const RunConsole = memo(function RunConsole({ draft, experimentId, active, open, onOpenChange, tab, onTabChange, executable, readiness, onConfigureModel, onLocate, mode, onModeChange, historyRequest }: {
  draft: Draft;
  experimentId: string;
  active: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tab: ConsoleTab;
  onTabChange: (tab: ConsoleTab) => void;
  executable: { ok: boolean; reason: string };
  readiness: ModelReadinessState;
  onConfigureModel: () => void;
  onLocate: (target: TrajectoryLocation) => void;
  mode: 'rollout' | 'collect' | 'demo';
  onModeChange: (mode: 'rollout' | 'collect' | 'demo') => void;
  historyRequest?: number;
}) {
  const root = useRef<HTMLElement>(null);
  const drag = useRef<{ y: number; height: number; parentHeight: number } | null>(null);
  const [height, setHeight] = useState(36);
  const live = useMasRolloutRun(experimentId, draft.executeWithSavedWorkflow, 'live');
  const demo = useMasRolloutRun(experimentId, draft.executeWithSavedWorkflow, 'mock');
  const rollout = mode === 'demo' ? demo : live;
  const [currentRequest, setCurrentRequest] = useState(0);
  const showResults = useCallback(() => {
    setCurrentRequest(value => value + 1);
    onTabChange('results');
  }, [onTabChange]);
  const showInput = useCallback(() => onTabChange('config'), [onTabChange]);
  const single = mode !== 'collect';
  const logs = single ? rollout.logs : draft.logs;
  const logText = useMemo(() => logs.map(entry => `${new Date(entry.at).toISOString()} [${entry.tone}] ${entry.message}`).join('\n'), [logs]);
  const visible = active && open;
  const label = mode === 'demo' ? '模拟演示 · Mock' : mode === 'collect' ? '数据集采集' : '真实调试';
  const clampHeight = (value: number) => Math.min(65, Math.max(22, value));
  const run = draft.lastRun;
  const result = draft.lastResult;
  const collectSummary = !run ? '尚未采集' : run.status === 'saving' ? '正在保存 Workflow…' : run.status === 'running' ? '正在采集…'
    : run.status === 'failed' ? '运行失败' : `运行完成 · ${result?.data.n ?? '—'} 条`;
  const summary = single ? rollout.attempt ? rolloutStatusLabel[rollout.attempt.status] : '尚未运行' : collectSummary;
  const running = single ? rollout.running : draft.running;
  const failed = single ? ['failed', 'interrupted', 'unknown'].includes(rollout.attempt?.status || '') : run?.status === 'failed';
  const hasRun = single ? Boolean(rollout.attempt) : Boolean(run);
  const StatusIcon = running ? LoaderCircle : failed ? CircleAlert : CircleCheck;

  return <section ref={root} className={`mas-run-console${open ? ' is-open' : ''}`}
    style={{ '--console-height': `${height}%` } as CSSProperties} aria-label={label}>
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
        {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}<Terminal size={14} /><strong>{label}</strong>
      </button>
      <span className={`mas-console-summary${failed ? ' is-error' : ''}`} role="status">
        {hasRun && <StatusIcon size={13} className={running ? 'animate-spin' : undefined} aria-hidden="true" />}
        <span className="mas-console-summary-text">{summary}
          {single && draft.running ? ' · 数据集采集中' : mode !== 'rollout' && live.running ? ' · 真实调试执行中' : mode !== 'demo' && demo.running ? ' · 模拟演示执行中' : ''}
        </span>
      </span>
      <Button size="sm" variant="ghost" aria-label="打开运行配置" title="运行配置"
        onClick={() => { onOpenChange(true); onTabChange('config'); }}><Settings2 size={14} /></Button>
    </div>
    <div id="mas-console-content" className="mas-console-content" hidden={!open}>
      {mode !== 'rollout' && <div className="debug-mode-banner">
        <span>{mode === 'demo' ? '独立演示入口，不产生真实模型结果。' : '独立采集入口；参数、奖励与单题调试分开。'}</span>
        <Button size="sm" variant="ghost" onClick={() => onModeChange('rollout')}>返回真实调试</Button>
      </div>}
      {single && rollout.attempt && <div className="debug-current-attempt" role="status">
        <strong>本次{rollout.execution === 'live' ? '真实调试' : '模拟演示'}：{rolloutStatusLabel[rollout.attempt.status]}</strong>
        <time>{new Date(rollout.attempt.startedAt).toLocaleTimeString()}</time>
        <code>{rollout.attempt.summary?.run_id || (rollout.running ? '等待服务端返回 Run ID' : '未取得 Run ID')}</code>
        <Button size="sm" variant="ghost" onClick={showResults}>查看本次结果</Button>
      </div>}
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
          {label}{id === 'results' && !single && result && <span className="mas-tab-count">{result.data.n}</span>}
        </button>)}
        {single && rollout.attempt && <span className="mas-run-version">
          {rollout.attempt.execution === 'mock' ? 'Mock' : 'Live'} · 已捕获 Workflow{rollout.attempt.workflow !== draft.workflow ? ' · 草稿有新编辑' : ''}
        </span>}
        {!single && run && <span className="mas-run-version">{run.mock ? '模拟' : '真实'} · {run.input === 'parquet' ? '数据集' : '示例'}
          {run.workflow !== draft.workflow ? ' · 草稿有新编辑' : ''}</span>}
      </div>
      {!single && run?.status === 'failed' && <div className="mas-console-error"><RunErrorDetails message={run.error || '未返回错误详情'} label="数据集采集失败" /></div>}
      {running && <p className="mas-run-hint">本次使用启动时保存的 Workflow，新编辑不会自动加入本次运行。</p>}
      <div className="mas-console-body" role="tabpanel" id="mas-console-panel-config" aria-labelledby="mas-console-tab-config" hidden={tab !== 'config'}>
        {visible && tab === 'config' && mode === 'rollout' && <ModelReadiness readiness={readiness} onConfigureModel={onConfigureModel} />}
        {visible && tab === 'config' && !single && <Button size="sm" variant="ghost" onClick={onConfigureModel}>配置采集模型</Button>}
        {visible && tab === 'config' && single && <RolloutConfig rollout={rollout} pending={Boolean(draft.pending)} executable={executable}
          readiness={readiness} onShowResults={showResults} />}
        <RetainedContent active={visible && tab === 'config' && !single}>
          <RunConfig draft={draft} executable={executable} readiness={readiness} />
        </RetainedContent>
      </div>
      <div className="mas-console-body" role="tabpanel" id="mas-console-panel-results" aria-labelledby="mas-console-tab-results" hidden={tab !== 'results'}>
        <RetainedContent active={visible && tab === 'results' && mode === 'rollout'}>
          <RolloutResults rollout={live} workflowChanged={Boolean(live.attempt && live.attempt.workflow !== draft.workflow)}
            workflow={draft.workflow} historyRequest={historyRequest} currentRequest={currentRequest} active={visible && tab === 'results' && mode === 'rollout'}
            onLocate={onLocate} onUseQuestion={showInput} />
        </RetainedContent>
        <RetainedContent active={visible && tab === 'results' && mode === 'demo'}>
          <RolloutResults rollout={demo} workflowChanged={Boolean(demo.attempt && demo.attempt.workflow !== draft.workflow)}
            workflow={draft.workflow} currentRequest={currentRequest} active={visible && tab === 'results' && mode === 'demo'}
            onLocate={onLocate} onUseQuestion={showInput} />
        </RetainedContent>
        {visible && tab === 'results' && !single && <RunResults draft={draft} />}
        {visible && tab === 'results' && logs.length > 0 && <div className="debug-log-download">
          <DownloadText text={logText} filename={`${single ? rollout.attempt?.summary?.run_id || mode : 'collect'}-operations.log`}
            label="下载本页操作日志" />
        </div>}
      </div>
    </div>
  </section>;
});
