import { memo, useMemo, useEffect, lazy, Suspense } from 'react';
import { Play, FileJson, Clock3 } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import { Textarea } from '../../../shared/ui/textarea';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { rolloutStatusLabel, type MasRolloutRun } from '../model/useMasRolloutRun';
import type { ModelReadinessState } from '../model/useModelReadiness';
import { useRolloutHistory } from '../model/useRolloutHistory';
import { RolloutHistory } from './RolloutHistory';
import { RolloutExport } from './RolloutExport';
import { RunErrorDetails } from './RunErrorDetails';
import type { TrajectoryLocation } from '../model/trajectoryView';
import type { WorkflowSpec } from '../../../shared/api/types';
const TrajectoryDetails = lazy(() => import('./TrajectoryDetails').then(module => ({ default: module.TrajectoryDetails })));

export function RolloutConfig({ rollout, pending, executable, readiness, onShowResults }: {
  rollout: MasRolloutRun;
  pending: boolean;
  executable: { ok: boolean; reason: string };
  readiness: ModelReadinessState;
  onShowResults: () => void;
}) {
  const liveError = rollout.execution === 'mock' ? null
    : readiness.loading ? '正在检查模型配置与依赖，请稍候。'
      : readiness.error ? `就绪检查失败：${readiness.error}`
        : !readiness.data ? '尚未取得就绪状态，请刷新检查。'
          : readiness.data.ready ? null : readiness.data.blocking_issues.map(issue => issue.message).join('；') || '模型尚未就绪。';
  const blocked = pending || !rollout.question.trim() || !executable.ok || Boolean(liveError);
  return <div className="mas-run-config mas-rollout-config">
    <form onSubmit={event => {
      event.preventDefault();
      if (blocked) return;
      void rollout.start();
      onShowResults();
    }}>
      <FormField label={rollout.execution === 'live' ? '调试任务' : '演示任务'} hint="从 Workflow 入口执行，答案与过程记录为同一份 Rollout。">
        <Textarea rows={4} required value={rollout.question} onChange={event => rollout.setQuestion(event.target.value)}
          placeholder="希望这个 Workflow 完成什么任务？" />
      </FormField>
      {!executable.ok && <InlineNotice tone="danger">{executable.reason}</InlineNotice>}
      <div className="mas-rollout-composer-footer">
        <div className="mas-rollout-composer-notes">
          <p className="debug-cost-hint">{rollout.execution === 'live' ? '真实调用当前模型及工具，可能产生费用。'
            : '模拟演示 · 不调用真实模型，不验证推理或工具能力。'}</p>
          {liveError && <p className="field-hint">请先处理上方的模型连接与执行条件。</p>}
        </div>
        <div className="mas-run-submit">
          <Button type="submit" size="sm" variant="primary" disabled={blocked} loading={rollout.running}>
            <Play size={13} aria-hidden="true" />{rollout.running ? '执行中' : rollout.execution === 'live' ? '发送调试' : '运行模拟演示'}
          </Button>
          <span>先保存 Workflow，再执行同一份快照；保存失败不执行。</span>
        </div>
      </div>
    </form>
  </div>;
}

const traceLabels = { complete: '完整 · complete', partial: '部分 · partial', unavailable: '不可用 · unavailable' };

export const RolloutResults = memo(function RolloutResults({ rollout, workflowChanged, workflow, onLocate, onUseQuestion, historyRequest, currentRequest = 0, active = true }: {
  rollout: MasRolloutRun;
  workflowChanged: boolean;
  workflow: WorkflowSpec | null;
  onLocate: (target: TrajectoryLocation) => void;
  onUseQuestion: () => void;
  historyRequest?: number;
  currentRequest?: number;
  active?: boolean;
}) {
  const history = useRolloutHistory(rollout.experimentId);
  useEffect(() => { if (currentRequest > 0) void history.select(null); }, [currentRequest, history.select]);
  const context = history.context;
  const historical = Boolean(history.selectedId);
  const run = historical ? context ? {
    summary: context.run, execution: context.run.execution, status: context.run.status,
    workflow: context.workflow, question: context.task.question, startedAt: Date.parse(context.run.started_at),
    endedAt: context.run.finished_at ? Date.parse(context.run.finished_at) : undefined, error: undefined,
  } : null : rollout.attempt;
  const data = historical ? history.trajectory : rollout.trajectory;
  const traceLoading = historical ? history.trajectoryLoading : rollout.trajectoryLoading;
  const traceError = historical ? history.trajectoryError : rollout.trajectoryError;
  const readTrajectory = historical ? history.readTrajectory : rollout.readTrajectory;
  const summary = run?.summary;
  const changed = useMemo(() => historical ? Boolean(context && JSON.stringify(context.workflow) !== JSON.stringify(workflow)) : workflowChanged,
    [historical, context, workflow, workflowChanged]);
  const elapsed = summary?.finished_at
    ? (Date.parse(summary.finished_at) - Date.parse(summary.started_at)) / 1000 : null;
  return <div className="mas-run-results mas-rollout-results">
    <RolloutHistory history={history} currentRunId={rollout.attempt?.summary?.run_id} openRequest={historyRequest} active={active} />
    {history.loading && <p className="mas-console-empty" role="status">正在读取历史运行…</p>}
    {history.error && <InlineNotice tone="danger">{history.error}
      <Button size="sm" onClick={() => void history.select(history.selectedId)}>重试读取</Button>
    </InlineNotice>}
    {!run && !history.loading && !history.error && <div className="mas-console-empty">输入一个任务，采集一次 Rollout，或选择已有的运行记录。</div>}
    {run && <>
    <header className="mas-result-header">
      <div className="mas-result-heading">
        <strong>{historical ? '历史运行' : '本次运行'}</strong>
        <StatusBadge tone={run.status === 'succeeded' ? 'success' : run.status === 'failed' ? 'danger'
          : run.status === 'interrupted' || run.status === 'unknown' ? 'warning' : 'info'}>
          {rolloutStatusLabel[run.status]}
        </StatusBadge>
        <span className="mas-result-execution">{run.execution === 'mock' ? '模拟执行 · Mock' : '真实推理 · Live'}</span>
      </div>
      <div className="mas-result-metrics">
        <time>{new Date(run.startedAt).toLocaleString()}</time>
        {elapsed !== null && Number.isFinite(elapsed) && elapsed >= 0 && <span><Clock3 size={13} aria-hidden="true" />总耗时 <b>{elapsed.toFixed(2)} 秒</b></span>}
        {elapsed === null && run.endedAt && <span><Clock3 size={13} aria-hidden="true" />本页耗时（含保存）<b>{((run.endedAt - run.startedAt) / 1000).toFixed(2)} 秒</b></span>}
      </div>
    </header>
    <ol className="mas-result-timeline">
      <li className="mas-result-step">
        <span className="mas-result-step-marker" aria-hidden="true">01</span>
        <section className="mas-result-card">
          <header className="mas-result-card-heading">
            <h3>本次提交的任务</h3>
            <Button size="sm" variant="ghost" onClick={() => { rollout.setQuestion(run.question); onUseQuestion(); }}>使用此问题</Button>
          </header>
          <div className="mas-result-card-body"><p className="mas-result-text">{run.question}</p></div>
        </section>
      </li>
      <li className="mas-result-step">
        <span className="mas-result-step-marker" aria-hidden="true">02</span>
        <section className="mas-result-card mas-result-card--answer">
          <header className="mas-result-card-heading">
            <h3>{summary?.final_answer && summary.status !== 'succeeded' ? '终止前保留的答案' : '最终答案'}</h3>
          </header>
          <div className="mas-result-card-body">
            {run.error && <RunErrorDetails message={run.error} unknown={run.status === 'unknown'} />}
            {summary?.error && <RunErrorDetails message={summary.error.message} stage={summary.error.stage} code={summary.error.code} />}
            {summary?.final_answer ? <p className="mas-result-text">{summary.final_answer}</p>
              : <p className="mas-result-placeholder" role="status">{!historical && rollout.running ? '正在执行，等待答案…' : '暂无最终答案'}</p>}
          </div>
        </section>
      </li>
      {summary && <li className="mas-result-step">
        <span className="mas-result-step-marker" aria-hidden="true">03</span>
        <section className="mas-result-card">
          <header className="mas-result-card-heading">
            <h3>Agent 执行过程</h3>
            <Button size="sm" onClick={() => void readTrajectory()} loading={traceLoading}
              disabled={traceLoading || !summary.trajectory_id}>
              <FileJson size={14} aria-hidden="true" />{traceError ? '重试读取过程' : data ? '刷新 Agent 过程' : '查看 Agent 输入输出'}
            </Button>
          </header>
          <div className="mas-result-card-body">
            {traceError && <InlineNotice tone="warning">轨迹读取失败：{traceError}</InlineNotice>}
            {data ? <Suspense fallback={<p className="mas-result-placeholder" role="status">正在加载轨迹视图…</p>}>
              <TrajectoryDetails key={data.run.run_id} data={data} experimentId={rollout.experimentId} onLocate={onLocate} active={active} />
            </Suspense> : !traceError && <p className="mas-result-placeholder" role="status">
              {traceLoading ? '正在读取执行过程…' : summary.trajectory_id ? '点击「查看 Agent 输入输出」展开执行过程。' : '未生成过程记录'}
            </p>}
          </div>
        </section>
      </li>}
    </ol>
    {summary && <div className="mas-result-secondary">
      <details className="mas-result-record"><summary>运行身份、模型与记录状态</summary>
      <dl className="mas-rollout-facts">
        <div><dt>Run ID</dt><dd><code>{summary.run_id}</code></dd></div>
        <div><dt>Trajectory ID</dt><dd><code>{summary.trajectory_id || '未生成'}</code></dd></div>
        <div><dt>实际模型</dt><dd>{summary.model ? `${summary.model.name} · ${summary.model.source}` : summary.execution === 'mock' ? 'Mock · 无真实模型' : '未返回'}</dd></div>
        <div><dt>策略版本</dt><dd>{summary.model?.policy_version ?? '未提供'}</dd></div>
        <div><dt>调用次数</dt><dd>模型 {summary.model_call_count} 次 · 工具 {summary.tool_call_count} 次</dd></div>
        <div><dt>格式检查</dt><dd>{summary.format_ok === null ? '未判定' : summary.format_ok ? '通过' : '未通过'}</dd></div>
        <div><dt>终止原因</dt><dd>{summary.termination_reason ?? '未返回'}</dd></div>
        <div><dt>轨迹记录</dt><dd>{traceLabels[summary.trace_status]}</dd></div>
        <div><dt>奖励评估</dt><dd>未执行</dd></div>
        {changed && <div><dt>Workflow</dt><dd>运行快照与当前草稿不同</dd></div>}
      </dl>
      </details>
      <RolloutExport key={summary.run_id} experimentId={rollout.experimentId} runId={summary.run_id} />
    </div>}
    </>}
  </div>;
}, (previous, next) => previous.active === next.active
  && previous.historyRequest === next.historyRequest
  && previous.currentRequest === next.currentRequest
  && previous.workflow === next.workflow && previous.workflowChanged === next.workflowChanged
  && previous.onLocate === next.onLocate && previous.onUseQuestion === next.onUseQuestion
  && previous.rollout.experimentId === next.rollout.experimentId
  && previous.rollout.attempt === next.rollout.attempt
  && previous.rollout.trajectory === next.rollout.trajectory
  && previous.rollout.trajectoryLoading === next.rollout.trajectoryLoading
  && previous.rollout.trajectoryError === next.rollout.trajectoryError
  && previous.rollout.readTrajectory === next.rollout.readTrajectory
  && previous.rollout.setQuestion === next.rollout.setQuestion);
