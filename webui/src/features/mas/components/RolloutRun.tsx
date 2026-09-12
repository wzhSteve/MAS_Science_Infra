import { memo, useMemo, useEffect, lazy, Suspense } from 'react';
import { Play, FileJson } from 'lucide-react';
import { Button } from '../../../shared/ui/button';
import { Textarea } from '../../../shared/ui/textarea';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
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
      <p className="debug-cost-hint">{rollout.execution === 'live' ? '真实调用当前模型及工具，可能产生费用。'
        : '模拟演示 · 不调用真实模型，不验证推理或工具能力。'}</p>
      {liveError && <p className="field-hint">请先处理上方的模型连接与执行条件。</p>}
      {!executable.ok && <InlineNotice tone="danger">{executable.reason}</InlineNotice>}
      <div className="mas-run-submit">
        <Button type="submit" size="sm" variant="primary" disabled={blocked} loading={rollout.running}>
          <Play size={13} aria-hidden="true" />{rollout.running ? '执行中' : rollout.execution === 'live' ? '发送调试' : '运行模拟演示'}
        </Button>
        <span>先保存 Workflow，再执行同一份快照；保存失败不执行。</span>
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
    <div className={`debug-result-identity${historical ? ' is-historical' : ''}`}>
      <strong>{historical ? '当前查看：历史记录' : '当前查看：本次结果'}</strong>
      {summary && <><time>{new Date(summary.started_at).toLocaleString()}</time><code>{summary.run_id}</code></>}
      {historical && <Button size="sm" onClick={() => void history.select(null)}>返回本次结果</Button>}
    </div>
    {history.loading && <p className="mas-console-empty" role="status">正在读取历史运行…</p>}
    {history.error && <InlineNotice tone="danger">{history.error}
      <Button size="sm" onClick={() => void history.select(history.selectedId)}>重试读取</Button>
    </InlineNotice>}
    {!run && !history.loading && !history.error && <div className="mas-console-empty">输入一个任务，采集一次 Rollout，或选择已有的运行记录。</div>}
    {run && <>
    <div className="mas-result-summary">
      <strong>{historical ? '历史运行 · ' : ''}{rolloutStatusLabel[run.status]}</strong>
      <span>{run.execution === 'mock' ? '模拟执行 · Mock' : '真实推理 · Live'}</span>
      {elapsed !== null && Number.isFinite(elapsed) && elapsed >= 0 && <span>总耗时 {elapsed.toFixed(2)} 秒</span>}
      {elapsed === null && run.endedAt && <span>本页耗时（含保存）{((run.endedAt - run.startedAt) / 1000).toFixed(2)} 秒</span>}
    </div>
    <p className="field-hint">结果绑定运行时快照，不会覆盖当前画布{changed ? '；当前配置与快照不一致，定位仅按已有实体标识进行' : ''}。</p>
    <details className="mas-rollout-task"><summary>本次提交的任务</summary><p>{run.question}</p></details>
    <div><Button size="sm" variant="ghost" onClick={() => { rollout.setQuestion(run.question); onUseQuestion(); }}>使用此问题</Button>
      <span className="field-hint">只填充问题；再次运行使用当前 Workflow 和模型配置。</span></div>
    {run.error && <RunErrorDetails message={run.error} unknown={run.status === 'unknown'} />}
    {!historical && rollout.running && <p className="field-hint">等待同步请求返回；关闭控制台不会取消执行，也不会自动重发。</p>}
    {summary && <>
      {summary.error && <RunErrorDetails message={summary.error.message} stage={summary.error.stage} code={summary.error.code} />}
      {summary.final_answer ? <section className="mas-rollout-answer">
        <h3>{summary.status === 'succeeded' ? '最终答案' : '终止前保留的答案'}</h3><p>{summary.final_answer}</p>
      </section> : !summary.error && <p className="field-hint">尚无最终答案，已有的部分记录可通过轨迹入口查看。</p>}
      <details className="mas-rollout-task"><summary>运行身份、模型与记录状态</summary>
      <dl className="mas-rollout-facts">
        <div><dt>Run ID</dt><dd><code>{summary.run_id}</code></dd></div>
        <div><dt>Trajectory ID</dt><dd><code>{summary.trajectory_id || '未生成'}</code></dd></div>
        <div><dt>实际模型</dt><dd>{summary.model ? `${summary.model.name} · ${summary.model.source}` : summary.execution === 'mock' ? 'Mock · 无真实模型' : '未返回'}</dd></div>
        <div><dt>策略版本</dt><dd>{summary.model?.policy_version ?? '未提供'}</dd></div>
        <div><dt>调用次数</dt><dd>模型 {summary.model_call_count} 次 · 工具 {summary.tool_call_count} 次</dd></div>
        <div><dt>格式检查</dt><dd>{summary.format_ok === null ? '未判定' : summary.format_ok ? '通过' : '未通过'}</dd></div>
        <div><dt>终止原因</dt><dd>{summary.termination_reason ?? '未返回'}</dd></div>
        <div><dt>轨迹记录</dt><dd>{traceLabels[summary.trace_status]}</dd></div>
      </dl>
      </details>
      <p className="field-hint">本次未进行奖励评估；记录完整不等于答案正确或满足全部训练要求。</p>
      <div className="mas-rollout-trace">
        <Button size="sm" onClick={() => void readTrajectory()} loading={traceLoading}
          disabled={traceLoading || !summary.trajectory_id}>
          <FileJson size={14} aria-hidden="true" />{traceError ? '重试读取过程' : data ? '刷新 Agent 过程' : '查看 Agent 输入输出'}
        </Button>
        <span className="field-hint">仅读取已有记录，不重新执行模型。</span>
      </div>
      {traceError && <InlineNotice tone="warning">轨迹读取失败：{traceError}。不改变已知的运行结果，可重试读取。</InlineNotice>}
      {data && <Suspense fallback={<p className="field-hint">正在加载轨迹视图…</p>}>
        <TrajectoryDetails key={data.run.run_id} data={data} experimentId={rollout.experimentId} onLocate={onLocate} active={active} />
      </Suspense>}
      <RolloutExport key={summary.run_id} experimentId={rollout.experimentId} runId={summary.run_id} />
    </>}
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
