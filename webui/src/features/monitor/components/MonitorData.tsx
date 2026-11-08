import { memo, useMemo } from 'react';
import type { LossPoint, MonitorResponse, RewardArchive, RewardPoint } from '../../../shared/api/types';
import { EmptyState } from '../../../shared/components/EmptyState';
import { Section } from '../../../shared/components/Section';
import { formatClock, TrainEta } from '../../training/components/TrainEta';
import { RewardChart, type ChartPoint, type ChartSeries } from './RewardChart';

const EMPTY_REWARDS: RewardPoint[] = [];
const EMPTY_LOSSES: LossPoint[] = [];

function formatScalar(value: number | null): string {
  if (value == null || Number.isNaN(value)) return '—';
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function latestNumber<T extends object>(points: T[], key: keyof T): number | null {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const value = points[index]?.[key];
    if (typeof value === 'number') return value;
  }
  return null;
}

export const MonitorSummary = memo(function MonitorSummary({ model }: { model: MonitorResponse | null }) {
  const trainPoints = model?.step_rewards?.length || 0;
  const valPoints = model?.val_rewards?.length || 0;
  const lossPoints = model?.step_losses?.length || 0;
  const latestReward = latestNumber(model?.step_rewards || [], 'reward');
  const latestValReward = latestNumber(model?.val_rewards || [], 'reward');
  const latestValRa = latestNumber(model?.val_answer_rewards || model?.val_rewards || [], 'reward')
    ?? latestNumber(model?.val_rewards || [], 'answer_reward');
  const latestValRc = latestNumber(model?.val_correct_rewards || [], 'reward')
    ?? latestNumber(model?.val_rewards || [], 'correct_reward');
  const losses = model?.step_losses || [];
  const pg = latestNumber(losses, 'pg_loss');
  const entropy = latestNumber(losses, 'entropy_loss');
  const latestLoss = pg != null && pg !== 0 ? pg : (entropy ?? pg);
  const elapsed = model?.progress ? formatClock(model.progress.elapsed_seconds) : '—';
  return (
    <dl className="metrics-grid" aria-label="Monitor 概览">
      <div className="metric"><dt className="metric-label">训练集 step reward 记录数</dt><dd className="metric-value">{trainPoints}</dd></div>
      <div className="metric"><dt className="metric-label">验证集 reward 记录数</dt><dd className="metric-value">{valPoints}</dd></div>
      <div className="metric"><dt className="metric-label">Loss 记录数</dt><dd className="metric-value">{lossPoints}</dd></div>
      <div className="metric"><dt className="metric-label">最新 train reward</dt><dd className="metric-value">{formatScalar(latestReward)}</dd></div>
      <div className="metric"><dt className="metric-label">最新 val reward</dt><dd className="metric-value">{formatScalar(latestValReward)}</dd></div>
      <div className="metric"><dt className="metric-label">最新 val Ra</dt><dd className="metric-value">{formatScalar(latestValRa)}</dd></div>
      <div className="metric"><dt className="metric-label">最新 val Rc</dt><dd className="metric-value">{formatScalar(latestValRc)}</dd></div>
      <div className="metric"><dt className="metric-label">最新 Loss</dt><dd className="metric-value">{formatScalar(latestLoss)}</dd></div>
      <div className="metric"><dt className="metric-label">算法</dt><dd className="metric-value">{model?.tir_algo || '—'}</dd></div>
      <div className="metric"><dt className="metric-label">已用时间</dt><dd className="metric-value">{elapsed}</dd></div>
      <div className="metric"><dt className="metric-label">预计剩余</dt><dd className="metric-value"><TrainEta progress={model?.progress} running={Boolean(model?.train_running)} /></dd></div>
    </dl>
  );
});

function chartPoints(points: RewardPoint[]): ChartPoint[] {
  return points.map(point => ({ i: point.index, reward: point.reward }));
}

function mergeRewardSeries(
  total: RewardPoint[],
  answer: RewardPoint[],
  correct: RewardPoint[],
): ChartPoint[] {
  const byIndex = new Map<number, ChartPoint>();
  for (const point of total) {
    byIndex.set(point.index, {
      i: point.index,
      reward: point.reward,
      answer_reward: point.answer_reward,
      correct_reward: point.correct_reward,
    });
  }
  for (const point of answer) {
    const existing = byIndex.get(point.index) || { i: point.index };
    existing.answer_reward = point.reward;
    byIndex.set(point.index, existing);
  }
  for (const point of correct) {
    const existing = byIndex.get(point.index) || { i: point.index };
    existing.correct_reward = point.reward;
    byIndex.set(point.index, existing);
  }
  return [...byIndex.values()].sort((a, b) => a.i - b.i);
}

function ArchiveCharts({ archive }: { archive: RewardArchive }) {
  const training = useMemo(() => chartPoints(archive.step_rewards), [archive.step_rewards]);
  const validation = useMemo(() => chartPoints(archive.val_rewards), [archive.val_rewards]);
  const critic = useMemo(() => chartPoints(archive.critic_rewards), [archive.critic_rewards]);
  return <article className="page-stack" aria-label={archive.id}>
    <p className="field-hint">{archive.note}</p>
    <p className="field-hint">文件 <span className="mono break-all">{archive.path}</span></p>
    {training.length ? <RewardChart data={training} title={`${archive.id} 训练集`} series={[{ key: 'reward', name: `训练集 · ${archive.train_file}`, color: 'var(--chart-train)' }]} /> : <p className="field-hint">这份历史没有训练集 step reward。</p>}
    {validation.length ? <RewardChart data={validation} title={`${archive.id} 验证集`} series={[{ key: 'reward', name: `验证集 · ${archive.val_file || 'val 文件'}`, color: '#228be6' }]} /> : <p className="field-hint">这份历史没有验证集 reward。</p>}
    {critic.length ? <RewardChart data={critic} title={`${archive.id} critic`} series={[{ key: 'reward', name: '训练集 critic/rewards/mean', color: '#f59f00' }]} /> : null}
  </article>;
}

function lossSeries(points: LossPoint[]): ChartSeries[] {
  const has = (key: keyof LossPoint) => points.some(point => typeof point[key] === 'number');
  const pgValues = points.map(point => point.pg_loss).filter((value): value is number => typeof value === 'number');
  const pgFlatZero = pgValues.length > 0 && pgValues.every(value => value === 0);
  const series: ChartSeries[] = [];
  if (has('pg_loss')) series.push({ key: 'pg_loss', name: 'actor/pg_loss', color: '#fa5252' });
  if (pgFlatZero || !has('pg_loss')) {
    if (has('entropy_loss')) series.push({ key: 'entropy_loss', name: 'actor/entropy_loss', color: '#f59f00' });
    if (has('kl_loss')) series.push({ key: 'kl_loss', name: 'actor/kl_loss', color: '#7950f2' });
  }
  return series;
}

interface MonitorDataProps {
  model: MonitorResponse | null;
  aglOnline: boolean;
  visible: boolean;
}

export const MonitorData = memo(function MonitorData({ model, aglOnline, visible }: MonitorDataProps) {
  const training = model?.step_rewards ?? EMPTY_REWARDS;
  const validation = model?.val_rewards ?? EMPTY_REWARDS;
  const critic = model?.critic_rewards ?? EMPTY_REWARDS;
  const trainAnswer = model?.train_answer_rewards ?? EMPTY_REWARDS;
  const trainCorrect = model?.train_correct_rewards ?? EMPTY_REWARDS;
  const valAnswer = model?.val_answer_rewards ?? EMPTY_REWARDS;
  const valCorrect = model?.val_correct_rewards ?? EMPTY_REWARDS;
  const archives = model?.reward_archives ?? [];
  const labels = model?.reward_datasets;
  const losses = model?.step_losses ?? EMPTY_LOSSES;
  const trainChart = useMemo(
    () => mergeRewardSeries(training, trainAnswer, trainCorrect),
    [training, trainAnswer, trainCorrect],
  );
  const valChart = useMemo(
    () => mergeRewardSeries(validation, valAnswer, valCorrect),
    [validation, valAnswer, valCorrect],
  );
  const criticChart = useMemo(() => chartPoints(critic), [critic]);
  const trainSeries = useMemo(() => {
    const series: ChartSeries[] = [{ key: 'reward', name: 'train reward (total)', color: 'var(--chart-train)' }];
    if (trainChart.some(point => typeof point.answer_reward === 'number')) {
      series.push({ key: 'answer_reward', name: 'train Ra (answer)', color: '#12b886' });
    }
    if (trainChart.some(point => typeof point.correct_reward === 'number')) {
      series.push({ key: 'correct_reward', name: 'train Rc (correct)', color: '#e8590c' });
    }
    return series;
  }, [trainChart]);
  const valSeries = useMemo(() => {
    const series: ChartSeries[] = [{ key: 'reward', name: 'val reward (total)', color: '#228be6' }];
    if (valChart.some(point => typeof point.answer_reward === 'number')) {
      series.push({ key: 'answer_reward', name: 'val Ra (answer)', color: '#12b886' });
    }
    if (valChart.some(point => typeof point.correct_reward === 'number')) {
      series.push({ key: 'correct_reward', name: 'val Rc (correct)', color: '#e8590c' });
    }
    return series;
  }, [valChart]);
  const lossChart = useMemo(
    () => losses.map(point => ({
      i: point.index,
      pg_loss: point.pg_loss,
      entropy_loss: point.entropy_loss,
      kl_loss: point.kl_loss,
    })),
    [losses],
  );
  const lossLines = useMemo(() => lossSeries(losses), [losses]);

  const trainName = labels?.train_label || '训练集 step reward · llm_binary 该 step rollout 平均';
  const valName = labels?.val_label || '验证集 reward · 只在 test_freq 触发时写入';
  const criticName = labels?.critic_label || '同一批训练样本的 critic/rewards/mean，不是验证集';
  return (
    <>
      <Section title="训练集 reward（total / Ra / Rc）">
        <p className="field-hint">{trainName}。Ra=`fcr/R_answer_mean`，Rc=`fcr/R_correct_mean`。</p>
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : trainChart.length ? (
          <RewardChart data={trainChart} title="训练集 reward" series={trainSeries} />
        ) : (
          <EmptyState title={aglOnline ? '训练已连接，尚无训练集 step reward' : '尚无训练集 step reward'}>
            {aglOnline
              ? '等第一个 actor update 结束，这里会出现 training/reward。'
              : '当前运行的 metrics.jsonl 里还没有 training/reward。历史曲线在下方单独列出。'}
          </EmptyState>
        )}
        {criticChart.length ? <>
          <p className="field-hint">{criticName}</p>
          <RewardChart data={criticChart} title="训练集 critic/rewards/mean" series={[{ key: 'reward', name: criticName, color: '#f59f00' }]} />
        </> : null}
      </Section>
      <Section title="验证集 reward（total / Ra / Rc）">
        <p className="field-hint">{valName}。Ra=`val/fcr/R_answer_mean`，Rc=`val/fcr/R_correct_mean`。</p>
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : valChart.length ? (
          <RewardChart data={valChart} title="验证集 reward" series={valSeries} />
        ) : (
          <EmptyState title="尚无验证集 reward">
            val/reward 只在验证触发时写入。test_freq 未到，或这次运行没有验证。
          </EmptyState>
        )}
      </Section>
      <Section title="保留的历史 Reward">
        {archives.length ? archives.map(archive => <ArchiveCharts key={archive.id} archive={archive} />) : (
          <EmptyState title="没有单独保留的历史 Reward">这份实验没有登记只读的旧 metrics 文件。</EmptyState>
        )}
      </Section>
      <Section title="Loss">
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : lossChart.length && lossLines.length ? (
          <RewardChart data={lossChart} title="Loss" series={lossLines} />
        ) : (
          <EmptyState title="尚无 Loss">
            actor/pg_loss 在第一次 actor update 之后写入 metrics.jsonl。若 pg_loss 全程为 0，图上会同时画出 entropy 或 kl。
          </EmptyState>
        )}
      </Section>
      {model?.metrics_path ? <p className="field-hint">metrics.jsonl: <span className="mono break-all">{model.metrics_path}</span></p> : null}
    </>
  );
});
