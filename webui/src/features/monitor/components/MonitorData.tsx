import { memo, useMemo } from 'react';
import type { LossPoint, MonitorResponse, RewardPoint } from '../../../shared/api/types';
import { EmptyState } from '../../../shared/components/EmptyState';
import { Section } from '../../../shared/components/Section';
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
  const losses = model?.step_losses || [];
  const pg = latestNumber(losses, 'pg_loss');
  const entropy = latestNumber(losses, 'entropy_loss');
  const latestLoss = pg != null && pg !== 0 ? pg : (entropy ?? pg);
  return (
    <dl className="metrics-grid" aria-label="Monitor 概览">
      <div className="metric"><dt className="metric-label">训练 Reward 记录数</dt><dd className="metric-value">{trainPoints}</dd></div>
      <div className="metric"><dt className="metric-label">验证 Reward 记录数</dt><dd className="metric-value">{valPoints}</dd></div>
      <div className="metric"><dt className="metric-label">Loss 记录数</dt><dd className="metric-value">{lossPoints}</dd></div>
      <div className="metric"><dt className="metric-label">最新 training/reward</dt><dd className="metric-value">{formatScalar(latestReward)}</dd></div>
      <div className="metric"><dt className="metric-label">最新 Loss</dt><dd className="metric-value">{formatScalar(latestLoss)}</dd></div>
    </dl>
  );
});

function mergeRewards(training: RewardPoint[], validation: RewardPoint[]): ChartPoint[] {
  const byStep = new Map<number, ChartPoint>();
  for (const point of training) {
    byStep.set(point.index, { i: point.index, reward: point.reward });
  }
  for (const point of validation) {
    const row = byStep.get(point.index) ?? { i: point.index };
    row.val = point.reward;
    byStep.set(point.index, row);
  }
  return [...byStep.values()].sort((left, right) => left.i - right.i);
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
  const losses = model?.step_losses ?? EMPTY_LOSSES;
  const rewardChart = useMemo(() => mergeRewards(training, validation), [training, validation]);
  const rewardSeries = useMemo(() => {
    const series: ChartSeries[] = [];
    if (training.length) series.push({ key: 'reward', name: 'training/reward', color: 'var(--chart-train)' });
    if (validation.length) series.push({ key: 'val', name: 'val/reward', color: '#228be6' });
    return series;
  }, [training.length, validation.length]);
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

  return (
    <>
      <Section title="训练 Reward（AGL）">
        {!visible ? <div className="h-[220px]" aria-hidden="true" /> : rewardChart.length ? (
          <RewardChart data={rewardChart} title="训练 Reward（AGL）" series={rewardSeries} />
        ) : (
          <EmptyState title={aglOnline ? '训练已连接，尚无训练 Reward' : '尚无训练 Reward'}>
            {aglOnline
              ? '训练已连接，但还没有 training/reward。等第一个 actor update 结束即可。'
              : 'metrics.jsonl 里还没有 training/reward 或 val/reward。训练写入后这里会画出历史曲线，不必等 LightningStore 在线。'}
          </EmptyState>
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
